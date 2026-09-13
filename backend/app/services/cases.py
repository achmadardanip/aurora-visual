import time
from uuid import uuid4

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from app.config import ServiceError
from app.models.contract import Analysis, Atom, AuroraBundle, RunInfo, atom_set_id, canonical, sha
from app.models.db import Audit, Case, Job, Snapshot
from app.services.pipeline import now


def dump(value):
    return value.model_dump(mode="json")


def snapshot(s, bundle, kind, parent=None, reason=None):
    s.add(
        Snapshot(
            id=str(uuid4()),
            case_id=bundle.case_id,
            revision=bundle.claim_revision,
            kind=kind,
            parent_id=parent,
            reason=reason,
            bundle=dump(bundle),
        )
    )


class CaseService:
    def __init__(self, settings, session, media):
        self.settings, self.session, self.media = settings, session, media

    def get(self, case_id, owner):
        with self.session() as s:
            row = s.get(Case, case_id)
            if not row or row.owner != owner:
                raise ServiceError("CASE_NOT_FOUND", "Kasus tidak ditemukan.", 404)
            return AuroraBundle.model_validate(row.bundle)

    def enqueue(self, raw, owner, key):
        payload_hash = sha(canonical(raw))
        if not key or len(key) > 128:
            raise ServiceError("IDEMPOTENCY_KEY_REQUIRED", "Idempotency-Key wajib berisi 1–128 karakter.")
        if raw.get("schema_version") != "1.0.0":
            raise ServiceError("SCHEMA_VERSION_UNSUPPORTED", "Versi kontrak tidak didukung.")
        with self.session() as s:
            existing = s.scalar(
                select(Job).where(Job.owner == owner, Job.endpoint == "analyze", Job.key == key)
            )
            if existing:
                if existing.payload_hash != payload_hash:
                    raise ServiceError(
                        "IDEMPOTENCY_CONFLICT", "Key yang sama dipakai untuk payload berbeda.", 409
                    )
                return existing
        bundle = AuroraBundle.model_validate(raw)
        if not bundle.input.image:
            raise ServiceError("IMAGE_REQUIRED", "Analisis visual memerlukan gambar.")
        self.media.resolve(bundle.input.image, owner)
        options = bundle.extensions.get("aurora_visual", {}).get("options", {})
        if not isinstance(options, dict) or set(options) - {
            "alignment",
            "backbone",
            "top_k",
            "parser",
            "head",
            "provider",
            "translation_shadow",
        }:
            raise ServiceError("INVALID_OPTIONS", "Opsi analisis tidak dikenal.")
        if (
            options.get("alignment", "uot")
            not in ("uot", "balanced-ot", "attention", "max-region", "mean-region", "global")
            or options.get("backbone", self.settings.backbone) not in ("local-color-v1", "openclip")
            or options.get("parser", "rules") not in ("rules", "llm", "hive-vlm")
            or options.get("head", "heuristic") not in ("heuristic", "trained")
            or options.get("provider", "local") not in ("local", "hive")
            or type(options.get("translation_shadow", False)) is not bool
        ):
            raise ServiceError("INVALID_OPTIONS", "Pilihan model/metode tidak didukung.")
        provider = options.get("provider", "local")
        parser = options.get("parser", "rules")
        translation_shadow = options.get("translation_shadow", False)
        if bundle.mode == "demo" and (provider != "local" or parser == "hive-vlm" or translation_shadow):
            raise ServiceError("INVALID_OPTIONS", "Mode demo hanya boleh memakai pemrosesan lokal.")
        if provider != "hive" and (parser == "hive-vlm" or translation_shadow):
            raise ServiceError(
                "HIVE_PROVIDER_REQUIRED",
                "Parser dan terjemahan Hive memerlukan pemilihan pemrosesan eksternal Hive.",
            )
        if provider == "hive" and not self.settings.hive_enabled:
            raise ServiceError("HIVE_DISABLED", "Pemrosesan eksternal Hive belum diaktifkan pada server.")
        if provider == "hive" and not self.settings.hive_v3_secret:
            raise ServiceError(
                "HIVE_V3_UNCONFIGURED",
                "Hive V3 VLM wajib dikonfigurasi untuk atomisasi dan observasi multimodal.",
            )
        if type(options.get("top_k", 16)) is not int or not 1 <= options.get("top_k", 16) <= 32:
            raise ServiceError("INVALID_OPTIONS", "Jumlah region harus 1–32.")
        try:
            with self.session.begin() as s:
                s.connection().exec_driver_sql("BEGIN IMMEDIATE")
                old = s.scalar(
                    select(Job).where(Job.owner == owner, Job.endpoint == "analyze", Job.key == key)
                )
                if old:
                    if old.payload_hash != payload_hash:
                        raise ServiceError(
                            "IDEMPOTENCY_CONFLICT", "Key yang sama dipakai untuk payload berbeda.", 409
                        )
                    return old
                count = s.scalar(
                    select(func.count())
                    .select_from(Job)
                    .where(Job.owner == owner, Job.status.in_(["queued", "running"]))
                )
                if count >= self.settings.max_pending:
                    raise ServiceError(
                        "QUEUE_FULL", "Antrean penuh. Coba lagi setelah pekerjaan selesai.", 429, True
                    )
                current = s.get(Case, bundle.case_id)
                if current and current.owner != owner:
                    raise ServiceError("CASE_NOT_FOUND", "Kasus tidak ditemukan.", 404)
                if current:
                    previous = AuroraBundle.model_validate(current.bundle)

                    def identity(b):
                        return (b.input.claim_text, b.input.image.sha256 if b.input.image else None)

                    changed = identity(previous) != identity(bundle)
                    if bundle.mode != previous.mode:
                        raise ServiceError(
                            "MODE_CONFLICT", "Mode kasus tetap; buat kasus baru untuk mode berbeda.", 409
                        )
                    if changed:
                        if (
                            bundle.claim_revision != current.revision + 1
                            or bundle.analysis
                            or bundle.retrieval
                            or bundle.decision
                        ):
                            raise ServiceError(
                                "REVISION_CONFLICT",
                                "Input berubah: naikkan revisi satu dan kosongkan hasil turunan.",
                                409,
                            )
                    elif bundle.claim_revision != current.revision:
                        raise ServiceError("REVISION_CONFLICT", "Revisi kasus tidak cocok.", 409)
                    elif (bundle.analysis.atom_set_id if bundle.analysis else None) != (
                        previous.analysis.atom_set_id if previous.analysis else None
                    ):
                        raise ServiceError(
                            "ATOM_SET_CONFLICT", "Atom set usang; gunakan endpoint koreksi.", 409
                        )
                    current.bundle, current.revision, current.updated_at = (
                        dump(bundle),
                        bundle.claim_revision,
                        time.time(),
                    )
                else:
                    s.add(
                        Case(
                            case_id=bundle.case_id,
                            owner=owner,
                            revision=bundle.claim_revision,
                            bundle=dump(bundle),
                        )
                    )
                snapshot(s, bundle, "submitted")
                job = Job(
                    job_id=str(uuid4()),
                    run_id=str(uuid4()),
                    case_id=bundle.case_id,
                    owner=owner,
                    key=key,
                    payload_hash=payload_hash,
                    snapshot_hash=sha(canonical(dump(bundle))),
                    payload=dump(bundle),
                    status="queued",
                )
                s.add(job)
                s.flush()
                return job
        except IntegrityError as exc:
            raise ServiceError(
                "REQUEST_CONFLICT", "Request bersamaan; ulangi dengan key yang sama.", 409, True
            ) from exc

    def correct(self, case_id, owner, atoms, expected, reason):
        if not reason.strip():
            raise ServiceError("CORRECTION_REASON_REQUIRED", "Alasan koreksi wajib diisi.")
        with self.session.begin() as s:
            s.connection().exec_driver_sql("BEGIN IMMEDIATE")
            row = s.get(Case, case_id)
            if not row or row.owner != owner:
                raise ServiceError("CASE_NOT_FOUND", "Kasus tidak ditemukan.", 404)
            bundle = AuroraBundle.model_validate(row.bundle)
            if not bundle.analysis or bundle.analysis.atom_set_id != expected:
                raise ServiceError(
                    "ATOM_SET_CONFLICT", "Atom telah berubah. Muat ulang sebelum koreksi.", 409
                )
            atoms = [Atom.model_validate(a) for a in atoms]
            new_id = atom_set_id(bundle, atoms)
            if new_id == expected:
                raise ServiceError("NO_ATOM_CHANGE", "Koreksi belum mengubah atom.", 409)
            old_run = bundle.analysis.run.run_id
            stamp = now()
            bundle.analysis = Analysis(
                run=RunInfo(
                    run_id=str(uuid4()),
                    mode=bundle.mode,
                    started_at=stamp,
                    finished_at=stamp,
                    status="partial",
                    versions={"parser": "human-correction-v1"},
                    warnings=[],
                ),
                atom_set_id=new_id,
                atomic_claims=atoms,
                visual_assessments=[],
                ocr=[],
            )
            bundle.retrieval, bundle.decision = None, None
            bundle.extensions["aurora_visual"] = {
                "options": bundle.extensions.get("aurora_visual", {}).get("options", {}),
                "correction": {"parent_atom_set_id": expected, "parent_run_id": old_run, "reason": reason},
            }
            bundle = AuroraBundle.model_validate(dump(bundle))
            row.bundle, row.updated_at = dump(bundle), time.time()
            snapshot(s, bundle, "atom_correction", expected, reason)
            s.add(
                Audit(
                    id=str(uuid4()),
                    case_id=case_id,
                    event="atom_correction",
                    details={"parent": expected, "atom_set_id": new_id, "reason_sha256": sha(reason)},
                )
            )
            return bundle

    def import_bundle(self, bundle, owner):
        with self.session.begin() as s:
            current = s.get(Case, bundle.case_id)
            if current:
                if current.owner != owner or canonical(current.bundle) != canonical(dump(bundle)):
                    raise ServiceError(
                        "IMPORT_CONFLICT",
                        "ID kasus sudah ada dengan snapshot berbeda. Riwayat lokal dipertahankan.",
                        409,
                    )
                return bundle
            s.add(
                Case(case_id=bundle.case_id, owner=owner, revision=bundle.claim_revision, bundle=dump(bundle))
            )
            snapshot(s, bundle, "import")
        return bundle
