"""Case persistence for module 2: enqueue, revision rules, import.

/retrieve keeps input+analysis, replaces retrieval and clears decision.
A changed caption/image bumps claim_revision and invalidates retrieval+decision;
an analysis atom-set change invalidates stale retrieval (contract invariant 8).
"""

import time
from uuid import uuid4

from sqlalchemy import func, select

from aurora_evidence.config import ServiceError
from aurora_evidence.contract import AuroraBundle, canonical, sha
from aurora_evidence.models.db import Case, Job, Snapshot


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
                select(Job).where(Job.owner == owner, Job.endpoint == "retrieve", Job.key == key)
            )
            if existing:
                if existing.payload_hash != payload_hash:
                    raise ServiceError(
                        "IDEMPOTENCY_CONFLICT", "Key yang sama dipakai untuk payload berbeda.", 409
                    )
                return existing
        bundle = AuroraBundle.model_validate(raw)
        for image in bundle.input.images:
            self.media.resolve(image, owner)
        with self.session.begin() as s:
            s.connection().exec_driver_sql("BEGIN IMMEDIATE")
            old = s.scalar(select(Job).where(Job.owner == owner, Job.endpoint == "retrieve", Job.key == key))
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
                if bundle.mode != previous.mode:
                    raise ServiceError(
                        "MODE_CONFLICT", "Mode kasus tetap; buat kasus baru untuk mode berbeda.", 409
                    )
                if bundle.claim_revision != current.revision:
                    raise ServiceError(
                        "REVISION_CONFLICT", "Revisi kasus tidak cocok dengan yang tersimpan.", 409
                    )
                # Retrieval is replaced wholesale; decision from the old retrieval is stale.
                bundle.decision = None
                current.bundle, current.updated_at = dump(bundle), time.time()
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
                endpoint="retrieve",
                key=key,
                payload_hash=payload_hash,
                snapshot_hash=sha(canonical(dump(bundle))),
                payload=dump(bundle),
                status="queued",
            )
            s.add(job)
            s.flush()
            return job

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
