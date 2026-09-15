"""Case persistence for module 3: enqueue fuse/pipeline, review, calibration.

/fuse keeps input/analysis/retrieval and fills decision; a new retrieval or
analysis run invalidates a stale decision (contract invariant 8). Human
review is an audit action that never overwrites model output.
"""

import time
from uuid import uuid4

from sqlalchemy import func, select

from aurora_decision.config import ServiceError
from aurora_decision.contract import AuroraBundle, HumanReview, canonical, sha
from aurora_decision.models.db import Audit, Case, Job, Snapshot


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

    def enqueue(self, raw, owner, key, endpoint):
        payload_hash = sha(canonical(raw))
        if not key or len(key) > 128:
            raise ServiceError("IDEMPOTENCY_KEY_REQUIRED", "Idempotency-Key wajib berisi 1–128 karakter.")
        if raw.get("schema_version") != "1.0.0":
            raise ServiceError("SCHEMA_VERSION_UNSUPPORTED", "Versi kontrak tidak didukung.")
        # Idempotency/conflict check runs before full contract validation so a
        # same-key-different-payload retry surfaces as 409, not a schema error.
        with self.session() as s:
            existing = s.scalar(
                select(Job).where(Job.owner == owner, Job.endpoint == endpoint, Job.key == key)
            )
            if existing:
                if existing.payload_hash != payload_hash:
                    raise ServiceError(
                        "IDEMPOTENCY_CONFLICT", "Key yang sama dipakai untuk payload berbeda.", 409
                    )
                return existing
        bundle = AuroraBundle.model_validate(raw)
        if endpoint == "fuse":
            # Contract requires a valid analysis for fusion; validate now for 422.
            if bundle.analysis is None:
                raise ServiceError(
                    "INPUT_ANALYSIS_REQUIRED",
                    "Fusi memerlukan analysis valid dari modul 1 atau impor standar.",
                )
        for image in bundle.input.images:
            self.media.resolve(image, owner)
        with self.session.begin() as s:
            s.connection().exec_driver_sql("BEGIN IMMEDIATE")
            old = s.scalar(select(Job).where(Job.owner == owner, Job.endpoint == endpoint, Job.key == key))
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
                    raise ServiceError("MODE_CONFLICT", "Mode kasus tetap; buat kasus baru.", 409)
                if bundle.claim_revision != current.revision:
                    raise ServiceError("REVISION_CONFLICT", "Revisi kasus tidak cocok.", 409)
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
                endpoint=endpoint,
                key=key,
                payload_hash=payload_hash,
                snapshot_hash=sha(canonical(dump(bundle))),
                payload=dump(bundle),
                status="queued",
            )
            s.add(job)
            s.flush()
            return job

    def review(self, case_id, owner, reviewer, verdict, reason):
        """Append human review; model output stays intact in the same bundle."""
        if verdict not in ("Supported", "Contradicted", "InsufficientEvidence"):
            raise ServiceError(
                "INVALID_VERDICT", "Verdict harus Supported/Contradicted/InsufficientEvidence."
            )
        if not reason.strip():
            raise ServiceError("REVIEW_REASON_REQUIRED", "Alasan tinjauan wajib diisi.")
        with self.session.begin() as s:
            s.connection().exec_driver_sql("BEGIN IMMEDIATE")
            row = s.get(Case, case_id)
            if not row or row.owner != owner:
                raise ServiceError("CASE_NOT_FOUND", "Kasus tidak ditemukan.", 404)
            bundle = AuroraBundle.model_validate(row.bundle)
            if bundle.decision is None:
                raise ServiceError("NO_DECISION", "Belum ada keputusan untuk ditinjau.", 409)
            review = HumanReview(
                reviewer=reviewer or "anonim",
                reviewed_at=f"{time.time():.0f}",
                verdict=verdict,
                reason=reason,
            )
            bundle.decision.human_review = review
            row.bundle, row.updated_at = dump(bundle), time.time()
            snapshot(s, bundle, "human_review", bundle.decision.run.run_id, reason)
            s.add(
                Audit(
                    id=str(uuid4()),
                    case_id=case_id,
                    event="human_review",
                    details={
                        "reviewer": reviewer,
                        "verdict": verdict,
                        "reason_sha256": sha(reason),
                        "decision_run_id": bundle.decision.run.run_id,
                    },
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
