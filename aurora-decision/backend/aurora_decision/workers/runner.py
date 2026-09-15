"""Persistent job worker for module 3: fuse and pipeline endpoints."""

import os
import re
import subprocess
import sys
import threading
import time
from uuid import uuid4

from sqlalchemy import select

from aurora_decision.config import Settings
from aurora_decision.contract import AuroraBundle, canonical, sha
from aurora_decision.core.fuse import fuse
from aurora_decision.core.orchestration import UpstreamClient, UpstreamError, run_pipeline
from aurora_decision.models.db import Audit, Base, Case, Job, WorkerState, database
from aurora_decision.services.cases import snapshot
from aurora_decision.services.media import MediaService


def fail(session, job_id, code, message, retryable=False, detail=None):
    with session.begin() as s:
        job = s.get(Job, job_id)
        if job and job.status == "running":
            job.status = "failed"
            job.progress = message
            error = {"code": code, "message": message, "retryable": retryable}
            if detail:
                error["detail"] = detail
            job.error = error
            s.add(
                Audit(
                    id=str(uuid4()),
                    case_id=job.case_id,
                    event="run_failed",
                    details={"run_id": job.run_id, "code": code},
                )
            )


def execute(settings, session, job_id):
    with session() as s:
        job = s.get(Job, job_id)
    if not job or job.status != "running":
        return

    def progress(message):
        with session.begin() as s:
            row = s.get(Job, job_id)
            if row and row.status == "running":
                row.progress = message

    try:
        if job.endpoint == "pipeline":
            progress("Menjalankan pipeline analisis -> pencarian -> fusi")
            media = MediaService(settings, session)
            visual = UpstreamClient(settings.visual_url)
            evidence = UpstreamClient(settings.evidence_url)

            def media_loader(asset_id):
                path, ref, _ = media.resolve(
                    next(
                        (
                            i
                            for i in AuroraBundle.model_validate(job.payload).input.images
                            if i.asset_id == asset_id
                        ),
                        None,
                    ),
                    job.owner,
                )
                return path.read_bytes(), ref.media_type

            bundle_dict = run_pipeline(
                job.payload,
                visual,
                evidence,
                media_loader,
                deadline_seconds=settings.upstream_deadline,
                progress=progress,
            )
            bundle = AuroraBundle.model_validate(bundle_dict)
        else:
            bundle = AuroraBundle.model_validate(job.payload)
        output = fuse(bundle, settings, progress)
        with session.begin() as s:
            s.connection().exec_driver_sql("BEGIN IMMEDIATE")
            current_job = s.get(Job, job_id)
            case = s.get(Case, job.case_id)
            if current_job.status != "running":
                return
            # Snapshot staleness applies to fuse jobs (user may have changed the
            # case while a fuse ran). Pipeline jobs enrich the bundle through
            # upstream services by design; the case is only written by us here.
            if job.endpoint == "fuse" and sha(canonical(case.bundle)) != job.snapshot_hash:
                current_job.status = "failed"
                current_job.error = {
                    "code": "STALE_RESULT",
                    "message": "Kasus telah berubah; hasil lama tidak dipasang.",
                    "retryable": False,
                }
                return
            result = output.model_dump(mode="json")
            current_job.status = "partial" if output.decision.run.status == "partial" else "succeeded"
            current_job.result, current_job.progress = result, "Fusi selesai"
            case.bundle, case.updated_at = result, time.time()
            snapshot(s, output, "decision", job.run_id)
            s.add(
                Audit(
                    id=str(uuid4()),
                    case_id=job.case_id,
                    event="run_completed",
                    details={
                        "run_id": job.run_id,
                        "endpoint": job.endpoint,
                        "final_verdict": output.decision.final_verdict,
                        "abstained": output.decision.abstention_flag,
                        "output_hash": sha(canonical(result)),
                    },
                )
            )
    except UpstreamError as exc:
        fail(session, job_id, exc.code, exc.message, exc.retryable)
    except ValueError as exc:
        text = str(exc)
        code = text if re.fullmatch(r"[A-Z][A-Z0-9_]{3,63}", text) else "FUSION_FAILED"
        fail(
            session,
            job_id,
            code,
            "Fusi gagal: input tidak memenuhi prasyarat kontrak.",
            False,
            text[:200],
        )
    except Exception as exc:
        fail(
            session,
            job_id,
            "FUSION_FAILED",
            "Fusi gagal; riwayat request tetap tersimpan.",
            False,
            type(exc).__name__,
        )
        print(f"Job {job_id}: {type(exc).__name__}", file=sys.stderr)


class Worker:
    def __init__(self, settings, session):
        self.settings, self.session = settings, session
        self.stop_event = threading.Event()
        self.thread = None

    def heartbeat(self):
        with self.session.begin() as s:
            s.merge(WorkerState(id="local", heartbeat=time.time()))

    def claim(self):
        with self.session.begin() as s:
            s.connection().exec_driver_sql("BEGIN IMMEDIATE")
            expired = s.scalars(
                select(Job).where(Job.status == "running", Job.lease_until < time.time())
            ).all()
            for job in expired:
                if job.attempts < 3:
                    job.status, job.progress = "queued", "Memulihkan pekerjaan setelah restart"
                else:
                    job.status, job.error = (
                        "failed",
                        {
                            "code": "RETRY_EXHAUSTED",
                            "message": "Batas pemulihan pekerjaan tercapai.",
                            "retryable": False,
                        },
                    )
            job = s.scalar(select(Job).where(Job.status == "queued").order_by(Job.created_at).limit(1))
            if not job:
                return None
            job.status, job.progress = "running", "Memulai fusi bukti"
            job.attempts += 1
            job.lease_until = time.time() + self.settings.job_timeout + 15
            return job.job_id

    def run_once(self, isolated=True):
        self.heartbeat()
        job_id = self.claim()
        if not job_id:
            return False
        if not isolated:
            execute(self.settings, self.session, job_id)
            return True
        env = {
            **os.environ,
            # Absolute path: the subprocess may run with a different cwd, and
            # a relative data dir would silently point elsewhere.
            "AURORA_DECISION_DATA_DIR": str(self.settings.data_dir.resolve()),
        }
        process = subprocess.Popen(
            [sys.executable, "-m", "aurora_decision.workers.runner", "--execute", job_id], env=env
        )
        started = time.monotonic()
        try:
            while process.poll() is None:
                self.heartbeat()
                with self.session() as s:
                    cancelled = s.get(Job, job_id).status != "running"
                if (
                    cancelled
                    or self.stop_event.is_set()
                    or time.monotonic() - started > self.settings.job_timeout
                ):
                    process.terminate()
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    if self.stop_event.is_set():
                        with self.session.begin() as s:
                            row = s.get(Job, job_id)
                            if row.status == "running":
                                row.lease_until = 0
                    elif not cancelled:
                        fail(self.session, job_id, "JOB_TIMEOUT", "Pekerjaan melewati batas waktu.", True)
                    break
                self.stop_event.wait(0.5)
            if process.returncode and not self.stop_event.is_set():
                fail(self.session, job_id, "WORKER_CRASH", "Proses komputasi berhenti.", True)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
        return True

    def loop(self):
        while not self.stop_event.is_set():
            try:
                if not self.run_once():
                    self.stop_event.wait(0.5)
            except Exception:
                self.stop_event.wait(1)

    def start(self):
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=6)


if __name__ == "__main__":
    settings = Settings()
    settings.prepare()
    from aurora_decision.services.settings_store import SettingsStore

    SettingsStore(settings, settings.data_dir / "settings.json").load()
    engine, session = database(settings.data_dir)
    Base.metadata.create_all(engine)
    if "--execute" in sys.argv:
        execute(settings, session, sys.argv[-1])
    else:
        Worker(settings, session).loop()
