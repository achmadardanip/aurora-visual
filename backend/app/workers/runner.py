"""Persistent queue with process isolation, bounded retries and supervisor leases."""

import os
import re
import subprocess
import sys
import threading
import time
from uuid import uuid4

from sqlalchemy import select

from app.config import Settings
from app.models.contract import AuroraBundle, canonical, sha
from app.models.db import Audit, Base, Case, Job, WorkerState, database
from app.services.cases import snapshot
from app.services.media import MediaService
from app.services.pipeline import analyze


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
                    details={"run_id": job.run_id, "code": code, "input_hash": job.payload_hash},
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
        output = analyze(
            AuroraBundle.model_validate(job.payload),
            job.run_id,
            settings,
            MediaService(settings, session),
            job.owner,
            progress,
        )
        with session.begin() as s:
            s.connection().exec_driver_sql("BEGIN IMMEDIATE")
            current_job = s.get(Job, job_id)
            case = s.get(Case, job.case_id)
            if current_job.status != "running":
                return
            if sha(canonical(case.bundle)) != job.snapshot_hash:
                current_job.status = "failed"
                current_job.error = {
                    "code": "STALE_RESULT",
                    "message": "Kasus telah berubah; hasil lama tidak dipasang.",
                    "retryable": False,
                }
                return
            result = output.model_dump(mode="json")
            current_job.status = "partial" if output.analysis.run.status == "partial" else "succeeded"
            current_job.result, current_job.progress = result, "Analisis selesai"
            case.bundle, case.updated_at = result, time.time()
            snapshot(s, output, "analysis", job.run_id)
            s.add(
                Audit(
                    id=str(uuid4()),
                    case_id=job.case_id,
                    event="run_completed",
                    details={
                        "claim_revision": output.claim_revision,
                        "run_id": job.run_id,
                        "input_hash": job.payload_hash,
                        "output_hash": sha(canonical(result)),
                        "config_hash": output.extensions["aurora_visual"]["config_sha256"],
                        "versions": output.analysis.run.versions,
                        "timing": output.extensions["aurora_visual"]["timing"],
                    },
                )
            )
    except Exception as exc:
        text = str(exc)
        # Only developer-defined contract codes (e.g. DEEPSEEK_KEY_REQUIRED,
        # pure A-Z/0-9/_) are surfaced verbatim; other exception text may
        # embed captions or secrets, so it degrades to the type name.
        detail = text if re.fullmatch(r"[A-Z][A-Z0-9_]{3,63}", text) else type(exc).__name__
        code = (
            "MODEL_UNAVAILABLE"
            if any(w in text.upper() for w in ("CHECKPOINT", "OPENCLIP", "CONFIGURED"))
            else "ANALYSIS_FAILED"
        )
        fail(
            session,
            job_id,
            code,
            "Analisis gagal. Periksa konfigurasi model atau input; riwayat request tetap tersimpan.",
            False,
            detail,
        )
        # Local diagnostics retain type only; no secrets/captions in logs.
        print(f"Job {job_id}: {type(exc).__name__}: {code}", file=sys.stderr)


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
            job.status, job.progress = "running", "Memulai pipeline"
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
        env = {**os.environ, "AURORA_DATA_DIR": str(self.settings.data_dir)}
        process = subprocess.Popen([sys.executable, "-m", "app.workers.runner", "--execute", job_id], env=env)
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
            except Exception as exc:
                print(f"Worker supervisor: {type(exc).__name__}", file=sys.stderr)
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
    # Subprocess workers rebuild Settings from env; re-apply the UI overlay
    # so jobs run with the same configuration as the API process.
    from app.services.settings_store import SettingsStore

    SettingsStore(settings, settings.data_dir / "settings.json").load()
    engine, session = database(settings.data_dir)
    Base.metadata.create_all(engine)
    if "--execute" in sys.argv:
        execute(settings, session, sys.argv[-1])
    else:
        Worker(settings, session).loop()
