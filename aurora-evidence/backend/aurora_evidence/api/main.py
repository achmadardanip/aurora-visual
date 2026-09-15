"""AURORA Evidence API (module 2): /api/v1/retrieve + shared endpoints."""

import os
import secrets
import time
import zipfile
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from io import BytesIO
from uuid import uuid4

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import ValidationError
from sqlalchemy import select, text
from starlette.middleware.trustedhost import TrustedHostMiddleware

from aurora_evidence.config import UI_FIELDS, ServiceError, Settings
from aurora_evidence.contract import AuroraBundle, sha, strict_json
from aurora_evidence.models.db import Base, Case, Job, Snapshot, WorkerState, database
from aurora_evidence.services.cases import CaseService, dump
from aurora_evidence.services.media import MediaService
from aurora_evidence.services.settings_store import MASKED, SettingsStore
from aurora_evidence.workers.runner import Worker


def create_app(settings=None, embedded_worker=True):
    settings = settings or Settings()
    settings.prepare()
    store = SettingsStore(settings, settings.data_dir / "settings.json")
    store.load()
    engine, session = database(settings.data_dir)
    Base.metadata.create_all(engine)
    media = MediaService(settings, session)
    cases = CaseService(settings, session, media)
    worker = Worker(settings, session)

    @asynccontextmanager
    async def lifespan(app):
        if embedded_worker:
            worker.start()
        yield
        worker.stop()
        engine.dispose()

    app = FastAPI(title="AURORA Evidence", version="1.0.0", lifespan=lifespan)
    app.state.settings, app.state.session, app.state.worker = settings, session, worker
    app.state.media, app.state.cases, app.state.store = media, cases, store
    origins = [
        origin.strip()
        for origin in os.getenv("AURORA_CORS", "http://localhost:5172,http://127.0.0.1:5172").split(",")
        if origin.strip()
    ]
    if not origins or any(origin == "*" for origin in origins):
        raise ValueError("AURORA_CORS must contain explicit origins")
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.allowed_hosts)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["GET", "POST", "PUT", "PATCH"],
        allow_headers=["Content-Type", "Authorization", "Idempotency-Key"],
    )

    def error(code, message, status=422, retryable=False, details=None):
        return JSONResponse(
            {"error": {"code": code, "message": message, "retryable": retryable, "details": details or {}}},
            status_code=status,
        )

    @app.middleware("http")
    async def security(request, call_next):
        def hardened(response):
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["Cache-Control"] = "no-store"
            response.headers["Referrer-Policy"] = "no-referrer"
            response.headers["X-Frame-Options"] = "DENY"
            return response

        if (
            request.method != "GET"
            and request.headers.get("origin")
            and request.headers["origin"] not in origins
        ):
            return hardened(error("ORIGIN_FORBIDDEN", "Origin tidak diizinkan.", 403))
        if settings.public and request.method != "OPTIONS" and request.url.path not in ("/health",):
            expected = "Bearer " + settings.token
            if not secrets.compare_digest(request.headers.get("authorization", ""), expected):
                return hardened(error("AUTH_REQUIRED", "Token akses diperlukan.", 401))
        request.state.owner = sha(settings.token) if settings.public else "local"
        return hardened(await call_next(request))

    @app.exception_handler(ServiceError)
    async def service_error(_, exc):
        return error(exc.code, exc.message, exc.status, exc.retryable)

    @app.exception_handler(RequestValidationError)
    async def request_error(_, exc):
        code = (
            "SCHEMA_VERSION_UNSUPPORTED"
            if any("schema_version" in e["loc"] for e in exc.errors())
            else "VALIDATION_ERROR"
        )
        return error(code, "Payload tidak sesuai kontrak AURORA 1.0.0.")

    @app.exception_handler(ValidationError)
    async def validation_error(_, exc):
        return error("VALIDATION_ERROR", "Data melanggar kontrak AURORA.")

    @app.exception_handler(ValueError)
    async def value_error(_, exc):
        return error("INVALID_DATA", "Data tidak valid: JSON kanonis atau referensi tidak sesuai.")

    @app.get("/health")
    def health():
        return {"status": "ok", "service": "aurora-evidence", "schema_version": "1.0.0"}

    @app.get("/ready")
    def ready():
        try:
            with session() as s:
                s.execute(text("SELECT 1"))
                state = s.get(WorkerState, "local")
                worker_ok = bool(state and time.time() - state.heartbeat < 10)
            database_ok = True
        except Exception:
            database_ok, worker_ok = False, False
        caps = [
            {
                "provider": "local-corpus",
                "mode": "live",
                "capability": "fact_check_search (BM25 lokal)",
                "status": "ok" if os.path.isfile(settings.corpus_path) else "unconfigured",
                "message": "Pencarian korpus lokal nyata (BM25)"
                if os.path.isfile(settings.corpus_path)
                else "Berkas korpus tidak ditemukan: " + settings.corpus_path,
            },
            {
                "provider": "tavily",
                "mode": "external opt-in",
                "capability": "web_search",
                "status": "ok" if settings.tavily_api_key else "unconfigured",
                "message": "Pencarian web Tavily siap"
                if settings.tavily_api_key
                else "Setel TAVILY_API_KEY server-side untuk mengaktifkan pencarian web",
            },
            {
                "provider": "hive-v3",
                "mode": "external opt-in",
                "capability": "ai_generation_detection_image",
                "status": "ok" if settings.hive_api_key else "unconfigured",
                "message": "Deteksi AI gambar Hive V3 siap"
                if settings.hive_api_key
                else "Setel HIVE_API_KEY server-side untuk deteksi AI gambar",
            },
            {
                "provider": "gptzero",
                "mode": "external opt-in",
                "capability": "ai_generation_detection_text",
                "status": "ok" if settings.gptzero_api_key else "unconfigured",
                "message": "Deteksi AI teks GPTZero siap"
                if settings.gptzero_api_key
                else "Setel GPTZERO_API_KEY server-side untuk deteksi AI teks",
            },
            {
                "provider": "local-image-index",
                "mode": "live",
                "capability": "local_image_index (dHash)",
                "status": "ok",
                "message": "Pencarian gambar korpus lokal; bukan reverse search seluruh web",
            },
            {
                "provider": "bounded-fetcher",
                "mode": "live",
                "capability": "content_fetch",
                "status": "ok" if settings.web_fetch else "disabled",
                "message": "Pengambilan halaman web dibatasi SSRF"
                if settings.web_fetch
                else "Pengambilan halaman web nonaktif (snippet saja)",
            },
        ]
        status = (
            "not_ready"
            if not database_ok or not worker_ok
            else "degraded"
            if any(c["status"] in {"unconfigured", "unavailable", "failed"} for c in caps)
            else "ready"
        )
        return JSONResponse(
            {
                "status": status,
                "database": "ok" if database_ok else "failed",
                "worker": "ok" if worker_ok else "unavailable",
                "capabilities": caps,
            },
            status_code=503 if status == "not_ready" else 200,
        )

    @app.get("/api/v1/settings")
    def get_settings():
        if settings.public:
            raise ServiceError("SETTINGS_DISABLED", "Pengaturan hanya tersedia pada instance lokal.", 404)
        return {
            "values": store.read(),
            "fields": {name: {"kind": kind, "secret": secret} for name, (kind, secret) in UI_FIELDS.items()},
            "masked": MASKED,
            "env_only": {
                "data_dir": str(settings.data_dir),
                "public": settings.public,
                "allowed_hosts": settings.allowed_hosts,
                "cors": origins,
            },
        }

    @app.put("/api/v1/settings")
    async def put_settings(request: Request):
        if settings.public:
            raise ServiceError("SETTINGS_DISABLED", "Pengaturan hanya tersedia pada instance lokal.", 404)
        merged = store.update(strict_json(await request.body()))
        return {
            "values": {
                name: (MASKED if UI_FIELDS[name][1] and value else value) for name, value in merged.items()
            },
            "applied": True,
        }

    @app.post("/api/v1/media")
    async def upload(request: Request, images: list[UploadFile] = File(...)):
        if len(images) > settings.max_images:
            raise ServiceError("UPLOAD_COUNT", f"Maksimal {settings.max_images} gambar per unggahan.", 413)
        contents = [await f.read(settings.max_upload + 1) for f in images]
        types = [f.content_type for f in images]
        refs = media.upload_many(contents, request.state.owner, types)
        return {"images": [r.model_dump(mode="json") for r in refs]}

    @app.get("/api/v1/media/{asset_id}")
    def get_media(asset_id: str, request: Request, preview: bool = False):
        path, ref, _ = media.resolve(asset_id, request.state.owner, preview)
        return FileResponse(path, media_type="image/png" if preview else ref.media_type)

    @app.get("/api/v1/media/{asset_id}/thumbnail")
    def get_thumbnail(asset_id: str, request: Request):
        path = media.thumbnail(asset_id, request.state.owner)
        return FileResponse(path, media_type="image/png")

    @app.post("/api/v1/retrieve", status_code=202)
    async def submit(bundle: AuroraBundle, request: Request):
        raw = strict_json(await request.body())
        job = cases.enqueue(raw, request.state.owner, request.headers.get("Idempotency-Key"))
        return {
            "job_id": job.job_id,
            "case_id": job.case_id,
            "claim_revision": job.payload["claim_revision"],
            "status": "queued",
        }

    @app.get("/api/v1/jobs/{job_id}")
    def get_job(job_id: str, request: Request):
        with session() as s:
            job = s.get(Job, job_id)
            if not job or job.owner != request.state.owner:
                raise ServiceError("JOB_NOT_FOUND", "Pekerjaan tidak ditemukan.", 404)
            return {
                "job_id": job.job_id,
                "status": job.status,
                "result": job.result,
                "error": job.error,
                "progress": job.progress,
                "attempts": job.attempts,
            }

    @app.post("/api/v1/jobs/{job_id}/cancel")
    def cancel(job_id: str, request: Request):
        with session.begin() as s:
            job = s.get(Job, job_id)
            if not job or job.owner != request.state.owner:
                raise ServiceError("JOB_NOT_FOUND", "Pekerjaan tidak ditemukan.", 404)
            if job.status in ("queued", "running"):
                job.status = "failed"
                job.error = {"code": "CANCELLED", "message": "Dibatalkan pengguna.", "retryable": False}
            return {"job_id": job.job_id, "status": job.status}

    @app.get("/api/v1/cases")
    def list_cases(request: Request, limit: int = 50):
        with session() as s:
            rows = s.scalars(
                select(Case)
                .where(Case.owner == request.state.owner)
                .order_by(Case.updated_at.desc())
                .limit(max(1, min(100, limit)))
            ).all()
        return [
            {
                "case_id": r.case_id,
                "claim_revision": r.revision,
                "caption": r.bundle["input"]["claim_text"],
                "mode": r.bundle["mode"],
                "updated_at": r.updated_at,
                "status": r.bundle["retrieval"]["run"]["status"] if r.bundle["retrieval"] else "pending",
            }
            for r in rows
        ]

    @app.get("/api/v1/cases/{case_id}", response_model=AuroraBundle)
    def get_case(case_id: str, request: Request):
        return cases.get(case_id, request.state.owner)

    @app.get("/api/v1/cases/{case_id}/history")
    def history(case_id: str, request: Request):
        cases.get(case_id, request.state.owner)
        with session() as s:
            rows = s.scalars(
                select(Snapshot).where(Snapshot.case_id == case_id).order_by(Snapshot.created_at.desc())
            ).all()
        return [
            {
                "snapshot_id": r.id,
                "kind": r.kind,
                "claim_revision": r.revision,
                "created_at": r.created_at,
                "reason": r.reason,
                "bundle": r.bundle,
            }
            for r in rows
        ]

    @app.get("/api/v1/cases/{case_id}/export")
    def export(case_id: str, request: Request, format: str = "json"):
        bundle = cases.get(case_id, request.state.owner)
        if format == "json":
            content, mime, ext = bundle.model_dump_json(indent=2).encode(), "application/json", "json"
        elif format == "zip":
            buffer = BytesIO()
            with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("bundle.json", bundle.model_dump_json(indent=2))
                for image in bundle.input.images:
                    try:
                        path, ref, _ = media.resolve(image, request.state.owner)
                        archive.write(path, f"media/{image.asset_id}.{path.suffix.lstrip('.')}")
                    except ServiceError:
                        continue
            content, mime, ext = buffer.getvalue(), "application/zip", "zip"
        elif format == "csv":
            import csv as _csv
            import io as _io

            rows = _io.StringIO(newline="")
            writer = _csv.writer(rows, quoting=_csv.QUOTE_ALL)
            writer.writerow(
                [
                    "evidence_id",
                    "kind",
                    "publisher",
                    "url",
                    "title",
                    "published_at",
                    "temporal_eligible",
                    "independence_group_id",
                    "relevance_score",
                    "status",
                ]
            )
            for e in bundle.retrieval.evidence_list if bundle.retrieval else []:
                writer.writerow(
                    [
                        e.evidence_id,
                        e.source.kind,
                        e.source.publisher or "",
                        e.source.url or "",
                        e.source.title,
                        e.source.published_at or "",
                        "" if e.provenance.temporal_eligible is None else e.provenance.temporal_eligible,
                        e.provenance.independence_group_id,
                        "" if e.relevance_score is None else e.relevance_score,
                        e.content.status,
                    ]
                )
            content, mime, ext = rows.getvalue().encode(), "text/csv", "csv"
        else:
            raise ServiceError("INVALID_FORMAT", "Format ekspor tidak dikenal.")
        return Response(
            content,
            media_type=mime,
            headers={"Content-Disposition": f"attachment; filename=aurora-evidence-{case_id}.{ext}"},
        )

    @app.post("/api/v1/import", response_model=dict)
    async def import_bundle(request: Request, file: UploadFile = File(...)):
        content = await file.read(settings.max_upload * 4 + 1)
        if file.filename and file.filename.endswith(".zip"):
            with zipfile.ZipFile(BytesIO(content)) as archive:
                names = archive.namelist()
                if "bundle.json" not in names or any(n.startswith("/") or ".." in n for n in names):
                    raise ServiceError(
                        "INVALID_ARCHIVE", "ZIP harus memuat bundle.json tanpa path berbahaya."
                    )
                bundle = AuroraBundle.model_validate(strict_json(archive.read("bundle.json")))
                for name in names:
                    if name.startswith("media/"):
                        media.upload(archive.read(name), request.state.owner)
        else:
            bundle = AuroraBundle.model_validate(strict_json(content))
        result = cases.import_bundle(bundle, request.state.owner)
        available = []
        for image in bundle.input.images:
            try:
                media.resolve(image, request.state.owner)
                available.append(image.asset_id)
            except ServiceError:
                pass
        return {"bundle": dump(result), "assets_available": available}

    @app.post("/api/v1/demo/{fixture}", response_model=AuroraBundle)
    def demo(fixture: str, request: Request):
        if fixture not in ("fact_check", "news"):
            raise ServiceError("FIXTURE_UNKNOWN", "Fixture tidak ditemukan.", 404)
        caption = (
            "Bidang ini berwarna merah"
            if fixture == "fact_check"
            else "Pemerintah mengumumkan kebijakan energi baru bulan depan"
        )
        return AuroraBundle(
            schema_version="1.0.0",
            case_id=str(uuid4()),
            claim_revision=1,
            mode="demo",
            created_at=datetime.now(timezone.utc).isoformat(),
            input={"claim_text": caption, "language": "id", "images": [], "as_of": None},
            analysis=None,
            retrieval=None,
            decision=None,
            warnings=[
                {
                    "code": "SYNTHETIC_FIXTURE",
                    "message": "Fixture demo sintetis untuk memeriksa alur pencarian bukti.",
                    "component": "demo",
                }
            ],
            extensions={"aurora_evidence": {"fixture": fixture}},
        )

    @app.get("/api/v1/schema")
    def schema():
        return AuroraBundle.model_json_schema()

    return app


app = create_app()
