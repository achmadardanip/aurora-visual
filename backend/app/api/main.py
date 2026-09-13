import io
import os
import secrets
import time
from contextlib import asynccontextmanager
from uuid import uuid4

from aurora_visual.mafindo import MafindoClient
from aurora_visual.ocr.engine import capability as ocr_capability
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from PIL import Image
from pydantic import ValidationError
from sqlalchemy import select, text
from starlette.middleware.trustedhost import TrustedHostMiddleware

from app.api.limits import RequestSizeLimit
from app.config import ServiceError, Settings
from app.models.contract import AuroraBundle, sha, strict_json
from app.models.db import Base, Case, Job, Snapshot, WorkerState, database
from app.services.cases import CaseService
from app.services.media import MediaService
from app.services.pipeline import now
from app.services.portable import csv_atoms, export_zip, import_data, overlay
from app.workers.runner import Worker


def create_app(settings=None, embedded_worker=True):
    settings = settings or Settings()
    settings.prepare()
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

    app = FastAPI(title="AURORA Visual", version="1.0.0", lifespan=lifespan)
    app.state.settings, app.state.session, app.state.worker = settings, session, worker
    app.state.media, app.state.cases = media, cases
    origins = [
        origin.strip()
        for origin in os.getenv("AURORA_CORS", "http://localhost:5171,http://127.0.0.1:5171").split(",")
        if origin.strip()
    ]
    if not origins or any(origin == "*" for origin in origins):
        raise ValueError("AURORA_CORS must contain explicit origins")
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.allowed_hosts)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["GET", "POST", "PATCH"],
        allow_headers=["Content-Type", "Authorization", "Idempotency-Key"],
    )

    def error(code, message, status=422, retryable=False, details=None):
        return JSONResponse(
            {"error": {"code": code, "message": message, "retryable": retryable, "details": details or {}}},
            status_code=status,
        )

    app.add_middleware(RequestSizeLimit, max_upload=settings.max_upload)

    @app.middleware("http")
    async def security(request, call_next):
        def hardened(response):
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["Cache-Control"] = "no-store"
            response.headers["Referrer-Policy"] = "no-referrer"
            response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
            response.headers["X-Frame-Options"] = "DENY"
            if settings.public:
                response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
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
        length = request.headers.get("content-length")
        if length and (not length.isdigit() or int(length) > settings.max_upload * 4 + 65536):
            return hardened(error("REQUEST_TOO_LARGE", "Request terlalu besar.", 413))
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
        return error(
            code,
            "Payload tidak sesuai kontrak AURORA 1.0.0.",
            details={"fields": [".".join(map(str, e["loc"])) for e in exc.errors()]},
        )

    @app.exception_handler(ValidationError)
    async def validation_error(_, exc):
        return error(
            "VALIDATION_ERROR",
            "Data melanggar kontrak AURORA.",
            details={"fields": [".".join(map(str, e["loc"])) for e in exc.errors()]},
        )

    @app.exception_handler(ValueError)
    async def value_error(_, exc):
        return error(
            "INVALID_DATA", "Data tidak valid: JSON kanonis, referensi, atau parameter tidak sesuai."
        )

    @app.get("/health")
    def health():
        return {"status": "ok", "service": "aurora-visual", "schema_version": "1.0.0"}

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
                "provider": "fixtures",
                "mode": "demo",
                "capability": "visual",
                "status": "ok",
                "message": "Demo deterministik; bukan evaluasi penelitian",
            },
            {
                "provider": "local-color-v1",
                "mode": "live",
                "capability": "visual",
                "status": "ok",
                "message": "Pengukuran warna bidang; tanpa pengenalan objek",
            },
            {
                "provider": "openclip",
                "mode": "live",
                "capability": "embeddings",
                "status": "ok" if os.getenv("AURORA_OPENCLIP_PRETRAINED") else "unconfigured",
                "message": "Memerlukan checkpoint; bahasa Indonesia belum tervalidasi",
            },
            {
                "provider": "tesseract",
                "mode": "live",
                "capability": "ocr",
                **ocr_capability(settings.ocr_lang),
            },
            {
                "provider": "origin-screen-v1",
                "mode": "live",
                "capability": "metadata, provenance markers, AI/deepfake detector status",
                "status": "ok",
                "message": "Metadata dan marker C2PA/JUMBF diperiksa lokal; detektor Hive opsional dilaporkan terpisah bila dikonfigurasi",
            },
            {
                "provider": "mafindo-v1",
                "mode": "diagnostic",
                "capability": "provider compatibility test only",
                "status": (
                    "disabled" if settings.public else "ok" if settings.mafindo_api_key else "unconfigured"
                ),
                "message": (
                    "Diagnostik dinonaktifkan pada public mode"
                    if settings.public
                    else "Read-only diagnostic tersedia; hasil bukan bukti visual atau verdict faktual"
                    if settings.mafindo_api_key
                    else "Konfigurasikan AURORA_MAFINDO_API_KEY hanya pada server untuk uji kompatibilitas"
                ),
            },
            {
                "provider": "three-state-head",
                "mode": "live",
                "capability": "trained",
                "status": "ok" if os.getenv("AURORA_CHECKPOINT") else "unconfigured",
                "message": "Checkpoint harus lolos metadata/split; smoke tidak boleh untuk live",
            },
            {
                "provider": "ollama",
                "mode": "live",
                "capability": "atomizer",
                "status": "ok" if os.getenv("AURORA_LLM_URL") else "unconfigured",
                "message": "Parser structured output opsional",
            },
        ]
        hive_capabilities = [
            ("hive-v3-vlm", "atomizer and multimodal observations", bool(settings.hive_v3_secret)),
            (
                "hive-v2-origin",
                "AI generation, deepfake, metadata observations",
                bool(settings.hive_v2_key("origin")),
            ),
            ("hive-v2-ocr", "OCR", bool(settings.hive_v2_key("ocr"))),
            ("hive-v2-object", "common object detection", bool(settings.hive_v2_key("object"))),
            ("hive-v2-scene", "contextual scene classification", bool(settings.hive_v2_key("scene"))),
            ("hive-v2-people", "people-count category", bool(settings.hive_v2_key("people"))),
            ("hive-v2-logo", "logo and logo-location proposals", bool(settings.hive_v2_key("logo"))),
            (
                "hive-v2-celebrity",
                "probabilistic celebrity proposal",
                bool(settings.hive_v2_key("celebrity")),
            ),
            ("hive-v2-translation", "optional shadow translation", bool(settings.hive_v2_key("translation"))),
        ]
        caps.extend(
            {
                "provider": provider,
                "mode": "external opt-in",
                "capability": capability,
                "status": "ok" if settings.hive_enabled and configured else "unconfigured",
                "message": (
                    "Konfigurasi server tersedia; readiness tidak menghubungi provider berbayar"
                    if settings.hive_enabled and configured
                    else "Nonaktif atau project key server belum dikonfigurasi"
                ),
            }
            for provider, capability, configured in hive_capabilities
        )
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

    @app.get("/api/v1/providers/mafindo/latest")
    def mafindo_latest(request: Request, limit: int = 1):
        """Opt-in diagnostic bridge only; never expose it from the public service."""
        if settings.public:
            raise ServiceError(
                "DIAGNOSTIC_DISABLED",
                "Diagnostik provider hanya tersedia pada instance lokal tepercaya.",
                404,
            )
        client = MafindoClient(settings.mafindo_api_key, settings.mafindo_timeout)
        return client.latest(limit)

    @app.post("/api/v1/media")
    async def upload(request: Request, image: UploadFile = File(...)):
        content = await image.read(settings.max_upload + 1)
        return media.upload(content, request.state.owner, image.content_type)

    @app.get("/api/v1/media/{asset_id}")
    def get_media(asset_id: str, request: Request, preview: bool = False):
        path, ref, _ = media.resolve(asset_id, request.state.owner, preview)
        return FileResponse(path, media_type="image/png" if preview else ref.media_type)

    @app.post("/api/v1/analyze", status_code=202)
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

    @app.get("/api/v1/jobs/{job_id}/bundle")
    def job_bundle(job_id: str, request: Request):
        job = get_job(job_id, request)
        if not job["result"]:
            raise ServiceError("RESULT_UNAVAILABLE", "Hasil belum tersedia.", 409)
        return JSONResponse(
            job["result"], headers={"Content-Disposition": "attachment; filename=bundle.json"}
        )

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
                "status": r.bundle["analysis"]["run"]["status"] if r.bundle["analysis"] else "pending",
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
                    "parent_id": r.parent_id,
                    "reason": r.reason,
                    "bundle": r.bundle,
                }
                for r in rows
            ]

    @app.patch("/api/v1/cases/{case_id}/atoms", response_model=AuroraBundle)
    async def correct(case_id: str, request: Request):
        data = strict_json(await request.body())
        if set(data) != {"atomic_claims", "expected_atom_set_id", "reason"}:
            raise ServiceError(
                "INVALID_CORRECTION", "Koreksi memerlukan atom, expected_atom_set_id, dan alasan."
            )
        return cases.correct(
            case_id, request.state.owner, data["atomic_claims"], data["expected_atom_set_id"], data["reason"]
        )

    @app.get("/api/v1/cases/{case_id}/export")
    def export(case_id: str, request: Request, format: str = "json", atom_id: str | None = None):
        bundle = cases.get(case_id, request.state.owner)
        if format == "json":
            content, mime, ext = bundle.model_dump_json(indent=2).encode(), "application/json", "json"
        elif format == "zip":
            content, mime, ext = export_zip(bundle, media, request.state.owner), "application/zip", "zip"
        elif format == "csv":
            content, mime, ext = csv_atoms(bundle), "text/csv", "csv"
        elif format == "overlay":
            content, mime, ext = overlay(bundle, media, request.state.owner, atom_id), "image/png", "png"
        else:
            raise ServiceError("INVALID_FORMAT", "Format ekspor tidak dikenal.")
        return Response(
            content,
            media_type=mime,
            headers={"Content-Disposition": f"attachment; filename=aurora-{case_id}.{ext}"},
        )

    @app.post("/api/v1/import", response_model=dict)
    async def import_bundle(request: Request, file: UploadFile = File(...)):
        content = await file.read(settings.max_upload * 4 + 1)
        bundle = import_data(content, file.filename or "", media, request.state.owner)
        result = cases.import_bundle(bundle, request.state.owner)
        available = False
        if bundle.input.image:
            try:
                media.resolve(bundle.input.image, request.state.owner)
                available = True
            except ServiceError:
                pass
        return {"bundle": result.model_dump(mode="json"), "asset_available": available}

    @app.post("/api/v1/demo/{fixture}", response_model=AuroraBundle)
    def demo(fixture: str, request: Request):
        if fixture not in ("supported", "contradicted", "unobservable"):
            raise ServiceError("FIXTURE_UNKNOWN", "Fixture tidak ditemukan.", 404)
        image = Image.new("RGB", (800, 600), "#e02222" if fixture == "supported" else "#2255dd")
        content = io.BytesIO()
        image.save(content, "PNG")
        ref = media.upload(content.getvalue(), request.state.owner)
        caption = (
            "Mahasiswa melakukan aksi di Monas pada September 2026"
            if fixture == "unobservable"
            else "Bidang ini berwarna merah"
        )
        return AuroraBundle(
            schema_version="1.0.0",
            case_id=str(uuid4()),
            claim_revision=1,
            mode="demo",
            created_at=now(),
            input={"claim_text": caption, "language": "id", "image": ref, "as_of": None},
            analysis=None,
            retrieval=None,
            decision=None,
            warnings=[
                {
                    "code": "SYNTHETIC_FIXTURE",
                    "message": "Gambar dan label demonstrasi sintetis, bukan foto peristiwa atau bukti penelitian.",
                    "component": "demo",
                }
            ],
            extensions={"aurora_visual": {"fixture": fixture}},
        )

    @app.get("/api/v1/schema")
    def schema():
        return AuroraBundle.model_json_schema()

    return app


app = create_app()
