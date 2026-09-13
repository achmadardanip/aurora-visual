import csv
import io
import re
import stat
import zipfile
from pathlib import PurePosixPath

from PIL import Image, ImageDraw

from app.config import ServiceError
from app.models.contract import AuroraBundle, sha, strict_json
from app.services.media import TYPES


def safe_path(name):
    path = PurePosixPath(name)
    if (
        "\\" in name
        or "\x00" in name
        or path.is_absolute()
        or ".." in path.parts
        or not path.parts
        or ":" in path.parts[0]
        or str(path) != name
    ):
        raise ServiceError("UNSAFE_ARCHIVE", "Path arsip tidak aman.")
    return path


def export_zip(bundle, media_service, owner):
    bundle = bundle.model_copy(deep=True)
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for image_ref in bundle.input.images:
            path, ref, _ = media_service.resolve(image_ref, owner)
            ext = next(v[1] for v in TYPES.values() if v[0] == ref.media_type)
            name = f"media/{ref.asset_id}.{ext}"
            image_ref.uri = name
            archive.write(path, name)
        for entry in bundle.extensions.get("aurora_visual", {}).get("artifacts", []):
            name = entry["path"]
            safe_path(name)
            if not re.fullmatch(r"artifacts/[0-9a-f-]{36}/[A-Za-z0-9_.-]+", name):
                raise ServiceError("UNSAFE_ARTIFACT", "Path artefak tidak diizinkan.")
            path = media_service.settings.data_dir / name
            if not path.is_file() or path.is_symlink() or sha(path.read_bytes()) != entry["sha256"]:
                raise ServiceError(
                    "ARTIFACT_UNAVAILABLE", "Artefak audit hilang atau berubah; ekspor JSON masih tersedia."
                )
            archive.write(path, name)
        archive.writestr("bundle.json", bundle.model_dump_json(indent=2))
    return stream.getvalue()


def import_data(content, filename, media_service, owner):
    if len(content) > media_service.settings.max_upload * 4:
        raise ServiceError("IMPORT_SIZE", "Berkas impor terlalu besar.", 413)
    entries = {}
    if filename.lower().endswith(".zip"):
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as archive:
                infos = archive.infolist()
                if len(infos) > 100 or sum(i.file_size for i in infos) > 50 * 1024**2:
                    raise ServiceError("ZIP_BOMB", "Ukuran ekstraksi arsip melewati batas.")
                for info in infos:
                    safe_path(info.filename)
                    if (
                        info.filename in entries
                        or stat.S_ISLNK(info.external_attr >> 16)
                        or info.is_dir()
                        or info.flag_bits & 1
                    ):
                        raise ServiceError(
                            "UNSAFE_ARCHIVE", "Entry duplikat, direktori, enkripsi, atau symlink ditolak."
                        )
                    if info.file_size > 20 * 1024**2 or info.file_size > max(1, info.compress_size) * 1000:
                        raise ServiceError("ZIP_BOMB", "Rasio atau ukuran kompresi tidak aman.")
                    if info.filename != "bundle.json" and not info.filename.startswith(
                        ("media/", "artifacts/")
                    ):
                        raise ServiceError("UNSAFE_ARCHIVE", "Entry arsip tidak dikenal.")
                    entries[info.filename] = archive.read(info)
                raw = strict_json(entries["bundle.json"])
        except (zipfile.BadZipFile, KeyError, RuntimeError, OSError) as exc:
            raise ServiceError("INVALID_ARCHIVE", "ZIP tidak valid atau bundle.json tidak tersedia.") from exc
    else:
        raw = strict_json(content)
    if raw.get("schema_version") != "1.0.0":
        raise ServiceError("SCHEMA_VERSION_UNSUPPORTED", "Versi kontrak tidak didukung.")
    bundle = AuroraBundle.model_validate(raw)
    expected_files = {"bundle.json"}
    images = bundle.input.images
    if entries and images:
        for image in images:
            safe_path(image.uri)
            if not image.uri.startswith("media/") or image.uri not in entries:
                raise ServiceError("ASSET_UNAVAILABLE", "ZIP tidak memuat media yang direferensikan.")
            if sha(entries[image.uri]) != image.sha256:
                raise ServiceError("ASSET_HASH_MISMATCH", "Hash media tidak cocok.")
            expected_files.add(image.uri)
    artifacts = bundle.extensions.get("aurora_visual", {}).get("artifacts", [])
    for entry in artifacts:
        safe_path(entry["path"])
        if not re.fullmatch(r"artifacts/[0-9a-f-]{36}/[A-Za-z0-9_.-]+", entry["path"]):
            raise ServiceError("UNSAFE_ARTIFACT", "Path artefak tidak valid.")
        if entries and (entry["path"] not in entries or sha(entries[entry["path"]]) != entry["sha256"]):
            raise ServiceError("ARTIFACT_HASH_MISMATCH", "Artefak tidak lengkap atau hash berbeda.")
        expected_files.add(entry["path"])
    if entries and set(entries) != expected_files:
        raise ServiceError("UNSAFE_ARCHIVE", "Arsip mengandung berkas tanpa referensi.")
    # All archive validation completes before writes; no extractall or URL fetching.
    if entries and images:
        for image in images:
            actual = media_service.upload(entries[image.uri], owner, image.media_type)
            if (actual.width, actual.height) != (image.width, image.height):
                raise ServiceError("ASSET_METADATA_MISMATCH", "Dimensi gambar tidak sesuai.")
            image.uri = actual.uri
    for entry in artifacts:
        if entries:
            path = media_service.settings.data_dir / entry["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() and sha(path.read_bytes()) != entry["sha256"]:
                raise ServiceError("ARTIFACT_CONFLICT", "Artefak lokal dengan ID yang sama berbeda.", 409)
            path.write_bytes(entries[entry["path"]])
    return bundle


def csv_atoms(bundle):
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(
        [
            "case_id",
            "claim_revision",
            "atom_set_id",
            "atom_id",
            "statement",
            "role",
            "visual_status",
            "gold_label",
            "gold_bbox",
            "reviewer",
            "notes",
        ]
    )
    if bundle.analysis:
        assessments = {a.atom_id: a for a in bundle.analysis.visual_assessments}
        for atom in bundle.analysis.atomic_claims:

            def safe(text):
                return "'" + text if text.startswith(("=", "+", "-", "@", "\t", "\r")) else text

            writer.writerow(
                [
                    bundle.case_id,
                    bundle.claim_revision,
                    bundle.analysis.atom_set_id,
                    atom.atom_id,
                    safe(atom.statement),
                    atom.role,
                    assessments[atom.atom_id].visual_status if atom.atom_id in assessments else "",
                    "",
                    "",
                    "",
                    "",
                ]
            )
    return stream.getvalue().encode("utf-8-sig")


def overlay(bundle, media_service, owner, atom_id=None, asset_id=None):
    images = {m.asset_id: m for m in bundle.input.images}
    if not images:
        raise ServiceError("IMAGE_REQUIRED", "Tidak ada gambar.")
    target = images.get(asset_id) if asset_id else next(iter(images.values()))
    if target is None:
        raise ServiceError("IMAGE_REQUIRED", "Gambar tidak ditemukan untuk aset ini.")
    path, _, _ = media_service.resolve(target, owner, preview=True)
    image = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(image)
    if bundle.analysis:
        for assessment in bundle.analysis.visual_assessments:
            if atom_id and assessment.atom_id != atom_id:
                continue
            for color, regions in (
                ("#16a574", assessment.supporting_regions),
                ("#d07a1e", assessment.contradicting_regions),
            ):
                for r in regions:
                    if r.asset_id != target.asset_id:
                        continue
                    x1, y1, x2, y2 = r.bbox
                    draw.rectangle(
                        (x1 * image.width, y1 * image.height, x2 * image.width - 1, y2 * image.height - 1),
                        outline=color,
                        width=max(2, image.width // 150),
                    )
    stream = io.BytesIO()
    image.save(stream, "PNG")
    return stream.getvalue()
