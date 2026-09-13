import io
import warnings

from PIL import Image, ImageOps, UnidentifiedImageError

from app.config import ServiceError
from app.models.contract import MediaRef, sha
from app.models.db import Media

TYPES = {"PNG": ("image/png", "png"), "JPEG": ("image/jpeg", "jpg"), "WEBP": ("image/webp", "webp")}
Image.MAX_IMAGE_PIXELS = 25_000_000


class MediaService:
    def __init__(self, settings, session):
        self.settings, self.session = settings, session

    def upload_many(self, contents, owner="local", claimed_types=None):
        if len(contents) > self.settings.max_images:
            raise ServiceError(
                "UPLOAD_COUNT", f"Maksimal {self.settings.max_images} gambar per unggahan.", 413
            )
        refs = [
            self.upload(content, owner, claimed_types[i] if claimed_types else None)
            for i, content in enumerate(contents)
        ]
        return refs

    def thumbnail(self, ref, owner="local", size=320):
        _, stored, _ = self.resolve(ref, owner, preview=True)
        path = self.settings.data_dir / "derived" / f"{stored.asset_id}_thumb.png"
        if not path.is_file():
            image = Image.open(self.settings.data_dir / "derived" / f"{stored.asset_id}.png").convert("RGB")
            image.thumbnail((size, size))
            image.save(path)
        return path

    def upload(self, content, owner="local", claimed_type=None):
        if not content or len(content) > self.settings.max_upload:
            raise ServiceError("UPLOAD_SIZE", "Gambar kosong atau melebihi batas ukuran.", 413)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                image = Image.open(io.BytesIO(content))
                image.verify()
                image = Image.open(io.BytesIO(content))
                if image.format not in TYPES:
                    raise ValueError("Unsupported image format")
                actual, ext = TYPES[image.format]
                if claimed_type and claimed_type not in (actual, "application/octet-stream"):
                    raise ValueError("MIME mismatch")
                orientation = image.getexif().get(274, 1)
                original_size = image.size
                normalized = ImageOps.exif_transpose(image).convert("RGB")
                normalized.load()
        except (
            OSError,
            ValueError,
            UnidentifiedImageError,
            Image.DecompressionBombError,
            Image.DecompressionBombWarning,
        ) as exc:
            raise ServiceError("INVALID_IMAGE", "Berkas bukan PNG/JPEG/WebP yang dapat didekode.") from exc
        digest = sha(content)
        asset_id = "asset_" + digest
        ref = MediaRef(
            asset_id=asset_id,
            sha256=digest,
            media_type=actual,
            width=normalized.width,
            height=normalized.height,
            uri=f"api/v1/media/{asset_id}",
        )
        (self.settings.data_dir / "media" / f"{asset_id}.{ext}").write_bytes(content)
        normalized.save(self.settings.data_dir / "derived" / f"{asset_id}.png")
        with self.session.begin() as s:
            row = s.get(Media, asset_id)
            if row:
                row.owners = sorted(set(row.owners + [owner]))
            else:
                s.add(
                    Media(
                        asset_id=asset_id,
                        ref=ref.model_dump(mode="json"),
                        owners=[owner],
                        transform={
                            "exif_orientation": orientation,
                            "original_size": original_size,
                            "normalized_size": list(normalized.size),
                            "operation": "EXIF transpose to RGB; bbox normalized after transpose",
                        },
                    )
                )
        return ref

    def resolve(self, ref, owner="local", preview=False):
        with self.session() as s:
            row = s.get(Media, ref.asset_id if isinstance(ref, MediaRef) else ref)
        if not row or owner not in row.owners:
            raise ServiceError(
                "ASSET_UNAVAILABLE", "Media belum tersedia. Unggah gambar dengan hash yang sesuai.", 422
            )
        stored = MediaRef.model_validate(row.ref)
        if isinstance(ref, MediaRef) and any(
            getattr(stored, k) != getattr(ref, k) for k in ("sha256", "width", "height", "media_type")
        ):
            raise ServiceError("ASSET_HASH_MISMATCH", "Metadata media tidak sesuai berkas tersimpan.")
        ext = next(v[1] for v in TYPES.values() if v[0] == stored.media_type)
        path = (
            self.settings.data_dir
            / ("derived" if preview else "media")
            / f"{stored.asset_id}.{'png' if preview else ext}"
        )
        if not path.is_file() or (not preview and sha(path.read_bytes()) != stored.sha256):
            raise ServiceError("ASSET_UNAVAILABLE", "Berkas media hilang atau hash berubah.", 422)
        return path, stored, row.transform
