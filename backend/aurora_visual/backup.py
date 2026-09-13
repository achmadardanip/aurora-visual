"""Operational helpers for backup creation, verification, and restore."""

import hashlib
import json
import os
import re
import sqlite3
import tarfile
import tempfile
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

BACKUP_FORMAT = "aurora-backup-v2"
SUPPORTED_FORMATS = {"aurora-backup-v1", BACKUP_FORMAT}
MAX_BACKUP_MEMBERS = 100_000
MAX_MANIFEST_BYTES = 8 * 1024**2
MAX_MEMBER_BYTES = 32 * 1024**3
MAX_BACKUP_BYTES = 64 * 1024**3
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _digest_stream(stream, output=None):
    digest = hashlib.sha256()
    size = 0
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        size += len(chunk)
        if size > MAX_MEMBER_BYTES:
            raise ValueError("Backup member exceeds the size limit")
        digest.update(chunk)
        if output is not None:
            output.write(chunk)
    return size, digest.hexdigest()


def _archive_stream(archive, member):
    stream = archive.extractfile(member)
    if stream is None:
        raise ValueError(f"Missing backup member: {member.name}")
    return closing(stream)


def file_digest(path: Path):
    with path.open("rb") as stream:
        return _digest_stream(stream)[1]


def safe_files(data_dir: Path, excluded: set[Path]):
    for path in sorted(data_dir.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Refusing symbolic link in data directory: {path.relative_to(data_dir)}")
        if not path.is_file() or path in excluded:
            continue
        yield path


def _stage_files(data_dir: Path, staging: Path, excluded: set[Path]):
    staged = []
    total_size = 0
    for source in safe_files(data_dir, excluded):
        relative = source.relative_to(data_dir)
        if _safe_member_name(relative.as_posix()).as_posix() != relative.as_posix():
            raise ValueError(f"Unsafe backup file path: {relative}")
        destination = staging / "files" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as input_stream, destination.open("xb") as output_stream:
            size, digest = _digest_stream(input_stream, output_stream)
        total_size += size
        if total_size > MAX_BACKUP_BYTES:
            raise ValueError("Backup exceeds the total size limit")
        staged.append({"path": relative.as_posix(), "size": size, "sha256": digest})
    return staged


def create_backup(data_dir: Path, output_dir: Path):
    requested_data_dir = Path(data_dir).absolute()
    if requested_data_dir.is_symlink():
        raise ValueError("Backup data directory must not be a symbolic link")
    data_dir = requested_data_dir.resolve()
    database = data_dir / "aurora.db"
    if database.is_symlink():
        raise ValueError("Backup database must not be a symbolic link")
    if not database.is_file():
        raise ValueError(f"Database not found: {database}")
    output_dir = output_dir.resolve()
    try:
        output_dir.relative_to(data_dir)
    except ValueError:
        pass
    else:
        raise ValueError("Backup output directory must be outside the data directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    destination = output_dir / f"aurora-backup-{timestamp}.tar.gz"
    with tempfile.TemporaryDirectory(prefix="aurora-backup-") as temporary:
        staging = Path(temporary)
        database_copy = staging / "aurora.db"
        with sqlite3.connect(database) as source, sqlite3.connect(database_copy) as target:
            source.backup(target)
        with sqlite3.connect(database_copy) as connection:
            if connection.execute("PRAGMA integrity_check").fetchone() != ("ok",):
                raise ValueError("SQLite backup failed integrity_check")
        files = _stage_files(
            data_dir,
            staging,
            {
                database,
                database.with_name(database.name + "-wal"),
                database.with_name(database.name + "-shm"),
            },
        )
        manifest = {
            "format": BACKUP_FORMAT,
            "created_at": datetime.now(UTC).isoformat(),
            "database": {
                "path": "aurora.db",
                "size": database_copy.stat().st_size,
                "sha256": file_digest(database_copy),
            },
            "files": files,
        }
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary_archive = output_dir / f".{destination.name}.{os.getpid()}.tmp"
        try:
            with tarfile.open(temporary_archive, "x:gz") as archive:
                archive.add(database_copy, arcname="aurora.db", recursive=False)
                archive.add(manifest_path, arcname="manifest.json", recursive=False)
                for row in files:
                    archive.add(
                        staging / "files" / row["path"],
                        arcname="files/" + row["path"],
                        recursive=False,
                    )
            verify_backup(temporary_archive)
            os.link(temporary_archive, destination)
        finally:
            temporary_archive.unlink(missing_ok=True)
    return destination, file_digest(destination)


def _safe_member_name(name):
    if not isinstance(name, str) or not name or "\\" in name or "\x00" in name:
        raise ValueError("Unsafe backup member")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "." in path.parts or any(not part for part in path.parts):
        raise ValueError("Unsafe backup member")
    return path


def _manifest_rows(manifest):
    if not isinstance(manifest, dict) or set(manifest) != {
        "format",
        "created_at",
        "database",
        "files",
    }:
        raise ValueError("Invalid backup manifest schema")
    backup_format = manifest["format"]
    if backup_format not in SUPPORTED_FORMATS or not isinstance(manifest["created_at"], str):
        raise ValueError("Unsupported backup format")
    if not 10 <= len(manifest["created_at"]) <= 64:
        raise ValueError("Invalid backup creation time")
    database = manifest["database"]
    files = manifest["files"]
    if not isinstance(database, dict) or not isinstance(files, list):
        raise ValueError("Invalid backup manifest schema")
    if len(files) > MAX_BACKUP_MEMBERS - 2:
        raise ValueError("Backup manifest contains too many files")
    allowed_database_keys = {"path", "sha256", "size"}
    required_database_keys = allowed_database_keys if backup_format == BACKUP_FORMAT else {"path", "sha256"}
    if set(database) - allowed_database_keys or not required_database_keys <= set(database):
        raise ValueError("Invalid backup database manifest")
    rows = [("aurora.db", database)]
    names = {"aurora.db"}
    if database["path"] != "aurora.db":
        raise ValueError("Invalid backup database path")
    for row in files:
        if not isinstance(row, dict) or set(row) != {"path", "size", "sha256"}:
            raise ValueError("Invalid backup file manifest")
        path = _safe_member_name(row["path"])
        if path.as_posix() != row["path"]:
            raise ValueError("Invalid backup file path")
        name = "files/" + row["path"]
        if name in names:
            raise ValueError("Duplicate backup manifest path")
        names.add(name)
        rows.append((name, row))
    return rows


def _validate_row(name, row, member):
    digest = row.get("sha256")
    size = row.get("size")
    if not isinstance(digest, str) or not SHA256_PATTERN.fullmatch(digest):
        raise ValueError(f"Invalid backup checksum: {name}")
    if size is not None and (not isinstance(size, int) or isinstance(size, bool) or size < 0):
        raise ValueError(f"Invalid backup size: {name}")
    if size is not None and member.size != size:
        raise ValueError(f"Backup size mismatch: {name}")


def _open_verified_archive(archive_path):
    archive_path = archive_path.resolve()
    try:
        archive = tarfile.open(archive_path, "r:gz")
    except tarfile.TarError as exc:
        raise ValueError("Invalid backup archive") from exc
    try:
        members = []
        names = set()
        total_size = 0
        for member in archive:
            if len(members) >= MAX_BACKUP_MEMBERS:
                raise ValueError("Backup contains too many members")
            path = _safe_member_name(member.name)
            if path.as_posix() != member.name or member.name in names:
                raise ValueError("Unsafe or duplicate backup member")
            if not member.isfile() or member.issym() or member.islnk():
                raise ValueError("Unsafe backup member")
            if member.size > MAX_MEMBER_BYTES:
                raise ValueError("Backup member exceeds the size limit")
            total_size += member.size
            if total_size > MAX_BACKUP_BYTES:
                raise ValueError("Backup exceeds the total size limit")
            members.append(member)
            names.add(member.name)
        if "manifest.json" not in names or "aurora.db" not in names:
            raise ValueError("Invalid backup member set")
        manifest_member = archive.getmember("manifest.json")
        if manifest_member.size > MAX_MANIFEST_BYTES:
            raise ValueError("Backup manifest exceeds the size limit")
        with _archive_stream(archive, manifest_member) as manifest_stream:
            try:
                manifest = json.load(manifest_stream)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("Invalid backup manifest JSON") from exc
        rows = _manifest_rows(manifest)
        expected_names = {name for name, _ in rows} | {"manifest.json"}
        if expected_names != set(names):
            raise ValueError("Backup manifest does not match archive members")
        for name, row in rows:
            _validate_row(name, row, archive.getmember(name))
        return archive, rows
    except Exception:
        archive.close()
        raise


def _verify_member(archive, name, row, output=None):
    member = archive.getmember(name)
    with _archive_stream(archive, member) as stream:
        size, digest = _digest_stream(stream, output)
    if size != member.size or digest != row["sha256"]:
        raise ValueError(f"Backup checksum mismatch: {name}")


def _check_database(path):
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
            if connection.execute("PRAGMA integrity_check").fetchone() != ("ok",):
                raise ValueError("Backup database failed integrity_check")
    except sqlite3.DatabaseError as exc:
        raise ValueError("Backup database failed integrity_check") from exc


def verify_backup(archive_path: Path):
    archive, rows = _open_verified_archive(archive_path)
    try:
        for name, row in rows:
            _verify_member(archive, name, row)
        with _archive_stream(archive, archive.getmember("aurora.db")) as database_stream:
            with tempfile.NamedTemporaryFile(suffix=".db") as temporary:
                _digest_stream(database_stream, temporary)
                temporary.flush()
                _check_database(temporary.name)
    finally:
        archive.close()
    return len(rows)


def restore_backup(archive_path: Path, data_dir: Path):
    requested_data_dir = Path(data_dir).absolute()
    if requested_data_dir.is_symlink():
        raise ValueError("Restore destination must not be a symbolic link")
    data_dir = requested_data_dir.resolve()
    existed = data_dir.exists()
    if existed:
        if data_dir.is_symlink() or not data_dir.is_dir():
            raise ValueError("Restore destination must be a directory")
        if any(data_dir.iterdir()):
            raise ValueError("Restore destination must be empty")
    data_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{data_dir.name}-restore-", dir=data_dir.parent) as temporary:
        staging = Path(temporary)
        archive, rows = _open_verified_archive(archive_path)
        try:
            for name, row in rows:
                destination = staging / ("aurora.db" if name == "aurora.db" else name.removeprefix("files/"))
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open("xb") as output:
                    _verify_member(archive, name, row, output)
            _check_database(staging / "aurora.db")
        finally:
            archive.close()
        if existed:
            data_dir.rmdir()
        os.replace(staging, data_dir)
    return len(rows)
