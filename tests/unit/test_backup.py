import io
import json
import tarfile
from pathlib import Path

import pytest
from app.api.main import create_app
from app.config import Settings
from app.models.db import Base, database
from aurora_visual.backup import create_backup, restore_backup, verify_backup
from fastapi.testclient import TestClient


def test_backup_and_verify_round_trip(tmp_path):
    data_dir = tmp_path / "data"
    app = create_app(Settings(data_dir=data_dir), embedded_worker=False)
    with TestClient(app) as client:
        response = client.post("/api/v1/demo/supported")
        assert response.status_code == 200
    archive, checksum = create_backup(data_dir, tmp_path / "backups")
    assert archive.stat().st_size > 0
    assert len(checksum) == 64
    assert verify_backup(archive) >= 1


def test_backup_database_copy_can_be_opened(tmp_path):
    data_dir = tmp_path / "data"
    settings = Settings(data_dir=data_dir)
    settings.prepare()
    engine, _ = database(data_dir)
    Base.metadata.create_all(engine)
    engine.dispose()
    backup_dir = tmp_path / "backups"
    archive, _ = create_backup(data_dir, backup_dir)
    assert any(Path(backup_dir).iterdir())
    assert verify_backup(archive) >= 1
    restored = tmp_path / "restored"
    assert restore_backup(archive, restored) >= 1
    with database(restored)[0].connect() as connection:
        assert connection.exec_driver_sql("PRAGMA integrity_check").scalar_one() == "ok"


def test_backup_excludes_sqlite_sidecars_and_rejects_nested_output(tmp_path):
    data_dir = tmp_path / "data"
    settings = Settings(data_dir=data_dir)
    settings.prepare()
    engine, _ = database(data_dir)
    Base.metadata.create_all(engine)
    engine.dispose()
    (data_dir / "aurora.db-wal").write_bytes(b"transient wal")
    (data_dir / "aurora.db-shm").write_bytes(b"transient shm")

    archive, _ = create_backup(data_dir, tmp_path / "backups")
    assert verify_backup(archive) == 1
    with pytest.raises(ValueError, match="outside the data directory"):
        create_backup(data_dir, data_dir / "backups")


def test_backup_rejects_symbolic_links(tmp_path):
    data_dir = tmp_path / "data"
    settings = Settings(data_dir=data_dir)
    settings.prepare()
    engine, _ = database(data_dir)
    Base.metadata.create_all(engine)
    engine.dispose()
    target = tmp_path / "outside.txt"
    target.write_text("outside")
    (data_dir / "media" / "link").symlink_to(target)

    with pytest.raises(ValueError, match="symbolic link"):
        create_backup(data_dir, tmp_path / "backups")


def test_backup_rejects_symbolic_database(tmp_path):
    real_data = tmp_path / "real"
    settings = Settings(data_dir=real_data)
    settings.prepare()
    engine, _ = database(real_data)
    Base.metadata.create_all(engine)
    engine.dispose()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "aurora.db").symlink_to(real_data / "aurora.db")
    with pytest.raises(ValueError, match="database must not be a symbolic link"):
        create_backup(data_dir, tmp_path / "backups")


def _write_archive(path, manifest, members):
    with tarfile.open(path, "w:gz") as archive:
        manifest_bytes = json.dumps(manifest).encode()
        info = tarfile.TarInfo("manifest.json")
        info.size = len(manifest_bytes)
        archive.addfile(info, io.BytesIO(manifest_bytes))
        for name, content in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))


def test_backup_rejects_malformed_manifest(tmp_path):
    archive = tmp_path / "bad.tar.gz"
    manifest = {
        "format": "aurora-backup-v2",
        "created_at": "2026-09-13T00:00:00Z",
        "database": {"path": "aurora.db", "size": 1, "sha256": "x" * 64},
        "files": [],
        "unexpected": True,
    }
    _write_archive(archive, manifest, {"aurora.db": b"x"})
    with pytest.raises(ValueError, match="manifest schema"):
        verify_backup(archive)


def test_restore_requires_empty_destination(tmp_path):
    data_dir = tmp_path / "data"
    settings = Settings(data_dir=data_dir)
    settings.prepare()
    engine, _ = database(data_dir)
    Base.metadata.create_all(engine)
    engine.dispose()
    archive, _ = create_backup(data_dir, tmp_path / "backups")
    destination = tmp_path / "restore"
    destination.mkdir()
    (destination / "keep.txt").write_text("keep")
    with pytest.raises(ValueError, match="must be empty"):
        restore_backup(archive, destination)
    assert (destination / "keep.txt").read_text() == "keep"
    symlink = tmp_path / "restore-link"
    symlink.symlink_to(destination)
    with pytest.raises(ValueError, match="symbolic link"):
        restore_backup(archive, symlink)


def test_backups_created_in_same_second_are_unique(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    settings = Settings(data_dir=data_dir)
    settings.prepare()
    engine, _ = database(data_dir)
    Base.metadata.create_all(engine)
    engine.dispose()
    import aurora_visual.backup as backup_module

    class FixedDateTime:
        @staticmethod
        def now(_):
            from datetime import datetime

            return datetime.fromisoformat("2026-09-13T12:00:00.123456+00:00")

    monkeypatch.setattr(backup_module, "datetime", FixedDateTime)
    first, _ = create_backup(data_dir, tmp_path / "backups")
    with pytest.raises(FileExistsError):
        create_backup(data_dir, tmp_path / "backups")
    assert first.is_file()
