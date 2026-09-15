"""Runtime configuration store: UI edits persisted as an overlay on env settings.

The overlay is a JSON file inside ``data_dir`` (mode 0600) containing only
UI-configurable fields (see ``aurora_decision.config.UI_FIELDS``). Values are merged over
the env-derived ``Settings`` so server restarts and worker subprocesses pick up
the same configuration. Secrets never leave the server: read responses mask
them with ``MASKED`` and write requests keep the stored value when the client
sends the sentinel back.
"""

import json
import os
import sys
from pathlib import Path

from aurora_decision.config import UI_FIELDS, ServiceError, Settings, normalize_overlay

MASKED = "__CONFIGURED__"


class SettingsStore:
    def __init__(self, settings: Settings, path: Path):
        self.settings, self.path = settings, path

    def load(self):
        """Apply the persisted overlay at startup; ignore malformed files loudly."""
        if not self.path.is_file():
            return
        try:
            data = json.loads(self.path.read_text())
            if isinstance(data, dict):
                overlay = normalize_overlay(data)
                self.settings.apply_overlay(overlay)
        except (ValueError, OSError, json.JSONDecodeError) as exc:
            print(f"Settings overlay ignored: {type(exc).__name__}", file=sys.stderr)

    def read(self) -> dict:
        """Current values with secrets masked for API responses."""
        values = {}
        for name, value in self.settings.overlay_values().items():
            secret = UI_FIELDS[name][1]
            values[name] = MASKED if secret and value else value
        return values

    def update(self, changes: dict) -> dict:
        """Validate a partial update, persist the merged overlay, apply it live."""
        if not isinstance(changes, dict) or not changes:
            raise ServiceError(
                "INVALID_SETTING", "Kiriman konfigurasi harus objek berisi kolom yang diubah.", 422
            )
        kept = {
            name: value
            for name, value in changes.items()
            if not (UI_FIELDS.get(name, ("", False))[1] and value == MASKED)
        }
        try:
            merged = normalize_overlay({**self.settings.overlay_values(), **kept})
        except ValueError as exc:
            raise ServiceError("INVALID_SETTING", str(exc), 422) from exc
        self._write(merged)
        self.settings.apply_overlay(merged)
        return merged

    def _write(self, overlay: dict):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp.json")
        tmp.write_text(json.dumps(overlay, indent=2, sort_keys=True))
        os.chmod(tmp, 0o600)
        tmp.replace(self.path)
