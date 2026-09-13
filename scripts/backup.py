#!/usr/bin/env python3
"""Create an atomic SQLite backup with integrity and file-manifest checks."""

import argparse
import json
import os
from pathlib import Path

from aurora_visual.backup import create_backup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(os.getenv("AURORA_DATA_DIR", "var")))
    parser.add_argument("--output-dir", type=Path, default=Path("backups"))
    args = parser.parse_args()
    try:
        destination, checksum = create_backup(args.data_dir, args.output_dir)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    print(json.dumps({"status": "ok", "backup": str(destination), "sha256": checksum}))


if __name__ == "__main__":
    main()
