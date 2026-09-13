#!/usr/bin/env python3
"""Verify and restore an AURORA backup into an empty data directory."""

import argparse
import json
from pathlib import Path

from aurora_visual.backup import restore_backup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        files = restore_backup(args.archive, args.data_dir)
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from None
    print(json.dumps({"status": "ok", "data_dir": str(args.data_dir.resolve()), "files": files}))


if __name__ == "__main__":
    main()
