#!/usr/bin/env python3
"""Verify an AURORA backup without extracting untrusted paths."""

import argparse
import json
from pathlib import Path

from aurora_visual.backup import verify_backup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    try:
        files = verify_backup(args.archive)
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from None
    print(json.dumps({"status": "ok", "archive": str(args.archive.resolve()), "files": files}))


if __name__ == "__main__":
    main()
