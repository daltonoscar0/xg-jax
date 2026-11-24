#!/usr/bin/env python3
"""Download the StatsBomb open data to data/raw/ and record the commit used.

    python scripts/fetch_data.py [--commit SHA] [--force]

Equivalent to `xg fetch`. Kept as a standalone script so the download is reproducible
without installing the package.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xg.data import default_root  # noqa: E402
from xg.fetch import fetch_open_data  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", help="pin a specific open-data commit")
    parser.add_argument("--root", type=Path, default=None, help="where to extract")
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args()
    fetch_open_data(args.root or default_root(), commit=args.commit, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
