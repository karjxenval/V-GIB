#!/usr/bin/env python3
"""Create a SHA256 release manifest for V-GIB."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--out", default="RELEASE_MANIFEST_SHA256.txt")
    args = parser.parse_args()

    _ensure_src_on_path()
    from vgib.release import write_manifest

    write_manifest(args.root, args.out)
    print(f"Wrote manifest: {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
