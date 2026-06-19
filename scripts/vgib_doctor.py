#!/usr/bin/env python3
"""Environment diagnostic for V-GIB.

Run from repository root:

    python scripts/vgib_doctor.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def main() -> None:
    _ensure_src_on_path()
    from vgib.reproducibility import environment_report

    report = environment_report()
    print(json.dumps(report, indent=2, sort_keys=True))

    missing = [name for name, version in report["packages"].items() if version == "not-installed"]
    if missing:
        print("\nMissing packages:", ", ".join(missing))
        print("Install with: pip install -r requirements.txt")
        raise SystemExit(1)

    print("\nV-GIB environment check passed.")


if __name__ == "__main__":
    main()
