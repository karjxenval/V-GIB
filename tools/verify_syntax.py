#!/usr/bin/env python3
"""Check Python syntax for all repository scripts without importing heavy libraries."""
from __future__ import annotations

import ast
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
paths = sorted([p for p in ROOT.rglob('*.py') if '__pycache__' not in p.parts])
failed = False
for path in paths:
    try:
        ast.parse(path.read_text(encoding='utf-8'))
        print(f'OK     {path.relative_to(ROOT)}')
    except SyntaxError as exc:
        failed = True
        print(f'FAIL   {path.relative_to(ROOT)}:{exc.lineno}:{exc.offset}: {exc.msg}')

if failed:
    sys.exit(1)
print(f'Checked {len(paths)} Python files.')
