.PHONY: install dev test lint format doctor smoke gate manifest build clean

install:
	pip install -e .

dev:
	pip install -e .
	pip install -r requirements-dev.txt

test:
	python -m pytest

lint:
	ruff check src scripts tests

format:
	ruff format src scripts tests

doctor:
	python scripts/vgib_doctor.py

smoke:
	python scripts/vgib_smoke_runner.py --outdir runs/smoke_industry --device cpu

gate:
	python scripts/vgib_result_gate.py --run-dir runs/smoke_industry --min-rows 1

manifest:
	python scripts/vgib_release_manifest.py --out RELEASE_MANIFEST_SHA256.txt

build:
	python -m build

clean:
	python -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in ['build','dist','.pytest_cache','.ruff_cache']]; [p.unlink() for p in pathlib.Path('.').glob('*.egg-info') if p.is_file()]"
