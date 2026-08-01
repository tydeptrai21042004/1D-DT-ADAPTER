#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-.}:$(pwd)"
python tools/preflight_dt1d_static.py --device auto
python -m pytest -q tests/test_dt1d_static_adapter.py
