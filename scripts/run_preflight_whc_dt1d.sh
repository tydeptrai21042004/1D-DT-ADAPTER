#!/usr/bin/env bash
set -Eeuo pipefail
REPO_DIR="${REPO_DIR:-$(pwd)}"
cd "$REPO_DIR"
python -m pip install -q timm pandas thop fvcore torchmetrics pycocotools opencv-python-headless || true
PYTHONPATH="$REPO_DIR" python tools/preflight_whc_dt1d.py --device "${DEVICE:-auto}"
PYTHONPATH="$REPO_DIR" python -m pytest -q tests/test_whc_dt1d_adapter.py
