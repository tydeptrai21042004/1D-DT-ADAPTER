#!/usr/bin/env bash
set -Eeuo pipefail
SEEDS="${SEEDS:-0,1,2}"
DATA_DIR="${DATA_DIR:-data}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/kaggle_hosq_lite_c1}"
RUN_COMPARISON="${RUN_COMPARISON:-1}"
RUN_ABLATION="${RUN_ABLATION:-1}"
[[ "$SEEDS" == "0,1,2" ]] || { echo "Publication runs require SEEDS=0,1,2" >&2; exit 2; }
python -m pip install -q --upgrade-strategy only-if-needed -r requirements-kaggle.txt
python -m pytest -q | tee "$OUTPUT_ROOT-pytest.txt"
python tools/validate_hosq_lite_c1.py
python tools/benchmark_hosq_lite_latency.py --batch-size 2 --warmup 3 --iters 15
if [[ "$RUN_COMPARISON" == "1" ]]; then
  python tools/run_cnn_paper.py --target table_14_15 --seeds "$SEEDS" \
    --data-path "$DATA_DIR" --device "$DEVICE" --skip-if-complete \
    --output-root "$OUTPUT_ROOT/comparison"
fi
if [[ "$RUN_ABLATION" == "1" ]]; then
  python tools/run_cnn_paper.py --manifest configs/experiments/hosq_lite_c1_ablation.yaml \
    --target hosq_lite_c1_ablation --seeds "$SEEDS" --data-path "$DATA_DIR" \
    --device "$DEVICE" --skip-if-complete --output-root "$OUTPUT_ROOT/ablation"
fi
python - <<'PY'
import shutil
from pathlib import Path
root=Path('outputs/kaggle_hosq_lite_c1')
if root.exists():
    print(shutil.make_archive(str(root), 'zip', root_dir=root.parent, base_dir=root.name))
PY
