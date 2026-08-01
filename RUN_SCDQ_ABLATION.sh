#!/usr/bin/env bash
set -Eeuo pipefail
DATA_PATH="${DATA_PATH:-data}"
DEVICE="${DEVICE:-cuda}"
SEEDS="${SEEDS:-0,1,2}"
OUT_ROOT="${OUT_ROOT:-outputs/table_14_15_scdq_ablation}"
python tools/run_cnn_paper.py \
  --manifest configs/experiments/scdq_three_seed_manifest.yaml \
  --target table_14_15_scdq_ablation \
  --seeds "$SEEDS" --methods target \
  --data-path "$DATA_PATH" --device "$DEVICE" \
  --output-root "$OUT_ROOT" --skip-if-complete
python tools/aggregate_cnn_paper.py \
  --root "$OUT_ROOT" --target table_14_15_scdq_ablation \
  --require-seeds "$SEEDS"
