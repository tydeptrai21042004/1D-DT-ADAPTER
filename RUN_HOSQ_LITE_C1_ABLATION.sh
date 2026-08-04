#!/usr/bin/env bash
set -Eeuo pipefail
python tools/run_cnn_paper.py \
  --manifest configs/experiments/hosq_lite_c1_ablation.yaml \
  --target hosq_lite_c1_ablation --seeds "${SEEDS:-0,1,2}" \
  --data-path "${DATA_DIR:-data}" --device "${DEVICE:-cuda}" --skip-if-complete
