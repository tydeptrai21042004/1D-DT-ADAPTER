#!/usr/bin/env bash
set -Eeuo pipefail

EXPECTED_BRANCH="dt1d-v8-cnn-three-seed"
CURRENT_BRANCH="$(git branch --show-current)"

if [[ "$CURRENT_BRANCH" != "$EXPECTED_BRANCH" ]]; then
  echo "ERROR: current branch is '$CURRENT_BRANCH'; expected '$EXPECTED_BRANCH'." >&2
  echo "This script will not create, rename, or switch branches." >&2
  exit 2
fi

# Refuse generated/runtime artifacts.
find . -type d \( -name __pycache__ -o -name .pytest_cache \) -prune -exec rm -rf {} + 2>/dev/null || true
find . -type f \( -name '*.pyc' -o -path './outputs/hosq_validation/*' \) -delete 2>/dev/null || true

python -m pytest -q tests/test_hosq_dt1d.py tests/test_v9_scdq_release.py
python tools/validate_hosq_theory.py

git add -- \
  BRANCH_NAME.txt CHANGELOG.md VERSION README.md HOSQ_FINAL_REPORT.md \
  KAGGLE_HOSQ_FULL_CELL.txt KAGGLE_HOSQ_FULL_RUN.sh \
  main.py models/dt1d_adapter.py models/tuning_modules/__init__.py \
  configs/paper/cnn_three_seed_manifest.yaml \
  configs/experiments/hosq_three_seed_manifest.yaml \
  tests/test_hosq_dt1d.py tests/test_v9_scdq_release.py \
  tools/benchmark_hosq_latency.py tools/validate_hosq_theory.py

echo
echo "Files staged for commit:"
git diff --cached --name-status

git commit -m "Add runnable HOSQ-DT1D method and ablations"
git push origin "$EXPECTED_BRANCH"
