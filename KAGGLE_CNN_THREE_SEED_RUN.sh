%%bash
set -Eeuo pipefail

REPO_URL="${REPO_URL:-https://github.com/tydeptrai21042004/1D-DT-ADAPTER.git}"
BRANCH="${BRANCH:-dt1d-v8-cnn-three-seed}"
TARGET="${TARGET:-table_14_15}"
SEEDS="${SEEDS:-0,1,2}"
METHODS="${METHODS:-target}"
WORKDIR="${WORKDIR:-/kaggle/working}"
REPO_DIR="${REPO_DIR:-$WORKDIR/1D-DT-ADAPTER}"
DATA_DIR="${DATA_DIR:-$WORKDIR/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$WORKDIR/dt1d_cnn_three_seed_results}"
RESULT_ZIP="${RESULT_ZIP:-$WORKDIR/dt1d_cnn_${TARGET}_three_seed_results.zip}"
DEVICE="${DEVICE:-cuda}"
export OUTPUT_ROOT RESULT_ZIP

mkdir -p "$WORKDIR" "$DATA_DIR" "$OUTPUT_ROOT"
if [[ ! -d "$REPO_DIR/.git" ]]; then
  git clone --branch "$BRANCH" --single-branch "$REPO_URL" "$REPO_DIR"
else
  git -C "$REPO_DIR" remote set-url origin "$REPO_URL"
  git -C "$REPO_DIR" fetch --prune origin "$BRANCH"
  git -C "$REPO_DIR" checkout -B "$BRANCH" "origin/$BRANCH"
  git -C "$REPO_DIR" reset --hard "origin/$BRANCH"
fi

cd "$REPO_DIR"
[[ "$(git branch --show-current)" == "$BRANCH" ]]
git rev-parse HEAD | tee "$OUTPUT_ROOT/git_commit.txt"
git status --short --branch | tee "$OUTPUT_ROOT/git_status.txt"

python -m pip install -q --upgrade pip
python -m pip install -q -r requirements-kaggle.txt

python tools/verify_reproducibility_package.py
python tools/preflight_cnn_matrix.py --target "$TARGET" --output "$OUTPUT_ROOT/${TARGET}_model_preflight.json"

python tools/run_cnn_paper.py \
  --target "$TARGET" \
  --seeds "$SEEDS" \
  --methods "$METHODS" \
  --data-path "$DATA_DIR" \
  --device "$DEVICE" \
  --output-root "$OUTPUT_ROOT" \
  --skip-if-complete

python tools/aggregate_cnn_paper.py \
  --root "$OUTPUT_ROOT" \
  --target "$TARGET" \
  --require-seeds "$SEEDS"

if [[ "$TARGET" == "figure_01" ]]; then
  python tools/plot_figure_01_three_seed.py --root "$OUTPUT_ROOT" --require-seeds "$SEEDS"
elif [[ "$TARGET" == "figure_04" ]]; then
  python tools/plot_figure_04_tradeoff.py --root "$OUTPUT_ROOT" --require-seeds "$SEEDS"
fi

python - <<'PY'
import os
import shutil
from pathlib import Path
root = Path(os.environ["OUTPUT_ROOT"])
zip_path = Path(os.environ["RESULT_ZIP"])
zip_path.parent.mkdir(parents=True, exist_ok=True)
created = Path(shutil.make_archive(str(zip_path.with_suffix("")), "zip", root_dir=root.parent, base_dir=root.name))
if created != zip_path:
    if zip_path.exists():
        zip_path.unlink()
    created.replace(zip_path)
print(f"Created: {zip_path}")
PY
