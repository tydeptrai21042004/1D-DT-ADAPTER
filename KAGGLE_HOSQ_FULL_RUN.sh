set -Eeuo pipefail

# ============================================================
# HOSQ-DT1D v0.10.0 — Git-clone Kaggle runner
# Existing Git branch is intentionally unchanged.
# Main comparison: 10 methods x 3 seeds = 30 runs
# HOSQ screening: 8 variants x 3 seeds = 24 runs
# Optional cross-setting confirmation: 3 more targets x 24 runs
# ============================================================

REPO_URL="${REPO_URL:-https://github.com/tydeptrai21042004/1D-DT-ADAPTER.git}"
BRANCH="${BRANCH:-dt1d-v8-cnn-three-seed}"
SEEDS="${SEEDS:-0,1,2}"
[[ "$SEEDS" == "0,1,2" ]] || { echo "Publication runs require SEEDS=0,1,2." >&2; exit 2; }

DEVICE="${DEVICE:-cuda}"
RUN_TESTS="${RUN_TESTS:-1}"
RUN_MAIN_COMPARISON="${RUN_MAIN_COMPARISON:-1}"
RUN_HOSQ_SCREEN="${RUN_HOSQ_SCREEN:-1}"
RUN_CROSS_SETTING="${RUN_CROSS_SETTING:-0}"

WORKDIR="${WORKDIR:-/kaggle/working}"
REPO_DIR="${REPO_DIR:-$WORKDIR/1D-DT-ADAPTER}"
DATA_DIR="${DATA_DIR:-$WORKDIR/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$WORKDIR/hosq_dt1d_results}"
MAIN_ROOT="$OUTPUT_ROOT/main_comparison"
HOSQ_ROOT="$OUTPUT_ROOT/hosq_screen"
CROSS_ROOT="$OUTPUT_ROOT/cross_setting"
RESULT_ZIP="${RESULT_ZIP:-$WORKDIR/hosq_dt1d_results.zip}"

export OUTPUT_ROOT MAIN_ROOT HOSQ_ROOT CROSS_ROOT RESULT_ZIP

log() {
  echo
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

package_results() {
  local code=$?
  trap - EXIT
  set +e
  if [[ -d "$OUTPUT_ROOT" ]]; then
    python - <<'PYZIP'
import os
import shutil
from pathlib import Path
root = Path(os.environ["OUTPUT_ROOT"])
zip_path = Path(os.environ["RESULT_ZIP"])
zip_path.parent.mkdir(parents=True, exist_ok=True)
if zip_path.exists():
    zip_path.unlink()
created = Path(shutil.make_archive(
    str(zip_path.with_suffix("")), "zip",
    root_dir=root.parent, base_dir=root.name,
))
if created != zip_path:
    created.replace(zip_path)
print(f"[package] Created: {zip_path}")
PYZIP
  fi
  exit "$code"
}
trap package_results EXIT

mkdir -p "$WORKDIR" "$DATA_DIR" "$OUTPUT_ROOT"
rm -rf "$REPO_DIR"

log "Cloning unchanged GitHub branch: $BRANCH"
git clone --branch "$BRANCH" --single-branch --depth 1 "$REPO_URL" "$REPO_DIR"
cd "$REPO_DIR"

ACTUAL_BRANCH="$(git branch --show-current)"
[[ "$ACTUAL_BRANCH" == "$BRANCH" ]] || {
  echo "Expected branch '$BRANCH', found '$ACTUAL_BRANCH'." >&2
  exit 2
}
[[ "$(tr -d '\r\n' < BRANCH_NAME.txt)" == "$BRANCH" ]] || {
  echo "BRANCH_NAME.txt does not match '$BRANCH'. Push the HOSQ release to the existing branch first." >&2
  exit 2
}
git rev-parse HEAD | tee "$OUTPUT_ROOT/git_commit.txt"
git status --short --branch | tee "$OUTPUT_ROOT/git_status.txt"

# ------------------------------------------------------------
# Preserve Kaggle's CUDA/PyTorch/NumPy stack. Install add-ons only.
# ------------------------------------------------------------
log "Installing runtime add-ons"
python -m pip install -q --upgrade-strategy only-if-needed \
  "timm==1.0.15" \
  "thop==0.1.1.post2209072238" \
  "fvcore==0.1.5.post20221221" \
  "torchmetrics==1.6.1" \
  "pytest>=8,<9"

python - <<'PYDEPS'
import importlib.util
import subprocess
import sys
required = {
    "yaml": "PyYAML", "pandas": "pandas", "numpy": "numpy",
    "scipy": "scipy", "sklearn": "scikit-learn", "PIL": "Pillow",
}
missing = [pkg for mod, pkg in required.items() if importlib.util.find_spec(mod) is None]
if missing:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "--upgrade-strategy", "only-if-needed", *missing,
    ])
print("Missing packages installed:" if missing else "Core packages present.", missing)
PYDEPS

log "Checking CUDA and runtime versions"
python - <<'PYCUDA'
import importlib.metadata as md
import torch, torchvision
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable. Enable a Kaggle GPU accelerator.")
print("PyTorch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("CUDA runtime:", torch.version.cuda)
print("cuDNN:", torch.backends.cudnn.version())
print("GPU:", torch.cuda.get_device_name(0))
for pkg in ("timm", "numpy", "pandas", "scipy", "scikit-learn", "PyYAML"):
    try:
        print(f"{pkg}:", md.version(pkg))
    except md.PackageNotFoundError:
        print(f"{pkg}: not installed")
PYCUDA

log "Auditing HOSQ source and executable assets"
python - <<'PYFILES'
from pathlib import Path
required = [
    "main.py", "engine.py", "models/dt1d_adapter.py",
    "configs/paper/cnn_three_seed_manifest.yaml",
    "configs/experiments/hosq_three_seed_manifest.yaml",
    "tools/run_from_config.py", "tools/run_cnn_paper.py",
    "tools/aggregate_cnn_paper.py", "tools/preflight_cnn_matrix.py",
    "tools/validate_hosq_theory.py", "tests/test_hosq_dt1d.py",
    "splits/caltech101/seed0_holdout20.json",
    "splits/caltech101/seed1_holdout20.json",
    "splits/caltech101/seed2_holdout20.json",
]
missing = [p for p in required if not Path(p).is_file()]
if missing:
    raise SystemExit("Missing required HOSQ files:\n" + "\n".join(missing))
text = Path("models/dt1d_adapter.py").read_text()
for token in ("hosq_realization", "_build_normalized_hosq_kernels", "hosq_detail4", "hosq_detail8"):
    if token not in text:
        raise SystemExit(f"HOSQ token missing from canonical source: {token}")
print(f"Asset audit passed: {len(required)} files and HOSQ source tokens.")
PYFILES

if [[ "$RUN_TESTS" == "1" ]]; then
  log "Running complete repository tests"
  python -m pytest -q | tee "$OUTPUT_ROOT/pytest.txt"
fi

log "Validating HOSQ mathematics and parameter budget"
python tools/validate_hosq_theory.py | tee "$OUTPUT_ROOT/hosq_theory_stdout.txt"
cp outputs/hosq_validation/hosq_validation.json "$OUTPUT_ROOT/hosq_validation.json"

# ------------------------------------------------------------
# Preflight without downloading datasets.
# ------------------------------------------------------------
log "Preflighting main Table 14-15 comparison"
python tools/preflight_cnn_matrix.py \
  --manifest configs/paper/cnn_three_seed_manifest.yaml \
  --target table_14_15 \
  --output "$OUTPUT_ROOT/main_preflight.json"

log "Preflighting HOSQ screening matrix"
python tools/preflight_cnn_matrix.py \
  --manifest configs/experiments/hosq_three_seed_manifest.yaml \
  --target hosq_caltech101_resnet18 \
  --output "$OUTPUT_ROOT/hosq_screen_preflight.json"

# ------------------------------------------------------------
# Main fair comparison: HOSQ proposal + nine controls/baselines.
# ------------------------------------------------------------
if [[ "$RUN_MAIN_COMPARISON" == "1" ]]; then
  log "Building 30-run main comparison plan"
  python tools/run_cnn_paper.py \
    --manifest configs/paper/cnn_three_seed_manifest.yaml \
    --target table_14_15 --seeds "$SEEDS" --methods target \
    --data-path "$DATA_DIR" --device "$DEVICE" \
    --output-root "$MAIN_ROOT" --plan-only

  python - <<'PYMAIN'
import json, os
from pathlib import Path
plan = json.loads((Path(os.environ["MAIN_ROOT"]) / "execution_plan.json").read_text())
expected = {"dt1d", "full", "linear", "conv_r4", "bam", "residual",
            "ssf", "lora_conv", "bitfit", "sidetune"}
methods = {Path(r["output_dir"]).parts[-2] for r in plan["runs"]}
if len(plan["runs"]) != 30:
    raise SystemExit(f"Expected 30 main runs, got {len(plan['runs'])}")
if methods != expected:
    raise SystemExit(f"Method mismatch: {sorted(methods)}")
if plan["seeds"] != [0, 1, 2]:
    raise SystemExit(f"Seed mismatch: {plan['seeds']}")
print("Main plan passed: 10 methods x 3 seeds.")
PYMAIN

  log "Running main comparison"
  python tools/run_cnn_paper.py \
    --manifest configs/paper/cnn_three_seed_manifest.yaml \
    --target table_14_15 --seeds "$SEEDS" --methods target \
    --data-path "$DATA_DIR" --device "$DEVICE" \
    --output-root "$MAIN_ROOT" --skip-if-complete

  python tools/aggregate_cnn_paper.py \
    --root "$MAIN_ROOT" --target table_14_15 --require-seeds "$SEEDS"
fi

# ------------------------------------------------------------
# HOSQ design screening: 8 variants x 3 seeds.
# ------------------------------------------------------------
if [[ "$RUN_HOSQ_SCREEN" == "1" ]]; then
  log "Building 24-run HOSQ screening plan"
  python tools/run_cnn_paper.py \
    --manifest configs/experiments/hosq_three_seed_manifest.yaml \
    --target hosq_caltech101_resnet18 --seeds "$SEEDS" --methods target \
    --data-path "$DATA_DIR" --device "$DEVICE" \
    --output-root "$HOSQ_ROOT" --plan-only

  python - <<'PYHOSQ'
import json, os
from pathlib import Path
plan = json.loads((Path(os.environ["HOSQ_ROOT"]) / "execution_plan.json").read_text())
variants = {Path(r["output_dir"]).parts[-2].split("dt1d__", 1)[-1] for r in plan["runs"]}
if len(plan["runs"]) != 24:
    raise SystemExit(f"Expected 24 HOSQ runs, got {len(plan['runs'])}")
if len(variants) != 8:
    raise SystemExit(f"Expected 8 HOSQ variants, got {len(variants)}: {sorted(variants)}")
if "hosq_r4_1_r8_2_final" not in variants:
    raise SystemExit("Final HOSQ variant missing from execution plan.")
print("HOSQ plan passed: 8 variants x 3 seeds.")
PYHOSQ

  log "Running HOSQ screening"
  python tools/run_cnn_paper.py \
    --manifest configs/experiments/hosq_three_seed_manifest.yaml \
    --target hosq_caltech101_resnet18 --seeds "$SEEDS" --methods target \
    --data-path "$DATA_DIR" --device "$DEVICE" \
    --output-root "$HOSQ_ROOT" --skip-if-complete

  python tools/aggregate_cnn_paper.py \
    --root "$HOSQ_ROOT" --target hosq_caltech101_resnet18 --require-seeds "$SEEDS"
fi

# ------------------------------------------------------------
# Optional two-dataset/two-backbone confirmation.
# ------------------------------------------------------------
if [[ "$RUN_CROSS_SETTING" == "1" ]]; then
  for target in hosq_dtd_resnet18 hosq_caltech101_resnet50 hosq_dtd_resnet50; do
    target_root="$CROSS_ROOT/$target"
    log "Running cross-setting HOSQ target: $target"
    python tools/run_cnn_paper.py \
      --manifest configs/experiments/hosq_three_seed_manifest.yaml \
      --target "$target" --seeds "$SEEDS" --methods target \
      --data-path "$DATA_DIR" --device "$DEVICE" \
      --output-root "$target_root" --skip-if-complete
    python tools/aggregate_cnn_paper.py \
      --root "$target_root" --target "$target" --require-seeds "$SEEDS"
  done
fi

log "Recording environment"
python -m pip freeze > "$OUTPUT_ROOT/pip_freeze.txt"
nvidia-smi > "$OUTPUT_ROOT/nvidia_smi.txt" || true

log "Important outputs"
find "$OUTPUT_ROOT" -type f \
  \( -name "execution_plan.json" -o -name "raw_runs.csv" \
     -o -name "mean_std_numeric.csv" -o -name "mean_std_pretty.csv" \
     -o -name "manuscript_compact.csv" -o -name "manuscript_compact.tex" \
     -o -name "seed_completeness.json" -o -name "aggregation_summary.json" \
     -o -name "*preflight.json" -o -name "hosq_validation.json" \) \
  -print | sort

log "COMPLETED"
echo "Results: $OUTPUT_ROOT"
echo "ZIP:     $RESULT_ZIP"
