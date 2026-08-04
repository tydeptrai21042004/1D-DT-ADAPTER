# HOSQ-Lite-C1-Orth DT1D-Adapter

This repository contains one revised proposal and one retained original-method baseline:

- **Proposal:** `HOSQ-Lite-C1-Orth`, implemented by `models.hosq_lite_c1_adapter.HOSQLiteC1Adapter`.
- **Original baseline:** `DT1D-Adapter`, implemented by `models.dt1d_adapter.DT1DAdapter`.

Alternative proposal families from earlier development branches have been removed. The remaining `detail_basis` and `detail_components` settings are component ablations of HOSQ-Lite-C1-Orth, not separate proposals.

## Final method

HOSQ-Lite-C1-Orth keeps the original Group-16 shifted axial DT1D structure. It learns five observable symmetric quotient coefficients at offsets

```text
0, ±1, ±2, ±4, ±8
```

and adds one zero-mean channel contrast per original group with two orthonormal zero-DC spectral coordinates. The adapter executes exactly one 17-tap depthwise convolution per enabled axis. The paper configuration uses height and width axes, `replicate` padding, no pointwise block, and a scalar residual gate.

## Install and validate

```bash
python -m pip install -r requirements.txt
pytest -q
python tools/validate_hosq_lite_c1.py
python tools/benchmark_hosq_lite_latency.py --batch-size 2 --warmup 3 --iters 15
```

## Run the focused ablation

```bash
SEEDS=0,1,2 DATA_DIR=/path/to/data DEVICE=cuda \
  bash RUN_HOSQ_LITE_C1_ABLATION.sh
```

The focused manifest is `configs/experiments/hosq_lite_c1_ablation.yaml`. It evaluates:

1. final HOSQ-Lite-C1-Orth;
2. original DT1D with pointwise mixing;
3. original DT1D core;
4. removal of the orthogonal detail;
5. offset-4 detail only;
6. offset-8 detail only;
7. raw zero-DC atoms instead of the orthogonal basis;
8. height-only and width-only filtering;
9. Group-8 and Group-32 sharing.

## Run the full CNN paper matrix

```bash
python tools/run_cnn_paper.py \
  --target all --seeds 0,1,2 \
  --data-path /path/to/data --device cuda --skip-if-complete
```

The paper manifest is `configs/paper/cnn_three_seed_manifest.yaml`. Main comparison tables use HOSQ-Lite-C1-Orth as `dt1d`; original DT1D appears only in the focused ablation.

## Direct CLI

Final proposal:

```bash
python main.py \
  --tuning_method dt --dt_variant hosq_lite_c1 \
  --dt_alpha_group 16 --dt_axis hw --dt_padding replicate \
  --dt_detail_basis orth --dt_detail_components both \
  --dt_contrast_split 8
```

Original submitted DT1D baseline:

```bash
python main.py \
  --tuning_method dt --dt_variant legacy \
  --dt_M 1 --dt_dilations 1,2,4 --dt_scale_adaptive true \
  --dt_alpha_group 16 --dt_axis hw --dt_padding replicate
```

## Reproducibility policy

The main experiments use independent seeds 0, 1, and 2; the same dataset split, pretrained weights, optimizer family, schedule, checkpoint rule, device class, precision, and profiling settings must be used across methods. Test accuracy is reported at the best validation checkpoint. Generated publication YAML files are stored under `configs/paper/generated/`.

## Important boundary rule

Use `replicate` padding for both HOSQ-Lite-C1-Orth and original DT1D in direct equivalence and latency comparisons. A fused radius-8 kernel and separate dilated branches do not share the same finite-boundary behavior under `reflect` padding on the final `7×7` ResNet stage.

See `HOSQ_LITE_C1_FINAL_REPORT.md`, `MANUSCRIPT_TO_CODE.md`, and `REPRODUCIBILITY.md` for details.
