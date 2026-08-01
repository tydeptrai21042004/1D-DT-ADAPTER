# SCDQ-DT1D v0.9.1

The final comparison preset now uses the **Spectrally Closed Dyadic Quotient**
parameterization with support cap 4, reflect padding, group size 16, and no
optional pointwise mixer. The original shifted-symmetric group-shared axial
convolution remains the trainable spatial operator.

For Kaggle, upload `DT1D_V9_SCDQ_KAGGLE_FINAL.zip` as a Dataset and paste
`KAGGLE_SCDQ_FULL_CELL.txt` into one notebook cell, or push branch
`dt1d-v9-scdq-math-latency` and let the same cell clone it.


This repository is the **CNN-only** experimental package for DT1D-Adapter. New paper runs use the canonical implementation `models/dt1d_adapter.py`, class `DT1DAdapter`, CLI method `--tuning_method dt`, and `--dt_*` arguments. `models/hcc_adapter.py` is retained only as a compatibility shim for older checkpoints.

## What changed in v0.8.0

- Every stochastic CNN classification table can be run with **three independent seeds: 0, 1, and 2**.
- **Full fine-tuning** and **Linear probing** are executable baselines and are included in every comparison target.
- One versioned manifest maps manuscript tables and figures to datasets, CNN backbones, epochs, batch sizes, methods, and seeds.
- **408 portable per-run YAML configurations** are committed for all selected target/method/seed combinations.
- Every run records its resolved YAML, exact command, seed, split manifest, Git revision, environment, pretrained weights, logs, metrics, and status.
- Aggregation exports raw runs and `mean ± standard deviation` summaries in CSV, JSON, and LaTeX formats.
- Figure 1 and Figure 4 are regenerated from three-seed outputs. Figures 2 and 3 are deterministic and do not require training seeds.
- ViT, Swin, Transformer, and token-prompt experiments are rejected by the CNN paper runner.

## V9 mathematical latency revision

The repository now includes **SCDQ-DT1D**, a Spectrally Closed Dyadic Quotient parameterization that retains reduced axial filtering and shifted symmetric group-shared kernels while removing the exact dyadic scale nullspace. See [`MLQ_SCDQ_FINAL_REPORT.md`](MLQ_SCDQ_FINAL_REPORT.md).

Key commands:

```bash
pytest -q
python tools/validate_mlq_theory.py
python tools/benchmark_mlq_latency.py --batch-size 4 --warmup 3 --iters 10
```

The final three-seed ablation configurations are under `configs/experiments/table_14_15_mlq_ablation/`.

## Supported manuscript targets

| Target | Dataset / CNN backbone | Protocol |
|---|---|---|
| Table 2 | DTD / ResNet-18 | 9 DT1D ablations × seeds 0,1,2 |
| Tables 3–4 | DTD or Flowers102 / ResNet-50 | 100 epochs × 3 seeds |
| Tables 5–7 | Flowers102 / ResNet-18 | 10 or 100 epochs × 3 seeds |
| Tables 8–9 | SVHN or Oxford-IIIT Pet / ResNet-50 | 10 epochs × 3 seeds |
| Tables 10–13 | Food-101 or Oxford-IIIT Pet / ResNet-18 or EfficientNet-B0 | 10 or 100 epochs × 3 seeds |
| Tables 14–15 | Caltech101 / ResNet-18 | accuracy and efficiency × 3 seeds |
| Tables 18–19 | EuroSAT / MobileNetV3-Small | accuracy and efficiency × 3 seeds |
| Figure 1 | Caltech101 / ResNet-18 / DT1D | mean convergence curve with ±1 std band |
| Figure 4 | FGVC-Aircraft / ResNet-18 | mean test accuracy ±1 std versus parameter count |

The experimental matrix is defined in [`configs/paper/cnn_three_seed_manifest.yaml`](configs/paper/cnn_three_seed_manifest.yaml).

## Installation

```bash
python -m pip install -r requirements.txt
```

or:

```bash
conda env create -f environment.yml
conda activate dt1d-adapter-0.8.0
```

## Run one table

```bash
DATA_DIR=/path/to/data \
DEVICE=cuda \
SEEDS=0,1,2 \
bash scripts/tables/table_14_15_three_seed.sh
```

Other table entry points are in `scripts/tables/`.

## Run a target directly

```bash
python tools/run_cnn_paper.py \
  --target table_03 \
  --seeds 0,1,2 \
  --data-path /path/to/data \
  --device cuda \
  --skip-if-complete

python tools/aggregate_cnn_paper.py \
  --root outputs/cnn_paper_three_seed \
  --target table_03 \
  --require-seeds 0,1,2
```

Use the table's publication method list with `--methods target`. To run every implemented CNN method instead:

```bash
python tools/run_cnn_paper.py \
  --target table_03 \
  --methods all-cnn \
  --seeds 0,1,2 \
  --data-path /path/to/data \
  --device cuda
```

## Full fine-tuning and Linear probing

Both methods use the same backbone, pretrained weights, data split, augmentation, optimizer family, epoch count, and seed as DT1D and the other baselines.

```bash
python tools/run_cnn_paper.py \
  --target table_14_15 \
  --methods full,linear \
  --seeds 0,1,2 \
  --data-path /path/to/data \
  --device cuda
```

- `full`: every backbone and classifier parameter is trainable.
- `linear`: only the replaced task classifier is trainable; the feature extractor and BatchNorm statistics are frozen.

## Figures

```bash
bash scripts/figures/figure_01_three_seed.sh
bash scripts/figures/figure_02_deterministic.sh
bash scripts/figures/figure_03_deterministic.sh
bash scripts/figures/figure_04_three_seed.sh
```

## Seed policy

For a fixed seed, all methods receive the same split and data-loader order. Seeds 0, 1, and 2 independently control Python, NumPy, PyTorch, CUDA, samplers, workers, model initialization, and generated validation splits. Datasets with official train/validation/test partitions retain their official partitions. Committed Caltech101 split files are provided for all three seeds.

## Output structure

```text
outputs/cnn_paper_three_seed/
  table_14_15/
    caltech101/resnet18/
      dt1d/seed_0/
      dt1d/seed_1/
      dt1d/seed_2/
      full/seed_0/...
      linear/seed_0/...
  aggregated/table_14_15/
    raw_runs.csv
    mean_std_numeric.csv
    mean_std_pretty.csv
    manuscript_compact.csv
    manuscript_compact.tex
    seed_completeness.json
```

Each run directory contains `args.json`, `resolved_config.json`, `command.sh`, `environment.json`, `run_metadata.json`, `run_status.json`, `stdout.log`, `history.json`, `convergence_summary.json`, `test_summary.json`, and efficiency output when profiling is enabled.

## Kaggle

Push branch `dt1d-v9-scdq-math-latency`, enable Internet and a GPU, then use [`KAGGLE_SCDQ_FULL_CELL.txt`](KAGGLE_SCDQ_FULL_CELL.txt). Select a target at the top of the cell:

```bash
BRANCH="dt1d-v9-scdq-math-latency"
TARGET="table_14_15"
SEEDS="0,1,2"
```

## Validation

```bash
python tools/verify_reproducibility_package.py
python tools/preflight_cnn_matrix.py --target all
python -m compileall -q main.py engine.py datasets models tools tests
pytest -q
```

A no-download CPU smoke run is also available:

```bash
python tools/run_cnn_paper.py \
  --target table_14_15 \
  --methods dt1d,full,linear \
  --seeds 0 \
  --smoke \
  --output-root /tmp/dt1d-smoke
```

## Release

Prepared branch: `dt1d-v9-scdq-math-latency`  
Prepared tag: `v0.8.0`

See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md), [`MANUSCRIPT_TO_CODE.md`](MANUSCRIPT_TO_CODE.md), [`MANUSCRIPT_ALIGNMENT_NOTES.md`](MANUSCRIPT_ALIGNMENT_NOTES.md), and [`PRETRAINED_WEIGHTS.md`](PRETRAINED_WEIGHTS.md).

## License

Apache License 2.0. See [`LICENSE`](LICENSE).
