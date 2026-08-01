# Manuscript-to-Code Mapping — CNN Package

All stochastic targets default to seeds `0,1,2`. Every comparison target includes Full fine-tuning and Linear probing.

| Manuscript item | Exact command |
|---|---|
| Table 2 | `bash scripts/tables/table_02_three_seed.sh` |
| Table 3 | `bash scripts/tables/table_03_three_seed.sh` |
| Table 4 | `bash scripts/tables/table_04_three_seed.sh` |
| Table 5 | `bash scripts/tables/table_05_three_seed.sh` |
| Table 6 | `bash scripts/tables/table_06_three_seed.sh` |
| Table 7 | `bash scripts/tables/table_07_three_seed.sh` |
| Table 8 | `bash scripts/tables/table_08_three_seed.sh` |
| Table 9 | `bash scripts/tables/table_09_three_seed.sh` |
| Table 10 | `bash scripts/tables/table_10_three_seed.sh` |
| Table 11 | `bash scripts/tables/table_11_three_seed.sh` |
| Table 12 | `bash scripts/tables/table_12_three_seed.sh` |
| Table 13 | `bash scripts/tables/table_13_three_seed.sh` |
| Tables 14–15 | `bash scripts/tables/table_14_15_three_seed.sh` |
| Tables 18–19 | `bash scripts/tables/table_18_19_three_seed.sh` |
| Figure 1 | `bash scripts/figures/figure_01_three_seed.sh` |
| Figure 2 | `bash scripts/figures/figure_02_deterministic.sh` |
| Figure 3 | `bash scripts/figures/figure_03_deterministic.sh` |
| Figure 4 | `bash scripts/figures/figure_04_three_seed.sh` |

The complete matrix is `configs/paper/cnn_three_seed_manifest.yaml`. Per-run resolved YAML files are generated under `configs/paper/generated/` and copied into each result directory through `resolved_config.json`.

## Run only Full fine-tuning and Linear probing

```bash
python tools/run_cnn_paper.py \
  --target table_03 \
  --methods full,linear \
  --seeds 0,1,2 \
  --data-path /path/to/data \
  --device cuda
```

## Run all implemented CNN methods

```bash
python tools/run_cnn_paper.py \
  --target table_03 \
  --methods all-cnn \
  --seeds 0,1,2 \
  --data-path /path/to/data \
  --device cuda
```

## Model-construction preflight

```bash
python tools/preflight_cnn_matrix.py --target all \
  --output outputs/cnn_model_preflight.json
```

The generated per-run configurations for each row are under `configs/paper/generated/<target>/`.
