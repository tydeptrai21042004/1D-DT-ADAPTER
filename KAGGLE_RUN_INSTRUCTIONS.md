# Kaggle execution

1. Enable a GPU and Internet.
2. Upload the full repository ZIP as a Kaggle Dataset or clone the repository branch.
3. Run `bash KAGGLE_CNN_THREE_SEED_RUN.sh` from the repository root.

Environment variables:

```bash
DATA_DIR=/kaggle/working/data
SEEDS=0,1,2
DEVICE=cuda
RUN_COMPARISON=1
RUN_ABLATION=1
```

The script runs tests, mathematical validation, the full paper comparison matrix, the focused HOSQ-Lite-C1-Orth ablation, aggregation, and result packaging. Set either run flag to `0` to skip that section.
