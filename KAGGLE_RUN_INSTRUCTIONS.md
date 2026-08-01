# Kaggle execution

1. Enable a GPU and Internet in the Kaggle notebook.
2. Upload `DT1D_V9_SCDQ_KAGGLE_FINAL.zip` as a Kaggle Dataset, or push the repository to branch `dt1d-v9-scdq-math-latency`.
3. Paste the complete contents of `KAGGLE_SCDQ_FULL_CELL.txt` into one Kaggle cell.

The cell automatically prefers the uploaded archive. If it is absent, it clones the branch. By default it runs:

- 30 main comparison runs: 10 methods × seeds 0, 1, 2;
- 39 ablation runs: 13 variants × seeds 0, 1, 2;
- source tests, mathematical validation, model preflight, aggregation, LaTeX export, and result packaging.

To run only the proposal ablations, set `RUN_COMPARISON=0` near the beginning of the cell. To run only the main Table 14–15 comparison, set `RUN_ABLATION=0`.
