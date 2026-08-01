# Changelog

## 0.8.0 — 2026-08-01

- Restricted the publication execution matrix to CNN classification only: manuscript Tables 2–15 and 18–19, plus result-driven Figures 1 and 4.
- Added independent seed support for seeds 0, 1, and 2, including Python, NumPy, PyTorch CPU/CUDA, samplers, data-loader workers, initialization, augmentation, and generated validation partitions.
- Added runnable Full fine-tuning and Linear probing controls to every cross-method CNN comparison target.
- Added 408 committed, portable, deterministic per-run YAML configurations generated from one versioned manifest.
- Added one shell entry point per table/figure, a complete CNN batch runner, and a configurable Kaggle `%%bash` runner.
- Added strict three-seed aggregation with raw-result export, mean ± sample-standard-deviation summaries, CSV/JSON/LaTeX output, and fail-fast seed-completeness checks.
- Added three-seed Figure 1 convergence and Figure 4 accuracy–parameter trade-off regeneration.
- Added deterministic Figure 2 spectral and CNN-only Figure 3 architecture generators and reference outputs.
- Corrected Caltech101 and EuroSAT evaluation to use disjoint 80% train / 10% validation / 10% test partitions instead of reusing one holdout for model selection and final testing.
- Added committed Caltech101 6,942/868/867 train/validation/test split manifests for seeds 0, 1, and 2 and automatic split recording in every run directory.
- Added a no-download CNN model preflight covering all 136 unique target/method/variant model configurations.
- Added real FakeData CPU execution smoke tests for all 13 supported method presets and regression tests for Full fine-tuning, Linear probing, split disjointness, manifest coverage, and aggregation completeness.
- Added `MANUSCRIPT_ALIGNMENT_NOTES.md`, including a transparent Table 9 trainable-parameter mismatch that must be resolved by regenerating the table from the canonical implementation.
- Standardized the implementation name as `DT1D-Adapter`, `DT1DAdapter`, `models/dt1d_adapter.py`, `--tuning_method dt`, and `--dt_*`; the legacy `hcc_adapter.py` remains only as a compatibility shim.
- Removed the unused Swin/Transformer source from this CNN-only release and added explicit non-CNN rejection in the paper runner.
- Fixed the missing `Path` import in exact split-file loading and removed runtime source patching from all execution scripts.

## 0.7.0 — 2026-08-01

- Integrated the AMP dtype correction into the canonical source.
- Added the initial publication reproducibility package and release metadata.
