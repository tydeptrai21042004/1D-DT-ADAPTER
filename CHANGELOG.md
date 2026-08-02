# v0.10.0 — HOSQ-DT1D hierarchical quotient release

- Added HOSQ-DT1D: an MLQ8 Group-32 coarse quotient plus low-rank orthogonal
  Group-8 channel details at offsets 4 and 8.
- Added fixed hierarchical Haar contrasts for four subgroups and canonical
  zero-mean orthogonal remainder-group contrasts.
- Added zero-DC spatial detail atoms and joint height/width L1 projection.
- Added CLI flags `dt_hosq_realization`, `dt_hosq_subgroup_size`,
  `dt_hosq_rank4`, and `dt_hosq_rank8`.
- Updated the main Table 14–15 proposal preset to HOSQ-DT1D without changing
  the Git branch name.
- Added an eight-variant HOSQ ablation matrix over four dataset/backbone
  settings, each with seeds 0, 1, and 2.
- Added deterministic theory validation, CPU structural benchmarking, and
  end-to-end FakeData training smoke validation.
- Verified the ResNet-18 HOSQ adapter budget: 968 adapter parameters and 52,781
  total trainable parameters on Caltech101.
- Validation: 316 passed, 1 skipped, 10 subtests passed.

# v0.9.1 — corrected SCDQ/Kaggle release

- Makes SCDQ4 with reflect padding and no pointwise mixer the Table 14–15 proposal preset.
- Adds a 13-variant, three-seed SCDQ ablation manifest.
- Adds one Kaggle cell for the 30-run comparison and 39-run ablation.
- Writes `run_metadata.json` for direct YAML runs so aggregation is complete.
- Fixes strict BitFit on torchvision ResNet by retaining trainable backbone biases while freezing BatchNorm statistics.
- Keeps frozen BatchNorm modules in evaluation mode after every `model.train(True)` call.
- Validates 52,781 trainable parameters for the final Caltech101/ResNet-18 configuration.

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

## v0.9.0 — SCDQ mathematical latency revision

- Added the Minimal Laurent Quotient parameterization for the dyadic `M=1`, `d={1,2,4}` shifted-symmetric axial operator.
- Proved and tested the rank-five quotient and exact scale-cancellation null direction.
- Added the Spectrally Closed Dyadic Quotient (`support_cap=4`) final variant.
- Added joint two-axis L1 projection and non-expansive stability tests.
- Added exact legacy-to-MLQ8 forward and input-gradient conversion tests.
- Added 30 three-seed Caltech101/ResNet-18 ablation YAML files.
- Added deterministic theory and structural latency validation scripts.
- Full suite: 304 passed, 1 skipped, 10 subtests passed.
