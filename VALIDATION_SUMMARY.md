# SCDQ-DT1D v0.9.1 Validation Summary

## Release scope

- Main comparison: Caltech101 + ResNet-18, 10 methods, seeds 0/1/2 (30 runs).
- Mathematical ablation: 13 variants, seeds 0/1/2 (39 runs).
- Final proposal preset: SCDQ-DT1D, quotient support cap 4, reflect padding, no optional pointwise mixer.

## Completed local validation

- Test suite: **308 passed, 1 skipped, 10 subtests passed**.
- Theory validation:
  - quotient rank: 5;
  - exact nullspace residual: 0;
  - legacy-to-MLQ forward maximum absolute error: 2.384185791015625e-07;
  - input-gradient maximum absolute error: 2.9103830456733704e-11;
  - SCDQ joint-axis L1 norm upper bound observed: 1.0.
- Execution plans:
  - main comparison: 30 runs;
  - ablation: 39 runs across 13 variants.
- End-to-end FakeData smoke training for the final SCDQ preset completed successfully.
- Bash syntax validation passed.
- Auxiliary/smoke manifests no longer overwrite committed publication YAML files.

## Important limitation

The complete 69-run Kaggle T4 experiment was not executed in this local environment. The supplied Kaggle cell performs the full GPU comparison, ablation, aggregation, environment capture, and result packaging.
