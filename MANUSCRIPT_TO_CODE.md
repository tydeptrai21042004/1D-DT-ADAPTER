# Manuscript-to-code map

| Manuscript concept | Code |
|---|---|
| Original DT1D baseline | `models/dt1d_adapter.py::DT1DAdapter` |
| HOSQ-Lite-C1-Orth proposal | `models/hosq_lite_c1_adapter.py::HOSQLiteC1Adapter` |
| Observable coarse quotient | `HOSQLiteC1Adapter.quotient_beta` |
| Weighted zero-mean channel contrast | `HOSQLiteC1Adapter._make_weighted_channel_contrast` |
| Orthonormal zero-DC atoms | `HOSQLiteC1Adapter._spectral_atoms` |
| Joint two-axis L1 projection | `HOSQLiteC1Adapter.build_kernels` |
| Exact original-DT1D warm start | `HOSQLiteC1Adapter.initialize_from_dt1d` |
| One-convolution-per-axis forward | `HOSQLiteC1Adapter.forward` |
| Main paper preset | `configs/paper/cnn_three_seed_manifest.yaml` |
| Focused ablation | `configs/experiments/hosq_lite_c1_ablation.yaml` |
| Mathematical validation | `tools/validate_hosq_lite_c1.py` |
| Paired latency benchmark | `tools/benchmark_hosq_lite_latency.py` |
| Unit and invariant tests | `tests/test_hosq_lite_c1.py` |

Only HOSQ-Lite-C1-Orth is the revised proposal. `legacy` selects the original DT1D baseline. `none`, `offset4`, `offset8`, and `raw` are ablations of the revised method.
