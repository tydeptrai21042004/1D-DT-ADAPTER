# Validation summary

The revised package is centered on HOSQ-Lite-C1-Orth and retains original DT1D only as the submitted-method baseline.

Required validation commands:

```bash
pytest -q
python tools/validate_hosq_lite_c1.py
python tools/benchmark_hosq_lite_latency.py --batch-size 2 --warmup 3 --iters 15
python tools/run_cnn_paper.py --target table_02 --seeds 0 --smoke --max-runs 2 --output-root /tmp/hosq-lite-smoke
```

The mathematical validator checks original-DT1D warm-start equivalence, zero-DC and orthogonal atoms, conditioning, focused spectral ablations, the joint two-axis L1 bound, parameter count, and convolution count. CPU latency is structural evidence only; use synchronized GPU profiling for manuscript numbers.
