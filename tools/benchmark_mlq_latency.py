#!/usr/bin/env python3
"""CPU structural benchmark; use the repository GPU profiler for paper numbers."""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
from types import SimpleNamespace

import torch
from torchvision.models import resnet18

from main import _add_adapters


def make_model(mode: str, no_pw: bool, cap: int) -> torch.nn.Module:
    model = resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 101)
    args = SimpleNamespace(
        tuning_method="dt", dt_M=1, dt_h=1, dt_dilations="1,2,4",
        dt_scale_adaptive=True, dt_separate_axis_kernels=True,
        dt_gate_temperature=1.0, dt_exact_cost_realization=mode != "branch",
        dt_closed_form_dyadic_realization=mode == "closed",
        dt_minimal_quotient_realization=mode == "mlq",
        dt_quotient_support_cap=cap, dt_input_adaptive_gate=False,
        dt_gate_reduction=4, dt_axis="hw", dt_alpha_group=16,
        dt_tie_sym=True, dt_no_pw=no_pw, dt_pw_ratio=32,
        dt_pw_groups=4, dt_use_bn=False, adapt_scale=1.0,
        dt_gate_init=0.01, dt_padding="replicate", kernel_size=3,
        adapt_size=4,
    )
    _add_adapters(model, args)
    return model.eval()


def measure(model, x, warmup: int, iters: int):
    values = []
    with torch.inference_mode():
        for _ in range(warmup):
            model(x)
        for _ in range(iters):
            t0 = time.perf_counter()
            model(x)
            values.append((time.perf_counter() - t0) * 1000.0)
    return {
        "median_ms_per_batch": statistics.median(values),
        "mean_ms_per_batch": statistics.mean(values),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=8)
    p.add_argument("--threads", type=int, default=5)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    x = torch.randn(args.batch_size, 3, 224, 224)
    variants = [
        ("legacy_closed_pointwise", "closed", False, 8),
        ("mlq8_pointwise", "mlq", False, 8),
        ("scdq4_pointwise", "mlq", False, 4),
        ("mlq8_core", "mlq", True, 8),
        ("scdq4_core_final", "mlq", True, 4),
    ]
    rows = []
    for name, mode, no_pw, cap in variants:
        r = measure(make_model(mode, no_pw, cap), x, args.warmup, args.iters)
        median = r["median_ms_per_batch"]
        rows.append({
            "variant": name,
            **r,
            "median_ms_per_image": median / args.batch_size,
            "median_fps": 1000.0 * args.batch_size / median,
        })
    baseline = rows[0]
    for r in rows:
        r["latency_reduction_vs_legacy_pct"] = 100.0 * (
            baseline["median_ms_per_batch"] - r["median_ms_per_batch"]
        ) / baseline["median_ms_per_batch"]
        r["throughput_gain_vs_legacy_pct"] = 100.0 * (
            r["median_fps"] - baseline["median_fps"]
        ) / baseline["median_fps"]
    payload = {"environment": {
        "torch": torch.__version__, "device": "cpu", "threads": args.threads,
        "batch_size": args.batch_size, "warmup": args.warmup, "iters": args.iters,
    }, "results": rows}
    out = Path("outputs/mlq_validation")
    out.mkdir(parents=True, exist_ok=True)
    (out / "cpu_structural_latency.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
