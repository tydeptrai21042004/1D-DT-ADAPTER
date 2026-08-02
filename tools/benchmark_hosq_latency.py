#!/usr/bin/env python3
"""CPU structural benchmark for HOSQ-DT1D.

This is a code-path sanity benchmark, not a substitute for the canonical Kaggle
GPU profiler used in the paper.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from torchvision.models import resnet18

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import _add_adapters, set_trainability_policy


def make_model(variant: str) -> torch.nn.Module:
    model = resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 101)
    common = dict(
        tuning_method="dt", dt_M=1, dt_h=1, dt_dilations="1,2,4",
        dt_scale_adaptive=True, dt_separate_axis_kernels=True,
        dt_gate_temperature=1.0, dt_exact_cost_realization=False,
        dt_closed_form_dyadic_realization=False,
        dt_input_adaptive_gate=False, dt_gate_reduction=4,
        dt_axis="hw", dt_tie_sym=True, dt_no_pw=True,
        dt_pw_ratio=32, dt_pw_groups=4, dt_use_bn=False,
        adapt_scale=1.0, dt_gate_init=0.01, dt_padding="reflect",
        kernel_size=3, adapt_size=4,
        dt_hosq_subgroup_size=8, dt_hosq_rank4=1, dt_hosq_rank8=2,
    )
    if variant == "scdq4_group32":
        common.update(dt_alpha_group=32, dt_minimal_quotient_realization=True,
                      dt_quotient_support_cap=4, dt_hosq_realization=False)
    elif variant == "mlq8_group32":
        common.update(dt_alpha_group=32, dt_minimal_quotient_realization=True,
                      dt_quotient_support_cap=8, dt_hosq_realization=False)
    elif variant == "mlq8_group8":
        common.update(dt_alpha_group=8, dt_minimal_quotient_realization=True,
                      dt_quotient_support_cap=8, dt_hosq_realization=False)
    elif variant == "hosq_final":
        common.update(dt_alpha_group=32, dt_minimal_quotient_realization=False,
                      dt_quotient_support_cap=8, dt_hosq_realization=True)
    else:
        raise ValueError(variant)
    args = SimpleNamespace(**common)
    model, adapter_ids = _add_adapters(model, args)
    set_trainability_policy(model, args, adapter_ids)
    return model.eval()


def measure(model, x, warmup: int, iters: int) -> dict[str, float]:
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


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=8)
    p.add_argument("--threads", type=int, default=5)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    x = torch.randn(args.batch_size, 3, 224, 224)
    rows = []
    for variant in ("scdq4_group32", "mlq8_group32", "mlq8_group8", "hosq_final"):
        model = make_model(variant)
        result = measure(model, x, args.warmup, args.iters)
        median = result["median_ms_per_batch"]
        rows.append({
            "variant": variant,
            **result,
            "median_ms_per_image": median / args.batch_size,
            "median_fps": 1000.0 * args.batch_size / median,
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        })
    payload = {
        "environment": {
            "torch": torch.__version__, "device": "cpu", "threads": args.threads,
            "batch_size": args.batch_size, "warmup": args.warmup, "iters": args.iters,
        },
        "results": rows,
        "warning": "Structural CPU benchmark only; use tools/profile_efficiency.py on one GPU for publication claims.",
    }
    out = ROOT / "outputs" / "hosq_validation"
    out.mkdir(parents=True, exist_ok=True)
    (out / "cpu_structural_latency.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
