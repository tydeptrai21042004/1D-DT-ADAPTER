"""Matched-GPU latency A/B/C benchmark for DT1D realizations.

Compares the same class, state_dict, input, precision, and device while changing only:
1. branch form (exact_cost_realization=False),
2. generic exact partition (exact=True, closed_form=False),
3. exact dyadic closed form (exact=True, closed_form=True).
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import random
import statistics
from pathlib import Path
from typing import Callable, Dict, Iterable, List

import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "dt1d_gpu_latency_ab_model", ROOT / "models" / "dt1d_adapter.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
DT1DAdapter = MODULE.DT1DAdapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=21)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--height", type=int, default=56)
    parser.add_argument("--width", type=int, default=56)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=20260801)
    return parser.parse_args()


def make_model(exact: bool, closed: bool, channels: int) -> torch.nn.Module:
    return DT1DAdapter(
        C=channels,
        M=1,
        h=1,
        axis="hw",
        alpha_group=16,
        no_pw=False,
        pw_ratio=32,
        pw_groups=4,
        use_bn=False,
        residual_scale=1.0,
        gate_init=0.01,
        padding_mode="reflect",
        dilations=(1, 2, 4),
        scale_adaptive=True,
        separate_axis_kernels=True,
        gate_temperature=1.0,
        exact_cost_realization=exact,
        closed_form_dyadic_realization=closed,
    ).cuda().eval()


def timed_ms(fn: Callable[[], None], iterations: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end)) / float(iterations)


def percentile(values: List[float], q: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = q * (len(ordered) - 1)
    lo = int(index)
    hi = min(lo + 1, len(ordered) - 1)
    fraction = index - lo
    return ordered[lo] * (1.0 - fraction) + ordered[hi] * fraction


def summarize(values: Iterable[float]) -> Dict[str, float]:
    data = list(values)
    return {
        "rounds": len(data),
        "median_ms": statistics.median(data),
        "mean_ms": statistics.fmean(data),
        "p05_ms": percentile(data, 0.05),
        "p95_ms": percentile(data, 0.95),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this matched-GPU benchmark")
    if args.rounds < 3 or args.iters < 1 or args.warmup < 1:
        raise SystemExit("Use rounds>=3, iters>=1, and warmup>=1")

    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    models = {
        "branch_form": make_model(False, False, args.channels),
        "dynamic_exact": make_model(True, False, args.channels),
        "closed_form": make_model(True, True, args.channels),
    }
    reference_state = copy.deepcopy(models["branch_form"].state_dict())
    for name, model in models.items():
        if name != "branch_form":
            model.load_state_dict(reference_state, strict=True)

    x = torch.randn(
        args.batch_size,
        args.channels,
        args.height,
        args.width,
        device="cuda",
    )
    dtype = torch.float16 if args.amp else torch.float32

    @torch.inference_mode()
    def forward(model: torch.nn.Module) -> torch.Tensor:
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=args.amp):
            return model(x)

    for _ in range(args.warmup):
        for model in models.values():
            forward(model)
    torch.cuda.synchronize()

    outputs = {name: forward(model).float() for name, model in models.items()}
    torch.testing.assert_close(
        outputs["dynamic_exact"], outputs["branch_form"], rtol=5e-3, atol=1e-3
    )
    torch.testing.assert_close(
        outputs["closed_form"], outputs["dynamic_exact"], rtol=5e-3, atol=1e-3
    )

    samples: Dict[str, List[float]] = {name: [] for name in models}
    rng = random.Random(args.seed)
    names = list(models)
    for _ in range(args.rounds):
        order = names[:]
        rng.shuffle(order)
        for name in order:
            samples[name].append(
                timed_ms(lambda model=models[name]: forward(model), args.iters)
            )

    summaries = {name: summarize(values) for name, values in samples.items()}
    branch_median = summaries["branch_form"]["median_ms"]
    dynamic_median = summaries["dynamic_exact"]["median_ms"]
    closed_median = summaries["closed_form"]["median_ms"]
    report = {
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "autocast_enabled": args.amp,
        "autocast_dtype": str(dtype),
        "shape": list(x.shape),
        "rounds": args.rounds,
        "iterations_per_round": args.iters,
        "same_class": True,
        "same_state_dict": True,
        "same_input": True,
        "same_precision": True,
        "runtime_source_patch_used": False,
        "parameter_count": sum(p.numel() for p in models["branch_form"].parameters()),
        "state_dict_keys_identical": all(
            tuple(model.state_dict()) == tuple(models["branch_form"].state_dict())
            for model in models.values()
        ),
        "max_abs_output_diff": {
            "branch_vs_dynamic_exact": float(
                (outputs["branch_form"] - outputs["dynamic_exact"]).abs().max().cpu()
            ),
            "dynamic_exact_vs_closed_form": float(
                (outputs["dynamic_exact"] - outputs["closed_form"]).abs().max().cpu()
            ),
        },
        "latency": summaries,
        "speedup": {
            "dynamic_exact_vs_branch": branch_median / dynamic_median,
            "closed_form_vs_branch": branch_median / closed_median,
            "closed_form_vs_dynamic_exact": dynamic_median / closed_median,
        },
        "raw_samples_ms": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
