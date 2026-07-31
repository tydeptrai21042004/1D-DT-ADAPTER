"""Fair benchmark of the branch and exact minimum-cost DT1D realizations.

Both variants use the same source file, kernel construction, channel expansion,
pointwise block, parameters, input, and PyTorch operators. The only changed switch
is exact_cost_realization, so the measured difference isolates the mathematical
branch-partition realization rather than caching or unrelated pipeline engineering.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('dt1d_math_benchmark', ROOT / 'models' / 'hcc_adapter.py')
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
DT1DAdapter = MODULE.DT1DAdapter


def timed_samples(fn, warmup: int, iterations: int, repeats: int):
    for _ in range(warmup):
        fn()
    values = []
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(iterations):
            fn()
        values.append((time.perf_counter() - start) * 1000.0 / iterations)
    values_sorted = sorted(values)
    return {
        'median_ms': float(statistics.median(values)),
        'min_ms': float(min(values)),
        'max_ms': float(max(values)),
        'samples_ms': [float(v) for v in values],
        'q1_ms': float(values_sorted[len(values_sorted) // 4]),
        'q3_ms': float(values_sorted[(3 * len(values_sorted)) // 4]),
    }


def make_pair(channels, args):
    common = dict(
        C=channels,
        M=args.radius,
        dilations=tuple(args.dilations),
        scale_adaptive=True,
        axis='hw',
        alpha_group=args.alpha_group,
        no_pw=args.no_pointwise,
        pw_ratio=args.pointwise_ratio,
        pw_groups=args.pointwise_groups,
        gate_init=0.2,
        padding_mode='reflect',
        separate_axis_kernels=True,
    )
    before = DT1DAdapter(**common, exact_cost_realization=False).to(args.device)
    after = DT1DAdapter(**common, exact_cost_realization=True).to(args.device)
    after.load_state_dict(before.state_dict(), strict=True)
    return before, after


def case(args, channels, height, width):
    before, after = make_pair(channels, args)
    x = torch.randn(args.batch_size, channels, height, width, device=args.device)
    before.eval(); after.eval()
    with torch.inference_mode():
        y_before = before(x)
        y_after = after(x)
        forward_before = timed_samples(lambda: before(x), args.warmup, args.iterations, args.repeats)
        forward_after = timed_samples(lambda: after(x), args.warmup, args.iterations, args.repeats)
    before.train(); after.train()
    xb = x.detach().clone().requires_grad_(True)
    xa = x.detach().clone().requires_grad_(True)

    def train_step(model, sample):
        model.zero_grad(set_to_none=True)
        sample.grad = None
        model(sample).square().mean().backward()

    train_iterations = max(2, args.iterations // 4)
    train_before = timed_samples(lambda: train_step(before, xb), max(1, args.warmup // 2), train_iterations, args.repeats)
    train_after = timed_samples(lambda: train_step(after, xa), max(1, args.warmup // 2), train_iterations, args.repeats)
    cost_h = after.exact_realization_cost(x, 'h')
    cost_w = after.exact_realization_cost(x, 'w')
    return {
        'shape': [args.batch_size, channels, height, width],
        'max_abs_output_difference': float((y_before - y_after).abs().max().item()),
        'prediction_proxy_sign_agreement': float(((y_before >= 0) == (y_after >= 0)).float().mean().item()),
        'cost_height': cost_h,
        'cost_width': cost_w,
        'before_forward': forward_before,
        'after_forward': forward_after,
        'forward_speedup': forward_before['median_ms'] / forward_after['median_ms'],
        'before_forward_backward': train_before,
        'after_forward_backward': train_after,
        'forward_backward_speedup': train_before['median_ms'] / train_after['median_ms'],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--radius', type=int, default=1)
    parser.add_argument('--dilations', type=int, nargs='+', default=[1, 2, 4])
    parser.add_argument('--alpha-group', type=int, default=16)
    parser.add_argument('--no-pointwise', action='store_true')
    parser.add_argument('--pointwise-ratio', type=int, default=8)
    parser.add_argument('--pointwise-groups', type=int, default=4)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--iterations', type=int, default=15)
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--threads', type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument('--output', default='benchmark_dt1d_math_realization.json')
    args = parser.parse_args()
    if args.device == 'cpu':
        torch.set_num_threads(args.threads)
    torch.manual_seed(20260731)
    cases = [(64, 56, 56), (128, 28, 28), (256, 14, 14), (512, 7, 7)]
    result = {
        'fairness_controls': {
            'same_class': True,
            'same_kernel_builder': True,
            'same_channel_expansion': True,
            'same_parameters': True,
            'same_pointwise_block': True,
            'same_input': True,
            'cache_used': False,
            'only_switch': 'exact_cost_realization',
        },
        'torch_version': torch.__version__,
        'device': args.device,
        'threads': torch.get_num_threads(),
        'results': [case(args, *shape) for shape in cases],
    }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
