"""Regression coverage for exact-kernel accumulation dtype safety."""

import importlib.util
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "dt1d_amp_dtype_safety", ROOT / "models" / "dt1d_adapter.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
DT1DAdapter = MODULE.DT1DAdapter


def make_model() -> torch.nn.Module:
    return DT1DAdapter(
        C=16,
        M=1,
        h=1,
        axis="hw",
        alpha_group=16,
        no_pw=False,
        pw_ratio=32,
        pw_groups=4,
        gate_init=0.01,
        padding_mode="reflect",
        dilations=(1, 2, 4),
        scale_adaptive=True,
        separate_axis_kernels=True,
        gate_temperature=1.0,
        exact_cost_realization=True,
        closed_form_dyadic_realization=False,
    )


@pytest.mark.parametrize("target_dtype", [torch.float16, torch.bfloat16])
def test_exact_group_kernel_casts_source_to_accumulator_dtype(target_dtype):
    """Reproduce the former float32-source/low-precision-index_add mismatch."""
    model = make_model()
    route_weights = model._compute_axis_scale_weights(torch.device("cpu"), torch.float32)
    kernel = model._build_exact_group_kernel_1d(
        axis_idx=0,
        scale_indices=(0, 1),
        axis_scale_weights=route_weights,
        device=torch.device("cpu"),
        dtype=target_dtype,
    )
    assert kernel.dtype == target_dtype
    assert torch.isfinite(kernel.float()).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_autocast_dynamic_exact_forward_and_backward():
    torch.manual_seed(20260801)
    model = make_model().cuda().train()
    x = torch.randn(2, 16, 19, 23, device="cuda", requires_grad=True)

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        y = model(x)
        loss = y.float().square().mean()
    loss.backward()

    assert torch.isfinite(y).all()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
