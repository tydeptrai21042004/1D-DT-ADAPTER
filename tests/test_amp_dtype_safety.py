"""Mixed-precision regression coverage for HOSQ-Lite-C1-Orth."""

import pytest
import torch

from models.hosq_lite_c1_adapter import HOSQLiteC1Adapter


def make_model():
    return HOSQLiteC1Adapter(
        C=16, axis="hw", alpha_group=16, gate_init=0.01,
        padding_mode="replicate", detail_basis="orth", detail_components="both",
    )


@pytest.mark.parametrize("target_dtype", [torch.float16, torch.bfloat16])
def test_kernel_builder_respects_requested_dtype(target_dtype):
    model = make_model()
    kernel = model.build_kernels(torch.device("cpu"), target_dtype)
    assert kernel.dtype == target_dtype
    assert torch.isfinite(kernel.float()).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_autocast_forward_and_backward():
    torch.manual_seed(20260804)
    model = make_model().cuda().train()
    x = torch.randn(2, 16, 19, 23, device="cuda", requires_grad=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        y = model(x)
        loss = y.float().square().mean()
    loss.backward()
    assert torch.isfinite(y).all()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for parameter in model.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()
