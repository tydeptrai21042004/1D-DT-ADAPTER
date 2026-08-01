import torch
import importlib.util
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("dt1d_adapter_direct", _ROOT / "models" / "dt1d_adapter.py")
_MOD = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MOD)
DT1DAdapter = _MOD.DT1DAdapter


def test_static_gate_no_router_created_even_if_flag_true():
    m = DT1DAdapter(C=16, M=1, dilations="1,2,4", scale_adaptive=True, axis="hw", input_adaptive_gate=True)
    assert m.axis_scale_router is None
    assert m.input_adaptive_gate is False
    bd = m.parameter_count_breakdown()
    assert "axis_scale_router" not in bd


def test_weighted_dt1d_kernel_length_and_l1():
    m = DT1DAdapter(C=16, M=2, dilations="1,2,4", scale_adaptive=True, axis="hw", alpha_group=8)
    k = m._build_weighted_dt1d_kernel_1d(0, 0, torch.device("cpu"), torch.float32)
    assert k.shape == (16, 1, 7)  # 2M+3 for M=2
    assert torch.all(k.squeeze(1).abs().sum(dim=1) <= 1.00001)


def test_axis_scale_weights_static_shape_sum():
    m = DT1DAdapter(C=16, M=1, dilations="1,2,4", scale_adaptive=True, axis="hw")
    w = m.axis_scale_weights()
    assert w.shape == (2, 3)
    assert torch.allclose(w.sum(), torch.tensor(1.0), atol=1e-6)


def test_forward_backward():
    torch.manual_seed(0)
    m = DT1DAdapter(C=8, M=1, dilations="1,2", scale_adaptive=True, axis="hw", no_pw=False, gate_init=0.01)
    x = torch.randn(2, 8, 12, 12, requires_grad=True)
    y = m(x)
    assert y.shape == x.shape
    y.mean().backward()
    assert x.grad is not None
    for p in m.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()


def test_single_dilation_mode():
    m = DT1DAdapter(C=8, M=1, h=1, dilations=None, scale_adaptive=False, axis="hw", no_pw=True)
    x = torch.randn(1, 8, 8, 8)
    y = m(x)
    assert y.shape == x.shape
