import tempfile
from pathlib import Path

import numpy as np
import torch

import importlib.util

REPO_ROOT = Path(__file__).resolve().parents[1]
_HCC_PATH = REPO_ROOT / "models" / "hcc_adapter.py"
_spec = importlib.util.spec_from_file_location("hcc_adapter_direct", _HCC_PATH)
_hcc_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_hcc_mod)
HCCAdapter = _hcc_mod.HCCAdapter


def safe_torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def test_weighted_hcc_kernel_shape_and_l1():
    m = HCCAdapter(C=7, M=2, h=1, axis="hw", alpha_group=3,
                   dilations="1,2,4", scale_adaptive=True,
                   input_adaptive_gate=True, no_pw=True)
    w = m._build_weighted_hcc_kernel_1d(0, 0, torch.device("cpu"), torch.float32)
    assert tuple(w.shape) == (7, 1, 7)  # 2M+3 for M=2
    assert torch.all(w.squeeze(1).abs().sum(dim=1) <= 1.0 + 1e-6)


def test_input_adaptive_routing_shape_and_sum():
    m = HCCAdapter(C=8, M=1, axis="hw", alpha_group=4,
                   dilations="1,2,4", scale_adaptive=True,
                   input_adaptive_gate=True, gate_reduction=4, no_pw=True)
    x = torch.randn(3, 8, 12, 12)
    weights = m.axis_scale_weights(x)
    assert tuple(weights.shape) == (3, 2, 3)
    assert torch.allclose(weights.reshape(3, -1).sum(dim=1), torch.ones(3), atol=1e-6)


def test_forward_backward_static_and_adaptive():
    for adaptive in (False, True):
        m = HCCAdapter(C=8, M=1, axis="hw", alpha_group=4,
                       dilations="1,2", scale_adaptive=True,
                       input_adaptive_gate=adaptive, no_pw=False,
                       pw_ratio=8, pw_groups=2, gate_init=0.01)
        x = torch.randn(2, 8, 10, 10, requires_grad=True)
        y = m(x)
        assert y.shape == x.shape
        y.mean().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


def test_small_feature_map_reflect_fallback():
    m = HCCAdapter(C=4, M=2, axis="hw", alpha_group=2,
                   dilations="3", scale_adaptive=False,
                   input_adaptive_gate=False, no_pw=True,
                   padding_mode="reflect")
    x = torch.randn(2, 4, 3, 3)
    y = m(x)
    assert y.shape == x.shape


def test_checkpoint_loading_with_numpy_scalar_payload():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "ckpt.pth"
        torch.save({"model": {"dummy": torch.tensor([1.0])}, "np_scalar": np.float64(1.0)}, path)
        ckpt = safe_torch_load(path)
        assert "model" in ckpt
        assert "np_scalar" in ckpt
