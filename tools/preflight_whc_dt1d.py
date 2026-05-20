#!/usr/bin/env python3
"""Preflight tests for WHC-DT1D / HCCAdapter before running long experiments.

This script is intentionally lightweight and dataset-free. It checks:
  1. finite weighted h-Hartley-cosine kernel shape and L1 normalization;
  2. static and input-adaptive routing shapes and softmax sums;
  3. forward/backward gradients for several axis/dilation settings;
  4. small-feature-map padding fallback;
  5. PyTorch 2.6-compatible checkpoint loading with weights_only=False;
  6. optional torchvision ResNet integration when torchvision is available.

Run from the repo root:
  PYTHONPATH=. python tools/preflight_whc_dt1d.py --device auto
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

# Allow running as `python tools/preflight_whc_dt1d.py` from any directory.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import importlib.util  # noqa: E402

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


def resolve_device(requested: str) -> Tuple[torch.device, bool]:
    req = str(requested).lower()
    if req == "auto":
        req = "cuda" if torch.cuda.is_available() else "cpu"
    if req.startswith("cuda") and torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        name = torch.cuda.get_device_name(0)
        print(f"[Device] CUDA GPU: {name}, capability={cap}")
        # Newer Kaggle PyTorch wheels may not support Tesla P100 sm_60.
        if cap[0] < 7:
            print("[Device] capability < 7.0; using CPU to avoid no-kernel-image crash")
            return torch.device("cpu"), False
        return torch.device("cuda"), True
    print("[Device] using CPU")
    return torch.device("cpu"), False


def assert_close(a, b, msg, atol=1e-6):
    if not torch.allclose(a, b, atol=atol, rtol=0):
        raise AssertionError(f"{msg}: max_abs={float((a-b).abs().max())}")


def test_kernel_shape_and_l1(device: torch.device):
    m = HCCAdapter(
        C=7, M=2, h=1, axis="hw", alpha_group=3,
        dilations="1,2,4", scale_adaptive=True,
        separate_axis_kernels=True, input_adaptive_gate=True,
        gate_reduction=4, no_pw=True, padding_mode="reflect",
    ).to(device)
    w = m._build_weighted_hcc_kernel_1d(0, 0, device, torch.float32)
    expected_k = 2 * m.M + 3
    assert tuple(w.shape) == (7, 1, expected_k), tuple(w.shape)
    l1 = w.squeeze(1).abs().sum(dim=1)
    assert torch.all(l1 <= 1.0 + 1e-5), l1
    print("[OK] weighted-HCC kernel shape and L1 normalization", tuple(w.shape))


def test_routing_and_forward_backward(device: torch.device):
    configs = [
        dict(axis="h", dilations="1", scale_adaptive=False, input_adaptive_gate=False),
        dict(axis="w", dilations="2", scale_adaptive=False, input_adaptive_gate=False),
        dict(axis="hw", dilations="1,2,4", scale_adaptive=True, input_adaptive_gate=False),
        dict(axis="hw", dilations="1,2,4", scale_adaptive=True, input_adaptive_gate=True),
    ]
    for cfg in configs:
        torch.manual_seed(123)
        m = HCCAdapter(
            C=8, M=1, h=1, alpha_group=4,
            separate_axis_kernels=True, gate_reduction=4,
            no_pw=False, pw_ratio=8, pw_groups=2,
            gate_init=0.01, padding_mode="reflect", **cfg,
        ).to(device)
        x = torch.randn(3, 8, 12, 10, device=device, requires_grad=True)
        y = m(x)
        assert y.shape == x.shape, (y.shape, x.shape, cfg)
        loss = y.square().mean()
        loss.backward()
        assert x.grad is not None and torch.isfinite(x.grad).all(), cfg
        for name, p in m.named_parameters():
            if p.requires_grad and p.grad is not None:
                assert torch.isfinite(p.grad).all(), (cfg, name)
        if m.scale_adaptive:
            weights = m.axis_scale_weights(x.detach())
            if m.input_adaptive_gate:
                assert tuple(weights.shape) == (3, m.num_axes, m.num_scales), tuple(weights.shape)
                sums = weights.reshape(3, -1).sum(dim=1)
                assert_close(sums, torch.ones_like(sums), "adaptive routing weights must sum to 1")
            else:
                assert tuple(weights.shape) == (m.num_axes, m.num_scales), tuple(weights.shape)
                assert_close(weights.sum(), torch.tensor(1.0, device=weights.device), "static routing weights must sum to 1")
        print("[OK] forward/backward/routing", cfg)


def test_small_feature_padding(device: torch.device):
    # pad can exceed H/W for large dilation. reflect should fall back to replicate.
    m = HCCAdapter(
        C=4, M=2, h=1, axis="hw", alpha_group=2,
        dilations="3", scale_adaptive=False,
        input_adaptive_gate=False, no_pw=True,
        padding_mode="reflect", gate_init=0.01,
    ).to(device)
    x = torch.randn(2, 4, 3, 3, device=device)
    y = m(x)
    assert y.shape == x.shape, (y.shape, x.shape)
    print("[OK] small-feature-map padding fallback")


def test_checkpoint_compatibility():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "checkpoint-best.pth"
        torch.save({"model": {"dummy": torch.tensor([1.0])}, "np_scalar": np.float64(1.23)}, path)
        ckpt = safe_torch_load(path, map_location="cpu")
        assert "model" in ckpt and "np_scalar" in ckpt
    print("[OK] checkpoint safe_torch_load handles NumPy scalar payload")


def test_torchvision_integration(device: torch.device):
    try:
        import torchvision
        from torchvision.models.resnet import BasicBlock
    except Exception as e:
        print(f"[SKIP] torchvision integration unavailable: {e}")
        return

    m = torchvision.models.resnet18(weights=None, num_classes=5)
    attached = 0
    for mod in list(m.modules()):
        if isinstance(mod, BasicBlock) and not hasattr(mod, "dt1d"):
            ch = mod.conv2.out_channels
            ad = HCCAdapter(
                C=ch, M=1, h=1, axis="hw", alpha_group=16,
                dilations="1,2", scale_adaptive=True,
                input_adaptive_gate=True, gate_reduction=4,
                no_pw=True, gate_init=0.01, padding_mode="reflect",
            )
            mod.add_module("dt1d", ad)
            mod.register_forward_hook(lambda mm, inp, out: mm.dt1d(out))
            attached += 1
    assert attached > 0
    m.to(device)
    m.train()
    x = torch.randn(2, 3, 64, 64, device=device)
    target = torch.tensor([0, 1], device=device)
    out = m(x)
    loss = nn.CrossEntropyLoss()(out, target)
    loss.backward()
    assert out.shape == (2, 5), out.shape
    print(f"[OK] torchvision ResNet integration, attached={attached}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto", help="auto, cpu, cuda")
    ap.add_argument("--skip-torchvision", action="store_true")
    args = ap.parse_args()

    device, cuda_ok = resolve_device(args.device)
    print(json.dumps({
        "torch_version": torch.__version__,
        "device": str(device),
        "cuda_ok": cuda_ok,
    }, indent=2))

    test_kernel_shape_and_l1(device)
    test_routing_and_forward_backward(device)
    test_small_feature_padding(device)
    test_checkpoint_compatibility()
    if not args.skip_torchvision:
        test_torchvision_integration(device)
    print("[ALL OK] WHC-DT1D preflight passed")


if __name__ == "__main__":
    main()
