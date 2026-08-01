#!/usr/bin/env python3
"""Preflight checks for DT1D-Adapter with static/global axis-scale gates.

Run before long Kaggle experiments:
    PYTHONPATH=. python tools/preflight_dt1d_static.py --device auto
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch

import importlib.util
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("dt1d_adapter_direct", _ROOT / "models" / "dt1d_adapter.py")
_MOD = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MOD)
DT1DAdapter = _MOD.DT1DAdapter


def resolve_device(requested: str) -> torch.device:
    req = requested.lower()
    if req == "auto":
        req = "cuda" if torch.cuda.is_available() else "cpu"
    if req.startswith("cuda") and torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        name = torch.cuda.get_device_name(0)
        print(f"[Device] CUDA GPU={name}, capability={cap}")
        if cap[0] >= 7:
            return torch.device("cuda")
        print("[Device] capability < 7.0; using CPU to avoid no-kernel-image crash.")
    return torch.device("cpu")


def assert_close(x: torch.Tensor, target: float, name: str, atol: float = 1e-5) -> None:
    if not torch.allclose(x, torch.full_like(x, target), atol=atol, rtol=0):
        raise AssertionError(f"{name} not close to {target}: {x}")


def make_adapter(C=16, M=1, no_pw=False) -> DT1DAdapter:
    return DT1DAdapter(
        C=C,
        M=M,
        h=1,
        dilations="1,2,4",
        scale_adaptive=True,
        separate_axis_kernels=True,
        gate_temperature=1.0,
        axis="hw",
        alpha_group=8,
        no_pw=no_pw,
        pw_ratio=16,
        pw_groups=4,
        gate_init=0.01,
        padding_mode="reflect",
        # These are intentionally accepted but ignored in the static-gate version.
        input_adaptive_gate=True,
        gate_reduction=4,
    )


def test_kernel_and_routing(device: torch.device) -> None:
    torch.manual_seed(0)
    m = make_adapter(C=16, M=2).to(device)
    assert not getattr(m, "input_adaptive_gate", True), "input_adaptive_gate should be disabled"
    assert getattr(m, "axis_scale_router", None) is None, "axis_scale_router should not exist"

    k = m._build_weighted_dt1d_kernel_1d(0, 0, device, torch.float32)
    expected_K = 2 * m.M + 3
    assert tuple(k.shape) == (16, 1, expected_K), tuple(k.shape)
    l1 = k.squeeze(1).abs().sum(dim=1)
    assert torch.all(l1 <= 1.00001), l1.max().item()

    w = m.axis_scale_weights()
    assert tuple(w.shape) == (2, 3), tuple(w.shape)
    assert_close(w.sum(), 1.0, "static axis-scale weight sum")

    bd = m.parameter_count_breakdown()
    assert "axis_scale_router" not in bd, bd
    print("[OK] kernel length, L1 normalization, static routing, no router params")


def test_forward_backward(device: torch.device) -> None:
    torch.manual_seed(1)
    m = make_adapter(C=16, M=1).to(device)
    x = torch.randn(2, 16, 16, 16, device=device, requires_grad=True)
    y = m(x)
    assert y.shape == x.shape, (y.shape, x.shape)
    loss = y.square().mean()
    loss.backward()
    bad = []
    for name, p in m.named_parameters():
        if p.requires_grad and p.grad is not None and not torch.isfinite(p.grad).all():
            bad.append(name)
    if bad:
        raise AssertionError(f"Non-finite gradients: {bad}")
    print("[OK] forward/backward finite")


def test_small_feature_maps(device: torch.device) -> None:
    torch.manual_seed(2)
    m = make_adapter(C=8, M=2, no_pw=True).to(device)
    for size in [2, 3, 5]:
        x = torch.randn(1, 8, size, size, device=device)
        y = m(x)
        assert y.shape == x.shape, (size, y.shape)
    print("[OK] small feature maps")


def test_checkpoint_load() -> None:
    # Reproduce the PyTorch 2.6 weights_only issue with numpy scalar payloads.
    # Prefer main.safe_torch_load when dependencies are installed; otherwise use
    # the same fallback locally so this preflight can run before pip install too.
    try:
        from main import safe_torch_load
    except Exception:
        def safe_torch_load(path, map_location="cpu"):
            try:
                return torch.load(path, map_location=map_location, weights_only=False)
            except TypeError:
                return torch.load(path, map_location=map_location)

    payload = {"model": {"x": torch.ones(1)}, "numpy_scalar": np.float64(1.23)}
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "ckpt.pth"
        torch.save(payload, p)
        got = safe_torch_load(str(p), map_location="cpu")
    assert "model" in got and "numpy_scalar" in got
    print("[OK] safe_torch_load checkpoint compatibility")


def test_torchvision_integration(device: torch.device) -> None:
    try:
        import torchvision
        from torchvision.models.resnet import BasicBlock, Bottleneck
    except Exception as e:
        print(f"[SKIP] torchvision integration unavailable: {e}")
        return

    m = torchvision.models.resnet18(weights=None, num_classes=10)
    attached = 0
    for mod in list(m.modules()):
        ch = None
        if isinstance(mod, BasicBlock):
            ch = mod.conv2.out_channels
        elif isinstance(mod, Bottleneck):
            ch = mod.conv3.out_channels
        if ch and not hasattr(mod, "pet_adapter"):
            ad = DT1DAdapter(C=ch, M=1, dilations="1,2,4", scale_adaptive=True, axis="hw", no_pw=True, gate_init=0.01)
            mod.add_module("pet_adapter", ad)
            mod.register_forward_hook(lambda mm, inp, out: mm.pet_adapter(out))
            attached += 1
    assert attached > 0
    m.to(device).eval()
    x = torch.randn(1, 3, 64, 64, device=device)
    with torch.no_grad():
        y = m(x)
    assert y.shape == (1, 10), y.shape
    print(f"[OK] torchvision ResNet integration, attached={attached}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--skip-torchvision", action="store_true")
    args = ap.parse_args()

    device = resolve_device(args.device)
    print(f"[Preflight] device={device}")
    test_kernel_and_routing(device)
    test_forward_backward(device)
    test_small_feature_maps(device)
    test_checkpoint_load()
    if not args.skip_torchvision:
        test_torchvision_integration(device)
    print("[ALL OK] DT1D-Adapter static-gate preflight passed")


if __name__ == "__main__":
    main()
