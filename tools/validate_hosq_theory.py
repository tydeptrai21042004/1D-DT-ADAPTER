#!/usr/bin/env python3
"""Deterministic validation for HOSQ-DT1D.

Checks the orthogonal channel basis, zero-DC detail atoms, MLQ8 embedding,
non-expansive joint L1 projection, convolution-call count, forward/backward
finiteness, and the ResNet-18 parameter budget.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.dt1d_adapter import DT1DAdapter

OUT = ROOT / "outputs" / "hosq_validation"


def hosq(C: int = 32, rank4: int = 1, rank8: int = 2) -> DT1DAdapter:
    return DT1DAdapter(
        C=C,
        M=1,
        dilations=(1, 2, 4),
        scale_adaptive=True,
        axis="hw",
        alpha_group=32,
        no_pw=True,
        gate_init=0.01,
        padding_mode="reflect",
        minimal_quotient_realization=False,
        quotient_support_cap=8,
        hosq_realization=True,
        hosq_subgroup_size=8,
        hosq_rank4=rank4,
        hosq_rank8=rank8,
    )


def mlq(C: int = 32) -> DT1DAdapter:
    return DT1DAdapter(
        C=C,
        M=1,
        dilations=(1, 2, 4),
        scale_adaptive=True,
        axis="hw",
        alpha_group=32,
        no_pw=True,
        gate_init=0.01,
        padding_mode="reflect",
        minimal_quotient_realization=True,
        quotient_support_cap=8,
    )


def main() -> int:
    torch.manual_seed(20260802)
    results: dict[str, object] = {}

    m = hosq(32)
    basis = m.hosq_basis[0, :4, :3].double()
    zero_mean_error = float(basis.sum(0).abs().max())
    gram_error = float((basis.T @ basis - torch.eye(3, dtype=torch.float64)).abs().max())
    results["haar_zero_mean_max_error"] = zero_mean_error
    results["haar_orthonormal_max_error"] = gram_error

    source = mlq(64)
    with torch.no_grad():
        source.quotient_beta.normal_(0.0, 0.25)
        source.gate.fill_(0.17)
    target = hosq(64)
    target.initialize_hosq_from_mlq(source)
    x1 = torch.randn(2, 64, 19, 21, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)
    y1 = source(x1)
    y2 = target(x2)
    forward_error = float((y1 - y2).abs().max().detach())
    y1.square().mean().backward()
    y2.square().mean().backward()
    gradient_error = float((x1.grad - x2.grad).abs().max().detach())
    results["mlq8_embedding_forward_max_error"] = forward_error
    results["mlq8_embedding_input_gradient_max_error"] = gradient_error

    detail = hosq(32)
    with torch.no_grad():
        detail.quotient_beta.zero_()
        detail.hosq_detail4.normal_(0.0, 1.0)
        detail.hosq_detail8.normal_(0.0, 1.0)
    detail_kernels = detail._build_normalized_hosq_kernels(torch.device("cpu"), torch.float64)
    results["detail_zero_dc_max_error"] = float(detail_kernels.squeeze(2).sum(-1).abs().max().detach())

    stable = hosq(64)
    with torch.no_grad():
        stable.quotient_beta.normal_(0.0, 4.0)
        stable.hosq_detail4.normal_(0.0, 4.0)
        stable.hosq_detail8.normal_(0.0, 4.0)
    kernels = stable._build_normalized_hosq_kernels(torch.device("cpu"), torch.float64)
    joint_l1 = kernels.squeeze(2).abs().sum(-1).sum(0)
    results["maximum_joint_axis_l1_norm"] = float(joint_l1.max().detach())

    calls: list[tuple[str, int, int]] = []
    original = stable._conv_axis

    def counted(x, axis_name, w1d, dilation):
        calls.append((axis_name, int(w1d.shape[-1]), int(dilation)))
        return original(x, axis_name, w1d, dilation)

    stable._conv_axis = counted
    z = torch.randn(1, 64, 18, 18, requires_grad=True)
    stable(z).mean().backward()
    results["convolution_calls"] = calls
    results["finite_backward"] = all(
        p.grad is not None and bool(torch.isfinite(p.grad).all())
        for p in (stable.quotient_beta, stable.hosq_detail4, stable.hosq_detail8, stable.gate)
    )

    channels = [64, 64, 128, 128, 256, 256, 512, 512]
    adapter_parameters = sum(
        sum(p.numel() for p in hosq(c).parameters() if p.requires_grad) for c in channels
    )
    results["resnet18_adapter_parameters"] = adapter_parameters
    results["caltech101_total_trainable_parameters"] = 51_813 + adapter_parameters
    results["coarse_dimension_per_axis_group"] = 5
    results["hosq_dimension_per_axis_group"] = 8
    results["full_group8_mlq_dimension_per_coarse_group_axis"] = 20

    thresholds = {
        "haar_zero_mean": zero_mean_error < 1e-6,
        "haar_orthonormal": gram_error < 1e-6,
        "mlq_embedding_forward": forward_error < 3e-6,
        "mlq_embedding_gradient": gradient_error < 3e-6,
        "detail_zero_dc": results["detail_zero_dc_max_error"] < 1e-10,
        "joint_l1": results["maximum_joint_axis_l1_norm"] <= 1.0 + 1e-12,
        "two_axis_calls": calls == [("h", 17, 1), ("w", 17, 1)],
        "finite_backward": bool(results["finite_backward"]),
        "parameter_budget": adapter_parameters == 968,
    }
    results["checks"] = thresholds
    results["status"] = "ok" if all(thresholds.values()) else "failed"

    OUT.mkdir(parents=True, exist_ok=True)
    output = OUT / "hosq_validation.json"
    output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    return 0 if results["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
