#!/usr/bin/env python3
"""Deterministic validation of the MLQ/SCDQ DT1D mathematical claims."""
from __future__ import annotations

import json
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import torch

from models.dt1d_adapter import DT1DAdapter


def build(*, mlq: bool, cap: int = 8, no_pw: bool = True) -> DT1DAdapter:
    return DT1DAdapter(
        C=32, M=1, axis="hw", alpha_group=8, no_pw=no_pw,
        pw_ratio=8, pw_groups=4, use_bn=False, gate_init=0.2,
        padding_mode="replicate", dilations=(1, 2, 4),
        scale_adaptive=True, separate_axis_kernels=True,
        exact_cost_realization=not mlq,
        closed_form_dyadic_realization=not mlq,
        minimal_quotient_realization=mlq,
        quotient_support_cap=cap,
    )


def main() -> None:
    torch.manual_seed(20260801)
    a = DT1DAdapter.dyadic_quotient_matrix(dtype=torch.float64)
    singular = torch.linalg.svdvals(a)
    null = torch.tensor([0., 1., -1., -1., 1., 0.], dtype=torch.float64)

    legacy = build(mlq=False, no_pw=True)
    with torch.no_grad():
        legacy.alpha.normal_(0.0, 0.5)
        legacy.axis_scale_logits.normal_(0.0, 0.3)
        legacy.gate.fill_(0.19)
    mlq8 = build(mlq=True, cap=8, no_pw=True)
    mlq8.initialize_quotient_from_legacy(legacy)

    x1 = torch.randn(2, 32, 24, 24, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)
    y1, y2 = legacy(x1), mlq8(x2)
    forward_max = float((y1 - y2).abs().max().detach())
    y1.square().mean().backward()
    y2.square().mean().backward()
    grad_max = float((x1.grad - x2.grad).abs().max().detach())

    scdq = build(mlq=True, cap=4, no_pw=True)
    with torch.no_grad():
        scdq.quotient_beta.normal_(0.0, 2.0)
    beta = scdq._normalized_quotient_beta(torch.device("cpu"), torch.float64)
    joint_l1 = beta[..., 0].abs() + 2.0 * beta[..., 1:].abs().sum(dim=-1)
    max_joint_l1 = float(joint_l1.sum(dim=0).max().detach())

    result = {
        "quotient_rank": int(torch.linalg.matrix_rank(a)),
        "quotient_singular_values": [float(v) for v in singular],
        "legacy_null_residual_max": float((a @ null).abs().max()),
        "legacy_to_mlq8_forward_max_abs_error": forward_max,
        "legacy_to_mlq8_input_gradient_max_abs_error": grad_max,
        "scdq4_max_joint_axis_l1_norm": max_joint_l1,
        "legacy_core_parameters": sum(p.numel() for p in legacy.parameters()),
        "mlq8_core_parameters": sum(p.numel() for p in mlq8.parameters()),
        "scdq4_core_parameters": sum(p.numel() for p in scdq.parameters()),
        "legacy_axial_calls_per_adapter": 4,
        "mlq8_axial_calls_per_adapter": 2,
        "scdq4_axial_calls_per_adapter": 2,
        "legacy_effective_max_offset": 8,
        "scdq4_effective_max_offset": 4,
    }
    out = Path("outputs/mlq_validation")
    out.mkdir(parents=True, exist_ok=True)
    (out / "theory_validation.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
