"""HOSQ-Lite-C1-Orth: the revised DT1D proposal.

The module keeps the original DT1D Group-16 coarse spatial structure, replaces
redundant axis--scale branch coordinates by their five observable quotient
coefficients, and adds one zero-mean channel contrast per original group with
up to two zero-DC spectral detail coordinates.  It always executes one
17-tap depthwise convolution per enabled axis.

Only ``HOSQLiteC1Adapter`` is the revised proposal. ``DT1DAdapter`` remains the
submitted/original baseline and is implemented in :mod:`models.dt1d_adapter`.
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn

from .dt1d_adapter import DT1DAdapter


_DETAIL_COMPONENTS = {"both", "offset4", "offset8", "none"}
_DETAIL_BASES = {"orth", "raw"}


class HOSQLiteC1Adapter(DT1DAdapter):
    """Minimal identifiable and low-latency extension of original DT1D.

    Coarse kernel for axis ``a`` and original DT1D channel group ``g``::

        beta_0 delta_0 + sum_{r in {1,2,4,8}} beta_r(delta_-r + delta_r).

    Channel detail for channel ``c`` in group ``g``::

        h_g(c) [eta_4 q_4 + eta_8 q_8],

    where ``h_g`` is a weighted zero-mean unit-norm Helmert contrast.  In the
    final ``orth`` basis, ``q_4`` and ``q_8`` are orthonormal, symmetric,
    zero-DC atoms spanning the same space as ``psi_4`` and ``psi_8``.

    ``detail_components`` is an ablation switch, not a separate proposal.
    """

    quotient_offsets = (0, 1, 2, 4, 8)

    def __init__(
        self,
        C: int,
        *,
        axis: str = "hw",
        alpha_group: int = 16,
        residual_scale: float = 1.0,
        gate_init: float = 0.01,
        padding_mode: str = "replicate",
        contrast_split: int = 8,
        detail_basis: str = "orth",
        detail_components: str = "both",
        **legacy_kwargs,
    ) -> None:
        detail_basis = str(detail_basis).lower()
        detail_components = str(detail_components).lower()
        if detail_basis not in _DETAIL_BASES:
            raise ValueError(f"detail_basis must be one of {_DETAIL_BASES}, got {detail_basis!r}")
        if detail_components not in _DETAIL_COMPONENTS:
            raise ValueError(
                f"detail_components must be one of {_DETAIL_COMPONENTS}, got {detail_components!r}"
            )

        # Build the original DT1D shell to reuse its validated boundary and axial
        # convolution routines. Its branch coordinates are then removed so the
        # revised module has no unused trainable parameters.
        super().__init__(
            C=C,
            M=1,
            h=1,
            axis=axis,
            alpha_group=alpha_group,
            tie_sym=True,
            no_pw=True,
            residual_scale=residual_scale,
            gate_init=gate_init,
            padding_mode=padding_mode,
            dilations=(1, 2, 4),
            scale_adaptive=True,
            separate_axis_kernels=True,
            **legacy_kwargs,
        )
        del self._parameters["alpha"]
        self.register_parameter("alpha", None)
        del self._parameters["axis_scale_logits"]
        self.register_parameter("axis_scale_logits", None)

        self.variant = "hosq_lite_c1"
        self.contrast_split = max(1, int(contrast_split))
        self.detail_basis = detail_basis
        self.detail_components = detail_components

        self.quotient_beta = nn.Parameter(
            torch.zeros(self.num_axes, self.num_alpha_groups, len(self.quotient_offsets))
        )
        active = self._active_component_indices(detail_components)
        self.register_buffer(
            "detail_component_indices", torch.tensor(active, dtype=torch.long), persistent=True
        )
        self.detail_eta = nn.Parameter(
            torch.zeros(self.num_axes, self.num_alpha_groups, len(active))
        )

        contrast, valid = self._make_weighted_channel_contrast()
        self.register_buffer("channel_contrast", contrast, persistent=True)
        self.register_buffer("valid_contrast_group", valid, persistent=True)
        self.register_buffer(
            "spectral_atoms",
            self._spectral_atoms(detail_basis),
            persistent=True,
        )

        # Equal-route legacy initialization: alpha=(1,0), six routes, and the
        # normalized branch coefficient at +/-d is 1/2.
        with torch.no_grad():
            init_side = 1.0 / float(2 * self.num_axes * 3)
            self.quotient_beta[..., 1:4].fill_(init_side)

    @staticmethod
    def _active_component_indices(detail_components: str) -> Tuple[int, ...]:
        if detail_components == "both":
            return (0, 1)
        if detail_components == "offset4":
            return (0,)
        if detail_components == "offset8":
            return (1,)
        return tuple()

    def _make_weighted_channel_contrast(self) -> Tuple[torch.Tensor, torch.Tensor]:
        contrast = torch.zeros(self.C, dtype=torch.float32)
        valid = torch.zeros(self.num_alpha_groups, dtype=torch.float32)
        start = 0
        for group in range(self.num_alpha_groups):
            n = min(self.alpha_group, self.C - start)
            n1 = min(self.contrast_split, n)
            n2 = n - n1
            if n1 > 0 and n2 > 0:
                pos = math.sqrt(n2 / (n1 * (n1 + n2)))
                neg = -math.sqrt(n1 / (n2 * (n1 + n2)))
                contrast[start : start + n1] = pos
                contrast[start + n1 : start + n] = neg
                valid[group] = 1.0
            start += n
        return contrast, valid

    @staticmethod
    def _psi(offset: int) -> torch.Tensor:
        atom = torch.zeros(17, dtype=torch.float64)
        atom[8 - offset] = 1.0
        atom[8 + offset] = 1.0
        atom[8] = -2.0
        return atom

    @classmethod
    def _spectral_atoms(cls, basis: str) -> torch.Tensor:
        psi4 = cls._psi(4)
        psi8 = cls._psi(8)
        if basis == "raw":
            return torch.stack((psi4, psi8)).to(torch.float32)
        q4 = psi4 / torch.linalg.vector_norm(psi4)
        psi8_perp = psi8 - torch.dot(psi8, q4) * q4
        q8 = psi8_perp / torch.linalg.vector_norm(psi8_perp)
        return torch.stack((q4, q8)).to(torch.float32)

    def _group_index(self, device: torch.device) -> torch.Tensor:
        return (torch.arange(self.C, device=device) // self.alpha_group).clamp_max(
            self.num_alpha_groups - 1
        )

    def build_kernels(
        self,
        device: torch.device,
        dtype: torch.dtype,
        *,
        project: bool = True,
    ) -> torch.Tensor:
        """Return kernels with shape ``(num_axes, C, 1, 17)``."""
        group_idx = self._group_index(device)
        beta = self.quotient_beta.to(device=device, dtype=dtype)[:, group_idx, :]
        kernel = torch.zeros(self.num_axes, self.C, 17, device=device, dtype=dtype)
        kernel[..., 8] = beta[..., 0]
        for coefficient, offset in enumerate((1, 2, 4, 8), start=1):
            kernel[..., 8 - offset] = beta[..., coefficient]
            kernel[..., 8 + offset] = beta[..., coefficient]

        if self.detail_eta.shape[-1] > 0:
            valid = self.valid_contrast_group.to(device=device, dtype=dtype)
            eta = self.detail_eta.to(device=device, dtype=dtype)
            eta = eta * valid.view(1, self.num_alpha_groups, 1)
            eta_channel = eta[:, group_idx, :]
            contrast = self.channel_contrast.to(device=device, dtype=dtype)
            atoms = self.spectral_atoms.to(device=device, dtype=dtype)
            atoms = atoms[self.detail_component_indices.to(device=device)]
            kernel = kernel + torch.einsum(
                "acr,rk->ack",
                eta_channel * contrast.view(1, self.C, 1),
                atoms,
            )

        if project:
            # Joint two-axis L1 projection.  It guarantees a non-expansive axial
            # branch and is identity on every original-DT1D-representable kernel.
            joint_l1 = kernel.abs().sum(dim=-1).sum(dim=0)
            scale = torch.maximum(joint_l1, torch.ones_like(joint_l1))
            kernel = kernel / scale.view(1, self.C, 1)
        return kernel.unsqueeze(2)

    @torch.no_grad()
    def initialize_from_dt1d(self, legacy: DT1DAdapter) -> None:
        """Exactly embed an original M=1, dilation-(1,2,4) DT1D module.

        Exact forward equality requires both modules to use the same linear
        boundary extension. The revised paper configuration standardizes
        ``padding_mode='replicate'`` for this reason.
        """
        if legacy.M != 1 or tuple(legacy.dilations) != (1, 2, 4):
            raise ValueError("legacy DT1D must use M=1 and dilations=(1,2,4)")
        if legacy.num_axes != self.num_axes or legacy.num_alpha_groups != self.num_alpha_groups:
            raise ValueError("legacy and HOSQ-Lite modules must have matching axes/groups")
        if not legacy.scale_adaptive or legacy.axis_scale_logits is None:
            raise ValueError("legacy DT1D must use static axis--scale routing")

        weights = legacy._compute_axis_scale_weights(legacy.alpha.device, legacy.alpha.dtype)
        betas = []
        for axis_index in range(self.num_axes):
            alpha_axis = axis_index if legacy.separate_axis_kernels else 0
            alpha = legacy.alpha[alpha_axis]  # (3,G,2)
            denom = (2.0 * alpha[..., 0].abs() + 4.0 * alpha[..., 1].abs()).clamp_min(1e-6)
            u = alpha[..., 0] / denom
            v = alpha[..., 1] / denom
            p = weights[axis_index].unsqueeze(1) * u
            q = weights[axis_index].unsqueeze(1) * v
            beta = torch.stack(
                (
                    2.0 * (q[0] + q[1] + q[2]),
                    p[0],
                    q[0] + p[1],
                    q[1] + p[2],
                    q[2],
                ),
                dim=1,
            )
            betas.append(beta)
        self.quotient_beta.copy_(torch.stack(betas).to(self.quotient_beta))
        if self.detail_eta.numel():
            self.detail_eta.zero_()
        self.gate.copy_(legacy.gate.to(self.gate))

    def parameter_count_breakdown(self) -> Dict[str, int]:
        coarse = self.quotient_beta.numel()
        detail = self.detail_eta.numel()
        gate = self.gate.numel()
        return {
            "coarse_quotient": int(coarse),
            "orthogonal_detail": int(detail),
            "gate": int(gate),
            "pointwise": 0,
            "total": int(coarse + detail + gate),
        }

    def extra_repr(self) -> str:
        return (
            f"C={self.C}, axis={self.axis}, alpha_group={self.alpha_group}, "
            f"detail_basis={self.detail_basis}, components={self.detail_components}, "
            f"contrast_split={self.contrast_split}, padding={self.padding_mode}, "
            f"gate={float(self.gate.detach().cpu()):.4g}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"HOSQLiteC1Adapter expects BCHW input, got {tuple(x.shape)}")
        if x.shape[1] != self.C:
            raise ValueError(f"Channel mismatch: adapter C={self.C}, input C={x.shape[1]}")
        kernels = self.build_kernels(x.device, x.dtype, project=True)
        response = torch.zeros_like(x)
        for axis_index, axis_name in enumerate(self.axis_names):
            response = response + self._conv_axis(x, axis_name, kernels[axis_index], dilation=1)
        return x + self.residual_scale * self.gate * response


__all__ = ["HOSQLiteC1Adapter"]
