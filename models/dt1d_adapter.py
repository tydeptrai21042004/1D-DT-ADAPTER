"""Canonical DT1D-Adapter implementation.

DT1D-Adapter is a lightweight spatial PEFT module based on finite weighted
h-Hartley-cosine axial convolution. The public class and source module are
``DT1DAdapter`` and ``models.dt1d_adapter``.

The release keeps only narrowly scoped compatibility aliases for pre-v0.7.0
checkpoints and scripts. New experiments, configuration files, logs, and
manuscript text must use the canonical name ``DT1D-Adapter``.
"""

from __future__ import annotations

import math
from math import gcd
from typing import Dict, Iterable, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


DilationLike = Optional[Union[str, int, Sequence[int]]]


def _parse_dilations(dilations: DilationLike, fallback: int) -> Tuple[int, ...]:
    """Parse dilation specification into a tuple of positive unique integers."""
    if dilations is None:
        values = [int(fallback)]
    elif isinstance(dilations, int):
        values = [int(dilations)]
    elif isinstance(dilations, str):
        text = dilations.strip()
        if not text:
            values = [int(fallback)]
        else:
            # Accept both comma-separated and whitespace-separated forms.
            text = text.replace(";", ",").replace(" ", ",")
            values = [int(v) for v in text.split(",") if v.strip()]
    else:
        values = [int(v) for v in dilations]

    clean = []
    for v in values:
        if v <= 0:
            raise ValueError(f"All dilations must be positive, got {values!r}")
        if v not in clean:
            clean.append(v)
    if not clean:
        clean = [int(fallback)]
    return tuple(clean)


class DT1DAdapter(nn.Module):
    def __init__(
        self,
        C: int,
        M: int = 1,
        h: int = 1,
        axis: str = "hw",
        alpha_group: int = 16,
        tie_sym: bool = True,
        no_pw: bool = False,
        pw_ratio: int = 32,
        pw_groups: int = 4,
        use_bn: bool = False,
        residual_scale: float = 1.0,
        gate_init: float = 0.0,
        padding_mode: str = "reflect",
        # New Step-2 arguments.
        dilations: DilationLike = None,
        scale_adaptive: bool = False,
        separate_axis_kernels: bool = True,
        gate_temperature: float = 1.0,
        exact_cost_realization: bool = True,
        closed_form_dyadic_realization: bool = True,
        minimal_quotient_realization: bool = False,
        quotient_support_cap: int = 8,
        # Hierarchically Orthogonal Spectral Quotient (HOSQ). This retains
        # the MLQ8 coarse quotient while adding a very small orthogonal
        # channel-detail space at offsets 4 and 8.
        hosq_realization: bool = False,
        hosq_subgroup_size: int = 8,
        hosq_rank4: int = 1,
        hosq_rank8: int = 2,
        input_adaptive_gate: bool = False,  # deprecated/ignored; kept for CLI compatibility
        gate_reduction: int = 4,            # deprecated/ignored; kept for CLI compatibility
        **legacy,
    ):
        super().__init__()

        if axis not in ("h", "w", "hw"):
            raise ValueError(f"axis must be one of 'h', 'w', 'hw', got {axis!r}")
        if padding_mode not in ("reflect", "replicate", "zeros", "constant"):
            raise ValueError(
                "padding_mode must be 'reflect', 'replicate', 'zeros', or 'constant', "
                f"got {padding_mode!r}"
            )

        # Backward-compatible translation from the pre-v0.7.0 API.
        if "per_channel" in legacy:
            per_channel = bool(legacy.pop("per_channel"))
            alpha_group = 1 if per_channel else alpha_group
        if "use_pw" in legacy:
            use_pw_legacy = bool(legacy.pop("use_pw"))
            no_pw = not use_pw_legacy
        if "hcc_dilations" in legacy and dilations is None:
            dilations = legacy.pop("hcc_dilations")
        # Deprecated input-adaptive routing options are accepted for backward
        # CLI compatibility but intentionally ignored in this static-gate version.
        legacy.pop("hcc_input_adaptive_gate", None)
        legacy.pop("hcc_gate_reduction", None)
        # Keep unknown legacy kwargs harmless, because older main.py may pass unused flags.

        self.C = int(C)
        self.M = int(M)
        self.h = int(h)
        self.axis = axis
        self.axis_names = tuple(a for a in ("h", "w") if a in axis)
        self.alpha_group = max(1, int(alpha_group))
        self.tie_sym = bool(tie_sym)
        self.no_pw = bool(no_pw)
        self.use_bn = bool(use_bn)
        self.residual_scale = float(residual_scale)
        self.padding_mode = "constant" if padding_mode == "zeros" else padding_mode
        self.dilations = _parse_dilations(dilations, fallback=self.h)
        # If multiple dilations are supplied, enable scale-adaptive gating automatically.
        self.scale_adaptive = bool(scale_adaptive or len(self.dilations) > 1)
        # Best-performance setting for Step 2: separate group-shared kernels per axis and scale.
        # For backward compatibility, single-dilation non-adaptive mode shares the old kernel.
        self.separate_axis_kernels = bool(separate_axis_kernels and self.scale_adaptive)
        self.gate_temperature = float(gate_temperature)
        # Exact mathematical realization selector. This changes neither the learned
        # parameters nor the represented DT1D operator. It only selects the globally
        # minimum-cost exact partition of static axis--scale branches.
        self.exact_cost_realization = bool(exact_cost_realization)
        # Exact closed-form evaluator for the paper configuration M=1 and
        # dilations=(1,2,4). It changes no parameters or represented operator.
        self.closed_form_dyadic_realization = bool(closed_form_dyadic_realization)
        # Minimal Laurent quotient (MLQ): for M=1 and dyadic dilations (1,2,4),
        # the three shifted-symmetric scale branches span a five-dimensional
        # symmetric Laurent kernel on offsets {0, +/-1, +/-2, +/-4, +/-8}.
        # This coordinate system removes the exact one-dimensional scale nullspace
        # while preserving axial convolution and group-shared shifted symmetry.
        self.minimal_quotient_realization = bool(minimal_quotient_realization)
        self.quotient_support_cap = int(quotient_support_cap)
        self.hosq_realization = bool(hosq_realization)
        self.hosq_subgroup_size = max(1, int(hosq_subgroup_size))
        self.hosq_rank4_requested = max(0, int(hosq_rank4))
        self.hosq_rank8_requested = max(0, int(hosq_rank8))
        # The GAP-MLP input-adaptive router has been removed. Keep these attributes
        # only so older scripts/checkpoints that inspect them do not fail.
        self.input_adaptive_gate = False
        self.gate_reduction = max(1, int(gate_reduction))

        if self.C <= 0:
            raise ValueError(f"C must be positive, got {self.C}")
        if self.M < 0:
            raise ValueError(f"M must be non-negative, got {self.M}")
        if self.h <= 0:
            raise ValueError(f"h/dilation must be positive, got {self.h}")
        if self.gate_temperature <= 0:
            raise ValueError(f"gate_temperature must be positive, got {self.gate_temperature}")

        # Number of coefficient-sharing groups. Use ceil, not floor, so remainder channels are handled.
        self.num_alpha_groups = math.ceil(self.C / self.alpha_group)
        ncoef = self.M + 1  # center + M symmetric side taps

        # alpha shape:
        #   old/single mode:        (1, 1, G, M+1)
        #   scale-adaptive shared:  (1, S, G, M+1)
        #   scale-adaptive full:    (A, S, G, M+1)
        self.num_axes = len(self.axis_names)
        self.num_scales = len(self.dilations)
        self.num_alpha_axes = self.num_axes if self.separate_axis_kernels else 1

        if self.hosq_realization and self.minimal_quotient_realization:
            raise ValueError(
                "hosq_realization and minimal_quotient_realization are separate modes; "
                "enable only hosq_realization for HOSQ."
            )

        if self.hosq_realization:
            if self.M != 1 or tuple(self.dilations) != (1, 2, 4):
                raise ValueError("hosq_realization requires M=1 and dilations=(1,2,4)")
            if not self.scale_adaptive:
                raise ValueError("hosq_realization requires scale_adaptive=True")
            if self.quotient_support_cap != 8:
                raise ValueError("HOSQ uses the full coarse MLQ8 support; set quotient_support_cap=8")
            if self.hosq_subgroup_size > self.alpha_group:
                raise ValueError("hosq_subgroup_size cannot exceed alpha_group")
            if self.alpha_group % self.hosq_subgroup_size != 0:
                raise ValueError("alpha_group must be divisible by hosq_subgroup_size")

            self.quotient_offsets = (0, 1, 2, 4, 8)
            max_subgroups = max(1, math.ceil(min(self.C, self.alpha_group) / self.hosq_subgroup_size))
            max_contrasts = max(0, max_subgroups - 1)
            self.hosq_rank4 = min(self.hosq_rank4_requested, max_contrasts)
            self.hosq_rank8 = min(self.hosq_rank8_requested, max_contrasts)

            self.register_parameter("alpha", None)
            self.quotient_beta = nn.Parameter(
                torch.zeros(self.num_axes, self.num_alpha_groups, len(self.quotient_offsets))
            )

            basis, channel_groups, channel_subgroups, subgroup_counts = self._make_hosq_index_buffers()
            detail4_groups, detail4_modes = self._make_hosq_detail_coordinate_map(
                subgroup_counts, self.hosq_rank4
            )
            detail8_groups, detail8_modes = self._make_hosq_detail_coordinate_map(
                subgroup_counts, self.hosq_rank8
            )
            # Flat coordinates allocate only mathematically valid contrasts. This
            # avoids unused parameters in remainder channel groups.
            self.hosq_detail4 = nn.Parameter(
                torch.zeros(self.num_axes, int(detail4_groups.numel()))
            )
            self.hosq_detail8 = nn.Parameter(
                torch.zeros(self.num_axes, int(detail8_groups.numel()))
            )
            with torch.no_grad():
                init_side = 1.0 / float(2 * self.num_axes * self.num_scales)
                self.quotient_beta[..., 1:4].fill_(init_side)

            self.register_buffer("hosq_basis", basis, persistent=True)
            self.register_buffer("hosq_channel_group", channel_groups, persistent=True)
            self.register_buffer("hosq_channel_subgroup", channel_subgroups, persistent=True)
            self.register_buffer("hosq_subgroup_counts", subgroup_counts, persistent=True)
            self.register_buffer("hosq_detail4_group", detail4_groups, persistent=True)
            self.register_buffer("hosq_detail4_mode", detail4_modes, persistent=True)
            self.register_buffer("hosq_detail8_group", detail8_groups, persistent=True)
            self.register_buffer("hosq_detail8_mode", detail8_modes, persistent=True)
            self.register_parameter("quotient_axis_logits", None)
            self.register_parameter("axis_scale_logits", None)
        elif self.minimal_quotient_realization:
            if self.M != 1 or tuple(self.dilations) != (1, 2, 4):
                raise ValueError(
                    "minimal_quotient_realization currently requires M=1 and dilations=(1,2,4)"
                )
            if not self.scale_adaptive:
                raise ValueError("minimal_quotient_realization requires scale_adaptive=True")
            if self.quotient_support_cap not in (4, 8):
                raise ValueError("quotient_support_cap must be 4 or 8")
            # Minimal quotient coordinates per axis/group, ordered by offsets
            # (0,1,2,4) or (0,1,2,4,8). Negative offsets are tied exactly.
            self.quotient_offsets = (0, 1, 2, 4, 8) if self.quotient_support_cap == 8 else (0, 1, 2, 4)
            self.register_parameter("alpha", None)
            self.quotient_beta = nn.Parameter(
                torch.zeros(self.num_axes, self.num_alpha_groups, len(self.quotient_offsets))
            )
            with torch.no_grad():
                # Exact legacy equal-route initialization. For alpha=(1,0), the
                # normalized branch coefficient is u=1/2 and each of A*S routes
                # has weight 1/(A*S).
                init_side = 1.0 / float(2 * self.num_axes * self.num_scales)
                self.quotient_beta[..., 1:4].fill_(init_side)
            self.register_parameter("hosq_detail4", None)
            self.register_parameter("hosq_detail8", None)
            self.register_parameter("quotient_axis_logits", None)
            self.register_parameter("axis_scale_logits", None)
        else:
            self.alpha = nn.Parameter(torch.zeros(self.num_alpha_axes, self.num_scales, self.num_alpha_groups, ncoef))
            self.register_parameter("hosq_detail4", None)
            self.register_parameter("hosq_detail8", None)
            with torch.no_grad():
                self.alpha[..., 0].fill_(1.0)  # identity-like axial filter before residual gate
            self.register_parameter("quotient_beta", None)
            self.register_parameter("quotient_axis_logits", None)
            # Static/global axis--scale logits. This is the old DT1D fusion mechanism:
            # one small set of learnable logits is shared by all input samples.
            if self.scale_adaptive:
                self.axis_scale_logits = nn.Parameter(torch.zeros(self.num_axes, self.num_scales))
            else:
                self.register_parameter("axis_scale_logits", None)
        self.axis_scale_router = None

        # Optional grouped pointwise channel mixing.
        if not self.no_pw:
            hidden = max(1, self.C // max(1, int(pw_ratio)))
            groups = max(1, int(pw_groups))
            # Groups must divide input and hidden channels for both 1x1 convs.
            groups = min(groups, self.C, hidden)
            groups = gcd(groups, self.C)
            groups = gcd(groups, hidden) or 1
            self.pw_groups = groups
            self.pw = nn.Sequential(
                nn.Conv2d(self.C, hidden, kernel_size=1, groups=groups, bias=False),
                nn.BatchNorm2d(hidden) if self.use_bn else nn.Identity(),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, self.C, kernel_size=1, groups=groups, bias=False),
                nn.BatchNorm2d(self.C) if self.use_bn else nn.Identity(),
            )
        else:
            self.pw_groups = 1
            self.pw = nn.Identity()

        # Scalar residual gate. gate_init=0.0 makes the whole adapter initially identity.
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    def extra_repr(self) -> str:
        return (
            f"C={self.C}, M={self.M}, dilations={self.dilations}, axis={self.axis}, "
            f"scale_adaptive={self.scale_adaptive}, static_axis_scale_gate=True, "
            f"exact_cost_realization={self.exact_cost_realization}, "
            f"closed_form_dyadic_realization={self.closed_form_dyadic_realization}, "
            f"minimal_quotient_realization={self.minimal_quotient_realization}, "
            f"hosq_realization={self.hosq_realization}, "
            f"quotient_support_cap={self.quotient_support_cap}, "
            f"hosq_subgroup_size={self.hosq_subgroup_size}, "
            f"hosq_ranks=({getattr(self, 'hosq_rank4', 0)},{getattr(self, 'hosq_rank8', 0)}), "
            f"separate_axis_kernels={self.separate_axis_kernels}, "
            f"alpha_group={self.alpha_group}, G={self.num_alpha_groups}, "
            f"no_pw={self.no_pw}, gate={float(self.gate.detach().cpu()):.4g}"
        )

    def parameter_count_breakdown(self) -> Dict[str, int]:
        if self.hosq_realization:
            axial = (
                self.quotient_beta.numel()
                + self.hosq_detail4.numel()
                + self.hosq_detail8.numel()
                + self.gate.numel()
            )
            axis_scale = 0
        elif self.minimal_quotient_realization:
            axial = self.quotient_beta.numel() + self.gate.numel()
            axis_scale = 0
        else:
            axial = self.alpha.numel() + self.gate.numel()
            axis_scale = 0 if self.axis_scale_logits is None else self.axis_scale_logits.numel()
        pw = sum(p.numel() for p in self.pw.parameters())
        return {
            "axial_alpha_and_gate": int(axial),
            "axis_scale_logits": int(axis_scale),
            "pointwise": int(pw),
            "total": int(axial + axis_scale + pw),
        }

    def axis_scale_weights(self, x: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        """Return detached static softmax weights with shape (num_axes, num_scales).

        The optional ``x`` argument is ignored and kept only for compatibility with
        earlier input-adaptive experiments.
        """
        if self.axis_scale_logits is None:
            return None
        logits = self.axis_scale_logits.detach() / self.gate_temperature
        return torch.softmax(logits.reshape(-1), dim=0).reshape(self.num_axes, self.num_scales)

    def _compute_axis_scale_weights(
        self,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Compute static routing weights with shape (num_axes, num_scales)."""
        if self.axis_scale_logits is None:
            raise RuntimeError("Axis--scale weights are only defined in scale_adaptive mode.")
        logits = self.axis_scale_logits.to(device=device, dtype=dtype) / self.gate_temperature
        weights = torch.softmax(logits.reshape(-1), dim=0)
        return weights.reshape(self.num_axes, self.num_scales)


    @staticmethod
    def _orthogonal_subgroup_basis(n: int, device=None, dtype=None) -> torch.Tensor:
        """Return a zero-mean orthonormal contrast basis on ``n`` subgroups.

        For four subgroups this is the fixed hierarchical Haar basis used in
        HOSQ. For a remainder group with two or three subgroups, a canonical
        Helmert basis is used so the zero-mean/orthogonality theorem remains
        valid instead of silently introducing unused coordinates.
        """
        dtype = dtype or torch.float32
        if n <= 1:
            return torch.zeros(n, 0, device=device, dtype=dtype)
        if n == 4:
            return torch.tensor(
                [
                    [0.5, 1.0 / math.sqrt(2.0), 0.0],
                    [0.5, -1.0 / math.sqrt(2.0), 0.0],
                    [-0.5, 0.0, 1.0 / math.sqrt(2.0)],
                    [-0.5, 0.0, -1.0 / math.sqrt(2.0)],
                ],
                device=device,
                dtype=dtype,
            )
        # Helmert contrasts: column k contrasts the first k entries with k+1.
        basis = torch.zeros(n, n - 1, device=device, dtype=dtype)
        for k in range(1, n):
            denom = math.sqrt(float(k * (k + 1)))
            basis[:k, k - 1] = 1.0 / denom
            basis[k, k - 1] = -float(k) / denom
        return basis

    def _make_hosq_index_buffers(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create padded group bases and channel-to-subgroup index maps."""
        max_subgroups = max(1, math.ceil(self.alpha_group / self.hosq_subgroup_size))
        # Store the complete orthogonal basis as a non-trainable buffer even when
        # the final model activates only a low-rank prefix of it.
        max_rank = max(0, max_subgroups - 1)
        basis = torch.zeros(self.num_alpha_groups, max_subgroups, max_rank)
        channel_group = torch.empty(self.C, dtype=torch.long)
        channel_subgroup = torch.empty(self.C, dtype=torch.long)
        subgroup_counts = torch.empty(self.num_alpha_groups, dtype=torch.long)

        start = 0
        for g in range(self.num_alpha_groups):
            group_channels = min(self.alpha_group, self.C - start)
            n_subgroups = max(1, math.ceil(group_channels / self.hosq_subgroup_size))
            subgroup_counts[g] = n_subgroups
            local_basis = self._orthogonal_subgroup_basis(n_subgroups)
            active_rank = min(max_rank, local_basis.shape[1])
            if active_rank:
                basis[g, :n_subgroups, :active_rank] = local_basis[:, :active_rank]
            for local_c in range(group_channels):
                channel_group[start + local_c] = g
                channel_subgroup[start + local_c] = min(
                    local_c // self.hosq_subgroup_size, n_subgroups - 1
                )
            start += group_channels
        return basis, channel_group, channel_subgroup, subgroup_counts

    @staticmethod
    def _make_hosq_detail_coordinate_map(
        subgroup_counts: torch.Tensor, requested_rank: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return flat valid ``(group, contrast-mode)`` coordinates."""
        groups = []
        modes = []
        for g, count in enumerate(subgroup_counts.tolist()):
            active = min(int(requested_rank), max(0, int(count) - 1))
            for mode in range(active):
                groups.append(g)
                modes.append(mode)
        return torch.tensor(groups, dtype=torch.long), torch.tensor(modes, dtype=torch.long)

    def _hosq_channel_detail(
        self,
        theta: torch.Tensor,
        coordinate_groups: torch.Tensor,
        coordinate_modes: torch.Tensor,
        group_idx: torch.Tensor,
        subgroup_idx: torch.Tensor,
        basis: torch.Tensor,
    ) -> torch.Tensor:
        """Expand flat valid detail coordinates to per-channel coefficients."""
        if theta.shape[1] == 0:
            return torch.zeros(self.num_axes, self.C, device=theta.device, dtype=theta.dtype)
        coord_groups = coordinate_groups.to(device=theta.device)
        coord_modes = coordinate_modes.to(device=theta.device)
        selected_groups = basis[coord_groups]  # T,S,R
        gather_index = coord_modes.view(-1, 1, 1).expand(-1, selected_groups.shape[1], 1)
        selected_basis = selected_groups.gather(2, gather_index).squeeze(2)  # T,S
        contributions = theta.unsqueeze(-1) * selected_basis.unsqueeze(0)  # A,T,S
        per_group = torch.zeros(
            self.num_axes, self.num_alpha_groups, basis.shape[1],
            device=theta.device, dtype=theta.dtype,
        )
        per_group.index_add_(1, coord_groups, contributions)
        return per_group[:, group_idx, subgroup_idx]

    def _build_normalized_hosq_kernels(
        self, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Build jointly normalized HOSQ kernels with shape ``(A,C,1,17)``.

        The coarse term is MLQ8 at Group-32 resolution. Fine terms use
        orthogonal channel contrasts and the zero-DC atoms

            psi_r = delta_-r + delta_r - 2 delta_0, r in {4,8}.

        Joint per-channel normalization across axes preserves the non-expansive
        operator bound without changing the convolution implementation.
        """
        group_idx = self.hosq_channel_group.to(device=device)
        subgroup_idx = self.hosq_channel_subgroup.to(device=device)
        beta = self.quotient_beta.to(device=device, dtype=dtype)[:, group_idx, :]  # A,C,5
        kernel = torch.zeros(self.num_axes, self.C, 17, device=device, dtype=dtype)
        center = 8
        kernel[..., center] = beta[..., 0]
        for j, offset in enumerate((1, 2, 4, 8), start=1):
            kernel[..., center - offset] = beta[..., j]
            kernel[..., center + offset] = beta[..., j]

        basis = self.hosq_basis.to(device=device, dtype=dtype)
        if self.hosq_detail4.shape[1]:
            d4 = self._hosq_channel_detail(
                self.hosq_detail4.to(device=device, dtype=dtype),
                self.hosq_detail4_group, self.hosq_detail4_mode,
                group_idx, subgroup_idx, basis,
            )
            kernel[..., center - 4] += d4
            kernel[..., center + 4] += d4
            kernel[..., center] -= 2.0 * d4
        if self.hosq_detail8.shape[1]:
            d8 = self._hosq_channel_detail(
                self.hosq_detail8.to(device=device, dtype=dtype),
                self.hosq_detail8_group, self.hosq_detail8_mode,
                group_idx, subgroup_idx, basis,
            )
            kernel[..., center - 8] += d8
            kernel[..., center + 8] += d8
            kernel[..., center] -= 2.0 * d8

        joint_l1 = kernel.abs().sum(dim=-1).sum(dim=0)  # C
        scale = torch.maximum(joint_l1, torch.ones_like(joint_l1)).view(1, self.C, 1)
        return (kernel / scale).unsqueeze(2)

    @torch.no_grad()
    def initialize_hosq_from_mlq(self, mlq: "DT1DAdapter") -> None:
        """Initialize HOSQ from a compatible MLQ8 model with zero fine details."""
        if not self.hosq_realization:
            raise RuntimeError("target module is not in HOSQ mode")
        if not mlq.minimal_quotient_realization or mlq.quotient_support_cap != 8:
            raise ValueError("source module must be MLQ8")
        if mlq.num_axes != self.num_axes or mlq.num_alpha_groups != self.num_alpha_groups:
            raise ValueError("source and target must use matching axes and coarse groups")
        self.quotient_beta.copy_(mlq.quotient_beta.to(self.quotient_beta))
        self.hosq_detail4.zero_()
        self.hosq_detail8.zero_()
        self.gate.copy_(mlq.gate.to(self.gate))
        if not self.no_pw and not mlq.no_pw:
            self.pw.load_state_dict(mlq.pw.state_dict())

    @staticmethod
    def dyadic_quotient_matrix(device=None, dtype=None) -> torch.Tensor:
        """Return the rank-five map from legacy dyadic branch coordinates to MLQ taps.

        Input coordinates are (p1,q1,p2,q2,p4,q4); output coordinates are
        (beta0,beta1,beta2,beta4,beta8). The null vector
        (0,1,-1,-1,1,0) is an exact scale-cancellation direction.
        """
        return torch.tensor(
            [[0, 2, 0, 2, 0, 2],
             [1, 0, 0, 0, 0, 0],
             [0, 1, 1, 0, 0, 0],
             [0, 0, 0, 1, 1, 0],
             [0, 0, 0, 0, 0, 1]],
            device=device, dtype=dtype or torch.float32,
        )

    def _normalized_quotient_beta(self, device, dtype) -> torch.Tensor:
        """Jointly project all axis kernels into the non-expansive L1 ball.

        The legacy routed operator already lies in this ball because its positive
        routing weights sum to one and every branch has L1 norm at most one. Hence
        this projection is identity on every legacy-representable kernel and only
        prevents newly learned quotient coordinates from becoming expansive.
        """
        beta = self.quotient_beta.to(device=device, dtype=dtype)  # (A,G,5)
        per_axis = beta[..., 0].abs() + 2.0 * beta[..., 1:].abs().sum(dim=-1)
        total = per_axis.sum(dim=0)  # (G,)
        scale = torch.maximum(total, torch.ones_like(total)).view(1, -1, 1)
        return beta / scale

    def _build_minimal_quotient_kernel_1d(
        self, normalized_beta: torch.Tensor, axis_idx: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Build the MLQ kernel on offsets {0,+/-1,+/-2,+/-4,+/-8}."""
        beta = normalized_beta[axis_idx]
        radius = self.quotient_support_cap
        wg = torch.zeros(self.num_alpha_groups, 2 * radius + 1, device=device, dtype=dtype)
        center = radius
        wg[:, center] = beta[:, 0]
        for j, off in enumerate(self.quotient_offsets[1:], start=1):
            wg[:, center - off] = beta[:, j]
            wg[:, center + off] = beta[:, j]
        return self._expand_group_kernel_to_channels(wg)

    @torch.no_grad()
    def initialize_quotient_from_legacy(self, legacy: "DT1DAdapter") -> None:
        """Project a legacy M=1,d=(1,2,4) module into exact MLQ coordinates.

        Exactness holds whenever all scale branches use the same linear boundary
        extension. The projection is groupwise and preserves the residual gate.
        """
        if not self.minimal_quotient_realization:
            raise RuntimeError("target module is not in minimal quotient mode")
        if legacy.M != 1 or tuple(legacy.dilations) != (1, 2, 4):
            raise ValueError("legacy module must use M=1 and dilations=(1,2,4)")
        if legacy.num_axes != self.num_axes or legacy.num_alpha_groups != self.num_alpha_groups:
            raise ValueError("legacy and quotient modules must have matching axes/groups")
        weights = legacy._compute_axis_scale_weights(legacy.alpha.device, legacy.alpha.dtype)
        betas = []
        for ai in range(self.num_axes):
            u, v = legacy._m1_dyadic_uv(ai, legacy.alpha.device, legacy.alpha.dtype)
            p = weights[ai].unsqueeze(1) * u
            q = weights[ai].unsqueeze(1) * v
            beta = torch.stack(
                (2.0 * (q[0] + q[1] + q[2]), p[0], q[0] + p[1], q[1] + p[2], q[2]),
                dim=1,
            )
            betas.append(beta)
        stacked = torch.stack(betas)
        if self.quotient_support_cap == 4:
            stacked = stacked[..., :4]
        self.quotient_beta.copy_(stacked.to(self.quotient_beta))
        self.gate.copy_(legacy.gate.to(self.gate))
        if not self.no_pw and not legacy.no_pw:
            self.pw.load_state_dict(legacy.pw.state_dict())

    def _expand_group_kernel_to_channels(self, wg: torch.Tensor) -> torch.Tensor:
        """Expand group-shared kernels from (G, K) to depthwise kernels (C, 1, K)."""
        chunks = []
        remaining = self.C
        for g in range(self.num_alpha_groups):
            rep = min(self.alpha_group, remaining)
            chunks.append(wg[g].unsqueeze(0).repeat(rep, 1))
            remaining -= rep
        w = torch.cat(chunks, dim=0)
        if w.shape[0] != self.C:
            raise RuntimeError(f"Internal error: built {w.shape[0]} channel kernels for C={self.C}")
        return w.unsqueeze(1)  # (C, 1, K)

    def _build_even_kernel_1d(
        self,
        axis_idx: int,
        scale_idx: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build normalized symmetric kernels with shape (C, 1, 2M+1)."""
        K = 2 * self.M + 1
        center = self.M

        alpha_axis_idx = axis_idx if self.separate_axis_kernels else 0
        alpha = self.alpha[alpha_axis_idx, scale_idx].to(device=device, dtype=dtype)  # (G, M+1)

        wg = torch.zeros(self.num_alpha_groups, K, device=device, dtype=dtype)
        wg[:, center] = alpha[:, 0]

        for m in range(1, self.M + 1):
            val = alpha[:, m]
            wg[:, center - m] = val
            # tie_sym is kept for legacy compatibility. In the current even-kernel formulation
            # both sides use the same coefficient.
            wg[:, center + m] = val if self.tie_sym else val

        # L1 normalization keeps the filter response numerically stable.
        denom = wg.abs().sum(dim=1, keepdim=True).clamp_min(1e-6)
        wg = wg / denom

        return self._expand_group_kernel_to_channels(wg)

    def _build_weighted_dt1d_kernel_1d(
        self,
        axis_idx: int,
        scale_idx: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build finite weighted h-Hartley-cosine axial kernels.

        The learnable sequence f[-M],...,f[M] is symmetric and group-shared:
            f[0]=alpha_0, f[+m]=f[-m]=alpha_m.

        For a 1D signal g[n], the finite weighted HCC branch is
            y[n] = 1/2 sum_m f[m] (
                       g[n-m-1] + g[n-m+1] + g[n+m+1] + g[n+m-1]).

        The four shifted terms can be aggregated into an ordinary depthwise
        1D convolution kernel supported on offsets [-(M+1), ..., M+1], hence
        the effective kernel length is 2M+3.
        """
        K_eff = 2 * self.M + 3
        center = self.M + 1

        alpha_axis_idx = axis_idx if self.separate_axis_kernels else 0
        alpha = self.alpha[alpha_axis_idx, scale_idx].to(device=device, dtype=dtype)  # (G, M+1)

        wg = torch.zeros(self.num_alpha_groups, K_eff, device=device, dtype=dtype)
        for m in range(-self.M, self.M + 1):
            val = alpha[:, abs(m)]
            for r in (-(m + 1), -(m - 1), (m + 1), (m - 1)):
                wg[:, center + r] += 0.5 * val

        # Normalize the implemented finite kernel to keep the neural branch stable.
        # This preserves the weighted DT1D shift pattern while preventing uncontrolled
        # amplification on finite feature maps.
        denom = wg.abs().sum(dim=1, keepdim=True).clamp_min(1e-6)
        wg = wg / denom
        return self._expand_group_kernel_to_channels(wg)



    # Pre-v0.7.0 private-name compatibility.
    def _build_weighted_hcc_kernel_1d(self, *args, **kwargs):
        return self._build_weighted_dt1d_kernel_1d(*args, **kwargs)

    def _build_m1_weighted_group_kernel_from_uv(
        self, u: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """Return the normalized M=1 HCC kernel [v,u,2v,u,v] in group space."""
        return torch.stack((v, u, 2.0 * v, u, v), dim=1)

    def _m1_dyadic_uv(
        self, axis_idx: int, device: torch.device, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Closed-form normalized M=1 coefficients for all scales.

        For alpha=(a,b), the shifted HCC kernel is [b,a,2b,a,b] and
        its L1 denominator is 2|a|+4|b|. The returned tensors are
        u=a/D and v=b/D with shape (S,G).
        """
        alpha_axis_idx = axis_idx if self.separate_axis_kernels else 0
        alpha = self.alpha[alpha_axis_idx].to(device=device, dtype=dtype)
        a = alpha[..., 0]
        b = alpha[..., 1]
        denom = (2.0 * a.abs() + 4.0 * b.abs()).clamp_min(1e-6)
        return a / denom, b / denom

    def _reflect_boundary_class(
        self, x: torch.Tensor, axis_name: str, scale_idx: int
    ) -> int:
        """Return 0 for reflect-valid and 1 for replicate fallback.

        Non-reflect padding has one common linear boundary operator and returns 0.
        """
        if self.padding_mode != "reflect":
            return 0
        spatial = int(x.shape[-2] if axis_name == "h" else x.shape[-1])
        pad = (self.M + 1) * int(self.dilations[scale_idx])
        return int(pad >= spatial)

    def _forward_closed_form_m1_dyadic_axis(
        self,
        x: torch.Tensor,
        axis_idx: int,
        axis_name: str,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Exact closed-form evaluator for M=1 and dilations (1,2,4).

        Define p_s=pi_s*a_s/D_s and q_s=pi_s*b_s/D_s. If scales 1 and
        2 use the same boundary operator, their exact fused K=9 kernel is

          [q_2,0,q_1+p_2,p_1,2(q_1+q_2),p_1,q_1+p_2,0,q_2].

        Scale 4 is [q_4,p_4,2q_4,p_4,q_4] with dilation four. If reflect
        padding makes scales 1 and 2 use different boundary operators, each is
        evaluated by its own closed-form K=5 kernel, preserving exactness.
        """
        u, v = self._m1_dyadic_uv(axis_idx, x.device, x.dtype)
        p = weights[axis_idx].unsqueeze(1) * u
        q = weights[axis_idx].unsqueeze(1) * v

        same_boundary = (
            self._reflect_boundary_class(x, axis_name, 0)
            == self._reflect_boundary_class(x, axis_name, 1)
        )
        if same_boundary:
            zero = torch.zeros_like(p[0])
            kernel_12_group = torch.stack(
                (
                    q[1],
                    zero,
                    q[0] + p[1],
                    p[0],
                    2.0 * (q[0] + q[1]),
                    p[0],
                    q[0] + p[1],
                    zero,
                    q[1],
                ),
                dim=1,
            )
            kernel_12 = self._expand_group_kernel_to_channels(kernel_12_group)
            y = self._conv_axis(x, axis_name, kernel_12, dilation=1)
        else:
            kernel_1 = self._expand_group_kernel_to_channels(
                self._build_m1_weighted_group_kernel_from_uv(p[0], q[0])
            )
            kernel_2 = self._expand_group_kernel_to_channels(
                self._build_m1_weighted_group_kernel_from_uv(p[1], q[1])
            )
            y = self._conv_axis(x, axis_name, kernel_1, dilation=1)
            y = y + self._conv_axis(x, axis_name, kernel_2, dilation=2)

        kernel_4 = self._expand_group_kernel_to_channels(
            self._build_m1_weighted_group_kernel_from_uv(p[2], q[2])
        )
        return y + self._conv_axis(x, axis_name, kernel_4, dilation=4)

    def _build_exact_group_kernel_1d(
        self,
        axis_idx: int,
        scale_indices: Sequence[int],
        axis_scale_weights: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build one exact Laurent-polynomial realization of a scale group.

        For a fixed axis u and a group G of dilation scales, the original branch sum

            sum_{s in G} pi[u,s] T_{d_s,k_{u,s}} x

        equals one unit-dilation convolution with kernel K_{u,G} satisfying

            K_{u,G}[t] = sum_{s in G} sum_r
                         pi[u,s] k_{u,s}[r] 1_{t=d_s r}.

        No coefficient, routing weight, normalization, boundary rule, or residual
        operation is changed.
        """
        group = tuple(int(i) for i in scale_indices)
        if not group:
            raise ValueError("scale_indices must be non-empty")
        radius = self.M + 1
        max_offset = radius * max(int(self.dilations[i]) for i in group)
        fused = torch.zeros(self.C, 2 * max_offset + 1, device=device, dtype=dtype)
        base_offsets = torch.arange(-radius, radius + 1, device=device, dtype=torch.long)
        for scale_idx in group:
            dilation = int(self.dilations[scale_idx])
            kernel = self._build_weighted_dt1d_kernel_1d(
                axis_idx, scale_idx, device, dtype
            ).squeeze(1)
            positions = max_offset + dilation * base_offsets
            source = (
                axis_scale_weights[axis_idx, scale_idx] * kernel
            ).to(device=fused.device, dtype=fused.dtype)
            fused = fused.index_add(1, positions, source)
        return fused.unsqueeze(1)

    def _group_dense_cost(self, scale_indices: Sequence[int]) -> int:
        """Dense tap count of an exact realization group."""
        group = tuple(int(i) for i in scale_indices)
        if len(group) == 1:
            return 2 * self.M + 3
        return 2 * (self.M + 1) * max(int(self.dilations[i]) for i in group) + 1

    def _optimal_exact_partition(
        self, scale_indices: Sequence[int]
    ) -> Tuple[Tuple[int, ...], ...]:
        """Globally minimize exact dense taps, then convolution launches.

        The admissible family contains the original all-singleton realization. Hence
        the optimum can never have more dense taps or more launches at equal tap cost
        than the original branch algorithm. The recurrence is

            J(i) = min_{0 <= j < i} J(j) + c({j,...,i-1}),

        after sorting scales by dilation. Because group cost depends only on its
        maximum dilation and singleton status, an optimal partition has a contiguous
        representative in this order.
        """
        ordered = tuple(sorted((int(i) for i in scale_indices), key=lambda i: self.dilations[i]))
        n = len(ordered)
        if n == 0:
            return tuple()
        # dp[i] = (tap_cost, launches, partition) for ordered[:i].
        dp = [(0, 0, tuple())] + [None] * n
        for i in range(1, n + 1):
            best = None
            for j in range(i):
                group = ordered[j:i]
                prev_cost, prev_launches, prev_partition = dp[j]
                candidate = (
                    prev_cost + self._group_dense_cost(group),
                    prev_launches + 1,
                    prev_partition + (group,),
                )
                if best is None or candidate[:2] < best[:2]:
                    best = candidate
            dp[i] = best
        return dp[n][2]

    def _exact_partition_for_axis(
        self, x: torch.Tensor, axis_name: str
    ) -> Tuple[Tuple[int, ...], ...]:
        """Return a boundary-equivalent exact partition for one axis.

        Under reflect padding, the uploaded implementation switches an individual
        branch to replicate padding when its required pad is not smaller than the
        spatial dimension. Branches with different boundary operators cannot be fused
        exactly, so they are optimized in separate equivalence classes.
        """
        scales = tuple(range(self.num_scales))
        if self.padding_mode != "reflect":
            classes = (scales,)
        else:
            spatial = int(x.shape[-2] if axis_name == "h" else x.shape[-1])
            radius = self.M + 1
            reflect_scales = tuple(
                i for i in scales if radius * int(self.dilations[i]) < spatial
            )
            replicate_scales = tuple(
                i for i in scales if radius * int(self.dilations[i]) >= spatial
            )
            classes = tuple(g for g in (reflect_scales, replicate_scales) if g)
        result = []
        for equivalence_class in classes:
            result.extend(self._optimal_exact_partition(equivalence_class))
        return tuple(result)

    def exact_realization_cost(self, x: torch.Tensor, axis_name: str) -> Dict[str, int]:
        """Report per-axis mathematical cost before and after optimization."""
        base = 2 * self.M + 3
        partition = self._exact_partition_for_axis(x, axis_name)
        return {
            "before_calls": int(self.num_scales),
            "after_calls": int(len(partition)),
            "before_dense_taps": int(self.num_scales * base),
            "after_dense_taps": int(sum(self._group_dense_cost(g) for g in partition)),
        }

    def _pad(self, x: torch.Tensor, pad_h: int, pad_w: int) -> torch.Tensor:
        if pad_h == 0 and pad_w == 0:
            return x
        if self.padding_mode == "constant":
            return F.pad(x, (pad_w, pad_w, pad_h, pad_h), mode="constant", value=0.0)

        # Reflect padding requires the padding size to be smaller than the corresponding dimension.
        # Fall back to replicate for very small feature maps.
        mode = self.padding_mode
        if mode == "reflect":
            H, W = x.shape[-2], x.shape[-1]
            if (pad_h >= H and pad_h > 0) or (pad_w >= W and pad_w > 0):
                mode = "replicate"
        return F.pad(x, (pad_w, pad_w, pad_h, pad_h), mode=mode)

    def _conv_axis(self, x: torch.Tensor, axis_name: str, w1d: torch.Tensor, dilation: int) -> torch.Tensor:
        K = int(w1d.shape[-1])
        radius = K // 2
        pad = radius * dilation
        if axis_name == "h":
            weight = w1d.view(self.C, 1, K, 1)
            x_pad = self._pad(x, pad_h=pad, pad_w=0)
            return F.conv2d(x_pad, weight, stride=1, padding=0, dilation=(dilation, 1), groups=self.C)
        if axis_name == "w":
            weight = w1d.view(self.C, 1, 1, K)
            x_pad = self._pad(x, pad_h=0, pad_w=pad)
            return F.conv2d(x_pad, weight, stride=1, padding=0, dilation=(1, dilation), groups=self.C)
        raise ValueError(f"Unknown axis_name={axis_name!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"DT1DAdapter expects BCHW input, got shape {tuple(x.shape)}")
        if x.shape[1] != self.C:
            raise ValueError(f"Channel mismatch: adapter C={self.C}, input C={x.shape[1]}")

        # HOSQ path: MLQ8 coarse quotient plus orthogonal zero-DC
        # channel-spectral detail, still using one convolution per enabled axis.
        if self.hosq_realization:
            kernels = self._build_normalized_hosq_kernels(x.device, x.dtype)
            y = torch.zeros_like(x)
            for ai, axis_name in enumerate(self.axis_names):
                y = y + self._conv_axis(x, axis_name, kernels[ai], dilation=1)
        # Minimal Laurent quotient path: one exact symmetric axial convolution
        # per enabled axis, with no scale-branch redundancy.
        elif self.minimal_quotient_realization:
            beta = self._normalized_quotient_beta(x.device, x.dtype)
            y = torch.zeros_like(x)
            for ai, axis_name in enumerate(self.axis_names):
                w1d = self._build_minimal_quotient_kernel_1d(
                    beta, ai, x.device, x.dtype
                )
                y = y + self._conv_axis(x, axis_name, w1d, dilation=1)
        # Step 2 path: static/global mixture over axis--dilation responses.
        elif self.scale_adaptive:
            weights = self._compute_axis_scale_weights(x.device, x.dtype)  # (A, S)
            y = torch.zeros_like(x)
            for ai, axis_name in enumerate(self.axis_names):
                if (
                    self.exact_cost_realization
                    and self.closed_form_dyadic_realization
                    and self.M == 1
                    and tuple(self.dilations) == (1, 2, 4)
                ):
                    y = y + self._forward_closed_form_m1_dyadic_axis(
                        x, ai, axis_name, weights
                    )
                elif self.exact_cost_realization:
                    partition = self._exact_partition_for_axis(x, axis_name)
                    for group in partition:
                        if len(group) == 1:
                            si = group[0]
                            w1d = self._build_weighted_dt1d_kernel_1d(
                                ai, si, x.device, x.dtype
                            )
                            yi = self._conv_axis(
                                x, axis_name, w1d, int(self.dilations[si])
                            )
                            y = y + weights[ai, si] * yi
                        else:
                            w1d = self._build_exact_group_kernel_1d(
                                ai, group, weights, x.device, x.dtype
                            )
                            y = y + self._conv_axis(x, axis_name, w1d, dilation=1)
                else:
                    # Uploaded branch realization retained as a fair ablation.
                    for si, dilation in enumerate(self.dilations):
                        w1d = self._build_weighted_dt1d_kernel_1d(
                            ai, si, x.device, x.dtype
                        )
                        yi = self._conv_axis(x, axis_name, w1d, dilation)
                        y = y + weights[ai, si] * yi
        else:
            # Single-dilation path: average selected weighted DT1D axial responses to
            # preserve the response scale when both axes are enabled.
            y = None
            n_axes = 0
            scale_idx = 0
            dilation = self.dilations[0]
            for ai, axis_name in enumerate(self.axis_names):
                w1d = self._build_weighted_dt1d_kernel_1d(ai, scale_idx, x.device, x.dtype)
                yi = self._conv_axis(x, axis_name, w1d, dilation)
                y = yi if y is None else y + yi
                n_axes += 1
            if y is None:
                y = x
                n_axes = 1
            y = y / float(max(1, n_axes))

        y = self.pw(y)
        return x + self.residual_scale * self.gate * y

