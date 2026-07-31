# models/hcc_adapter.py
"""
DT1D-Adapter / HCCAdapter
-------------------------
A lightweight spatial PEFT adapter based on finite weighted h-Hartley-cosine
axial convolution.

This static-gate version keeps the new finite weighted h-Hartley-cosine
axial convolution kernels, but removes the input-adaptive GAP-MLP router.
Axis--scale responses are fused by the old global learnable softmax logits,
so the parameter count remains close to the original DT1D design.

Backward compatibility:
    * If `dilations=None` and `scale_adaptive=False`, the module behaves like the
      original routing behavior: a single dilation `h` is used and height/width
      responses are averaged in two-axis mode, but each branch uses the finite
      weighted h-Hartley-cosine axial kernel.
    * If `dilations=(1, 2, 4)` or `scale_adaptive=True`, the module evaluates
      multiple axial responses and combines them with global/static softmax gates
      over axis--dilation pairs.

Backward-compatible aliases are kept:
    HCCAdapter = DT1DAdapter
    H1D_DT_Adapter = DT1DAdapter
    OneDDTAdapter = DT1DAdapter
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

        # Backward-compatible translation from the old HCC API.
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
        self.alpha = nn.Parameter(torch.zeros(self.num_alpha_axes, self.num_scales, self.num_alpha_groups, ncoef))
        with torch.no_grad():
            self.alpha[..., 0].fill_(1.0)  # identity-like axial filter before residual gate

        # Static/global axis--scale logits. This is the old DT1D fusion mechanism:
        # one small set of learnable logits is shared by all input samples.
        # For A=2 axes and S=3 dilation scales this adds only A*S=6 parameters
        # per inserted adapter. No GAP-MLP router is created.
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
            f"separate_axis_kernels={self.separate_axis_kernels}, "
            f"alpha_group={self.alpha_group}, G={self.num_alpha_groups}, "
            f"no_pw={self.no_pw}, gate={float(self.gate.detach().cpu()):.4g}"
        )

    def parameter_count_breakdown(self) -> Dict[str, int]:
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

    def _build_weighted_hcc_kernel_1d(
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
        # This preserves the weighted-HCC shift pattern while preventing uncontrolled
        # amplification on finite feature maps.
        denom = wg.abs().sum(dim=1, keepdim=True).clamp_min(1e-6)
        wg = wg / denom
        return self._expand_group_kernel_to_channels(wg)


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
            kernel = self._build_weighted_hcc_kernel_1d(
                axis_idx, scale_idx, device, dtype
            ).squeeze(1)
            positions = max_offset + dilation * base_offsets
            fused = fused.index_add(
                1,
                positions,
                axis_scale_weights[axis_idx, scale_idx] * kernel,
            )
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

        # Step 2 path: static/global mixture over axis--dilation responses.
        if self.scale_adaptive:
            weights = self._compute_axis_scale_weights(x.device, x.dtype)  # (A, S)
            y = torch.zeros_like(x)
            for ai, axis_name in enumerate(self.axis_names):
                if self.exact_cost_realization:
                    partition = self._exact_partition_for_axis(x, axis_name)
                    for group in partition:
                        if len(group) == 1:
                            si = group[0]
                            w1d = self._build_weighted_hcc_kernel_1d(
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
                        w1d = self._build_weighted_hcc_kernel_1d(
                            ai, si, x.device, x.dtype
                        )
                        yi = self._conv_axis(x, axis_name, w1d, dilation)
                        y = y + weights[ai, si] * yi
        else:
            # Single-dilation path: average selected weighted-HCC axial responses to
            # preserve the response scale when both axes are enabled.
            y = None
            n_axes = 0
            scale_idx = 0
            dilation = self.dilations[0]
            for ai, axis_name in enumerate(self.axis_names):
                w1d = self._build_weighted_hcc_kernel_1d(ai, scale_idx, x.device, x.dtype)
                yi = self._conv_axis(x, axis_name, w1d, dilation)
                y = yi if y is None else y + yi
                n_axes += 1
            if y is None:
                y = x
                n_axes = 1
            y = y / float(max(1, n_axes))

        y = self.pw(y)
        return x + self.residual_scale * self.gate * y


# Backward-compatible aliases.
HCCAdapter = DT1DAdapter
H1D_DT_Adapter = DT1DAdapter
OneDDTAdapter = DT1DAdapter
