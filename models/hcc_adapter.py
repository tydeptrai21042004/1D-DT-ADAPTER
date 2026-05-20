# models/hcc_adapter.py
"""
DT1D-Adapter / HCCAdapter
-------------------------
A lightweight spatial PEFT adapter based on finite weighted h-Hartley-cosine
axial convolution.

This version implements two revised-method changes:
    1. Finite weighted h-Hartley-cosine axial convolution kernels.
    2. Optional input-adaptive axis--scale routing gates.

Backward compatibility:
    * If `dilations=None` and `scale_adaptive=False`, the module behaves like the
      original routing behavior: a single dilation `h` is used and height/width
      responses are averaged in two-axis mode, but each branch uses the finite
      weighted h-Hartley-cosine axial kernel.
    * If `dilations=(1, 2, 4)` or `scale_adaptive=True`, the module evaluates
      multiple axial responses and combines them with either global softmax gates
      or input-adaptive routing weights over axis--dilation pairs.

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
        input_adaptive_gate: bool = True,
        gate_reduction: int = 4,
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
        if "hcc_input_adaptive_gate" in legacy:
            input_adaptive_gate = bool(legacy.pop("hcc_input_adaptive_gate"))
        if "hcc_gate_reduction" in legacy:
            gate_reduction = int(legacy.pop("hcc_gate_reduction"))
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
        self.input_adaptive_gate = bool(input_adaptive_gate and self.scale_adaptive)
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

        # Global axis--scale logits. In input-adaptive mode these act as learnable
        # bias/prior logits and the routing MLP adds sample-dependent offsets.
        if self.scale_adaptive:
            self.axis_scale_logits = nn.Parameter(torch.zeros(self.num_axes, self.num_scales))
        else:
            self.register_parameter("axis_scale_logits", None)

        # Input-adaptive routing:
        #   r(x)=GAP(x), eta(x)=W2(ReLU(W1 r(x))),
        #   pi_{b,u,s}=softmax((eta_{b,u,s}+global_logit_{u,s})/tau).
        # The final Linear is zero-initialized, so training starts from uniform/global
        # routing while still allowing sample-specific gates to emerge.
        if self.input_adaptive_gate:
            route_hidden = max(1, self.C // self.gate_reduction)
            self.axis_scale_router = nn.Sequential(
                nn.Linear(self.C, route_hidden, bias=True),
                nn.ReLU(inplace=True),
                nn.Linear(route_hidden, self.num_axes * self.num_scales, bias=True),
            )
            nn.init.zeros_(self.axis_scale_router[-1].weight)
            nn.init.zeros_(self.axis_scale_router[-1].bias)
        else:
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
            f"scale_adaptive={self.scale_adaptive}, input_adaptive_gate={self.input_adaptive_gate}, "
            f"separate_axis_kernels={self.separate_axis_kernels}, "
            f"alpha_group={self.alpha_group}, G={self.num_alpha_groups}, "
            f"no_pw={self.no_pw}, gate={float(self.gate.detach().cpu()):.4g}"
        )

    def parameter_count_breakdown(self) -> Dict[str, int]:
        axial = self.alpha.numel() + self.gate.numel()
        axis_scale = 0 if self.axis_scale_logits is None else self.axis_scale_logits.numel()
        router = 0 if self.axis_scale_router is None else sum(p.numel() for p in self.axis_scale_router.parameters())
        pw = sum(p.numel() for p in self.pw.parameters())
        return {
            "axial_alpha_and_gate": int(axial),
            "axis_scale_logits": int(axis_scale),
            "axis_scale_router": int(router),
            "pointwise": int(pw),
            "total": int(axial + axis_scale + router + pw),
        }

    def axis_scale_weights(self, x: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        """Return softmax axis--scale weights.

        If ``x`` is provided and input-adaptive routing is enabled, the return shape
        is ``(B, num_axes, num_scales)``. Otherwise, the return shape is
        ``(num_axes, num_scales)`` for the global/static routing weights.
        """
        if self.axis_scale_logits is None:
            return None
        if x is not None and self.input_adaptive_gate:
            return self._compute_axis_scale_weights(x).detach()
        logits = self.axis_scale_logits.detach() / self.gate_temperature
        return torch.softmax(logits.reshape(-1), dim=0).reshape(self.num_axes, self.num_scales)

    def _compute_axis_scale_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Compute routing weights with shape (B, num_axes, num_scales)."""
        if self.axis_scale_logits is None:
            raise RuntimeError("Axis--scale weights are only defined in scale_adaptive mode.")
        B = x.shape[0]
        logits = self.axis_scale_logits.to(device=x.device, dtype=x.dtype).unsqueeze(0).expand(B, -1, -1)
        if self.axis_scale_router is not None:
            pooled = F.adaptive_avg_pool2d(x, output_size=1).flatten(1)
            route_logits = self.axis_scale_router(pooled).reshape(B, self.num_axes, self.num_scales)
            logits = logits + route_logits.to(device=x.device, dtype=x.dtype)
        weights = torch.softmax((logits / self.gate_temperature).reshape(B, -1), dim=1)
        return weights.reshape(B, self.num_axes, self.num_scales)

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

        # Step 2 path: mixture over axis--dilation responses. The weights are
        # either global/static or input-adaptive per sample.
        if self.scale_adaptive:
            weights = self._compute_axis_scale_weights(x).to(device=x.device, dtype=x.dtype)  # (B, A, S)
            y = torch.zeros_like(x)
            for ai, axis_name in enumerate(self.axis_names):
                for si, dilation in enumerate(self.dilations):
                    w1d = self._build_weighted_hcc_kernel_1d(ai, si, x.device, x.dtype)
                    yi = self._conv_axis(x, axis_name, w1d, dilation)
                    y = y + weights[:, ai, si].view(-1, 1, 1, 1) * yi
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
