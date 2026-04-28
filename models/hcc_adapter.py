# models/hcc_adapter.py
"""
DT1D-Adapter / HCCAdapter
-------------------------
A lightweight spatial PEFT adapter based on axial depthwise 1D filtering.

Main revision fixes:
1. The two-axis mode averages height/width responses instead of summing them,
   which avoids doubling the identity-like response at initialization.
2. Group count uses ceil(C / alpha_group), so remainder channels are handled correctly.
3. The default residual gate is identity-safe: gate_init=0.0.
4. The module exposes a small parameter-count helper for paper/debug reporting.

Backward-compatible aliases are kept:
    HCCAdapter = DT1DAdapter
    H1D_DT_Adapter = DT1DAdapter
    OneDDTAdapter = DT1DAdapter
"""

from __future__ import annotations

import math
from math import gcd
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        # keep unknown legacy kwargs harmless, because older main.py may pass unused flags.

        self.C = int(C)
        self.M = int(M)
        self.h = int(h)
        self.axis = axis
        self.alpha_group = max(1, int(alpha_group))
        self.tie_sym = bool(tie_sym)
        self.no_pw = bool(no_pw)
        self.use_bn = bool(use_bn)
        self.residual_scale = float(residual_scale)
        self.padding_mode = "constant" if padding_mode == "zeros" else padding_mode

        if self.C <= 0:
            raise ValueError(f"C must be positive, got {self.C}")
        if self.M < 0:
            raise ValueError(f"M must be non-negative, got {self.M}")
        if self.h <= 0:
            raise ValueError(f"h/dilation must be positive, got {self.h}")

        # Number of coefficient-sharing groups. Use ceil, not floor.
        self.num_alpha_groups = math.ceil(self.C / self.alpha_group)
        ncoef = self.M + 1  # center + M symmetric side taps

        self.alpha = nn.Parameter(torch.zeros(self.num_alpha_groups, ncoef))
        with torch.no_grad():
            self.alpha[:, 0].fill_(1.0)  # identity-like axial filter before residual gate

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
            f"C={self.C}, M={self.M}, h={self.h}, axis={self.axis}, "
            f"alpha_group={self.alpha_group}, G={self.num_alpha_groups}, "
            f"no_pw={self.no_pw}, gate={float(self.gate.detach().cpu()):.4g}"
        )

    def parameter_count_breakdown(self) -> Dict[str, int]:
        axial = self.alpha.numel() + self.gate.numel()
        pw = sum(p.numel() for p in self.pw.parameters())
        return {
            "axial_alpha_and_gate": int(axial),
            "pointwise": int(pw),
            "total": int(axial + pw),
        }

    def _build_even_kernel_1d(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Build normalized symmetric kernels with shape (C, 1, 2M+1)."""
        K = 2 * self.M + 1
        center = self.M

        wg = torch.zeros(self.num_alpha_groups, K, device=device, dtype=dtype)
        wg[:, center] = self.alpha[:, 0].to(device=device, dtype=dtype)

        for m in range(1, self.M + 1):
            val = self.alpha[:, m].to(device=device, dtype=dtype)
            wg[:, center - m] = val
            wg[:, center + m] = val if self.tie_sym else val

        # L1 normalization keeps the filter response numerically stable.
        denom = wg.abs().sum(dim=1, keepdim=True).clamp_min(1e-6)
        wg = wg / denom

        # Expand each group filter to the channels assigned to that group.
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

    def _pad(self, x: torch.Tensor, pad_h: int, pad_w: int) -> torch.Tensor:
        if pad_h == 0 and pad_w == 0:
            return x
        if self.padding_mode == "constant":
            return F.pad(x, (pad_w, pad_w, pad_h, pad_h), mode="constant", value=0.0)

        # reflect padding requires the padding size to be smaller than the corresponding dimension.
        # Fall back to replicate for very small feature maps.
        mode = self.padding_mode
        if mode == "reflect":
            H, W = x.shape[-2], x.shape[-1]
            if (pad_h >= H and pad_h > 0) or (pad_w >= W and pad_w > 0):
                mode = "replicate"
        return F.pad(x, (pad_w, pad_w, pad_h, pad_h), mode=mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"DT1DAdapter expects BCHW input, got shape {tuple(x.shape)}")
        if x.shape[1] != self.C:
            raise ValueError(f"Channel mismatch: adapter C={self.C}, input C={x.shape[1]}")

        w1d = self._build_even_kernel_1d(x.device, x.dtype)
        y = None
        n_axes = 0

        if "h" in self.axis:
            wh = w1d.view(self.C, 1, 2 * self.M + 1, 1)
            xh = self._pad(x, pad_h=self.M * self.h, pad_w=0)
            yh = F.conv2d(xh, wh, stride=1, padding=0, dilation=(self.h, 1), groups=self.C)
            y = yh if y is None else y + yh
            n_axes += 1

        if "w" in self.axis:
            ww = w1d.view(self.C, 1, 1, 2 * self.M + 1)
            xw = self._pad(x, pad_h=0, pad_w=self.M * self.h)
            yw = F.conv2d(xw, ww, stride=1, padding=0, dilation=(1, self.h), groups=self.C)
            y = yw if y is None else y + yw
            n_axes += 1

        if y is None:
            y = x
            n_axes = 1

        # Revision fix: average across selected axes to preserve response scale.
        y = y / float(max(1, n_axes))
        y = self.pw(y)
        return x + self.residual_scale * self.gate * y


# Backward-compatible aliases.
HCCAdapter = DT1DAdapter
H1D_DT_Adapter = DT1DAdapter
OneDDTAdapter = DT1DAdapter
