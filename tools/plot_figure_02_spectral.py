#!/usr/bin/env python3
"""Generate deterministic DT1D spectral-response Figure 2 (no training seed)."""
from __future__ import annotations

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def direct_response(omega: np.ndarray, alpha=(1.0, 0.65, 0.25), dilation=1) -> np.ndarray:
    return alpha[0] + 2 * sum(alpha[m] * np.cos(m * dilation * omega) for m in range(1, len(alpha)))


def shifted_response(omega: np.ndarray, alpha=(1.0, 0.65, 0.25), dilation=1) -> np.ndarray:
    return 2 * np.cos(dilation * omega) * direct_response(omega, alpha, dilation)


def normalized(values: np.ndarray) -> np.ndarray:
    scale = np.max(np.abs(values))
    return values / scale if scale else values


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=ROOT / "outputs" / "cnn_paper_three_seed" / "figures" / "figure_02_spectral.png")
    a = p.parse_args()
    omega = np.linspace(-np.pi, np.pi, 1000)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(omega, normalized(direct_response(omega, dilation=1)), linestyle="--", label="Direct symmetric, d=1")
    axes[0].plot(omega, normalized(shifted_response(omega, dilation=1)), label="DT1D shifted, d=1")
    axes[0].set_title("Direct symmetric vs. shifted DT1D")
    for dilation in (1, 2, 4):
        axes[1].plot(omega, normalized(shifted_response(omega, dilation=dilation)), label=f"d={dilation}")
    axes[1].set_title("Effect of dilation")
    for ax in axes:
        ax.set_xlabel("Frequency ω")
        ax.set_ylabel("Normalized response")
        ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], ["−π", "−π/2", "0", "π/2", "π"])
        ax.grid(True, alpha=0.25)
        ax.legend()
    fig.tight_layout()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(a.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
