#!/usr/bin/env python3
"""Generate deterministic CNN-only DT1D-Adapter architecture Figure 3."""
from __future__ import annotations

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]


def box(ax, x, y, w, h, text):
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", fill=False)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="->", mutation_scale=12))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=ROOT / "outputs" / "cnn_paper_three_seed" / "figures" / "figure_03_cnn_architecture.png")
    a = p.parse_args()
    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")
    box(ax, 0.3, 2.3, 1.5, 1.0, "Frozen CNN\nbackbone block")
    box(ax, 2.2, 2.3, 1.4, 1.0, "Feature map\nB×C×H×W")
    box(ax, 4.1, 3.9, 2.0, 0.8, "Kernel bank\nα → sym → shift → norm")
    box(ax, 4.1, 2.6, 2.0, 0.8, "Height axis\nd = 1, 2, 4")
    box(ax, 4.1, 1.3, 2.0, 0.8, "Width axis\nd = 1, 2, 4")
    box(ax, 6.8, 2.3, 1.8, 1.0, "Axis–scale fusion\nπ = softmax(γ/τ)")
    box(ax, 9.1, 2.3, 1.0, 1.0, "+")
    box(ax, 10.6, 2.3, 1.1, 1.0, "Adapted\nfeature")
    arrow(ax, 1.8, 2.8, 2.2, 2.8)
    arrow(ax, 3.6, 2.8, 4.1, 3.0)
    arrow(ax, 5.1, 3.9, 5.1, 3.4)
    arrow(ax, 5.1, 3.9, 5.1, 2.1)
    arrow(ax, 6.1, 3.0, 6.8, 2.9)
    arrow(ax, 6.1, 1.7, 6.8, 2.6)
    arrow(ax, 8.6, 2.8, 9.1, 2.8)
    arrow(ax, 10.1, 2.8, 10.6, 2.8)
    # Residual skip.
    ax.plot([3.0, 3.0, 9.6], [2.3, 0.55, 0.55], linestyle="--")
    arrow(ax, 9.6, 0.55, 9.6, 2.3)
    ax.text(6.2, 5.35, "DT1D-Adapter (CNN-only package)", ha="center", fontsize=13, fontweight="bold")
    fig.tight_layout()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(a.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
