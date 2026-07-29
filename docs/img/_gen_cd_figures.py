"""Regenerate the CD-diagram figures used on the docs home page.

Produces a light-mode and a dark-mode PNG from the same synthetic
BenchmarkReport so the two figures share layout/data exactly.

Run:  ../.venv/bin/python docs/img/_gen_cd_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from graphnetz.benchmark import BenchmarkReport  # noqa: E402

sys.path.insert(0, str(ROOT))
from paper.experiments._style import style_cd_axes  # noqa: E402

# Per-task accuracy means engineered to reproduce mean Friedman ranks of
# GCN=1.67, GraphSAGE=2.00, GraphTransformer=2.67, GAT=3.67 across three
# tasks (matches the figure that was previously committed to the repo).
TASK_MEANS = {
    "task1": {"GCN": 0.90, "GraphSAGE": 0.80, "GraphTransformer": 0.70, "GAT": 0.60},
    "task2": {"GCN": 0.90, "GraphSAGE": 0.80, "GAT": 0.70, "GraphTransformer": 0.60},
    "task3": {"GraphTransformer": 0.90, "GraphSAGE": 0.80, "GCN": 0.70, "GAT": 0.60},
}


def _histories():
    histories = {}
    for task, model_means in TASK_MEANS.items():
        histories[task] = {model: [{"test_acc": [acc]} for _ in range(5)] for model, acc in model_means.items()}
    return histories


def _save_light(out: Path) -> None:
    report = BenchmarkReport(seeds=(0, 1, 2, 3, 4), histories=_histories())
    fig, ax = report.plot_critical_difference(alpha=0.05)
    # Same rank colouring as the paper's figure, so the docs and the manuscript
    # show the same diagram rather than two conventions.
    style_cd_axes(ax)
    fig.savefig(out, dpi=600, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def _recolor_for_dark(fig: plt.Figure) -> None:
    """Walk the figure and swap dark-on-light for light-on-dark."""
    fg = "#e8ecf2"
    muted = "#adb5bd"

    # Classify by luminance and chroma rather than by a list of hex values. The
    # list version silently rotted when the palette changed: the rank tick
    # numbers and the CD label take their colour from rcParams rather than from
    # an explicit argument, so the new ink was not in the table and they stayed
    # dark on a dark page. A rule cannot fall out of date this way.
    def _swap(current: str) -> str | None:
        r, g, b = mpl.colors.to_rgb(current)
        # Hue-bearing colours are the rank ramp, which is solved against the
        # dark page separately; leave them alone.
        if max(r, g, b) - min(r, g, b) > 0.12:
            return None
        linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in (r, g, b)]
        lum = 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]
        if lum < 0.18:  # ink
            return fg
        if lum < 0.55:  # connectors, captions, the Friedman note
            return muted
        return None  # already light enough for a dark page

    # Transparent canvas so the surrounding page colour (whatever Furo's
    # current --color-background-primary happens to be) shows through.
    fig.patch.set_alpha(0.0)
    for ax in fig.get_axes():
        ax.patch.set_alpha(0.0)
        for spine in ax.spines.values():
            spine.set_edgecolor(fg)
        ax.tick_params(colors=fg, which="both")
        ax.xaxis.label.set_color(fg)
        ax.yaxis.label.set_color(fg)
        ax.title.set_color(fg)
        for text in ax.texts:
            new = _swap(mpl.colors.to_hex(text.get_color()).lower())
            if new is not None:
                text.set_color(new)
        for line in ax.lines:
            new = _swap(mpl.colors.to_hex(line.get_color()).lower())
            if new is not None:
                line.set_color(new)


def _save_dark(out: Path) -> None:
    report = BenchmarkReport(seeds=(0, 1, 2, 3, 4), histories=_histories())
    fig, ax = report.plot_critical_difference(alpha=0.05)
    _recolor_for_dark(fig)
    style_cd_axes(ax, on_dark=True)
    fig.savefig(
        out,
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.05,
        transparent=True,
    )
    plt.close(fig)


if __name__ == "__main__":
    static = Path(__file__).resolve().parent
    _save_light(static / "critical_difference.png")
    _save_dark(static / "critical_difference_dark.png")
    print("wrote:")
    print(" ", static / "critical_difference.png")
    print(" ", static / "critical_difference_dark.png")
