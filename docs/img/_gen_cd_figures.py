"""Regenerate the CD-diagram figures used on the docs home page.

Produces a light-mode and a dark-mode PNG from the *real* breadth run — the
ten-category sweep whose per-dataset means live in
``paper/experiments/_artifacts/breadth_summary.csv`` — so the home page shows
the same diagram as Figure 3 of the paper rather than an illustrative mock-up.

Run:  ../.venv/bin/python docs/img/_gen_cd_figures.py

The reconstruction is exact for everything the diagram displays. A CD diagram
is a function of the *rank table* alone: per-task ranks, the Friedman
:math:`\\chi^2`, and the Nemenyi critical difference. All three follow from the
per-(dataset, model) means, which are read verbatim from the artefact. The
per-seed spread inside each cell is not reconstructed — it cannot change a
rank — so each cell is replayed as a constant series. ``_verify`` asserts the
resulting statistics against ``headline.json``, so a drift in either the
artefacts or the ranking code fails the run instead of silently publishing a
figure that no longer matches the paper.

Only models present in every task enter the diagram (``strict=True``), which
drops GIN: it is defined for the two graph-classification slots only.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from graphnetz.benchmark import BenchmarkReport  # noqa: E402

sys.path.insert(0, str(ROOT))
from paper.experiments._style import style_cd_axes  # noqa: E402

ARTIFACTS = ROOT / "paper" / "experiments" / "_artifacts"
SUMMARY = ARTIFACTS / "breadth_summary.csv"
HEADLINE = ARTIFACTS / "headline.json"

ALPHA = 0.05
N_SEEDS = 10


def _report() -> BenchmarkReport:
    """Replay the breadth run's per-cell means as a `BenchmarkReport`."""
    frame = pd.read_csv(SUMMARY)
    histories: dict[str, dict[str, list[dict[str, list[float]]]]] = defaultdict(dict)
    for row in frame.itertuples(index=False):
        cell = [{row.metric: [float(row.mean)]} for _ in range(N_SEEDS)]
        histories[str(row.dataset)][str(row.model)] = cell
    return BenchmarkReport(seeds=tuple(range(N_SEEDS)), histories=dict(histories))


def _verify(report: BenchmarkReport) -> None:
    """Fail loudly if the replay no longer reproduces the published stats."""
    expected = json.loads(HEADLINE.read_text())
    ranks = report.mean_ranks(strict=True).round(2).to_dict()
    published = {m: round(r, 2) for m, r in expected["mean_ranks"].items()}
    if ranks != published:
        msg = f"mean ranks drifted from headline.json:\n  got      {ranks}\n  expected {published}"
        raise SystemExit(msg)
    print(f"  verified against headline.json: {published}")


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

    # Transparent canvas so the surrounding page colour shows through.
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


def _save(out: Path, *, dark: bool) -> None:
    fig, ax = _report().plot_critical_difference(alpha=ALPHA)
    if dark:
        _recolor_for_dark(fig)
    # Same rank colouring as the paper's figure, so the docs and the manuscript
    # show the same diagram rather than two conventions.
    style_cd_axes(ax, on_dark=dark)
    fig.savefig(out, dpi=600, bbox_inches="tight", pad_inches=0.05, transparent=dark)
    plt.close(fig)
    print(f"  wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    static = Path(__file__).resolve().parent
    _verify(_report())
    _save(static / "critical_difference.png", dark=False)
    _save(static / "critical_difference_dark.png", dark=True)
