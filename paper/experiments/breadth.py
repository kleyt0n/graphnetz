"""Section 3: the cross-category benchmark.

Trains one representative dataset per research category (ten slots) against
every compatible architecture over ten seeds, then emits

    figures/breadth_dashboard.png            -- Figure 2
    figures/breadth_critical_difference.png  -- Figure 3
    tables/breadth.tex                       -- Table 3
    _artifacts/breadth_summary.csv           -- the same cells, as numbers
    _artifacts/breadth_ranks.csv             -- the rank table behind Figure 3
    _artifacts/headline.json                 -- the statistics quoted in prose

The four node-level encoders are present in every slot and GIN only in the
two graph-classification slots, so the CD diagram reduces to ``k = 4``,
``N = 10`` on its own -- the paper's ``k`` is a consequence of the protocol,
not a filter applied afterwards.
"""

from __future__ import annotations

import json
from pathlib import Path
from string import ascii_lowercase
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from paper.experiments import _latex, _runs
from paper.experiments._config import (
    ALPHA,
    BREADTH_SLOTS,
    CI,
    METRIC_AXIS_LABELS,
    MODEL_NAMES,
    SEEDS,
    Cell,
    Slot,
)
from paper.experiments._style import PAPER_COLORS, style_cd_axes

from graphnetz import BenchmarkReport, panel_label

NCOLS = 3

PANEL_SIZE: tuple[float, float] = (2.35, 1.95)
H_PAD = 2.4
W_PAD = 1.0
PANEL_LABEL_X = -0.22
"""Dashboard geometry, tuned against the published Figure 2 (~7 x 7.6 in at
600 dpi, four rows of three panels).  Exact pixel dimensions depend on the
installed font metrics -- ``main.tex`` embeds the figure at ``\\linewidth``, so
what matters is the panel grid and the per-panel aspect, not the raw size."""


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #


def train(
    *,
    seeds: tuple[int, ...] = SEEDS,
    data_root: str,
    cache_dir: Path,
    slots: tuple[Slot, ...] = BREADTH_SLOTS,
    epochs: int | None = None,
    force: bool = False,
    verbose: bool = True,
    cache_only: bool = False,
    chunk: int = 2,
    max_seconds: float | None = None,
) -> dict[str, BenchmarkReport]:
    """Train (or load) every slot; returns ``{slot.key: report}`` in slot order.

    Slots are independent cells, so a run that dies partway through (a dead
    dataset mirror, an interrupted download) resumes from the cache rather
    than starting over.  Under ``cache_only`` uncached slots are omitted.
    """
    reports: dict[str, BenchmarkReport] = {}
    for slot in slots:
        task = _runs.resolve_task(slot)
        report = _runs.train_cell(
            label=slot.key,
            tasks=[task],
            category=slot.category,
            model_names=MODEL_NAMES,
            seeds=seeds,
            data_root=data_root,
            cache_path=cache_dir / f"breadth_{slot.category}_{slot.task_name}.json",
            epochs=epochs,
            force=force,
            verbose=verbose,
            cache_only=cache_only,
            chunk=chunk,
            max_seconds=max_seconds,
        )
        if report is not None:
            reports[slot.key] = report
    return reports


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #


def cells(
    reports: dict[str, BenchmarkReport],
    slots: tuple[Slot, ...] = BREADTH_SLOTS,
) -> list[Cell]:
    """Reduce each slot report to a :class:`Cell`, in paper (table) order.

    Statistics come from ``BenchmarkReport.summary`` so the interval is the
    library's Student's-t half-width (Eq. 2), not a re-implementation.
    """
    out: list[Cell] = []
    for slot in slots:
        report = reports[slot.key]
        task_name = next(iter(report.histories))
        per_task = report.summary(ci=CI).loc[task_name]
        stats = {
            model: (float(r["mean"]), float((r["ci_high"] - r["ci_low"]) / 2.0)) for model, r in per_task.iterrows()
        }
        out.append(
            Cell(
                slot=slot,
                metric=report.metric_name(),
                stats={m: stats[m] for m in MODEL_NAMES if m in stats},
            )
        )
    return out


def summary_frame(rows: list[Cell]) -> pd.DataFrame:
    """Long-format view of Table 3, one line per (category, dataset, model)."""
    records = []
    for cell in rows:
        slot = cell.slot
        for model, (mean, half) in cell.stats.items():
            records.append(
                {
                    "category": slot.category_label,
                    "dataset": slot.dataset_label,
                    "task_type": slot.task_type,
                    "metric": cell.metric,
                    "model": model,
                    "mean": mean,
                    "ci_half": half,
                    "ci_low": mean - half,
                    "ci_high": mean + half,
                }
            )
    return pd.DataFrame(records).set_index(["category", "dataset", "model"])


def headline(merged: BenchmarkReport) -> tuple[dict[str, Any], pd.DataFrame]:
    """The cross-category numbers quoted in Sections 3.4 / 4 and the abstract.

    Returns ``(statistics, rank_table)``.  The Nemenyi ``CD`` is recomputed
    here with the same formula ``plot_critical_difference`` uses (Eq. 6), so
    the number written to ``headline.json`` and the number drawn on Figure 3
    cannot disagree.
    """
    from scipy.stats import studentized_range

    friedman = merged.friedman(alpha=ALPHA)
    finals = merged.final_metrics()
    metrics = {task: _metric_of(merged, task) for task in finals}
    ranks = _latex.rank_table(finals, metrics)
    mean_ranks = ranks.mean(axis=0).sort_values()
    k, n = len(mean_ranks), len(ranks)
    q = float(studentized_range.ppf(1 - ALPHA, k, np.inf) / np.sqrt(2))
    cd = q * float(np.sqrt(k * (k + 1) / (6 * n)))
    gap = float(mean_ranks.max() - mean_ranks.min())
    stats: dict[str, Any] = {
        "alpha": ALPHA,
        "k": k,
        "n": n,
        "friedman_chi2": round(float(friedman["chi2"]), 4),
        "friedman_p": round(float(friedman["p_value"]), 4),
        "friedman_rejected": bool(friedman["rejected"]),
        "critical_difference": round(cd, 4),
        "mean_ranks": {m: round(float(r), 4) for m, r in mean_ranks.items()},
        "max_rank_gap": round(gap, 4),
        "single_clique": bool(gap < cd),
    }
    return stats, ranks


def merge_slots(reports: dict[str, BenchmarkReport]) -> BenchmarkReport:
    """Merge per-slot reports into one whose tasks are the ten category slots.

    The merged report is what the across-task analyses operate on: the CD
    diagram, the rank-stability bootstrap, and the aggregation comparison all
    need every slot in a single metric tensor.
    """
    return _runs.merge(reports)


def aggregate(
    reports: dict[str, BenchmarkReport],
    slots: tuple[Slot, ...] = BREADTH_SLOTS,
) -> tuple[dict[str, Any], list[Cell], pd.DataFrame]:
    """Compute everything Section 3 reports, writing nothing.

    Returns ``(headline stats, Table-3 rows, rank table)`` -- the read-only
    half of :func:`emit`, used by the ``verify`` stage.
    """
    rows = cells(reports, slots)
    merged = _runs.merge({slot.key: reports[slot.key] for slot in slots})
    stats, ranks = headline(merged)
    return stats, rows, ranks


def _metric_of(report: BenchmarkReport, task: str) -> str:
    """The history key a task is scored on (``test_auc``, ``val_mae``, ...)."""
    sample = next(iter(report.histories[task].values()))[0]
    for key in ("test_acc", "test_auc", "val_acc", "val_auc", "val_mae"):
        if key in sample:
            return key
    return next(iter(sample))


# --------------------------------------------------------------------------- #
# Figure 2: the cross-category dashboard
# --------------------------------------------------------------------------- #


def dashboard(
    rows: list[Cell],
    *,
    h_pad: float = H_PAD,
    w_pad: float = W_PAD,
) -> tuple[plt.Figure, np.ndarray]:
    """One bar panel per category: seed mean with a 95 % Student's t interval.

    Each panel keeps its own y-limits and its own metric label -- the panels
    are deliberately *not* commensurable (Section 3.6), so no shared axis is
    drawn that would invite cross-panel reading.

    ``h_pad`` / ``w_pad`` are the ``tight_layout`` paddings; they are exposed
    because the panel data area (and hence the y-tick density) is sensitive to
    them, and the paper's figure was laid out at the module defaults.
    """
    n = len(rows)
    nrows = -(-n // NCOLS)
    fig, axes = plt.subplots(
        nrows,
        NCOLS,
        figsize=(PANEL_SIZE[0] * NCOLS, PANEL_SIZE[1] * nrows),
        squeeze=False,
    )
    for idx, cell in enumerate(rows):
        slot, stats = cell.slot, cell.stats
        ax = axes[idx // NCOLS][idx % NCOLS]
        models = sorted(stats)
        means = np.array([stats[m][0] for m in models])
        halves = np.array([stats[m][1] for m in models])
        xs = np.arange(len(models))

        ax.bar(
            xs,
            means,
            width=0.8,
            color=[PAPER_COLORS[i % len(PAPER_COLORS)] for i in range(len(models))],
            linewidth=0,
        )
        ax.errorbar(xs, means, yerr=halves, fmt="none", ecolor="0.2", elinewidth=0.9, capsize=2.0)

        lo = float((means - halves).min())
        hi = float((means + halves).max())
        pad = 0.025 * max(hi - lo, 1e-9)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlim(-0.6, len(models) - 0.4)
        ax.set_xticks(xs)
        ax.set_xticklabels(models, rotation=40, ha="right")
        ax.tick_params(axis="x", which="both", length=0)
        ax.set_ylabel(METRIC_AXIS_LABELS[cell.metric])
        ax.set_title(f"{slot.category_label} • {slot.dataset_label}", loc="center")
        ax.set_axisbelow(True)
        ax.yaxis.grid(True)
        panel_label(ax, ascii_lowercase[idx], x=PANEL_LABEL_X, y=1.02)

    for idx in range(n, nrows * NCOLS):
        axes[idx // NCOLS][idx % NCOLS].axis("off")
    fig.tight_layout(h_pad=h_pad, w_pad=w_pad)
    return fig, axes


# --------------------------------------------------------------------------- #
# Stage entry point
# --------------------------------------------------------------------------- #


def emit(
    reports: dict[str, BenchmarkReport],
    *,
    fig_dir: Path,
    tab_dir: Path,
    artifact_dir: Path,
    slots: tuple[Slot, ...] = BREADTH_SLOTS,
    formats: tuple[str, ...] = ("png",),
) -> tuple[dict[str, Any], list[Cell]]:
    """Write every Section-3 artefact; returns ``(headline stats, table rows)``."""
    from graphnetz import save_figure

    stats, rows, ranks = aggregate(reports, slots)

    fig, _ = dashboard(rows)
    save_figure(fig, fig_dir / "breadth_dashboard", formats=formats)
    plt.close(fig)

    _latex.breadth_table(tab_dir / "breadth.tex", rows)
    _latex.summary_csv(artifact_dir / "breadth_summary.csv", summary_frame(rows))
    _latex.summary_csv(artifact_dir / "breadth_ranks.csv", ranks)

    merged = _runs.merge({slot.key: reports[slot.key] for slot in slots})

    fig_cd, ax_cd = merged.plot_critical_difference(alpha=ALPHA)
    style_cd_axes(ax_cd)
    save_figure(fig_cd, fig_dir / "breadth_critical_difference", formats=formats)
    plt.close(fig_cd)

    (artifact_dir / "headline.json").write_text(json.dumps(stats, indent=2) + "\n")

    print(f"  Friedman chi2_{stats['k'] - 1} = {stats['friedman_chi2']:.2f}, p = {stats['friedman_p']:.3g}")
    print(f"  CD(alpha={ALPHA}, k={stats['k']}, N={stats['n']}) = {stats['critical_difference']:.3f}")
    ordered = ", ".join(f"{m} {r:.2f}" for m, r in stats["mean_ranks"].items())
    print(f"  mean ranks: {ordered}")
    print(f"  max rank gap = {stats['max_rank_gap']:.2f} -> single Nemenyi clique: {stats['single_clique']}")
    return stats, rows
