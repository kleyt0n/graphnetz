"""Revision analyses added in response to the NeurIPS reviews.

Every analysis here reads the cached metric tensor and retrains nothing, which
is the point: the reviewers' central questions about the *interpretation* of a
benchmark are answerable from the data already collected.

    tables/capacity.tex          -- CD vs N: what breadth the claim would need
    tables/power.tex             -- per-task MDE and what the design detected
    tables/equivalence.tex       -- TOST verdicts: different / equivalent / undetermined
    tables/epoch_selection.tex   -- fixed-epoch vs validation-selected checkpoints
    tables/aggregation.tex       -- mean ranks under four task weightings
    tables/rank_stability.tex    -- bootstrap-over-tasks and jackknife
    figures/rank_stability.png   -- the same, as a figure
    figures/epoch_selection.png  -- per-comparison effect shift under selection

Reviewer trace: capacity/power/equivalence answer 6FbE W1 and Q2; rank
stability answers 6FbE Q1 and HVs6 Q1; epoch selection answers 6FbE W2,
Bvni Q3 and HVs6 W3; aggregation answers Bvni Q4 and HVs6 W4.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from paper.experiments import _latex
from paper.experiments._config import ALPHA, BREADTH_SLOTS, SEEDS
from paper.experiments._style import (
    EPOCH_OUTCOME_LABELS,
    EPOCH_OUTCOMES,
    PAPER_COLORS,
    classify_epoch_outcome,
    epoch_outcome_colors,
)

from graphnetz import BenchmarkReport, save_figure
from graphnetz.benchmark._stats import _nemenyi_cd

SESOI = 0.01
"""Smallest effect size of interest for the equivalence tests, in metric units.

One accuracy/AUC point.  Chosen because it is the order of difference the
GNN literature routinely reports as an improvement, so it is the margin at
which "these architectures are interchangeable" would actually be
interesting.  It is a judgement call and is reported as such.
"""

CAPACITY_N = (10, 15, 20, 24, 29, 35, 40, 50, 75, 100)


# --------------------------------------------------------------------------- #
# Design capacity: how much benchmark breadth would the claim need?
# --------------------------------------------------------------------------- #


def capacity(report: BenchmarkReport, *, alpha: float = ALPHA) -> dict[str, Any]:
    """CD as a function of N, and the N at which the observed gap would resolve.

    ``CD_alpha = q_alpha sqrt(k(k+1)/6N)`` depends only on ``(k, N, alpha)``,
    so this is a property of the *design*, not of the data -- which is exactly
    why it can be reported honestly without running anything.  The observed
    max mean-rank gap enters only as the target to be resolved.
    """
    fried = report.friedman(alpha=alpha)
    k = int(fried["k"])
    gap = float(fried["max_rank_gap"])
    rows = []
    for n in CAPACITY_N:
        cd = _nemenyi_cd(k, n, alpha)
        rows.append({"n_tasks": n, "cd": cd, "resolves_observed_gap": bool(np.isfinite(cd) and cd < gap)})
    return {
        "k": k,
        "n_current": int(fried["n"]),
        "cd_current": float(fried["critical_difference"]),
        "max_rank_gap": gap,
        "n_for_observed_gap": int(fried["n_for_observed_gap"]),
        "curve": pd.DataFrame(rows),
    }


def emit_capacity(report: BenchmarkReport, *, tab_dir: Path) -> dict[str, Any]:
    cap = capacity(report)
    curve: pd.DataFrame = cap["curve"]
    rows = []
    for _, r in curve.iterrows():
        n = int(r["n_tasks"])
        label = f"{n}"
        if n == cap["n_current"]:
            label = rf"\textbf{{{n}}} (this paper)"
        elif n == 24:
            label = "24 (whole curated catalogue)"
        elif n == 29:
            label = "29 (curated catalogue + OGB)"
        cd_cell = f"{r['cd']:.3f}"
        verdict = _latex._sig("yes") if r["resolves_observed_gap"] else _latex._quiet("no")
        rows.append([label, cd_cell, verdict])
    _latex.booktabs(
        tab_dir / "capacity.tex",
        ["Benchmark tasks $N$", r"$CD_{0.05}$", rf"Resolves the observed ${cap['max_rank_gap']:.2f}$ gap?"],
        rows,
        align="lcc",
    )
    return cap


# --------------------------------------------------------------------------- #
# Power: what could this design have detected?
# --------------------------------------------------------------------------- #


def emit_power(
    report: BenchmarkReport,
    *,
    tab_dir: Path,
    artifact_dir: Path,
    pretty_tasks: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Per-task minimum detectable effect, and how many cells cleared it."""
    pw = report.power(alpha=ALPHA)
    _latex.summary_csv(artifact_dir / "power.csv", pw.set_index(["task", "model_a", "model_b"]))
    pretty = pretty_tasks or {}
    rows = []
    for task, block in pw.groupby("task", sort=False):
        n_det = int(block["detectable"].sum())
        rows.append(
            [
                pretty.get(str(task), str(task)),
                str(len(block)),
                f"{block['sigma_d'].median():.4f}",
                f"{block['mde'].median():.4f}",
                f"{n_det}/{len(block)}",
                str(int(block["seeds_for_0.01"].median())),
                str(int(block["seeds_for_0.005"].median())),
            ]
        )
    total = [
        r"\textbf{All}",
        str(len(pw)),
        f"{pw['sigma_d'].median():.4f}",
        f"{pw['mde'].median():.4f}",
        f"{int(pw['detectable'].sum())}/{len(pw)}",
        str(int(pw["seeds_for_0.01"].median())),
        str(int(pw["seeds_for_0.005"].median())),
    ]
    _latex.booktabs(
        tab_dir / "power.tex",
        [
            "Task",
            "Comp.",
            r"med.\ $\hat\sigma_d$",
            r"med.\ MDE",
            "Detectable",
            r"$S$ for $0.01$",
            r"$S$ for $0.005$",
        ],
        [*rows, total],
        align="lcccccc",
        rules_before={len(rows)},
    )
    return {
        "n_comparisons": len(pw),
        "median_mde": float(pw["mde"].median()),
        "n_detectable": int(pw["detectable"].sum()),
        "n_undetectable": int((~pw["detectable"]).sum()),
        "median_seeds_for_0.01": int(pw["seeds_for_0.01"].median()),
        "median_seeds_for_0.005": int(pw["seeds_for_0.005"].median()),
    }


# --------------------------------------------------------------------------- #
# Equivalence: turning "not significant" into a stated verdict
# --------------------------------------------------------------------------- #


def emit_equivalence(
    report: BenchmarkReport,
    *,
    tab_dir: Path,
    artifact_dir: Path,
    margin: float = SESOI,
    pretty_tasks: dict[str, str] | None = None,
) -> dict[str, Any]:
    eq = report.equivalence(margin=margin, alpha=ALPHA)
    _latex.summary_csv(artifact_dir / "equivalence.csv", eq.set_index(["task", "model_a", "model_b"]))
    pretty = pretty_tasks or {}
    order = ("different", "equivalent", "trivial", "undetermined")
    rows = []
    for task, block in eq.groupby("task", sort=False):
        counts = block["verdict"].value_counts()
        rows.append([pretty.get(str(task), str(task)), *[str(int(counts.get(v, 0))) for v in order]])
    counts_all = eq["verdict"].value_counts()
    rows.append([r"\textbf{All}", *[rf"\textbf{{{int(counts_all.get(v, 0))}}}" for v in order]])
    _latex.booktabs(
        tab_dir / "equivalence.tex",
        ["Task", "Different", "Equivalent", "Trivial", "Undetermined"],
        rows,
        align="lcccc",
        rules_before={len(rows) - 1},
    )
    return {
        "margin": margin,
        **{v: int(counts_all.get(v, 0)) for v in order},
        "n_comparisons": len(eq),
    }


# --------------------------------------------------------------------------- #
# Epoch selection: the fixed-epoch protocol, measured
# --------------------------------------------------------------------------- #


def epoch_comparison(report: BenchmarkReport) -> pd.DataFrame:
    """Join the pairwise verdicts under both epoch-selection rules.

    ``strict=False`` lets a mixed benchmark through: tasks whose trainer keeps
    no held-out series stay on the final epoch and are flagged, rather than
    silently being selected on the series they are then reported on.
    """
    final = report.pairwise(alpha=ALPHA, epoch_selection="final")
    best = report.pairwise(alpha=ALPHA, epoch_selection="best_val", strict=False)
    key = ["task", "model_a", "model_b"]
    merged = final.merge(best, on=key, suffixes=("_final", "_best"), validate="one_to_one")
    support = report.epoch_selection_support()["best_val_supported"]
    merged["selected"] = merged["task"].map(support).fillna(False)
    merged["flipped"] = merged["significant_final"] != merged["significant_best"]
    return merged


def emit_epoch_selection(
    report: BenchmarkReport,
    *,
    tab_dir: Path,
    fig_dir: Path,
    artifact_dir: Path,
    pretty_tasks: dict[str, str] | None = None,
    formats: tuple[str, ...] = ("png",),
) -> dict[str, Any]:
    cmp = epoch_comparison(report)
    _latex.summary_csv(artifact_dir / "epoch_selection.csv", cmp.set_index(["task", "model_a", "model_b"]))
    pretty = pretty_tasks or {}
    rows = []
    for task, block in cmp.groupby("task", sort=False):
        sel = bool(block["selected"].iloc[0])
        n_f = int(block["significant_final"].sum())
        n_b = int(block["significant_best"].sum())
        rows.append(
            [
                pretty.get(str(task), str(task)),
                r"\checkmark" if sel else r"{\color{nac}\textit{n/a}}",
                str(len(block)),
                str(n_f),
                str(n_b),
                str(int(block["flipped"].sum())),
            ]
        )
    rows.append(
        [
            r"\textbf{All}",
            "",
            rf"\textbf{{{len(cmp)}}}",
            rf"\textbf{{{int(cmp['significant_final'].sum())}}}",
            rf"\textbf{{{int(cmp['significant_best'].sum())}}}",
            rf"\textbf{{{int(cmp['flipped'].sum())}}}",
        ]
    )
    _latex.booktabs(
        tab_dir / "epoch_selection.tex",
        ["Task", "Selectable", "Comp.", "Sig.\\ final", "Sig.\\ best-val", "Flipped"],
        rows,
        align="llcccc",
        rules_before={len(rows) - 1},
    )

    fig = _plot_epoch_shift(cmp)
    save_figure(fig, fig_dir / "epoch_selection", formats=formats)
    plt.close(fig)

    flips = cmp[cmp["flipped"]].copy()
    flips["shrink"] = flips["mean_diff_final"].abs() - flips["mean_diff_best"].abs()
    biggest = flips.sort_values("shrink", ascending=False).head(1)
    out: dict[str, Any] = {
        "n_comparisons": len(cmp),
        "significant_final": int(cmp["significant_final"].sum()),
        "significant_best_val": int(cmp["significant_best"].sum()),
        "flipped": int(cmp["flipped"].sum()),
        "tasks_selectable": int(cmp.groupby("task")["selected"].first().sum()),
        "tasks_total": int(cmp["task"].nunique()),
    }
    if not biggest.empty:
        b = biggest.iloc[0]
        out["largest_reversal"] = {
            "task": str(b["task"]),
            "comparison": f"{b['model_a']} vs {b['model_b']}",
            "mean_diff_final": round(float(b["mean_diff_final"]), 4),
            "p_holm_final": float(b["p_holm_final"]),
            "mean_diff_best_val": round(float(b["mean_diff_best"]), 4),
            "p_holm_best_val": float(b["p_holm_best"]),
        }
    return out


def _plot_epoch_shift(cmp: pd.DataFrame) -> plt.Figure:
    """Paired dumbbell of |mean difference| under both selection rules."""
    from graphnetz.plotting import COLUMN_INCHES, set_plot_style

    set_plot_style()
    d = cmp.copy()
    d["label"] = d["task"] + ": " + d["model_a"] + " vs " + d["model_b"]
    d = d.sort_values("mean_diff_final", key=lambda s: s.abs(), ascending=True)
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(COLUMN_INCHES["double"] * 0.62, 0.24 * len(d) + 1.2))
    a = d["mean_diff_final"].abs().to_numpy()
    b = d["mean_diff_best"].abs().to_numpy()

    # Colour carries the *outcome* -- whether the verdict survived the change of
    # selection rule -- so the figure's finding is visible before any annotation
    # is read. The two rules themselves stay distinguishable by shape (filled
    # versus hollow), which leaves colour free for the judgement and keeps the
    # encoding readable in greyscale.
    palette = epoch_outcome_colors()
    outcomes = [
        classify_epoch_outcome(bool(sf), bool(sb))
        for sf, sb in zip(d["significant_final"], d["significant_best"], strict=True)
    ]
    row_colors = [palette[o] for o in outcomes]

    for i, (x0, x1) in enumerate(zip(a, b, strict=False)):
        ax.plot([x0, x1], [i, i], color=row_colors[i], linewidth=1.4, alpha=0.55, zorder=1)
    ax.scatter(a, y, s=24, color=row_colors, zorder=3, label="fixed final epoch")
    ax.scatter(
        b,
        y,
        s=24,
        facecolor="white",
        edgecolor=row_colors,
        linewidth=1.1,
        zorder=3,
        label="best validation epoch",
    )
    for i, outcome in enumerate(outcomes):
        if outcome == "inflated":
            ax.text(max(a[i], b[i]) + 0.002, i, "flips", va="center", fontsize=7, color=palette["inflated"])
    ax.set_yticks(y)
    ax.set_yticklabels(d["label"], fontsize=7.5)
    for tick, color in zip(ax.get_yticklabels(), row_colors, strict=True):
        tick.set_color(color)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel(r"$|\Delta\mu|$ (metric units)")
    ax.set_axisbelow(True)
    ax.xaxis.grid(True)

    # Two legends: shape says which rule, colour says what happened to the verdict.
    rule_handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=5, color="0.25", label="fixed final epoch"),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markersize=5,
            markerfacecolor="white",
            markeredgecolor="0.25",
            label="best validation epoch",
        ),
    ]
    outcome_handles = [
        Line2D([], [], color=palette[o], linewidth=2.4, label=EPOCH_OUTCOME_LABELS[o]) for o in EPOCH_OUTCOMES
    ]
    # The outcome legend sits above the axes; the rule legend goes inside, at
    # lower right, where the short-|dmu| rows leave the panel empty. Stacking
    # both above the axes put one of them outside the canvas that
    # ``tight_layout`` keeps.
    rules = ax.legend(
        handles=rule_handles,
        loc="lower right",
        frameon=True,
        framealpha=0.9,
        edgecolor="0.85",
        fontsize=7.5,
        handletextpad=0.4,
        borderpad=0.6,
    )
    ax.add_artist(rules)
    ax.legend(
        handles=outcome_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=3,
        frameon=False,
        handletextpad=0.5,
        fontsize=7.5,
    )
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Aggregation: is the conclusion an artefact of equal task weighting?
# --------------------------------------------------------------------------- #


def emit_aggregation(
    report: BenchmarkReport,
    *,
    tab_dir: Path,
    groups: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Mean ranks under uniform, reliability and hierarchical weighting.

    Only the uniform column has a Friedman/Nemenyi null; the others are
    diagnostics, and the table says so.
    """
    variants: list[tuple[str, dict[str, Any]]] = [
        ("uniform", {"aggregation": "uniform"}),
        ("reliability", {"aggregation": "reliability"}),
    ]
    if groups:
        variants.append(("hierarchical", {"aggregation": "hierarchical", "groups": groups}))

    series = {name: report.mean_ranks(**kw) for name, kw in variants}
    models = list(series["uniform"].index)
    rows = [[model, *[f"{series[name][model]:.2f}" for name, _ in variants]] for model in models]
    gaps = ["Max gap", *[f"{s.max() - s.min():.2f}" for s in series.values()]]
    orders = [
        "Order preserved",
        r"\textbf{--}",
        *[
            (_latex._sig("yes") if list(series[name].index) == models else _latex._quiet("no"))
            for name, _ in variants[1:]
        ],
    ]
    _latex.booktabs(
        tab_dir / "aggregation.tex",
        ["Model", *[n.capitalize() for n, _ in variants]],
        [*rows, gaps, orders],
        align="l" + "c" * len(variants),
        rules_before={len(rows)},
    )
    return {name: {m: round(float(v), 4) for m, v in s.items()} for name, s in series.items()} | {
        "order_preserved": {name: list(series[name].index) == models for name, _ in variants[1:]}
    }


# --------------------------------------------------------------------------- #
# Rank stability: how much does the conclusion depend on which tasks ran?
# --------------------------------------------------------------------------- #


def emit_stability(
    report: BenchmarkReport,
    *,
    tab_dir: Path,
    fig_dir: Path,
    artifact_dir: Path,
    n_boot: int = 10_000,
    formats: tuple[str, ...] = ("png",),
) -> dict[str, Any]:
    st = report.rank_stability(n_boot=n_boot, alpha=ALPHA, seed=0)
    if st.get("mean_ranks") is None or len(st["mean_ranks"]) == 0:
        return {}
    sep: pd.DataFrame = st["separation"]
    jack: pd.DataFrame = st["jackknife"]
    _latex.summary_csv(artifact_dir / "rank_stability_jackknife.csv", jack.set_index(["dropped_task", "model"]))

    rows = [
        [f"{a} vs {b}", f"{g:.2f}", f"{f * 100:.1f}\\%"]
        for a, b, g, f in zip(sep["model_a"], sep["model_b"], sep["observed_gap"], sep["separates_frac"], strict=False)
    ]
    _latex.booktabs(
        tab_dir / "rank_stability.tex",
        ["Comparison", "Observed rank gap", r"Separates in \% of task resamples"],
        rows,
        align="lcc",
    )

    fig = _plot_stability(st)
    save_figure(fig, fig_dir / "rank_stability", formats=formats)
    plt.close(fig)

    return {
        "n_boot": n_boot,
        "order": st["order"],
        "order_stability": round(float(st["order_stability"]), 4),
        "gap_observed": round(float(st["gap_observed"]), 4),
        "critical_difference": round(float(st["critical_difference"]), 4),
        "gap_quantiles": {str(k): round(v, 4) for k, v in st["gap_quantiles"].items()},
        "gap_exceeds_cd_frac": round(float(st["gap_exceeds_cd_frac"]), 4),
        "max_jackknife_shift": round(float(jack["shift"].abs().max()), 4),
        "separation": {
            f"{a} vs {b}": round(float(f), 4)
            for a, b, f in zip(sep["model_a"], sep["model_b"], sep["separates_frac"], strict=False)
        },
    }


def _plot_stability(st: dict[str, Any]) -> plt.Figure:
    """Left: bootstrap distribution of the max rank gap. Right: jackknife shifts."""
    from graphnetz.plotting import COLUMN_INCHES, panel_label, set_plot_style

    set_plot_style()
    jack: pd.DataFrame = st["jackknife"]
    models = list(st["mean_ranks"].index)
    fig, axes = plt.subplots(1, 2, figsize=(COLUMN_INCHES["double"] * 0.82, 2.6))

    ax = axes[0]
    quant = st["gap_quantiles"]
    ax.axvspan(
        quant[0.025], quant[0.975], color=PAPER_COLORS[2], alpha=0.35, linewidth=0, label="95\\% of task resamples"
    )
    ax.axvline(st["gap_observed"], color=PAPER_COLORS[0], linewidth=1.6, label="observed gap")
    ax.axvline(
        st["critical_difference"],
        color=PAPER_COLORS[1],
        linewidth=1.2,
        linestyle="--",
        label=rf"$CD_{{0.05}}={st['critical_difference']:.2f}$",
    )
    ax.set_xlabel("Max mean-rank gap")
    ax.set_yticks([])
    ax.set_title(f"Rank order stable in {st['order_stability'] * 100:.0f}\\% of resamples", loc="left")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    panel_label(ax, "a", x=-0.06, y=1.10)

    ax = axes[1]
    for j, model in enumerate(models):
        sub = jack[jack["model"] == model]
        ax.scatter(sub["shift"], np.full(len(sub), j), s=16, color=PAPER_COLORS[j % len(PAPER_COLORS)], alpha=0.85)
    ax.axvline(0.0, color="0.5", linewidth=0.7, linestyle=":")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=8)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Mean-rank shift when one task is dropped")
    ax.set_axisbelow(True)
    ax.xaxis.grid(True)
    ax.set_title("Leave-one-task-out influence", loc="left")
    panel_label(ax, "b", x=-0.22, y=1.10)

    fig.tight_layout(w_pad=2.0)
    return fig


# --------------------------------------------------------------------------- #
# Stage entry point
# --------------------------------------------------------------------------- #


def emit(
    *,
    breadth: BenchmarkReport | None,
    planetoid: BenchmarkReport | None,
    fig_dir: Path,
    tab_dir: Path,
    artifact_dir: Path,
    formats: tuple[str, ...] = ("png",),
    n_boot: int = 10_000,
) -> dict[str, Any]:
    """Run every revision analysis for whichever reports are available."""
    out: dict[str, Any] = {"seeds": len(SEEDS), "sesoi": SESOI}
    pretty_slot = {s.task_name: f"{s.category_label}/{s.dataset_label}" for s in BREADTH_SLOTS}
    pretty_slot |= {s.key: f"{s.category_label}/{s.dataset_label}" for s in BREADTH_SLOTS}
    groups = {s.key: s.category for s in BREADTH_SLOTS}

    if planetoid is not None:
        from paper.experiments._config import PLANETOID_PRETTY

        print("  [planetoid] epoch selection, power, equivalence")
        out["planetoid"] = {
            "epoch_selection": emit_epoch_selection(
                planetoid,
                tab_dir=tab_dir,
                fig_dir=fig_dir,
                artifact_dir=artifact_dir,
                pretty_tasks=PLANETOID_PRETTY,
                formats=formats,
            ),
            "power": emit_power(planetoid, tab_dir=tab_dir, artifact_dir=artifact_dir, pretty_tasks=PLANETOID_PRETTY),
            "equivalence": emit_equivalence(
                planetoid, tab_dir=tab_dir, artifact_dir=artifact_dir, pretty_tasks=PLANETOID_PRETTY
            ),
        }
        e = out["planetoid"]["epoch_selection"]
        print(
            f"    epoch selection: {e['significant_final']} -> {e['significant_best_val']} "
            f"significant of {e['n_comparisons']} ({e['flipped']} flipped)"
        )
        p = out["planetoid"]["power"]
        print(f"    median MDE {p['median_mde']:.4f}; {p['n_undetectable']}/{p['n_comparisons']} undetectable")
        q = out["planetoid"]["equivalence"]
        print(
            f"    verdicts at margin {SESOI}: {q['different']} different, "
            f"{q['equivalent']} equivalent, {q['undetermined']} undetermined"
        )

    if breadth is not None:
        print("  [breadth] capacity, stability, aggregation")
        cap = emit_capacity(breadth, tab_dir=tab_dir)
        cap.pop("curve", None)
        out["breadth"] = {
            "capacity": cap,
            "stability": emit_stability(
                breadth,
                tab_dir=tab_dir,
                fig_dir=fig_dir,
                artifact_dir=artifact_dir,
                n_boot=n_boot,
                formats=formats,
            ),
            "aggregation": emit_aggregation(breadth, tab_dir=tab_dir, groups=groups),
        }
        print(
            f"    CD(k={cap['k']}, N={cap['n_current']}) = {cap['cd_current']:.3f}; "
            f"observed gap {cap['max_rank_gap']:.2f} would need N >= {cap['n_for_observed_gap']}"
        )
        st = out["breadth"]["stability"]
        if st:
            print(
                f"    rank order stable in {st['order_stability'] * 100:.0f}% of task resamples; "
                f"max jackknife shift {st['max_jackknife_shift']:.2f}"
            )

    (artifact_dir / "revision_analyses.json").write_text(json.dumps(out, indent=2, default=str) + "\n")
    return out
