"""Section 3.5: sensitivity of the within-task pairwise test on Planetoid.

Runs the four node-level encoders on Cora / CiteSeer / PubMed over ten seeds
and reports the same 18 comparisons under two tests -- paired Student's $t$
and Wilcoxon signed-rank -- both Holm--Bonferroni corrected.  This is also
the demo sweep the paper's illustrative figures are drawn from.

    figures/benchmark_bars.png          -- grouped bars, mean +- 95 % t-CI
    figures/forest_plot.png             -- the same cells as a forest plot
    figures/learning_curves.png         -- per-epoch mean +- t-CI
    figures/critical_difference.png     -- CD diagram at k = 4, N = 3
    figures/pairwise_diff.png           -- Holm p / effect-size matrices (t)
    figures/pairwise_diff_wilcoxon.png  -- the same under Wilcoxon
    tables/results.tex                  -- mean +- t-CI per (model, dataset)
    tables/pairwise.tex                 -- Holm-corrected paired t
    tables/pairwise_wilcoxon.tex        -- Holm-corrected Wilcoxon
    tables/pairwise_compare.tex         -- Table 2, the two tests side by side
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from paper.experiments import _latex, _runs
from paper.experiments._config import (
    ALPHA,
    CI,
    PLANETOID_MODEL_NAMES,
    PLANETOID_PRETTY,
    PLANETOID_TASK_NAMES,
    SEEDS,
)
from paper.experiments._style import style_cd_axes

from graphnetz import BenchmarkReport, Task
from graphnetz.benchmark import BENCHMARK_TASKS

CATEGORY = "social"
TASK_TYPE = "node_cls"


def _tasks() -> list[Task]:
    """The curated ``social``/``node_cls`` tasks named in :data:`PLANETOID_TASK_NAMES`."""
    curated = {t.name: t for t in BENCHMARK_TASKS[CATEGORY][TASK_TYPE]}
    missing = [n for n in PLANETOID_TASK_NAMES if n not in curated]
    if missing:
        msg = f"missing curated social/node_cls tasks: {missing}"
        raise KeyError(msg)
    return [curated[n] for n in PLANETOID_TASK_NAMES]


def train(
    *,
    seeds: tuple[int, ...] = SEEDS,
    data_root: str,
    cache_dir: Path,
    epochs: int | None = None,
    force: bool = False,
    verbose: bool = True,
    cache_only: bool = False,
    chunk: int = 2,
    max_seconds: float | None = None,
) -> BenchmarkReport | None:
    """Train (or load) the three-dataset demo as a single report.

    All three tasks live in one report -- not one per task -- because the
    Planetoid CD diagram (k = 4, N = 3) and the shared learning-curve figure
    need them side by side.
    """
    return _runs.train_cell(
        label="social/planetoid",
        tasks=_tasks(),
        category=CATEGORY,
        model_names=PLANETOID_MODEL_NAMES,
        seeds=seeds,
        data_root=data_root,
        cache_path=cache_dir / "planetoid.json",
        epochs=epochs,
        force=force,
        verbose=verbose,
        cache_only=cache_only,
        chunk=chunk,
        max_seconds=max_seconds,
    )


def emit(
    report: BenchmarkReport,
    *,
    fig_dir: Path,
    tab_dir: Path,
    artifact_dir: Path,
    formats: tuple[str, ...] = ("png",),
) -> None:
    """Write every Section-3.5 artefact."""
    from graphnetz import save_figure

    pretty = PLANETOID_PRETTY

    fig, _ = report.plot(pretty_tasks=pretty, ci=CI)
    save_figure(fig, fig_dir / "benchmark_bars", formats=formats)
    plt.close(fig)

    fig, _ = report.plot_forest(pretty_tasks=pretty, ci=CI)
    save_figure(fig, fig_dir / "forest_plot", formats=formats)
    plt.close(fig)

    fig, _ = report.plot_learning_curves(pretty_tasks=pretty, ci=CI)
    save_figure(fig, fig_dir / "learning_curves", formats=formats)
    plt.close(fig)

    fig, ax = report.plot_critical_difference(alpha=ALPHA)
    style_cd_axes(ax)
    save_figure(fig, fig_dir / "critical_difference", formats=formats)
    plt.close(fig)

    for method, stem in (("t", "pairwise_diff"), ("wilcoxon", "pairwise_diff_wilcoxon")):
        fig, _ = report.plot_pairwise(layout="matrix", pretty_tasks=pretty, alpha=ALPHA, method=method)
        save_figure(fig, fig_dir / stem, formats=formats)
        plt.close(fig)

    report.to_latex(tab_dir / "results.tex", ci=CI, pretty_tasks=pretty)
    report.pairwise_to_latex(
        tab_dir / "pairwise.tex",
        alpha=ALPHA,
        method="t",
        caption=r"Per-task pairwise paired-$t$ tests with Holm--Bonferroni correction.",
        label="tab:pairwise_t",
    )
    report.pairwise_to_latex(
        tab_dir / "pairwise_wilcoxon.tex",
        alpha=ALPHA,
        method="wilcoxon",
        caption="Per-task pairwise Wilcoxon signed-rank tests with Holm--Bonferroni correction.",
        label="tab:pairwise_wilcoxon",
    )

    t_df = report.pairwise(alpha=ALPHA, method="t")
    w_df = report.pairwise(alpha=ALPHA, method="wilcoxon")
    _latex.pairwise_compare_table(tab_dir / "pairwise_compare.tex", t_df, w_df, pretty, alpha=ALPHA)
    _latex.summary_csv(artifact_dir / "planetoid_summary.csv", report.summary(ci=CI))

    _report_agreement(report, t_df, w_df)


def _report_agreement(report: BenchmarkReport, t_df: pd.DataFrame, w_df: pd.DataFrame) -> None:
    """Print the Section-3.5 claims so they can be checked against the prose."""
    n_sig_t = int(t_df["significant"].sum())
    n_sig_w = int(w_df["significant"].sum())
    disagree = int((t_df["significant"].to_numpy() != w_df["significant"].to_numpy()).sum())
    s = len(report.seeds)
    floor = 2.0 * 2.0**-s
    per_task = len(t_df) // max(len(report.histories), 1)
    print(f"  comparisons: {len(t_df)} ({per_task} per task, Holm family m = {per_task})")
    print(f"  significant at alpha={ALPHA}: paired-t {n_sig_t}, Wilcoxon {n_sig_w}, disagreements {disagree}")
    print(f"  smallest exact two-sided Wilcoxon p at S={s}: {floor:.5f} -> Holm x {per_task} = {floor * per_task:.4f}")
