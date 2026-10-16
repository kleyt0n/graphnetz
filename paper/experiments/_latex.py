"""LaTeX emitters for the hand-shaped tables in ``main.tex``.

Three of the paper's tables are not plain ``booktabs`` dumps and therefore
cannot come from ``BenchmarkReport.to_latex`` / ``pairwise_to_latex``:

``tables/breadth.tex``
    Table 3 -- category x architecture, cells shaded ``winnerc`` (best),
    ``tiec`` (CI overlaps the best) or unshaded, with ``n/a`` for
    incompatible (model, task) pairs.
``tables/pairwise_compare.tex``
    Table 2 -- paired-$t$ and Wilcoxon Holm-adjusted $p$-values side by side,
    shaded by significance, with an agreement column.
``tables/categories.tex``
    Table 4 -- the dataset catalogue, grouped ``\\multirow`` blocks per
    research category.

The colour names used here (``winnerc``, ``tiec``, ``paperink``,
``paperaccent``, ``nac``) are declared in the preamble of ``main.tex``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from paper.experiments._config import (
    ALPHA,
    CATEGORY_LABELS,
    LOWER_IS_BETTER,
    METRIC_LABELS,
    MODEL_HEADERS,
    MODEL_NAMES,
    PRETTY_LOADERS,
    TASK_ABBREV,
    TASK_ORDER,
    Cell,
)


def _write(path: Path, lines: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


def _tex_escape(text: str) -> str:
    return text.replace("_", r"\_")


# --------------------------------------------------------------------------- #
# Table 3: the cross-category results table
# --------------------------------------------------------------------------- #

_NA_CELL = r"{\color{nac}\textit{n/a}}"


def _stat_cell(mean: float, half: float, shade: str) -> str:
    """One ``$x_{\\pm h}$`` cell, shaded by its relation to the row winner."""
    body = f"${mean:.3f}_{{\\pm {half:.3f}}}$"
    if shade == "winner":
        return rf"\cellcolor{{winnerc}}{{\color{{paperink}}\textbf{{{body}}}}}"
    if shade == "tie":
        return rf"\cellcolor{{tiec}}{{\color{{paperaccent}}{body}}}"
    return body


def breadth_table(path: Path, rows: list[Cell]) -> Path:
    """Emit Table 3, one row per :class:`~paper.experiments._config.Cell`.

    Shading is decided here: the best mean by the metric's direction is
    ``winnerc``, any model whose 95 % CI overlaps the winner's is ``tiec``
    (statistically indistinguishable at this sample size), the rest are
    unshaded, and models the dispatcher skipped are ``n/a``.
    """
    header_models = [MODEL_HEADERS[m] for m in MODEL_NAMES]
    lines = [
        r"\begin{tabular}{lll" + "c" * (len(MODEL_NAMES) + 1) + "}",
        r"\toprule",
        "Category & Dataset & Kind & Metric & " + " & ".join(header_models) + r" \\",
        r"\midrule",
    ]
    for cell in rows:
        slot, metric, stats = cell.slot, cell.metric, cell.stats
        lower_better = metric in LOWER_IS_BETTER
        ranked = sorted(stats.items(), key=lambda kv: kv[1][0], reverse=not lower_better)
        winner = ranked[0][0]
        w_mean, w_half = stats[winner]
        w_lo, w_hi = w_mean - w_half, w_mean + w_half

        cells: list[str] = []
        for model in MODEL_NAMES:
            if model not in stats:
                cells.append(_NA_CELL)
                continue
            mean, half = stats[model]
            if model == winner:
                shade = "winner"
            elif (mean - half) <= w_hi and (mean + half) >= w_lo:
                shade = "tie"
            else:
                shade = "plain"
            cells.append(_stat_cell(mean, half, shade))

        lines.append(
            f"{slot.category_label} & {slot.dataset_label} & "
            f"{_tex_escape(slot.task_type)} & {METRIC_LABELS[metric]} & " + " & ".join(cells) + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return _write(path, lines)


# --------------------------------------------------------------------------- #
# Table 2: paired-t vs Wilcoxon, side by side
# --------------------------------------------------------------------------- #


def _p_cell(p: float, alpha: float) -> str:
    if pd.isna(p):
        return _NA_CELL
    body = f"{p:#.3g}"  # three significant digits, trailing zeros kept: 0.110
    if p < alpha:
        return rf"\cellcolor{{winnerc}}{{\color{{paperink}}\textbf{{{body}}}}}"
    return rf"\cellcolor{{tiec}}{{\color{{paperaccent}}{body}}}"


def pairwise_compare_table(
    path: Path,
    t_df: pd.DataFrame,
    w_df: pd.DataFrame,
    pretty_tasks: dict[str, str],
    alpha: float = ALPHA,
) -> Path:
    """Emit Table 2 by joining the paired-$t$ and Wilcoxon pairwise frames.

    Both frames come from ``BenchmarkReport.pairwise`` (same task order, same
    comparison order), so they are joined on (task, model_a, model_b).  The
    ``Agree`` column flags comparisons where the two tests reach the same
    verdict at ``alpha``.
    """
    key = ["task", "model_a", "model_b"]
    merged = t_df.merge(w_df, on=key, suffixes=("_t", "_w"), validate="one_to_one")
    lines = [
        r"\begin{tabular}{l l c c c c}",
        r"\toprule",
        r"Task & Comparison & $\Delta\mu$ & $p_{\text{Holm}}^{\,t}$ & "
        r"$p_{\text{Holm}}^{\,W}$ & Agree \\",
        r"\midrule",
    ]
    tasks = list(dict.fromkeys(merged["task"]))
    for t_idx, task in enumerate(tasks):
        block = merged[merged["task"] == task]
        label = pretty_tasks.get(task, task)
        for i, (_, row) in enumerate(block.iterrows()):
            head = rf"\multirow{{{len(block)}}}{{*}}{{\textbf{{{label}}}}}" if i == 0 else ""
            sig_t = (not pd.isna(row["p_holm_t"])) and row["p_holm_t"] < alpha
            sig_w = (not pd.isna(row["p_holm_w"])) and row["p_holm_w"] < alpha
            agree = r"{\color{paperink}\checkmark}" if sig_t == sig_w else r"{\color{paperaccent}$\times$}"
            lines.append(
                f"{head} & {row['model_a']} \\textsc{{vs.}} {row['model_b']} & "
                f"${row['mean_diff_t']:+.3f}$ & {_p_cell(row['p_holm_t'], alpha)} & "
                f"{_p_cell(row['p_holm_w'], alpha)} & {agree} \\\\"
            )
        if t_idx < len(tasks) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return _write(path, lines)


# --------------------------------------------------------------------------- #
# Table 4: the dataset catalogue
# --------------------------------------------------------------------------- #


def categories_table(path: Path, registry: dict[str, dict[str, list[str]]]) -> Path:
    """Emit Table 4 from ``graphnetz.datasets.list_datasets()``.

    The ``#`` column counts *distinct* loaders per category: a loader serving
    two task types (Cora is both ``node_cls`` and ``link_pred``) is listed on
    both rows but counted once, which is how the catalogue totals 62.
    """
    lines = [
        r"\begin{tabular}{l r l >{\raggedright\arraybackslash}p{0.62\linewidth}}",
        r"\toprule",
        r"Category & \# & Kind & Loaders \\",
        r"\midrule",
    ]
    grand_total = 0
    for category, per_task in registry.items():
        present = [t for t in TASK_ORDER if per_task.get(t)]
        if not present:
            continue
        unique = {name for t in present for name in per_task[t]}
        grand_total += len(unique)
        span = len(present)
        for i, task_type in enumerate(present):
            names = ", ".join(PRETTY_LOADERS.get(n, _tex_escape(n)) for n in per_task[task_type])
            kind = rf"{{\color{{paperink}}\textbf{{{TASK_ABBREV[task_type]}}}}}"
            if i == 0:
                head = (
                    rf"\multirow{{{span}}}{{*}}{{{CATEGORY_LABELS.get(category, category.title())}}} & "
                    rf"\multirow{{{span}}}{{*}}{{{len(unique)}}}"
                )
            else:
                head = " & "
            lines.append(f"{head} & {kind} & {names} \\\\")
        lines.append(r"\midrule")
    lines += [
        rf"\textbf{{Total}} & \textbf{{{grand_total}}} & & \\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    return _write(path, lines)


# --------------------------------------------------------------------------- #
# Plain-text companion, for the record
# --------------------------------------------------------------------------- #


def summary_csv(path: Path, frame: pd.DataFrame) -> Path:
    """Dump a stats frame next to the LaTeX so numbers can be diffed by eye."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, float_format="%.6f")
    return path


def rank_table(finals: dict[str, dict[str, list[float]]], metrics: dict[str, str]) -> pd.DataFrame:
    """Per-task ranks of seed-mean metrics, with the direction applied.

    Reproduces the rank table behind Figure 3 (and the Friedman statistic) in
    a form that can be printed and inspected: rows are tasks, columns models,
    rank 1 = best.  Only models present in every task are kept, which is the
    ``k = 4`` node-level slice the paper reports.
    """
    from scipy import stats as sps

    common = sorted(set.intersection(*[set(per) for per in finals.values()]))
    tasks = list(finals)
    rows = []
    for task in tasks:
        means = np.array([float(np.mean(finals[task][m])) for m in common])
        sign = 1.0 if metrics[task] in LOWER_IS_BETTER else -1.0
        rows.append(sps.rankdata(sign * means, method="average"))
    return pd.DataFrame(rows, index=tasks, columns=common)


# --------------------------------------------------------------------------- #
# Generic booktabs emitter, for the tables added in revision
# --------------------------------------------------------------------------- #


def booktabs(
    path: Path,
    header: list[str],
    rows: list[list[str]],
    *,
    align: str | None = None,
    rules_before: set[int] | None = None,
) -> Path:
    """Write a plain ``booktabs`` table body (no float wrapper).

    ``align`` defaults to left for the first column and centred for the rest.
    ``rules_before`` inserts a ``\\midrule`` before the given 0-based row
    indices, which is how the summary line at the bottom of a count table is
    separated from the per-task lines.
    """
    spec = align or ("l" + "c" * (len(header) - 1))
    lines = [rf"\begin{{tabular}}{{{spec}}}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    for i, row in enumerate(rows):
        if rules_before and i in rules_before:
            lines.append(r"\midrule")
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return _write(path, lines)


def _sig(value: str) -> str:
    """Shade a cell as significant, matching the palette of Table 2/3."""
    return rf"\cellcolor{{winnerc}}{{\color{{paperink}}\textbf{{{value}}}}}"


def _quiet(value: str) -> str:
    """Shade a cell as non-significant / undetermined."""
    return rf"\cellcolor{{tiec}}{{\color{{paperaccent}}{value}}}"


VERDICT_CELL = {
    "different": lambda: _sig("different"),
    "equivalent": lambda: r"\cellcolor{tiec}{\color{paperink}\textbf{equivalent}}",
    "trivial": lambda: _quiet("trivial"),
    "undetermined": lambda: r"{\color{nac}\textit{undetermined}}",
}
"""Verdict -> shaded LaTeX cell, so the three-way outcome reads at a glance."""
