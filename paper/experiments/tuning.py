"""A bounded hyperparameter-search study on one category.

Reviewer HVs6 (W3/Q2) objects that the comparison may be unfair because no
architecture receives model-specific tuning, and reviewer 6FbE (W2) raises the
same tension with \\citet{errica2020fair}.  The framework's answer is structural
--- ``run_benchmark(search=...)`` adds the inner loop --- but a mechanism without
a measurement is still a promise, so this stage runs the loop on one category
and reports whether the within-task verdicts change.

Scope is deliberately bounded and stated: the social node-classification trio,
a small grid over the optimiser, and the same ten seeds.  A full tuned sweep
across all ten categories multiplies compute by the grid size and is explicitly
*not* claimed.

    tables/tuning.tex   -- default vs searched means, and the verdict changes
    _artifacts/tuning.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from paper.experiments import _latex, _runs
from paper.experiments._config import ALPHA, CI, PLANETOID_MODEL_NAMES, PLANETOID_PRETTY, SEEDS

from graphnetz import SearchSpace

CATEGORY = "social"
TASK_TYPE = "node_cls"

GRID = SearchSpace(lr=(1e-2, 1e-3), weight_decay=(5e-4, 0.0))
"""A two-axis optimiser grid: four candidates per (task, model, seed).

Deliberately small. The point is not to find the best configuration for each
architecture but to test whether *any* per-model tuning moves the within-task
verdicts -- which is what the reviewers asked. The axes chosen are the two the
literature's default recipes actually disagree on between GCN-style and
attention-style models.
"""


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
) -> Any:
    """Train the searched variant of the Planetoid trio (or load it from cache)."""
    from paper.experiments.planetoid import _tasks

    return _runs.train_cell(
        label=f"social/planetoid+search({len(GRID)})",
        tasks=_tasks(),
        category=CATEGORY,
        model_names=PLANETOID_MODEL_NAMES,
        seeds=seeds,
        data_root=data_root,
        cache_path=cache_dir / "planetoid_tuned.json",
        epochs=epochs,
        force=force,
        verbose=verbose,
        cache_only=cache_only,
        chunk=chunk,
        max_seconds=max_seconds,
        search=GRID,
    )


def emit(default: Any, tuned: Any, *, tab_dir: Path, artifact_dir: Path) -> dict[str, Any]:
    """Compare the default and searched protocols on the same tasks and seeds."""
    d_sum, t_sum = default.summary(ci=CI), tuned.summary(ci=CI)
    d_pw, t_pw = default.pairwise(alpha=ALPHA), tuned.pairwise(alpha=ALPHA)

    rows: list[list[str]] = []
    gains: dict[str, list[float]] = {}
    for (task, model), r in d_sum.iterrows():
        if (task, model) not in t_sum.index:
            continue
        after = float(t_sum.loc[(task, model), "mean"])
        delta = after - float(r["mean"])
        gains.setdefault(model, []).append(delta)
        rows.append(
            [
                PLANETOID_PRETTY.get(str(task), str(task)),
                str(model),
                f"{float(r['mean']):.4f}",
                f"{after:.4f}",
                (_latex._sig(f"{delta:+.4f}") if delta > 0.005 else f"{delta:+.4f}"),
            ]
        )
    summary_rows = [
        [
            r"\textbf{Mean gain}",
            model,
            "",
            "",
            rf"\textbf{{{sum(v) / len(v):+.4f}}}",
        ]
        for model, v in sorted(gains.items(), key=lambda kv: -sum(kv[1]) / len(kv[1]))
    ]
    _latex.booktabs(
        tab_dir / "tuning.tex",
        ["Task", "Model", "Default", "Searched", r"$\Delta$"],
        [*rows, *summary_rows],
        align="llccc",
        rules_before={len(rows)},
    )

    key = ["task", "model_a", "model_b"]
    merged = d_pw.merge(t_pw, on=key, suffixes=("_d", "_t"), validate="one_to_one")
    flipped = merged[merged["significant_d"] != merged["significant_t"]]
    out: dict[str, Any] = {
        "grid": GRID.describe(),
        "n_candidates": len(GRID),
        "mean_gain": {m: round(sum(v) / len(v), 4) for m, v in gains.items()},
        "significant_default": int(merged["significant_d"].sum()),
        "significant_searched": int(merged["significant_t"].sum()),
        "flipped": len(flipped),
        "flipped_comparisons": [f"{r['task']}: {r['model_a']} vs {r['model_b']}" for _, r in flipped.iterrows()],
        "selected": tuned.config.get("search_selected", {}),
    }
    (artifact_dir / "tuning.json").write_text(json.dumps(out, indent=2) + "\n")
    best = max(out["mean_gain"].items(), key=lambda kv: kv[1])
    print(f"  grid: {GRID.describe()}")
    print("  mean gain from tuning: " + ", ".join(f"{m} {g:+.4f}" for m, g in out["mean_gain"].items()))
    print(f"  largest beneficiary: {best[0]} ({best[1]:+.4f})")
    print(
        f"  significant comparisons: {out['significant_default']} default -> "
        f"{out['significant_searched']} searched ({out['flipped']} flipped)"
    )
    return out
