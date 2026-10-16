"""Compare a fresh run against the numbers published in ``main.tex``.

``expected.json`` holds every value the paper prints for the cross-category
sweep (Table 3), the across-task aggregation (Section 3.4) and the Planetoid
demo (results.tex), transcribed from the committed tables.  This stage recomputes
those quantities from the cache and reports the deltas, so "reproduces the
paper" is a checkable claim rather than an assertion.

The tolerances are loose on purpose.  Training is deterministic given a fixed
software stack (Appendix E), but the stack moves: a different BLAS, a
different ``torch`` minor version, or a CUDA rather than CPU runner will
shift a metric in the third decimal without anything being wrong.  A drift
past ``--tol`` is a signal to investigate, not proof of a bug.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from paper.experiments._config import Cell

EXPECTED_PATH = Path(__file__).with_name("expected.json")

DEFAULT_TOL = 0.02
"""Absolute tolerance on a metric mean, in metric units."""

RANK_TOL = 0.25
"""Absolute tolerance on a mean rank (one task flipping a rank moves it 0.1)."""


def load_expected() -> dict[str, Any]:
    return json.loads(EXPECTED_PATH.read_text())


def _check(label: str, got: float, want: float, tol: float, out: list[tuple[bool, str]]) -> None:
    delta = got - want
    ok = abs(delta) <= tol
    out.append((ok, f"{'ok  ' if ok else 'DIFF'} {label:<48} got {got:+.4f}  paper {want:+.4f}  Δ {delta:+.4f}"))


def check_breadth(rows: list[Cell], expected: dict[str, Any]) -> list[tuple[bool, str]]:
    """Compare each Table-3 cell mean against the published value."""
    out: list[tuple[bool, str]] = []
    published = {(r["category"], r["dataset"]): r["cells"] for r in expected["breadth"]}
    for cell in rows:
        slot = cell.slot
        key = (slot.category_label, slot.dataset_label)
        if key not in published:
            out.append((False, f"DIFF {slot.key:<48} not present in expected.json"))
            continue
        for model, want in published[key].items():
            got = cell.stats.get(model)
            if want is None:
                if got is not None:
                    out.append((False, f"DIFF {slot.key}/{model:<20} paper reports n/a but a value was produced"))
                continue
            if got is None:
                out.append((False, f"DIFF {slot.key}/{model:<20} missing (paper: {want[0]:.3f})"))
                continue
            _check(f"{slot.key}/{model}", got[0], want[0], DEFAULT_TOL, out)
    return out


def check_headline(stats: dict[str, Any], expected: dict[str, Any]) -> list[tuple[bool, str]]:
    """Compare the across-task aggregation quoted in Section 3.4."""
    out: list[tuple[bool, str]] = []
    want = expected["headline"]
    for field in ("k", "n"):
        got_i, want_i = int(stats[field]), int(want[field])
        ok = got_i == want_i
        out.append((ok, f"{'ok  ' if ok else 'DIFF'} {field:<48} got {got_i}      paper {want_i}"))
    # CD depends only on (k, N), so it must match exactly once k and N do.
    _check("critical_difference", float(stats["critical_difference"]), float(want["critical_difference"]), 0.005, out)
    _check("friedman_chi2", float(stats["friedman_chi2"]), float(want["friedman_chi2"]), 0.5, out)
    for model, rank in want["mean_ranks"].items():
        got = stats["mean_ranks"].get(model)
        if got is None:
            out.append((False, f"DIFF mean_rank/{model:<38} absent from the CD rank table"))
            continue
        _check(f"mean_rank/{model}", float(got), float(rank), RANK_TOL, out)
    ok = bool(stats["single_clique"]) == bool(want["single_clique"])
    out.append((ok, f"{'ok  ' if ok else 'DIFF'} {'single_nemenyi_clique':<48} got {stats['single_clique']}"))
    return out


def check_planetoid(report: Any, expected: dict[str, Any], alpha: float) -> list[tuple[bool, str]]:
    """Compare the Planetoid means and the paired-test agreement of Table 2."""
    out: list[tuple[bool, str]] = []
    want = expected["planetoid"]
    frame = report.summary(ci=expected["protocol"]["ci"])
    for task, per_model in want["tasks"].items():
        for model, cell in per_model.items():
            if (task, model) not in frame.index:
                out.append((False, f"DIFF {task}/{model:<40} missing from the report"))
                continue
            _check(f"{task}/{model}", float(frame.loc[(task, model), "mean"]), cell[0], DEFAULT_TOL, out)

    t_df = report.pairwise(alpha=alpha, method="t")
    w_df = report.pairwise(alpha=alpha, method="wilcoxon")
    for label, got, key in (
        ("n_significant_t", int(t_df["significant"].sum()), "n_significant_t"),
        ("n_significant_wilcoxon", int(w_df["significant"].sum()), "n_significant_wilcoxon"),
        (
            "n_disagreements",
            int((t_df["significant"].to_numpy() != w_df["significant"].to_numpy()).sum()),
            "n_disagreements",
        ),
    ):
        ok = got == int(want[key])
        out.append((ok, f"{'ok  ' if ok else 'DIFF'} {label:<48} got {got}      paper {want[key]}"))
    return out


def report(results: list[tuple[bool, str]], *, quiet_ok: bool = True) -> bool:
    """Print ``results`` and return True when everything is within tolerance."""
    failures = [line for ok, line in results if not ok]
    if not quiet_ok:
        for _, line in results:
            print(f"  {line}")
    else:
        for line in failures:
            print(f"  {line}")
    n_ok = len(results) - len(failures)
    print(f"  {n_ok}/{len(results)} checks within tolerance")
    return not failures
