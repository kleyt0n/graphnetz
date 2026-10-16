"""Audit the dataset catalogue: does every advertised loader actually load?

The paper claims 62 loaders across ten research areas.  Reviewer 6FbE asked
whether that number is verifiable and how many are validated end to end;
reviewer HVs6 read the claim as "the experiment ran all 62".  Both are answered
by running every loader and publishing the result rather than asserting it.

    tables/loader_matrix.tex     -- per-loader status, grouped by category
    tables/loader_summary.tex    -- status counts per category, with a total
    _artifacts/loader_matrix.csv -- the full matrix, with shapes and timings

This is the one stage that is *expected* to be slow and to touch the network:
it downloads every raw archive in the catalogue.  ``--only-cached`` restricts
it to what is already on disk, which is what a network-free CI job wants.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from paper.experiments import _latex
from paper.experiments._config import CATEGORY_LABELS, PRETTY_LOADERS, TASK_ABBREV

STATUS_ORDER = ("ok", "needs-extra", "download-failed", "error", "skipped", "not-audited")

STATUS_CELL = {
    "ok": r"\cellcolor{winnerc}{\color{paperink}\textbf{ok}}",
    "needs-extra": r"\cellcolor{tiec}{\color{paperaccent}extra}",
    "download-failed": r"{\color{nac}\textit{download}}",
    "error": r"{\color{nac}\textit{error}}",
    "skipped": r"{\color{nac}\textit{skipped}}",
    "not-audited": r"{\color{nac}\textit{--}}",
}


def _fmt_int(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return r"{\color{nac}--}"
    return f"{int(value):,}".replace(",", r"\,")


def _fmt_size(value: Any, n_graphs: Any) -> str:
    """Nodes/edges cell: an exact count for one graph, a per-graph mean otherwise.

    ``validate_loaders`` reports means over the dataset's graphs, so a
    multi-graph row keeps one decimal even when the mean happens to be whole.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return r"{\color{nac}--}"
    single = n_graphs is None or (isinstance(n_graphs, float) and pd.isna(n_graphs)) or int(n_graphs) <= 1
    if single:
        return _fmt_int(value)
    whole, frac = f"{float(value):.1f}".split(".")
    return f"{int(whole):,}".replace(",", r"\,") + f".{frac}"


def _tasks_cell(tasks: str) -> str:
    parts = [TASK_ABBREV.get(t, t) for t in tasks.split("+")]
    return " ".join(rf"{{\color{{paperink}}\textbf{{{p}}}}}" for p in parts)


def run(
    *,
    root: str,
    categories: list[str] | None = None,
    only_cached: bool = False,
    resume_from: Path | None = None,
    max_seconds: float | None = None,
) -> tuple[pd.DataFrame, int]:
    """Audit catalogued loaders; returns ``(matrix, n_expected)``.

    The audit is resumable: rows already present in ``resume_from`` are not
    re-run, and ``max_seconds`` bounds a single call. Several bounded calls
    accumulate into the same matrix as one long run, which is how a
    multi-gigabyte audit gets done inside a time limit.
    """
    from graphnetz import validate_loaders
    from graphnetz.datasets import LOADER_REGISTRY

    expected = sum(len({n for tt in per for n, _ in per[tt]}) for per in LOADER_REGISTRY.values())
    done = pd.read_csv(resume_from) if resume_from and resume_from.exists() else None
    skip = set(done["loader"]) if done is not None else set()

    # Checkpoint after every loader rather than at the end. The walk downloads
    # tens of gigabytes over hours, so being interrupted is the normal case; a
    # run that only wrote its results on completion threw away every status it
    # had established, even though the archives were already on disk.
    checkpointed: list[dict[str, Any]] = []

    def checkpoint(row: dict[str, Any]) -> None:
        checkpointed.append(row)
        if resume_from is None:
            return
        partial = pd.DataFrame(checkpointed)
        merged = partial if done is None else pd.concat([done, partial], ignore_index=True)
        merged = merged.drop_duplicates(subset=["category", "loader"], keep="last")
        resume_from.parent.mkdir(parents=True, exist_ok=True)
        merged.sort_values(["category", "loader"]).to_csv(resume_from, index=False)

    fresh = validate_loaders(
        categories,
        root=root,
        only_cached=only_cached,
        skip=skip,
        max_seconds=max_seconds,
        on_row=checkpoint,
    )
    frame = fresh if done is None else pd.concat([done, fresh], ignore_index=True)
    if not frame.empty:
        frame = frame.drop_duplicates(subset=["category", "loader"], keep="last")
        frame = frame.sort_values(["category", "loader"]).reset_index(drop=True)
    return frame, expected


def _with_unaudited(frame: pd.DataFrame) -> pd.DataFrame:
    """Add a row for every catalogued loader the audit has not reached yet.

    The published table then lists all 62 loaders whatever the audit's progress,
    marking the ones it has not run as ``not-audited`` rather than omitting them.
    A table that silently listed only the audited subset would misrepresent
    coverage in exactly the way this appendix exists to prevent.
    """
    from graphnetz.datasets import LOADER_REGISTRY

    seen = set(zip(frame["category"], frame["loader"], strict=False)) if not frame.empty else set()
    extra: list[dict[str, Any]] = []
    for category, per_cat in LOADER_REGISTRY.items():
        merged: dict[str, list[str]] = {}
        for task_type, entries in per_cat.items():
            for name, _ in entries:
                merged.setdefault(name, []).append(task_type)
        for name, task_types in merged.items():
            if (category, name) in seen:
                continue
            extra.append(
                {
                    "category": category,
                    "loader": name,
                    "tasks": "+".join(sorted(task_types)),
                    "status": "not-audited",
                    "seconds": 0.0,
                    "error": "",
                }
            )
    if not extra:
        return frame
    out = pd.concat([frame, pd.DataFrame(extra)], ignore_index=True)
    return out.sort_values(["category", "loader"]).reset_index(drop=True)


def emit(
    frame: pd.DataFrame,
    *,
    tab_dir: Path,
    artifact_dir: Path,
    expected: int | None = None,
) -> dict[str, Any]:
    """Write the per-loader matrix, the per-category summary, and the CSV.

    ``expected`` is the catalogue size.  When fewer loaders have been audited the
    tables are still written and the shortfall is printed, so a partial audit is
    visible as partial instead of passing for complete.
    """
    # The CSV keeps only genuinely audited rows, so a later call resumes
    # correctly; the LaTeX table shows the whole catalogue.
    _latex.summary_csv(artifact_dir / "loader_matrix.csv", frame.set_index(["category", "loader"]))
    frame = _with_unaudited(frame)

    # ---- per-loader matrix -------------------------------------------------
    rows: list[list[str]] = []
    rules: set[int] = set()
    last_category: str | None = None
    for _, r in frame.iterrows():
        if last_category is not None and r["category"] != last_category:
            rules.add(len(rows))
        last_category = str(r["category"])
        rows.append(
            [
                CATEGORY_LABELS.get(str(r["category"]), str(r["category"])),
                PRETTY_LOADERS.get(str(r["loader"]), str(r["loader"]).replace("_", r"\_")),
                _tasks_cell(str(r["tasks"])),
                STATUS_CELL.get(str(r["status"]), str(r["status"])),
                _fmt_int(r.get("n_graphs")),
                _fmt_size(r.get("n_nodes"), r.get("n_graphs")),
                _fmt_size(r.get("n_edges"), r.get("n_graphs")),
                f"{float(r['seconds']):.1f}",
            ]
        )
    _latex.booktabs(
        tab_dir / "loader_matrix.tex",
        ["Category", "Loader", "Kind", "Status", "Graphs", "Nodes", "Edges", "s"],
        rows,
        align="llccrrrr",
        rules_before=rules,
    )

    # ---- per-category summary --------------------------------------------
    counts = frame.pivot_table(index="category", columns="status", values="loader", aggfunc="count").fillna(0)
    present = [s for s in STATUS_ORDER if s in counts.columns]
    summary_rows: list[list[str]] = []
    for category in frame["category"].drop_duplicates():
        c = counts.loc[category]
        total = int(sum(int(c.get(s, 0)) for s in present))
        summary_rows.append(
            [
                CATEGORY_LABELS.get(str(category), str(category)),
                str(total),
                *[str(int(c.get(s, 0))) for s in present],
            ]
        )
    totals = {s: int(counts[s].sum()) for s in present}
    grand = int(sum(totals.values()))
    summary_rows.append([r"\textbf{Total}", rf"\textbf{{{grand}}}", *[rf"\textbf{{{totals[s]}}}" for s in present]])
    _latex.booktabs(
        tab_dir / "loader_summary.tex",
        ["Category", "Loaders", *[s.replace("-", "-\\allowbreak ") for s in present]],
        summary_rows,
        align="lc" + "c" * len(present),
        rules_before={len(summary_rows) - 1},
    )

    by_status = {s: int((frame["status"] == s).sum()) for s in STATUS_ORDER}
    n_loaders = len(frame)
    total_seconds = round(float(frame["seconds"].sum()), 1)
    out: dict[str, Any] = {
        "n_loaders": n_loaders,
        "by_status": by_status,
        "total_seconds": total_seconds,
    }
    if expected is not None and n_loaders < expected:
        out["expected"] = expected
        print(f"  PARTIAL: {n_loaders}/{expected} loaders audited so far; rerun to continue")
    print(f"  validated {by_status['ok']}/{n_loaders} loaders end to end in {total_seconds:.0f}s")
    for status in STATUS_ORDER[1:]:
        n = by_status[status]
        if n:
            names = frame.loc[frame["status"] == status, "loader"].tolist()
            print(f"    {status}: {n} ({', '.join(names[:8])}{' ...' if n > 8 else ''})")
    return out
