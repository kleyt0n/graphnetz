"""Training drivers: resolve curated tasks, train a cell, merge cells.

Every run goes through :func:`graphnetz.run_benchmark`, so the reproduction
inherits the library's per-(task, model, seed) RNG reseeding (Appendix E)
rather than re-implementing it.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from paper.experiments import _cache, _config
from paper.experiments._config import HIDDEN_CHANNELS, Slot

from graphnetz import BenchmarkReport, Task, run_benchmark
from graphnetz.benchmark import BENCHMARK_TASKS


def model_registry(names: Iterable[str]) -> dict[str, type]:
    """Resolve model names to the classes exported by ``graphnetz``.

    Task compatibility comes from the library's model registry (GIN is
    graph-level only), so incompatible (model, task) pairs are skipped by the
    dispatcher rather than filtered here.
    """
    import graphnetz

    return {name: getattr(graphnetz, name) for name in names}


def resolve_task(slot: Slot) -> Task:
    """Look up ``slot``'s curated :class:`Task` in ``BENCHMARK_TASKS``.

    The task is taken from the library registry rather than redeclared, so
    the reproduction cannot drift from the shipped catalogue.  A mismatch
    against the epoch budget the paper recorded is reported as a warning
    instead of being silently reproduced.
    """
    candidates = BENCHMARK_TASKS.get(slot.category, {}).get(slot.task_type, [])
    for task in candidates:
        if task.name == slot.task_name:
            if task.epochs != slot.epochs:
                print(
                    f"  ! {slot.key}: registry epochs={task.epochs} but the paper ran "
                    f"epochs={slot.epochs}; using the registry value"
                )
            return task
    available = [t.name for t in candidates]
    msg = (
        f"{slot.key}: no curated task {slot.task_name!r} under "
        f"BENCHMARK_TASKS[{slot.category!r}][{slot.task_type!r}] (have: {available})"
    )
    raise KeyError(msg)


def train_cell(
    *,
    label: str,
    tasks: list[Task],
    category: str,
    model_names: Iterable[str],
    seeds: tuple[int, ...],
    data_root: str,
    cache_path: Path,
    epochs: int | None = None,
    force: bool = False,
    verbose: bool = True,
    cache_only: bool = False,
    search: Any = None,
    chunk: int = 2,
    max_seconds: float | None = None,
    models: dict[str, Any] | None = None,
) -> BenchmarkReport | None:
    """Train ``tasks`` x ``model_names`` x ``seeds``, or load the cached result.

    ``category`` only namespaces the dataset download directory
    (``<data_root>/<category>/<task>``); the tasks themselves are passed
    explicitly so the sweep is pinned to the datasets the paper used.

    **Seed-level resume.** The cache is written after every ``chunk`` seeds, and
    a cache holding a subset of ``seeds`` is completed rather than discarded, so
    an interrupted cell resumes where it stopped instead of restarting.  A
    multi-hour slot is thus safe to run in short sittings, and a killed job
    costs at most one chunk.  ``max_seconds`` stops cleanly after the current
    chunk once the budget is spent, which is what makes a slot trainable inside
    a bounded time limit.

    With ``cache_only`` a missing cache returns ``None`` instead of training,
    so read-only stages (``verify``) can never kick off a multi-hour sweep.

    ``models`` overrides the name-to-class lookup for cells whose models are
    not library exports -- the planted-effect control of
    ``positive_control`` defines its own three.  The dispatcher still decides
    task compatibility, so nothing about the run path differs.
    """
    cached = _cache.load(cache_path) if cache_path.exists() and not force else None
    if cached is not None and set(seeds) <= set(cached.seeds):
        print(f"  = {label}: cached ({len(cached.seeds)} seeds) <- {cache_path.name}")
        return _subset_seeds(cached, seeds)
    if cache_only:
        have = f"{len(cached.seeds)}/{len(seeds)} seeds cached" if cached else f"not cached ({cache_path.name})"
        print(f"  ? {label}: {have}; train it first")
        return None

    todo = [s for s in seeds if cached is None or s not in set(cached.seeds)]
    n_models = len(list(model_names))
    origin = f"resuming, {len(cached.seeds)} of {len(seeds)} seeds already done" if cached else "from scratch"
    print(f"  + {label}: training <= {n_models} models x {len(todo)} seeds ({origin})")

    started = time.monotonic()
    report = cached
    for i in range(0, len(todo), max(1, chunk)):
        batch = tuple(todo[i : i + max(1, chunk)])
        part = run_benchmark(
            category,
            models if models is not None else model_registry(model_names),
            tasks=tasks,
            root=data_root,
            hidden_channels=HIDDEN_CHANNELS,
            epochs=epochs,
            seeds=batch,
            verbose=verbose,
            search=search,
            device=_config.DEVICE,
        )
        report = part if report is None else _merge_seeds(report, part)
        _cache.save(cache_path, report)
        done = len(report.seeds)
        print(f"    saved {done}/{len(seeds)} seeds ({time.monotonic() - started:.0f}s elapsed)")
        if max_seconds is not None and time.monotonic() - started >= max_seconds and done < len(seeds):
            print(
                f"    stopping after {done}/{len(seeds)} seeds: time budget of {max_seconds:.0f}s spent. "
                f"Rerun the same command to continue from here."
            )
            return None
    return _subset_seeds(report, seeds) if report is not None else None


def _merge_seeds(a: BenchmarkReport, b: BenchmarkReport) -> BenchmarkReport:
    """Concatenate two reports over the seed axis, keeping seed order aligned.

    Both must cover the same (task, model) cells; the seed lists are disjoint.
    Pairing is by position within each report, so the merged history lists stay
    aligned with the merged seed tuple -- which is what every paired test relies
    on.
    """
    order = [*a.seeds, *b.seeds]
    index = sorted(range(len(order)), key=lambda i: order[i])
    seeds = tuple(order[i] for i in index)
    histories: dict[str, dict[str, list[dict[str, list[float]]]]] = {}
    for task in a.histories:
        histories[task] = {}
        for model, runs in a.histories[task].items():
            combined = [*runs, *b.histories.get(task, {}).get(model, [])]
            if len(combined) != len(order):
                msg = f"cannot merge seeds for {task}/{model}: {len(combined)} runs for {len(order)} seeds"
                raise ValueError(msg)
            histories[task][model] = [combined[i] for i in index]
    return BenchmarkReport(seeds=seeds, histories=histories, config={**a.config, **b.config})


def _subset_seeds(report: BenchmarkReport, seeds: tuple[int, ...]) -> BenchmarkReport:
    """Restrict a report to ``seeds``, so a wider cache serves a narrower run."""
    if tuple(report.seeds) == tuple(seeds):
        return report
    keep = [report.seeds.index(s) for s in seeds]
    histories = {
        task: {model: [runs[i] for i in keep] for model, runs in per.items()} for task, per in report.histories.items()
    }
    return BenchmarkReport(seeds=tuple(seeds), histories=histories, config=dict(report.config))


def merge(reports: dict[str, BenchmarkReport]) -> BenchmarkReport:
    """Stitch per-slot reports into one, re-keying tasks by slot label.

    The merged report is what feeds the cross-category CD diagram: models
    absent from any single task drop out of the rank table, which is exactly
    how GIN (graph-level only) is excluded from Figure 3 without a filter.
    """
    if not reports:
        msg = "merge() needs at least one report"
        raise ValueError(msg)
    seed_sets = {rep.seeds for rep in reports.values()}
    if len(seed_sets) != 1:
        msg = f"cannot merge reports trained on different seed sets: {sorted(seed_sets)}"
        raise ValueError(msg)
    histories: dict[str, dict[str, list[dict[str, list[float]]]]] = {}
    for label, rep in reports.items():
        if len(rep.histories) != 1:
            msg = f"{label}: expected exactly one task per slot report, got {list(rep.histories)}"
            raise ValueError(msg)
        histories[label] = next(iter(rep.histories.values()))
    return BenchmarkReport(
        seeds=next(iter(seed_sets)),
        histories=histories,
        config={"merged_from": len(reports)},
    )
