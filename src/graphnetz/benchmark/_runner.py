"""The benchmark driver: per-task training dispatch and the public runner."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm

from graphnetz.benchmark._report import BenchmarkReport
from graphnetz.benchmark._search import SearchSpace, select
from graphnetz.benchmark._specs import TASK_TYPES, ModelSpec, Task, _spec_from
from graphnetz.benchmark._stats import _final_metric
from graphnetz.benchmark._tasks import BENCHMARK_TASKS, iter_benchmark_tasks
from graphnetz.plotting import plot_grouped_bars, set_plot_style
from graphnetz.training import (
    train_graph_classification,
    train_graph_regression,
    train_link_prediction,
    train_node_classification,
)

# --------------------------------------------------------------------------- #
# Task runner
# --------------------------------------------------------------------------- #

GRAPH_SPLIT: tuple[float, float, float] = (0.8, 0.1, 0.1)
"""Train/validation/test fractions for graph-level datasets without a canonical split."""


def _three_way_split(ds: Any) -> tuple[Any, Any, Any]:
    """Shuffle a graph-level dataset and cut it into train/validation/test.

    The shuffle draws from torch's global RNG, which the runner reseeds before
    every (task, model, seed) triple, so the split changes with the seed but is
    identical across the models compared at that seed. The held-out third split
    is what lets ``epoch_selection="best_val"`` and the hyperparameter search
    select on validation without reporting on the series they selected on.
    """
    n = len(ds)
    if n < 3:
        msg = f"a graph-level dataset needs at least 3 graphs for a train/val/test split; got {n}"
        raise ValueError(msg)
    perm = torch.randperm(n)
    n_val = max(1, int(round(GRAPH_SPLIT[1] * n)))
    n_test = max(1, int(round(GRAPH_SPLIT[2] * n)))
    n_train = n - n_val - n_test
    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]
    return ds[train_idx], ds[val_idx], ds[test_idx]


def _graph_out_channels(ds: Any) -> int:
    """Output width of a graph-classification head.

    Multi-label targets (``y`` of shape ``[1, C]`` per graph, e.g.
    ogbg-molpcba) need one logit per label, while PyG's ``num_classes``
    counts distinct label *values* there and would report 2.
    """
    y = getattr(getattr(ds, "_data", None), "y", None)
    if y is None:
        y = getattr(ds[0], "y", None)
    if y is not None and y.dim() == 2 and y.size(1) > 1:
        return int(y.size(1))
    return int(ds.num_classes)


def _categorical_vocab(ds: Any) -> list[int] | None:
    """Per-column vocabulary sizes when node features are integer codes.

    Molecular datasets (OGB ``ogbg-*``, LRGB Peptides) store atoms as rows of
    categorical ids. Feeding those to a convolution as numbers is meaningless,
    so the graph-level path embeds them instead. Returns ``None`` for float
    features.
    """
    x = getattr(getattr(ds, "_data", None), "x", None)
    if x is None:
        x = getattr(ds[0], "x", None)
    if x is None or x.is_floating_point():
        return None
    codes = x.view(x.size(0), -1)
    return [int(v) + 1 for v in codes.max(dim=0).values.tolist()]


class _CategoricalEncoder(torch.nn.Module):
    """Sum of one embedding per categorical feature column, then ``inner``."""

    def __init__(self, vocab: list[int], hidden: int, inner: torch.nn.Module) -> None:
        super().__init__()
        self.embeds = torch.nn.ModuleList(torch.nn.Embedding(v, hidden) for v in vocab)
        self.inner = inner

    def forward(self, batch: Any) -> torch.Tensor:
        codes = batch.x.view(batch.x.size(0), -1).long()
        batch = batch.clone()
        batch.x = sum(embed(codes[:, i]) for i, embed in enumerate(self.embeds))
        return self.inner(batch)


def _run_task(
    task_type: Task,
    ds: Any,
    spec: ModelSpec,
    hidden: int,
    epochs: int,
    verbose: bool = False,
    device: torch.device | str | None = "auto",
    hyper: Mapping[str, Any] | None = None,
) -> dict[str, list[float]]:
    """Train one (task, model) pair once.

    ``hyper`` optionally overrides the trainer's optimiser defaults and the
    encoder width; it is how the inner hyperparameter search
    (`graphnetz.benchmark._search`) varies a candidate. Keys the relevant
    trainer does not accept are dropped rather than raising, because the
    searchable surface differs by task type (only node classification takes a
    weight decay).
    """
    hp = dict(hyper or {})
    hidden = int(hp.pop("hidden_channels", hidden))

    def _opt(*accepted: str) -> dict[str, Any]:
        return {k: v for k, v in hp.items() if k in accepted}

    if task_type.task_type == "node_cls":
        data = ds[0]
        model = spec.build(ds.num_features, hidden, ds.num_classes, task_type="node_cls")
        return train_node_classification(
            model, data, epochs=epochs, verbose=verbose, device=device, **_opt("lr", "weight_decay")
        )

    if task_type.task_type == "graph_cls":
        train_ds, val_ds, test_ds = _three_way_split(ds)
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=32)
        test_loader = DataLoader(test_ds, batch_size=32)
        vocab = _categorical_vocab(ds)
        out_channels = _graph_out_channels(ds)
        if vocab is None:
            model = spec.build(ds.num_features, hidden, out_channels, task_type="graph_cls")
        else:
            inner = spec.build(hidden, hidden, out_channels, task_type="graph_cls")
            model = _CategoricalEncoder(vocab, hidden, inner)
        return train_graph_classification(
            model,
            train_loader,
            val_loader,
            epochs=epochs,
            verbose=verbose,
            device=device,
            test_loader=test_loader,
            **_opt("lr"),
        )

    if task_type.task_type == "graph_reg":
        # Loader may return a single dataset (split here 80/10/10, as for
        # graph classification), a ``(train, val, test)`` triple (real
        # benchmarks with a canonical split, e.g. ZINC), or a legacy
        # ``(train, val)`` pair, which records no held-out series.
        test_ds = None
        if isinstance(ds, tuple) and len(ds) == 3:
            train_ds, val_ds, test_ds = ds
        elif isinstance(ds, tuple) and len(ds) == 2:
            train_ds, val_ds = ds
        elif isinstance(ds, tuple):
            msg = f"graph_reg loader returned a tuple of {len(ds)} datasets; expected 2 or 3"
            raise ValueError(msg)
        else:
            train_ds, val_ds, test_ds = _three_way_split(ds)
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=64)
        reg_test_loader = DataLoader(test_ds, batch_size=64) if test_ds is not None else None
        # Integer node features (ZINC atom ids) are embedded, float features
        # are used as they are. The vocabulary spans every split so an id
        # first seen at validation or test time still has an embedding.
        vocabs = [v for v in (_categorical_vocab(d) for d in (train_ds, val_ds, test_ds) if d is not None) if v]
        if vocabs:
            vocab = [max(col) for col in zip(*vocabs, strict=False)]
            reg_model: torch.nn.Module = _CategoricalEncoder(
                vocab, hidden, spec.build(hidden, hidden, 1, task_type="graph_reg")
            )
        else:
            reg_model = spec.build(train_ds.num_features, hidden, 1, task_type="graph_reg")

        return train_graph_regression(
            reg_model,
            train_loader,
            val_loader,
            epochs=epochs,
            verbose=verbose,
            device=device,
            test_loader=reg_test_loader,
            **_opt("lr"),
        )

    if task_type.task_type == "link_pred":
        import math

        from torch_geometric.transforms import RandomLinkSplit
        from torch_geometric.utils import degree

        data = ds[0]

        def _fabricate_log_degree_features(d: Data, edge_index: torch.Tensor) -> Data:
            """Build a 3-D log-degree feature from `edge_index` only.

            Used as the fallback when a loader ships no node features. We
            keep the source of degree restricted to the caller-supplied
            `edge_index` so val/test edges never bleed into the feature
            matrix at training time.
            """
            n = int(d.num_nodes)
            deg = degree(edge_index[0], num_nodes=n, dtype=torch.float)
            log_deg = torch.log1p(deg) / math.log(max(n, 2))
            ones = torch.ones(n)
            out = d.clone()
            out.x = torch.stack([log_deg, log_deg.pow(2), ones], dim=1)
            return out

        # Relational link prediction (knowledge graphs with edge_type) is
        # detected on the *raw* data because PyG's RelLinkPredDataset
        # already restricts edge_index to training edges -- we only
        # fabricate features when missing, using train_edge_index.
        if hasattr(data, "edge_type") and hasattr(data, "train_edge_index"):
            from graphnetz.models._adapters import RelationalLinkPredWrapper
            from graphnetz.training import train_relational_link_prediction

            if getattr(data, "x", None) is None:
                data = _fabricate_log_degree_features(data, data.train_edge_index)

            num_relations = ds.num_relations if hasattr(ds, "num_relations") else int(data.edge_type.max()) + 1
            built = spec.build(data.num_features, hidden, hidden, task_type="link_pred")
            # spec.build returns a LinkPredWrapper for task_type="link_pred"; unwrap it
            # so RelationalLinkPredWrapper can drive the bare encoder directly
            # (otherwise its forward expects data.edge_label_index).
            from typing import cast

            encoder = cast(
                torch.nn.Module,
                built.encoder if hasattr(built, "encoder") else built,
            )
            model = RelationalLinkPredWrapper(encoder, hidden, num_relations)

            # Create separate Data objects for train/val/test splits
            train_split = Data(
                x=data.x, edge_index=data.train_edge_index, edge_type=data.train_edge_type, num_nodes=data.num_nodes
            )
            val_split = Data(
                x=data.x, edge_index=data.valid_edge_index, edge_type=data.valid_edge_type, num_nodes=data.num_nodes
            )
            test_split = Data(
                x=data.x, edge_index=data.test_edge_index, edge_type=data.test_edge_type, num_nodes=data.num_nodes
            )

            return train_relational_link_prediction(
                model,
                train_split,
                val_split,
                test_split,
                epochs=epochs,
                verbose=verbose,
                device=device,
                **_opt("lr"),
            )

        # Detect graph direction from the data itself instead of forcing
        # ``is_undirected=True`` -- on a directed graph the latter silently
        # de-duplicates reciprocal edges and halves the supervision signal.
        is_undirected = not bool(data.is_directed())
        transform = RandomLinkSplit(
            num_val=0.05,
            num_test=0.10,
            is_undirected=is_undirected,
            add_negative_train_samples=True,
            neg_sampling_ratio=1.0,
        )
        train_data, val_data, test_data = transform(data)
        # Fabricate features *after* the split so val/test edges never
        # leak into the node features the encoder consumes.  Use only the
        # training message-passing edges (edge_index, not edge_label_index)
        # for the degree statistic.
        if getattr(train_data, "x", None) is None:
            train_data = _fabricate_log_degree_features(train_data, train_data.edge_index)
            val_data = val_data.clone()
            val_data.x = train_data.x
            test_data = test_data.clone()
            test_data.x = train_data.x
        # ``spec.build(task_type="link_pred")`` returns a LinkPredWrapper, which
        # satisfies the ``_LinkPredLike`` protocol of the trainer; mypy
        # only sees the declared ``Module`` return so we narrow here.
        from typing import cast as _cast

        from graphnetz.training import _LinkPredLike

        lp_model = _cast(_LinkPredLike, spec.build(train_data.num_features, hidden, hidden, task_type="link_pred"))
        return train_link_prediction(
            lp_model, train_data, val_data, test_data, epochs=epochs, verbose=verbose, device=device, **_opt("lr")
        )

    msg = f"Unknown task task_type: {task_type.task_type}"
    raise ValueError(msg)


def _seed_all(seed: int) -> None:
    """Seed every RNG that benchmark training touches."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _normalize_seeds(
    seeds: int | Iterable[int] | None,
    seed: int | None,
) -> tuple[int, ...]:
    if seed is not None:
        return (int(seed),)
    if seeds is None:
        return (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    if isinstance(seeds, int):
        return (int(seeds),)
    return tuple(int(s) for s in seeds)


def run_benchmark(
    category: str | None = None,
    models: type | tuple[Any, ...] | ModelSpec | dict[str, type | tuple[Any, ...] | ModelSpec] | None = None,
    root: str = "data/benchmark",
    hidden_channels: int = 64,
    epochs: int | None = None,
    only: list[str] | None = None,
    verbose: bool = True,
    seeds: int | Iterable[int] | None = None,
    seed: int | None = None,
    task_type: str | None = None,
    tasks: Iterable[Task] | None = None,
    device: torch.device | str | None = "auto",
    search: SearchSpace | None = None,
) -> BenchmarkReport:
    """Run a benchmark across one or more (model, task, seed) combinations.

    Two ways to choose tasks:

    1. **By category** (default) -- tasks come from
       [`BENCHMARK_TASKS`][graphnetz.benchmark.BENCHMARK_TASKS] indexed as
       ``[category][task_type] -> list[Task]``. Pass ``category="social"``
       (etc.) and optionally restrict with ``task_type`` and ``only=``.
    2. **Ad-hoc** -- pass ``tasks=[Task(...), ...]`` to bypass the registry
       entirely. Useful for benchmarking custom datasets without mutating
       global state. ``category`` then defaults to ``"custom"`` and is used
       only to namespace ``root/`` cache directories.

    The runner trains every compatible (model, task) pair across each
    value in ``seeds`` (default ``(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)``) and aggregates the per-seed
    histories into a [`BenchmarkReport`][graphnetz.benchmark.BenchmarkReport].

    ``search`` optionally adds an inner hyperparameter loop (Algorithm 1,
    line 4a): for every (task, model, seed) the runner trains each candidate in
    the [`SearchSpace`][graphnetz.benchmark.SearchSpace], scores them on the task's
    **validation** series, and keeps the winner's history. Candidates are never
    scored on a held-out metric. The selected configuration and the full search
    trace land in ``report.config["search_selected"]``, and the cost multiplier
    is announced before training starts, because a grid of ``G`` candidates
    multiplies compute by ``G``.
    """
    if models is None:
        msg = "run_benchmark requires `models` (a class, dict, or ModelSpec)"
        raise ValueError(msg)
    if task_type is not None and task_type not in TASK_TYPES:
        msg = f"Unknown task type {task_type!r}. Choices: {sorted(TASK_TYPES)}"
        raise ValueError(msg)
    if not isinstance(models, dict):
        spec = _spec_from(models)
        models = {spec.cls.__name__: spec}

    resolved = {name: _spec_from(value) for name, value in models.items()}
    seed_list = _normalize_seeds(seeds, seed)

    if tasks is not None:
        task_list = list(tasks)
        for t in task_list:
            if not isinstance(t, Task):
                msg = f"`tasks` must contain Task instances, got {type(t).__name__}"
                raise TypeError(msg)
            if t.task_type not in TASK_TYPES:
                msg = f"Task {t.name!r} has unknown task type {t.task_type!r}; choices: {sorted(TASK_TYPES)}"
                raise ValueError(msg)
        if task_type is not None:
            task_list = [t for t in task_list if t.task_type == task_type]
        if category is None:
            category = "custom"
    else:
        if category is None:
            msg = "run_benchmark requires either `category` or `tasks=`"
            raise ValueError(msg)
        if category not in BENCHMARK_TASKS:
            msg = f"Unknown category {category!r}. Choices: {sorted(BENCHMARK_TASKS)}"
            raise KeyError(msg)
        task_list = iter_benchmark_tasks(category=category, task_type=task_type)
    if only is not None:
        task_list = [t for t in task_list if t.name in only]
    tasks = task_list  # the loop below treats this as the working list

    histories: dict[str, dict[str, list[dict[str, list[float]]]]] = {}
    search_selected: dict[str, Any] = {}
    candidates = search.candidates() if search is not None else [{}]
    if search is not None and not search.is_empty:
        print(f"hyperparameter search: {search.describe()} -- multiplies training cost by {len(candidates)}x")
    total_combinations = (
        sum(1 for spec in resolved.values() for task in tasks if task.task_type in spec.task_type)
        * len(seed_list)
        * len(candidates)
    )
    overall_pbar = tqdm(
        total=total_combinations,
        desc="Benchmark",
        unit="run",
        disable=not verbose,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )
    import inspect

    for task in tasks:
        try:
            seed_aware = "seed" in inspect.signature(task.loader).parameters
        except (TypeError, ValueError):
            seed_aware = False
        ds_cache: Any = None  # for seed-agnostic loaders, load once
        histories[task.name] = {}
        for model_name, spec in resolved.items():
            if task.task_type not in spec.task_type:
                continue
            histories[task.name][model_name] = []
            for s in seed_list:
                _seed_all(s)
                if seed_aware:
                    # Seed-aware loaders (e.g. synthetic combinatorial graphs)
                    # produce a fresh dataset per seed, so cross-seed variance
                    # captures data resampling rather than only model init.
                    ds = task.loader(f"{root}/{category}/{task.name}/seed{s}", seed=s)
                else:
                    if ds_cache is None:
                        ds_cache = task.loader(f"{root}/{category}/{task.name}")
                    ds = ds_cache
                if len(candidates) == 1:
                    history = _run_task(
                        task,
                        ds,
                        spec,
                        hidden_channels,
                        epochs or task.epochs,
                        verbose=verbose,
                        device=device,
                        hyper=candidates[0],
                    )
                else:
                    # Inner search: reseed before every candidate so the
                    # comparison between them is paired too, then keep the one
                    # that wins on validation.
                    def _trials(t=task, sp=spec, d=ds, s_=s):
                        for cand in candidates:
                            _seed_all(s_)
                            yield (
                                cand,
                                _run_task(
                                    t,
                                    d,
                                    sp,
                                    hidden_channels,
                                    epochs or t.epochs,
                                    verbose=False,
                                    device=device,
                                    hyper=cand,
                                ),
                            )
                            overall_pbar.update(1)

                    best_config, history, trace = select(_trials())
                    search_selected[f"{task.name}/{model_name}/seed{s}"] = {
                        "selected": best_config,
                        "trace": trace,
                    }
                histories[task.name][model_name].append(history)
                # Update overall progress with latest metric
                last_metrics = {k: v[-1] for k, v in history.items() if v}
                metric_str = " ".join(f"{k[:3]}={v:.3f}" for k, v in last_metrics.items())
                overall_pbar.set_postfix_str(f"{task.name}/{model_name}/s{s} | {metric_str}", refresh=False)
                if len(candidates) == 1:
                    overall_pbar.update(1)
    overall_pbar.close()

    from graphnetz.training import _resolve_device

    config = {
        "category": category,
        "task": task,
        "hidden_channels": hidden_channels,
        "epochs": epochs,
        "only": only,
        "device": str(_resolve_device(device)),
        **_provenance(),
    }
    if search is not None and not search.is_empty:
        config["search"] = search.describe()
        config["search_selected"] = search_selected  # type: ignore[assignment]
    return BenchmarkReport(seeds=seed_list, histories=histories, config=config)


def _provenance() -> dict[str, str]:
    """Record the software stack in the report.

    Determinism is only meaningful relative to a stack (Appendix E), so the
    report carries its own: a later re-analysis can attribute a drifted number
    to a library upgrade instead of guessing.
    """
    import platform
    import sys

    out: dict[str, str] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for name in ("graphnetz", "torch", "torch_geometric", "scipy", "numpy", "pandas"):
        try:
            module = __import__(name)
            out[f"{name}_version"] = str(getattr(module, "__version__", "unknown"))
        except ImportError:  # pragma: no cover - optional at analysis time
            out[f"{name}_version"] = "absent"
    return out


def plot_benchmark(
    results: BenchmarkReport | Mapping[str, Mapping[str, Mapping[str, list[float]]]],
    errors: Mapping[str, Mapping[str, float]] | None = None,
    ax: plt.Axes | None = None,
    title: str | None = None,
    annotate: bool = True,
    ci: float = 0.95,
) -> tuple[plt.Figure, plt.Axes]:
    """Grouped bar chart with mean ± CI error bars.

    Accepts a [`BenchmarkReport`][graphnetz.benchmark.BenchmarkReport] (preferred) or the legacy dict form for
    a single seed. ``errors`` overrides the default t-CI half-width.
    """
    if isinstance(results, BenchmarkReport):
        return results.plot(ax=ax, title=title, annotate=annotate, ci=ci)

    set_plot_style()
    values: dict[str, dict[str, float]] = {}
    metric_label: str | None = None
    for task_name, per_task in results.items():
        per_value: dict[str, float] = {}
        for model_name, history in per_task.items():
            metric, value = _final_metric(history)
            metric_label = metric_label or metric
            per_value[model_name] = value
        values[task_name] = per_value

    return plot_grouped_bars(
        values,
        errors=errors,
        ax=ax,
        title=title,
        ylabel=metric_label or "metric",
        annotate=annotate,
    )
