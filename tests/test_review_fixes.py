"""Regression tests for the fixes prompted by the LoG 2026 review.

Each section pins one defect: the symmetrised bootstrap interval, graph-level
tasks without a held-out split (and the search leak that followed), the OGB
node/link loaders returning a bare ``Data``, the loader audit's shape columns,
ModelNet meshes without edges, max-cut aliasing coloring, and the smaller
reporting issues.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch
from torch_geometric.data import Data, InMemoryDataset

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _report(histories: dict, seeds: tuple[int, ...] | None = None):
    from graphnetz.benchmark import BenchmarkReport

    n = len(next(iter(next(iter(histories.values())).values())))
    return BenchmarkReport(seeds=seeds or tuple(range(n)), histories=histories)


def _const(values: list[float], key: str = "test_acc") -> list[dict[str, list[float]]]:
    return [{key: [v]} for v in values]


class _Graphs(InMemoryDataset):
    """An in-memory graph-level dataset built from a list of ``Data``."""

    def __init__(self, data_list: list[Data]) -> None:
        super().__init__(None)
        self._data, self.slices = self.collate(data_list)


def _toy_graphs(n: int = 40, *, integer_x: bool = False, regression: bool = False) -> _Graphs:
    gen = torch.Generator().manual_seed(0)
    graphs = []
    for i in range(n):
        nodes = 5 + i % 4
        ring = torch.arange(nodes)
        edge_index = torch.stack([torch.cat([ring, (ring + 1) % nodes]), torch.cat([(ring + 1) % nodes, ring])])
        x = torch.randint(0, 6, (nodes, 3), generator=gen) if integer_x else torch.randn(nodes, 4, generator=gen)
        y = torch.tensor([float(i % 3)]) if regression else torch.tensor([i % 2])
        graphs.append(Data(x=x, edge_index=edge_index, y=y))
    return _Graphs(graphs)


# --------------------------------------------------------------------------- #
# Bootstrap intervals keep their endpoints
# --------------------------------------------------------------------------- #


def test_bootstrap_interval_is_asymmetric_and_respects_the_ceiling() -> None:
    from graphnetz.benchmark._stats import _bootstrap_ci

    values = np.array([1.0] * 9 + [0.9])
    lo, hi = _bootstrap_ci(values)
    assert hi <= 1.0
    assert lo < values.mean() < hi
    # The old symmetric envelope put the upper end above the ceiling.
    assert values.mean() + (hi - lo) / 2 > 1.0


def test_summary_reports_percentile_endpoints_under_bootstrap() -> None:
    report = _report({"t": {"A": _const([1.0] * 9 + [0.9]), "B": _const([0.5 + 0.01 * i for i in range(10)])}})
    boot = report.summary(method="bootstrap")
    assert boot.loc[("t", "A"), "ci_high"] <= 1.0
    mean = boot.loc[("t", "A"), "mean"]
    assert mean - boot.loc[("t", "A"), "ci_low"] != pytest.approx(boot.loc[("t", "A"), "ci_high"] - mean)

    t = report.summary(method="t")
    row = t.loc[("t", "B")]
    assert row["mean"] - row["ci_low"] == pytest.approx(row["ci_high"] - row["mean"])


def test_to_latex_prints_asymmetric_bootstrap_offsets(tmp_path: Path) -> None:
    report = _report({"t": {"A": _const([1.0] * 9 + [0.9]), "B": _const([0.6] * 10)}})
    text = report.to_latex(tmp_path / "boot.tex", method="bootstrap").read_text()
    assert "^{+" in text
    assert "_{-" in text
    assert r"\pm" not in text
    text_t = report.to_latex(tmp_path / "t.tex", method="t").read_text()
    assert r"\pm" in text_t


def test_plots_accept_asymmetric_errors() -> None:
    import matplotlib

    matplotlib.use("Agg")
    report = _report({"t": {"A": _const([1.0] * 9 + [0.9]), "B": _const([0.6 + 0.01 * i for i in range(10)])}})
    report.ci_method = "bootstrap"
    report.plot()
    report.plot_forest()
    report.plot_pairwise()


# --------------------------------------------------------------------------- #
# Reporting details
# --------------------------------------------------------------------------- #


def test_p_values_keep_three_significant_digits() -> None:
    from graphnetz.benchmark._report import _format_p

    assert _format_p(0.11) == "0.110"
    assert _format_p(0.1) == "0.100"
    assert _format_p(0.0547) == "0.0547"
    assert _format_p(float("nan")) == "n/a"


def test_best_cell_uses_each_tasks_own_direction(tmp_path: Path) -> None:
    report = _report(
        {
            "acc_task": {"A": _const([0.9, 0.91, 0.9]), "B": _const([0.5, 0.52, 0.51])},
            "mae_task": {"A": _const([3.0, 3.1, 3.2], "val_mae"), "B": _const([1.0, 1.1, 1.0], "val_mae")},
        }
    )
    best = report._best_per_task()
    assert best == {"acc_task": "A", "mae_task": "B"}


def test_reliability_weights_ignore_models_outside_the_rank_table() -> None:
    noisy = [0.1, 0.9, 0.2, 0.8, 0.3]
    base = {
        "t1": {"A": _const([0.8, 0.81, 0.8, 0.79, 0.8]), "B": _const([0.7, 0.71, 0.7, 0.69, 0.7])},
        "t2": {"A": _const([0.6, 0.61, 0.6, 0.59, 0.6]), "B": _const([0.65, 0.66, 0.65, 0.64, 0.65])},
    }
    with_extra = {k: dict(v) for k, v in base.items()}
    with_extra["t1"]["C"] = _const(noisy)  # present on one task only, so unranked
    a = _report(base).mean_ranks(aggregation="reliability")
    b = _report(with_extra).mean_ranks(aggregation="reliability")
    assert list(b.index) == list(a.index)
    np.testing.assert_allclose(a.to_numpy(), b.to_numpy())


def test_hierarchical_with_singleton_groups_warns() -> None:
    report = _report(
        {
            "t1": {"A": _const([0.8, 0.81, 0.8]), "B": _const([0.7, 0.71, 0.7])},
            "t2": {"A": _const([0.6, 0.61, 0.6]), "B": _const([0.65, 0.66, 0.65])},
        }
    )
    with pytest.warns(UserWarning, match="identical to 'uniform'"):
        report.mean_ranks(aggregation="hierarchical", groups={"t1": "x", "t2": "y"})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        report.mean_ranks(aggregation="hierarchical", groups={"t1": "x", "t2": "x"})


def test_seeds_for_effect_undefined_inputs_are_nan() -> None:
    from graphnetz.benchmark._power import minimum_detectable_effect, seeds_for_effect

    assert math.isnan(seeds_for_effect(0.01, float("nan")))
    assert math.isnan(seeds_for_effect(0.0, 0.01))
    assert seeds_for_effect(0.01, 0.0) == 2
    n = seeds_for_effect(0.01, 0.0138)
    assert minimum_detectable_effect(0.0138, n) <= 0.01 < minimum_detectable_effect(0.0138, n - 1)


def test_register_model_accepts_task_types_alias() -> None:
    from graphnetz.benchmark._specs import _REGISTRY, register_model

    class _A(torch.nn.Module):
        pass

    class _B(torch.nn.Module):
        pass

    try:
        register_model(_A, task_types={"node_cls", "link_pred"})
        register_model(_B, task_type="graph_cls")
        assert _REGISTRY[_A].task_type == frozenset({"node_cls", "link_pred"})
        assert _REGISTRY[_B].task_type == frozenset({"graph_cls"})
        with pytest.raises(TypeError):
            register_model(_A, task_type="node_cls", task_types="node_cls")
        with pytest.raises(TypeError):
            register_model(_A)
    finally:
        _REGISTRY.pop(_A, None)
        _REGISTRY.pop(_B, None)


# --------------------------------------------------------------------------- #
# Graph-level tasks get a held-out split
# --------------------------------------------------------------------------- #


def _run(task_type: str, loader, *, seeds=(0, 1), search=None):
    from graphnetz import GCN, run_benchmark
    from graphnetz.benchmark import Task

    task = Task(name="toy", task_type=task_type, loader=loader, epochs=2)
    return run_benchmark(models={"GCN": GCN}, tasks=[task], seeds=seeds, verbose=False, device="cpu", search=search)


def test_three_way_split_is_disjoint_and_seed_paired() -> None:
    from graphnetz.benchmark._runner import _three_way_split

    ds = _toy_graphs(40)
    torch.manual_seed(3)
    tr, va, te = _three_way_split(ds)
    assert (len(tr), len(va), len(te)) == (32, 4, 4)
    ids = [set(part.indices()) for part in (tr, va, te)]
    assert not (ids[0] & ids[1])
    assert not (ids[0] & ids[2])
    assert not (ids[1] & ids[2])
    torch.manual_seed(3)
    again = _three_way_split(ds)
    assert [p.indices() for p in again] == [p.indices() for p in (tr, va, te)]


def test_graph_classification_records_a_held_out_series_and_supports_best_val() -> None:
    report = _run("graph_cls", lambda _root: _toy_graphs(40))
    history = report.histories["toy"]["GCN"][0]
    assert {"val_acc", "test_acc"} <= set(history)
    assert report.metric_name() == "test_acc"
    summary = report.summary(epoch_selection="best_val")
    assert summary.loc[("toy", "GCN"), "n_seeds"] == 2


def test_graph_classification_embeds_integer_features() -> None:
    report = _run("graph_cls", lambda _root: _toy_graphs(30, integer_x=True), seeds=(0,))
    assert len(report.histories["toy"]["GCN"][0]["test_acc"]) == 2


def test_graph_regression_single_dataset_is_split_not_evaluated_on_train() -> None:
    report = _run("graph_reg", lambda _root: _toy_graphs(30, regression=True), seeds=(0,))
    history = report.histories["toy"]["GCN"][0]
    assert {"val_mae", "test_mae"} <= set(history)
    assert report.metric_name() == "test_mae"


def test_graph_regression_accepts_canonical_triples_and_legacy_pairs() -> None:
    def triple(_root: str):
        ds = _toy_graphs(30, regression=True)
        return ds[:20], ds[20:25], ds[25:]

    def pair(_root: str):
        ds = _toy_graphs(30, regression=True)
        return ds[:20], ds[20:]

    assert "test_mae" in _run("graph_reg", triple, seeds=(0,)).histories["toy"]["GCN"][0]
    assert "test_mae" not in _run("graph_reg", pair, seeds=(0,)).histories["toy"]["GCN"][0]


def test_search_refuses_tasks_that_report_their_validation_series() -> None:
    from graphnetz.benchmark import SearchSpace

    def pair(_root: str):
        ds = _toy_graphs(30, regression=True)
        return ds[:20], ds[20:]

    space = SearchSpace(lr=(1e-2, 1e-3))
    with pytest.raises(ValueError, match="leak"):
        _run("graph_reg", pair, seeds=(0,), search=space)
    report = _run("graph_cls", lambda _root: _toy_graphs(30), seeds=(0,), search=space)
    assert report.config["search_selected"]


def test_multi_label_loss_ignores_missing_labels() -> None:
    from torch_geometric.loader import DataLoader

    from graphnetz.training import train_graph_classification

    graphs = []
    for i in range(12):
        y = torch.tensor([[float(i % 2), float("nan"), 1.0]])
        graphs.append(Data(x=torch.randn(4, 3), edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]), y=y))
    loader = DataLoader(graphs, batch_size=4)

    class _Head(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = torch.nn.Linear(3, 3)

        def forward(self, batch):
            from torch_geometric.nn import global_mean_pool

            return self.lin(global_mean_pool(batch.x, batch.batch))

    history = train_graph_classification(_Head(), loader, loader, epochs=2, device="cpu", test_loader=loader)
    assert all(math.isfinite(v) for v in history["train_loss"])
    assert all(0.0 <= v <= 1.0 for v in history["test_acc"])


# --------------------------------------------------------------------------- #
# OGB node/link loaders and the loader audit
# --------------------------------------------------------------------------- #


def test_single_graph_dataset_behaves_like_a_dataset() -> None:
    from graphnetz.datasets._ogb import SingleGraphDataset

    data = Data(x=torch.randn(5, 2), edge_index=torch.tensor([[0, 1], [1, 2]]), y=torch.tensor([0, 1, 0, 1, 0]))
    ds = SingleGraphDataset(data, num_features=2, num_classes=2)
    assert len(ds) == 1
    assert ds[0] is data
    assert ds[-1] is data
    with pytest.raises(IndexError):
        ds[1]
    assert ds.num_features == 2
    assert ds.num_classes == 2
    assert torch.equal(ds.x, data.x)  # forwarded for backward compatibility


def test_single_graph_dataset_runs_through_node_classification() -> None:
    from graphnetz.datasets._ogb import SingleGraphDataset

    n = 20
    data = Data(x=torch.randn(n, 4), edge_index=torch.randint(0, n, (2, 60)), y=torch.randint(0, 2, (n,)))
    data.train_mask = torch.arange(n) < 10
    data.val_mask = (torch.arange(n) >= 10) & (torch.arange(n) < 15)
    data.test_mask = torch.arange(n) >= 15
    report = _run("node_cls", lambda _root: SingleGraphDataset(data, num_features=4, num_classes=2), seeds=(0,))
    assert "test_acc" in report.histories["toy"]["GCN"][0]


def test_audit_shape_counts_graphs_and_averages_sizes() -> None:
    from graphnetz.datasets._ogb import SingleGraphDataset
    from graphnetz.datasets._validate import _shape

    data = Data(x=torch.randn(7, 2), edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]))
    for ds in (data, SingleGraphDataset(data, num_features=2)):
        shape = _shape(ds)
        assert (shape["n_graphs"], shape["n_nodes"], shape["n_edges"]) == (1, 7, 3)

    multi = _toy_graphs(8)  # 5, 6, 7, 8 nodes, repeated
    shape = _shape(multi)
    assert shape["n_graphs"] == 8
    assert shape["n_nodes"] == pytest.approx(6.5)
    assert shape["n_edges"] == pytest.approx(13.0)
    assert shape["shape_sampled"] is False


# --------------------------------------------------------------------------- #
# ModelNet meshes become graphs; max-cut is its own generator
# --------------------------------------------------------------------------- #


def test_modelnet_transform_builds_edges_and_features() -> None:
    from graphnetz.datasets.vision import _mesh_to_graph

    mesh = Data(pos=torch.rand(4, 3), face=torch.tensor([[0, 0], [1, 2], [2, 3]]))
    graph = _mesh_to_graph()(mesh)
    assert graph.edge_index.size(1) == 10  # 5 undirected mesh edges, both directions
    assert torch.equal(graph.x, graph.pos)
    assert getattr(graph, "face", None) is None


def test_max_cut_is_a_distinct_generator_with_locally_optimal_labels(tmp_path: Path) -> None:
    from graphnetz.datasets.combinatorial import RandomColoring, RandomMaxCut

    maxcut = RandomMaxCut(str(tmp_path / "maxcut"), num_graphs=3)
    coloring = RandomColoring(str(tmp_path / "coloring"), num_graphs=3)
    assert RandomMaxCut is not RandomColoring
    assert maxcut.processed_file_names != coloring.processed_file_names

    g = maxcut[0]
    n = g.num_nodes
    w = torch.zeros(n, n)
    w[g.edge_index[0], g.edge_index[1]] = g.edge_attr.view(-1)
    assert torch.allclose(w, w.T)
    same = (g.y[:, None] == g.y[None, :]).float()
    gain = (w * same).sum(1) - (w * (1.0 - same)).sum(1)
    assert float(gain.max()) <= 1e-6
    cut = w[g.y[:, None] != g.y[None, :]].sum() / 2.0
    assert float(cut) == pytest.approx(float(g.cut_value))

    again = RandomMaxCut(str(tmp_path / "maxcut2"), num_graphs=3)
    assert torch.equal(again[0].y, g.y)


def test_multi_label_head_has_one_logit_per_label() -> None:
    from graphnetz.benchmark._runner import _graph_out_channels

    graphs = [
        Data(x=torch.randn(3, 2), edge_index=torch.tensor([[0, 1], [1, 2]]), y=torch.tensor([[0.0, 1.0, float("nan")]]))
        for _ in range(6)
    ]
    ds = _Graphs(graphs)
    assert _graph_out_channels(ds) == 3
    assert _graph_out_channels(_toy_graphs(6)) == 2
    report = _run("graph_cls", lambda _root: _Graphs(graphs * 3), seeds=(0,))
    assert "test_acc" in report.histories["toy"]["GCN"][0]
