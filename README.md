<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo-banner-dark.svg">
    <img src="assets/logo-banner.svg" alt="GraphNetz" width="320">
  </picture>
</p>

<p align="center"><em>Statistically rigorous GNN benchmarking</em></p>

<p align="center">
  <a href="https://github.com/Kleyt0n/graphnetz/actions"><img alt="Build" src="https://img.shields.io/badge/build-passing-212529?style=flat-square&labelColor=e9ecef"></a>
  <a href="https://kleyt0n.github.io/graphnetz/"><img alt="Docs" src="https://img.shields.io/badge/passing-docs-212529?style=flat-square&labelColor=e9ecef"></a>
  <a href="https://www.python.org/downloads/"><img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-212529?style=flat-square&labelColor=e9ecef"></a>
  <a href="LICENCE.txt"><img alt="License" src="https://img.shields.io/badge/license-MIT-212529?style=flat-square&labelColor=e9ecef"></a>
  <a href="https://arxiv.org/pdf/2605.09099"><img alt="Paper" src="https://img.shields.io/badge/paper-PDF-212529?style=flat-square&labelColor=e9ecef"></a>
</p>

---

## Why GraphNetz

Whether you are proposing a new GNN architecture, testing a model on a new graph domain, or comparing existing methods across graph types, GraphNetz turns the usual "train, evaluate, table of accuracies" workflow into a proper statistical report. It gives you confidence intervals for each result, paired model comparisons with multiple-testing correction, and rank-based summaries across datasets via critical-difference diagrams. The point isn't to crown a leaderboard winner. It's to give researchers a way to quantify uncertainty, compare methods fairly, and produce the evidence reviewers actually ask for in graph-learning papers.


Most GNN benchmarks report point-estimate accuracies on a handful of citation graphs and declare a winner without confidence intervals, multiple-comparison correction, or rank aggregation across datasets. GraphNetz's default output is a **structured statistical report**, not a raw accuracy table:

- multi-seed Student's *t* confidence intervals per cell,
- Holm–Bonferroni paired *t*-tests (or Wilcoxon signed-rank) within each task,
- Demšar critical-difference diagrams from Friedman ranks with a Nemenyi post-hoc.

The catalogue is organised along a **category × task** taxonomy: 

- 62 dataset loaders across 10 scientific categories
- 4 task types (node classification, graph classification, graph regression, link prediction)
- 5 canonical architectures (GCN, GAT, GIN, GraphSAGE, Graph Transformer) plug into every tasl via a small set of task adapters;

## Install

```bash
uv add graphnetz
# or, in an existing environment:
pip install graphnetz
```

For local development:

```bash
git clone https://github.com/Kleyt0n/graphnetz
cd graphnetz
uv sync --group dev
```

GraphNetz requires Python ≥ 3.10, `torch ≥ 2.6`, and `torch-geometric ≥ 2.6`.

## Quick start

```python
from graphnetz import GCN, train_node_classification, plot_history
from graphnetz.datasets.social import cora

ds = cora("data/cora")
model = GCN(ds.num_features, 64, ds.num_classes)
history = train_node_classification(model, ds[0], epochs=200)
fig, ax = plot_history(history, title="GCN on Cora")
```

For a full benchmark run with the default statistical report:

```python
from graphnetz import GAT, GCN, GraphSAGE, GraphTransformer, run_benchmark

report = run_benchmark(
    "social",
    {"GCN": GCN, "GAT": GAT, "GraphSAGE": GraphSAGE, "GraphTransformer": GraphTransformer},
    seeds=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    task_type="node_cls",          # restrict to one task family
)
print(report.summary())       # per-(task, model) mean ± t-CI
print(report.pairwise())      # Holm-corrected paired t-tests (or Wilcoxon)
fig, _ = report.plot_critical_difference(alpha=0.05)
```

## Tasks

| Kind | Symbol | Metric | Examples |
|---|---|---|---|
| Node classification | `node_cls` | test accuracy | Cora, Roman-empire |
| Graph classification | `graph_cls` | val accuracy | MUTAG, MNIST-superpixels |
| Graph regression | `graph_reg` | val MAE | ZINC, QM9 |
| Link prediction | `link_pred` | test AUC | FB15k-237, Internet AS |

Unlabelled graphs (Netzschleuder, synthetic combinatorial, Ising lattice)
enter the benchmark through link prediction on a held-out edge split, so
every cell carries a real test-time metric; there is no self-supervised
*pretext* loss in the headline report.

## Dataset categories

| Category | # | Tasks | Loaders |
|---|---:|---|---|
| Combinatorial | 6 | LP | random TSP, VRP, max-flow, bipartite matching, coloring, max-cut |
| Biology | 12 | GC, GR, LP | MUTAG, PROTEINS, ENZYMES, Peptides-func/struct, PPI, C. elegans, Budapest connectome, hospital/high-school contacts, ogbg-molhiv†, ogbg-molpcba† |
| Social | 16 | NC, LP | Cora, CiteSeer, PubMed, WikiCS, Roman-empire, Amazon-ratings, Minesweeper, Tolokers, Questions, MovieLens-100k, Karate, Facebook friends, DBLP coauthor, DNC emails, ogbn-arxiv†, ogbl-collab† |
| Knowledge | 3 | LP | FB15k-237, WordNet18-RR, WordNet (Netz) |
| Infrastructure | 6 | LP | power grid, EuroRoad, US roads, EU airlines, London transport, urban streets |
| Finance | 5 | NC, LP | Elliptic Bitcoin, product space, board of directors, US patents, ogbn-products† |
| Computing | 4 | LP | Internet AS, Internet topology, AS-Skitter, route views |
| Vision | 4 | GC | MNIST/CIFAR-10 superpixels, ModelNet10/40 |
| Physics | 3 | GR, LP | QM9, ZINC, Ising lattice |
| Security | 3 | GC, LP | MalNet-Tiny, 9/11 terrorists, train terrorists |

† Requires the optional `ogb` extra (`pip install graphnetz[ogb]`). The
five OGB loaders are folded into their domain categories rather than
exposed as a separate `ogb` category, so they appear in
`run_benchmark(category, ...)` alongside the curated built-ins.

```python
from graphnetz.datasets.social import cora, roman_empire
from graphnetz.datasets.biology import peptides_func
from graphnetz.datasets.computing import internet_as

# Optional OGB loaders live in their domain modules (require `pip install graphnetz[ogb]`):
from graphnetz.datasets.social import ogbn_arxiv     # node_cls
from graphnetz.datasets.biology import ogbg_molhiv   # graph_cls

ds_cora = cora("data/cora")
ds_rom  = roman_empire("data/roman_empire")        # heterophilic
ds_pep  = peptides_func("data/peptides_func")      # LRGB
ds_inet = internet_as("data/internet_as")          # Netzschleuder
```

For arbitrary [Netzschleuder](https://networks.skewed.de/) networks:

```python
from graphnetz import Netz
ds = Netz(root="data", dataset_name="urban_streets", network_name="brasilia")
```

## Models

| Model | Kinds | Source |
|---|---|---|
| `GCN`  | all four | Kipf & Welling, ICLR 2017 |
| `GAT`  | all four | Veličković et al., ICLR 2018 |
| `GIN`  | `graph_cls`, `graph_reg` | Xu et al., ICLR 2019 |
| `GraphSAGE` | all four | Hamilton et al., NeurIPS 2017 |
| `GraphTransformer` | all four | Shi et al., 2021 |
| `DGI` | *(utility)* | Veličković et al., ICLR 2019 |

Node-level encoders enter every task through three small adapters:
graph-level pooling head, dot-product link-prediction head, and the DGI
self-supervised wrapper for optional unsupervised pre-training.

## Bring your own model

The contract is two methods: `__init__(in_channels, hidden_channels,
out_channels)` and `forward(data)` taking a PyG `Data`. Declare which task
your model supports and it becomes a first-class citizen of the benchmark —
same seeds, same splits, same corrections as the built-ins.

```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from graphnetz import GAT, GCN, register_model, run_benchmark


@register_model(task_type="node_cls")
class ResGCN(torch.nn.Module):
    """Three-layer GCN with a residual hop — your model goes here."""

    def __init__(self, in_channels, hidden_channels, out_channels, *, dropout=0.5):
        super().__init__()
        self.inp = GCNConv(in_channels, hidden_channels)
        self.mid = GCNConv(hidden_channels, hidden_channels)
        self.out = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.inp(x, edge_index))
        x = F.relu(self.mid(x, edge_index)) + x          # residual hop
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.out(x, edge_index)


report = run_benchmark(
    "social",
    {"GCN": GCN, "GAT": GAT, "ResGCN": ResGCN},
    only=["cora"],
    task_type="node_cls",
    seeds=range(10),
)
print(report.summary())     # mean ± t-CI, your model beside the baselines
print(report.pairwise())    # is it *really* better, after Holm correction?
```

`summary()` returns one row per (task, model); `pairwise()` returns every
comparison with its raw and Holm-adjusted *p*-value, so a claim like "ResGCN
beats GCN" either survives correction or it does not:

```
             n_seeds      mean       std       sem    ci_low   ci_high
task model
cora GAT           3  0.802333  0.013013  0.007513  0.770008  0.834659
     GCN           3  0.790667  0.003215  0.001856  0.782681  0.798652
     ResGCN        3  0.790667  0.011150  0.006438  0.762967  0.818366
```

Three ways to declare the task, depending on how permanent the model is:

```python
# 1. Decorator — permanent registration at import time.
@register_model(task_type="node_cls")
class MyGNN(torch.nn.Module): ...

# 2. Class attribute — same effect, no import-time dependency on graphnetz.
class MyGNN(torch.nn.Module):
    task_types = {"node_cls", "graph_cls"}

# 3. Inline tuple — one-shot variants, e.g. a hyperparameter sweep. The third
#    slot is a factory (in_channels, hidden_channels, out_channels) -> Module.
run_benchmark(
    "social",
    {
        "MyGNN-d0.3": (MyGNN, "node_cls", lambda i, h, o: MyGNN(i, h, o, dropout=0.3)),
        "MyGNN-d0.5": (MyGNN, "node_cls", lambda i, h, o: MyGNN(i, h, o, dropout=0.5)),
    },
)
```

Custom **datasets** follow the same pattern — pass `tasks=[Task(...)]` to
`run_benchmark` to bypass the built-in catalogue entirely. See
[Custom models & datasets](https://kleyt0n.github.io/graphnetz/getting-started/custom/).

## The statistical report

`run_benchmark(...)` returns a `BenchmarkReport` with the following methods:

| Method | Output |
|---|---|
| `report.summary(ci=0.95)` | per-(task, model) mean ± *t*-CI half-width DataFrame |
| `report.pairwise(alpha=0.05)` | Holm-corrected paired *t*-tests or Wilcoxon signed-rank tests within each task |
| `report.plot_critical_difference()` | Demšar / Nemenyi CD diagram across tasks |
| `report.plot_pairwise(layout=...)` | matrix or list view of pairwise significance |
| `report.plot_forest()` | per-task forest plot of mean ± CI |
| `report.plot_learning_curves()` | shared-y learning curves with t-CI bands |
| `report.to_latex(path)` | publication-ready bold-best LaTeX table |
| `report.pairwise_to_latex(path)` | Holm pairwise LaTeX table (parametric or non-parametric) |

## Notebooks

Worked examples live under `examples/`:

- `01_benchmark.ipynb`: the cross-category dashboard (multi-seed report,
  bootstrap CIs, custom-model integration).
- `02_knowledge.ipynb`: relational link prediction on FB15k-237 / WN18-RR
  using the DistMult decoder.

## Contributing

Pull requests welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md) first. The
short version: every benchmark cell must carry a real held-out metric,
every change must thread through the multi-seed pipeline, and every PR must
be `ruff` clean.

```bash
uv run pytest
uv run ruff check
```

## Citation

If GraphNetz is useful in your work, please cite the accompanying paper:

```bibtex
@misc{dacosta2026graphnetz,
  title={GraphNetz: Statistical Benchmarking of Graph Neural Networks with Paired Tests and Rank Aggregation}, 
  author={Kleyton da Costa and Bernardo Modenesi},
  year={2026},
  eprint={2605.09099},
  archivePrefix={arXiv},
  primaryClass={cs.CE},
  url={https://arxiv.org/abs/2605.09099}, 
}
```

## License

MIT. See [`LICENCE.txt`](LICENCE.txt).
