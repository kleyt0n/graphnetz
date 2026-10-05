<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo-banner-dark.svg">
    <img src="assets/logo-banner.svg" alt="graphnetz" width="300">
  </picture>
</p>

<p align="center">Statistically rigorous benchmarking for graph neural networks.</p>

<p align="center">
  <a href="https://github.com/kleyt0n/graphnetz/actions/workflows/ci.yaml"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/kleyt0n/graphnetz/ci.yaml?branch=main&style=flat-square&label=ci"></a>
  <a href="https://pypi.org/project/graphnetz/"><img alt="PyPI" src="https://img.shields.io/pypi/v/graphnetz?style=flat-square&color=8da9c4"></a>
  <a href="https://kleyt0n.github.io/graphnetz/"><img alt="Docs" src="https://img.shields.io/badge/docs-online-8da9c4?style=flat-square"></a>
  <a href="https://arxiv.org/abs/2605.09099"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2605.09099-8da9c4?style=flat-square"></a>
  <a href="LICENCE.txt"><img alt="License" src="https://img.shields.io/badge/license-MIT-8da9c4?style=flat-square"></a>
</p>

---

graphnetz trains every *(task, model, seed)* triple through one pipeline and
returns a statistical report instead of an accuracy table:

- a Student's *t* confidence interval for every cell,
- paired *t*-tests or Wilcoxon signed-rank tests within each task, Holm-corrected,
- Friedman ranks and a Nemenyi critical difference across tasks,
- power, minimum detectable effect and equivalence tests, so a
  non-significant result can be read as "tied" or "underpowered".

The catalogue holds 62 dataset loaders across 10 domains and 4 task types,
with 5 architectures and a DGI pre-training utility.

**Documentation:** [kleyt0n.github.io/graphnetz](https://kleyt0n.github.io/graphnetz/)

## Install

```bash
uv add graphnetz                # core
uv add "graphnetz[ogb]"         # plus the OGB loaders
uv add "graphnetz[chem]"        # plus RDKit, for the molecular loaders
```

Requires Python 3.10+, `torch` 2.6+ and `torch-geometric` 2.6+.

For development:

```bash
git clone https://github.com/kleyt0n/graphnetz
cd graphnetz
uv sync --group dev
```

## Quick start

```python
from graphnetz import GAT, GCN, GraphSAGE, run_benchmark

report = run_benchmark(
    "social",
    {"GCN": GCN, "GAT": GAT, "GraphSAGE": GraphSAGE},
    seeds=range(10),
    task_type="node_cls",
)

report.summary()                        # mean and t-CI per (task, model)
report.pairwise()                       # Holm-corrected paired tests
report.friedman()                       # omnibus test on ranks across tasks
report.power()                          # what this design could detect
report.plot_critical_difference()       # Demšar diagram
report.to_latex("results.tex")          # booktabs table, row-best in bold
```

## The report

`run_benchmark` returns a `BenchmarkReport`. Every method reads the same
`X[task, model, seed]` tensor, so nothing downstream retrains.

| Method | Returns |
|---|---|
| `summary(ci=0.95)` | mean, std, sem and CI bounds per (task, model) |
| `pairwise(alpha=0.05)` | paired *t* or Wilcoxon tests within each task, raw and Holm *p*-values |
| `friedman(alpha=0.05)` | Friedman omnibus statistic and *p*-value across tasks |
| `power(alpha=0.05)` | minimum detectable effect and observed power per comparison |
| `equivalence(margin)` | two one-sided tests (TOST) with a verdict per comparison |
| `plot_critical_difference()` | Demšar diagram from Friedman ranks and the Nemenyi CD |
| `plot_pairwise(layout="matrix")` | pairwise significance, as a matrix or a list |
| `plot_forest()` | per-task forest plot of mean and CI |
| `plot_learning_curves()` | learning curves with CI bands |
| `to_latex(path)` | publication table, best per row in bold |
| `pairwise_to_latex(path)` | pairwise test table |
| `to_json(path)` | the full report, reloadable with `BenchmarkReport.from_json` |

See [Reading the report](https://kleyt0n.github.io/graphnetz/guides/report/).

## Tasks

| Task | Symbol | Metric |
|---|---|---|
| Node classification | `node_cls` | test accuracy |
| Graph classification | `graph_cls` | validation accuracy |
| Graph regression | `graph_reg` | validation MAE (lower is better) |
| Link prediction | `link_pred` | test ROC-AUC |

Unlabelled graphs (Netzschleuder, synthetic combinatorial, Ising lattice)
enter through link prediction on a held-out edge split, so every cell carries
a held-out metric.

## Datasets

| Category | Loaders | Tasks | Examples |
|---|---:|---|---|
| Combinatorial | 6 | LP | random TSP, VRP, max-flow, matching, coloring, max-cut |
| Biology | 12 | GC, GR, LP | MUTAG, PROTEINS, ENZYMES, Peptides, PPI, C. elegans, ogbg-molhiv† |
| Social | 16 | NC, LP | Cora, CiteSeer, PubMed, WikiCS, Roman-empire, Tolokers, ogbn-arxiv† |
| Knowledge | 3 | LP | FB15k-237, WordNet18-RR, WordNet |
| Infrastructure | 6 | LP | power grid, EuroRoad, US roads, EU airlines, London transport |
| Finance | 5 | NC, LP | Elliptic Bitcoin, product space, board interlocks, ogbn-products† |
| Computing | 4 | LP | Internet AS, AS-Skitter, route views |
| Vision | 4 | GC | MNIST and CIFAR-10 superpixels, ModelNet10/40 |
| Physics | 3 | GR, LP | QM9, ZINC, Ising lattice |
| Security | 3 | GC, LP | MalNet-Tiny, terrorist networks |

† Requires the `ogb` extra. OGB loaders live in their domain category.

```python
from graphnetz import Netz
from graphnetz.datasets.social import cora

ds = cora("data/cora")
streets = Netz(root="data", dataset_name="urban_streets", network_name="brasilia")
```

`Netz` loads any network from the [Netzschleuder](https://networks.skewed.de/)
archive. See [Datasets](https://kleyt0n.github.io/graphnetz/guides/datasets/).

## Models

| Model | Tasks | Reference |
|---|---|---|
| `GCN` | all four | Kipf & Welling, ICLR 2017 |
| `GAT` | all four | Veličković et al., ICLR 2018 |
| `GIN` | `graph_cls`, `graph_reg` | Xu et al., ICLR 2019 |
| `GraphSAGE` | all four | Hamilton et al., NeurIPS 2017 |
| `GraphTransformer` | all four | Shi et al., IJCAI 2021 |
| `DGI` | pre-training utility | Veličković et al., ICLR 2019 |

Node-level encoders reach graph-level and link tasks through adapters the
runner attaches automatically.

## Custom models

A model needs `__init__(in_channels, hidden_channels, out_channels)`,
`forward(data)` and a declared task type. It then runs with the same seeds,
splits and corrections as the built-ins.

```python
import torch
from graphnetz import GCN, register_model, run_benchmark


@register_model(task_type="node_cls")
class MyGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels): ...
    def forward(self, data): ...


report = run_benchmark("social", {"GCN": GCN, "MyGNN": MyGNN}, seeds=range(10))
```

Task types can also be declared with a `task_types` class attribute or an
inline `(cls, task, factory)` tuple. Custom datasets go in through
`run_benchmark(tasks=[Task(...)])`. See
[Custom models](https://kleyt0n.github.io/graphnetz/guides/custom-models/) and
[Custom datasets](https://kleyt0n.github.io/graphnetz/guides/custom-datasets/).

## Examples

| Notebook | Covers |
|---|---|
| [`01_benchmark.ipynb`](examples/01_benchmark.ipynb) | cross-category benchmark and report |
| [`02_knowledge.ipynb`](examples/02_knowledge.ipynb) | relational link prediction on FB15k-237 and WN18-RR |
| [`03_custom_artifacts.ipynb`](examples/03_custom_artifacts.ipynb) | your own model and dataset in the same pipeline |
| [`04_ogb.ipynb`](examples/04_ogb.ipynb) | the OGB loaders |

## Citation

```bibtex
@misc{dacosta2026graphnetz,
  title         = {{GraphNetz}: Statistical Benchmarking of Graph Neural Networks with Paired Tests and Rank Aggregation},
  author        = {Kleyton da Costa and Bernardo Modenesi},
  year          = {2026},
  eprint        = {2605.09099},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CE},
  url           = {https://arxiv.org/abs/2605.09099}
}
```

## License

MIT. See [`LICENCE.txt`](LICENCE.txt).
