# API reference

Five modules, in the order they compose. Each page renders the module's own
docstring and then every public symbol, so a citation or a caveat sits beside
the implementation it describes.

<div class="grid cards" markdown>

-   __Running a benchmark__

    ---

    [`graphnetz.benchmark`](benchmark.md) · the runner, `BenchmarkReport`, the
    task catalogue and the model registry<br>
    [`graphnetz.training`](training.md) · one trainer per task family

-   __Data__

    ---

    [`graphnetz.datasets`](datasets.md) · 62 loaders across 10 categories<br>
    `Netz` · any network from the Netzschleuder catalogue<br>
    `validate_loaders` · does the catalogue actually load?

-   __Models__

    ---

    [`graphnetz.models`](models.md) · GCN, GAT, GIN, GraphSAGE,
    GraphTransformer, DGI<br>
    `_adapters` · the wrappers that reach every task type

-   __Output__

    ---

    [`graphnetz.plotting`](plotting.md) · the brand and figure palettes,
    figure helpers, plot builders

</div>

## Top-level convenience imports

Importable directly from `graphnetz`:

**Models**

`GCN` · `GAT` · `GIN` · `GraphSAGE` · `GraphTransformer` · `DGI`
— see [Models](models.md).

**Datasets**

`CATEGORIES` · `Netz` · `download_all_networks_netz` · `list_datasets` ·
`validate_loaders`
— see [Datasets](datasets.md).

**Benchmark**

`BENCHMARK_TASKS` · `BenchmarkReport` · `ModelSpec` · `SearchSpace` · `Task` ·
`iter_benchmark_tasks` · `plot_benchmark` · `register_model` ·
`register_task` · `run_benchmark` · `task_from_dataset` · `unregister_task`
— see [Benchmark](benchmark.md).

**Training utilities**

`train_node_classification` · `train_graph_classification` ·
`train_graph_regression` · `train_link_prediction` · `train_dgi` ·
`train_node_degree_regression` · `train_relational_link_prediction`
— see [Training](training.md).

**Plotting**

`figure` · `panel_label` · `plot_grouped_bars` · `plot_history` ·
`save_figure` · `set_plot_style`
— see [Plotting](plotting.md).

## Citations

Every model carries its reference in its own docstring, so the paper sits
beside the code:

```python
help(graphnetz.GIN)
```

| model | paper |
| --- | --- |
| [GCN](../models/gcn.md) | Kipf & Welling, ICLR 2017 · [arXiv:1609.02907](https://arxiv.org/abs/1609.02907) |
| [GAT](../models/gat.md) | Veličković et al., ICLR 2018 · [arXiv:1710.10903](https://arxiv.org/abs/1710.10903) |
| [GIN](../models/gin.md) | Xu et al., ICLR 2019 · [arXiv:1810.00826](https://arxiv.org/abs/1810.00826) |
| [GraphSAGE](../models/graphsage.md) | Hamilton, Ying & Leskovec, NeurIPS 2017 · [arXiv:1706.02216](https://arxiv.org/abs/1706.02216) |
| [GraphTransformer](../models/graph-transformer.md) | Shi et al., IJCAI 2021 · [arXiv:2009.03509](https://arxiv.org/abs/2009.03509) |
| [DGI](../models/dgi.md) | Veličković et al., ICLR 2019 · [arXiv:1809.10341](https://arxiv.org/abs/1809.10341) |

The statistical procedures carry theirs too:

| procedure | reference |
| --- | --- |
| Friedman + Nemenyi critical difference | Demšar, *JMLR* 7:1-30, 2006 |
| Wilcoxon signed-rank for classifier comparison | Benavoli et al., *JMLR* 17(5):1-36, 2016 |
| Holm step-down correction | Holm, *Scand. J. Statist.* 6(2):65-70, 1979 |
