# API reference

Auto-generated reference for the public surface of GraphNetz. The reference
is organised by module so deep links stay stable as the package grows; the
top-level convenience imports below cover everything you'll typically need.

## Top-level convenience imports

Importable directly from `graphnetz`:

**Models**

`GCN` · `GAT` · `GIN` · `GraphSAGE` · `GraphTransformer` · `DGI`
— see [Models](models.md).

**Datasets**

`CATEGORIES` · `Netz` · `download_all_networks_netz` · `list_datasets`
— see [Datasets](datasets.md).

**Benchmark**

`BENCHMARK_TASKS` · `BenchmarkReport` · `ModelSpec` · `Task` ·
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
