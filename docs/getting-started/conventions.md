# Conventions

The vocabulary the rest of the documentation assumes. Everything here is a
default you can override; knowing what it is saves reading the source later.

## Naming

| term | means |
| --- | --- |
| **category** | one of ten research domains — `social`, `biology`, `physics`, … |
| **task type** | one of four families — `node_cls`, `graph_cls`, `graph_reg`, `link_pred` |
| **task** | a `Task(name, task_type, loader, epochs)`: a dataset plus how to train on it |
| **cell** | one *(task, model)* pair, summarised over its seeds |
| **run** | one *(task, model, seed)* triple: a single training |

A **report** is what `run_benchmark` returns. It is never a table of
accuracies; it is the raw metric tensor plus the methods that reduce it.

## The model contract

Two methods, and nothing else:

```python
class MyModel(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int): ...
    def forward(self, data): ...     # data is a PyG Data object
```

`in_channels` and `out_channels` come from the dataset; `hidden_channels`
comes from the benchmark and defaults to **64**. A node-level encoder returns
per-node features and reaches the other three task families through
[adapters](../models/adapters.md).

## Defaults worth knowing

| setting | default | override |
| --- | --- | --- |
| seeds | `(0, 1, …, 9)` | `seeds=range(20)`, or any iterable |
| hidden width | `64` | `hidden_channels=` |
| epochs | per-task, from the catalogue (`Task.epochs`, itself defaulting to 30) | `epochs=` |
| device | `"auto"` — CUDA, then MPS, then CPU | `device="cpu"` or any `torch.device` |
| cache root | `data/benchmark/<category>/<task>/` | `root=` |
| CI method | Student's *t* | `method="bootstrap"`, or `report.ci_method` |
| pairwise test | paired *t* | `method="wilcoxon"`, or `report.pairwise_method` |
| scoring epoch | the final one | `report.epoch_selection = "best_val"` |
| correction | Holm, within each task | — |

`seeds=range(10)` is the reproducible default sweep and the setting every
number in [Findings](../findings.md) was produced under.

## Metric direction

Three of the four families are higher-is-better. `graph_reg` reports MAE and
is **lower**-is-better.

The rank aggregation normalises direction per task before averaging, so a
regression task can share a critical-difference diagram with classification
tasks. A pairwise table does not normalise: `mean_diff = mean(a) − mean(b)`,
so on MAE a *negative* difference means `a` is the better model. This is the
one place where reading a sign carelessly gives the opposite conclusion.

## Argument order

`run_benchmark` takes `category` first and `models` second, both positionally:

```python
run_benchmark("social", {"GCN": GCN, "GAT": GAT}, seeds=range(10))
```

A single class is accepted where a dict is expected, for one-off checks:

```python
run_benchmark("infrastructure", GAT, task_type="link_pred")
```

Passing `tasks=[Task(...)]` bypasses the catalogue entirely; `category` then
defaults to `"custom"` and is used only to namespace cache directories.

## Where state lives

GraphNetz keeps two global registries, and both have explicit escape hatches
because global state in a benchmark is a reproducibility hazard:

| registry | holds | mutated by | bypassed by |
| --- | --- | --- | --- |
| `BENCHMARK_TASKS` | the curated tasks per category | `register_task` / `unregister_task` | `tasks=` |
| the model registry | which task types each model serves | `register_model`, or a `task_types` class attribute | an inline `(cls, task_type, factory)` tuple |

For a one-off experiment prefer the bypasses. They leave no trace between
runs, which means a notebook cell executed twice cannot produce two different
benchmarks.

## Type conventions

- Datasets are **PyTorch Geometric** objects: `ds[0]` returns a `Data`, and
  the task-relevant attributes are `num_features`, `num_classes` or
  `num_relations`.
- Trainers return a per-epoch **history dict**, ready for
  [`plot_history`][graphnetz.plotting.plot_history].
- `report.final_metrics()` returns a nested `dict[task][model] -> list[float]`
  indexed by seed order. That is the canonical access path for your own
  analysis.

## Next

- [Quickstart](quickstart.md) — run the defaults above.
- [Concepts](../concepts/index.md) — what the statistics mean.
- [Benchmark protocol](../guides/benchmark.md) — the five stages in full.
