# Quickstart

This page takes you from a working install to a five-seed benchmark with a
LaTeX-ready summary table and a critical-difference diagram.

## Train one model

The single-task trainers accept any `nn.Module` and return a per-epoch
history dict ready for plotting:

```python
from graphnetz import GCN, train_node_classification, plot_history
from graphnetz.datasets.social import cora

ds = cora("data/cora")
model = GCN(ds.num_features, 64, ds.num_classes)
history = train_node_classification(model, ds[0], epochs=200)
fig, ax = plot_history(history, title="GCN on Cora")
```

Use this when you only need *one* model on *one* dataset and don't care about
cross-seed variance. For everything else — multi-seed, multi-task,
multi-model — reach for `run_benchmark`.

## Run a multi-seed benchmark

```python
from graphnetz import GAT, GCN, GraphSAGE, GraphTransformer, run_benchmark

report = run_benchmark(
    "social",
    {"GCN": GCN, "GAT": GAT, "GraphSAGE": GraphSAGE, "GraphTransformer": GraphTransformer},
    seeds=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    task_type="node_cls",
)

print(report.summary())          # per-(task, model) mean ± t-CI
print(report.pairwise())         # Holm-corrected paired t-tests
fig, _ = report.plot_critical_difference(alpha=0.05)
report.to_latex("results.tex")   # publication-ready table
```

The same call works for any category and task family: pass
`task_type="graph_cls"` to benchmark on graph classification,
`task_type="link_pred"` for link prediction, and so on. Pass
`only=[task_name, ...]` to restrict to specific loaders.

## Five-minute tour

1. **Pick a category.** `combinatorial`, `biology`, `social`, `knowledge`,
   `infrastructure`, `finance`, `computing`, `vision`, `physics`, `security`.
   Installing the optional `ogb` extra adds OGB datasets to the existing
   domain categories (e.g. `ogbn-arxiv` joins `social/node_cls`,
   `ogbg-molhiv` joins `biology/graph_cls`).
2. **Pick a task.** `node_cls`, `graph_cls`, `graph_reg`, or
   `link_pred`. The runner skips models that don't declare support for the
   chosen task.
3. **Pick architectures.** Any subset of the five built-ins, or your own —
   see [Models & adapters](../guide/models.md#custom-models).
4. **Run.** `run_benchmark(category, models, task_type=..., seeds=...)`. Use
   `seeds=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)` for the default reproducible 10-seed sweep.
5. **Report.** Call `summary`, `pairwise`, `plot_critical_difference`,
   `plot_pairwise`, `plot_forest`, `plot_learning_curves`, `to_latex`,
   `pairwise_to_latex`. Every method works on the same
   `BenchmarkReport`.

## Next steps

- Bring your own encoder or dataset: [Custom models & datasets](custom.md).
- Browse the [dataset taxonomy](../guide/datasets.md) to find loaders that
  match your domain.
- Read [Reading the report](../guide/report.md) to learn which plot or table
  answers which question.
- Skim [Contributing](../contributing.md) before adding a new loader, model,
  or task so your additions thread through the same pipeline.
