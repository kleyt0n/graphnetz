# Custom datasets

Your dataset gets the same statistical pipeline as the built-ins. The
contract is the standard PyTorch Geometric one, so anything you already have
loading into PyG is close to done.

## The contract

- `ds[0]` returns a `Data` object,
- plus the attributes the task needs: `num_features` and `num_classes` for
  classification, `num_relations` for relational link prediction.

The per-task conventions — which attributes are required, how splits are
encoded — live in the trainer docstrings:
[`train_node_classification`][graphnetz.training.train_node_classification],
[`train_graph_classification`][graphnetz.training.train_graph_classification],
[`train_graph_regression`][graphnetz.training.train_graph_regression],
[`train_link_prediction`][graphnetz.training.train_link_prediction] and
[`train_relational_link_prediction`][graphnetz.training.train_relational_link_prediction].

## Quickest path: `tasks=`

Wrap an already-loaded dataset and hand it straight to the runner:

```python
from graphnetz import GCN, run_benchmark, task_from_dataset

ds = my_loader("data/my_dataset")          # any PyG-shaped object

task = task_from_dataset("my_dataset", "node_cls", ds, epochs=100)
report = run_benchmark(
    models={"GCN": GCN},
    tasks=[task],
    seeds=range(10),
)
```

No `BENCHMARK_TASKS` mutation and no global state: `tasks=` bypasses the
registry entirely. `category` defaults to `"custom"`, and is used only to
namespace cache paths.

This is the right default for research code. A notebook cell that registers a
task globally behaves differently the second time it is run; one that passes
`tasks=` does not.

## Permanent registration

When the dataset should be visible to `run_benchmark(category, ...)` and to
`iter_benchmark_tasks`:

```python
from graphnetz import register_task, task_from_dataset, unregister_task

register_task("biology", task_from_dataset("my_assay", "graph_cls", ds, epochs=50))

# ... later, if you want it gone again:
unregister_task("biology", "my_assay")
```

`unregister_task` exists because registration is global. If a test suite or a
long-lived session registers a task, everything downstream in that process
sees it — including a later `run_benchmark("biology", ...)` that did not
expect it.

## Seed-aware loaders

For synthetic datasets, a fresh sample per seed makes cross-seed variance
reflect **both** model initialisation and data resampling — which is usually
what you want a confidence interval to cover. Write a loader taking a `seed`
keyword; the dispatcher detects it via `inspect.signature` and threads the
benchmark seed through:

```python
from graphnetz.benchmark import Task

def my_loader(root: str, *, seed: int):
    return MySyntheticDataset(root, num_graphs=100, seed=seed)

task = Task("synthetic_g100", "graph_cls", my_loader, epochs=20)
report = run_benchmark(models={"GCN": GCN}, tasks=[task], seeds=range(10))
```

A loader without the keyword is simply called `f(root)`, and its data is
constant across seeds by construction. Both are legitimate; they answer
slightly different questions, and the built-in combinatorial loaders take the
seed-aware form deliberately. See
[Reproducibility](../concepts/reproducibility.md#what-gets-reseeded-and-when).

## Netzschleuder networks

Any network in the [Netzschleuder catalogue](https://networks.skewed.de/) is
available without writing a loader at all:

```python
from graphnetz import Netz

ds = Netz(root="data", dataset_name="urban_streets", network_name="brasilia")
data = ds[0]

# Multiplex / transit / airline networks need parallel-edge support:
ds_air = Netz(
    root="data",
    dataset_name="eu_airlines",
    network_name="eu_airlines",
    multigraph=True,
)
```

These arrive unlabelled, so they enter the benchmark through `link_pred` on a
held-out edge split. See
[Tasks and metrics](../concepts/tasks.md#why-there-is-no-self-supervised-headline).

## Checking it before you trust it

A dataset that loads is not the same as a dataset that loads *correctly* for
the task you assigned it. `validate_loaders` instantiates entries and reports
what happened, including for tasks you registered yourself:

```python
from graphnetz import validate_loaders

matrix = validate_loaders(["biology"])
matrix[["category", "loader", "status", "n_nodes", "seconds"]]
```

See [Auditing the catalogue](datasets.md#auditing-the-catalogue) for what the
statuses mean and why a dead mirror is classified apart from a defect.

## Next

- [Dataset taxonomy](datasets.md) — the 62 built-in loaders and what they serve.
- [Custom models](custom-models.md) — the other half of the same problem.
- [Contributing](contributing.md) — upstreaming a loader into the catalogue.
