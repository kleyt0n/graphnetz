# Custom models & datasets

Custom models and datasets go through the same statistical pipeline as the
built-ins — multi-seed, Holm-corrected, CD-diagrammed.

## Plug in your own model

Once a model declares which task it supports, three integration paths cover
the common cases:

**Decorator** — permanent registration at import time. Best for libraries
or shared modules:

```python
import torch
from torch_geometric.nn import GCNConv
from graphnetz import register_model

@register_model(task_type={"node_cls"})
class MyGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x, ei = data.x, data.edge_index
        return self.conv2(torch.relu(self.conv1(x, ei)), ei)

run_benchmark("social", {"MyGNN": MyGNN}, task_type="node_cls", seeds=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
```

**Class attribute** — same effect, no decorator dependency:

```python
class MyGNN(torch.nn.Module):
    task_types = {"node_cls"}
    ...
```

**Inline tuple** — one-shot variants for hyperparameter sweeps. The third
slot is a factory `(in_channels, hidden_channels, out_channels) -> Module`:

```python
run_benchmark(
    "social",
    {
        "MyGNN-d0.3": (MyGNN, "node_cls", lambda i, h, o: MyGNN(i, h, o, dropout=0.3)),
        "MyGNN-d0.5": (MyGNN, "node_cls", lambda i, h, o: MyGNN(i, h, o, dropout=0.5)),
    },
)
```

For node-level encoders that should run on **all four** task types without
writing the adapter glue, see the
[multi-task factory](../guide/models.md#multi-task-factory).

## Plug in your own dataset

Custom datasets get the same statistical pipeline as the built-ins.
The minimal contract is the standard PyG one — your dataset object exposes
`ds[0]` returning a `Data`, plus the relevant attributes for the task
(`num_features`, `num_classes`, or `num_relations`).

**Quickest path** — wrap an already-loaded dataset and pass it via
`tasks=`:

```python
from graphnetz import GCN, run_benchmark, task_from_dataset

# Your dataset (any PyG-shaped object).
ds = my_loader("data/my_dataset")

task = task_from_dataset("my_dataset", "node_cls", ds, epochs=100)
report = run_benchmark(
    models={"GCN": GCN},
    tasks=[task],
    seeds=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
)
```

No `BENCHMARK_TASKS` mutation, no global state — `tasks=` bypasses the
registry entirely. `category` defaults to `"custom"` for cache-path
namespacing.

**Permanent registration** — make your dataset visible to
`run_benchmark(category, ...)` and `iter_benchmark_tasks`:

```python
from graphnetz import register_task, task_from_dataset, unregister_task

register_task("biology", task_from_dataset("my_assay", "graph_cls", ds, epochs=50))

# ... later, if you want to remove it:
unregister_task("biology", "my_assay")
```

**Seed-aware loaders** — for synthetic datasets where each seed should
produce a fresh sample, write a loader that takes a `seed` keyword. The
dispatcher detects it via `inspect.signature` and passes the benchmark
seed in:

```python
from graphnetz.benchmark import Task

def my_loader(root: str, *, seed: int):
    return MySyntheticDataset(root, num_graphs=100, seed=seed)

task = Task("synthetic_g100", "graph_cls", my_loader, epochs=20)
report = run_benchmark(models={"GCN": GCN}, tasks=[task], seeds=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
```

!!! tip
    The conventions for each task (which attributes the dataset must
    expose, how splits are encoded) live in the trainer docstrings:
    `graphnetz.train_node_classification`,
    `graphnetz.train_graph_classification`,
    `graphnetz.train_graph_regression`,
    `graphnetz.train_link_prediction`, and
    `graphnetz.train_relational_link_prediction`.
