# Custom models

Your model runs through the *same* pipeline as the built-ins — same seeds,
same splits, same Holm correction, same critical-difference diagram. There is
no separate code path for user models, which is the point: a comparison
against baselines is only worth anything if the baselines were not given a
different protocol.

## The contract

Two methods:

```python
class MyGNN(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int): ...
    def forward(self, data): ...     # data is a PyG Data object
```

plus a declaration of which [task types](../concepts/tasks.md) it supports.
`data` is a PyG `Data`; read what you need from it. The runner owns the
split, the optimiser and the seeding.

## Three ways to register

### Decorator

Permanent registration at import time. Best for libraries and shared modules,
where the model should be visible to every `run_benchmark` call by name:

```python
import torch
from torch_geometric.nn import GCNConv

from graphnetz import register_model, run_benchmark


@register_model(task_type={"node_cls"})
class MyGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x, ei = data.x, data.edge_index
        return self.conv2(torch.relu(self.conv1(x, ei)), ei)


run_benchmark("social", {"MyGNN": MyGNN}, task_type="node_cls", seeds=range(10))
```

`register_model` validates the task types against `TASK_TYPES` and raises on
an unknown one, rather than registering a model the dispatcher would silently
never schedule.

### Class attribute

The same effect with no import-time dependency on `register_model` — useful
when you are benchmarking someone else's encoder and want to touch their code
as little as possible:

```python
class MyGNN(torch.nn.Module):
    task_types = {"node_cls", "graph_cls"}
    ...
```

### Inline tuple (one-shot variants)

A `(cls, task_type, factory)` triple, where the factory is
`(in_channels, hidden_channels, out_channels) -> Module`. Nothing is
registered globally, so a notebook cell run twice cannot accumulate state.
This is the path for hyperparameter variants, each of which is just another
"model":

```python
run_benchmark(
    "social",
    {
        "MyGNN-d0.3": (MyGNN, "node_cls", lambda i, h, o: MyGNN(i, h, o, dropout=0.3)),
        "MyGNN-d0.5": (MyGNN, "node_cls", lambda i, h, o: MyGNN(i, h, o, dropout=0.5)),
    },
)
```

For a real search with validation-based selection, use a
[`SearchSpace`](search.md) instead: the inline-tuple approach compares
variants as if they were different models, which is fine for two but does not
scale and does not record what was selected.

### Choosing

| scenario | use |
| --- | --- |
| publishing a new architecture | **decorator** — clean import surface, name-based discovery |
| benchmarking someone else's encoder | **class attribute** — no edits beyond adding `task_types` |
| sweeping a couple of variants | **inline tuple** — one factory per variant |
| tuning properly | [**`SearchSpace`**](search.md) — an inner loop scored on validation |
| a node-level encoder that should run everywhere | **multi-task factory**, below |

## All four task types, for free

A node-level encoder has no idea what graph classification is, and should not
have to. `_multi_task_factory` wraps it in the right
[adapter](../models/adapters.md) per task:

```python
from graphnetz.benchmark import _multi_task_factory, register_model

class MyEncoder(torch.nn.Module):
    """Returns per-node embeddings of shape [N, out_channels]."""
    ...

_ALL = {"node_cls", "graph_cls", "graph_reg", "link_pred"}
register_model(MyEncoder, task_type=_ALL, factory=_multi_task_factory(MyEncoder))
```

Two requirements on the encoder: return **per-node** features, and do not
read `data.batch` (the wrapper does that). See
[Writing an encoder that adapts cleanly](../models/adapters.md#writing-an-encoder-that-adapts-cleanly).

## A worked example

Your model beside the baselines, with the comparison that decides whether the
difference is real:

```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from graphnetz import GAT, GCN, register_model, run_benchmark


@register_model(task_type="node_cls")  # (1)!
class ResGCN(torch.nn.Module):
    """Three-layer GCN with a residual hop — your model goes here."""

    def __init__(self, in_channels, hidden_channels, out_channels, *, dropout=0.5):
        super().__init__()
        self.inp = GCNConv(in_channels, hidden_channels)
        self.mid = GCNConv(hidden_channels, hidden_channels)
        self.out = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, data):  # (2)!
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.inp(x, edge_index))
        x = F.relu(self.mid(x, edge_index)) + x
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.out(x, edge_index)


report = run_benchmark(
    "social",
    {"GCN": GCN, "GAT": GAT, "ResGCN": ResGCN},  # (3)!
    only=["cora"],
    task_type="node_cls",
    seeds=range(10),
)
print(report.summary())     # mean ± t-CI, your model beside the baselines
print(report.pairwise())    # is it *really* better, after Holm correction?
print(report.power())       # and could this design have told you either way?
```

1. Or set `task_types = {"node_cls"}` as a class attribute, or pass an inline
   tuple `(cls, task_type, factory)` for one-shot variants.
2. `data` is a PyG `Data` object. Read what you need from it; the runner owns
   the split, the optimiser, and the seeding.
3. Baselines are ordinary entries in the same dict. Nothing about your model
   takes a different code path.

The payoff is the second and third `print`. A difference in means that does
not survive Holm correction never becomes a claim — and a *non*-difference
that the design could never have detected does not become a claim of
equivalence either. See [Adequacy](../concepts/adequacy.md).

## Next

- [Custom datasets](custom-datasets.md) — the other half of the same problem.
- [Adapters](../models/adapters.md) — what wraps your encoder, and when.
- [Hyperparameter search](search.md) — when shared settings are not a fair test.
- [Contributing](contributing.md) — upstreaming a model into the catalogue.
