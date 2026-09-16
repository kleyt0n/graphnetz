# Models

GraphNetz follows one rule: **a node-level encoder should run on every task
without rewriting code**. Five architectures ship as benchmark models and one
as a pre-training utility; four of the five are node-level encoders that
reach graph classification, graph regression and link prediction through
[adapters](adapters.md) the runner attaches for them.

All of them share the same constructor signature, which is the whole contract
the benchmark asks of a model:

```python
Model(in_channels: int, hidden_channels: int, out_channels: int)
```

plus a `forward(data)` taking a PyTorch Geometric `Data` object. Anything
satisfying those two runs through the same pipeline as the built-ins — same
seeds, same splits, same Holm correction, same CD diagram. See
[Custom models](../guides/custom-models.md).

## The five benchmark architectures

| model | depth | aggregation | supported tasks |
| --- | --- | --- | --- |
| [`GCN`](gcn.md) | 2 layers | degree-normalised mean | all four |
| [`GAT`](gat.md) | 2 layers, 8 heads | learned attention | all four |
| [`GIN`](gin.md) | 3 layers | sum, then an MLP | `graph_cls`, `graph_reg` |
| [`GraphSAGE`](graphsage.md) | 2 layers | mean, self kept separate | all four |
| [`GraphTransformer`](graph-transformer.md) | 2 layers, 4 heads | dot-product attention | all four |

They are deliberately small and deliberately plain. The benchmark's purpose
is to measure a *protocol*, and a set of two-layer reference implementations
at shared hyperparameters is the honest baseline for that — not a set of
individually tuned state-of-the-art recipes, each of which would be carrying
its own paper's worth of tricks into a comparison that claims to hold
everything else fixed. If you need per-model tuning, that is what
[hyperparameter search](../guides/search.md) is for, and the report records
what it selected.

## What separates them

<div class="grid cards" markdown>

-   __Normalisation__

    ---

    [`GCN`](gcn.md) divides by $\sqrt{d_i d_j}$, so high-degree neighbours
    count for less. This is the assumption that makes it strong on citation
    graphs and weak where degree carries signal.

-   __Attention__

    ---

    [`GAT`](gat.md) and [`GraphTransformer`](graph-transformer.md) both learn
    edge weights, but GAT scores a concatenated pair with a single vector
    while the transformer uses scaled dot-product over projected queries and
    keys.

-   __Injectivity__

    ---

    [`GIN`](gin.md) uses sum aggregation and an MLP because mean and max lose
    multiset information. It is the only built-in with a proof attached, and
    the only one that is graph-level by construction.

-   __Self versus neighbourhood__

    ---

    [`GraphSAGE`](graphsage.md) keeps a node's own representation on a
    separate weight matrix from its neighbours' — the difference that lets it
    hold up on heterophilic graphs.

</div>

## Pre-training

[`DGI`](dgi.md) is exposed as a self-supervised pre-training utility, not a
benchmark task. Its loss is its own metric, so it cannot serve as a held-out
evaluation; the runner routes unlabelled graphs through `link_pred` instead.
See [Tasks and metrics](../concepts/tasks.md#why-there-is-no-self-supervised-headline).

## Instantiating one directly

Nothing forces you through `run_benchmark`. The models are ordinary
`torch.nn.Module`s:

```python
from graphnetz import GAT, train_node_classification
from graphnetz.datasets.social import cora

ds = cora("data/cora")
model = GAT(ds.num_features, 64, ds.num_classes)
history = train_node_classification(model, ds[0], epochs=200)
```

The benchmark builds them with `hidden_channels=64` by default, and
`in_channels` / `out_channels` come from the dataset.

## Next

- [Adapters](adapters.md) — how a node-level encoder reaches all four tasks.
- [Custom models](../guides/custom-models.md) — three ways to register your own.
- [`graphnetz.models`](../reference/models.md) — the full API.
