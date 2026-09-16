# Adapters

A node-level encoder maps a graph to one vector per node. Graph
classification needs one vector per *graph*; link prediction needs a score
per *edge*. Three thin wrappers close that gap, and the benchmark dispatcher
attaches the right one automatically based on the requested `task_type`.

This is what makes the claim "every cell goes through the same pipeline"
literal rather than aspirational: [GCN](gcn.md), [GAT](gat.md),
[GraphSAGE](graphsage.md) and [GraphTransformer](graph-transformer.md)
contain no task-specific code at all.

## The three wrappers

| wrapper | turns an encoder into | used for |
| --- | --- | --- |
| [`GraphLevelWrapper`](#graphlevelwrapper) | a graph classifier or regressor | `graph_cls`, `graph_reg` |
| [`LinkPredWrapper`](#linkpredwrapper) | an edge scorer | `link_pred` |
| [`DGIWrapper`](#dgiwrapper) | a Deep Graph Infomax model | pre-training only |

`node_cls` needs no wrapper: the encoder is built with the dataset's class
count as `out_channels` and used directly.

## How the dispatcher chooses

`_multi_task_factory` is the single place that decides, and the rule is short
enough to read in full:

| `task_type` | encoder built as | then |
| --- | --- | --- |
| `node_cls` | `cls(in, hidden, num_classes)` | used directly |
| `graph_cls` | `cls(in, hidden, hidden)` | `GraphLevelWrapper(enc, hidden, num_classes)` |
| `graph_reg` | `cls(in, hidden, hidden)` | `GraphLevelWrapper(enc, hidden, 1)` |
| `link_pred` | `cls(in, hidden, hidden)` | `LinkPredWrapper(enc)` |

Note the pattern: for every task except node classification the encoder's
`out_channels` is set to `hidden_channels`, because its output is an
intermediate representation rather than a prediction. The wrapper owns the
head.

You will not usually instantiate any of this. `run_benchmark` does it:

```python
from graphnetz import GAT, run_benchmark

# One model, four task families, no adapter code written by you.
report = run_benchmark("social", {"GAT": GAT}, seeds=range(10))
```

## GraphLevelWrapper

Adds a `global_mean_pool` over the batch index and a linear head:

```python
z = encoder(data)                      # [N, hidden]
pooled = global_mean_pool(z, data.batch)   # [B, hidden]
out = Linear(hidden, out_channels)(pooled) # [B, out]
```

Mean pooling rather than sum is a deliberate default for the *adapted*
models: it is scale-invariant to graph size, which matters when a
category mixes graphs of very different orders. [`GIN`](gin.md) pools with
`global_add_pool` instead, because sum aggregation is the thing GIN is *for*
— see its expressiveness argument. The difference between the two readouts
is visible in the [findings](../findings.md), where GIN beats every adapted
encoder on both graph-classification rows.

[:octicons-arrow-right-24: `GraphLevelWrapper` API](../reference/models.md#graphnetz.models._adapters.GraphLevelWrapper)

## LinkPredWrapper

Scores an edge from the two endpoint embeddings. Homogeneous graphs use a
dot product; relational graphs (knowledge graphs, with `num_relations`) use
DistMult, so the relation type participates in the score.

Link prediction is the workhorse of this catalogue: it is how every
unlabelled graph enters the benchmark with a real held-out metric instead of
a pretext loss. See
[Tasks and metrics](../concepts/tasks.md#why-there-is-no-self-supervised-headline).

[:octicons-arrow-right-24: `LinkPredWrapper` API](../reference/models.md#graphnetz.models._adapters.LinkPredWrapper)

## DGIWrapper

Adapts any `forward(data)` encoder into PyG's `DeepGraphInfomax`, which calls
its encoder with positional `(x, edge_index)`. A small `Data` shim bridges
the two calling conventions.

This one is **not** reachable from `run_benchmark`: `"dgi"` is not a member
of `TASK_TYPES`, and [`register_model`][graphnetz.benchmark.register_model]
rejects it. It exists so that the unsupervised objective can be run against
an arbitrary encoder as a pre-training step. See [DGI](dgi.md).

[:octicons-arrow-right-24: `DGIWrapper` API](../reference/models.md#graphnetz.models._adapters.DGIWrapper)

## Writing an encoder that adapts cleanly

The contract the wrappers depend on is narrow:

```python
class MyEncoder(torch.nn.Module):
    """Returns per-node embeddings of shape [N, out_channels]."""

    def __init__(self, in_channels, hidden_channels, out_channels): ...
    def forward(self, data): ...   # data is a PyG Data object
```

Two things to get right:

- **Return per-node features, not pooled ones.** A wrapper that receives an
  already-pooled tensor will pool it again, silently, and the shapes may even
  work out.
- **Read `data.batch` only if you need it.** `GraphLevelWrapper` uses it; a
  node-level encoder should not, or it will not run on a single-graph task
  where `batch` is absent.

To get all four task types without writing the glue:

```python
from graphnetz.benchmark import _multi_task_factory, register_model

_ALL = {"node_cls", "graph_cls", "graph_reg", "link_pred"}
register_model(MyEncoder, task_type=_ALL, factory=_multi_task_factory(MyEncoder))
```

## Next

- [Custom models](../guides/custom-models.md) — the three registration paths.
- [Tasks and metrics](../concepts/tasks.md) — what each family expects.
- [`graphnetz.models`](../reference/models.md) — full API.
