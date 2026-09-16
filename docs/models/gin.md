# GIN

Graph Isomorphism Network — Xu et al., ICLR 2019
([arXiv:1810.00826](https://arxiv.org/abs/1810.00826)).

The only built-in with a theoretical guarantee attached, and the only one
that is graph-level by construction.

## What it computes

$$h_i^{(l+1)} \;=\; \mathrm{MLP}^{(l)}\!\left((1 + \epsilon^{(l)})\,h_i^{(l)} \;+\; \sum_{j \in \mathcal{N}(i)} h_j^{(l)}\right)$$

Two choices distinguish this from [GCN](gcn.md), and both are deliberate:

**Sum, not mean or max.** Mean aggregation cannot tell a neighbourhood of
three red nodes from one of six, and max cannot tell three from one. Sum
preserves the multiset, which is the information a graph-level readout needs.

**An MLP, not a linear map.** Sum aggregation is only injective if what
follows it is, and a single linear layer is not. The MLP is what makes the
composition injective in principle.

Together these make GIN as discriminative as the Weisfeiler–Lehman graph
isomorphism test, which is the provable upper bound for this family of
message-passing networks. It cannot distinguish two graphs that 1-WL cannot,
and neither can any of the other built-ins.

## Architecture as shipped

Three `GINConv` layers with learnable $\epsilon$, each wrapping a two-layer
MLP with batch normalisation, then a global **sum** pool and a linear
classifier:

```python
GIN(in_channels, hidden_channels, out_channels, num_layers=3)
```

| | |
| --- | --- |
| layers | 3 (`num_layers`) |
| per-layer MLP | `Linear → BatchNorm1d → ReLU → Linear → ReLU` |
| `train_eps` | `True` — $\epsilon$ is learned, not fixed at 0 |
| readout | `global_add_pool`, then a linear head |
| supported tasks | `graph_cls`, `graph_reg` only |

## Why it is graph-level only

GIN pools internally. It maps a batch of graphs to one vector per graph, not
one per node, so it has nothing to hand a node classifier or a link-prediction
decoder. The other four built-ins are node-level encoders that acquire
graph-level behaviour through [adapters](adapters.md); GIN is the reverse case
and is simply absent from the node-level and edge-level families.

!!! warning "A missing cell is not a zero"
    In the ten-category [findings](../findings.md) GIN appears on the two
    graph-classification rows and is omitted from the other eight — so it is
    also excluded from the critical-difference diagram, which needs models
    common to every task. It is not ranked last there; it is not ranked at
    all. See [Tasks and metrics](../concepts/tasks.md#which-models-can-serve-which-family).

## Where it does well

On both rows where it can run, it beats every other built-in by a wide
margin: **0.847 ± 0.044** on MUTAG against GraphTransformer's 0.721, and
**0.273 ± 0.049** on MNIST-superpixels against 0.142. That is the
sum-aggregation argument showing up as a number — graph classification is
exactly the task where losing multiset information costs you.

`BatchNorm1d` in the MLP means GIN needs batches of more than one graph, which
graph-level loaders provide but a single-graph node task would not. It is
another reason the family restriction is structural rather than conservative.

## Usage

```python
from graphnetz import GIN, run_benchmark
from graphnetz.datasets.biology import mutag

ds = mutag("data/mutag")
model = GIN(ds.num_features, 64, ds.num_classes, num_layers=3)

report = run_benchmark(
    "biology",
    {"GIN": GIN},
    seeds=range(10),
    task_type="graph_cls",
)
```

## API

Full signature, parameters and source:
[`graphnetz.models.GIN`](../reference/models.md#graphnetz.models.GIN).
