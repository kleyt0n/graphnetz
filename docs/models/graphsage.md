# GraphSAGE

SAmple and aggreGatE — Hamilton, Ying & Leskovec, NeurIPS 2017
([arXiv:1706.02216](https://arxiv.org/abs/1706.02216)).

The best mean rank in GraphNetz's own [findings](../findings.md), on a
design whose distinguishing feature is that it keeps a node separate from its
neighbourhood.

## What it computes

$$h_i' \;=\; W_{\text{self}}\,h_i \;+\; W_{\text{neigh}} \cdot \mathrm{AGG}\big(\{h_j : j \in \mathcal{N}(i)\}\big)$$

The two weight matrices are the whole idea. [GCN](gcn.md) folds a node into
its own neighbourhood average via the added self-loop, so a node's
representation and its neighbours' are transformed identically. GraphSAGE
gives the node its own parameters, which lets the model learn *contrast*
between a node and its surroundings rather than only agreement.

`AGG` is configurable — `mean` by default, with `max`, `lstm` and `sum`
available through PyG.

## Architecture as shipped

Two `SAGEConv` layers with ReLU and dropout between them:

```python
GraphSAGE(in_channels, hidden_channels, out_channels, aggr="mean", dropout=0.5)
```

| | |
| --- | --- |
| layers | 2 |
| aggregation | `"mean"` (`aggr=`) |
| activation | ReLU |
| dropout | 0.5, between the layers |
| supported tasks | all four, via [adapters](adapters.md) |

## Where it does well, and why

The separate self-transform is what makes GraphSAGE robust on **heterophilic**
graphs — those where connected nodes tend to have different labels. Averaging
a node into its neighbourhood, as GCN does, destroys exactly the signal such
a graph carries. Keeping the two apart preserves it.

In the ten-category sweep GraphSAGE wins outright only once (Combinatorial,
TSP-random, 0.890) but is rarely bad, and that consistency gives it the best
mean rank of the four: **2.10**, ahead of GCN's 2.20. It is a useful
illustration of the difference between winning categories and ranking well —
GCN takes four rows to GraphSAGE's one and still ranks second.

!!! note "The best mean rank is still not a win"
    GraphSAGE's 2.10 against GraphTransformer's 2.90 is a gap of 0.80, well
    inside the Nemenyi critical difference of 1.483 at $N = 10$, and the
    Friedman omnibus does not reject ($p = 0.392$). The ordering is not
    evidence. See [Comparison](../concepts/comparison.md#across-tasks-friedman-and-nemenyi).

The original paper's other contribution — neighbourhood *sampling* for
inductive learning on graphs too large to fit in memory — is not exercised by
the built-in, which does full-neighbourhood aggregation. The catalogue's
largest graphs arrive through the OGB loaders, where sampling would be the
natural next step.

## Usage

```python
from graphnetz import GraphSAGE, run_benchmark
from graphnetz.datasets.social import roman_empire

# Heterophilic node classification, where the separate self-weights matter.
ds = roman_empire("data/roman_empire")
model = GraphSAGE(ds.num_features, 64, ds.num_classes, aggr="max")

report = run_benchmark(
    "social",
    {"GraphSAGE": GraphSAGE},
    seeds=range(10),
    task_type="node_cls",
)
```

## API

Full signature, parameters and source:
[`graphnetz.models.GraphSAGE`](../reference/models.md#graphnetz.models.GraphSAGE).
