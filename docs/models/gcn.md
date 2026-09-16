# GCN

Graph Convolutional Network — Kipf & Welling, ICLR 2017
([arXiv:1609.02907](https://arxiv.org/abs/1609.02907)).

The reference baseline of the field, and in GraphNetz's own
[findings](../findings.md) still the model that takes the most categories.

## What it computes

Each layer averages a node's neighbourhood under a symmetric degree
normalisation, then applies a linear map:

$$H^{(l+1)} \;=\; \sigma\!\left(\tilde{D}^{-1/2}\,\tilde{A}\,\tilde{D}^{-1/2}\,H^{(l)}\,W^{(l)}\right)$$

with $\tilde{A} = A + I$ the adjacency plus self-loops and $\tilde{D}$ its
degree matrix. The normalisation is the substantive choice: a neighbour of
degree $d_j$ contributes with weight $1/\sqrt{d_i d_j}$, so a hub influences
each of its many neighbours only weakly.

## Architecture as shipped

Two `GCNConv` layers with a ReLU between them, no dropout:

```python
GCN(in_channels, hidden_channels, out_channels)
```

| | |
| --- | --- |
| layers | 2 |
| activation | ReLU |
| dropout | none |
| parameters | `in × hidden + hidden × out`, plus biases |
| supported tasks | all four, via [adapters](adapters.md) |

Two layers means a two-hop receptive field. That is not a placeholder: GCN
degrades with depth rather than improving, because repeated normalised
averaging drives node representations toward a degree-dependent fixed point
— oversmoothing. The original paper found two layers best on citation graphs
and so does everyone since.

## Where it does well, and why

GCN's normalisation encodes an assumption: **a node resembles its
neighbours, and a high-degree neighbour is less informative per edge**. That
holds on citation networks and on many infrastructure and routing graphs,
and the findings bear it out — GCN wins Social (Cora), Knowledge (FB15k-237),
Computing (Internet AS) and Physics (Ising lattice).

The same assumption is what hurts it on heterophilic graphs, where connected
nodes tend to differ, and where the averaging washes out the contrast the
label depends on. [`GraphSAGE`](graphsage.md), which keeps the self
representation on its own weights, is the natural comparison there.

!!! note "The strongest baseline is a real result"
    In the ten-category sweep GCN takes the most rows of any model while
    ranking second by mean rank, and the Friedman omnibus does not reject.
    A two-layer 2017 baseline being indistinguishable from three later
    architectures is the finding, not a failure to tune it. See
    [Findings](../findings.md).

## Usage

```python
from graphnetz import GCN, run_benchmark, train_node_classification
from graphnetz.datasets.social import cora

# Directly
ds = cora("data/cora")
model = GCN(ds.num_features, 64, ds.num_classes)
history = train_node_classification(model, ds[0], epochs=200)

# In a benchmark
report = run_benchmark("social", {"GCN": GCN}, seeds=range(10), task_type="node_cls")
```

## API

Full signature, parameters and source:
[`graphnetz.models.GCN`](../reference/models.md#graphnetz.models.GCN).
