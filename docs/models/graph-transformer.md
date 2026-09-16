# GraphTransformer

Unified Message Passing (UniMP) — Shi et al., IJCAI 2021
([arXiv:2009.03509](https://arxiv.org/abs/2009.03509)), via PyG's
`TransformerConv`.

Attention again, but computed the way a transformer computes it rather than
the way [GAT](gat.md) does.

## What it computes

Queries, keys and values are separate projections, and the attention weight
is a scaled dot product:

$$\alpha_{ij} \;=\; \mathrm{softmax}_j\!\left(\frac{(W_q h_i)^\top (W_k h_j)}{\sqrt{d}}\right)$$

$$h_i' \;=\; W_o\,h_i \;+\; \sum_{j \in \mathcal{N}(i)} \alpha_{ij}\,W_v h_j$$

Two differences from GAT are worth naming. GAT scores a *concatenated* pair
against one learned vector, which is a weaker function of the pair than a
bilinear query-key product. And `TransformerConv` keeps a separate
transformation of the node's own features — the same self/neighbour split
that [GraphSAGE](graphsage.md) relies on.

Attention is still restricted to the graph's edges. This is a *sparse*
transformer: there is no all-pairs attention and no positional encoding, so
it does not escape the 1-WL expressiveness ceiling that binds every
message-passing model here.

## Architecture as shipped

Two `TransformerConv` layers, four heads in the first and one in the second:

```python
GraphTransformer(in_channels, hidden_channels, out_channels, heads=4, dropout=0.1)
```

| | |
| --- | --- |
| layers | 2 |
| heads | 4 in layer 1 (concatenated), 1 in layer 2 (averaged) |
| activation | ReLU |
| dropout | 0.1, on the attention coefficients and between layers |
| hidden width | `hidden_channels × heads` entering layer 2 |
| supported tasks | all four, via [adapters](adapters.md) |

Dropout at 0.1 rather than GAT's 0.6 is the visible consequence of the
separate value projection: the model does not lean on coefficient dropout as
its main regulariser.

## Where it does well, and why

In the [findings](../findings.md) GraphTransformer takes Biology (MUTAG,
0.721) and Vision (MNIST-superpixels, 0.142) — both graph classification, and
both cases where a richer per-edge weighting helps a pooled readout. It also
takes CiteSeer in the Planetoid sweep, where GCN wins the other two.

It ranks **last of the four** by mean rank (2.90), and its worst rows are the
large high-degree graphs: Computing (Internet AS, 0.598 against GCN's 0.967)
and Knowledge (FB15k-237, 0.537 against 0.675). The pattern it shares with
GAT is instructive — the two attention models fail on the same graphs, which
suggests the problem is attention over large neighbourhoods rather than
either implementation.

!!! note "Last place here is not a verdict"
    The gap between GraphTransformer's 2.90 and GraphSAGE's 2.10 is 0.80,
    inside the Nemenyi critical difference of 1.483, and the Friedman omnibus
    does not reject. Ten categories cannot order these four. See
    [Adequacy](../concepts/adequacy.md).

## Usage

```python
from graphnetz import GraphTransformer, run_benchmark

model = GraphTransformer(in_channels=7, hidden_channels=64, out_channels=2, heads=4)

report = run_benchmark(
    "biology",
    {"GraphTransformer": GraphTransformer},
    seeds=range(10),
    task_type="graph_cls",
)
```

## API

Full signature, parameters and source:
[`graphnetz.models.GraphTransformer`](../reference/models.md#graphnetz.models.GraphTransformer).
