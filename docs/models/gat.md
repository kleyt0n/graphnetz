# GAT

Graph Attention Network — Veličković et al., ICLR 2018
([arXiv:1710.10903](https://arxiv.org/abs/1710.10903)).

Replaces [GCN](gcn.md)'s fixed degree normalisation with attention weights
the model learns per edge.

## What it computes

For each edge, a shared attention vector scores the concatenated pair of
transformed endpoints, and the scores are softmaxed over each node's
neighbourhood:

$$\alpha_{ij} \;=\; \frac{\exp\!\big(\mathrm{LeakyReLU}(\mathbf{a}^\top[\,W h_i \,\|\, W h_j\,])\big)}
{\sum_{k \in \mathcal{N}(i)} \exp\!\big(\mathrm{LeakyReLU}(\mathbf{a}^\top[\,W h_i \,\|\, W h_k\,])\big)}$$

$$h_i' \;=\; \sigma\Big(\sum_{j \in \mathcal{N}(i)} \alpha_{ij}\, W h_j\Big)$$

Several such heads run in parallel and are concatenated, so different heads
can attend to different aspects of a neighbourhood.

## Architecture as shipped

Two `GATConv` layers, eight heads in the first and one in the second, with
ELU and heavy dropout — the configuration from the original paper:

```python
GAT(in_channels, hidden_channels, out_channels, heads=8, dropout=0.6)
```

| | |
| --- | --- |
| layers | 2 |
| heads | 8 in layer 1 (concatenated), 1 in layer 2 (averaged) |
| activation | ELU |
| dropout | 0.6, applied to inputs **and** to the attention coefficients |
| hidden width | `hidden_channels × heads` entering layer 2 |
| supported tasks | all four, via [adapters](adapters.md) |

Dropout at 0.6 is unusually high and is not an accident: attention has far
more capacity to overfit a small citation graph than a fixed averaging
kernel, and dropping attention coefficients is the regulariser the paper
relies on. It also means GAT is the built-in most sensitive to the epoch
budget, which is worth remembering when reading a fixed-epoch sweep.

## Where it does well, and why

Learned edge weights pay off when edges are genuinely unequal — when a node's
neighbourhood contains a few informative links buried in many uninformative
ones. In the [findings](../findings.md) GAT takes Infrastructure (Euroroad),
Finance (board directors) and Security (9/11 terrorists): small, irregular,
heavy-tailed graphs where "average your neighbours" is a poor prior.

It loses badly on Knowledge (FB15k-237, 0.523 against GCN's 0.675) and
Computing (Internet AS, 0.655 against 0.967). Both are large and
high-degree, where softmax attention over hundreds of neighbours becomes
close to uniform anyway while costing the parameters and the dropout noise
that come with it.

!!! tip "Attention is not free capacity"
    GAT has the widest spread across categories of any built-in: best on
    three, and last or near-last on three others. Reporting it on one
    dataset would support either story, which is the argument for the
    breadth view in [Comparison](../concepts/comparison.md#across-tasks-friedman-and-nemenyi).

## Usage

```python
from graphnetz import GAT, run_benchmark

# Defaults follow the paper; both keywords are exposed.
model = GAT(in_channels=1433, hidden_channels=8, out_channels=7, heads=8, dropout=0.6)

report = run_benchmark("infrastructure", {"GAT": GAT}, seeds=range(10), task_type="link_pred")
```

To sweep `heads` or `dropout` inside the benchmark, register each variant as
its own entry — see
[Custom models](../guides/custom-models.md#inline-tuple-one-shot-variants).

## API

Full signature, parameters and source:
[`graphnetz.models.GAT`](../reference/models.md#graphnetz.models.GAT).
