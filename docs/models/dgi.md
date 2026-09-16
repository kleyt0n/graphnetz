# DGI

Deep Graph Infomax — Veličković et al., ICLR 2019
([arXiv:1809.10341](https://arxiv.org/abs/1809.10341)).

A self-supervised **pre-training utility**, not a benchmark task. It is the
one built-in that never appears as a row in a report, and the reason is
worth stating precisely.

## What it computes

DGI learns node embeddings with no labels by maximising the mutual
information between a node's representation and a summary of the whole
graph. Concretely:

1. encode the real graph to per-node embeddings $h_i$,
2. build a **corrupted** graph by row-permuting the node features, and encode
   that to $\tilde{h}_i$,
3. summarise the real graph as $s = \sigma\!\left(\frac{1}{N}\sum_i h_i\right)$,
4. train a bilinear discriminator to tell $(h_i, s)$ from $(\tilde{h}_i, s)$:

$$\mathcal{L} \;=\; -\frac{1}{2N}\sum_{i}\Big[\log \mathcal{D}(h_i, s) \;+\; \log\big(1 - \mathcal{D}(\tilde{h}_i, s)\big)\Big]$$

The corruption is the whole trick. Permuting features keeps the degree
sequence and the topology identical, so the only way to win the
discrimination is to encode how a node's features relate to its structural
position.

## Architecture as shipped

A single `GCNConv` with a PReLU activation, wrapped in PyG's
`DeepGraphInfomax`:

```python
DGI(in_channels, hidden_channels=512)
```

| | |
| --- | --- |
| encoder | one `GCNConv` + `PReLU` |
| default width | 512 |
| summary | sigmoid of the mean node embedding |
| corruption | row-permutation of the node feature matrix |
| `out_channels` | none — there is no head |

The signature is the giveaway: DGI takes no `out_channels`, because it
produces representations rather than predictions.

## Why it is not a benchmark task

!!! warning "A pretext loss is its own metric"
    DGI's loss is computed on the same graph it trained on, against a
    corruption it generated itself. There is no held-out signal, so a lower
    number cannot be distinguished from a model that has simply got better at
    recognising its own permutation. Reporting it beside a held-out accuracy
    would put two incomparable quantities in one column.

The catalogue is mostly unlabelled — Netzschleuder networks, synthetic
combinatorial instances, the Ising lattice — so the temptation to score them
with a pretext loss is real. GraphNetz routes them through `link_pred` on a
held-out edge split instead, which gives a genuine AUC. See
[Tasks and metrics](../concepts/tasks.md#why-there-is-no-self-supervised-headline).

Accordingly `TASK_TYPES` does not contain `"dgi"`, and
[`register_model`][graphnetz.benchmark.register_model] rejects it.

## Using it anyway

Pre-training an encoder unsupervised and then fine-tuning it is a perfectly
good workflow; it just is not a benchmark cell. `train_dgi` runs the
objective, and `DGIWrapper` lets any node-level encoder in the library stand
in for the default GCN one:

```python
from graphnetz import DGI, train_dgi
from graphnetz.datasets.social import cora

ds = cora("data/cora")
model = DGI(ds.num_features, hidden_channels=512)
history = train_dgi(model, ds[0], epochs=300)

# Embeddings for a downstream task
pos_z, _, _ = model(ds[0])
```

To pre-train an arbitrary encoder instead, see
[`DGIWrapper`](adapters.md#dgiwrapper), which adapts a `forward(data)`
encoder to the interface PyG's `DeepGraphInfomax` expects.

## API

Full signature, parameters and source:
[`graphnetz.models.DGI`](../reference/models.md#graphnetz.models.DGI).
