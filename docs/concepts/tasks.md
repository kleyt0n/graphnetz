# Tasks and metrics

Every cell in the benchmark — $(\text{category}, \text{task type},
\text{dataset}, \text{model}, \text{seed})$ — belongs to one of four task
families. The family decides three things: which trainer runs, which adapter
wraps the encoder, and which number the report headlines.

## The four families

| symbol | kind | default metric | direction | adapter |
| --- | --- | --- | --- | --- |
| `node_cls` | node classification | test accuracy | higher is better | encoder used directly |
| `graph_cls` | graph classification | val accuracy | higher is better | mean-pool + linear head |
| `graph_reg` | graph regression | val MAE | **lower** is better | mean-pool + linear head |
| `link_pred` | link prediction | test ROC-AUC | higher is better | dot-product, or DistMult when relational |

`TASK_TYPES` is the authoritative set, and
[`register_model`][graphnetz.benchmark.register_model] rejects anything
outside it rather than registering a model the dispatcher would silently
never schedule.

!!! note "Mixed directions are handled, not assumed away"
    `graph_reg` is the only lower-is-better family. The rank aggregation in
    [Comparison](comparison.md) normalises direction per task *before*
    averaging ranks, so a regression task can sit in the same
    critical-difference diagram as three classification tasks without
    inverting the result. In a pairwise table the sign is yours to read:
    `mean_diff = mean(a) − mean(b)`, so on MAE a **negative** difference
    means `a` is the better model.

## Why there is no self-supervised headline

Unlabelled graphs — the Netzschleuder networks, the synthetic combinatorial
instances, the Ising lattice — are the majority of the catalogue, and the
obvious way to score them is a pretext loss such as Deep Graph Infomax.
GraphNetz does not do that.

A DGI loss is its own metric: there is no held-out signal, so a "better"
number can always be obtained by a model that reconstructs its own corruption
more aggressively, and nothing in the protocol can tell that apart from
learning something about the graph. Instead the runner routes unlabelled
graphs through `link_pred` on a held-out edge split, so **every cell in the
headline report carries a real held-out metric**.

[`DGI`](../models/dgi.md) remains available as a pre-training utility, and
`train_dgi` will happily optimise it. It is simply not a benchmark task.

## Which models can serve which family

A model declares the families it supports, and the dispatcher drops
incompatible *(model, task)* pairs before training rather than failing
mid-sweep:

```python
from graphnetz import GCN, GIN, run_benchmark

# GIN is graph-level only, so this call schedules GCN on all four tasks
# in the category and GIN only on the graph-level ones.
report = run_benchmark("biology", {"GCN": GCN, "GIN": GIN}, seeds=range(10))
```

| model | `node_cls` | `graph_cls` | `graph_reg` | `link_pred` |
| --- | :-: | :-: | :-: | :-: |
| [`GCN`](../models/gcn.md) | ● | ● | ● | ● |
| [`GAT`](../models/gat.md) | ● | ● | ● | ● |
| [`GIN`](../models/gin.md) | — | ● | ● | — |
| [`GraphSAGE`](../models/graphsage.md) | ● | ● | ● | ● |
| [`GraphTransformer`](../models/graph-transformer.md) | ● | ● | ● | ● |

The four node-level encoders reach the graph-level and edge-level families
through [adapters](../models/adapters.md), not through per-model special
cases. GIN is the exception in the other direction: it pools internally, so
it is native to the graph-level families and absent from the rest.

!!! warning "A missing cell is not a zero"
    An unsupported pair produces **no cell at all**, not a cell scored badly.
    That matters for the cross-task ranking: Friedman needs models common to
    every task, so a diagram over a category containing `link_pred` tasks
    will exclude GIN rather than rank it last. The
    [Findings](../findings.md) table shows the consequence — GIN appears on
    two rows and is omitted from the other eight.

## Choosing a task type

```python
run_benchmark("social", MODELS, task_type="node_cls")   # one family
run_benchmark("social", MODELS)                          # every family in the category
```

Restricting to one family is usually what you want when comparing
architectures, because it keeps the metric comparable down a column. Leaving
`task_type` unset is the broader sweep, and the report handles the mixed
metrics — but read [Adequacy](adequacy.md) before reading an ordering off
it.

## Next

- [Uncertainty](uncertainty.md) — the interval attached to each of these metrics.
- [Dataset taxonomy](../guides/datasets.md) — which loaders serve which family.
- [`graphnetz.training`](../reference/training.md) — the trainer behind each family.
