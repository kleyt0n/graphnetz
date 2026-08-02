<div class="gn-hero" markdown>

<div class="gn-hero__lockup">
  <!-- Material hides whichever does not match the active scheme, via the
       #only-light / #only-dark src suffixes. The hero ground follows the
       scheme, so a single-ink mark would disappear in one of them. -->
  <img class="gn-hero__logo" src="logo-ink.png#only-light" alt="">
  <img class="gn-hero__logo" src="logo.png#only-dark" alt="">
  <h1 class="gn-hero__title">GraphNetz</h1>
</div>

<p class="gn-hero__tagline">
A GNN benchmark whose default output is a statistical report, not a leaderboard.
</p>

[Get started](getting-started/installation.md){ .md-button .md-button--primary }
[View on GitHub](https://github.com/Kleyt0n/graphnetz){ .md-button }

</div>

<!-- No markdown="1" on the cells: the spans carry no markdown, and the
     extension would wrap each pair in a <p>, collapsing the flex column. -->
<div class="gn-stats">
  <div class="gn-stat"><span class="gn-stat__n">62</span><span class="gn-stat__l">dataset loaders</span></div>
  <div class="gn-stat"><span class="gn-stat__n">10</span><span class="gn-stat__l">categories</span></div>
  <div class="gn-stat"><span class="gn-stat__n">4</span><span class="gn-stat__l">task types</span></div>
  <div class="gn-stat"><span class="gn-stat__n">5</span><span class="gn-stat__l">architectures</span></div>
</div>

## Why GraphNetz

Whether you are proposing a new GNN architecture, testing a model on a new graph domain, or comparing existing methods across graph types, GraphNetz turns the usual "train, evaluate, table of accuracies" workflow into a proper statistical report. It gives you confidence intervals for each result, paired model comparisons with multiple-testing correction, and rank-based summaries across datasets via critical-difference diagrams. The point isn't to crown a leaderboard winner. It's to give researchers a way to quantify uncertainty, compare methods fairly, and produce the evidence reviewers actually ask for in graph-learning papers.

<div class="gn-grid" markdown>

<div class="gn-card" markdown>
### Honest comparisons by default
Per-cell Student's-*t* (or percentile-bootstrap) CIs, Holm-adjusted paired
*t*-tests within each task, Friedman ranks plus Nemenyi CD across tasks —
no extra bookkeeping.
</div>

<div class="gn-card" markdown>
### One call, every metric
`run_benchmark(category, models, seeds=...)` trains every compatible
*(task, model, seed)* triple and returns a `BenchmarkReport`.
</div>

<div class="gn-card" markdown>
### Publication-ready artefacts
`report.to_latex(...)`, `plot_forest()`, `plot_pairwise()`,
`plot_critical_difference()`.
</div>

<div class="gn-card" markdown>
### Pluggable models
Decorator, class attribute, or inline tuple — your encoder runs through the
same statistical pipeline as the built-ins.
</div>

</div>

## Benchmark results

Every number on this page comes from the repository's own experiment pipeline
(`paper/experiments/`), at **10 seeds per cell** under the fixed-epoch
protocol. Nothing here is illustrative.

### One protocol, ten categories

One representative dataset per category, all four general-purpose encoders.
Values are mean ± Student's-*t* CI half-width, higher is better throughout.
<strong class="gn-win-key">Bold green</strong> is the best cell in the row; *italic* marks a model whose
interval overlaps the best one, so the two are not distinguishable. Task tags: **NC**
node classification and **GC** graph classification, both scored by accuracy;
**LP** link prediction, scored by AUC.

<div class="gn-wide gn-bench" markdown>

| Category | Dataset | GCN | GAT | SAGE | GT |
|---|---|---|---|---|---|
| Combinatorial | TSP-random · LP | *0.864 ± 0.025* | 0.832 ± 0.028 | **0.890 ± 0.023** | *0.872 ± 0.019* |
| Biology | MUTAG · GC&nbsp;† | 0.705 ± 0.060 | 0.637 ± 0.038 | 0.700 ± 0.060 | **0.721 ± 0.050** |
| Social | Cora · NC | **0.811 ± 0.001** | 0.797 ± 0.009 | 0.799 ± 0.004 | 0.792 ± 0.004 |
| Knowledge | FB15k-237 · LP | **0.675 ± 0.008** | 0.523 ± 0.008 | 0.572 ± 0.027 | 0.537 ± 0.008 |
| Infrastructure | Euroroad · LP | 0.524 ± 0.019 | **0.593 ± 0.022** | 0.540 ± 0.028 | 0.519 ± 0.042 |
| Finance | Board-directors · LP | 0.968 ± 0.007 | **0.989 ± 0.002** | 0.901 ± 0.012 | 0.853 ± 0.015 |
| Computing | Internet AS · LP | **0.967 ± 0.001** | 0.655 ± 0.013 | 0.745 ± 0.013 | 0.598 ± 0.068 |
| Vision | MNIST-superpixels · GC&nbsp;† | 0.105 ± 0.011 | 0.106 ± 0.012 | 0.120 ± 0.021 | **0.142 ± 0.027** |
| Physics | Ising-lattice · LP | **0.676 ± 0.030** | *0.616 ± 0.034* | *0.630 ± 0.020* | 0.620 ± 0.018 |
| Security | 9/11 terrorists · LP | *0.685 ± 0.096* | **0.738 ± 0.065** | *0.730 ± 0.093* | *0.704 ± 0.081* |

</div>

† GIN is defined only for the two graph-classification slots, so it has no
value in the other eight rows and is left out of the table. On both rows it
beats every model shown: **0.847 ± 0.044** on MUTAG and **0.273 ± 0.049** on
MNIST-superpixels.

Read down a column and no architecture wins everywhere: GCN takes four
categories, GAT three, GraphTransformer two, GraphSAGE one — and GIN, where
it can run at all, takes both. That is the point, and it is exactly what a
single-dataset benchmark cannot show you.

### What the statistics actually say

<div class="gn-cd" markdown>
![Demšar critical-difference diagram over ten categories. Mean ranks: GraphSAGE 2.10, GCN 2.20, GAT 2.80, GraphTransformer 2.90. All four are joined by one clique bar.](img/critical_difference.png#only-light)
![Demšar critical-difference diagram over ten categories. Mean ranks: GraphSAGE 2.10, GCN 2.20, GAT 2.80, GraphTransformer 2.90. All four are joined by one clique bar.](img/critical_difference_dark.png#only-dark)
</div>

Aggregate the ten per-category rankings and the honest answer is a negative
result:

<div class="gn-result" markdown>

| | |
|---|---|
| Friedman omnibus | $\chi^2_3 = 3.00$, $p = 0.392$ — **do not reject** |
| Nemenyi critical difference | $CD = 1.483$ ($\alpha = 0.05$, $k = 4$, $N = 10$) |
| Observed mean ranks | GraphSAGE 2.10 · GCN 2.20 · GAT 2.80 · GraphTransformer 2.90 |
| Largest rank gap | 0.80 — well inside $CD$ |
| Cliques | **one**: no architecture separates from any other |

</div>

Ten categories are not enough evidence to order these four architectures. A
benchmark that reported only the means would have declared a winner anyway.

### Where differences *do* appear

Zoom in on a single task family and the resolution improves. Node
classification on the three Planetoid citation graphs, 10 seeds:

<div class="gn-bench" markdown>

| Model | Cora | CiteSeer | PubMed |
|---|---|---|---|
| GCN | **0.811 ± 0.001** | 0.683 ± 0.002 | **0.791 ± 0.002** |
| GAT | 0.795 ± 0.010 | 0.655 ± 0.013 | 0.752 ± 0.015 |
| GraphSAGE | 0.799 ± 0.004 | 0.683 ± 0.012 | 0.767 ± 0.002 |
| GraphTransformer | 0.792 ± 0.004 | **0.700 ± 0.006** | 0.759 ± 0.006 |

</div>

Here **11 of 18** pairwise comparisons survive Holm–Bonferroni correction — a
real signal that the cross-category view washes out. Note also that the
per-task winner flips: GCN on Cora and PubMed, GraphTransformer on CiteSeer.

### Is the evidence sufficient?

The report also grades itself. Across the same 18 comparisons:

| | Cora | CiteSeer | PubMed | All |
|---|---:|---:|---:|---:|
| Median minimum detectable effect | 0.0114 | 0.0169 | 0.0150 | **0.0137** |
| Comparisons resolved at 10 seeds | 3/6 | 5/6 | 3/6 | **11/18** |
| Seeds needed for a 0.01 gap | 13 | 25 | 23 | **17** |
| Seeds needed for a 0.005 gap | 44 | 93 | 84 | **62** |

And a jackknife over the ten categories shows how fragile the rank ordering
is: the six pairwise rank gaps separate in only **0.1 %–11.3 %** of task
resamples. No pair is stable.

[Read the adequacy guide](guide/adequacy.md){ .md-button }
[How the report is built](guide/report.md){ .md-button }

## Quickstart

```python
from graphnetz import GAT, GCN, GraphSAGE, run_benchmark

report = run_benchmark(
    "social",
    {"GCN": GCN, "GAT": GAT, "GraphSAGE": GraphSAGE},
    seeds=range(10),
    task_type="node_cls",
)

print(report.summary())          # per-(task, model) mean ± t-CI
print(report.pairwise())         # Holm-corrected paired t-tests
report.plot_critical_difference(alpha=0.05)
report.to_latex("results.tex")   # publication-ready table
```

## Bring your own model

The contract is two methods: `__init__(in_channels, hidden_channels,
out_channels)` and `forward(data)` taking a PyG `Data`. Declare which task
your model supports and it runs through the *same* pipeline as the built-ins
— same seeds, same splits, same Holm correction, same CD diagram.

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
```

1. Or set `task_types = {"node_cls"}` as a class attribute, or pass an inline
   tuple `(cls, task_type, factory)` for one-shot variants — handy for a
   hyperparameter sweep where each candidate is a separate "model".
2. `data` is a PyG `Data` object. Read what you need from it; the runner owns
   the split, the optimiser, and the seeding.
3. Baselines are ordinary entries in the same dict. Nothing about your model
   takes a different code path.

The payoff is the second `print`: a difference in means that does not survive
Holm correction never becomes a claim.

<div class="gn-result" markdown>

| | |
|---|---|
| Different task? | Swap `task_type` for `graph_cls`, `graph_reg` or `link_pred` |
| Custom dataset? | Pass `tasks=[Task(...)]` and skip the built-in catalogue |
| Sweeping hyperparameters? | One dict entry per variant, or hand `run_benchmark` a `SearchSpace` |

</div>

[Custom models & datasets](getting-started/custom.md){ .md-button }
[Models & adapters](guide/models.md){ .md-button }

## At a glance

| | |
|---|---|
| **Tasks** | `node_cls` · `graph_cls` · `graph_reg` · `link_pred` |
| **Architectures** | GCN · GAT · GIN · GraphSAGE · GraphTransformer (DGI as a pre-training utility) |
| **Loaders** | 62 across 10 categories (combinatorial, biology, social, knowledge, infrastructure, finance, computing, vision, physics, security) |
| **Default report** | per-cell mean ± Student's-*t* CI · Holm-adjusted paired *t* / Wilcoxon · Demšar–Nemenyi CD |
| **Adequacy layer** | minimum detectable effect · observed power · TOST equivalence · rank stability |
| **Source** | [github.com/Kleyt0n/graphnetz](https://github.com/Kleyt0n/graphnetz) |

!!! tip "New here?"
    Start with [Installation](getting-started/installation.md), then the
    [Quickstart](getting-started/quickstart.md). To understand *how* the
    pieces fit together — the taxonomy, the adapters, and the five-stage
    pipeline — read the [User guide](guide/datasets.md).

!!! question "Already have a report?"
    A benchmark should also say when its own evidence is thin.
    [Is the evidence sufficient?](guide/adequacy.md) covers the minimum
    detectable effect, equivalence testing, and how much benchmark breadth a
    rank ordering would actually need.
