---
title: GraphNetz
description: A GNN benchmark whose default output is a statistical report, not a leaderboard.
hide:
  - navigation
---

<div class="gn-hero" markdown="0">
  <div class="gn-hero__lockup">
    <!-- Material hides whichever does not match the active scheme, via the
         #only-light / #only-dark src suffixes. The hero ground follows the
         scheme, so a single-ink mark would disappear in one of them. -->
    <img class="gn-hero__logo" src="logo-ink.png#only-light" alt="">
    <img class="gn-hero__logo" src="logo.png#only-dark" alt="">
    <div class="gn-hero__word">GraphNetz</div>
  </div>
  <div class="gn-hero__kicker">Graph Learning Benchmark</div>
  <p class="gn-hero__tagline">
    A GNN benchmark whose default output is a statistical report, not a
    leaderboard. Sixty-two loaders, four task families, one pipeline, and a
    report that says when its own evidence is thin.
  </p>
</div>

<!-- No markdown="1" on the cells: the spans carry no markdown, and the
     extension would wrap each pair in a <p>, collapsing the flex column. -->
<div class="gn-stats">
  <div class="gn-stat"><span class="gn-stat__n">62</span><span class="gn-stat__l">dataset loaders</span></div>
  <div class="gn-stat"><span class="gn-stat__n">10</span><span class="gn-stat__l">categories</span></div>
  <div class="gn-stat"><span class="gn-stat__n">4</span><span class="gn-stat__l">task types</span></div>
  <div class="gn-stat"><span class="gn-stat__n">6</span><span class="gn-stat__l">architectures</span></div>
</div>

## Overview

`graphnetz` is a Python library for benchmarking graph neural networks. You
give it a category, a set of models and a list of seeds; it trains every
compatible *(task, model, seed)* triple and hands back a **report** rather
than a table of accuracies.

The report is the point. It carries a confidence interval for every cell,
paired model comparisons with multiple-testing correction within each task,
and rank-based aggregation across tasks. It also grades itself: the same
seed-paired data answers *could this design have detected a difference at
all?*, which is the question a non-significant result actually raises.

## Install

```bash
uv add graphnetz                # core
uv add "graphnetz[ogb]"         # core plus the OGB loaders
uv add "graphnetz[chem]"        # core plus RDKit, for the molecular loaders
```

Python 3.10 or newer, with `torch` ≥ 2.6 and `torch-geometric` ≥ 2.6. The
optional extras are listed in
[Installation](getting-started/installation.md).

## Quick start

```python
from graphnetz import GAT, GCN, GraphSAGE, run_benchmark

report = run_benchmark(
    "social",
    {"GCN": GCN, "GAT": GAT, "GraphSAGE": GraphSAGE},
    seeds=range(10),
    task_type="node_cls",
)

print(report.summary())          # per-cell mean ± Student's t CI
print(report.pairwise())         # Holm-corrected paired t-tests
print(report.power())            # what this design could have detected
report.plot_critical_difference(alpha=0.05)
report.to_latex("results.tex")   # publication-ready table
```

The [Quickstart](getting-started/quickstart.md) takes this from a fresh
install to a ten-seed sweep with a LaTeX table and a critical-difference
diagram.

## Models

Six architectures share the same trainers, splits and statistics. What
separates them is how a node embedding is computed, not how it is evaluated.

| model | supported tasks | mechanism | reference |
| --- | --- | --- | --- |
| [`GCN`](models/gcn.md) | all four | symmetric-normalised neighbour averaging | Kipf & Welling, ICLR 2017 |
| [`GAT`](models/gat.md) | all four | learned attention over neighbours | Veličković et al., ICLR 2018 |
| [`GIN`](models/gin.md) | `graph_cls`, `graph_reg` | sum aggregation plus an MLP | Xu et al., ICLR 2019 |
| [`GraphSAGE`](models/graphsage.md) | all four | separate self and neighbour transforms | Hamilton et al., NeurIPS 2017 |
| [`GraphTransformer`](models/graph-transformer.md) | all four | multi-head transformer convolution | Shi et al., IJCAI 2021 |
| [`DGI`](models/dgi.md) | *(pre-training utility)* | mutual-information maximisation | Veličković et al., ICLR 2019 |

Four of the six are node-level encoders that never learned about graph
classification or link prediction. They reach those tasks through
[adapters](models/adapters.md) the runner attaches automatically, which is
what makes "the same pipeline for every cell" more than a slogan.

## Documentation

<div class="grid cards" markdown>

-   :material-rocket-launch-outline:{ .lg .middle } __Getting started__

    ---

    Install the right extras, train one model, then run your first multi-seed
    benchmark.

    [:octicons-arrow-right-24: Installation](getting-started/installation.md) ·
    [Quickstart](getting-started/quickstart.md) ·
    [Conventions](getting-started/conventions.md)

-   :material-lightbulb-outline:{ .lg .middle } __Concepts__

    ---

    What the four task families are, where the intervals come from, which
    correction applies where, and how to tell a tie from an underpowered test.

    [:octicons-arrow-right-24: Concepts](concepts/index.md) ·
    [Uncertainty](concepts/uncertainty.md) ·
    [Comparison](concepts/comparison.md) ·
    [Adequacy](concepts/adequacy.md)

-   :material-cog-outline:{ .lg .middle } __Guides__

    ---

    The five-stage protocol, the dataset catalogue, every view on the report,
    and how to plug in your own model or dataset.

    [:octicons-arrow-right-24: Benchmark protocol](guides/benchmark.md) ·
    [Datasets](guides/datasets.md) ·
    [Reading the report](guides/report.md) ·
    [Custom models](guides/custom-models.md)

-   :material-chart-line-variant:{ .lg .middle } __Findings and reference__

    ---

    Every number this library claims, including the negative ones, and the
    full API organised by module.

    [:octicons-arrow-right-24: Findings](findings.md) ·
    [API reference](reference/index.md)

</div>

## Library layout

| module | contents |
| --- | --- |
| [`graphnetz.benchmark`](reference/benchmark.md) | the runner, the task catalogue, the model registry, `BenchmarkReport` |
| [`graphnetz.datasets`](reference/datasets.md) | 62 loaders across 10 categories, the Netzschleuder client, the catalogue auditor |
| [`graphnetz.models`](reference/models.md) | GCN, GAT, GIN, GraphSAGE, GraphTransformer, DGI, and the task adapters |
| [`graphnetz.training`](reference/training.md) | one trainer per task family, each returning a per-epoch history |
| [`graphnetz.plotting`](reference/plotting.md) | the brand and figure palettes, figure helpers, plot builders |

`BenchmarkReport` is the centre of the library rather than an add-on. The
runner's only job is to fill in a metric tensor `X[task, model, seed]`;
everything a user asks of a benchmark — intervals, corrected tests, ranks,
power, equivalence, LaTeX — is a method on the report reading that same
tensor. Nothing downstream requires retraining.

## Project

- **Source**: [github.com/Kleyt0n/graphnetz](https://github.com/Kleyt0n/graphnetz)
- **Package**: [pypi.org/project/graphnetz](https://pypi.org/project/graphnetz/)
- **License**: MIT
- **Results**: [every number, including the negative ones](findings.md)

!!! question "Already have a report?"
    A benchmark should also say when its own evidence is thin.
    [Adequacy](concepts/adequacy.md) covers the minimum detectable effect,
    equivalence testing, and how much benchmark breadth a rank ordering would
    actually need.
