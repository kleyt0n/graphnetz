---
title: graphnetz
description: A GNN benchmark whose default output is a statistical report, not a leaderboard.
hide:
  - toc
---

<!-- The hero carries the visible title. A real h1 stays for screen readers and
     search, and stops Material inserting a "Home" heading above the hero. -->
# graphnetz { .gn-sr-only }

<div class="gn-hero" markdown>

<div class="gn-hero__lockup" markdown="0">
  <!-- Material hides whichever does not match the active scheme, via the
       #only-light / #only-dark src suffixes. -->
  <img class="gn-hero__logo" src="logo-ink.svg#only-light" alt="">
  <img class="gn-hero__logo" src="logo-light.svg#only-dark" alt="">
  <div class="gn-hero__word">graphnetz</div>
</div>

<p class="gn-hero__tagline">
  Benchmark graph neural networks and get back a statistical report,
  not a leaderboard.
</p>

<div class="gn-hero__actions" markdown>
[Get started](getting-started/quickstart.md){ .md-button .md-button--primary }
[:fontawesome-brands-github: GitHub](https://github.com/Kleyt0n/graphnetz){ .md-button }
</div>

</div>

## Why graphnetz

Most GNN benchmarks report one accuracy per cell and name a winner. graphnetz
trains every *(task, model, seed)* triple through one pipeline and answers the
questions a reviewer actually asks.

<div class="grid cards gn-features" markdown>

-   :material-chart-bell-curve:{ .lg } __Intervals on every cell__

    Mean and Student's *t* confidence interval over paired seeds, so a
    difference comes with its uncertainty attached.

    [:octicons-arrow-right-24: Uncertainty](concepts/uncertainty.md)

-   :material-scale-balance:{ .lg } __Corrected comparisons__

    Paired *t*-tests or Wilcoxon signed-rank within each task, Holm-corrected
    for the number of pairs.

    [:octicons-arrow-right-24: Comparison](concepts/comparison.md)

-   :material-podium:{ .lg } __Ranks across tasks__

    Friedman omnibus, Nemenyi critical difference, and the Demšar diagram for
    the "does it win in general?" question.

    [:octicons-arrow-right-24: Comparison](concepts/comparison.md#across-tasks-friedman-and-nemenyi)

-   :material-magnify-scan:{ .lg } __A report that grades itself__

    Minimum detectable effect, observed power and equivalence tests say
    whether a non-significant result means "tied" or "underpowered".

    [:octicons-arrow-right-24: Adequacy](concepts/adequacy.md)

-   :material-graph-outline:{ .lg } __One pipeline, four task families__

    Node and graph classification, graph regression and link prediction.
    Adapters let node encoders reach every task without a second code path.

    [:octicons-arrow-right-24: Tasks and metrics](concepts/tasks.md)

-   :material-database-outline:{ .lg } __A catalogue across ten domains__

    PyG built-ins, OGB and the Netzschleuder archive, from citation graphs to
    power grids, molecules and connectomes.

    [:octicons-arrow-right-24: Datasets](guides/datasets.md)

</div>

## From install to report

=== "Install"

    ```bash
    uv add graphnetz                # core
    uv add "graphnetz[ogb]"         # plus the OGB loaders
    uv add "graphnetz[chem]"        # plus RDKit, for the molecular loaders
    ```

    Python 3.10 or newer, with `torch` ≥ 2.6 and `torch-geometric` ≥ 2.6. See
    [Installation](getting-started/installation.md) for the extras.

=== "Run"

    ```python
    from graphnetz import GAT, GCN, GraphSAGE, run_benchmark

    report = run_benchmark(
        "social",
        {"GCN": GCN, "GAT": GAT, "GraphSAGE": GraphSAGE},
        seeds=range(10),
        task_type="node_cls",
    )
    ```

    Every compatible *(task, model, seed)* triple is trained with the same
    splits, the same seeding and the same epoch budget.

=== "Read"

    ```python
    report.summary()       # per-cell mean ± Student's t CI
    report.pairwise()      # Holm-corrected paired tests within each task
    report.friedman()      # omnibus test on ranks across tasks
    report.power()         # what this design could have detected
    ```

    Every view reads the same `X[task, model, seed]` tensor. Nothing
    downstream retrains. See [Reading the report](guides/report.md).

=== "Publish"

    ```python
    report.plot_critical_difference(alpha=0.05)
    report.to_latex("results.tex")       # booktabs table, row-best in bold
    report.to_json("report.json")        # reload later with from_json
    ```

    Figures follow single and double column widths and save as vector output.
    See [Figures and tables](guides/figures.md).

## How a run works

<div class="gn-steps" markdown>

1.  __Catalogue__

    The category maps to its curated tasks.

2.  __Encoders__

    Models that cannot serve a task are dropped.

3.  __Training__

    Each triple is reseeded and trained.

4.  __Statistics__

    Intervals, paired tests, ranks.

5.  __Report__

    One object holds every view.

</div>

The [benchmark protocol](guides/benchmark.md) covers each stage in full.

## What the evidence says

<div class="gn-evidence" markdown>

<div class="gn-evidence__figure" markdown>
![Demšar critical-difference diagram over ten categories. Mean ranks: GraphSAGE 2.10, GCN 2.20, GAT 2.80, GraphTransformer 2.90. All four are joined by one clique bar.](img/critical_difference.png#only-light)
![Demšar critical-difference diagram over ten categories. Mean ranks: GraphSAGE 2.10, GCN 2.20, GAT 2.80, GraphTransformer 2.90. All four are joined by one clique bar.](img/critical_difference_dark.png#only-dark)
</div>

<div class="gn-evidence__text" markdown>
Four general-purpose encoders, one dataset from each of ten categories, ten
seeds per cell. The Friedman test does not reject
($\chi^2_3 = 3.00$, $p = 0.392$) and all four sit in one clique.

Ten categories are not enough evidence to order these architectures. A
benchmark that reported only the means would have declared a winner anyway.

[Read the findings :octicons-arrow-right-24:](findings.md){ .md-button }
</div>

</div>

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

Bring your own with a decorator, a class attribute or an inline tuple. See
[Custom models](guides/custom-models.md).

## Explore the docs

<div class="grid cards" markdown>

-   :material-rocket-launch-outline:{ .lg .middle } __Getting started__

    ---

    Install the right extras, train one model, then run your first multi-seed
    benchmark.

    [:octicons-arrow-right-24: Quickstart](getting-started/quickstart.md)

-   :material-lightbulb-outline:{ .lg .middle } __Concepts__

    ---

    Where the intervals come from, which correction applies where, and how to
    tell a tie from an underpowered test.

    [:octicons-arrow-right-24: Concepts](concepts/index.md)

-   :material-book-open-variant:{ .lg .middle } __Guides__

    ---

    The five-stage protocol, the dataset catalogue, every view on the report,
    and hyperparameter search.

    [:octicons-arrow-right-24: Benchmark protocol](guides/benchmark.md)

-   :material-code-braces:{ .lg .middle } __API reference__

    ---

    Every public symbol, organised by module, with its docstring and source.

    [:octicons-arrow-right-24: API reference](reference/index.md)

</div>

<div class="gn-project" markdown>
[:fontawesome-brands-github: Source](https://github.com/Kleyt0n/graphnetz)
[:fontawesome-brands-python: PyPI](https://pypi.org/project/graphnetz/)
[:material-file-document-outline: Paper](https://arxiv.org/pdf/2605.09099)
[:material-scale-balance: MIT License](https://github.com/Kleyt0n/graphnetz/blob/main/LICENCE.txt)
</div>
