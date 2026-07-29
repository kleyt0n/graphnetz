---
hide:
  - toc
---

<div class="gn-hero" markdown>

<img class="gn-hero__logo" src="logo.png" alt="GraphNetz logo">

<h1 class="gn-hero__title">GraphNetz</h1>

<p class="gn-hero__tagline">
A GNN benchmark whose default output is a statistical report, not a leaderboard.
</p>

[Get started](getting-started/installation.md){ .md-button .md-button--primary }
[View on GitHub](https://github.com/Kleyt0n/graphnetz){ .md-button }

<figure class="gn-anim__wrap">

  <svg class="gn-anim" viewBox="48 28 424 126" role="img"
       aria-label="Eight seeds per model collapse into a mean with a confidence interval; a clique bar then joins all four models, showing no difference is detected.">
    <line class="gn-anim__axis" x1="60" y1="124" x2="460" y2="124"/>
    <g class="gn-anim__model" style="--m: 0">
      <line class="gn-anim__seed" style="--s: 0" x1="84" y1="40" x2="106" y2="40"/>
      <line class="gn-anim__seed" style="--s: 1" x1="84" y1="48" x2="106" y2="48"/>
      <line class="gn-anim__seed" style="--s: 2" x1="84" y1="55" x2="106" y2="55"/>
      <line class="gn-anim__seed" style="--s: 3" x1="84" y1="60" x2="106" y2="60"/>
      <line class="gn-anim__seed" style="--s: 4" x1="84" y1="64" x2="106" y2="64"/>
      <line class="gn-anim__seed" style="--s: 5" x1="84" y1="70" x2="106" y2="70"/>
      <line class="gn-anim__seed" style="--s: 6" x1="84" y1="78" x2="106" y2="78"/>
      <line class="gn-anim__seed" style="--s: 7" x1="84" y1="85" x2="106" y2="85"/>
      <line class="gn-anim__whisker" x1="95" y1="36" x2="95" y2="88"/>
      <line class="gn-anim__cap" x1="88" y1="36" x2="102" y2="36"/>
      <line class="gn-anim__cap" x1="88" y1="88" x2="102" y2="88"/>
      <circle class="gn-anim__mean" cx="95" cy="62" r="4.5"/>
    </g>
    <g class="gn-anim__model" style="--m: 1">
      <line class="gn-anim__seed" style="--s: 0" x1="194" y1="56" x2="216" y2="56"/>
      <line class="gn-anim__seed" style="--s: 1" x1="194" y1="62" x2="216" y2="62"/>
      <line class="gn-anim__seed" style="--s: 2" x1="194" y1="68" x2="216" y2="68"/>
      <line class="gn-anim__seed" style="--s: 3" x1="194" y1="72" x2="216" y2="72"/>
      <line class="gn-anim__seed" style="--s: 4" x1="194" y1="76" x2="216" y2="76"/>
      <line class="gn-anim__seed" style="--s: 5" x1="194" y1="81" x2="216" y2="81"/>
      <line class="gn-anim__seed" style="--s: 6" x1="194" y1="87" x2="216" y2="87"/>
      <line class="gn-anim__seed" style="--s: 7" x1="194" y1="93" x2="216" y2="93"/>
      <line class="gn-anim__whisker" x1="205" y1="53" x2="205" y2="95"/>
      <line class="gn-anim__cap" x1="198" y1="53" x2="212" y2="53"/>
      <line class="gn-anim__cap" x1="198" y1="95" x2="212" y2="95"/>
      <circle class="gn-anim__mean" cx="205" cy="74" r="4.5"/>
    </g>
    <g class="gn-anim__model" style="--m: 2">
      <line class="gn-anim__seed" style="--s: 0" x1="304" y1="42" x2="326" y2="42"/>
      <line class="gn-anim__seed" style="--s: 1" x1="304" y1="52" x2="326" y2="52"/>
      <line class="gn-anim__seed" style="--s: 2" x1="304" y1="60" x2="326" y2="60"/>
      <line class="gn-anim__seed" style="--s: 3" x1="304" y1="66" x2="326" y2="66"/>
      <line class="gn-anim__seed" style="--s: 4" x1="304" y1="71" x2="326" y2="71"/>
      <line class="gn-anim__seed" style="--s: 5" x1="304" y1="79" x2="326" y2="79"/>
      <line class="gn-anim__seed" style="--s: 6" x1="304" y1="88" x2="326" y2="88"/>
      <line class="gn-anim__seed" style="--s: 7" x1="304" y1="96" x2="326" y2="96"/>
      <line class="gn-anim__whisker" x1="315" y1="38" x2="315" y2="98"/>
      <line class="gn-anim__cap" x1="308" y1="38" x2="322" y2="38"/>
      <line class="gn-anim__cap" x1="308" y1="98" x2="322" y2="98"/>
      <circle class="gn-anim__mean" cx="315" cy="68" r="4.5"/>
    </g>
    <g class="gn-anim__model" style="--m: 3">
      <line class="gn-anim__seed" style="--s: 0" x1="414" y1="62" x2="436" y2="62"/>
      <line class="gn-anim__seed" style="--s: 1" x1="414" y1="69" x2="436" y2="69"/>
      <line class="gn-anim__seed" style="--s: 2" x1="414" y1="75" x2="436" y2="75"/>
      <line class="gn-anim__seed" style="--s: 3" x1="414" y1="79" x2="436" y2="79"/>
      <line class="gn-anim__seed" style="--s: 4" x1="414" y1="84" x2="436" y2="84"/>
      <line class="gn-anim__seed" style="--s: 5" x1="414" y1="90" x2="436" y2="90"/>
      <line class="gn-anim__seed" style="--s: 6" x1="414" y1="97" x2="436" y2="97"/>
      <line class="gn-anim__seed" style="--s: 7" x1="414" y1="103" x2="436" y2="103"/>
      <line class="gn-anim__whisker" x1="425" y1="59" x2="425" y2="105"/>
      <line class="gn-anim__cap" x1="418" y1="59" x2="432" y2="59"/>
      <line class="gn-anim__cap" x1="418" y1="105" x2="432" y2="105"/>
      <circle class="gn-anim__mean" cx="425" cy="82" r="4.5"/>
    </g>
    <line class="gn-anim__clique" x1="80" y1="148" x2="440" y2="148"/>
  </svg>

<figcaption class="gn-anim__caption">
  <span class="gn-anim__step" style="--k: 0">every model, every seed</span>
  <span class="gn-anim__step" style="--k: 1">mean ± confidence interval</span>
  <span class="gn-anim__step" style="--k: 2">no difference detected</span>
</figcaption>
</figure>

</div>

## Why GraphNetz

Whether you are proposing a new GNN architecture, testing a model on a new
graph domain, or comparing existing methods across graph types, GraphNetz
turns the usual “train, evaluate, table of accuracies” workflow into a
reproducible statistical report. Instead of reporting point estimates alone,
it provides confidence intervals for each result, paired model comparisons
with multiple-testing correction, and rank-based summaries across datasets
using critical-difference diagrams. The goal is not just to crown a
leaderboard winner, but to give researchers a principled way to quantify
uncertainty, compare methods fairly, and produce the exact evidence reviewers
often ask for in graph-learning papers.

<div class="gn-cd" markdown>
![Demšar critical-difference diagram comparing four GNN architectures by mean rank.](img/critical_difference.png#only-light)
![Demšar critical-difference diagram comparing four GNN architectures by mean rank.](img/critical_difference_dark.png#only-dark)
</div>

A Demšar critical-difference diagram. Models are ordered by mean Friedman
rank; the horizontal bar connects groups whose ranks are not significantly
different at the chosen $\alpha$ under the Nemenyi post-hoc.

## What is inside

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
