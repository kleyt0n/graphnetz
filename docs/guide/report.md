# Reading the report

Every `run_benchmark` call returns a
`BenchmarkReport`. It carries the full
per-seed history and exposes a small, opinionated set of views — each
designed to answer a different question.

## Choosing the right view

| Question | Method | Returns |
|---|---|---|
| What's the headline number per *(task, model)*? | `summary()` | DataFrame: `n_seeds, mean, std, sem, ci_low, ci_high` |
| Are these two models significantly different on this task? | `pairwise()` | DataFrame: per-pair `p_raw, p_holm, significant` (parametric or Wilcoxon) |
| Which model wins overall across many tasks? | `plot_critical_difference()` | Demšar / Nemenyi CD diagram |
| How do all tasks compare side by side? | `plot_forest()` | one row per task, models jittered within row |
| Is the win on task X driven by an outlier seed? | `plot_pairwise(layout="list")` | per-comparison CI whiskers |
| How fast did each model converge? | `plot_learning_curves()` | mean ± *t*-CI band, one panel per task |
| I want raw seed values for my own analysis. | `final_metrics()` | nested dict `task → model → [seed_values]` |
| Could this design have detected a difference at all? | `power()` | per-pair MDE, observed power, seeds needed |
| Can I claim two models are *equivalent*? | `equivalence(margin=…)` | per-pair TOST verdict |
| Does the ranking depend on *which* tasks I ran? | `rank_stability()` | bootstrap-over-tasks + jackknife |
| What if I score at the best epoch instead of the last? | `epoch_selection` | re-derives every statistic |
| How would a different task weighting rank them? | `mean_ranks(aggregation=…)` | weighted mean ranks |

The last four are covered in [Is the evidence sufficient?](adequacy.md) and
below.

## Summary table

```python
print(report.summary())                     # default: Student's t CI
print(report.summary(method="bootstrap"))   # percentile-bootstrap CI
```

Per-(task, model) DataFrame with `n_seeds, mean, std, sem, ci_low, ci_high`.
The metric is auto-detected from the per-epoch history; call
`report.metric_name()` to confirm. Set `report.ci_method = "bootstrap"`
once if you want every downstream plot/table to follow.

## Pairwise tests

```python
print(report.pairwise(alpha=0.05))              # default: paired t-test
print(report.pairwise(alpha=0.05, method="wilcoxon"))  # non-parametric
```

Paired *t*-test (default) or Wilcoxon signed-rank test across seed-aligned
final metrics per task with Holm step-down adjustment. Columns: `task,
model_a, model_b, mean_diff, p_raw, p_holm, significant`.

The `method` argument accepts `"t"` (parametric, default) or
`"wilcoxon"` (non-parametric).  You can also set
`report.pairwise_method = "wilcoxon"` once to change the default for all
downstream calls.  Wilcoxon is recommended at small seed counts where the
paired *t*-test's normality assumption is most fragile (Benavoli et al.,
*JMLR* 17(5):1-36, 2016).

If every paired difference is exactly zero the signed-rank statistic is
undefined; the row reports `NaN` rather than an artificial `p = 1.0`.

!!! note
    `mean_diff = mean(model_a) − mean(model_b)`. For lower-is-better metrics
    (MAE), a *negative* `mean_diff` means `model_a` is the better model.

## Forest plot

```python
fig, _ = report.plot_forest(ci=0.95, sort_within=True, band=True)
```

One row per task; models are jittered within the row. The figure scales
with the number of *tasks* — adding more models widens the within-row
jitter rather than adding new rows — so you can compare a dozen models on a
dozen tasks in a single column-width figure.

## Pairwise heatmap

```python
fig, _ = report.plot_pairwise(layout="matrix", alpha=0.05)
```

One heatmap panel per task. The lower triangle shows
$-\log_{10}(p_{\text{Holm}})$ (darker = more significant); the upper triangle
shows the signed mean difference. Switch to `layout="list"` for one row per
comparison with CI whiskers — better when you have only a few pairs.

Pass `method="wilcoxon"` to use the non-parametric p-values in the plot
(and optionally set `report.pairwise_method = "wilcoxon"` as a default).

## Critical-difference diagram

```python
fig, _ = report.plot_critical_difference(alpha=0.05)
```

A Friedman–Nemenyi diagram with average ranks across tasks and horizontal
clique bars joining models within the Nemenyi critical difference. This is
the canonical scalable view for multi-method × multi-dataset comparisons.

!!! tip
    Needs ≥ 2 tasks and ≥ 2 models common to *every* task. With $N \le 4$
    tasks the diagram is informative for rank ordering but rarely strong enough
    to certify pairwise differences — fall back to the pairwise heatmap at that
    scale.

## Epoch selection

By default every statistic is computed at the **final** epoch. That is a
protocol choice, not a neutral one: scoring models at an arbitrary point of
their trajectories rather than at their best can manufacture significance. The
report keeps the full per-epoch history, so the alternative costs nothing:

```python
report.epoch_selection = "best_val"   # or pass epoch_selection= per call
report.pairwise()                     # every statistic re-derived, no retraining
```

`"best_val"` takes each run's metric at the epoch that maximised the *paired
validation* series. The distinction that makes it honest is that the value
reported comes from a split that was not used to choose it.

```python
report.epoch_selection_support()   # which tasks can be selected, and why not
report.selected_epochs()           # the epoch chosen per (task, model, seed)
```

!!! warning "It is not always available, and the framework says so"

    Checkpoint selection needs a held-out series *distinct* from the one being
    selected on. Node classification and both link-prediction routines record
    validation **and** test metrics per epoch; the graph-level routines record
    only a validation metric. Asking for `"best_val"` there raises rather than
    silently selecting and reporting on the same split. Pass `strict=False` to
    let such tasks fall back to the final epoch — and check
    `epoch_selection_support()` so you know which ones did.

`selected_epochs()` is the audit surface: an epoch pinned at the last one for
every seed means selection did nothing, and a wildly varying one means the run
had not converged.

## Task weighting

The Demšar procedure weights every task equally, so a 200-node synthetic
lattice counts as much as a large public benchmark. That is a modelling
choice, and it is exposed:

```python
report.mean_ranks()                              # uniform (the default)
report.mean_ranks(aggregation="reliability")     # inverse CI width
report.mean_ranks(aggregation="hierarchical", groups=task_to_category)
report.mean_ranks(aggregation="custom", weights={"cora": 2.0, "citeseer": 1.0})
report.rank_table()                              # the per-task ranks as data
```

!!! danger "Only the uniform weighting has a Friedman/Nemenyi null"

    Weighting breaks the exchangeability those procedures assume, so a weighted
    mean rank **must not** be compared against $CD_\alpha$.
    `plot_critical_difference(aggregation=…)` enforces this: on a weighted
    diagram it refuses to draw the CD reference and the clique bars, and labels
    the figure a diagnostic. Use `rank_stability()` for uncertainty on the
    weighted variants instead.

## Learning curves

```python
fig, _ = report.plot_learning_curves(ci=0.95, ylabel="Test accuracy")
```

One panel per task, sharing the y-axis, with the mean ± *t*-CI band across
seeds. Useful for diagnosing under- vs over-training before you commit to a
final epoch budget.

## LaTeX exports

```python
report.to_latex("results.tex", ci=0.95, bold_best=True)
report.pairwise_to_latex("pairwise.tex")
report.pairwise_to_latex("pairwise_wilcoxon.tex", method="wilcoxon")
```

`to_latex` writes a booktabs table with the row-best in bold green; ties
get a lighter shade. `pairwise_to_latex` writes the Holm-adjusted
pairwise comparison table with significance markers.  Pass
`method="wilcoxon"` to emit the non-parametric variant.

## Reading a single cell

```python
finals = report.final_metrics()                # dict[task][model] -> [v_seed_0, ...]
arr = finals["cora"]["GCN"]                    # ten seeds for GCN on Cora
mean = sum(arr) / len(arr)
```

`final_metrics()` is the canonical access path for downstream analyses
(custom plots, ad-hoc statistics, exporting to W&B / MLflow).

## Persistence

Use the JSON round-trip, which stores the metric tensor
`X[task, model, seed]` *before* any statistic is applied:

```python
report.to_json("report_social.json")
report = BenchmarkReport.from_json("report_social.json")
```

JSON rather than pickle so the artefact stays readable, diffable, and
independent of the library version that produced it. Because it holds raw
per-epoch histories rather than reduced numbers, a published bundle can be
re-analysed later under a different CI method, pairwise test, epoch-selection
rule or aggregation — without anyone re-running the training.

`BenchmarkReport` is also a plain dataclass, so `pickle` still works for a
short-lived cache within one environment; prefer JSON for anything you ship.

See `examples/experiment.py` for a small driver that caches per-category
reports and emits figures + LaTeX tables on top of them.

## Legacy mapping access

`BenchmarkReport` implements the read-only mapping protocol so older code
that expected a plain `dict[task][model][history]` still works — at the cost
of seeing only seed 0:

```python
for task, per_task in report.items():
    for model, history in per_task.items():
        # history is the seed-0 per-epoch dict (legacy single-seed view)
        ...
```

This view is intentionally lossy. Use `report.histories` for the full
nested dict and `report.final_metrics()` for analysis-ready arrays.
