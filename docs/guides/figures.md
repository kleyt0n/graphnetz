# Figures and tables

Every figure and table in a GraphNetz paper is one method call on a
[`BenchmarkReport`](report.md), drawn on a palette the library owns. This
page covers the output side: which artefact to emit, how the palette is
constructed, and how to keep a figure and its table from disagreeing.

## One call per artefact

```python
# Mean ± t-CI table; row-best in bold with a shaded cell, ties shaded lighter
report.to_latex("results.tex", ci=0.95, bold_best=True)

# Holm-corrected pairwise test table
report.pairwise_to_latex("pairwise.tex")
report.pairwise_to_latex("pairwise_wilcoxon.tex", method="wilcoxon")

# Per-task forest plot, models jittered within rows
fig, _ = report.plot_forest(ci=0.95)

# Pairwise significance heatmap, one panel per task
fig, _ = report.plot_pairwise(layout="matrix")

# Demšar critical-difference diagram across tasks
fig, _ = report.plot_critical_difference(alpha=0.05)

# Mean ± t-CI bands over epochs, one panel per task
fig, _ = report.plot_learning_curves(ci=0.95, ylabel="Test accuracy")
```

Each returns a Matplotlib `(fig, ax)` pair, so anything you would normally do
to a figure still works.

## Keeping the figure and the table consistent

Set the method once on the report rather than per call. Otherwise it is
entirely possible to ship a table of bootstrap intervals beside a forest plot
of *t* intervals, and nothing will warn you:

```python
report.ci_method = "bootstrap"        # every interval, table and plot
report.pairwise_method = "wilcoxon"   # every test, table and heatmap
report.epoch_selection = "best_val"   # every statistic, re-derived
```

These are the same knobs described in [Uncertainty](../concepts/uncertainty.md)
and [Comparison](../concepts/comparison.md); setting them on the report is
what makes the choice apply uniformly.

## Sizing for a paper

```python
from graphnetz import figure, save_figure, set_plot_style

set_plot_style()
fig, ax = figure(width=COLUMN_INCHES)     # single-column by default
save_figure(fig, "figure_3", formats=("pdf", "png"))
```

`COLUMN_INCHES` is the single-column width most venues use.
[`figure`][graphnetz.plotting.figure] applies `NATURE_RC` so a figure's fonts
match the body text at final size rather than after a scaling step, and
[`save_figure`][graphnetz.plotting.save_figure] writes every requested format
from the same draw. [`panel_label`][graphnetz.plotting.panel_label] adds the
**(a)**, **(b)** markers for a multi-panel arrangement.

Figures scale with the number of **tasks**, not models:
`plot_forest` widens the within-row jitter rather than adding rows, so a
dozen models on a dozen tasks still fits one column.

## The two palettes

[`BRAND_COLORS`][graphnetz.plotting.BRAND_COLORS] is the identity palette — a
nine-step neutral ramp, light to dark — shared by this site, the logo and the
figures. [`NATURE_COLORS`][graphnetz.plotting.NATURE_COLORS] selects and
**orders** seven of those steps for plot series.

The two used to be independent, so that restyling the site could not change a
published figure. They now share the ramp, which means the guarantee has to
come from somewhere else: `NATURE_COLORS` names its steps explicitly rather
than slicing `BRAND_COLORS` programmatically, so adding or reordering an
identity colour cannot silently repaint a figure. A figure's colours change
only when that tuple is edited.

Ordering carries the weight that hue used to. The first four steps all clear
4.5:1 against the page, because a series colour also tints markers and, in a
critical-difference diagram, the model labels themselves. The three lightest
steps come last and are reached only by a plot with five or more series,
where they appear as bar fills stroked in ink.

!!! tip "Luminance is a weaker cue than hue"
    Prefer a marker shape or a direct label over colour alone when a series
    has to be identified at a glance — especially in print, and especially
    past four series.

## Which plot answers which question

| question | plot |
| --- | --- |
| How do all tasks compare side by side? | `plot_forest()` |
| Is this win driven by one outlier seed? | `plot_pairwise(layout="list")` |
| Which comparisons survived correction? | `plot_pairwise(layout="matrix")` |
| Which model wins overall? | `plot_critical_difference()` |
| Did anything converge? | `plot_learning_curves()` |

The longer version, including what each one hides, is in
[Reading the report](report.md#choosing-the-right-view).

## Next

- [Reading the report](report.md) — every view, and when it misleads.
- [`graphnetz.plotting`](../reference/plotting.md) — the full figure API.
- [Findings](../findings.md) — these calls, on real numbers.
