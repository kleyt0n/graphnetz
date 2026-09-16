# Uncertainty

A single accuracy is a sample, not a property of a model. GraphNetz never
reports one on its own: every cell in a summary carries an interval over the
seeds that produced it.

## Where the interval comes from

For a cell with $S$ paired seeds, the default is a Student's-*t* interval on
the mean of the per-seed final metrics:

$$\bar{x} \;\pm\; t_{1-\alpha/2,\,S-1}\,\frac{s}{\sqrt{S}}$$

```python
print(report.summary())            # n_seeds, mean, std, sem, ci_low, ci_high
print(report.metric_name())        # which metric this is, auto-detected
```

The *t* quantile rather than the normal one is not a formality at the seed
counts benchmarks actually use. At $S = 10$, $t_{0.975, 9} = 2.262$ against
$z_{0.975} = 1.96$: a normal approximation would report intervals about 15 %
too narrow, and would do so in the direction that makes results look more
conclusive.

## What the interval is over

It is the uncertainty of **this protocol's mean**, across the randomness the
seed controls. That is a narrower claim than it may look, and the boundary is
worth stating:

| varies across seeds | does not vary |
| --- | --- |
| model initialisation | the architecture and its hyperparameters |
| the train/val/test split, where the trainer draws one | the dataset itself |
| data resampling, for the seed-aware synthetic loaders | the epoch budget |
| non-deterministic kernel scheduling on GPU | the metric definition |

So a narrow interval means *this recipe lands here reliably*. It says nothing
about how the model would do on a different dataset, at a different epoch
budget, or under a hyperparameter search — for which see
[Hyperparameter search](../guides/search.md).

## When Student's *t* is the wrong instrument

The *t* interval assumes the per-seed metrics are roughly normal around their
mean. That is reasonable for accuracy on a balanced split with a decent
number of test nodes. It gets fragile for:

- **bounded metrics near their ceiling** — AUC at 0.99, where the sampling
  distribution is squeezed against 1.0 and visibly skewed,
- **ranking metrics** — Hits@K and MRR, which are averages of a heavy-tailed
  per-query quantity,
- **AUC on badly imbalanced splits**, where a handful of positives dominate.

In those cases take the percentile bootstrap instead, which makes no
distributional assumption:

```python
print(report.summary(method="bootstrap"))   # per call
report.ci_method = "bootstrap"              # or once, for every downstream view
```

Setting `ci_method` on the report changes the default for the plots and the
LaTeX exports too, so the figure and the table in a paper cannot disagree
about which interval they show.

!!! note "The bootstrap resamples seeds, not test examples"
    It is resampling the same $S$ numbers the *t* interval uses, so it cannot
    manufacture precision that ten runs do not contain. What it drops is the
    normality assumption, not the sample size. At $S < 10$ a percentile
    bootstrap is coarse in its own way — the 2.5th percentile of ten values
    is barely defined — so the honest move at small $S$ is to report the
    interval *and* the [minimum detectable effect](adequacy.md).

## Overlapping intervals are not a test

Two cells whose CIs overlap may still differ significantly, because the
paired test uses the per-seed *differences*, whose variance is typically far
smaller than either cell's. This is the single most common misreading of a
benchmark table.

$$\sigma_d^2 \;=\; \sigma_a^2 + \sigma_b^2 - 2\,\rho\,\sigma_a\sigma_b$$

With seeds paired, $\rho$ is usually large and positive: the seed that gave
every model a hard split gave *all* of them a hard split. The shared
difficulty cancels in the difference and does not cancel in either interval.

Read the intervals for *how well do we know this number*, and read
[`pairwise()`](comparison.md) for *is A better than B*. The
[Findings](../findings.md) page marks overlap in italics for exactly this
reason — as a caution, never as a verdict.

## Next

- [Comparison](comparison.md) — the paired tests these intervals do not replace.
- [Adequacy](adequacy.md) — what a wide interval implies about the design.
- [Reading the report](../guides/report.md) — `summary`, `plot_forest` and the exports.
