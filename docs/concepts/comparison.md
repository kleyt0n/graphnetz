# Comparison

Two different questions get asked of a benchmark, and they need two different
procedures:

- **within a task** — is model A better than model B *on this dataset*?
- **across tasks** — does one model win *in general*?

GraphNetz answers the first with Holm-corrected paired tests and the second
with Friedman ranks and a Nemenyi critical difference. Using either
procedure for the other question is a mistake the library tries hard to make
difficult.

## Within a task: paired tests

Seeds are paired, so the natural statistic is the per-seed difference:

```python
print(report.pairwise(alpha=0.05))                     # paired t-test (default)
print(report.pairwise(alpha=0.05, method="wilcoxon"))  # non-parametric
```

Columns: `task, model_a, model_b, mean_diff, p_raw, p_holm, significant`.

The default is a paired *t*-test on $d_s = x_{a,s} - x_{b,s}$. Pairing is
what makes this powerful at ten seeds: the seed-to-seed difficulty that
dominates each model's own variance cancels in the difference. See
[Uncertainty](uncertainty.md#overlapping-intervals-are-not-a-test).

### Choosing t or Wilcoxon

| | paired *t* | Wilcoxon signed-rank |
| --- | --- | --- |
| assumes | differences roughly normal | differences symmetric |
| uses | the size of each difference | the rank of each difference |
| best at | $S \gtrsim 10$, well-behaved metrics | small $S$, skewed or bounded metrics |
| cost | fragile when normality fails | discards magnitude, so less powerful when it holds |

Wilcoxon is the recommendation at small seed counts, where the *t*-test's
normality assumption is least defensible (Benavoli et al., *JMLR*
17(5):1-36, 2016):

```python
report.pairwise_method = "wilcoxon"   # every downstream table and plot follows
```

!!! note "Exact ties are reported as undefined, not as p = 1"
    If every paired difference is exactly zero the signed-rank statistic has
    no value. The row reports `NaN` rather than an artificial $p = 1.0$,
    because "the test does not apply here" and "the test ran and found
    nothing" are different states and only one of them is evidence.

## Correcting for multiplicity

Comparing $k$ models on a task means $\binom{k}{2}$ tests. At $k = 4$ that is
six, and at $\alpha = 0.05$ the chance of at least one false positive among
six independent tests is about 26 %. Reporting raw *p*-values across a table
of comparisons is how benchmarks manufacture significance for free.

GraphNetz applies **Holm step-down** within each task, controlling the
family-wise error rate:

$$p_i^{\text{adj}} \;=\; \min\!\big(p_{(i)}\,(k - i),\, 1\big)$$

Holm rather than plain Bonferroni because it is uniformly more powerful and
requires no extra assumptions: it is strictly better, not a different
trade-off. The family is the **task**, so a comparison is corrected against
the other comparisons on its own dataset, not against every comparison in the
sweep — correcting across tasks as well would confound the two questions this
page is about keeping apart.

Both `p_raw` and `p_holm` are in the table. Report the adjusted one.

## Across tasks: Friedman and Nemenyi

Rank the $k$ models on each of $N$ tasks, average the ranks, and ask whether
the spread is larger than chance. This is the canonical scalable view for
multi-method, multi-dataset comparison (Demšar, 2006).

```python
fig, _ = report.plot_critical_difference(alpha=0.05)
f = report.friedman(alpha=0.05)
```

The Friedman test is the omnibus: *is any model different from any other?*
If it does not reject, the ordering you are looking at is not evidence of
anything and the procedure stops there. If it does, Nemenyi gives the
resolution of the diagram:

$$CD_\alpha \;=\; q_\alpha\,\sqrt{\frac{k(k+1)}{6N}}$$

Two models are distinguishable only if their mean ranks differ by more than
$CD_\alpha$; the clique bars in the diagram join those that are not. Note
what $N$ does here: the resolution improves only as $\sqrt{N}$, so halving
the detectable rank gap takes **four times** the benchmark breadth. That is
the arithmetic behind the [Findings](../findings.md) result, and it is
computable before running anything — see
[Adequacy](adequacy.md#how-much-benchmark-would-the-ranking-need).

Ranks also normalise the metric away, which is why a lower-is-better `graph_reg`
task can share a diagram with accuracy tasks; the direction is applied per
task before averaging.

## Task weighting, and the line it crosses

Demšar weights every task equally, so a 200-node synthetic lattice counts as
much as a large public benchmark. That is a modelling choice, and it is
exposed:

```python
report.mean_ranks()                              # uniform (the default)
report.mean_ranks(aggregation="reliability")     # inverse CI width
report.mean_ranks(aggregation="hierarchical", groups=task_to_category)
report.mean_ranks(aggregation="custom", weights={"cora": 2.0, "citeseer": 1.0})
```

!!! danger "Only the uniform weighting has a Friedman/Nemenyi null"
    Weighting breaks the exchangeability those procedures assume, so a
    weighted mean rank **must not** be compared against $CD_\alpha$.
    `plot_critical_difference(aggregation=…)` enforces this: on a weighted
    diagram it refuses to draw the CD reference and the clique bars, and
    labels the figure a diagnostic. Use
    [`rank_stability()`](adequacy.md#how-much-does-the-task-sample-matter)
    for uncertainty on the weighted variants instead.

## Choosing the right procedure

| you want to say | use | do not use |
| --- | --- | --- |
| "A beats B on Cora" | `pairwise()`, Holm-adjusted | overlapping CIs |
| "A beats B overall" | Friedman, then Nemenyi CD | a count of per-task wins |
| "A and B are equivalent" | [`equivalence(margin=…)`](adequacy.md#can-i-claim-equivalence) | a large *p*-value |
| "this weighting favours A" | `mean_ranks(aggregation=…)` | a weighted CD diagram |

## Next

- [Adequacy](adequacy.md) — what to do when nothing reaches significance.
- [Reading the report](../guides/report.md) — the plots these tests feed.
- [Benchmark protocol](../guides/benchmark.md) — how the seed-paired data is produced.
