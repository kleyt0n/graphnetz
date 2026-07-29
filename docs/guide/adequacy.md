# Is the evidence sufficient?

A non-significant paired test is compatible with two very different states of
the world:

- the models really are close, or
- the design could not have told the difference either way.

Reporting `p > 0.05` without distinguishing them is how benchmarks end up
claiming that architectures are "equivalent" when all they have shown is that
they ran too few seeds. This page covers the three views that keep the two
apart. They read the same seed-paired data as
[`pairwise`][graphnetz.benchmark.BenchmarkReport.pairwise], so none of them
requires retraining.

!!! abstract "The short version"

    | Question | Method |
    |---|---|
    | What is the smallest difference this design could detect? | [`power()`][graphnetz.benchmark.BenchmarkReport.power] |
    | Can I claim these two models are *equivalent*? | [`equivalence()`][graphnetz.benchmark.BenchmarkReport.equivalence] |
    | How many benchmark tasks would the observed rank gap need? | [`friedman()`][graphnetz.benchmark.BenchmarkReport.friedman] |

## What could this design detect?

```python
power = report.power(alpha=0.05, target_power=0.80)
power[["task", "model_a", "model_b", "mean_diff", "mde", "observed_power", "detectable"]]
```

`mde` is the **minimum detectable effect**: the smallest true paired difference
the test would reject at `alpha` with probability `target_power`. A comparison
whose observed `|mean_diff|` falls below its own `mde` is *undetermined* — the
test could not have found an effect of that size even if one existed.

$$
\mathrm{MDE} = \bigl(t_{S-1,\,1-\alpha/2} + t_{S-1,\,1-\beta}\bigr)\,
               \frac{\hat\sigma_d}{\sqrt{S}}
$$

The `seeds_for_*` columns invert the relationship, which is usually the more
actionable direction:

```python
power[["task", "model_a", "model_b", "sigma_d", "seeds_for_0.01", "seeds_for_0.005"]]
```

!!! example "Ten seeds is less than it sounds"

    On a three-dataset Planetoid sweep with `S = 10`, the median MDE is about
    **0.014** accuracy points and 7 of 18 comparisons fall below their own MDE.
    Resolving half a point at that noise level would take roughly **60 seeds**,
    not ten. Nothing is wrong with the runs — the design simply has a floor,
    and it is better to report it than to imply the models are tied.

`observed_power` comes from the non-central *t* distribution rather than a
normal approximation, which matters at the seed counts benchmarks actually use.

## Can I claim equivalence?

Not from a large *p*-value. Use two one-sided tests (TOST) against a
**smallest effect size of interest** — the difference below which you would not
care:

```python
eq = report.equivalence(margin=0.01)          # 0.01 = one accuracy/AUC point
eq["verdict"].value_counts()
```

Crossing the difference test with the equivalence test partitions every
comparison into exactly one of four states:

| Difference test | TOST | Verdict | Meaning |
|---|---|---|---|
| rejects | does not reject | `different` | a real effect larger than `margin` |
| does not reject | rejects | `equivalent` | any effect is smaller than `margin` |
| rejects | rejects | `trivial` | real, but smaller than `margin` |
| does not reject | does not reject | `undetermined` | underpowered — the data support neither claim |

Both families are Holm-corrected within a task, matching `pairwise()`.

!!! warning "`margin` is a judgement call, so state it"

    There is no default that is right for every metric. Pick the smallest
    difference that would change a decision, and report the number alongside
    the verdicts — an equivalence claim is meaningless without it.

## How much benchmark would the ranking need?

The Nemenyi critical difference depends only on the number of models $k$, the
number of tasks $N$, and $\alpha$:

$$
CD_\alpha = q_\alpha\sqrt{\frac{k(k+1)}{6N}}
$$

so the benchmark breadth required to resolve a given mean-rank gap is a
property of the *design*, computable without running anything:

```python
f = report.friedman(alpha=0.05)
f["critical_difference"]   # the resolution of the current diagram
f["max_rank_gap"]          # the largest gap actually observed
f["n_for_observed_gap"]    # tasks at which that gap would become significant
```

Because $CD_\alpha \propto 1/\sqrt{N}$, this often reveals that the ordering
you are looking at is far out of reach: a gap of 0.80 rank units with $k = 4$
needs $N \geq 35$ tasks, which is more than most catalogues expose. Reporting
that is more useful than promising that a slightly larger sweep would settle it.

## How much does the task sample matter?

A CD diagram at small $N$ says nothing about whether the ordering would survive
a different draw of benchmark tasks.
[`rank_stability`][graphnetz.benchmark.BenchmarkReport.rank_stability] resamples
the tasks themselves:

```python
st = report.rank_stability(n_boot=10_000)
st["order_stability"]        # fraction of resamples reproducing the full order
st["separation"]             # per pair: how often it separates by more than CD
st["jackknife"]              # mean-rank shift when each task is dropped
```

A low `order_stability` with a uniformly low `separation` is the honest
signature of "the ordering is noise, and the non-result is robust" — which is a
finding, not a failure.

## Putting it together

```python
report = run_benchmark("social", MODELS, seeds=range(10))

pw   = report.pairwise()                  # is there a difference?
pwr  = report.power()                     # could we have seen one?
eq   = report.equivalence(margin=0.01)    # can we claim there isn't?
f    = report.friedman()                  # what would it take across tasks?
```

Read together, these four answer the question a benchmark is actually asked —
*what does this evidence support?* — rather than the narrower question of
whether one number exceeded another.

## See also

- [Reading the report](report.md) — the other views and the export helpers.
- [Benchmark protocol](benchmark.md) — how the seed-paired data is produced.
- [`graphnetz.benchmark`](../reference/benchmark.md) — full API reference.
