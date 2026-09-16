# Hyperparameter search

By default every model on a task sees the same optimiser settings. That makes
the benchmark a measurement of **model quality at those settings**, which is
a defensible protocol and a clearly stated one — but it systematically
disadvantages architectures whose usual recipes differ from the shared
default.

A `SearchSpace` adds an inner loop per *(task, model, seed)* so each model is
measured closer to its own best.

## Running one

```python
from graphnetz import SearchSpace, run_benchmark

grid = SearchSpace(lr=(1e-2, 1e-3), weight_decay=(5e-4, 0.0))
report = run_benchmark("social", MODELS, seeds=range(10), search=grid)
```

Any keyword accepted by the trainers can be an axis. Beyond two or three
axes, an exhaustive grid stops being the best use of a fixed budget — use
`n_random=` for a random search instead, which is a strictly better allocation
in that regime.

## Selection happens on validation, never on test

Candidates are scored on the task's **validation** series. The framework
**refuses to run** when a task exposes no validation series, rather than
falling back to the held-out metric.

!!! danger "Why the refusal matters"
    A search that peeks at test inflates every downstream statistic — the
    interval, the *p*-value, the rank — while looking like a *more* careful
    protocol than one that does not search at all. It is the failure mode
    most likely to survive review, because the extra machinery reads as
    rigour. Refusing is the only safe default.

## What gets recorded

The selected configuration and the full trace are stored per run, so the
search is auditable rather than a black box between you and the number:

```python
report.config["search_selected"]["cora/GAT/seed0"]
# {'selected': {'lr': 0.001, 'weight_decay': 0.0},
#  'trace': [{'lr': 0.01, 'weight_decay': 0.0005, 'val_score': 0.772}, ...]}
```

Two things worth looking at in the trace:

- **A configuration selected at the edge of the grid** means the grid was too
  narrow, and the reported number is a lower bound on what the model can do.
- **A selection that varies wildly across seeds** means the validation series
  is too noisy to choose on, and the inner loop is adding variance rather
  than removing bias.

## The cost

!!! warning "It multiplies compute"
    A grid of $G$ candidates multiplies training by $G$, per *(task, model,
    seed)*. With 4 models × 6 tasks × 10 seeds × a 4-point grid, that is 960
    trainings rather than 240. The runner announces the multiplier before it
    starts.

    Search **one category first and measure**, rather than launching a full
    catalogue sweep and discovering the budget afterwards.

## Search and the statistics

Selection is inside the seed loop, so it is part of what the seed varies.
That is deliberate: the per-seed metric remains an honest sample of *the
whole procedure*, search included, and the paired tests in
[Comparison](../concepts/comparison.md) stay valid without modification.

The practical consequence is that $\sigma_d$ usually **grows** — the inner
loop adds its own variance — so a search can widen intervals and cost you
significance even while raising every mean. Check
[`power()`](../concepts/adequacy.md#what-could-this-design-detect) after a
search, not only before it.

## When not to search

- **Comparing against published numbers** that were themselves produced at
  fixed settings. Searching one side of that comparison and not the other is
  the same error in a different direction.
- **At small seed counts**, where the added variance costs more than the bias
  it removes.
- **When the claim is about the protocol**, not the models — which is what
  the [findings](../findings.md) are. Every number there is fixed-epoch,
  fixed-hyperparameter, and says so.

## Next

- [Benchmark protocol](benchmark.md) — the five stages this sits inside.
- [Adequacy](../concepts/adequacy.md) — what the added variance does to power.
- [Custom models](custom-models.md) — variants as models, for very small sweeps.
