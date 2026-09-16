# Concepts

A benchmark answers *"is this model better?"*. Answering it honestly takes
three separate things, and conflating them is how graph-learning papers end
up claiming more than they measured:

1. an **estimate** with its uncertainty, per cell,
2. a **comparison** that accounts for how many comparisons were made,
3. a statement of what the design **could** have detected, so a
   non-significant result is not silently read as a tie.

GraphNetz treats all three as the default output. This section covers the
vocabulary; the [Guides](../guides/benchmark.md) cover the calls.

## The shape of the data

Every run reduces to one metric tensor, and every statistic on this site is a
function of it:

$$X[t, m, s] \;=\; \text{the final metric of model } m \text{ on task } t
\text{ at seed } s$$

Seeds are **paired**: seed $s$ means the same split, the same initialisation
draw and the same data resample for every model on task $t$. That pairing is
what licenses the paired tests in [Comparison](comparison.md), and it is the
one property the runner protects above all others.

| axis | size | set by |
| --- | --- | --- |
| task $t$ | $N$ | the category, or an explicit `tasks=` list |
| model $m$ | $k$ | the `models` dict |
| seed $s$ | $S$ | `seeds=`, default `range(10)` |

## Three questions, three answers

| question | what answers it | page |
| --- | --- | --- |
| How good is this cell, and how sure are we? | mean ± CI over $S$ seeds | [Uncertainty](uncertainty.md) |
| Is model A better than model B here? | Holm-corrected paired test | [Comparison](comparison.md) |
| Which model wins across tasks? | Friedman ranks, Nemenyi CD | [Comparison](comparison.md) |
| Could we have detected a difference at all? | MDE, observed power, TOST | [Adequacy](adequacy.md) |
| Would the same seeds give the same answer? | reseeding, cache, epoch choice | [Reproducibility](reproducibility.md) |

## Read in this order

<div class="grid cards" markdown>

-   __1. [Tasks and metrics](tasks.md)__

    ---

    The four task families, the default metric for each, and why unlabelled
    graphs enter through link prediction rather than a pretext loss.

-   __2. [Uncertainty](uncertainty.md)__

    ---

    Where the interval on every cell comes from, when Student's *t* is the
    wrong instrument, and what the bootstrap buys you.

-   __3. [Comparison](comparison.md)__

    ---

    Paired tests within a task, Holm correction across a family, and
    Friedman–Nemenyi ranks across tasks. Which one answers which question.

-   __4. [Adequacy](adequacy.md)__

    ---

    A large *p*-value is not evidence of a tie. Minimum detectable effect,
    equivalence testing, and how much benchmark breadth a ranking needs.

-   __5. [Reproducibility](reproducibility.md)__

    ---

    What gets reseeded and when, why the epoch you score at is a protocol
    choice, and what the JSON round-trip guarantees.

</div>
