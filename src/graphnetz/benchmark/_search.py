"""Optional inner hyperparameter search for the benchmark protocol.

Errica et al. (2020) argue that a benchmark should be honest over the
hyperparameter grid, not only over seeds.  A framework that fixes
hyperparameters across models therefore reports something narrower than model
quality: it reports model quality *at those hyperparameters*.  This module adds
the inner loop that closes the gap, as line 4a of Algorithm 1.

The search is deliberately constrained in one respect: candidates are scored on
the task's **validation** series and never on its held-out metric.  A search
that peeked at test would inflate every downstream statistic while looking like
a more careful protocol, so `select` refuses to run when the task exposes
no validation series.

Cost is the obvious caveat and is reported up front by the runner: a grid of
$G$ candidates multiplies training by $G$, per (task, model, seed).
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from graphnetz.benchmark._stats import _LOWER_IS_BETTER

__all__ = ["SearchSpace", "select"]

# Validation series a candidate may legitimately be scored on, in preference
# order. Deliberately excludes every ``test_*`` key.
_VALIDATION_KEYS: tuple[str, ...] = ("val_acc", "val_auc", "val_mae")


@dataclass(frozen=True)
class SearchSpace:
    """Hyperparameters to search over, and how to enumerate them.

    Only knobs the training routines already expose are searchable:
    ``lr`` and ``weight_decay`` (the optimiser) and ``hidden_channels`` (the
    encoder width).  An empty space means "no search", so
    ``SearchSpace()`` reproduces the unsearched protocol exactly.

    ``n_random`` switches from an exhaustive grid to that many random draws
    (without replacement where the grid is smaller), which is the better use of
    a fixed budget once the grid has more than two or three axes
    (Bergstra & Bengio, 2012).
    """

    lr: Sequence[float] = ()
    weight_decay: Sequence[float] = ()
    hidden_channels: Sequence[int] = ()
    n_random: int | None = None
    seed: int = 0
    _axes: tuple[str, ...] = field(default=("lr", "weight_decay", "hidden_channels"), repr=False)

    def candidates(self) -> list[dict[str, Any]]:
        """Enumerate the candidate configurations, always at least one."""
        axes = {name: tuple(getattr(self, name)) for name in self._axes}
        active = {k: v for k, v in axes.items() if v}
        if not active:
            return [{}]
        keys = list(active)
        grid = [dict(zip(keys, combo, strict=False)) for combo in itertools.product(*(active[k] for k in keys))]
        if self.n_random is None or self.n_random >= len(grid):
            return grid
        rng = np.random.default_rng(self.seed)
        idx = rng.choice(len(grid), size=self.n_random, replace=False)
        return [grid[int(i)] for i in sorted(idx)]

    def __len__(self) -> int:
        return len(self.candidates())

    @property
    def is_empty(self) -> bool:
        return len(self.candidates()) <= 1

    def describe(self) -> str:
        axes = ", ".join(f"{name}={tuple(getattr(self, name))}" for name in self._axes if getattr(self, name))
        mode = "grid" if self.n_random is None else f"random({self.n_random})"
        return f"{mode} over {{{axes}}} -> {len(self)} candidates"


def _validation_key(history: Mapping[str, Any]) -> str | None:
    for key in _VALIDATION_KEYS:
        if key in history and len(history[key]):
            return key
    return None


def score(history: Mapping[str, Any]) -> tuple[float, str]:
    """Selection score for one candidate: its final validation value.

    Returns ``(score, key)`` where higher is always better, so the caller can
    maximise regardless of the metric's natural direction.  Raises when the
    history exposes no validation series, rather than silently falling back to
    a held-out metric.
    """
    key = _validation_key(history)
    if key is None:
        msg = (
            f"hyperparameter search needs a validation series to score candidates; "
            f"this task's history has {sorted(history)} and exposes none of {list(_VALIDATION_KEYS)}. "
            f"Scoring on a held-out metric would bias every downstream statistic."
        )
        raise ValueError(msg)
    value = float(history[key][-1])
    return (-value if key in _LOWER_IS_BETTER else value), key


def select(
    trials: Iterator[tuple[dict[str, Any], dict[str, list[float]]]],
) -> tuple[dict[str, Any], dict[str, list[float]], list[dict[str, Any]]]:
    """Pick the best candidate from ``(config, history)`` pairs.

    Returns ``(best_config, best_history, trace)``; the trace records every
    candidate's configuration and validation score so a run's search is
    auditable after the fact.  Ties keep the first candidate, which makes the
    outcome deterministic given a fixed candidate order.
    """
    best: tuple[float, dict[str, Any], dict[str, list[float]]] | None = None
    trace: list[dict[str, Any]] = []
    for config, history in trials:
        value, key = score(history)
        trace.append({**config, "val_metric": key, "val_score": round(value, 6)})
        if best is None or value > best[0]:
            best = (value, config, history)
    if best is None:
        msg = "hyperparameter search received no trials"
        raise ValueError(msg)
    return best[1], best[2], trace
