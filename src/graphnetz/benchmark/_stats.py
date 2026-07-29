"""Statistical helpers shared by the report and the runner."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy import stats

_METRIC_KEYS: tuple[str, ...] = (
    "test_acc",
    "test_auc",
    "val_acc",
    "val_auc",
    "val_mae",
)
_LOWER_IS_BETTER: frozenset[str] = frozenset({"val_mae", "train_loss"})

# Reported metric -> the validation metric that may legitimately select its
# epoch. Only pairs that live on *different* splits appear here: selecting the
# epoch on the same series that is then reported is optimistic bias, not
# checkpoint selection, so ``val_acc`` / ``val_mae`` have no entry.
_SELECTION_SOURCE: dict[str, str] = {
    "test_acc": "val_acc",
    "test_auc": "val_auc",
}

EPOCH_SELECTIONS: frozenset[str] = frozenset({"final", "best_val"})
"""How to reduce a per-epoch history to one number.

``"final"``
    The value at the last epoch -- a fixed-epoch protocol, and the default so
    that existing reports keep their meaning.
``"best_val"``
    The value at the epoch that optimises the paired *validation* metric, i.e.
    honest checkpoint selection. Requires the history to carry both a
    validation and a held-out series (see `_SELECTION_SOURCE`).
"""

# --------------------------------------------------------------------------- #
# Statistical helpers
# --------------------------------------------------------------------------- #


def _ci_half_width(values: np.ndarray, ci: float = 0.95) -> float:
    """Half-width of a t-distribution confidence interval for the mean."""
    n = values.size
    if n < 2:
        return 0.0
    sem = stats.sem(values)
    return float(sem * stats.t.ppf((1 + ci) / 2, n - 1))


def _bootstrap_ci_half_width(
    values: np.ndarray,
    ci: float = 0.95,
    n_resamples: int = 10000,
    random_state: int = 0,
) -> float:
    """Half-width of a percentile-bootstrap CI for the mean.

    Robust for non-Gaussian metrics (e.g. Hits@K, MRR, AUC) where the
    Student's-t assumption is poor. Returns ``(hi - lo) / 2`` -- the
    half-width of a symmetric envelope with the same total width as the
    percentile interval, so callers reporting ``mean ± half`` recover
    the bootstrap interval's spread without inflating asymmetric tails.
    """
    arr = np.asarray(values, dtype=float).ravel()
    n = arr.size
    if n < 2:
        return 0.0
    rng = np.random.default_rng(random_state)
    idx = rng.integers(0, n, size=(n_resamples, n))
    means = arr[idx].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(means, [alpha, 1.0 - alpha])
    return float((hi - lo) / 2.0)


def _resolve_ci_half(
    values: np.ndarray,
    ci: float,
    method: str,
    n_resamples: int,
    random_state: int,
) -> float:
    if method == "t":
        return _ci_half_width(values, ci)
    if method == "bootstrap":
        return _bootstrap_ci_half_width(values, ci, n_resamples, random_state)
    msg = f"Unknown CI method: {method!r}; choices: 't', 'bootstrap'"
    raise ValueError(msg)


def _paired_pvalue(a: np.ndarray, b: np.ndarray, method: str) -> float:
    """p-value of a paired test between two seed-aligned metric arrays.

    ``method="t"`` is the paired Student's t-test (parametric). ``method=
    "wilcoxon"`` is the Wilcoxon signed-rank test on the paired
    differences -- recommended at small seed counts where the paired
    t-test's normality assumption is most fragile (Benavoli et al.,
    JMLR 2016).
    """
    if a.size < 2 or b.size < 2 or a.size != b.size:
        return float("nan")
    if method == "t":
        return float(stats.ttest_rel(a, b).pvalue)
    if method == "wilcoxon":
        diffs = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
        # All-zero paired differences -> the signed-rank statistic has no
        # ranks to assign; return NaN so the row is reported as undefined
        # rather than as an artificial 1.0.
        if not np.any(diffs != 0):
            return float("nan")
        try:
            return float(stats.wilcoxon(diffs, zero_method="wilcox").pvalue)
        except ValueError:
            return float("nan")
    msg = f"Unknown pairwise method: {method!r}; choices: 't', 'wilcoxon'"
    raise ValueError(msg)


def _holm_correction(p_values: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values (max-monotone).

    NaN inputs (e.g. tests that were undefined for that pair) are
    excluded from the rank table and propagated as NaN in the output;
    they are *not* counted toward the family size, so the remaining
    valid tests retain their proper power.
    """
    p = np.asarray(p_values, dtype=float)
    n = p.size
    if n == 0:
        return p
    valid = ~np.isnan(p)
    n_valid = int(valid.sum())
    adjusted = np.full(n, np.nan, dtype=float)
    if n_valid == 0:
        return adjusted
    valid_idx = np.where(valid)[0]
    p_valid = p[valid_idx]
    order = np.argsort(p_valid)
    running = 0.0
    out_valid = np.empty(n_valid, dtype=float)
    for rank, idx in enumerate(order):
        adj = float(min(p_valid[idx] * (n_valid - rank), 1.0))
        running = max(running, adj)
        out_valid[idx] = running
    adjusted[valid_idx] = out_valid
    return adjusted


def _auto_metric_key(history: Mapping[str, Any]) -> str:
    for key in _METRIC_KEYS:
        if key in history:
            return key
    return next(iter(history))


def _final_metric(history: Mapping[str, list[float]]) -> tuple[str, float]:
    key = _auto_metric_key(history)
    return key, history[key][-1]


# --------------------------------------------------------------------------- #
# Epoch selection
# --------------------------------------------------------------------------- #


def _selection_reason(history: Mapping[str, Any], key: str) -> str | None:
    """Why ``best_val`` is unavailable for this history, or ``None`` if it is.

    Kept separate from `_selected_index` so callers can *ask* whether a
    task supports checkpoint selection without triggering an exception.
    """
    source = _SELECTION_SOURCE.get(key)
    if source is None:
        return (
            f"metric {key!r} is itself a validation metric; selecting its epoch "
            f"on the same series and then reporting it would be optimistically "
            f"biased. The trainer records no held-out series for this task."
        )
    if source not in history:
        return f"history has {sorted(history)} and lacks the {source!r} series needed to select an epoch"
    return None


def _selected_index(history: Mapping[str, list[float]], key: str, selection: str) -> int:
    """Index into ``history[key]`` chosen by ``selection``.

    ``"best_val"`` picks the epoch optimising the *paired validation* series
    and returns that index, so the reported value comes from a split that was
    not used to choose it.
    """
    if selection == "final":
        return -1
    if selection not in EPOCH_SELECTIONS:
        msg = f"Unknown epoch_selection {selection!r}; choices: {sorted(EPOCH_SELECTIONS)}"
        raise ValueError(msg)
    reason = _selection_reason(history, key)
    if reason is not None:
        msg = f"epoch_selection='best_val' unavailable: {reason}"
        raise ValueError(msg)
    source = _SELECTION_SOURCE[key]
    values = np.asarray(history[source], dtype=float)
    if values.size == 0:
        msg = f"epoch_selection='best_val' unavailable: {source!r} series is empty"
        raise ValueError(msg)
    return int(values.argmin() if source in _LOWER_IS_BETTER else values.argmax())


def _selected_value(history: Mapping[str, list[float]], key: str, selection: str) -> tuple[float, int]:
    """The reduced metric plus the (1-based) epoch it was taken from."""
    idx = _selected_index(history, key, selection)
    series = history[key]
    return float(series[idx]), (len(series) if idx == -1 else idx + 1)


# --------------------------------------------------------------------------- #
# Rank aggregation across tasks
# --------------------------------------------------------------------------- #


def _rank_rows(
    finals: Mapping[str, Mapping[str, list[float]]],
    directions: Mapping[str, bool],
    models: list[str],
    tasks: list[str],
) -> np.ndarray:
    """Per-task ranks of the seed means, rank 1 = best, ties averaged.

    ``directions[task]`` is True when lower is better for that task, so a
    benchmark mixing accuracy with MAE ranks correctly in one table.
    """
    rows: list[np.ndarray] = []
    for task in tasks:
        means = np.array([float(np.mean(finals[task][m])) for m in models])
        sign = 1.0 if directions[task] else -1.0
        rows.append(stats.rankdata(sign * means, method="average"))
    return np.asarray(rows, dtype=float)


def _nemenyi_cd(k: int, n: int, alpha: float = 0.05) -> float:
    """Nemenyi critical difference ``q_alpha sqrt(k(k+1)/6N)``."""
    from scipy.stats import studentized_range

    if k < 2 or n < 1:
        return float("nan")
    q = float(studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2))
    return q * float(np.sqrt(k * (k + 1) / (6 * n)))


def _friedman_chi2(ranks: np.ndarray) -> float:
    """Friedman statistic from a ``[n_tasks, k]`` rank table."""
    n, k = ranks.shape
    avg = ranks.mean(axis=0)
    return float((12.0 * n) / (k * (k + 1)) * (float(np.sum(avg**2)) - k * (k + 1) ** 2 / 4.0))


def _normalized_weights(weights: Mapping[str, float] | None, tasks: list[str]) -> np.ndarray:
    """Task weights as a length-``n_tasks`` array summing to ``n_tasks``.

    Normalising to ``n_tasks`` (not 1) keeps a weighted mean rank on the same
    scale as the uniform one, so the two are directly comparable on a diagram.
    """
    if weights is None:
        return np.ones(len(tasks), dtype=float)
    w = np.array([float(weights.get(t, 0.0)) for t in tasks], dtype=float)
    if np.any(w < 0):
        msg = "task weights must be non-negative"
        raise ValueError(msg)
    total = w.sum()
    if total <= 0:
        msg = "task weights must not all be zero"
        raise ValueError(msg)
    return w * (len(tasks) / total)
