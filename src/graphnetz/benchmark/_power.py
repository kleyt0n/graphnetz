"""Power, minimum detectable effect, and equivalence testing.

A non-significant paired test is not evidence of equivalence.  Reporting
``p > alpha`` without saying what the design *could* have detected invites
exactly the over-reading this framework exists to prevent, so the report
exposes three companion quantities on the same seed-paired scaffolding as
[`pairwise`][graphnetz.benchmark.BenchmarkReport.pairwise]:

**Minimum detectable effect (MDE).**  The smallest true paired difference the
design would reject at level ``alpha`` with probability ``power``.  Under the
paired-*t* null with ``S`` seeds,

$$
\\mathrm{MDE} = \\bigl(t_{S-1,1-\\alpha/2} + t_{S-1,\\mathrm{power}}\\bigr)
\\frac{\\hat\\sigma_d}{\\sqrt{S}},
$$

the standard normal-approximation form with *t* quantiles substituted.  A cell
whose observed ``|d̄|`` falls below its own MDE is *undetermined*, not tied.

**Observed power.**  The probability of rejecting at the observed effect size,
computed from the non-central *t* distribution rather than the normal
approximation, so it is correct at the small ``S`` this framework targets.

**Equivalence (TOST).**  Two one-sided tests against a smallest effect size of
interest ``margin``.  Rejecting both one-sided nulls licenses the positive
claim "these models are equivalent within ``margin``" that a large *p*-value
alone does not support (Lakens, 2017).

Combining TOST with the paired test partitions every comparison into exactly
one of four verdicts, which is the vocabulary the report should have had from
the start:

| paired test | TOST | verdict |
| --- | --- | --- |
| rejects | does not reject | `different` |
| does not reject | rejects | `equivalent` |
| does not reject | does not reject | `undetermined` (underpowered) |
| rejects | rejects | `trivial` (real but < margin) |
"""

from __future__ import annotations

import numpy as np
from scipy import stats

__all__ = [
    "equivalence_verdict",
    "minimum_detectable_effect",
    "observed_power",
    "paired_tost",
    "seeds_for_effect",
]


def minimum_detectable_effect(
    sigma_d: float,
    n: int,
    alpha: float = 0.05,
    power: float = 0.80,
) -> float:
    """Smallest paired difference detectable at ``power`` with ``n`` observations.

    ``sigma_d`` is the standard deviation of the paired differences.  Returns
    NaN when ``n < 2`` (no variance estimate) so the value propagates rather
    than silently reading as zero.
    """
    if n < 2 or not np.isfinite(sigma_d):
        return float("nan")
    t_alpha = float(stats.t.ppf(1 - alpha / 2, n - 1))
    t_power = float(stats.t.ppf(power, n - 1))
    return float((t_alpha + t_power) * sigma_d / np.sqrt(n))


def seeds_for_effect(
    effect: float,
    sigma_d: float,
    alpha: float = 0.05,
    power: float = 0.80,
    max_n: int = 10_000,
) -> int:
    """Smallest ``n`` whose MDE is at most ``effect``.

    The design question in the other direction: "how many seeds would I need
    to call a difference of this size?"  Returns ``max_n`` if the target is
    not reached within the search bound.
    """
    if effect <= 0 or not np.isfinite(sigma_d) or sigma_d <= 0:
        return 2
    for n in range(2, max_n + 1):
        if minimum_detectable_effect(sigma_d, n, alpha, power) <= effect:
            return n
    return max_n


def observed_power(
    effect: float,
    sigma_d: float,
    n: int,
    alpha: float = 0.05,
) -> float:
    """Power of the two-sided paired *t*-test at the observed effect size.

    Uses the non-central *t* distribution with non-centrality
    ``|effect| √n / sigma_d``, which is exact for the paired-*t* design; the
    normal approximation is optimistic at the seed counts used here.
    """
    if n < 2 or sigma_d <= 0 or not np.isfinite(sigma_d) or not np.isfinite(effect):
        return float("nan")
    df = n - 1
    ncp = abs(effect) * np.sqrt(n) / sigma_d
    crit = float(stats.t.ppf(1 - alpha / 2, df))
    upper = float(stats.nct.sf(crit, df, ncp))
    lower = float(stats.nct.cdf(-crit, df, ncp))
    return float(min(1.0, upper + lower))


def paired_tost(
    a: np.ndarray,
    b: np.ndarray,
    margin: float,
    alpha: float = 0.05,
) -> float:
    """*p*-value of the two one-sided tests for equivalence within ``margin``.

    Tests ``H01: d >= margin`` and ``H02: d <= -margin`` on the paired
    differences ``d = a - b``; the TOST *p*-value is the larger of the two
    one-sided *p*-values, so ``p < alpha`` rejects both nulls and concludes
    equivalence.  ``margin`` must be positive and is expressed in the metric's
    own units.
    """
    if margin <= 0:
        msg = f"TOST margin must be positive, got {margin}"
        raise ValueError(msg)
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    n = d.size
    if n < 2:
        return float("nan")
    sd = float(d.std(ddof=1))
    if sd == 0.0:
        # Zero variance: equivalent iff the constant difference is inside the
        # margin. No test statistic exists, so answer degenerately but honestly.
        return 0.0 if abs(float(d.mean())) < margin else 1.0
    se = sd / np.sqrt(n)
    mean = float(d.mean())
    df = n - 1
    p_lower = float(stats.t.sf((mean + margin) / se, df))  # H02: d <= -margin
    p_upper = float(stats.t.cdf((mean - margin) / se, df))  # H01: d >=  margin
    return float(max(p_lower, p_upper))


def equivalence_verdict(
    p_difference: float,
    p_equivalence: float,
    alpha: float = 0.05,
) -> str:
    """Combine a difference test and a TOST into one of four verdicts.

    See the module docstring for the table.  ``undetermined`` is the honest
    answer when neither test rejects -- the data are compatible both with a
    real difference and with equivalence.
    """
    if np.isnan(p_difference) or np.isnan(p_equivalence):
        return "undetermined"
    diff = p_difference < alpha
    equiv = p_equivalence < alpha
    if diff and not equiv:
        return "different"
    if equiv and not diff:
        return "equivalent"
    if diff and equiv:
        return "trivial"
    return "undetermined"
