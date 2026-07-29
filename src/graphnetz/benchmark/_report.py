"""The [`BenchmarkReport`][graphnetz.benchmark.BenchmarkReport] container: statistics and LaTeX tables."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from graphnetz.benchmark._power import (
    equivalence_verdict,
    minimum_detectable_effect,
    observed_power,
    paired_tost,
    seeds_for_effect,
)
from graphnetz.benchmark._report_plots import _ReportPlotsMixin
from graphnetz.benchmark._stats import (
    _LOWER_IS_BETTER,
    _auto_metric_key,
    _friedman_chi2,
    _holm_correction,
    _nemenyi_cd,
    _normalized_weights,
    _paired_pvalue,
    _rank_rows,
    _resolve_ci_half,
    _selected_value,
    _selection_reason,
)

# --------------------------------------------------------------------------- #
# Benchmark report
# --------------------------------------------------------------------------- #


@dataclass
class BenchmarkReport(_ReportPlotsMixin):
    """Structured outcome of a multi-seed benchmark run.

    ``histories[task][model]`` is a list with one history dict per seed (in
    seed order). The report is also a read-only mapping ``task -> {model:
    history_seed_0}`` for backward compatibility with single-seed callers.
    """

    seeds: tuple[int, ...]
    histories: dict[str, dict[str, list[dict[str, list[float]]]]]
    config: dict[str, Any] = field(default_factory=dict)
    ci_method: str = "t"
    bootstrap_n: int = 10000
    bootstrap_seed: int = 0
    pairwise_method: str = "t"
    epoch_selection: str = "final"
    """How each per-epoch history is reduced to one number.

    ``"final"`` (default) is the fixed-epoch protocol.  ``"best_val"`` selects
    the epoch optimising the paired validation series and reports the held-out
    metric there -- proper checkpoint selection.  Setting it once switches the
    summary, the pairwise tests, the rank aggregation, every plot and the LaTeX
    exporters together.
    """

    def _ci_half(
        self,
        values: np.ndarray,
        ci: float,
        method: str | None = None,
    ) -> float:
        return _resolve_ci_half(
            values,
            ci,
            method or self.ci_method,
            self.bootstrap_n,
            self.bootstrap_seed,
        )

    # ----- Pickle compatibility ---------------------------------------------

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore from pickle, backfilling fields added since serialisation.

        Older [`BenchmarkReport`][graphnetz.benchmark.BenchmarkReport] pickles predate the ``ci_method`` /
        ``bootstrap_*`` / ``pairwise_method`` fields. ``__setstate__``
        ensures they load cleanly with sensible defaults so the experiment
        cache (``_cache_*.pkl``) survives library upgrades.
        """
        self.__dict__.update(state)
        self.__dict__.setdefault("ci_method", "t")
        self.__dict__.setdefault("bootstrap_n", 10000)
        self.__dict__.setdefault("bootstrap_seed", 0)
        self.__dict__.setdefault("pairwise_method", "t")
        self.__dict__.setdefault("epoch_selection", "final")
        self.__dict__.setdefault("config", {})

    # ----- Mapping protocol (backward compat with the legacy dict shape) -----

    def __iter__(self):
        return iter(self.histories)

    def __len__(self) -> int:
        return len(self.histories)

    def __getitem__(self, task_type: str) -> dict[str, dict[str, list[float]]]:
        per_task = self.histories[task_type]
        return {model: per_task[model][0] for model in per_task}

    def items(self):
        for task in self.histories:
            yield task, self[task]

    def keys(self):
        return self.histories.keys()

    def values(self):
        return [self[task] for task in self.histories]

    # ----- Statistics --------------------------------------------------------

    def final_metrics(
        self,
        key: str | None = None,
        *,
        epoch_selection: str | None = None,
        strict: bool = True,
    ) -> dict[str, dict[str, list[float]]]:
        """Reduced metric value per (task, model, seed).

        ``epoch_selection`` overrides [`epoch_selection`][graphnetz.benchmark.BenchmarkReport.epoch_selection] for this call.
        Under ``"best_val"`` a task whose trainer records no held-out series
        cannot be selected honestly; ``strict=True`` raises, ``strict=False``
        falls back to the final epoch for that task alone (use
        [`epoch_selection_support`][graphnetz.benchmark.BenchmarkReport.epoch_selection_support] to see which tasks fell back).
        """
        selection = epoch_selection or self.epoch_selection
        out: dict[str, dict[str, list[float]]] = {}
        for task, per_task in self.histories.items():
            out[task] = {}
            task_selection = selection
            if selection != "final" and not strict:
                sample = next(iter(per_task.values()), None)
                if sample and _selection_reason(sample[0], key or _auto_metric_key(sample[0])) is not None:
                    task_selection = "final"
            for model, seed_histories in per_task.items():
                vals: list[float] = []
                for h in seed_histories:
                    k = key or _auto_metric_key(h)
                    vals.append(_selected_value(h, k, task_selection)[0])
                out[task][model] = vals
        return out

    def epoch_selection_support(self, key: str | None = None) -> pd.DataFrame:
        """Per task: whether ``"best_val"`` is available, and why not if it is not.

        Checkpoint selection needs a validation series *and* a held-out series
        in the same history.  Node classification and both link-prediction
        trainers record both; the graph-level trainers record only a validation
        metric, so selecting on it and reporting it would be optimistically
        biased.  This surfaces that asymmetry instead of hiding it.
        """
        rows = []
        for task, per_task in self.histories.items():
            sample = next(iter(per_task.values()), None)
            if not sample:
                continue
            metric = key or _auto_metric_key(sample[0])
            reason = _selection_reason(sample[0], metric)
            rows.append(
                {
                    "task": task,
                    "metric": metric,
                    "best_val_supported": reason is None,
                    "reason": "" if reason is None else reason,
                }
            )
        return pd.DataFrame(rows).set_index("task")

    def selected_epochs(
        self,
        key: str | None = None,
        *,
        epoch_selection: str | None = None,
        strict: bool = True,
    ) -> pd.DataFrame:
        """The epoch each (task, model, seed) metric was taken from.

        Auditing surface for ``"best_val"``: a selected epoch pinned at the
        last epoch for every seed means selection did nothing, and a wildly
        varying one means the run had not converged.
        """
        selection = epoch_selection or self.epoch_selection
        rows = []
        for task, per_task in self.histories.items():
            task_selection = selection
            if selection != "final" and not strict:
                sample = next(iter(per_task.values()), None)
                if sample and _selection_reason(sample[0], key or _auto_metric_key(sample[0])) is not None:
                    task_selection = "final"
            for model, seed_histories in per_task.items():
                for seed, h in zip(self.seeds, seed_histories, strict=False):
                    k = key or _auto_metric_key(h)
                    value, epoch = _selected_value(h, k, task_selection)
                    rows.append(
                        {
                            "task": task,
                            "model": model,
                            "seed": seed,
                            "selection": task_selection,
                            "epoch": epoch,
                            "n_epochs": len(h[k]),
                            "value": value,
                        }
                    )
        return pd.DataFrame(rows)

    def metric_name(self) -> str:
        for per_task in self.histories.values():
            for seed_histories in per_task.values():
                if seed_histories:
                    return _auto_metric_key(seed_histories[0])
        return "metric"

    def _task_directions(self, key: str | None = None) -> dict[str, bool]:
        """``task -> lower is better`` for each task, from its own metric."""
        out: dict[str, bool] = {}
        for task, per_task in self.histories.items():
            sample = next(iter(per_task.values()), None)
            metric = key or (_auto_metric_key(sample[0]) if sample else "metric")
            out[task] = metric in _LOWER_IS_BETTER
        return out

    def summary(
        self,
        ci: float = 0.95,
        method: str | None = None,
        *,
        epoch_selection: str | None = None,
        strict: bool = True,
    ) -> pd.DataFrame:
        """Per-(task, model) mean, std, sem, CI half-width and bounds.

        ``method`` overrides ``self.ci_method`` for this call only; choose
        ``"t"`` for Student's-t intervals (default) or ``"bootstrap"`` for
        percentile-bootstrap intervals (better for non-Gaussian metrics
        such as Hits@K, MRR, or AUC).
        """
        rows = []
        finals = self.final_metrics(epoch_selection=epoch_selection, strict=strict)
        for task, per_task in finals.items():
            for model, values in per_task.items():
                arr = np.asarray(values, dtype=float)
                mean = float(arr.mean())
                std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
                sem = float(stats.sem(arr)) if arr.size > 1 else 0.0
                half = self._ci_half(arr, ci, method=method)
                rows.append(
                    {
                        "task": task,
                        "model": model,
                        "n_seeds": arr.size,
                        "mean": mean,
                        "std": std,
                        "sem": sem,
                        "ci_low": mean - half,
                        "ci_high": mean + half,
                    }
                )
        return pd.DataFrame(rows).set_index(["task", "model"]).sort_index()

    def pairwise(
        self,
        alpha: float = 0.05,
        method: str | None = None,
        *,
        epoch_selection: str | None = None,
        strict: bool = True,
    ) -> pd.DataFrame:
        """Paired pairwise tests between models per task with Holm adjustment.

        ``method`` overrides ``self.pairwise_method`` for this call only:

        - ``"t"`` (default) -- paired Student's t-test on per-seed final metrics.
        - ``"wilcoxon"`` -- non-parametric Wilcoxon signed-rank test on the
          paired differences. Recommended at small seed counts where the
          paired t-test's normality assumption is most fragile; see
          Benavoli et al., *JMLR* 17(5):1-36, 2016.

        ``epoch_selection`` overrides [`epoch_selection`][graphnetz.benchmark.BenchmarkReport.epoch_selection], so the same
        comparison can be run under a fixed-epoch and a checkpoint-selected
        protocol without retraining.
        """
        finals = self.final_metrics(epoch_selection=epoch_selection, strict=strict)
        test = method or self.pairwise_method
        rows = []
        for task, per_task in finals.items():
            models = sorted(per_task)
            pairs: list[tuple[str, str, float, float, float]] = []
            ps: list[float] = []
            for i, model_a in enumerate(models):
                for model_b in models[i + 1 :]:
                    a = np.asarray(per_task[model_a], dtype=float)
                    b = np.asarray(per_task[model_b], dtype=float)
                    p = _paired_pvalue(a, b, test)
                    d = a - b
                    sigma_d = float(d.std(ddof=1)) if d.size > 1 else float("nan")
                    pairs.append((model_a, model_b, float(d.mean()), sigma_d, p))
                    ps.append(p)
            adj = _holm_correction(np.asarray(ps, dtype=float))
            for (model_a, model_b, diff, sigma_d, p_raw), p_holm in zip(pairs, adj, strict=False):
                n = len(per_task[model_a])
                rows.append(
                    {
                        "task": task,
                        "model_a": model_a,
                        "model_b": model_b,
                        "mean_diff": diff,
                        # Paired effect size (Appendix A): invariant to the seed
                        # count, so it complements the p-value with a magnitude.
                        "sigma_d": sigma_d,
                        "cohens_dz": diff / sigma_d
                        if sigma_d and np.isfinite(sigma_d) and sigma_d > 0
                        else float("nan"),
                        "p_raw": p_raw,
                        "p_holm": p_holm,
                        "significant": (not np.isnan(p_holm)) and p_holm < alpha,
                        "n_seeds": n,
                    }
                )
        return pd.DataFrame(rows)

    # ----- Power, equivalence, and what the design could detect --------------

    def power(
        self,
        alpha: float = 0.05,
        target_power: float = 0.80,
        *,
        targets: tuple[float, ...] = (0.02, 0.01, 0.005),
        epoch_selection: str | None = None,
        strict: bool = True,
    ) -> pd.DataFrame:
        """Per-comparison minimum detectable effect and observed power.

        A non-significant pairwise row means one of two very different things:
        the models are close, or the design could not tell.  This separates
        them.  ``mde`` is the smallest true paired difference detectable at
        ``target_power``; ``detectable`` flags comparisons whose observed
        effect clears their own MDE; ``seeds_for_*`` says how many seeds a
        given effect size would need.

        See `graphnetz.benchmark._power` for the formulae.
        """
        pw = self.pairwise(alpha=alpha, epoch_selection=epoch_selection, strict=strict)
        if pw.empty:
            return pw
        rows = []
        for _, r in pw.iterrows():
            n = int(r["n_seeds"])
            sigma_d = float(r["sigma_d"])
            effect = float(r["mean_diff"])
            mde = minimum_detectable_effect(sigma_d, n, alpha, target_power)
            row = {
                "task": r["task"],
                "model_a": r["model_a"],
                "model_b": r["model_b"],
                "mean_diff": effect,
                "sigma_d": sigma_d,
                "n_seeds": n,
                "mde": mde,
                "observed_power": observed_power(effect, sigma_d, n, alpha),
                "detectable": bool(np.isfinite(mde) and abs(effect) >= mde),
            }
            for t in targets:
                row[f"seeds_for_{t:g}"] = seeds_for_effect(t, sigma_d, alpha, target_power)
            rows.append(row)
        return pd.DataFrame(rows)

    def equivalence(
        self,
        margin: float,
        alpha: float = 0.05,
        *,
        method: str | None = None,
        epoch_selection: str | None = None,
        strict: bool = True,
    ) -> pd.DataFrame:
        """Two one-sided tests for equivalence within ``margin``, plus a verdict.

        ``margin`` is the smallest effect size of interest, in the metric's own
        units (e.g. ``0.01`` accuracy points).  Combined with the difference
        test, every comparison lands in exactly one of ``different``,
        ``equivalent``, ``trivial`` or ``undetermined`` -- so "we found no
        difference" can be stated as a positive claim where the data support it
        and withheld where they do not.

        Both families are Holm-corrected over the comparisons within a task,
        matching [`pairwise`][graphnetz.benchmark.BenchmarkReport.pairwise].
        """
        finals = self.final_metrics(epoch_selection=epoch_selection, strict=strict)
        diff_df = self.pairwise(alpha=alpha, method=method, epoch_selection=epoch_selection, strict=strict)
        rows = []
        for task, per_task in finals.items():
            models = sorted(per_task)
            keys: list[tuple[str, str]] = []
            ps: list[float] = []
            for i, model_a in enumerate(models):
                for model_b in models[i + 1 :]:
                    a = np.asarray(per_task[model_a], dtype=float)
                    b = np.asarray(per_task[model_b], dtype=float)
                    ps.append(paired_tost(a, b, margin, alpha))
                    keys.append((model_a, model_b))
            adj = _holm_correction(np.asarray(ps, dtype=float))
            for (model_a, model_b), p_raw, p_holm in zip(keys, ps, adj, strict=False):
                match = diff_df[
                    (diff_df["task"] == task) & (diff_df["model_a"] == model_a) & (diff_df["model_b"] == model_b)
                ]
                p_diff = float(match["p_holm"].iloc[0]) if not match.empty else float("nan")
                mean_diff = float(match["mean_diff"].iloc[0]) if not match.empty else float("nan")
                rows.append(
                    {
                        "task": task,
                        "model_a": model_a,
                        "model_b": model_b,
                        "mean_diff": mean_diff,
                        "margin": margin,
                        "p_tost_raw": p_raw,
                        "p_tost_holm": p_holm,
                        "p_difference_holm": p_diff,
                        "verdict": equivalence_verdict(p_diff, p_holm, alpha),
                    }
                )
        return pd.DataFrame(rows)

    # ----- Across-task rank aggregation --------------------------------------

    def rank_table(
        self,
        *,
        epoch_selection: str | None = None,
        strict: bool = True,
    ) -> pd.DataFrame:
        """Per-task ranks of the seed means; rows tasks, columns models.

        Rank 1 is best, ties averaged, and the metric direction is applied per
        task, so a benchmark mixing accuracy with MAE ranks correctly.  Only
        models present in *every* task appear -- the same restriction the CD
        diagram applies, surfaced as data.
        """
        finals = self.final_metrics(epoch_selection=epoch_selection, strict=strict)
        if not finals:
            return pd.DataFrame()
        common: set[str] = set.intersection(*[set(per.keys()) for per in finals.values()])
        if len(common) < 2:
            return pd.DataFrame()
        models = sorted(common)
        tasks = sorted(finals)
        ranks = _rank_rows(finals, self._task_directions(), models, tasks)
        return pd.DataFrame(ranks, index=pd.Index(tasks, name="task"), columns=models)

    def mean_ranks(
        self,
        *,
        aggregation: str = "uniform",
        weights: Mapping[str, float] | None = None,
        groups: Mapping[str, str] | None = None,
        epoch_selection: str | None = None,
        strict: bool = True,
    ) -> pd.Series:
        r"""Mean rank per model across tasks, under a choice of task weighting.

        The Demšar procedure weights every task equally, which means a small
        synthetic task counts as much as a large public benchmark.  That is a
        modelling choice, not a law, so it is exposed:

        ``"uniform"``
            The Demšar default, $\bar r_i = \frac1N \sum_n r_{n,i}$.
            **This is the only variant whose null distribution the Friedman and
            Nemenyi procedures describe.**
        ``"reliability"``
            Weight each task by the inverse mean 95 % CI half-width of its
            cells, so tasks whose seed variance makes their ranking unreliable
            contribute less.
        ``"hierarchical"``
            Average within each group in ``groups`` (e.g. research category),
            then across groups, so a category with many tasks does not
            dominate one with few.
        ``"custom"``
            Use ``weights`` directly, keyed by task name.

        Non-uniform variants are **diagnostics**: they break the exchangeability
        the Friedman null assumes, so a weighted mean rank must not be compared
        against $CD_\alpha$.  Use [`rank_stability`][graphnetz.benchmark.BenchmarkReport.rank_stability] for uncertainty
        on them.
        """
        table = self.rank_table(epoch_selection=epoch_selection, strict=strict)
        if table.empty:
            return pd.Series(dtype=float)
        tasks = list(table.index)

        if aggregation == "uniform":
            w = _normalized_weights(None, tasks)
        elif aggregation == "custom":
            if weights is None:
                msg = "aggregation='custom' requires weights={task: weight}"
                raise ValueError(msg)
            w = _normalized_weights(weights, tasks)
        elif aggregation == "reliability":
            summary = self.summary(epoch_selection=epoch_selection, strict=strict)
            half = (summary["ci_high"] - summary["ci_low"]) / 2.0
            per_task = half.groupby(level="task").mean()
            inv = {t: 1.0 / max(float(per_task.get(t, np.nan)), 1e-12) for t in tasks}
            w = _normalized_weights(inv, tasks)
        elif aggregation == "hierarchical":
            if groups is None:
                msg = "aggregation='hierarchical' requires groups={task: group}"
                raise ValueError(msg)
            sizes: dict[str, int] = {}
            for t in tasks:
                sizes[groups.get(t, t)] = sizes.get(groups.get(t, t), 0) + 1
            # Each group contributes equally; tasks share their group's budget.
            w = _normalized_weights({t: 1.0 / sizes[groups.get(t, t)] for t in tasks}, tasks)
        else:
            msg = f"Unknown aggregation {aggregation!r}; choices: 'uniform', 'reliability', 'hierarchical', 'custom'"
            raise ValueError(msg)

        values = np.asarray(table.to_numpy(dtype=float))
        weighted = (values * w[:, None]).sum(axis=0) / w.sum()
        return pd.Series(weighted, index=table.columns).sort_values()

    def friedman(
        self,
        alpha: float = 0.05,
        *,
        epoch_selection: str | None = None,
        strict: bool = True,
    ) -> dict[str, float | int | bool]:
        r"""Friedman omnibus test on per-task ranks of seed-mean metrics.

        Returns the statistic ``chi2``, the asymptotic $\chi^2_{k-1}$ p-value,
        the rejection flag at ``alpha``, the $(k, N)$ shape, and the Nemenyi
        ``critical_difference`` implied by that shape.  Also returns
        ``n_for_observed_gap``: the number of tasks at which $CD_\alpha$ would
        shrink below the largest observed mean-rank gap -- i.e. how much
        benchmark breadth the current ordering would need before it could be
        called significant.  Since $CD_\alpha \propto 1/\sqrt N$ this is a
        design quantity, computed from the observed gap and independent of any
        assumption about how new tasks would rank.

        The Nemenyi post-hoc surfaced in [`plot_critical_difference`][graphnetz.benchmark.BenchmarkReport.plot_critical_difference]
        should only be interpreted when ``rejected`` is true (Demšar, 2006).
        Only models present in every task are included.
        """
        empty = {
            "chi2": float("nan"),
            "p_value": float("nan"),
            "k": 0,
            "n": 0,
            "rejected": False,
            "critical_difference": float("nan"),
            "max_rank_gap": float("nan"),
            "n_for_observed_gap": 0,
        }
        table = self.rank_table(epoch_selection=epoch_selection, strict=strict)
        if table.empty or table.shape[0] < 2 or table.shape[1] < 2:
            finals = self.final_metrics(epoch_selection=epoch_selection, strict=strict)
            common = set.intersection(*[set(p) for p in finals.values()]) if finals else set()
            return {**empty, "k": len(common), "n": len(finals)}

        ranks = table.to_numpy(dtype=float)
        n, k = ranks.shape
        chi2 = _friedman_chi2(ranks)
        p = float(stats.chi2.sf(chi2, df=k - 1))
        cd = _nemenyi_cd(k, n, alpha)
        avg = ranks.mean(axis=0)
        gap = float(avg.max() - avg.min())
        # CD(N) = q sqrt(k(k+1)/6N) < gap  <=>  N > q^2 k(k+1)/(6 gap^2),
        # so scaling the current CD gives the N at which the gap would resolve.
        n_needed = int(np.ceil((cd**2) * n / (gap**2))) if gap > 0 and np.isfinite(cd) else 0
        return {
            "chi2": float(chi2),
            "p_value": p,
            "k": k,
            "n": n,
            "rejected": bool(p < alpha),
            "critical_difference": float(cd),
            "max_rank_gap": gap,
            "n_for_observed_gap": n_needed,
        }

    def rank_stability(
        self,
        *,
        n_boot: int = 10_000,
        alpha: float = 0.05,
        seed: int = 0,
        epoch_selection: str | None = None,
        strict: bool = True,
    ) -> dict[str, Any]:
        """How much the across-task ranking depends on *which* tasks were run.

        A CD diagram at small ``N`` says nothing about whether the ordering
        would survive a different draw of benchmark tasks.  Two answers:

        *Bootstrap over tasks.*  Resample the ``N`` tasks with replacement
        ``n_boot`` times, recompute mean ranks, and report the distribution of
        the largest mean-rank gap, the fraction of resamples reproducing the
        observed rank *order*, and for each pair the fraction of resamples in
        which the pair separates by more than that resample's own $CD_\\alpha$.

        *Leave-one-task-out jackknife.*  Drop each task in turn and record how
        far every mean rank moves -- exposing single tasks that carry the
        ordering.

        Returns ``mean_ranks``, ``order``, ``gap_observed``, ``gap_quantiles``,
        ``order_stability``, ``separation`` and ``jackknife``.
        """
        table = self.rank_table(epoch_selection=epoch_selection, strict=strict)
        if table.empty or table.shape[0] < 2:
            return {"mean_ranks": pd.Series(dtype=float), "order_stability": float("nan")}
        ranks = table.to_numpy(dtype=float)
        models = list(table.columns)
        tasks = list(table.index)
        n, k = ranks.shape

        observed = ranks.mean(axis=0)
        order = [models[i] for i in np.argsort(observed)]
        gap_observed = float(observed.max() - observed.min())

        rng = np.random.default_rng(seed)
        idx = rng.integers(0, n, size=(n_boot, n))
        boot = ranks[idx].mean(axis=1)  # [n_boot, k]
        gaps = boot.max(axis=1) - boot.min(axis=1)
        boot_order = np.argsort(boot, axis=1)
        target = np.argsort(observed)
        order_stability = float(np.mean(np.all(boot_order == target[None, :], axis=1)))

        cd_boot = _nemenyi_cd(k, n, alpha)  # same (k, N) in every resample
        separation = []
        for i in range(k):
            for j in range(i + 1, k):
                diff = np.abs(boot[:, i] - boot[:, j])
                separation.append(
                    {
                        "model_a": models[i],
                        "model_b": models[j],
                        "observed_gap": float(abs(observed[i] - observed[j])),
                        "separates_frac": float(np.mean(diff > cd_boot)),
                    }
                )

        jack = []
        for t in range(n):
            kept = np.delete(ranks, t, axis=0).mean(axis=0)
            for m, before, after in zip(models, observed, kept, strict=False):
                jack.append(
                    {
                        "dropped_task": tasks[t],
                        "model": m,
                        "mean_rank": float(after),
                        "shift": float(after - before),
                    }
                )
        return {
            "mean_ranks": pd.Series(observed, index=models).sort_values(),
            "order": order,
            "gap_observed": gap_observed,
            "critical_difference": float(cd_boot),
            "gap_quantiles": {q: float(np.quantile(gaps, q)) for q in (0.025, 0.25, 0.5, 0.75, 0.975)},
            "gap_exceeds_cd_frac": float(np.mean(gaps > cd_boot)),
            "order_stability": order_stability,
            "separation": pd.DataFrame(separation),
            "jackknife": pd.DataFrame(jack),
            "n_boot": n_boot,
        }

    # ----- Serialisation -----------------------------------------------------

    SCHEMA: int = 1
    """Version of the JSON payload written by [`to_json`][graphnetz.benchmark.BenchmarkReport.to_json]."""

    def to_json(self, path: str | Path | None = None) -> str:
        """Serialise the report to JSON: seeds, config, and the raw histories.

        The payload is the metric tensor ``X[task, model, seed]`` *before* any
        statistic is applied, so a published bundle can be re-analysed under a
        different CI method, pairwise test, epoch selection or aggregation
        without retraining.  JSON rather than pickle so the artefact stays
        readable, diffable, and independent of the library version that wrote
        it.  Returns the JSON text and writes it to ``path`` when given.
        """
        import json

        payload = {
            "schema": self.SCHEMA,
            "seeds": list(self.seeds),
            "config": {k: v for k, v in self.config.items() if isinstance(v, (str, int, float, bool, type(None)))},
            "options": {
                "ci_method": self.ci_method,
                "bootstrap_n": self.bootstrap_n,
                "bootstrap_seed": self.bootstrap_seed,
                "pairwise_method": self.pairwise_method,
                "epoch_selection": self.epoch_selection,
            },
            "histories": self.histories,
        }
        text = json.dumps(payload, indent=1) + "\n"
        if path is not None:
            out = Path(path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text)
        return text

    @classmethod
    def from_json(cls, source: str | Path) -> BenchmarkReport:
        """Rebuild a report from [`to_json`][graphnetz.benchmark.BenchmarkReport.to_json] output (a path or the text itself)."""
        import json

        text = Path(source).read_text() if Path(str(source)).exists() else str(source)
        payload = json.loads(text)
        schema = payload.get("schema")
        if schema != cls.SCHEMA:
            msg = f"unsupported report schema {schema!r}; this version reads {cls.SCHEMA}"
            raise ValueError(msg)
        options = payload.get("options", {})
        return cls(
            seeds=tuple(payload["seeds"]),
            histories=payload["histories"],
            config=payload.get("config", {}),
            ci_method=options.get("ci_method", "t"),
            bootstrap_n=options.get("bootstrap_n", 10000),
            bootstrap_seed=options.get("bootstrap_seed", 0),
            pairwise_method=options.get("pairwise_method", "t"),
            epoch_selection=options.get("epoch_selection", "final"),
        )

    # ----- Reporting helpers -------------------------------------------------

    def _best_per_task(self) -> dict[str, str]:
        finals = self.final_metrics()
        metric = self.metric_name()
        lower_is_better = metric in _LOWER_IS_BETTER
        best: dict[str, str] = {}
        for task, per_task in finals.items():
            scored = [(model, float(np.mean(values))) for model, values in per_task.items()]
            if lower_is_better:
                best[task] = min(scored, key=lambda x: x[1])[0]
            else:
                best[task] = max(scored, key=lambda x: x[1])[0]
        return best

    def to_latex(
        self,
        path: str | Path,
        *,
        ci: float = 0.95,
        bold_best: bool = True,
        pretty_tasks: Mapping[str, str] | None = None,
        caption: str | None = None,
        label: str | None = None,
        method: str | None = None,
    ) -> Path:
        """Booktabs LaTeX table of mean ± CI half-width with bold-best per task.

        ``method`` overrides ``self.ci_method`` (``"t"`` or ``"bootstrap"``).
        """
        finals = self.final_metrics()
        tasks = sorted(finals)
        models = sorted({m for per in finals.values() for m in per})
        best = self._best_per_task() if bold_best else {}
        pretty = dict(pretty_tasks or {})

        lines: list[str] = []
        if caption is not None or label is not None:
            lines.extend([r"\begin{table}[t]", r"  \centering"])
            if caption is not None:
                lines.append(rf"  \caption{{{caption}}}")
            if label is not None:
                lines.append(rf"  \label{{{label}}}")
        lines.append(r"\begin{tabular}{l" + "c" * len(tasks) + "}")
        lines.append(r"\toprule")
        header = "Model & " + " & ".join(pretty.get(t, t) for t in tasks) + r" \\"
        lines.append(header)
        lines.append(r"\midrule")
        for model in models:
            cells = []
            for task in tasks:
                values = np.asarray(finals[task].get(model, []), dtype=float)
                if values.size == 0:
                    cells.append("--")
                    continue
                mean = float(values.mean())
                half = self._ci_half(values, ci, method=method)
                if bold_best and best.get(task) == model:
                    cell = rf"$\mathbf{{{mean:.3f} \pm {half:.3f}}}$"
                else:
                    cell = rf"${mean:.3f} \pm {half:.3f}$"
                cells.append(cell)
            lines.append(f"{model} & " + " & ".join(cells) + r" \\")
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        if caption is not None or label is not None:
            lines.append(r"\end{table}")
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(lines) + "\n")
        return out

    def pairwise_to_latex(
        self,
        path: str | Path,
        *,
        alpha: float = 0.05,
        caption: str | None = None,
        label: str | None = None,
        method: str | None = None,
    ) -> Path:
        """LaTeX booktabs table of pairwise Holm-adjusted p-values.

        ``method`` overrides ``self.pairwise_method`` (``"t"`` or
        ``"wilcoxon"``) for this call only.
        """
        df = self.pairwise(alpha=alpha, method=method)
        lines: list[str] = []
        if caption is not None or label is not None:
            lines.extend([r"\begin{table}[t]", r"  \centering"])
            if caption is not None:
                lines.append(rf"  \caption{{{caption}}}")
            if label is not None:
                lines.append(rf"  \label{{{label}}}")
        lines.append(r"\begin{tabular}{llcccl}")
        lines.append(r"\toprule")
        lines.append(r"Task & Comparison & $\Delta\mu$ & $p_{\text{raw}}$ & $p_{\text{Holm}}$ & Sig. \\")
        lines.append(r"\midrule")
        for _, row in df.iterrows():
            sig = r"\textbf{*}" if row["significant"] else ""
            p_raw = "n/a" if pd.isna(row["p_raw"]) else f"{row['p_raw']:.3g}"
            p_holm = "n/a" if pd.isna(row["p_holm"]) else f"{row['p_holm']:.3g}"
            lines.append(
                f"{row['task']} & {row['model_a']} vs.\\ {row['model_b']} & "
                f"${row['mean_diff']:+.3f}$ & {p_raw} & {p_holm} & {sig} \\\\"
            )
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        if caption is not None or label is not None:
            lines.append(r"\end{table}")
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(lines) + "\n")
        return out
