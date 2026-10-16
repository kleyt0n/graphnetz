"""Positive control: does the reporting layer detect an effect that is there?

Every result in Section 3 is a non-rejection, which leaves the framework's
central claim only half-tested.  A layer that never rejects anything would
produce exactly the same output, so the null result on its own is not evidence
that the layer works -- it is evidence that the layer is *conservative*, at
best.  This stage supplies the missing half in two independent ways.

**(A) Planted-effect benchmark, trained end to end.**  Twelve synthetic
node-classification tasks in which the label is a stochastic-block-model
community and the node features are pure noise, so *all* of the signal is
structural and by construction only a model that propagates over edges can
recover it.  Three models with a designed quality ordering are run through the
ordinary ``run_benchmark`` path:

``GCN-2``   two-hop message passing -- should recover the communities;
``GCN-1``   one-hop -- less propagation, so strictly weaker;
``MLP-0``   an identical-width MLP that never reads ``edge_index`` -- has no
            access to the signal at all and should sit at chance.

Nothing about the statistical layer is special-cased: the tasks go through the
same dispatcher, trainer and ``BenchmarkReport`` as Cora does.  The design
question the stage answers is whether the same report that returned a single
Nemenyi clique in Section 3.4 returns *separated* cliques here.

**(B) Monte-Carlo calibration of the estimators.**  The planted-effect sweep
shows the layer rejects one true effect; it does not show the rejection rate is
the advertised one.  So the stage also simulates the paired scaffolding
directly -- 20,000 replicates at a known effect size -- and checks the three
rates the paper actually claims: the type-I error under the null is at most
alpha, the rejection rate at an effect exactly equal to the reported MDE is the
target power, and the Friedman-Nemenyi step first resolves a planted rank gap
at the N that Eq. (10) predicts in closed form.

    tables/positive_control.tex   -- the planted-effect sweep (A)
    tables/calibration.tex        -- the Monte-Carlo rates (B)
    _artifacts/positive_control.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from paper.experiments import _runs
from paper.experiments._config import ALPHA, CI, SEEDS
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

from graphnetz.benchmark import Task
from graphnetz.benchmark._power import minimum_detectable_effect
from graphnetz.benchmark._stats import _holm_correction, _nemenyi_cd

CATEGORY = "positive_control"
EPOCHS = 100
MODEL_NAMES: tuple[str, ...] = ("GCN-2", "GCN-1", "MLP-0")

# Twelve SBM slots.  Communities are the label, features are noise, so the
# planted ordering is a property of the generator and not of any tuning: the
# two-hop model can average over a wider neighbourhood than the one-hop model,
# and the MLP cannot see the graph at all.  Difficulty is varied through the
# intra/inter edge-probability ratio and the graph size so the twelve tasks are
# not twelve copies of one task.
SLOTS: tuple[tuple[str, int, int, float, float], ...] = (
    #  name          nodes  classes  p_in    p_out
    ("sbm_a", 600, 3, 0.060, 0.004),
    ("sbm_b", 600, 3, 0.045, 0.006),
    ("sbm_c", 600, 4, 0.055, 0.005),
    ("sbm_d", 800, 3, 0.040, 0.004),
    ("sbm_e", 800, 4, 0.035, 0.004),
    ("sbm_f", 800, 2, 0.030, 0.006),
    ("sbm_g", 1000, 3, 0.030, 0.003),
    ("sbm_h", 1000, 4, 0.028, 0.003),
    ("sbm_i", 1000, 5, 0.032, 0.003),
    ("sbm_j", 1200, 3, 0.024, 0.003),
    ("sbm_k", 1200, 4, 0.022, 0.002),
    ("sbm_l", 1200, 5, 0.026, 0.002),
)


# --------------------------------------------------------------------------- #
# (A) the planted-effect tasks
# --------------------------------------------------------------------------- #


class _SBM:
    """Dataset-shaped wrapper over one seeded SBM draw.

    Satisfies the loader contract the dispatcher expects for ``node_cls``
    (``ds[0]``, ``num_features``, ``num_classes``) without depending on
    ``InMemoryDataset``, so no file is written to disk.
    """

    def __init__(self, data: Data, num_features: int, num_classes: int) -> None:
        self._data, self.num_features, self.num_classes = data, num_features, num_classes

    def __getitem__(self, _i: int) -> Data:
        return self._data

    def __len__(self) -> int:
        return 1


def _sbm_loader(n: int, k: int, p_in: float, p_out: float, n_feat: int = 16):
    """Build a seed-aware loader for one slot.

    The ``seed`` keyword makes the dispatcher redraw the graph per seed
    (``_runner`` threads it through for seed-aware loaders), so the reported
    cross-seed variance covers data resampling as well as initialisation --
    the harder and more honest of the two settings for a positive control.
    """

    def load(_root: str, *, seed: int = 0) -> _SBM:
        g = torch.Generator().manual_seed(10_000 + seed)
        y = torch.randint(0, k, (n,), generator=g)
        # Undirected Bernoulli SBM, upper triangle only, then symmetrised.
        iu, ju = torch.triu_indices(n, n, offset=1)
        same = y[iu] == y[ju]
        prob = torch.where(same, torch.tensor(p_in), torch.tensor(p_out))
        keep = torch.rand(iu.numel(), generator=g) < prob
        src, dst = iu[keep], ju[keep]
        edge_index = torch.cat([torch.stack([src, dst]), torch.stack([dst, src])], dim=1)
        # Features carry no label information whatsoever.  This is what makes
        # the ordering planted rather than hoped for.
        x = torch.randn(n, n_feat, generator=g)
        perm = torch.randperm(n, generator=g)
        n_tr, n_va = int(0.2 * n), int(0.2 * n)
        masks = {}
        for name, idx in (
            ("train_mask", perm[:n_tr]),
            ("val_mask", perm[n_tr : n_tr + n_va]),
            ("test_mask", perm[n_tr + n_va :]),
        ):
            m = torch.zeros(n, dtype=torch.bool)
            m[idx] = True
            masks[name] = m
        return _SBM(Data(x=x, edge_index=edge_index, y=y, **masks), n_feat, k)

    return load


def _tasks() -> list[Task]:
    return [
        Task(name=name, task_type="node_cls", loader=_sbm_loader(n, k, p_in, p_out), epochs=EPOCHS)
        for name, n, k, p_in, p_out in SLOTS
    ]


class GCN2(nn.Module):
    """Two-hop propagation: sees the two-step neighbourhood."""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int) -> None:
        super().__init__()
        self.c1 = GCNConv(in_channels, hidden_channels)
        self.c2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data: Data) -> torch.Tensor:
        x, ei = data.x, data.edge_index
        return self.c2(torch.relu(self.c1(x, ei)), ei)


class GCN1(nn.Module):
    """One-hop propagation, same parameter budget as ``GCN2``."""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int) -> None:
        super().__init__()
        self.c1 = GCNConv(in_channels, hidden_channels)
        self.head = nn.Linear(hidden_channels, out_channels)

    def forward(self, data: Data) -> torch.Tensor:
        return self.head(torch.relu(self.c1(data.x, data.edge_index)))


class MLP0(nn.Module):
    """Identical width, never reads ``edge_index``: no access to the signal."""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, data: Data) -> torch.Tensor:
        return self.net(data.x)


MODELS: dict[str, Any] = {
    "GCN-2": (GCN2, "node_cls"),
    "GCN-1": (GCN1, "node_cls"),
    "MLP-0": (MLP0, "node_cls"),
}


def train(
    *,
    seeds: tuple[int, ...] = SEEDS,
    data_root: str,
    cache_dir: Path,
    epochs: int | None = None,
    force: bool = False,
    verbose: bool = True,
    cache_only: bool = False,
    chunk: int = 2,
    max_seconds: float | None = None,
) -> Any:
    """Train the planted-effect sweep (or load it from cache)."""
    return _runs.train_cell(
        label=f"positive_control/{len(SLOTS)} SBM tasks",
        tasks=_tasks(),
        category=CATEGORY,
        model_names=MODEL_NAMES,
        seeds=seeds,
        data_root=data_root,
        cache_path=cache_dir / "positive_control.json",
        epochs=epochs,
        force=force,
        verbose=verbose,
        cache_only=cache_only,
        chunk=chunk,
        max_seconds=max_seconds,
        models=MODELS,
    )


# --------------------------------------------------------------------------- #
# (B) Monte-Carlo calibration of the estimators
# --------------------------------------------------------------------------- #


def _calibrate(n_rep: int = 20_000, n_seeds: int = 10, alpha: float = ALPHA) -> dict[str, Any]:
    """Empirical type-I error, power at the reported MDE, and Nemenyi breadth.

    All three are rates the manuscript asserts, so all three are measured
    rather than assumed.  The simulation is of the paired differences directly,
    which is the scaffolding every within-task test in the report sits on.

    Every rate is returned with its Monte-Carlo standard error, because a
    simulated rate quoted without one invites exactly the over-reading of a
    point estimate this paper is about: at ``n_rep`` replicates a rate near
    ``alpha`` carries an SE of roughly 0.0015, so agreement should be judged
    against that and not against the fourth decimal.
    """
    from scipy import stats

    rng = np.random.default_rng(0)
    sigma = 1.0
    crit = stats.t.ppf(1 - alpha / 2, n_seeds - 1)

    def _se(rate: float) -> float:
        return float(np.sqrt(rate * (1.0 - rate) / n_rep))

    def _reject_rate(delta: float) -> tuple[float, float]:
        """Rejection rate of both pairwise tests at a true effect of ``delta``."""
        d = rng.normal(delta, sigma, size=(n_rep, n_seeds))
        t = d.mean(axis=1) / (d.std(axis=1, ddof=1) / np.sqrt(n_seeds))
        para = float(np.mean(np.abs(t) > crit))
        # The exact Wilcoxon null is what the report uses at S = 10, so it is
        # evaluated per replicate rather than through a normal approximation.
        w = np.array([stats.wilcoxon(row, zero_method="wilcox").pvalue for row in d])
        return para, float(np.mean(w < alpha))

    # Type-I error under the exact null.
    t1_t, t1_w = _reject_rate(0.0)

    # Holm family-wise error over the six comparisons within a four-model task,
    # all six null.  Holm should hold this at or below alpha; the uncorrected
    # family is reported beside it to quantify what the correction buys.
    fam = rng.normal(0.0, sigma, size=(n_rep, 6, n_seeds))
    tf = fam.mean(axis=2) / (fam.std(axis=2, ddof=1) / np.sqrt(n_seeds))
    pf = 2 * stats.t.sf(np.abs(tf), n_seeds - 1)
    holm_any = float(np.mean([np.nanmin(_holm_correction(row)) < alpha for row in pf]))
    raw_any = float(np.mean(pf.min(axis=1) < alpha))

    # The same family as it arises in a real four-model table: four shared
    # per-seed score vectors and the six paired differences formed from them,
    # which are correlated (rho = +/-0.5 for pairs sharing a model).  Drawn
    # from its own generator so the rows above and below keep their stream.
    shared_rng = np.random.default_rng(1)
    scores = shared_rng.normal(0.0, sigma, size=(n_rep, 4, n_seeds))
    pairs = [(i, j) for i in range(4) for j in range(i + 1, 4)]
    diffs = np.stack([scores[:, i] - scores[:, j] for i, j in pairs], axis=1)
    ts = diffs.mean(axis=2) / (diffs.std(axis=2, ddof=1) / np.sqrt(n_seeds))
    ps = 2 * stats.t.sf(np.abs(ts), n_seeds - 1)
    holm_shared = float(np.mean([np.nanmin(_holm_correction(row)) < alpha for row in ps]))
    raw_shared = float(np.mean(ps.min(axis=1) < alpha))

    # Power at an effect exactly equal to the design's own reported MDE.
    mde = minimum_detectable_effect(sigma, n_seeds, alpha=alpha, power=0.80)
    pw_t, pw_w = _reject_rate(mde)

    # Eq. (10) in closed form against an independent scan of CD(N): plant a
    # mean-rank gap of 1.0 between the best and worst of three models and ask
    # each route for the N at which Nemenyi first resolves it.  The closed form
    # recovers q_alpha from CD at a reference N rather than re-tabulating it,
    # so the two routes share no arithmetic beyond _nemenyi_cd itself.
    k, gap, n_ref = 3, 1.0, 10
    q_alpha = _nemenyi_cd(k, n_ref, alpha) / np.sqrt(k * (k + 1) / (6 * n_ref))
    n_closed = int(np.ceil((q_alpha**2) * k * (k + 1) / (6 * gap**2)))
    n_scan = next(n for n in range(3, 500) if _nemenyi_cd(k, n, alpha) < gap)
    return {
        "n_rep": n_rep,
        "n_seeds": n_seeds,
        "alpha": alpha,
        "type1_t": t1_t,
        "type1_t_se": _se(t1_t),
        "type1_wilcoxon": t1_w,
        "type1_wilcoxon_se": _se(t1_w),
        "holm_fwer": holm_any,
        "holm_fwer_se": _se(holm_any),
        "uncorrected_fwer": raw_any,
        "uncorrected_fwer_se": _se(raw_any),
        "holm_fwer_shared": holm_shared,
        "holm_fwer_shared_se": _se(holm_shared),
        "uncorrected_fwer_shared": raw_shared,
        "uncorrected_fwer_shared_se": _se(raw_shared),
        "mde_at_sigma1": mde,
        "power_at_mde_t": pw_t,
        "power_at_mde_t_se": _se(pw_t),
        "power_at_mde_wilcoxon": pw_w,
        "power_at_mde_wilcoxon_se": _se(pw_w),
        "nemenyi_k": k,
        "nemenyi_gap": gap,
        "nemenyi_n_closed_form": n_closed,
        "nemenyi_n_scan": n_scan,
        "nemenyi_cd_at_n": _nemenyi_cd(k, n_scan, alpha),
        "nemenyi_cd_one_below": _nemenyi_cd(k, n_scan - 1, alpha),
    }


# --------------------------------------------------------------------------- #
# emit
# --------------------------------------------------------------------------- #


def _write(path: Path, lines: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


def emit(report: Any, *, tab_dir: Path, artifact_dir: Path) -> dict[str, Any]:
    """Emit the planted-effect table, the calibration table, and the JSON."""
    summary = report.summary(ci=CI)
    pw = report.pairwise(alpha=ALPHA)
    fr = report.friedman(alpha=ALPHA)
    ranks = report.mean_ranks()
    power = report.power(alpha=ALPHA)

    # --- Table: the planted-effect sweep ---------------------------------- #
    lines = [
        r"\begin{tabular}{l ccc c}",
        r"\toprule",
        r"Task & GCN-2 & GCN-1 & MLP-0 & Holm-significant pairs \\",
        r"\midrule",
    ]
    for task in [name for name, *_ in SLOTS]:
        cells = []
        best = max(
            (float(summary.loc[(task, m), "mean"]) for m in MODEL_NAMES if (task, m) in summary.index),
            default=float("nan"),
        )
        for m in MODEL_NAMES:
            if (task, m) not in summary.index:
                cells.append(r"{\color{nac}\textit{n/a}}")
                continue
            row = summary.loc[(task, m)]
            mean = float(row["mean"])
            half = (float(row["ci_high"]) - float(row["ci_low"])) / 2.0
            body = f"${mean:.3f}_{{\\pm {half:.3f}}}$"
            cells.append(rf"\cellcolor{{winnerc}}{{\color{{paperink}}\textbf{{{body}}}}}" if mean == best else body)
        sig = pw[(pw["task"] == task) & pw["significant"]]
        name = task.replace("_", r"\_")
        lines.append(f"{name} & " + " & ".join(cells) + f" & {len(sig)} of 3" + r" \\")
    lines += [
        r"\midrule",
        r"\textbf{Mean rank} & "
        + " & ".join(rf"\textbf{{{float(ranks.loc[m]):.2f}}}" for m in MODEL_NAMES)
        + r" & \textbf{"
        + f"{int(pw['significant'].sum())} of {len(pw)}"
        + r"} \\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    _write(tab_dir / "positive_control.tex", lines)

    # --- Table: Monte-Carlo calibration ----------------------------------- #
    cal = _calibrate()
    a = cal["alpha"]

    def _pm(key: str) -> str:
        return f"${cal[key]:.4f} \\pm {cal[key + '_se']:.4f}$"

    rows = [
        (r"Type-I error, paired $t$ ($\delta = 0$)", f"${a:.3f}$", _pm("type1_t")),
        (r"Type-I error, exact Wilcoxon ($\delta = 0$)", f"${a:.3f}$", _pm("type1_wilcoxon")),
        (r"Family-wise error, 6 null pairs of 4 models, \textbf{Holm}", rf"$\leq {a:.3f}$", _pm("holm_fwer_shared")),
        (r"Family-wise error, 6 null pairs of 4 models, \emph{uncorrected}", r"---", _pm("uncorrected_fwer_shared")),
        (r"Family-wise error, 6 independent null pairs, \textbf{Holm}", rf"$\leq {a:.3f}$", _pm("holm_fwer")),
        (
            r"Family-wise error, 6 independent null pairs, \emph{uncorrected}",
            rf"${1 - (1 - a) ** 6:.3f}$",
            _pm("uncorrected_fwer"),
        ),
        (
            r"Power at $\delta = \mathrm{MDE}_{0.05,\,0.80}$, paired $t$",
            r"$0.800$",
            _pm("power_at_mde_t"),
        ),
        (
            r"Power at $\delta = \mathrm{MDE}_{0.05,\,0.80}$, exact Wilcoxon",
            r"$\leq 0.800$",
            _pm("power_at_mde_wilcoxon"),
        ),
        (
            rf"Tasks $N$ resolving a planted rank gap of ${cal['nemenyi_gap']:.1f}$ "
            rf"($k = {cal['nemenyi_k']}$)",
            f"${cal['nemenyi_n_closed_form']}$",
            f"${cal['nemenyi_n_scan']}$",
        ),
    ]
    cal_lines = [
        r"\begin{tabular}{l cc}",
        r"\toprule",
        r"Quantity & Nominal / closed form & Empirical \\",
        r"\midrule",
    ]
    cal_lines += [f"{label} & {nominal} & {observed}" + r" \\" for label, nominal, observed in rows]
    cal_lines += [r"\bottomrule", r"\end{tabular}"]
    _write(tab_dir / "calibration.tex", cal_lines)

    out = {
        "friedman": fr,
        "mean_ranks": {m: float(ranks.loc[m]) for m in ranks.index},
        "n_significant": int(pw["significant"].sum()),
        "n_comparisons": len(pw),
        "median_observed_power": float(np.nanmedian(power["observed_power"]))
        if "observed_power" in power
        else float("nan"),
        "calibration": cal,
        "summary": {f"{t}/{m}": float(r["mean"]) for (t, m), r in summary.iterrows()},
    }
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "positive_control.json").write_text(json.dumps(out, indent=2, default=str))
    print(
        f"  positive control: Friedman chi2={fr['chi2']:.2f} (p={fr['p_value']:.2g}), "
        f"CD={fr['critical_difference']:.3f}, gap={fr['max_rank_gap']:.2f}, "
        f"{out['n_significant']}/{out['n_comparisons']} pairs significant"
    )
    return out
