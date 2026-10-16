#!/usr/bin/env python3
"""Reproduce every figure and table in ``paper/main.tex``.

    python paper/experiments/run.py all              # everything (long)
    python paper/experiments/run.py catalog          # Table 4 only, seconds
    python paper/experiments/run.py planetoid        # Section 3.5 + demo figures
    python paper/experiments/run.py breadth          # Section 3, Figs 2-3, Table 3
    python paper/experiments/run.py analysis         # revision analyses, cache only
    python paper/experiments/run.py tuning           # bounded hyperparameter-search study
    python paper/experiments/run.py positive_control # planted-effect control + calibration
    python paper/experiments/run.py usability        # measured cost vs raw PyG
    python paper/experiments/run.py loaders          # audit all 62 loaders (slow)
    python paper/experiments/run.py verify           # compare cache vs the paper
    python paper/experiments/run.py all --smoke      # 2 seeds/2 epochs, throwaway

``analysis`` retrains nothing: it re-reads the cached metric tensor and emits
the power, equivalence, epoch-selection, aggregation and rank-stability
artefacts added in revision.

Runs are cached per benchmark cell under ``paper/experiments/_artifacts/cache``
as JSON, so re-emitting the figures after a plotting tweak costs nothing.
``--force`` retrains.  See ``paper/experiments/README.md``.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
# Importable both as `python paper/experiments/run.py` and as
# `python -m paper.experiments.run` from the repository root.
for candidate in (REPO, REPO / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from paper.experiments import (  # noqa: E402
    analysis,
    breadth,
    catalog,
    loaders,
    planetoid,
    positive_control,
    tuning,
    usability,
    verify,
)
from paper.experiments._config import ALPHA, BREADTH_SLOTS, SEEDS  # noqa: E402
from paper.experiments._style import install_paper_style  # noqa: E402

STAGES = (
    "catalog",
    "usability",
    "planetoid",
    "breadth",
    "analysis",
    "positive_control",
    "tuning",
    "loaders",
    "verify",
)
DEFAULT_STAGES = (
    "catalog",
    "usability",
    "planetoid",
    "breadth",
    "analysis",
    "positive_control",
    "verify",
)
"""``all`` runs everything except ``loaders``, which downloads the whole
catalogue (~10 GB) and is opt-in."""


def _parse(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "stages",
        nargs="*",
        default=["all"],
        help=f"one or more of {', '.join(STAGES)}, or 'all' (default)",
    )
    p.add_argument("--seeds", type=int, default=len(SEEDS), help="number of seeds, 0..N-1 (paper: 10)")
    p.add_argument("--epochs", type=int, default=None, help="override every per-task epoch budget")
    p.add_argument("--data-root", default=str(REPO / "data" / "benchmark"), help="dataset download root")
    p.add_argument("--out-dir", default=None, help="write figures/ and tables/ here instead of paper/")
    p.add_argument("--only", nargs="+", default=None, help="restrict the breadth sweep to these categories")
    p.add_argument("--force", action="store_true", help="retrain instead of using the cache")
    p.add_argument("--pdf", action="store_true", help="also write vector PDFs beside the PNGs")
    p.add_argument("--quiet", action="store_true", help="hide per-run training progress bars")
    p.add_argument("--device", default="cpu", help="training device: cpu (paper), cuda, mps or auto")
    p.add_argument("--show-ok", action="store_true", help="verify: print passing checks too")
    p.add_argument("--smoke", action="store_true", help="2 seeds, 2 epochs, throwaway output dir")
    p.add_argument("--n-boot", type=int, default=10_000, help="analysis: task-resampling replicates")
    p.add_argument("--chunk", type=int, default=2, help="seeds to train between cache writes (resume granularity)")
    p.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="stop cleanly after the current chunk once this many seconds have elapsed in a cell; "
        "rerun the same command to continue. Lets a multi-hour slot be trained in short sittings.",
    )
    p.add_argument(
        "--allow-partial",
        action="store_true",
        help="analysis: run the across-task analyses on the cached slot subset. For "
        "developing the manuscript before a sweep completes; the resulting N is not "
        "the paper's, and the run says so.",
    )
    p.add_argument(
        "--only-cached",
        action="store_true",
        help="loaders: audit only datasets already on disk (no downloads)",
    )
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse(argv)
    from paper.experiments import _config

    _config.DEVICE = args.device
    stages = list(DEFAULT_STAGES) if "all" in args.stages else list(args.stages)
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        print(f"unknown stage(s): {unknown}; choices: {list(STAGES)}", file=sys.stderr)
        return 2

    if args.smoke:
        args.seeds, args.epochs = min(args.seeds, 2), args.epochs or 2
        args.out_dir = args.out_dir or str(HERE / "_artifacts" / "smoke")

    out_root = Path(args.out_dir) if args.out_dir else REPO / "paper"
    fig_dir, tab_dir = out_root / "figures", out_root / "tables"
    # Smoke output is quarantined so a throwaway run can never overwrite the
    # cache, the CSVs, or the figures/tables the paper actually builds from.
    artifact_dir = HERE / "_artifacts" / "smoke" if args.smoke else HERE / "_artifacts"
    cache_dir = HERE / "_artifacts" / ("cache_smoke" if args.smoke else "cache")
    formats = ("png", "pdf") if args.pdf else ("png",)
    seeds = tuple(range(args.seeds))
    # `--only` narrows what gets *trained*; Figure 2/3 and Table 3 always
    # cover all ten categories or are not written at all, so a partial run can
    # never leave a truncated Table 3 behind.
    train_slots = BREADTH_SLOTS
    if args.only:
        train_slots = tuple(s for s in BREADTH_SLOTS if s.category in set(args.only))
        if not train_slots:
            print(f"--only {args.only} matched no category", file=sys.stderr)
            return 2

    # A partial sweep produces across-task numbers at the wrong N. Writing them
    # into the manuscript's own table directory would put figures in the paper
    # that contradict its prose -- exactly the failure this paper argues
    # against -- so the combination is refused rather than warned about.
    if args.allow_partial and args.out_dir is None:
        print(
            "--allow-partial writes across-task tables at a reduced N, which must not land in\n"
            "paper/tables/. Pass --out-dir to send them somewhere else, e.g.\n"
            "  python paper/experiments/run.py analysis --allow-partial --out-dir /tmp/partial",
            file=sys.stderr,
        )
        return 2

    install_paper_style()
    print(f"stages     : {', '.join(stages)}")
    print(f"seeds      : {seeds}" + (f"   epochs override: {args.epochs}" if args.epochs else ""))
    print(f"figures    : {fig_dir}")
    print(f"tables     : {tab_dir}")
    print(f"cache      : {cache_dir}")
    if args.smoke:
        print("MODE       : smoke -- fast, statistically meaningless, not the paper's numbers")

    # `verify` on its own is read-only: it reads whatever is cached and never
    # starts a training run. Combined with a training stage it verifies what
    # that stage just produced.
    cache_only = not ({"planetoid", "breadth", "tuning", "positive_control"} & set(stages))
    reads_cache = bool({"verify", "analysis"} & set(stages))
    want_planetoid = bool({"planetoid", "tuning"} & set(stages)) or (reads_cache and cache_only)
    want_breadth = "breadth" in stages or (reads_cache and cache_only)

    checks: list[tuple[bool, str]] = []
    planetoid_report = None
    breadth_reports: dict = {}
    breadth_stats: dict = {}
    breadth_rows: list = []
    started = time.time()

    if "catalog" in stages:
        print("\n[catalog] Table 4: dataset catalogue")
        catalog.emit(tab_dir=tab_dir)

    if "usability" in stages:
        print("\n[usability] measured cost of the same report without the framework")
        usability.emit(tab_dir=tab_dir, artifact_dir=artifact_dir)

    if want_planetoid:
        print("\n[planetoid] Section 3.5: Cora / CiteSeer / PubMed, two pairwise tests")
        planetoid_report = planetoid.train(
            seeds=seeds,
            data_root=args.data_root,
            cache_dir=cache_dir,
            epochs=args.epochs,
            force=args.force,
            verbose=not args.quiet,
            cache_only=cache_only,
            chunk=args.chunk,
            max_seconds=args.max_seconds,
        )
        if "planetoid" in stages and planetoid_report is not None:
            planetoid.emit(
                planetoid_report,
                fig_dir=fig_dir,
                tab_dir=tab_dir,
                artifact_dir=artifact_dir,
                formats=formats,
            )

    if want_breadth:
        print(f"\n[breadth] Section 3: {len(train_slots)} categories x <= 5 models x {len(seeds)} seeds")
        breadth_reports = breadth.train(
            seeds=seeds,
            data_root=args.data_root,
            cache_dir=cache_dir,
            slots=train_slots,
            epochs=args.epochs,
            force=args.force,
            verbose=not args.quiet,
            cache_only=cache_only,
            chunk=args.chunk,
            max_seconds=args.max_seconds,
        )
        if train_slots != BREADTH_SLOTS:
            # Pull the untrained slots from cache so a narrowed run can still
            # emit the complete figures when everything else is already there.
            rest = tuple(s for s in BREADTH_SLOTS if s not in train_slots)
            breadth_reports |= breadth.train(
                seeds=seeds,
                data_root=args.data_root,
                cache_dir=cache_dir,
                slots=rest,
                epochs=args.epochs,
                verbose=False,
                cache_only=True,
            )
        missing = [s.key for s in BREADTH_SLOTS if s.key not in breadth_reports]
        if missing:
            print(f"  ! {len(missing)} of 10 slots not available; skipping Figure 2/3 and Table 3: {missing}")
        elif "breadth" in stages:
            breadth_stats, breadth_rows = breadth.emit(
                breadth_reports,
                fig_dir=fig_dir,
                tab_dir=tab_dir,
                artifact_dir=artifact_dir,
                slots=BREADTH_SLOTS,
                formats=formats,
            )
        elif cache_only:
            breadth_stats, breadth_rows, _ = breadth.aggregate(breadth_reports, BREADTH_SLOTS)

    if "analysis" in stages:
        print("\n[analysis] revision analyses (cache only, no retraining)")
        merged = None
        if cached := {s.key: r for s in BREADTH_SLOTS if (r := breadth_reports.get(s.key))}:
            if len(cached) == len(BREADTH_SLOTS):
                merged = breadth.merge_slots(cached)
            elif args.allow_partial:
                merged = breadth.merge_slots(cached)
                print(
                    f"  ! PARTIAL: {len(cached)}/{len(BREADTH_SLOTS)} slots cached, so the "
                    f"across-task analyses use N={len(cached)}, not the paper's "
                    f"N={len(BREADTH_SLOTS)}. Rerun without --allow-partial once the sweep finishes."
                )
            else:
                print(
                    f"  ! only {len(cached)}/{len(BREADTH_SLOTS)} slots cached; skipping the "
                    f"across-task analyses (pass --allow-partial to run them anyway)"
                )
        analysis.emit(
            breadth=merged,
            planetoid=planetoid_report,
            fig_dir=fig_dir,
            tab_dir=tab_dir,
            artifact_dir=artifact_dir,
            formats=formats,
            n_boot=args.n_boot,
        )

    if "positive_control" in stages:
        print("\n[positive_control] planted-effect sweep + Monte-Carlo calibration")
        control = positive_control.train(
            seeds=seeds,
            data_root=args.data_root,
            cache_dir=cache_dir,
            epochs=args.epochs,
            force=args.force,
            verbose=not args.quiet,
            cache_only=cache_only,
            chunk=args.chunk,
            max_seconds=args.max_seconds,
        )
        if control is None:
            print("  ! not cached; rerun without cache-only stages to train it")
        else:
            pc_stats = positive_control.emit(control, tab_dir=tab_dir, artifact_dir=artifact_dir)
            # The control is only a control if it actually rejects.  A silent
            # failure here would leave the manuscript claiming a positive
            # result the pipeline did not produce, so it fails the stage
            # outright rather than waiting for the `verify` stage to notice --
            # this stage is run on its own often enough that a check only
            # reported under `verify` would be a check in name only.
            rejected = bool(pc_stats["friedman"]["rejected"])
            checks.append(
                (
                    rejected,
                    "positive control: Friedman rejects the equal-rank null on the planted-effect sweep",
                )
            )
            if not rejected:
                print(
                    "\n  FAIL: the planted-effect control did not reject the equal-rank null.\n"
                    "  The manuscript's Appendix I.2 claims it does, so this is a hard failure:\n"
                    "  either the statistical layer regressed or the generator no longer plants\n"
                    "  an effect. Do not ship the appendix until this passes.",
                    file=sys.stderr,
                )
                return 1

    if "tuning" in stages:
        print("\n[tuning] bounded hyperparameter-search study on social/node_cls")
        if planetoid_report is None:
            print("  ! needs the planetoid cache; run the 'planetoid' stage first")
        else:
            tuned = tuning.train(
                seeds=seeds,
                data_root=args.data_root,
                cache_dir=cache_dir,
                epochs=args.epochs,
                force=args.force,
                verbose=not args.quiet,
                cache_only=cache_only,
                chunk=args.chunk,
                max_seconds=args.max_seconds,
            )
            if tuned is None:
                print("  ! not cached; rerun without cache-only stages to train it")
            else:
                tuning.emit(planetoid_report, tuned, tab_dir=tab_dir, artifact_dir=artifact_dir)

    if "loaders" in stages:
        print("\n[loaders] auditing every catalogued loader")
        matrix_csv = artifact_dir / "loader_matrix.csv"
        frame, n_expected = loaders.run(
            root=str(Path(args.data_root).parent / "validate"),
            categories=args.only,
            only_cached=args.only_cached,
            resume_from=matrix_csv,
            max_seconds=args.max_seconds,
        )
        loaders.emit(frame, tab_dir=tab_dir, artifact_dir=artifact_dir, expected=n_expected)

    if "verify" in stages:
        print("\n[verify] comparing against the values published in main.tex")
        if args.smoke:
            print("  skipped: --smoke runs cannot match the paper's numbers")
        else:
            expected = verify.load_expected()
            if breadth_rows:
                checks += verify.check_breadth(breadth_rows, expected)
            if breadth_stats:
                checks += verify.check_headline(breadth_stats, expected)
            if planetoid_report is not None:
                checks += verify.check_planetoid(planetoid_report, expected, ALPHA)
            if not checks:
                print("  nothing cached to check: run the 'breadth' and/or 'planetoid' stages first")
            elif not verify.report(checks, quiet_ok=not args.show_ok):
                print(f"\ndone in {time.time() - started:.0f}s -- with differences beyond tolerance")
                return 1

    print(f"\ndone in {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
