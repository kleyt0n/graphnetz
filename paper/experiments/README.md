# Reproducing the paper

This directory regenerates every table and figure of *GraphNetz: Statistical
Benchmarking of Graph Neural Networks with Paired Tests and Rank Aggregation*
(LoG 2026). The LaTeX source lives in a separate repository,
[`kleyt0n/paper-graphnetz`](https://github.com/kleyt0n/paper-graphnetz); its
`tables/` and `figures/` are this pipeline's output.

Nothing here re-implements the statistics. Intervals, Holm correction,
Friedman/Nemenyi, power, equivalence and the CD diagram all come from
`graphnetz.BenchmarkReport`. This directory chooses the datasets, drives the
seeds and typesets the result.

## Run it

```bash
# from the repository root
uv sync --extra ogb                       # ogb is needed for the 62-loader count
uv run python paper/experiments/run.py all --out-dir out/
```

`all` runs every stage except the loader audit. With the committed cache it
retrains nothing and finishes in seconds; `verify` then checks the result
against the published values and should report `67/67 checks within
tolerance`. Without `--out-dir` the output goes to `paper/figures/` and
`paper/tables/`.

| Stage | Produces | Cost |
|---|---|---|
| `catalog` | Table 11 (catalogue) | seconds |
| `usability` | Table 10 (cost without the framework) | seconds |
| `planetoid` | Table 4 and the demo figures | ~20 min to train, seconds from cache |
| `breadth` | Table 2, Figures 2 and 6 | hours to train, seconds from cache |
| `analysis` | Tables 3, 5–9, Figures 4–5 | seconds (cache only) |
| `positive_control` | Tables 14 and 15 | minutes to train, seconds from cache |
| `tuning` | the bounded search study (opt-in) | grid size × the untuned cost |
| `loaders` | Tables 12 and 13 (opt-in; downloads every dataset) | hours |
| `verify` | cache vs. the published values | seconds |

Useful flags: `--force` (retrain instead of reading the cache), `--seeds N`,
`--epochs N`, `--only CATEGORY ...`, `--smoke` (2 seeds, 2 epochs, throwaway
output), `--pdf`, `--quiet`, `--show-ok`.

## What is committed

* `_artifacts/cache/*.json`: the per-seed training histories of every cell,
  the `X[t, m, s]` tensor of Algorithm 1 before any statistic is applied. This
  is what makes the tables reproducible without retraining, and every analysis
  in the appendix is a re-reduction of it.
* `_artifacts/*.csv`, `*.json`: the table cells as plain numbers, the loader
  audit matrix, and the analysis summaries.
* `expected.json`: the values the paper prints, which `verify` compares against.

Raw datasets are not committed. The training stages download them into
`data/benchmark/` on first use (about 2 GB for the cross-category sweep).

## Versions

The paper's numbers were produced with graphnetz **0.1.2** (git tag `v0.1.2`)
on Python 3.12.13, `torch` 2.11.0, `torch-geometric` 2.7.0, `scipy` 1.17.1 and
`numpy` 2.4.4, CPU (Apple silicon). Each cached cell written after stack
recording was added carries its stack in `config`.

Running this pipeline on **0.2.0** from the cache reproduces every table and
figure. One table re-reduces differently by design: the reliability weights of
Table 8 now average CI widths over the ranked encoders only, and the paper
quotes the corrected values. Retraining with 0.2.0 (`--force`) changes two
rows of Table 2: 0.2.0 scores graph classification on a held-out 80/10/10
split, while 0.1.2 reported validation accuracy on an 80/20 split, so the MUTAG
and MNIST-superpixels cells move. Install `v0.1.2` to retrain exactly as the
paper did.

Training is deterministic for a fixed stack with one observed exception: GAT's
CPU kernels did not reproduce bit for bit across two runs of the Cora cell (seed
6 ended at 0.793 in one and 0.778 in the other). `verify` therefore allows 0.02
in metric units and 0.25 in mean-rank units and reports drift rather than
asserting bit-equality.

## Layout

```
_config.py      frozen protocol: slots, models, seeds, LaTeX spelling
_style.py       the paper palette and CD-label tinting
_runs.py        resolve curated tasks, train a cell, merge cells
_cache.py       JSON round-trip of a BenchmarkReport
_latex.py       the hand-shaped tables
breadth.py      Section 3: sweep, dashboard, Table 2, CD diagram
planetoid.py    Section 3.6: the two pairwise tests and the demo figures
analysis.py     capacity, power, equivalence, epoch selection, aggregation,
                rank stability (cache only)
positive_control.py  the planted-effect SBM sweep and the Monte Carlo calibration
catalog.py      the dataset-catalogue table, from the loader registry
loaders.py      the loader audit matrix
usability.py    measured cost of the same report without the framework
usability_variants/  with_graphnetz.py and with_pyg.py, the two scripts it measures
tuning.py       the bounded hyperparameter-search study
verify.py       cache vs. published values
run.py          CLI
```
