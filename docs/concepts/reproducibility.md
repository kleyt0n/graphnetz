# Reproducibility

A benchmark whose numbers move between runs cannot support any of the
statistics in [Comparison](comparison.md), because the pairing those tests
rely on would be fictional. This page states exactly what GraphNetz
guarantees, and where the guarantee stops.

## What gets reseeded, and when

Before **each** $(task, model, seed)$ triple — not once per sweep — the
runner reseeds every RNG the training code can reach:

| RNG | call |
| --- | --- |
| Python | `random.seed(s)` |
| NumPy | `numpy.random.seed(s)` |
| Torch CPU | `torch.manual_seed(s)` |
| Torch CUDA | `torch.cuda.manual_seed_all(s)` |

Per-triple rather than per-sweep is the load-bearing detail. Seeding once at
the start would make each model's initialisation depend on how many models
ran before it, so `{"GCN": …, "GAT": …}` and `{"GAT": …, "GCN": …}` would
produce different numbers from the same request — and the per-seed
differences the paired tests consume would no longer be paired.

Combinatorial loaders thread the seed through to their data generator, so
cross-seed variance on those tasks reflects **both** model initialisation and
data resampling, not only the former.

```python
def my_loader(root: str, *, seed: int):
    return MySyntheticDataset(root, num_graphs=100, seed=seed)
```

The dispatcher detects the `seed` keyword via `inspect.signature` and passes
the benchmark seed in. A loader without it is simply called `f(root)`, and
its data is then constant across seeds by construction.

## What the guarantee covers

A run with the same seed list and software stack reproduces bit-for-bit **on
the same hardware**.

The hardware qualifier is not hedging. Floating-point reduction order differs
between CPU and GPU and between CUDA kernel selections, so the same seed on a
different device gives a slightly different trajectory. That is a property of
the arithmetic, not of the library. What is preserved across devices is the
protocol: the same splits, the same initialisation draws, the same data.

| stable across | |
| --- | --- |
| reruns on one machine | bit-for-bit |
| model ordering in the `models` dict | yes, by construction |
| CPU ↔ CUDA ↔ MPS | protocol yes, last digits no |
| library versions | no — pin them for a published result |

## Device placement

`run_benchmark(..., device="auto")` is the default. The runtime picks CUDA
when available, then Apple-silicon MPS, then CPU, and moves the model and
data for you. Pin it explicitly when a result has to be comparable to an
earlier one:

```python
run_benchmark("social", MODELS, seeds=range(10), device="cpu")
```

## The cache is keyed, not shared

Datasets are downloaded and processed into
`data/benchmark/<category>/<task>/`, overridable with `root=`. Reruns hit the
on-disk cache, which is what makes iterating on report and plotting logic
fast. Ad-hoc runs passing `tasks=` default to `category="custom"` purely so
their cache directories do not collide with the curated catalogue.

A cache hit returns the same processed graph, so it does not weaken the
reproducibility guarantee. What it does mean is that a corrupt half-written
download is sticky — see the
[catalogue auditor](../guides/datasets.md#auditing-the-catalogue), which
clears and retries rather than failing identically forever.

## Scoring epoch is a protocol choice

By default every statistic is computed at the **final** epoch. This is a
choice, not a neutral position: scoring models at an arbitrary point of their
trajectories rather than at their best can manufacture significance, and
scoring them at their best on the *same* split it was chosen from manufactures
more.

The report keeps the full per-epoch history, so the alternative costs nothing
and requires no retraining:

```python
report.epoch_selection = "best_val"   # or pass epoch_selection= per call
report.pairwise()                     # every statistic re-derived
report.selected_epochs()              # the epoch chosen per (task, model, seed)
```

`"best_val"` takes each run's metric at the epoch that maximised the *paired
validation* series. What makes it honest is that the reported value comes
from a split that was not used to choose it — which is also why it is not
available everywhere. Node classification and both link-prediction routines
record validation **and** test metrics per epoch; the graph-level routines
record only a validation metric, so asking for `"best_val"` there raises
rather than silently selecting and reporting on the same split. Pass
`strict=False` to let such tasks fall back to the final epoch, and check
`epoch_selection_support()` to see which ones did.

`selected_epochs()` is the audit surface: an epoch pinned at the last one for
every seed means selection did nothing, and a wildly varying one means the
run had not converged.

## Publishing a result

Use the JSON round-trip. It stores the metric tensor $X[t, m, s]$ and the raw
per-epoch histories **before** any statistic is applied:

```python
report.to_json("report_social.json")
report = BenchmarkReport.from_json("report_social.json")
```

JSON rather than pickle so the artefact stays readable, diffable and
independent of the library version that produced it. Because it holds
histories rather than reduced numbers, a published bundle can be re-analysed
later under a different CI method, pairwise test, epoch-selection rule or
aggregation — by a reviewer, without anyone re-running the training. That is
the strongest reproducibility claim in the library, and it is one line.

## Next

- [Benchmark protocol](../guides/benchmark.md) — the five stages around this.
- [Reading the report](../guides/report.md) — every view on the stored tensor.
- [Adequacy](adequacy.md) — what the stored tensor can and cannot settle.
