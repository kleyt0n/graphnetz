# Dataset taxonomy

GraphNetz organises **62 loaders across 10 scientific categories**, each declaring the task types it can serve. The taxonomy is the single source of truth: the curated benchmark, the per-category notebooks, and the documented loader names all derive from `graphnetz.datasets.LOADER_REGISTRY`.

```python
from graphnetz.datasets import LOADER_REGISTRY, list_datasets

list_datasets(category="biology")
# {'biology': {'graph_cls': [...], 'graph_reg': [...], 'node_cls': [...], 'link_pred': [...]}}
```

## Tasks

Every cell in the benchmark — `(category, task_type, dataset, model, seed)` —
maps to one of four task families. The default metric for each is what the
report headlines:

| Symbol | Kind | Default metric | Adapter |
|---|---|---|---|
| `node_cls` | Node classification | test accuracy | encoder used directly |
| `graph_cls` | Graph classification | val accuracy | mean-pool + linear head |
| `graph_reg` | Graph regression | val MAE | mean-pool + linear head |
| `link_pred` | Link prediction | test ROC-AUC | dot-product (homogeneous) or DistMult (relational) decoder |

!!! note
    Unlabelled graphs (Netzschleuder networks, synthetic combinatorial
    instances, the Ising lattice) enter the benchmark through `link_pred` on a
    held-out edge split, so every cell carries a real held-out metric — there
    is no self-supervised pretext loss in the headline report.

## Categories

| Category | # | Tasks | Loaders |
|---|---:|---|---|
| **Combinatorial** | 6 | LP | random TSP, VRP, max-flow, bipartite matching, coloring, max-cut |
| **Biology** | 12 | GC · GR · LP | MUTAG, PROTEINS, ENZYMES, Peptides-func/struct, PPI, *C. elegans*, Budapest connectome, hospital / high-school contacts, ogbg-molhiv†, ogbg-molpcba† |
| **Social** | 16 | NC · LP | Cora, CiteSeer, PubMed, WikiCS, Roman-empire, Amazon-ratings, Minesweeper, Tolokers, Questions, MovieLens-100k, Karate, Facebook friends, DBLP coauthor, DNC emails, ogbn-arxiv†, ogbl-collab† |
| **Knowledge** | 3 | LP | FB15k-237, WordNet18-RR, WordNet (Netzschleuder) |
| **Infrastructure** | 6 | LP | power grid, EuroRoad, US roads, EU airlines, London transport, urban streets |
| **Finance** | 5 | NC · LP | Elliptic Bitcoin, product space, board of directors, US patents, ogbn-products† |
| **Computing** | 4 | LP | Internet AS, Internet topology, AS-Skitter, route views |
| **Vision** | 4 | GC | MNIST/CIFAR-10 superpixels, ModelNet10/40 |
| **Physics** | 3 | GR · LP | QM9, ZINC, Ising lattice |
| **Security** | 3 | GC · LP | MalNet-Tiny, 9/11 terrorists, train terrorists |

!!! note
    Loaders marked with † come from OGB and require the optional `ogb`
    extra (`pip install graphnetz[ogb]`): `ogbn-arxiv` and `ogbl-collab`
    in **Social**, `ogbn-products` in **Finance**, and `ogbg-molhiv` and
    `ogbg-molpcba` in **Biology**. They are folded into their domain
    categories rather than exposed as a separate `ogb` category, so they
    appear in `run_benchmark(category, ...)` alongside the curated
    built-ins. Without the extra installed, the catalogue exposes 57 of
    the 62 loaders.

## Loading individual datasets

Each category exposes thin loader functions returning a PyG dataset:

```python
from graphnetz.datasets.social import cora, roman_empire
from graphnetz.datasets.biology import mutag, peptides_func
from graphnetz.datasets.computing import internet_as

ds_cora = cora("data/cora")                       # node_cls + link_pred
ds_rom  = roman_empire("data/roman_empire")       # heterophilic node_cls
ds_mut  = mutag("data/mutag")                     # graph_cls
ds_pep  = peptides_func("data/peptides_func")     # LRGB graph_cls
ds_inet = internet_as("data/internet_as")         # link_pred

# Optional OGB loaders live in their domain modules
# (require `pip install graphnetz[ogb]`):
from graphnetz.datasets.social import ogbn_arxiv, ogbl_collab
from graphnetz.datasets.biology import ogbg_molhiv

ds_arxiv  = ogbn_arxiv("data/ogb")                # node_cls (in Social)
ds_collab = ogbl_collab("data/ogb")               # link_pred (in Social)
ds_hiv    = ogbg_molhiv("data/ogb")               # graph_cls (in Biology)
```

The first call downloads + processes into the directory you pass; subsequent
calls hit the on-disk cache.

## Arbitrary Netzschleuder networks

The `Netz` loader fetches any network from the [Netzschleuder
catalogue](https://networks.skewed.de/) on demand and converts it to the PyG
format used by the rest of the library:

```python
from graphnetz import Netz

ds = Netz(root="data", dataset_name="urban_streets", network_name="brasilia")
data = ds[0]   # PyG Data object

# Multiplex / transit / airline networks need parallel-edge support:
ds_air = Netz(
    root="data",
    dataset_name="eu_airlines",
    network_name="eu_airlines",
    multigraph=True,
)
```

## Choosing a dataset

| If you want… | Try |
|---|---|
| A small node-classification sanity check | `social.cora`, `social.citeseer` |
| Heterophilic node classification | `social.roman_empire`, `social.minesweeper` |
| Long-range graph classification | `biology.peptides_func` (LRGB) |
| Molecular regression | `physics.zinc`, `physics.qm9` |
| Knowledge-graph link prediction | `knowledge.fb15k_237`, `knowledge.wordnet18rr` |
| A real-world infrastructure network | `infrastructure.power_grid`, `infrastructure.euroroad` |
| Synthetic, deterministic, fast | `combinatorial.random_coloring`, `combinatorial.random_tsp` |

## Auditing the catalogue

A catalogue is a claim about software, and it is worth testing rather than
trusting. `validate_loaders` instantiates every entry in the registry and
reports what happened:

```python
from graphnetz import validate_loaders

matrix = validate_loaders()                     # the whole catalogue
matrix = validate_loaders(["social", "physics"])  # or a few categories
matrix[["category", "loader", "status", "n_nodes", "seconds"]]
```

Failures are classified so that neither a missing optional dependency nor a
flaky mirror is reported as a broken loader:

| Status | Meaning |
|---|---|
| `ok` | instantiated and exposed the attributes its task types need |
| `needs-extra` | an optional extra (`ogb`, `rdkit`) is not installed — not a defect |
| `download-failed` | dead mirror, or an archive that arrived truncated |
| `error` | a genuine defect, with the exception text recorded |
| `skipped` | excluded by the caller |

The distinction is load-bearing in both directions, and getting it wrong is
easy. Calling a dropped connection a defect slanders a loader that works;
calling a defect a dropped connection hides a bug behind someone else's dead
mirror, which is worse, because nobody goes looking for it. Transport failures
are therefore matched on exception *type* — every `ConnectionError` subclass,
whose message may say only `[Errno 32] Broken pipe` — and on a filesystem probe
for an empty or zero-byte download, rather than on hopeful string matching.

!!! tip "It is slow, resumable, and self-repairing"

    A full audit downloads every raw archive in the catalogue. `max_seconds=`
    bounds one call and `skip=` continues from what you have already checked,
    so the walk completes in stages; `on_row=` fires as each loader finishes,
    which is how a caller checkpoints progress that an interruption would
    otherwise discard. `only_cached=True` restricts it to datasets already on
    disk, which is what a network-free CI job wants.

    A failed download clears the half-written archive it left behind and is
    retried once (`retries=`). Without that, one interrupted transfer is
    permanent: the downloaders in this ecosystem reuse whatever archive is on
    disk without checking it, so the loader fails identically for ever.

## Adding a new loader

See [Contributing → Adding a dataset loader](../contributing.md#adding-a-dataset-loader).
The short version:

1. Write a thin loader function under the right category module.
2. Register it in `LOADER_REGISTRY` for each task it supports.
3. Optionally add a `Task(...)` to `BENCHMARK_TASKS` for the curated run.
4. Add a one-line smoke test in `tests/test_smoke.py`.
