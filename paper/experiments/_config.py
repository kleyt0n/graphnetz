"""Frozen configuration of the experiments reported in ``paper/main.tex``.

Everything that pins a number in the paper lives here: which dataset fills
each category slot, which architectures compete, how many seeds, and how
loaders/categories/models are spelled in the LaTeX tables.  The training
hyperparameters themselves are *not* duplicated -- they are the per-task
defaults of ``graphnetz.training`` (Adam, 64 hidden channels; lr/weight
decay per task type), exactly as Section 3.1 of the paper states.
"""

from __future__ import annotations

from dataclasses import dataclass

# --------------------------------------------------------------------------- #
# Protocol
# --------------------------------------------------------------------------- #

SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
"""Section 3.1: ``S = 10`` seeds, 0--9, for every (task, model) cell."""

HIDDEN_CHANNELS: int = 64
"""Section 3.1: 64 hidden channels for every architecture."""

DEVICE: str = "cpu"
"""Training device. The paper ran on CPU; ``run.py --device`` overrides it.

Kept explicit rather than left to the library's ``"auto"``, which picks CUDA or
Apple-silicon MPS when present: a different device shifts metrics in the third
decimal, and MPS can run out of memory on cells a CPU trains comfortably.
"""

ALPHA: float = 0.05
"""Significance level for the paired tests and the Nemenyi post-hoc."""

CI: float = 0.95
"""Confidence level of the per-cell Student's t interval (Eq. 2)."""

MODEL_NAMES: tuple[str, ...] = ("GCN", "GAT", "GraphSAGE", "GraphTransformer", "GIN")
"""The five benchmarked architectures, in the order the paper introduces them.

``GIN`` is registered for graph-level tasks only, so the dispatcher skips it
on ``node_cls`` / ``link_pred`` slots and those cells come out ``n/a``.
"""

PLANETOID_MODEL_NAMES: tuple[str, ...] = ("GCN", "GAT", "GraphSAGE", "GraphTransformer")
"""The Planetoid demo (Section 3.5) is node classification only -- no GIN."""

# Metric keys whose *lower* values are better; everything else is higher-better.
LOWER_IS_BETTER: frozenset[str] = frozenset({"val_mae", "train_loss"})


# --------------------------------------------------------------------------- #
# The cross-category sweep (Section 3, Figure 2, Table 3)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Slot:
    """One row of Table 3 / one panel of Figure 2.

    ``category`` / ``task_type`` / ``task_name`` address a curated task in
    ``graphnetz.benchmark.BENCHMARK_TASKS``; the ``*_label`` fields are how
    the slot is spelled in the paper.  ``epochs`` records the budget the
    paper ran with so a drift in the library's curated registry is reported
    rather than silently reproduced.
    """

    category: str
    task_type: str
    task_name: str
    category_label: str
    dataset_label: str
    epochs: int

    @property
    def key(self) -> str:
        """Stable identifier used for cache filenames and merged report keys."""
        return f"{self.category}/{self.task_name}"


@dataclass(frozen=True)
class Cell:
    """A trained slot reduced to what the figure and the table need.

    ``metric`` is the history key the slot is scored on (``test_auc``,
    ``val_acc``, ...) and ``stats`` maps each model that actually ran to its
    ``(seed mean, 95 % t-CI half-width)``.  Models the dispatcher skipped are
    simply absent, which is what produces the ``n/a`` cells of Table 3.
    """

    slot: Slot
    metric: str
    stats: dict[str, tuple[float, float]]


BREADTH_SLOTS: tuple[Slot, ...] = (
    Slot("combinatorial", "link_pred", "random_tsp", "Combinatorial", "TSP-random", 80),
    Slot("biology", "graph_cls", "mutag", "Biology", "MUTAG", 40),
    Slot("social", "node_cls", "cora", "Social", "Cora", 100),
    Slot("knowledge", "link_pred", "fb15k_237", "Knowledge", "FB15k-237", 20),
    Slot("infrastructure", "link_pred", "euroroad", "Infrastructure", "Euroroad", 80),
    Slot("finance", "link_pred", "board_directors", "Finance", "Board-directors", 40),
    Slot("computing", "link_pred", "internet_as", "Computing", "Internet AS", 40),
    Slot("vision", "graph_cls", "mnist_superpixels", "Vision", "MNIST-superpixels", 4),
    Slot("physics", "link_pred", "ising_lattice", "Physics", "Ising-lattice", 60),
    Slot("security", "link_pred", "terrorists_911", "Security", "9/11 terrorists", 120),
)
"""One representative dataset per research category, in paper (table) order.

Two things worth knowing about this list.  The Physics slot is the Ising
lattice (link prediction), not ZINC -- Section 3.1 quotes an epoch budget for
ZINC as well, but no graph-regression row reaches Table 3, so the ten
executed tasks carry no ``val_mae`` cell.  Adding one needs no code change:
the metric labels and the rank direction already handle lower-is-better.

The Social slot is Cora under ``node_cls``; the ``link_pred`` view of the same
dataset is a different curated task and is deliberately not used here, so no
dataset appears twice in the rank table.
"""


# --------------------------------------------------------------------------- #
# The Planetoid sensitivity demo (Section 3.5, Table 2 and the demo tables, demo figures)
# --------------------------------------------------------------------------- #

PLANETOID_TASK_NAMES: tuple[str, ...] = ("cora", "citeseer", "pubmed")
"""Section 3.5 runs the two pairwise tests on the social node-classification trio."""

PLANETOID_PRETTY: dict[str, str] = {
    "cora": "Cora",
    "citeseer": "CiteSeer",
    "pubmed": "PubMed",
}


# --------------------------------------------------------------------------- #
# LaTeX spelling
# --------------------------------------------------------------------------- #

MODEL_HEADERS: dict[str, str] = {
    "GCN": "GCN",
    "GAT": "GAT",
    "GraphSAGE": "SAGE",
    "GraphTransformer": "GT",
    "GIN": "GIN",
}
"""Column headers of Table 3 -- abbreviated so the table fits the text width."""

METRIC_LABELS: dict[str, str] = {
    "test_acc": r"Acc.\,$\uparrow$",
    "val_acc": r"Acc.\,$\uparrow$",
    "test_auc": r"AUC\,$\uparrow$",
    "val_auc": r"AUC\,$\uparrow$",
    "val_mae": r"MAE\,$\downarrow$",
}
"""``Metric`` column of Table 3: name plus the ``best`` direction."""

METRIC_AXIS_LABELS: dict[str, str] = {
    "test_acc": "Test acc.",
    "val_acc": "Val acc.",
    "test_auc": "Test AUC",
    "val_auc": "Val AUC",
    "val_mae": "Val MAE",
}
"""Short y-axis labels for the dashboard panels of Figure 2."""

TASK_ABBREV: dict[str, str] = {
    "node_cls": "NC",
    "graph_cls": "GC",
    "graph_reg": "GR",
    "link_pred": "LP",
}
"""Table 4 legend: NC / GC / GR / LP."""

TASK_ORDER: tuple[str, ...] = ("node_cls", "graph_cls", "graph_reg", "link_pred")
"""Canonical task ordering used for the rows of Table 4."""

CATEGORY_LABELS: dict[str, str] = {
    "combinatorial": "Combinatorial",
    "biology": "Biology",
    "social": "Social",
    "knowledge": "Knowledge",
    "infrastructure": "Infrastructure",
    "finance": "Finance",
    "computing": "Computing",
    "vision": "Vision",
    "physics": "Physics",
    "security": "Security",
}

PRETTY_LOADERS: dict[str, str] = {
    # combinatorial
    "random_bipartite_matching": "bipartite matching",
    "random_tsp": "TSP",
    "random_vrp": "VRP",
    "random_coloring": "coloring",
    "random_maxcut": "max-cut",
    "random_maxflow": "max-flow",
    # biology
    "mutag": "MUTAG",
    "proteins": "PROTEINS",
    "enzymes": "ENZYMES",
    "peptides_func": "Peptides-func",
    "peptides_struct": "Peptides-struct",
    "celegans": r"C.\,elegans",
    "budapest_connectome": "Budapest connectome",
    "hospital_contacts": "hospital contacts",
    "high_school_contacts": "high-school contacts",
    "ppi": "PPI",
    "ogbg_molhiv": "ogbg-molhiv",
    "ogbg_molpcba": "ogbg-molpcba",
    # social
    "cora": "Cora",
    "citeseer": "CiteSeer",
    "pubmed": "PubMed",
    "wikics": "WikiCS",
    "roman_empire": "Roman-empire",
    "amazon_ratings": "Amazon-ratings",
    "minesweeper": "Minesweeper",
    "tolokers": "Tolokers",
    "questions": "Questions",
    "ogbn_arxiv": "ogbn-arxiv",
    "movielens100k": "MovieLens-100k",
    "karate": "Karate",
    "facebook_friends": "Facebook friends",
    "dblp_coauthor": "DBLP coauthor",
    "dnc_emails": "DNC emails",
    "ogbl_collab": "ogbl-collab",
    # knowledge
    "fb15k_237": "FB15k-237",
    "wordnet18rr": "WordNet18-RR",
    "wordnet_netz": "WordNet (Netz)",
    # infrastructure
    "power_grid": "power grid",
    "euroroad": "EuroRoad",
    "us_roads": "US roads",
    "eu_airlines": "EU airlines",
    "london_transport": "London transport",
    "urban_streets": "urban streets",
    # finance
    "elliptic_bitcoin": "Elliptic Bitcoin",
    "ogbn_products": "ogbn-products",
    "product_space": "product space",
    "board_directors": "board of directors",
    "us_patents": "US patents",
    # computing
    "internet_as": "Internet AS",
    "topology": "Internet topology",
    "as_skitter": "AS-Skitter",
    "route_views": "route views",
    # vision
    "mnist_superpixels": "MNIST superpixels",
    "cifar10_superpixels": "CIFAR-10 superpixels",
    "modelnet10": "ModelNet10",
    "modelnet40": "ModelNet40",
    # physics
    "qm9": "QM9",
    "zinc": "ZINC",
    "ising_lattice": "Ising lattice",
    # security
    "malnet_tiny": "MalNet-Tiny",
    "terrorists_911": "9/11 terrorists",
    "train_terrorists": "train terrorists",
}
"""Loader name -> the label used in Table 4 (the dataset-catalog table)."""
