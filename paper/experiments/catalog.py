"""Table 4: the dataset catalogue, generated from the shipped loader registry.

The table is derived, not curated: it reads
``graphnetz.datasets.LOADER_REGISTRY`` through
:func:`graphnetz.list_datasets`, so the counts in the paper cannot drift from
the loaders the library actually exposes.

Note on the total.  The registry folds in the five OGB loaders only when the
``ogb`` extra is importable, so the catalogue totals 62 loaders with the
extra installed and 57 without it.  The paper reports 62; this stage warns
loudly when it would produce a different number.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from paper.experiments import _latex
from paper.experiments._config import TASK_ORDER

PAPER_TOTAL = 62
OGB_LOADERS = ("ogbn_arxiv", "ogbl_collab", "ogbn_products", "ogbg_molhiv", "ogbg_molpcba")


def emit(*, tab_dir: Path) -> int:
    """Write ``tables/categories.tex``; returns the loader total it recorded."""
    from graphnetz import list_datasets

    registry = list_datasets()
    total = sum(len({n for t in TASK_ORDER if per.get(t) for n in per[t]}) for per in registry.values())

    if importlib.util.find_spec("ogb") is None:
        print(
            f"  ! ogb is not installed: {len(OGB_LOADERS)} OGB loaders are absent from the registry, "
            f"so the catalogue totals {total} instead of the paper's {PAPER_TOTAL}.\n"
            f"    Install the extra to reproduce Table 4 exactly:  uv sync --extra ogb"
        )
    elif total != PAPER_TOTAL:
        print(f"  ! catalogue totals {total} loaders, but the paper reports {PAPER_TOTAL}")

    path = _latex.categories_table(tab_dir / "categories.tex", registry)
    print(f"  wrote {path} ({total} loaders across {len(registry)} categories)")
    return total
