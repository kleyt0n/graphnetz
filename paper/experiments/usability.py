"""Measure the cost of producing the same report with and without GraphNetz.

Reviewers Bvni (W1) and HVs6 (Q3) both ask for evidence that the framework is
more than an integration of existing statistical tools.  A checkmark table
cannot answer that; a measurement can.

``usability_variants/with_graphnetz.py`` and ``usability_variants/with_pyg.py``
are two runnable
scripts that produce the *same three artefacts* -- per-cell 95 % t-CI,
Holm-corrected paired tests within each task, and a Demšar critical-difference
diagram across three Planetoid tasks and four architectures over ten seeds.
This stage counts, for each:

*effective lines*
    Non-blank, non-comment, non-docstring lines of user code.
*statistical decisions*
    Marked ``# --- DECISION:`` in the baseline -- choices a user must make
    correctly and that the framework otherwise fixes (paired vs unpaired test,
    Holm family scope, rank direction, the ``q_alpha/√2`` conversion, ...).
    This is the number that matters: lines are effort, decisions are risk.
*imports*
    Distinct third-party modules the user must know about.

    tables/usability.tex   -- the comparison
    _artifacts/usability.json
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

from paper.experiments import _latex

HERE = Path(__file__).resolve().parent / "usability_variants"

VARIANTS: tuple[tuple[str, str], ...] = (
    ("GraphNetz", "with_graphnetz.py"),
    ("PyTorch Geometric + scipy", "with_pyg.py"),
)

DECISION_MARKER = "# --- DECISION:"


def _docstring_lines(tree: ast.Module) -> set[int]:
    """Line numbers occupied by module/function/class docstrings."""
    out: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        body = getattr(node, "body", [])
        if not body:
            continue
        first = body[0]
        is_str_const = (
            isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str)
        )
        if is_str_const:
            out.update(range(first.lineno, (first.end_lineno or first.lineno) + 1))
    return out


def measure(path: Path) -> dict[str, Any]:
    """Effective lines, decision markers, and third-party imports of one script."""
    text = path.read_text()
    lines = text.splitlines()
    tree = ast.parse(text)
    skip = _docstring_lines(tree)

    effective = 0
    for i, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or i in skip:
            continue
        effective += 1

    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imports.add(node.module.split(".")[0])
    stdlib = {"itertools", "random", "math", "json", "pathlib", "typing", "ast", "collections"}

    return {
        "file": path.name,
        "effective_lines": effective,
        "total_lines": len(lines),
        "decisions": text.count(DECISION_MARKER),
        "imports": sorted(imports - stdlib),
        "n_imports": len(imports - stdlib),
    }


def emit(*, tab_dir: Path, artifact_dir: Path) -> dict[str, Any]:
    """Write the usability comparison table; returns the measurements."""
    results: dict[str, dict[str, Any]] = {}
    for label, filename in VARIANTS:
        path = HERE / filename
        if not path.exists():
            msg = f"missing usability script: {path}"
            raise FileNotFoundError(msg)
        results[label] = measure(path)

    base = results["PyTorch Geometric + scipy"]
    ours = results["GraphNetz"]
    ratio = base["effective_lines"] / max(ours["effective_lines"], 1)

    rows = [
        [
            label,
            str(r["effective_lines"]),
            str(r["decisions"]) if r["decisions"] else r"{\color{nac}0}",
            str(r["n_imports"]),
            rf"\texttt{{{r['file'].replace('_', chr(92) + '_')}}}",
        ]
        for label, r in results.items()
    ]
    _latex.booktabs(
        tab_dir / "usability.tex",
        ["Implementation", "Lines of user code", "Statistical decisions", "Imports", "Script"],
        rows,
        align="lcccl",
    )

    out = {
        "variants": results,
        "line_ratio": round(ratio, 1),
        "decisions_avoided": base["decisions"],
    }
    (artifact_dir / "usability.json").write_text(json.dumps(out, indent=2) + "\n")
    print(
        f"  same three artefacts: {ours['effective_lines']} lines with GraphNetz vs "
        f"{base['effective_lines']} with PyG + scipy ({ratio:.0f}x), and "
        f"{base['decisions']} statistical decisions the framework fixes"
    )
    return out
