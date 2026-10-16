"""On-disk cache of trained runs, so figures/tables regenerate without retraining.

Appendix E of the paper promises that the metric tensor is part of the
artefact bundle.  Here that bundle is one JSON file per benchmark cell
holding the raw per-seed training histories -- the ``X[t, m, s]`` tensor of
Algorithm 1, before any statistics are applied.  JSON rather than pickle so
the cache stays readable, diffable, and independent of the library version
that produced it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from graphnetz import BenchmarkReport

SCHEMA = 1


def _safe_config(config: dict[str, Any]) -> dict[str, Any]:
    """Drop unserialisable entries (``run_benchmark`` stores a ``Task``)."""
    return {k: v for k, v in config.items() if isinstance(v, (str, int, float, bool, type(None)))}


def save(path: Path, report: BenchmarkReport) -> Path:
    """Write ``report``'s seeds/histories/config to ``path`` as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": SCHEMA,
        "seeds": list(report.seeds),
        "config": _safe_config(report.config),
        "histories": report.histories,
    }
    path.write_text(json.dumps(payload, indent=1, sort_keys=False) + "\n")
    return path


def load(path: Path) -> BenchmarkReport:
    """Rebuild a :class:`BenchmarkReport` from a cache file written by :func:`save`."""
    payload = json.loads(path.read_text())
    if payload.get("schema") != SCHEMA:
        msg = f"{path}: cache schema {payload.get('schema')!r} != {SCHEMA}; delete it and rerun"
        raise ValueError(msg)
    return BenchmarkReport(
        seeds=tuple(payload["seeds"]),
        histories=payload["histories"],
        config=payload["config"],
    )
