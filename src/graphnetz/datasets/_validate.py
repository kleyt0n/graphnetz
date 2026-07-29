"""Validate that catalogued loaders actually load.

A catalogue is a claim, and a claim about 62 loaders should be auditable
rather than asserted.  [`validate_loaders`][graphnetz.datasets.validate_loaders] instantiates every entry in
[`LOADER_REGISTRY`][graphnetz.datasets.LOADER_REGISTRY] and reports, per loader, whether it
returned a usable dataset, what shape it has, and how long it took.

Statuses are deliberately fine-grained so that a missing optional dependency
is never reported as a broken loader:

``ok``
    Instantiated and exposed the attributes its task types require.
``needs-extra``
    Raised `ImportError` -- an optional extra (``ogb``, ``rdkit``) is
    not installed.  Not a defect.
``download-failed``
    Network or archive error while fetching the raw data.  Not a code defect,
    but it does mean the loader is unusable right now.
``error``
    Anything else, with the exception text recorded.
``skipped``
    Excluded by the caller (e.g. ``allow_download=False`` for a loader with no
    local cache).
"""

from __future__ import annotations

import inspect
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, cast

__all__ = ["LoaderStatus", "validate_loaders"]

LoaderStatus = str

_DOWNLOAD_HINTS = (
    "urlopen",
    "URLError",
    "HTTPError",
    "Connection",
    "timed out",
    "Temporary failure",
    "Name or service not known",
    "certificate",
    "BadZipFile",
    "tarfile",
    "No such file or directory",
    # A raw archive that arrived truncated -- an interrupted download leaves a
    # short or zero-byte file that the reader then chokes on.  This is a
    # transport failure, not a defective loader, and must not be reported as
    # one: the audit table is evidence about the catalogue, so a
    # misclassification here becomes a false claim in the paper.
    "EOFError",
    "No data left in file",
    "Ran out of input",
    "not a zip file",
    "invalid load key",
    "truncated",
    # A transfer that stalled or was cut off mid-stream. Same reasoning: the
    # loader never got a fair test, so this is not evidence against it.
    "FSTimeoutError",
    "TimeoutError",
    "IncompleteRead",
    "Connection broken",
    "ChunkedEncodingError",
)


# Failures whose text merely *resembles* a truncated archive but that are in
# fact defects on our side. Excusing these hid two real bugs behind a
# ``download-failed`` label, which is the same misreporting this classifier
# exists to prevent -- only in the direction that flatters the catalogue.
_NOT_A_DOWNLOAD_FAILURE = (
    # PyTorch 2.6 flipped ``torch.load(weights_only=...)`` to True, so reading a
    # pickled artifact now fails on a download that completed perfectly.
    "weights only load failed",
    "weights_only",
)


def _classify(exc: BaseException, root: str | None = None) -> LoaderStatus:
    """Sort a loader failure into a status a reader can act on.

    Order matters: a missing optional dependency is not a defect, a failed
    download is not a defect, and only what remains is reported as ``error``.
    ``root`` lets an incomplete download be detected from the filesystem, which
    catches the cases where the exception text is uninformative.

    The excuses are checked against `_NOT_A_DOWNLOAD_FAILURE` first, because a
    classifier that is generous in the wrong direction publishes our own bugs as
    somebody else's dead mirror.
    """
    text = f"{type(exc).__name__}: {exc}"
    lowered = text.lower()
    if isinstance(exc, ImportError):
        return "needs-extra"
    if any(marker in lowered for marker in _NOT_A_DOWNLOAD_FAILURE):
        return "error"
    # Matching on exception *type* catches what the message does not: a
    # ``BrokenPipeError`` reads "[Errno 32] Broken pipe" and mentions neither a
    # connection nor a download, so ModelNet10 losing its transfer 374 MB into a
    # 451 MB archive was filed as a catalogue defect. Every ``ConnectionError``
    # subclass -- broken pipe, reset, aborted -- is a transport failure.
    if isinstance(exc, (EOFError, TimeoutError, ConnectionError)):
        return "download-failed"
    if root is not None and _download_incomplete(root):
        return "download-failed"
    if any(hint.lower() in lowered for hint in _DOWNLOAD_HINTS):
        return "download-failed"
    return "error"


def _download_incomplete(root: str) -> bool:
    """True when the download left nothing usable behind.

    Two symptoms of a transfer that failed rather than a loader that is broken:
    a zero-byte file (truncated mid-write), or no file at all (the fetch never
    produced one, e.g. a timeout). In either case the loader has not been given
    a fair test, so reporting it as a catalogue defect would be wrong.
    """
    base = Path(root)
    if not base.exists():
        return True
    files = [f for f in base.rglob("*") if f.is_file()]
    if not files:
        return True
    return any(f.stat().st_size == 0 for f in files)


# Extensions a failed transfer can leave behind half-written.
_ARCHIVE_SUFFIXES = (".zip", ".gz", ".tgz", ".tar", ".bz2", ".xz", ".7z", ".npz")

# Symptoms of an archive that is present but unreadable, as opposed to absent.
_CORRUPT_ARCHIVE_HINTS = (
    "badzipfile",
    "not a zip file",
    "ran out of input",
    "no data left in file",
    "compressed file ended",
    "invalid load key",
    "truncated",
    "tarfile",
)


def _purge_partial_archives(root: str, exc_text: str) -> list[str]:
    """Delete half-written archives so the next pass can re-download.

    A killed transfer leaves a truncated archive, and the downloaders in this
    ecosystem reuse whatever is on disk without an integrity check --- OGB even
    logs ``Using exist file`` before failing to unzip it. The loader then appears
    *permanently* broken, and no number of retries recovers it, which quietly
    turns a one-off interruption into a false catalogue defect.

    Only files this harness downloaded are removed, only inside the single
    loader's own cache directory, and only when they are empty or the failure
    says the archive itself is unreadable. Everything removed is public data that
    re-downloads, and the caller records what was cleared.
    """
    base = Path(root)
    if not base.exists():
        return []
    corrupt = any(hint in exc_text.lower() for hint in _CORRUPT_ARCHIVE_HINTS)
    removed: list[str] = []
    for path in sorted(base.rglob("*")):
        if not path.is_file():
            continue
        empty = path.stat().st_size == 0
        archive = path.name.lower().endswith(_ARCHIVE_SUFFIXES)
        if empty or (corrupt and archive):
            try:
                path.unlink()
            except OSError:  # pragma: no cover - racing another process
                continue
            removed.append(path.name)
    return removed


def _shape(ds: Any) -> dict[str, Any]:
    """Best-effort shape summary that works for PyG datasets and tuples."""
    if isinstance(ds, tuple):
        ds = ds[0]
    out: dict[str, Any] = {}
    try:
        out["n_graphs"] = len(ds)
    except TypeError:
        out["n_graphs"] = None

    def _int_attr(name: str) -> int | None:
        try:
            return int(getattr(ds, name))
        except (AttributeError, TypeError, ValueError):
            return None

    for attr in ("num_features", "num_classes", "num_relations"):
        out[attr] = _int_attr(attr)
    try:
        data = ds[0]
        out["n_nodes"] = int(data.num_nodes) if data.num_nodes is not None else None
        out["n_edges"] = int(data.num_edges) if data.num_edges is not None else None
        out["has_x"] = getattr(data, "x", None) is not None
        out["has_y"] = getattr(data, "y", None) is not None
        out["masks"] = all(hasattr(data, f"{s}_mask") for s in ("train", "val", "test"))
    except Exception:
        out.setdefault("n_nodes", None)
        out.setdefault("n_edges", None)
    return out


def _call(loader: Callable[..., Any], root: str, seed: int) -> Any:
    """Invoke a loader, threading ``seed`` when its signature accepts one."""
    try:
        seed_aware = "seed" in inspect.signature(loader).parameters
    except (TypeError, ValueError):
        seed_aware = False
    return loader(root, seed=seed) if seed_aware else loader(root)


def validate_loaders(
    categories: Iterable[str] | None = None,
    *,
    root: str = "data/validate",
    seed: int = 0,
    allow_download: bool = True,
    only_cached: bool = False,
    skip: Iterable[str] | None = None,
    max_seconds: float | None = None,
    retries: int = 1,
    on_row: Callable[[dict[str, Any]], None] | None = None,
) -> Any:
    """Instantiate every catalogued loader and report a status matrix.

    Returns a `pandas.DataFrame` with one row per (category, loader):
    the task types it serves, its status, shape columns, elapsed seconds, and
    the error text when it failed.  One row per *unique* loader name within a
    category -- a loader serving two task types is validated once and its
    ``tasks`` column lists both.

    ``only_cached=True`` skips loaders whose root directory does not already
    exist, which is what a network-free CI job wants.

    Auditing the whole catalogue downloads tens of gigabytes and can take hours,
    so the walk is resumable: pass already-audited loader names in ``skip`` and a
    wall-clock budget in ``max_seconds``, and the call returns what it finished.
    Accumulating several bounded calls yields the same matrix as one long one,
    which is what makes the audit runnable in a constrained environment.

    ``retries`` re-attempts a loader that failed to *download*, after clearing
    whatever half-written archive the attempt left behind; a loader that failed
    for any other reason is not retried, because the second attempt would fail
    the same way. Retries respect ``max_seconds``.

    ``on_row`` is invoked with each row as soon as that loader finishes. Resuming
    is only as good as what reached disk, so a caller that checkpoints here keeps
    its progress when the walk is interrupted -- which, over a multi-hour
    download, is the normal way for it to end.
    """
    import pandas as pd

    from graphnetz.datasets import LOADER_REGISTRY

    cats = list(categories) if categories is not None else list(LOADER_REGISTRY)
    already = set(skip or ())
    rows: list[dict[str, Any]] = []
    # Two clocks, deliberately separate: ``run_started`` bounds the whole call
    # against ``max_seconds``, ``loader_started`` times one loader for the
    # ``seconds`` column. Sharing one variable silently disabled the budget,
    # because each loader reset the clock the budget was measured against.
    run_started = time.perf_counter()
    exhausted = False

    for category in cats:
        per_cat = LOADER_REGISTRY.get(category, {})
        # loader name -> (callable, [task types])
        merged: dict[str, tuple[Callable[..., Any], list[str]]] = {}
        for task_type, entries in per_cat.items():
            for name, fn in entries:
                if name in merged:
                    merged[name][1].append(task_type)
                else:
                    merged[name] = (cast("Callable[..., Any]", fn), [task_type])

        for name, (fn, task_types) in merged.items():
            if name in already:
                continue
            if exhausted or (max_seconds is not None and time.perf_counter() - run_started >= max_seconds):
                exhausted = True
                break
            target = f"{root}/{category}/{name}"
            row: dict[str, Any] = {
                "category": category,
                "loader": name,
                "tasks": "+".join(sorted(task_types)),
                "status": "skipped",
                "seconds": 0.0,
                "error": "",
            }
            if only_cached and not Path(target).exists():
                row["error"] = "not cached and only_cached=True"
                rows.append(row)
                continue
            if not allow_download and not Path(target).exists():
                row["error"] = "would require a download and allow_download=False"
                rows.append(row)
                continue
            loader_started = time.perf_counter()
            # A transport failure says nothing about the loader, so one attempt
            # is not evidence. Most of the failures observed here were transient
            # -- a reset connection, a stalled mirror -- and cleared on a second
            # try against a cache the purge had just made clean.
            for attempt in range(retries + 1):
                try:
                    ds = _call(fn, target, seed)
                    row.update(_shape(ds))
                    row["status"] = "ok"
                    row["error"] = ""
                    break
                except BaseException as exc:
                    row["status"] = _classify(exc, target)
                    row["error"] = f"{type(exc).__name__}: {exc}"[:400]
                    if row["status"] != "download-failed":
                        break
                    cleared = _purge_partial_archives(target, row["error"])
                    if cleared:
                        row["error"] = f"{row['error']} [cleared partial: {', '.join(cleared[:3])}]"
                    if attempt == retries:
                        break
                    if max_seconds is not None and time.perf_counter() - run_started >= max_seconds:
                        row["error"] = f"{row['error']} [no retry: time budget spent]"
                        break
            row["seconds"] = round(time.perf_counter() - loader_started, 2)
            rows.append(row)
            if on_row is not None:
                on_row(row)

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(["category", "loader"]).reset_index(drop=True)
    return frame
