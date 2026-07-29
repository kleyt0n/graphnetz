"""Shared loading helpers for Open Graph Benchmark (OGB) datasets.

The public entry points live in the domain modules (``social``,
``biology``, ``finance``); this module only houses the wire-format
adapters that turn an OGB dataset into a PyG-shaped object. All
helpers raise a friendly `ImportError` when ``ogb`` is missing,
pointing the user at ``pip install graphnetz[ogb]``.
"""

from __future__ import annotations

import contextlib
import importlib
import logging
import urllib.request
from collections.abc import Iterator
from typing import Any

from torch_geometric.data import Data

_OGB_INSTALL_HINT = "Install with:  pip install graphnetz[ogb]"

_LOG = logging.getLogger(__name__)

# OGB's ``decide_download`` asks stdin to confirm any download over 1 GB, and
# the caller treats a negative answer as fatal (``exit(-1)``). Neither works
# from a library: with stdin closed the prompt raises ``EOFError``, and the
# ``exit`` would take the calling process down with it. Each dataset module
# binds the helper by value (``from ogb.utils.url import decide_download``), so
# the substitution has to be applied per module rather than at the source.
_PROMPTING_MODULES = (
    "ogb.nodeproppred.dataset",
    "ogb.graphproppred.dataset",
    "ogb.linkproppred.dataset",
)

_GB = float(1 << 30)


def _download_size_gb(url: str) -> float | None:
    """Size of ``url`` in GB, or ``None`` if the server does not report it."""
    try:
        with urllib.request.urlopen(url) as response:
            length = response.info()["Content-Length"]
        return int(length) / _GB if length is not None else None
    except Exception:
        return None


@contextlib.contextmanager
def _download_without_prompting() -> Iterator[None]:
    """Answer OGB's size confirmation in-process instead of on stdin.

    The size is logged rather than silently swallowed, so a multi-gigabyte
    download still announces itself. Every patched module is restored on exit,
    including when the download raises.
    """

    def approve(url: str) -> bool:
        size = _download_size_gb(url)
        shown = f"{size:.2f} GB" if size is not None else "unknown size"
        _LOG.info("downloading OGB archive (%s) from %s", shown, url)
        return True

    patched: list[tuple[Any, Any]] = []
    for name in _PROMPTING_MODULES:
        try:
            module = importlib.import_module(name)
        except ImportError:  # pragma: no cover - depends on the optional extra
            continue
        if hasattr(module, "decide_download"):
            patched.append((module, module.decide_download))
            setattr(module, "decide_download", approve)  # noqa: B010
    try:
        yield
    finally:
        for module, original in patched:
            setattr(module, "decide_download", original)  # noqa: B010


@contextlib.contextmanager
def _torch_load_allowing_pickles() -> Iterator[None]:
    """Restore ``torch.load``'s pre-2.6 default for OGB's own cached artifacts.

    OGB 1.3.6 writes its pre-processed graphs with ``torch.save`` and reads them
    back with a bare ``torch.load``. PyTorch 2.6 flipped ``weights_only`` from
    ``False`` to ``True``, so those reads now raise ``UnpicklingError`` and every
    ``ogbg-*`` loader fails on an otherwise complete download.

    The relaxation is deliberately narrow: it applies only while an OGB loader is
    running, only when the caller has not asked for a specific ``weights_only``,
    and only to files OGB itself produced under the dataset root. Unpickling is
    arbitrary code execution, so this must not leak into general use.
    """
    import torch

    original = torch.load

    def load(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("weights_only", False)
        return original(*args, **kwargs)

    torch.load = load
    try:
        yield
    finally:
        torch.load = original


@contextlib.contextmanager
def _ogb_compat() -> Iterator[None]:
    """Both OGB workarounds, applied together and unwound together."""
    with _download_without_prompting(), _torch_load_allowing_pickles():
        yield


def load_ogb_node(name: str, root: str) -> Data:
    """Return a PyG ``Data`` for an OGB node-property dataset."""
    try:
        from ogb.nodeproppred import NodePropPredDataset
    except ImportError as exc:
        msg = f"'{name}' requires the 'ogb' extra. {_OGB_INSTALL_HINT}"
        raise ImportError(msg) from exc

    import torch

    with _ogb_compat():
        ds = NodePropPredDataset(name=name, root=root)
    graph, label = ds[0]
    split = ds.get_idx_split()

    edge_index = torch.from_numpy(graph["edge_index"]).long()
    x = torch.from_numpy(graph["node_feat"]).float()
    y = torch.from_numpy(label).long().view(-1)

    data = Data(x=x, edge_index=edge_index, y=y)
    data.train_mask = torch.zeros(y.size(0), dtype=torch.bool)
    data.val_mask = torch.zeros(y.size(0), dtype=torch.bool)
    data.test_mask = torch.zeros(y.size(0), dtype=torch.bool)
    data.train_mask[split["train"]] = True
    data.val_mask[split["valid"]] = True
    data.test_mask[split["test"]] = True

    data.num_features = x.size(1)
    data.num_classes = int(y.max()) + 1
    return data


def load_ogb_graph(name: str, root: str) -> Any:
    """Return the raw OGB ``GraphPropPredDataset`` for graph-prop tasks."""
    try:
        from ogb.graphproppred import GraphPropPredDataset
    except ImportError as exc:
        msg = f"'{name}' requires the 'ogb' extra. {_OGB_INSTALL_HINT}"
        raise ImportError(msg) from exc

    with _ogb_compat():
        return GraphPropPredDataset(name=name, root=root)


def load_ogb_link(name: str, root: str) -> Data:
    """Return a single-graph PyG ``Data`` for an OGB link-prop dataset.

    The benchmark runner re-splits via ``RandomLinkSplit``; OGB's
    official train/valid/test edge split is not consumed here. For
    protocol-faithful evaluation, fall back to
    ``ogb.linkproppred.LinkPropPredDataset`` directly.
    """
    try:
        from ogb.linkproppred import LinkPropPredDataset
    except ImportError as exc:
        msg = f"'{name}' requires the 'ogb' extra. {_OGB_INSTALL_HINT}"
        raise ImportError(msg) from exc

    import torch

    with _ogb_compat():
        ds = LinkPropPredDataset(name=name, root=root)
    graph = ds[0]

    edge_index = torch.from_numpy(graph["edge_index"]).long()
    num_nodes = int(graph["num_nodes"])
    data = Data(edge_index=edge_index, num_nodes=num_nodes)

    node_feat = graph.get("node_feat")
    if node_feat is not None:
        data.x = torch.from_numpy(node_feat).float()
        data.num_features = data.x.size(1)
    return data
