"""Check that catalogued loaders name data that actually exists upstream.

Five loaders shipped asking Netzschleuder for a network name the catalogue does
not publish -- ``product_space/product_space``, ``karate/karate``,
``sp_high_school/sp_high_school``, ``budapest_connectome/100m_avg`` and
``route_views/20030701``. Every one returned HTTP 404 on first use, and the
audit initially reported them as dead mirrors rather than as our own bugs.

The failure was invisible to unit tests because the name is only wrong *upstream*:
nothing local contradicts it. So this check asks the catalogue. It needs the
network and is therefore deselected by default; run it with ``-m network``, and
in particular before claiming the catalogue is validated.
"""

from __future__ import annotations

import inspect
import re

import pytest


def _declared_netz_pairs() -> dict[str, tuple[str, str | None]]:
    """Map loader name -> (dataset_name, network_name) for every Netz wrapper.

    Read statically from each loader's source, so the pairs are collected without
    instantiating anything or touching the network.
    """
    from graphnetz.datasets import LOADER_REGISTRY

    pairs: dict[str, tuple[str, str | None]] = {}
    for per_category in LOADER_REGISTRY.values():
        for entries in per_category.values():
            for name, fn in entries:
                if name in pairs:
                    continue
                try:
                    src = inspect.getsource(fn)
                except (OSError, TypeError):  # pragma: no cover - builtins
                    continue
                if "Netz(" not in src:
                    continue
                dataset = re.search(r'dataset_name="([^"]+)"', src)
                network = re.search(r'network_name[:=] *(?:str *= *)?"([^"]+)"', src)
                if dataset:
                    pairs[name] = (dataset.group(1), network.group(1) if network else None)
    return pairs


@pytest.mark.network
def test_every_netzschleuder_loader_names_a_real_network() -> None:
    requests = pytest.importorskip("requests")

    from graphnetz.datasets._netz import NETZ_API

    pairs = _declared_netz_pairs()
    assert pairs, "no Netzschleuder loaders found -- has the registry moved?"

    # A 404 means the name is wrong; a timeout means we could not ask. Conflating
    # the two is the exact mistake the loader audit made in the other direction,
    # and here it would fail the build for a slow mirror -- so an unreachable API
    # skips rather than accuses.
    catalogue: dict[str, list[str]] = {}
    wrong: list[str] = []
    unreachable: list[str] = []
    for loader, (dataset, network) in sorted(pairs.items()):
        if dataset not in catalogue and dataset not in unreachable:
            try:
                response = requests.get(f"{NETZ_API}/{dataset}", timeout=60)
            except requests.RequestException as exc:
                unreachable.append(f"{dataset} ({type(exc).__name__})")
                continue
            if response.status_code == 404:
                wrong.append(f"{loader}: dataset {dataset!r} returns 404")
                continue
            if not response.ok:
                unreachable.append(f"{dataset} (HTTP {response.status_code})")
                continue
            try:
                catalogue[dataset] = list(response.json().get("nets", []))
            except ValueError:
                # A non-JSON body from a live server means the dataset path is
                # not a dataset -- which is how `sp_high_school` was found.
                wrong.append(f"{loader}: dataset {dataset!r} did not return a catalogue entry")
                continue
        if dataset in catalogue and network not in catalogue[dataset]:
            wrong.append(f"{loader}: {dataset}/{network!r} not among {catalogue[dataset][:8]}")

    assert not wrong, "loaders naming data that does not exist:\n  " + "\n  ".join(wrong)
    if unreachable:
        pytest.skip("catalogue API unreachable for: " + ", ".join(sorted(set(unreachable))))
