"""Computing and systems networks (Netzschleuder).

Internet topology, autonomous-system graphs, and routing snapshots.
"""

from graphnetz.datasets._netz import Netz


def internet_as(root: str, network_name: str = "internet_as") -> Netz:
    """Internet AS-level topology snapshot (Karrer-Newman-Zdeborová, 2014)."""
    return Netz(root=root, dataset_name="internet_as", network_name=network_name)


def as_skitter(root: str) -> Netz:
    """CAIDA Skitter AS-level network."""
    return Netz(root=root, dataset_name="as_skitter", network_name="as_skitter")


def topology(root: str) -> Netz:
    """Internet router-level topology."""
    return Netz(root=root, dataset_name="topology", network_name="topology")


def route_views(root: str, network_name: str = "20000102") -> Netz:
    """Route Views BGP snapshot.

    The catalogue holds 733 daily snapshots spanning 1997-11-08 to 2000-01-02,
    named by date; the default is the last one (6,474 ASes, 13,895 links). The
    previous default, ``20030701``, is outside that range and always 404'd.
    """
    return Netz(root=root, dataset_name="route_views", network_name=network_name)


__all__ = ["as_skitter", "internet_as", "route_views", "topology"]
