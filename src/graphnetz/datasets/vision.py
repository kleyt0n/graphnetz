"""Geometry and vision datasets.

Coverage:
- Image-derived superpixel graphs: ``MNISTSuperpixels``, ``CIFAR10`` (GNN benchmark).
- Meshes / point clouds: PyG ``ModelNet`` (10/40 classes).

ShapeNet part segmentation was dropped from the catalogue: its only host,
``shapenet.cs.stanford.edu``, answers neither ICMP nor TCP, upstream PyG still
points at it, and the alternative its maintainers document requires an account.
Rather than advertise a loader that cannot fetch its data, or repoint it at an
unofficial mirror and silently change the dataset's provenance and licence, the
entry is removed. Restoring it means adding a loader whose source can be
downloaded.
"""

from torch_geometric import transforms as T
from torch_geometric.data import Data
from torch_geometric.datasets import GNNBenchmarkDataset, MNISTSuperpixels, ModelNet
from torch_geometric.transforms import BaseTransform


def mnist_superpixels(root: str, train: bool = True) -> MNISTSuperpixels:
    """MNIST images converted to 75-superpixel graphs."""
    return MNISTSuperpixels(root=root, train=train)


def cifar10_superpixels(root: str, split: str = "train") -> GNNBenchmarkDataset:
    """CIFAR10 superpixel graphs (GNN benchmark suite)."""
    return GNNBenchmarkDataset(root=root, name="CIFAR10", split=split)


class _PositionsAsFeatures(BaseTransform):
    """Use the vertex coordinates as node features (``x = pos``)."""

    def forward(self, data: Data) -> Data:
        data.x = data.pos
        return data


def _mesh_to_graph() -> T.Compose:
    """Turn a ModelNet triangle mesh into a graph a GNN can consume.

    PyG's ``ModelNet`` stores each shape as vertices (``pos``) and triangles
    (``face``) with no ``edge_index`` and no ``x``, so message passing over the
    raw objects sees no edges and has no features. The mesh edges become an
    undirected ``edge_index`` and the coordinates become ``x``. Applied as a
    ``transform`` (on access), not a ``pre_transform``, so an existing
    processed cache stays valid.
    """
    return T.Compose([T.FaceToEdge(remove_faces=True), _PositionsAsFeatures()])


def modelnet10(root: str, train: bool = True) -> ModelNet:
    """ModelNet10 3D shapes (10 classes), as mesh-edge graphs with ``x = pos``."""
    return ModelNet(root=root, name="10", train=train, transform=_mesh_to_graph())


def modelnet40(root: str, train: bool = True) -> ModelNet:
    """ModelNet40 3D shapes (40 classes), as mesh-edge graphs with ``x = pos``."""
    return ModelNet(root=root, name="40", train=train, transform=_mesh_to_graph())


__all__ = [
    "cifar10_superpixels",
    "mnist_superpixels",
    "modelnet10",
    "modelnet40",
]
