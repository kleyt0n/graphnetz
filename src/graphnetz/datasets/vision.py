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

from torch_geometric.datasets import GNNBenchmarkDataset, MNISTSuperpixels, ModelNet


def mnist_superpixels(root: str, train: bool = True) -> MNISTSuperpixels:
    """MNIST images converted to 75-superpixel graphs."""
    return MNISTSuperpixels(root=root, train=train)


def cifar10_superpixels(root: str, split: str = "train") -> GNNBenchmarkDataset:
    """CIFAR10 superpixel graphs (GNN benchmark suite)."""
    return GNNBenchmarkDataset(root=root, name="CIFAR10", split=split)


def modelnet10(root: str, train: bool = True) -> ModelNet:
    """ModelNet10 3D shapes (10 classes)."""
    return ModelNet(root=root, name="10", train=train)


def modelnet40(root: str, train: bool = True) -> ModelNet:
    """ModelNet40 3D shapes (40 classes)."""
    return ModelNet(root=root, name="40", train=train)


__all__ = [
    "cifar10_superpixels",
    "mnist_superpixels",
    "modelnet10",
    "modelnet40",
]
