# Installation

This page takes you from a clean environment to a working GraphNetz install.
It assumes familiarity with PyTorch and PyG; everything else is covered as
we go.

## Install

```bash
uv add graphnetz
# or, in an existing environment:
pip install graphnetz
```

Requires **Python ≥ 3.10**, **PyTorch ≥ 2.6**, and **torch-geometric ≥ 2.6**.

## Optional extras

| Extra | Install | Unlocks |
|---|---|---|
| `ogb` | `pip install graphnetz[ogb]` | OGB loaders (`ogbn-arxiv`, `ogbl-collab`, `ogbn-products`, `ogbg-molhiv`, `ogbg-molpcba`) |
| `chem` | `pip install graphnetz[chem]` | RDKit — required by OGB molecular loaders such as `ogbg-molhiv` |

## Development install

For local development, clone the repo and use the `dev` group:

```bash
git clone https://github.com/Kleyt0n/graphnetz
cd graphnetz
uv sync --group dev
```

## Verify

```python
from graphnetz import GCN, train_node_classification
from graphnetz.datasets.social import cora

ds = cora("data/cora")
model = GCN(ds.num_features, 64, ds.num_classes)
history = train_node_classification(model, ds[0], epochs=10)
```

!!! tip
    **GPU is automatic.** Both the standalone trainers and `run_benchmark`
    accept `device='auto'` (the default). The runtime picks CUDA when
    available, then Apple-silicon MPS, then CPU, and moves the model and
    data onto it for you. Pin placement explicitly with `device='cpu'` (or
    any `torch.device`) when you need to.

Next: the [Quickstart](quickstart.md) runs your first multi-seed benchmark.
