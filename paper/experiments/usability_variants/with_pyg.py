"""The same report -- with PyTorch Geometric and scipy directly.

A faithful, runnable equivalent of ``with_graphnetz.py``: the same three
Planetoid tasks, the same four architectures, the same ten seeds, and the same
three outputs (per-cell 95% t-CI, Holm-corrected paired tests within each task,
Demsar critical-difference diagram across tasks).

This is not a strawman. Nothing here is gratuitous: every block is work the
user must do because the layer below does not do it. The comment markers name
the statistical decision each block encodes, because the count that matters is
not lines but *decisions a user can get wrong*.
"""

import itertools
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from scipy.stats import studentized_range
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, TransformerConv

# --- DECISION: architecture depth, width, activation, dropout placement -----
CONVS = {"GCN": GCNConv, "GAT": GATConv, "GraphSAGE": SAGEConv, "GraphTransformer": TransformerConv}


class Net(torch.nn.Module):
    def __init__(self, conv, in_channels, hidden, out_channels):
        super().__init__()
        self.c1 = conv(in_channels, hidden)
        self.c2 = conv(hidden, out_channels)

    def forward(self, data):
        x = F.relu(self.c1(data.x, data.edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.c2(x, data.edge_index)


# --- DECISION: which RNGs to reseed, and when (before every run, not once) ---
def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --- DECISION: optimiser, lr, weight decay, epochs, which metric to record --
def train_once(name, dataset, seed, epochs=100):
    seed_all(seed)
    data = dataset[0]
    model = Net(CONVS[name], dataset.num_features, 64, dataset.num_classes)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(data)
        F.cross_entropy(out[data.train_mask], data.y[data.train_mask]).backward()
        opt.step()
    # --- DECISION: report at the final epoch, or select a checkpoint? -------
    model.eval()
    with torch.no_grad():
        pred = model(data).argmax(dim=1)
    mask = data.test_mask
    return float((pred[mask] == data.y[mask]).float().mean())


TASKS = {"cora": "Cora", "citeseer": "CiteSeer", "pubmed": "PubMed"}
SEEDS = range(10)

# --- DECISION: the loop order that keeps seeds paired across models ---------
finals = {}
for task, pyg_name in TASKS.items():
    dataset = Planetoid(root=f"data/usability/{task}", name=pyg_name)
    finals[task] = {name: [train_once(name, dataset, s) for s in SEEDS] for name in CONVS}

# --- DECISION: t-CI vs normal-approximation vs bootstrap; ddof=1 ------------
summary = {}
for task, per_task in finals.items():
    for name, values in per_task.items():
        arr = np.asarray(values, dtype=float)
        half = float(stats.sem(arr) * stats.t.ppf(0.975, arr.size - 1))
        summary[task, name] = (float(arr.mean()), half)

# --- DECISION: paired vs unpaired test; Holm vs Bonferroni vs none;
# --- DECISION: family = within-task (6 tests) vs across all tasks (18) ------
pairwise = []
for task, per_task in finals.items():
    names = sorted(per_task)
    pairs, raw = [], []
    for a, b in itertools.combinations(names, 2):
        x, y = np.asarray(per_task[a]), np.asarray(per_task[b])
        pairs.append((a, b, float(x.mean() - y.mean())))
        raw.append(float(stats.ttest_rel(x, y).pvalue))
    # Holm step-down, with the running max that makes it monotone
    order = np.argsort(raw)
    m, running, adj = len(raw), 0.0, [0.0] * len(raw)
    for rank, idx in enumerate(order):
        running = max(running, min(raw[idx] * (m - rank), 1.0))
        adj[idx] = running
    for (a, b, diff), p_raw, p_adj in zip(pairs, raw, adj, strict=False):
        pairwise.append((task, a, b, diff, p_raw, p_adj, p_adj < 0.05))

# --- DECISION: rank direction per task (higher accuracy = rank 1) ----------
# --- DECISION: average ranks for ties, and drop models absent from a task ---
models = sorted(set.intersection(*[set(p) for p in finals.values()]))
ranks = np.array([stats.rankdata([-float(np.mean(finals[t][m])) for m in models], method="average") for t in finals])
avg = ranks.mean(axis=0)
k, n = len(models), len(finals)

# --- DECISION: Friedman before Nemenyi; the q_alpha / sqrt(2) conversion ----
chi2 = (12.0 * n) / (k * (k + 1)) * (float(np.sum(avg**2)) - k * (k + 1) ** 2 / 4.0)
friedman_p = float(stats.chi2.sf(chi2, df=k - 1))
q = float(studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2))
cd = q * float(np.sqrt(k * (k + 1) / (6 * n)))

# --- DECISION: clique construction -- maximal contiguous runs within CD -----
order = np.argsort(avg)
sorted_models = [models[i] for i in order]
sorted_ranks = avg[order]
raw_cliques = []
i = 0
while i < k:
    j = i
    while j + 1 < k and sorted_ranks[j + 1] - sorted_ranks[i] < cd:
        j += 1
    if j > i:
        raw_cliques.append((i, j))
    i += 1
cliques = []
for a, b in sorted(set(raw_cliques)):
    if any(c <= a and b <= d for c, d in cliques):
        continue
    cliques = [(c, d) for c, d in cliques if not (a <= c and d <= b)]
    cliques.append((a, b))

# --- the CD diagram itself: axis, leaders, staggered labels, clique bars ----
fig, ax = plt.subplots(figsize=(10, max(2.2, 1.6 + 0.22 * k)))
ax.plot([1.0, float(k)], [0, 0], color="black", linewidth=0.8)
for r in range(1, k + 1):
    ax.plot([r, r], [0, -0.04], color="black", linewidth=0.6)
    ax.text(r, -0.08, str(r), ha="center", va="top", fontsize=8)
half_k = (k + 1) // 2
for idx, (model, r) in enumerate(zip(sorted_models, sorted_ranks, strict=False)):
    left = idx < half_k
    ly = 0.32 + ((half_k - idx - 1) if left else (idx - half_k)) * 0.16
    lx = 0.5 if left else float(k) + 0.5
    ax.plot([r, r], [0, ly], color="0.55", linewidth=0.5)
    ax.plot([r, lx], [ly, ly], color="0.55", linewidth=0.5)
    ax.plot([r], [0], marker="o", markersize=5.0, color="black")
    ax.text(lx, ly, f"{model} ({r:.2f})", va="center", ha="right" if left else "left", fontsize=9)
bar_y = -0.16
for a, b in cliques:
    ax.plot([sorted_ranks[a] - 0.06, sorted_ranks[b] + 0.06], [bar_y, bar_y], color="black", linewidth=3.5)
    bar_y -= 0.06
cd_y = 0.32 + max(half_k - 1, 0) * 0.16 + 0.22
ax.plot([1.0, 1.0 + cd], [cd_y, cd_y], color="black", linewidth=1.0)
ax.text(1.0 + cd / 2, cd_y + 0.04, f"CD = {cd:.3f}", ha="center", va="bottom", fontsize=8)
ax.text(1.0 + cd / 2, cd_y + 0.18, f"Friedman p = {friedman_p:.3g}", ha="center", va="bottom", fontsize=7)
ax.set_xlim(-0.7, float(k) + 1.7)
ax.set_ylim(bar_y - 0.16, cd_y + 0.3)
ax.axis("off")
fig.tight_layout()
fig.savefig("pyg_cd.png", dpi=600, bbox_inches="tight")
