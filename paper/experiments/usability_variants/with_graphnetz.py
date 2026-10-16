"""Multi-seed, Holm-corrected, CD-diagrammed report -- with GraphNetz.

Produces: per-cell mean +- 95% t-CI, Holm-corrected paired tests within each
task, and a Demsar critical-difference diagram across tasks.

Counterpart: ``with_pyg.py`` does the same thing with PyTorch Geometric and
scipy directly. Both are runnable; ``usability.py`` counts their effective
lines of user code.
"""

from graphnetz import GAT, GCN, GraphSAGE, GraphTransformer, run_benchmark, save_figure
from graphnetz.benchmark import BENCHMARK_TASKS

TASKS = [t for t in BENCHMARK_TASKS["social"]["node_cls"] if t.name in ("cora", "citeseer", "pubmed")]
MODELS = {"GCN": GCN, "GAT": GAT, "GraphSAGE": GraphSAGE, "GraphTransformer": GraphTransformer}

report = run_benchmark("social", MODELS, tasks=TASKS, seeds=range(10), root="data/usability")

summary = report.summary(ci=0.95)
pairwise = report.pairwise(alpha=0.05)
fig, _ = report.plot_critical_difference(alpha=0.05)
save_figure(fig, "graphnetz_cd")
