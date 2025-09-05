from __future__ import annotations
import math
from typing import Dict, Any, Iterable, Optional, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from tcdfkg.causality.tcg import load_tcg
from .utils import ensure_dir, default_figsize


def _tcg_to_nx(tcg: Dict[str, Any]) -> nx.DiGraph:
    G = nx.DiGraph()
    for e in tcg.get("edges", []):
        i, j = int(e["src"]), int(e["dst"])
        G.add_node(i); G.add_node(j)
        G.add_edge(i, j, score=float(e.get("score", 0.0)), lag=int(e.get("lag", 0)))
    return G


def plot_tcg(
    tcg: Dict[str, Any] | str,
    out_path: str,
    top_edges: Optional[int] = 200,
    node_labels: Optional[List[str]] = None,
    layout: str = "kamada_kawai",
    seed: int = 42,
) -> None:
    """
    Vẽ đồ thị TCG: màu & độ dày ∝ score, nhãn cạnh = lag.
    - tcg: dict hoặc path tới tcg.json
    - top_edges: vẽ tối đa bao nhiêu cạnh có score cao nhất (None = tất cả)
    - node_labels: danh sách tên cột (để hiển thị label nút)
    - layout: 'spring' | 'kamada_kawai'
    """
    tcg_obj = load_tcg(tcg) if isinstance(tcg, str) else tcg
    G = _tcg_to_nx(tcg_obj)
    if G.number_of_edges() == 0:
        raise ValueError("TCG has no edges to plot.")

    # chọn top-k cạnh theo score
    E = [(u, v, d["score"]) for u, v, d in G.edges(data=True)]
    E = sorted(E, key=lambda x: x[2], reverse=True)
    if top_edges is not None:
        keep = set((u, v) for (u, v, _) in E[:top_edges])
        G = G.edge_subgraph(keep).copy()

    # layout
    rng = np.random.default_rng(seed)
    if layout == "spring":
        pos = nx.spring_layout(G, seed=seed, k=1 / math.sqrt(max(1, G.number_of_nodes())))
    else:
        pos = nx.kamada_kawai_layout(G)

    # node labels
    lab_map = {}
    if node_labels is not None:
        for i, name in enumerate(node_labels):
            lab_map[i] = str(name)

    # style từ score
    scores = np.array([d["score"] for _, _, d in G.edges(data=True)], dtype=float)
    if scores.size == 0:
        scores = np.array([0.0])
    smin, smax = float(scores.min()), float(scores.max())
    norm = (scores - smin) / (smax - smin + 1e-12)
    widths = 0.5 + 2.5 * norm
    cmap = plt.cm.viridis

    # vẽ
    plt.figure(figsize=default_figsize(G.number_of_nodes(), base=(10, 8)))
    nx.draw_networkx_nodes(G, pos, node_size=200, node_color="#DDE8F8", linewidths=0.5, edgecolors="#4878CF")
    nx.draw_networkx_edges(
        G, pos,
        edge_color=cmap(norm),
        width=widths,
        arrowsize=12,
        alpha=0.9,
    )
    # nhãn nút
    if node_labels is None and G.number_of_nodes() <= 50:
        nx.draw_networkx_labels(G, pos, font_size=8)
    elif node_labels is not None and G.number_of_nodes() <= 100:
        nx.draw_networkx_labels(G, pos, labels=lab_map, font_size=7)

    # nhãn lag trên cạnh (chỉ khi đồ thị nhỏ)
    if G.number_of_edges() <= 80:
        edge_labels = {(u, v): d.get("lag", 0) for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)

    plt.axis("off")
    ensure_dir(out_path)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
