from __future__ import annotations
from typing import Dict, Any, Optional, List, Sequence
import numpy as np
import matplotlib.pyplot as plt

from .utils import ensure_dir


def plot_propagation_timeline(
    prop: Dict[str, Any],
    out_path: str,
    topk_per_delta: int = 10,
    id2name: Optional[Sequence[str]] = None,
) -> None:
    """
    Vẽ xác suất lan truyền theo từng Δ (delta). Ở mỗi Δ, hiển thị top-k nút có p cao nhất (barh).
    """
    tl = prop.get("timeline", [])
    if not tl:
        raise ValueError("Propagation timeline is empty.")

    ncols = 1
    nrows = len(tl)
    nrows = min(nrows, 10)  # tránh quá dài; nếu nhiều hơn, vẽ 10 delta đầu
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(9, max(2.5, 2.0 * nrows)))
    if nrows == 1:
        axes = [axes]

    for ax, step in zip(axes, tl[:nrows]):
        delta = step["delta"]
        nodes = step["nodes"]
        # sort theo p giảm dần và cắt top-k
        nodes = sorted(nodes, key=lambda x: x["p"], reverse=True)[:topk_per_delta]
        if not nodes:
            continue
        ids = [n["id"] for n in nodes]
        ps = [n["p"] for n in nodes]
        labels = [id2name[i] if (id2name and i < len(id2name)) else f"var_{i}" for i in ids]
        ax.barh(range(len(ids)), ps)
        ax.set_yticks(range(len(ids)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_title(f"Δ = {delta}")
        ax.set_xlabel("probability")
        ax.grid(True, axis="x", linewidth=0.3, alpha=0.6)

    fig.suptitle("Propagation timeline (top-k per delta)", y=1.02, fontsize=12)
    fig.tight_layout()
    ensure_dir(out_path)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
