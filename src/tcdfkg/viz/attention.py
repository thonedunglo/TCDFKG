from __future__ import annotations
from typing import Optional, Sequence
import numpy as np
import matplotlib.pyplot as plt

from .utils import ensure_dir, default_figsize


def plot_attention_heatmap(
    M: np.ndarray,
    out_path: str,
    title: str = "Attention / Adjacency",
    xlabels: Optional[Sequence[str]] = None,
    ylabels: Optional[Sequence[str]] = None,
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> None:
    """
    Vẽ heatmap N x N (ví dụ: S_att hoặc adjacency theo score/lag).
    Nếu không có S_att, bạn có thể xây ma trận từ tcg.json:
      A[i,j] = score(i->j) (hoặc normalize về [0,1]).
    """
    assert M.ndim == 2 and M.shape[0] == M.shape[1], "Matrix must be square."
    N = M.shape[0]
    fig = plt.figure(figsize=default_figsize(N, base=(7, 6)))
    ax = fig.add_subplot(111)
    im = ax.imshow(M, vmin=vmin, vmax=vmax, aspect="auto", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("dst (j)"); ax.set_ylabel("src (i)")

    if xlabels is not None and len(xlabels) == N and N <= 60:
        ax.set_xticks(np.arange(N)); ax.set_xticklabels(xlabels, rotation=90, fontsize=6)
    if ylabels is not None and len(ylabels) == N and N <= 60:
        ax.set_yticks(np.arange(N)); ax.set_yticklabels(ylabels, fontsize=6)

    fig.colorbar(im, ax=ax, shrink=0.85)
    ensure_dir(out_path)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def tcg_to_adj_matrix(tcg: dict, N: Optional[int] = None, normalize: bool = True) -> np.ndarray:
    """
    Chuyển tcg (edge list) -> ma trận N x N với giá trị = score.
    Nếu normalize=True: scale về [0,1].
    """
    if N is None:
        N = 0
        for e in tcg.get("edges", []):
            N = max(N, int(e["src"]), int(e["dst"]))
        N += 1
    A = np.zeros((N, N), dtype=float)
    scores = []
    for e in tcg.get("edges", []):
        i, j = int(e["src"]), int(e["dst"])
        s = float(e.get("score", 0.0))
        A[i, j] = s
        scores.append(s)
    if normalize and len(scores) > 0:
        smin, smax = float(min(scores)), float(max(scores))
        if smax > smin:
            A = (A - smin) / (smax - smin + 1e-12)
    return A
