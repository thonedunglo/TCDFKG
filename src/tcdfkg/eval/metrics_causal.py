"""
Đánh giá đồ thị nhân quả (TCG) khi có ground-truth:
- Precision/Recall/F1 (directed edges)
- Average Precision (AP) khi dự đoán có score
- Structural Hamming Distance (SHD)
- (tuỳ chọn) dung sai trễ: một cạnh đúng nếu |lag_pred - lag_true| <= tau_tol
"""
from __future__ import annotations
from typing import Dict, Any, Iterable, List, Tuple, Optional
import numpy as np


def _normalize_edges(edges: Iterable, with_score: bool) -> List[Tuple[int,int,float,Optional[int]]]:
    """
    Hỗ trợ các format:
      - [{'src':i,'dst':j,'score':s,'lag':d}, ...]
      - [(i,j), ...]  (with_score=False)
    Trả: list (i, j, score, lag)
    """
    out = []
    for e in edges:
        if isinstance(e, dict):
            i = int(e["src"]); j = int(e["dst"])
            s = float(e.get("score", 1.0))
            d = e.get("lag", None)
            out.append((i, j, s, None if d is None else int(d)))
        else:
            i, j = int(e[0]), int(e[1])
            s = 1.0 if with_score else 1.0
            out.append((i, j, s, None))
    return out


def prf1(
    pred_edges: Iterable,
    true_edges: Iterable,
    threshold: float = 0.5,
    tau_tol: Optional[int] = None,
) -> Dict[str, float]:
    """
    Tính Precision/Recall/F1 dùng ngưỡng score cho pred_edges (nếu có).
    - tau_tol: dung sai trễ (nếu true có lag); None = bỏ qua lag trong matching.
    """
    P = _normalize_edges(pred_edges, with_score=True)
    T = _normalize_edges(true_edges, with_score=False)

    # filter theo threshold
    P_f = [(i, j, s, d) for (i, j, s, d) in P if s >= threshold]

    # set ground truth
    true_set = {(i, j) for (i, j, _, _) in T}
    true_lag = {(i, j): d for (i, j, _, d) in T if d is not None}

    tp = 0
    for (i, j, s, d_pred) in P_f:
        if (i, j) in true_set:
            if tau_tol is not None and (i, j) in true_lag and d_pred is not None:
                if abs(d_pred - true_lag[(i, j)]) <= tau_tol:
                    tp += 1
                else:
                    # lag sai quá mức, coi là FP
                    pass
            else:
                tp += 1

    fp = len(P_f) - tp
    fn = len(true_set) - tp

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def average_precision(
    pred_edges: Iterable,
    true_edges: Iterable,
) -> float:
    """
    Average Precision (AP) trên danh sách cạnh sắp theo score giảm dần.
    """
    P = _normalize_edges(pred_edges, with_score=True)
    T = _normalize_edges(true_edges, with_score=False)
    P_sorted = sorted(P, key=lambda x: x[2], reverse=True)
    true_set = {(i, j) for (i, j, _, _) in T}

    tp = 0
    precisions = []
    for k, (i, j, s, d) in enumerate(P_sorted, start=1):
        if (i, j) in true_set:
            tp += 1
            precisions.append(tp / k)
    if not true_set:
        return 0.0
    return float(np.mean(precisions)) if precisions else 0.0


def shd(
    pred_edges: Iterable,
    true_edges: Iterable,
    undirected: bool = False,
) -> int:
    """
    Structural Hamming Distance:
      - directed: |E_pred Δ E_true|
      - undirected: so sánh trên tập cạnh vô hướng (i,j)==(j,i).
    """
    P = _normalize_edges(pred_edges, with_score=False)
    T = _normalize_edges(true_edges, with_score=False)

    if undirected:
        def und(x): return frozenset((x[0], x[1]))
        set_p = {und((i, j)) for (i, j, _, _) in P}
        set_t = {und((i, j)) for (i, j, _, _) in T}
    else:
        set_p = {(i, j) for (i, j, _, _) in P}
        set_t = {(i, j) for (i, j, _, _) in T}
    return int(len(set_p ^ set_t))
