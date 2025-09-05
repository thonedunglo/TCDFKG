import math
import json
import heapq
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np


def _build_out_edges(tcg: Dict) -> Tuple[Dict[int, List[Tuple[int, float, int]]], int]:
    """
    Trả về:
      - out_edges[i] = [(j, S_ij, d_ij), ...]
      - N: số biến (suy ra từ id lớn nhất trong tcg)
    """
    out = defaultdict(list)
    max_id = -1
    for e in tcg.get("edges", []):
        i = int(e["src"]); j = int(e["dst"])
        s = float(e["score"]); d = int(e.get("lag", 0))
        out[i].append((j, s, d))
        max_id = max(max_id, i, j)
    return out, max_id + 1


def _noisy_or(p_old: float, p_add: float) -> float:
    """p_new = 1 - (1 - p_old) * (1 - p_add)"""
    p_old = max(0.0, min(1.0, p_old))
    p_add = max(0.0, min(1.0, p_add))
    return 1.0 - (1.0 - p_old) * (1.0 - p_add)


def propagate(
    tcg: Dict,
    A_t: np.ndarray,
    lam: float = 0.1,
    L: int = 3,
    max_delta: Optional[int] = None,
    min_score: float = 0.0,
    eps_push: float = 1e-6,
) -> Dict:
    """
    Lan truyền xác suất bất thường từ vector gốc A_t (N,), theo TCG.
    p_j,t+Δ từ i:  p = A_t[i] * S_ij * exp(-lam * d_ij)
    Gộp nhiều nguồn về j tại cùng Δ bằng noisy-OR.
    Sau đó tiếp tục lan thêm ≤ L bậc (hop) với cùng công thức.

    Args:
      tcg: {'edges': [{'src','dst','lag','score'}, ...]}
      A_t: (N,) ∈ [0,1], thường là nhãn bất thường ở t (0/1) hoặc xác suất gốc.
      lam: hệ số suy giảm theo trễ
      L:   số bậc lan truyền tối đa (hop count)
      max_delta: giới hạn thời gian tương lai (tính bằng số bước), None = không giới hạn
      min_score: bỏ qua cạnh có score < min_score
      eps_push: chỉ mở rộng khi p mới tăng đáng kể (> eps_push)

    Returns:
      {
        "N": N,
        "timeline": [ {"delta": Δ, "nodes": [{"id": j, "p": p_jΔ}, ...]}, ... ],
        "by_node": { j: [{"delta": Δ, "p": ...}, ...] }  # tiện tra cứu
      }
    """
    out_edges, N = _build_out_edges(tcg)
    assert A_t.shape[0] == N, f"A_t shape {A_t.shape} != N={N}"

    # Ưu tiên theo thời gian đến (delta nhỏ xử lý trước)
    # hàng đợi phần tử: (arrival_delta, hop, node_id, prob_at_arrival)
    pq: List[Tuple[int, int, int, float]] = []

    # Xác suất theo timeline: timeline[delta][j] = prob
    timeline: Dict[int, Dict[int, float]] = defaultdict(lambda: defaultdict(float))

    # Khởi tạo từ các hạt giống (seed) tại delta=0
    seeds = np.where(A_t > 0)[0].tolist()
    for i in seeds:
        p0 = float(A_t[i])
        if p0 <= 0: 
            continue
        # ghi xác suất seed tại Δ=0
        timeline[0][i] = _noisy_or(timeline[0][i], p0)
        heapq.heappush(pq, (0, 0, i, p0))

    # Duyệt lan truyền
    while pq:
        cur_delta, hop, u, p_u = heapq.heappop(pq)
        if hop >= L:
            continue
        # mở rộng theo các cạnh đi ra từ u
        for (v, s, d) in out_edges.get(u, []):
            if s < min_score:
                continue
            arr = cur_delta + int(max(0, d))
            if max_delta is not None and arr > max_delta:
                continue
            p_add = p_u * s * math.exp(-lam * max(0, d))
            if p_add <= 0:
                continue
            p_old = timeline[arr][v]
            p_new = _noisy_or(p_old, p_add)
            if p_new - p_old > eps_push:
                timeline[arr][v] = p_new
                # đẩy tiếp sóng lan truyền từ v ở thời điểm arr
                heapq.heappush(pq, (arr, hop + 1, v, p_new))

    # Chuẩn hoá output
    deltas = sorted(timeline.keys())
    timeline_arr = []
    by_node = defaultdict(list)
    for dlt in deltas:
        nodes = [{"id": int(j), "p": float(p)} for j, p in sorted(timeline[dlt].items()) if p > 0]
        if not nodes:
            continue
        timeline_arr.append({"delta": int(dlt), "nodes": nodes})
        for item in nodes:
            by_node[item["id"]].append({"delta": int(dlt), "p": float(item["p"])})

    return {"N": N, "timeline": timeline_arr, "by_node": dict(by_node)}
