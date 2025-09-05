"""
GraphStore: lớp tiện ích lưu/đọc đồ thị (schema graph & TCG), truy vấn đường đi nhanh
với cache khoảng cách, và chuyển đổi qua lại giữa JSON & NetworkX.
"""
from __future__ import annotations
from typing import Dict, Any, Tuple, Optional
import json
import networkx as nx


class GraphStore:
    def __init__(self):
        self.schema_graph: Optional[nx.DiGraph] = None
        self.tcg_graph: Optional[nx.DiGraph] = None
        self._dist_cache: Dict[Tuple[str, str], int] = {}

    # ---------------- Schema Graph ----------------
    def set_schema_graph(self, G: nx.DiGraph):
        self.schema_graph = G
        self._dist_cache.clear()

    def shortest_path_length(self, u: str, v: str) -> Optional[int]:
        """
        Khoảng cách ngắn nhất trong schema_graph (edge weight=1).
        Trả None nếu không có đường.
        """
        if self.schema_graph is None:
            return None
        key = (u, v)
        if key in self._dist_cache:
            return self._dist_cache[key]
        try:
            d = nx.shortest_path_length(self.schema_graph, u, v)
            self._dist_cache[key] = int(d)
            return int(d)
        except nx.NetworkXNoPath:
            self._dist_cache[key] = None  # type: ignore
            return None

    # ---------------- TCG Graph ----------------
    def set_tcg_from_json(self, tcg: Dict[str, Any]):
        """
        tcg = {'edges': [{'src','dst','lag','score'}, ...]}
        """
        G = nx.DiGraph()
        for e in tcg.get("edges", []):
            i = int(e["src"]); j = int(e["dst"])
            G.add_edge(i, j, score=float(e.get("score", 0.0)), lag=int(e.get("lag", 0)))
        self.tcg_graph = G

    def to_tcg_json(self) -> Dict[str, Any]:
        assert self.tcg_graph is not None, "No TCG graph set."
        edges = []
        for u, v, d in self.tcg_graph.edges(data=True):
            edges.append({"src": int(u), "dst": int(v), "lag": int(d.get("lag", 0)), "score": float(d.get("score", 0.0))})
        return {"edges": edges}

    # ---------------- IO helpers ----------------
    @staticmethod
    def save_json(obj: Dict[str, Any], path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load_json(path: str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
