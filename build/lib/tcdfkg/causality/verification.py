import json, networkx as nx
from typing import Dict, Any, Tuple
import math

class KGVerifier:
    """
    Tính S_KG(i->j) = α S_struct + β S_type + δ S_rule.
    Parsed theo format schema đã trích xuất (machines/components/feature_types/rules/events). 
    """
    def __init__(self, schema_path: str, alpha=0.5, beta=0.3, delta=0.2, kappa=0.7, rho_rule=0.2):
        self.alpha, self.beta, self.delta, self.kappa, self.rho_rule = alpha, beta, delta, kappa, rho_rule
        with open(schema_path, "r", encoding="utf-8") as f:
            self.schema = json.load(f)
        self.G = self._build_graph(self.schema)

    def _build_graph(self, s: Dict[str, Any]) -> nx.DiGraph:
        G = nx.DiGraph()
        # Nodes
        for m in s.get("machines", []):
            for c in m.get("components", []):
                cname = c.get("name"); 
                if cname: G.add_node(f"Component::{cname}")
            for ft in m.get("feature_types", []):
                fname = ft.get("name"); 
                if fname: G.add_node(f"Feature::{fname}")
            # weak links component->feature (đồng máy)
            for c in m.get("components", []):
                for ft in m.get("feature_types", []):
                    if c.get("name") and ft.get("name"):
                        G.add_edge(f"Component::{c['name']}", f"Feature::{ft['name']}")
        for r in s.get("rules", []):
            rname = r.get("name"); 
            if rname: G.add_node(f"Rule::{rname}")
            for lg in r.get("logic", []):
                comp = lg.get("of_component"); ftype = lg.get("feature_type")
                if comp: G.add_node(f"Component::{comp}")
                if ftype: G.add_node(f"Feature::{ftype}")
                if comp and ftype:
                    G.add_edge(f"Rule::{rname}", f"Feature::{ftype}")
                    G.add_edge(f"Component::{comp}", f"Feature::{ftype}")
        for ev in s.get("events", []):
            rule_ref = ev.get("rule_ref")
            if rule_ref:
                G.add_node(f"Rule::{rule_ref}")
                for src in ev.get("sources", []):
                    ft = src.get("feature_type")
                    if ft:
                        G.add_node(f"Feature::{ft}")
                        G.add_edge(f"Rule::{rule_ref}", f"Feature::{ft}")
        return G

    def _s_struct(self, type_i: str, type_j: str) -> float:
        u, v = f"Feature::{type_i}", f"Feature::{type_j}"
        if u not in self.G or v not in self.G: return 0.0
        try:
            d = nx.shortest_path_length(self.G, u, v)
            return math.exp(-self.kappa * d)
        except nx.NetworkXNoPath:
            return 0.0

    def _s_type(self, type_i: str, type_j: str) -> float:
        if not type_i or not type_j: return 0.5
        return 1.0 if type_i == type_j else 0.5

    def _s_rule(self, type_i: str, type_j: str) -> float:
        Ui, Uj = f"Feature::{type_i}", f"Feature::{type_j}"
        rules_i = {n for n in self.G.predecessors(Ui) if str(n).startswith("Rule::")} if Ui in self.G else set()
        rules_j = {n for n in self.G.predecessors(Uj) if str(n).startswith("Rule::")} if Uj in self.G else set()
        if rules_i & rules_j:  # cùng rule nhưng không chỉ hướng rõ
            return self.rho_rule
        return 0.0

    def score(self, meta: Dict[str, Any]) -> Dict[Tuple[int,int], float]:
        out = {}
        types = meta.get("type", {})
        for (i, j) in meta.get("candidates", []):
            ti, tj = types.get(i, ""), types.get(j, "")
            s = ( self.alpha * self._s_struct(ti, tj)
                + self.beta  * self._s_type(ti, tj)
                + self.delta * self._s_rule(ti, tj) )
            out[(i, j)] = float(s)
        return out
