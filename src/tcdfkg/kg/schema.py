"""
Schema loader & typed-graph builder.

Đọc extracted_schema.json và dựng đồ thị KG dị chủng (Feature/Component/Rule/Event).
Cung cấp API truy vấn type, liệt kê nodes/edges, và tạo NetworkX DiGraph sẵn dùng.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import json
import networkx as nx


# --------- Dataclasses dành cho tài liệu & introspection (tuỳ chọn dùng) ---------
@dataclass
class FeatureType:
    name: str

@dataclass
class Component:
    name: str

@dataclass
class RuleLogic:
    of_component: Optional[str] = None
    feature_type: Optional[str] = None
    # có thể mở rộng: condition, threshold, operator,...

@dataclass
class Rule:
    name: str
    logic: List[RuleLogic]

@dataclass
class EventSource:
    feature_type: Optional[str] = None
    # có thể mở rộng: sensor, bounds,...

@dataclass
class Event:
    description: Optional[str] = None
    rule_ref: Optional[str] = None
    sources: List[EventSource] = None


# --------------------------------- Schema wrapper ---------------------------------
class Schema:
    """
    Wrapper quanh JSON schema đã trích xuất:
      {
        "machines": [{"name":..., "components":[...], "feature_types":[...]}],
        "rules": [{"name":..., "logic":[{of_component, feature_type}, ...]}],
        "events": [{"description":..., "rule_ref":..., "sources":[{feature_type}, ...]}]
      }

    Giao diện chính:
      - Schema.from_json(path)
      - build_graph(): nx.DiGraph với node kiểu: "Feature::X", "Component::Y", "Rule::Z", "Event::K"
      - list_feature_types(), list_components(), list_rules(), list_events()
      - has_feature_type(name), has_component(name)
    """
    def __init__(self, raw: Dict[str, Any]):
        self.raw = raw or {}
        self._G: Optional[nx.DiGraph] = None

    @classmethod
    def from_json(cls, path: str) -> "Schema":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(data)

    # ----------- tiện ích liệt kê ----------
    def list_feature_types(self) -> List[str]:
        out: List[str] = []
        for m in self.raw.get("machines", []):
            for ft in m.get("feature_types", []):
                n = ft.get("name")
                if n and n not in out:
                    out.append(n)
        return out

    def list_components(self) -> List[str]:
        out: List[str] = []
        for m in self.raw.get("machines", []):
            for c in m.get("components", []):
                n = c.get("name")
                if n and n not in out:
                    out.append(n)
        return out

    def list_rules(self) -> List[str]:
        out: List[str] = []
        for r in self.raw.get("rules", []):
            n = r.get("name")
            if n:
                out.append(n)
        return out

    def list_events(self) -> List[str]:
        out: List[str] = []
        for e in self.raw.get("events", []):
            d = e.get("description")
            if d:
                out.append(d)
        return out

    def has_feature_type(self, name: str) -> bool:
        return name in set(self.list_feature_types())

    def has_component(self, name: str) -> bool:
        return name in set(self.list_components())

    # ------------- graph dựng từ schema --------------
    def build_graph(self, force_rebuild: bool = False) -> nx.DiGraph:
        """
        Tạo đồ thị dị chủng:
          - Node nhãn: "Feature::<name>", "Component::<name>", "Rule::<name>", "Event::<desc>"
          - Cạnh:
              * Component -> Feature (cùng machine) [liên hệ cấu trúc yếu]
              * Rule -> Feature (từ logic)
              * (tuỳ chọn) Event -> Feature (từ sources & rule_ref)
        """
        if self._G is not None and not force_rebuild:
            return self._G

        G = nx.DiGraph()

        # Machines
        for m in self.raw.get("machines", []):
            comps = m.get("components", []) or []
            ftypes = m.get("feature_types", []) or []
            for c in comps:
                cname = c.get("name")
                if cname:
                    G.add_node(f"Component::{cname}")
            for ft in ftypes:
                fname = ft.get("name")
                if fname:
                    G.add_node(f"Feature::{fname}")
            # liên hệ yếu: cùng máy -> có thể có ảnh hưởng cấu trúc
            for c in comps:
                for ft in ftypes:
                    cname, fname = c.get("name"), ft.get("name")
                    if cname and fname:
                        G.add_edge(f"Component::{cname}", f"Feature::{fname}")

        # Rules
        for r in self.raw.get("rules", []) or []:
            rname = r.get("name")
            if rname:
                G.add_node(f"Rule::{rname}")
            for lg in r.get("logic", []) or []:
                comp = lg.get("of_component")
                ftype = lg.get("feature_type")
                if comp:
                    G.add_node(f"Component::{comp}")
                if ftype:
                    G.add_node(f"Feature::{ftype}")
                if rname and ftype:
                    G.add_edge(f"Rule::{rname}", f"Feature::{ftype}")
                if comp and ftype:
                    G.add_edge(f"Component::{comp}", f"Feature::{ftype}")

        # Events
        for ev in self.raw.get("events", []) or []:
            desc = ev.get("description")
            if desc:
                G.add_node(f"Event::{desc}")
            rule_ref = ev.get("rule_ref")
            if rule_ref:
                G.add_node(f"Rule::{rule_ref}")
            for s in ev.get("sources", []) or []:
                ft = s.get("feature_type")
                if ft:
                    G.add_node(f"Feature::{ft}")
                    # kết nối rule/event -> feature (ngữ nghĩa: liên quan)
                    if rule_ref:
                        G.add_edge(f"Rule::{rule_ref}", f"Feature::{ft}")
                    if desc:
                        G.add_edge(f"Event::{desc}", f"Feature::{ft}")

        self._G = G
        return self._G

    # ----------------- helpers -----------------
    @staticmethod
    def feature_node(name: str) -> str:
        return f"Feature::{name}"

    @staticmethod
    def component_node(name: str) -> str:
        return f"Component::{name}"

    @staticmethod
    def rule_node(name: str) -> str:
        return f"Rule::{name}"

    @staticmethod
    def event_node(desc: str) -> str:
        return f"Event::{desc}"
