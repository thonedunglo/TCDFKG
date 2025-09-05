"""
Rule utilities: trích xuất quan hệ hướng từ Rule/Event tới Feature,
và các predicate hỗ trợ verification:
  - is_explicit_rule_edge(ft_i -> ft_j): có “quy tắc” nào ngầm gợi ý hướng?
  - share_rule_context(ft_i, ft_j): hai feature cùng liên quan một Rule/Event?
"""
from __future__ import annotations
from typing import Dict, Any, Set
from .schema import Schema


def _pred_rules_to_feature(schema: Schema) -> Dict[str, Set[str]]:
    """
    Map Rule::<name> -> {Feature::<type>...}
    """
    G = schema.build_graph()
    out: Dict[str, Set[str]] = {}
    for n in G.nodes:
        if str(n).startswith("Rule::"):
            outs = {v for _, v in G.out_edges(n) if str(v).startswith("Feature::")}
            if outs:
                out[str(n)] = outs
    return out


def _pred_events_to_feature(schema: Schema) -> Dict[str, Set[str]]:
    """
    Map Event::<desc> -> {Feature::<type>...}
    """
    G = schema.build_graph()
    out: Dict[str, Set[str]] = {}
    for n in G.nodes:
        if str(n).startswith("Event::"):
            outs = {v for _, v in G.out_edges(n) if str(v).startswith("Feature::")}
            if outs:
                out[str(n)] = outs
    return out


def share_rule_context(schema: Schema, ft_i: str, ft_j: str) -> bool:
    """
    Hai feature cùng thuộc ngữ cảnh một Rule/Event (gián tiếp) hay không?
    Dùng trong S_rule "mềm" (ρ).
    """
    Gi = Schema.feature_node(ft_i)
    Gj = Schema.feature_node(ft_j)
    G = schema.build_graph()

    # cùng là đích của một Rule hay Event
    preds_i = {u for u in G.predecessors(Gi)}
    preds_j = {u for u in G.predecessors(Gj)}
    if preds_i & preds_j:
        return True
    return False


def is_explicit_rule_edge(schema: Schema, ft_i: str, ft_j: str) -> bool:
    """
    Có bằng chứng Rule/Event “hướng” từ i -> j không?
    Ở bản chuẩn hoá này, ta xem:
      - i và j đều là đích của *cùng* Rule R thì không đủ kết luận hướng i->j.
      - Nếu tồn tại Rule R nối tới j, và i là 'nguồn cấu trúc' liên kết mạnh tới Rule R,
        thì có thể coi là gợi ý hướng. Do schema hiện tại không nối Feature -> Rule,
        ta giữ mặc định là False (tránh overclaim).
    Bạn có thể thay đổi hàm này nếu schema/rules chi tiết hơn (ví dụ có trigger chains).
    """
    # Giữ conservative => False
    return False
