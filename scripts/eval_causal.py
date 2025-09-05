"""
Đánh giá TCG đối chiếu ground-truth graph.

Ground truth format chấp nhận:
- JSON {"edges":[{"src":i,"dst":j, "lag": <optional>}, ...]} hoặc
- JSON [{"src":i,"dst":j}, ...] hoặc
- CSV 2 cột src,dst (không header hoặc có header)
"""
import argparse, os, json, csv
from typing import List, Tuple, Dict, Any
from tcdfkg.causality.tcg import load_tcg
from tcdfkg.eval.metrics_causal import prf1, average_precision, shd


def _load_truth(path: str) -> Dict[str, Any]:
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "edges" in data:
            return data
        elif isinstance(data, list):
            return {"edges": data}
        else:
            raise ValueError("Unsupported JSON format for ground-truth.")
    elif path.endswith(".csv"):
        edges: List[Dict[str, int]] = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
            # thử bỏ header
            try:
                int(rows[0][0]); int(rows[0][1])
                rows_iter = rows
            except Exception:
                rows_iter = rows[1:]
            for r in rows_iter:
                edges.append({"src": int(r[0]), "dst": int(r[1])})
        return {"edges": edges}
    else:
        raise ValueError("Unsupported ground-truth format (use .json or .csv).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_tcg", default="artifacts/tcg.json")
    ap.add_argument("--truth_path", required=True)
    ap.add_argument("--tau_tol", type=int, default=None, help="dung sai trễ (nếu ground-truth có lag)")
    ap.add_argument("--score_threshold", type=float, default=0.5)
    ap.add_argument("--save_path", default="artifacts/metrics_causal.json")
    args = ap.parse_args()

    pred = load_tcg(args.pred_tcg)
    truth = _load_truth(args.truth_path)

    m_prf = prf1(pred["edges"], truth["edges"], threshold=args.score_threshold, tau_tol=args.tau_tol)
    m_ap = average_precision(pred["edges"], truth["edges"])
    m_shd = shd(pred["edges"], truth["edges"], undirected=False)

    out = {
        "threshold": args.score_threshold,
        "tau_tol": args.tau_tol,
        "precision": m_prf["precision"],
        "recall": m_prf["recall"],
        "f1": m_prf["f1"],
        "tp": m_prf["tp"], "fp": m_prf["fp"], "fn": m_prf["fn"],
        "AP": m_ap,
        "SHD": m_shd,
    }
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    with open(args.save_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("[OK] saved causal metrics ->", args.save_path)


if __name__ == "__main__":
    main()
