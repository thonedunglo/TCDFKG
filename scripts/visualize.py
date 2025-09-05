import argparse, os, json
import numpy as np
import pandas as pd

from tcdfkg.causality.tcg import load_tcg
from tcdfkg.viz.tcg import plot_tcg
from tcdfkg.viz.attention import plot_attention_heatmap, tcg_to_adj_matrix
from tcdfkg.viz.anomaly import plot_series_pred_anomaly
from tcdfkg.viz.propagation import plot_propagation_timeline
from tcdfkg.data.loaders import load_mts
from tcdfkg.utils.config import load_cfg
from tcdfkg.viz.utils import ensure_dir


def _load_colnames_from_dataset(dataset_yaml: str) -> list[str]:
    cfg = load_cfg(dataset_yaml, dataset_yaml, save_dir="artifacts")  # trick chỉ để parse yaml
    # ưu tiên processed/train để lấy thứ tự cột
    path = cfg.dataset.get("mts", {}).get("train") or ""
    if path and os.path.isfile(path):
        df = load_mts(path)
        return list(df.columns)
    return []


def main():
    ap = argparse.ArgumentParser(description="Visualization toolkit")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # --- TCG graph ---
    sp_tcg = sub.add_parser("tcg", help="plot TCG graph")
    sp_tcg.add_argument("--tcg_path", default="artifacts/tcg.json")
    sp_tcg.add_argument("--dataset_yaml", default="configs/dataset.yaml", help="để lấy tên cột cho labels (nếu có)")
    sp_tcg.add_argument("--out", default="artifacts/plots/tcg.png")
    sp_tcg.add_argument("--top_edges", type=int, default=200)
    sp_tcg.add_argument("--layout", choices=["kamada_kawai","spring"], default="kamada_kawai")

    # --- Attention/Adjacency heatmap ---
    sp_hm = sub.add_parser("heatmap", help="plot attention/adjacency heatmap")
    sp_hm.add_argument("--tcg_path", default="artifacts/tcg.json", help="dùng để dựng adjacency nếu không có S_att")
    sp_hm.add_argument("--satt_path", default="", help="npz/npy chứa S_att (N,N). Nếu trống, dùng adjacency từ TCG.")
    sp_hm.add_argument("--dataset_yaml", default="configs/dataset.yaml")
    sp_hm.add_argument("--out", default="artifacts/plots/heatmap.png")

    # --- Anomaly series ---
    sp_an = sub.add_parser("anomaly", help="plot one variable truth vs pred + anomaly marks")
    sp_an.add_argument("--truth_path", default="data/processed/test.parquet")
    sp_an.add_argument("--pred_path", default="artifacts/preds.parquet")
    sp_an.add_argument("--labels_path", default="artifacts/anomaly_labels.parquet")
    sp_an.add_argument("--var", required=True)
    sp_an.add_argument("--out", default="artifacts/plots/anomaly_VAR.png")

    # --- Propagation timeline ---
    sp_pr = sub.add_parser("propagation", help="plot propagation timeline (top-k per delta)")
    sp_pr.add_argument("--prop_path", default="artifacts/propagation.json")
    sp_pr.add_argument("--dataset_yaml", default="configs/dataset.yaml")
    sp_pr.add_argument("--out", default="artifacts/plots/propagation.png")
    sp_pr.add_argument("--topk", type=int, default=10)

    args = ap.parse_args()

    if args.cmd == "tcg":
        names = _load_colnames_from_dataset(args.dataset_yaml)
        plot_tcg(args.tcg_path, out_path=args.out, top_edges=args.top_edges, node_labels=names, layout=args.layout)
        print(f"[viz] saved {args.out}")

    elif args.cmd == "heatmap":
        names = _load_colnames_from_dataset(args.dataset_yaml)
        if args.satt_path and os.path.isfile(args.satt_path):
            # np.save('artifacts/S_att.npy', S_att) hoặc np.savez(..., S_att=S_att)
            if args.satt_path.endswith(".npz"):
                M = np.load(args.satt_path)["S_att"]
            else:
                M = np.load(args.satt_path)
            title = "TCDF Attention (S_att)"
        else:
            tcg = load_tcg(args.tcg_path)
            M = tcg_to_adj_matrix(tcg, normalize=True)
            title = "Adjacency from TCG (score scaled)"
        plot_attention_heatmap(M, out_path=args.out, title=title, xlabels=names, ylabels=names, vmin=0.0, vmax=1.0)
        print(f"[viz] saved {args.out}")

    elif args.cmd == "anomaly":
        truth = pd.read_parquet(args.truth_path)
        preds = pd.read_parquet(args.pred_path)
        labels = None
        if os.path.isfile(args.labels_path):
            labels = pd.read_parquet(args.labels_path)
        out = args.out.replace("VAR", args.var)
        plot_series_pred_anomaly(truth, preds, labels, var_name=args.var, out_path=out, title=f"Anomaly - {args.var}")
        print(f"[viz] saved {out}")

    elif args.cmd == "propagation":
        with open(args.prop_path, "r", encoding="utf-8") as f:
            prop = json.load(f)
        names = _load_colnames_from_dataset(args.dataset_yaml)
        plot_propagation_timeline(prop, out_path=args.out, topk_per_delta=args.topk, id2name=names)
        print(f"[viz] saved {args.out}")

if __name__ == "__main__":
    main()
