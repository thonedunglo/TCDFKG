import argparse, os, json
import numpy as np
import pandas as pd

from tcdfkg.causality.tcg import load_tcg
from tcdfkg.propagation.propagate import propagate
from tcdfkg.data.loaders import load_mts
from tcdfkg.utils.config import load_cfg

def _load_seeds_from_labels(parquet_path: str, time_index: int | None):
    df = pd.read_parquet(parquet_path)
    if time_index is None:
        time_index = len(df) - 1
    row = df.iloc[time_index].to_numpy(dtype=float)  # shape (N,)
    # coi labels là 0/1 => dùng làm xác suất gốc; nếu bạn muốn seed mềm, sửa ở đây.
    return row, time_index, list(df.columns)

def _load_seeds_from_indices(N: int, idx_csv: str):
    A = np.zeros(N, dtype=float)
    if idx_csv.strip():
        for tok in idx_csv.split(","):
            k = tok.strip()
            if not k: 
                continue
            A[int(k)] = 1.0
    return A

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_yaml", required=True, help="để lấy tên biến (cột) nếu cần")
    ap.add_argument("--tcg_path", default="artifacts/tcg.json")
    ap.add_argument("--labels_path", default="artifacts/anomaly_labels.parquet", help="nhãn bất thường per-feature theo thời gian")
    ap.add_argument("--time_index", type=int, default=None, help="dùng hàng t trong labels; mặc định = hàng cuối")
    ap.add_argument("--seed_idx", default="", help="ví dụ '3,7,15' nếu không dùng labels")
    ap.add_argument("--lam", type=float, default=0.1)
    ap.add_argument("--L", type=int, default=3)
    ap.add_argument("--max_delta", type=int, default=None)
    ap.add_argument("--min_score", type=float, default=0.0)
    ap.add_argument("--save_path", default="artifacts/propagation.json")
    args = ap.parse_args()

    cfg = load_cfg(args.dataset_yaml, args.dataset_yaml, save_dir="artifacts")  # hack: chỉ để dùng dataset path
    tcg = load_tcg(args.tcg_path)

    # Lấy N từ tcg hoặc từ dữ liệu
    N = 0
    for e in tcg.get("edges", []):
        N = max(N, int(e["src"]), int(e["dst"]))
    N += 1

    # Lấy seeds A_t
    col_names = None
    if os.path.isfile(args.labels_path) and not args.seed_idx:
        A, t_idx, col_names = _load_seeds_from_labels(args.labels_path, args.time_index)
        print(f"[propagate] use labels row t={t_idx} as seeds.")
    else:
        A = _load_seeds_from_indices(N, args.seed_idx)
        print(f"[propagate] use manual seeds: {args.seed_idx or '(none)'}")

    # Lan truyền
    res = propagate(
        tcg, A,
        lam=args.lam, L=args.L, max_delta=args.max_delta, min_score=args.min_score
    )

    # Đính tên biến (nếu có)
    if col_names is not None:
        id2name = {i: col_names[i] for i in range(len(col_names))}
        for step in res["timeline"]:
            for n in step["nodes"]:
                n["name"] = id2name.get(n["id"], f"var_{n['id']}")

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    with open(args.save_path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"[OK] Saved propagation graph to {args.save_path}")

if __name__ == "__main__":
    main()
