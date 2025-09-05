import argparse, os, json
from typing import List, Optional
import pandas as pd
import numpy as np

from src.tcdfkg.data.preprocessing import (
    load_timeseries_any,
    resample_df,
    align_columns,
    impute_df,
)

def _infer_paths_from_cfg(cfg_path: str):
    try:
        import yaml
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        cfg = {}
    data_dir = cfg.get("data_dir", "./data")
    raw_dir = cfg.get("raw_dir", os.path.join(data_dir, "raw"))
    processed_dir = cfg.get("processed_dir", os.path.join(data_dir, "processed"))
    kg_dir = cfg.get("kg", {}).get("dir", os.path.join(data_dir, "kg"))

    freq = (cfg.get("time") or {}).get("freq", None)
    split = cfg.get("split") or {}
    val_ratio = float(split.get("val_ratio", 0.1))
    test_ratio = float(split.get("test_ratio", 0.1))
    return {
        "data_dir": data_dir,
        "raw_dir": raw_dir,
        "processed_dir": processed_dir,
        "kg_dir": kg_dir,
        "freq": freq,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
    }

def _save_parquet(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path)
    print(f"[prepare] saved {path} shape={df.shape}")

def main():
    ap = argparse.ArgumentParser(description="Prepare MTS data: resample/align/impute/split (NO OUTLIER CLIP)")
    ap.add_argument("--dataset_yaml", default="configs/dataset.yaml")
    ap.add_argument("--input", default="", help="file/dir dữ liệu thô; trống -> dùng data/raw")
    ap.add_argument("--freq", default="", help="vd 1min, 10S; trống -> dùng dataset.yaml (time.freq)")
    ap.add_argument("--val_ratio", type=float, default=-1, help="override; <0 -> dùng cfg")
    ap.add_argument("--test_ratio", type=float, default=-1, help="override; <0 -> dùng cfg")
    ap.add_argument("--impute", default="ffill_bfill",
                    choices=["ffill","bfill","ffill_bfill","mean"],
                    help="cách bù khuyết (không đụng tới giá trị ngoại lai)")
    ap.add_argument("--save_train", default="data/processed/train.parquet")
    ap.add_argument("--save_val",   default="data/processed/val.parquet")
    ap.add_argument("--save_test",  default="data/processed/test.parquet")
    args = ap.parse_args()

    cfg = _infer_paths_from_cfg(args.dataset_yaml)
    raw_input = args.input or cfg["raw_dir"]
    freq = args.freq or cfg["freq"]
    val_ratio = cfg["val_ratio"] if args.val_ratio < 0 else args.val_ratio
    test_ratio = cfg["test_ratio"] if args.test_ratio < 0 else args.test_ratio

    # 1) Load thô
    df = load_timeseries_any(raw_input)
    print(f"[prepare] loaded raw df shape={df.shape} from {raw_input}")

    # 2) Resample & align (không đụng giá trị ngoại lai)
    if freq:
        df = resample_df(df, rule=freq, how="mean")  # cần robust hơn thì đổi 'median'
        print(f"[prepare] resampled to freq={freq} shape={df.shape}")
    df = align_columns(df)

    # 3) Impute missing (giữ nguyên biên độ giá trị)
    df = impute_df(df, method=args.impute)

    # 4) Split theo thời gian
    n = len(df)
    n_test = int(round(n * test_ratio))
    n_val  = int(round((n - n_test) * val_ratio))
    n_train = n - n_val - n_test
    if min(n_train, n_val, n_test) <= 0:
        raise ValueError(f"Split too small: train={n_train}, val={n_val}, test={n_test}")

    df_train = df.iloc[:n_train].copy()
    df_val   = df.iloc[n_train:n_train + n_val].copy()
    df_test  = df.iloc[n_train + n_val:].copy()

    # 5) Save
    _save_parquet(df_train, args.save_train)
    _save_parquet(df_val,   args.save_val)
    _save_parquet(df_test,  args.save_test)

    # 6) Meta
    meta = {
        "freq": freq,
        "columns": list(df.columns),
        "shape": {"train": df_train.shape, "val": df_val.shape, "test": df_test.shape},
        "split": {"train": n_train, "val": n_val, "test": n_test},
        "impute": args.impute,
        "outlier_clip": False
    }
    with open(os.path.join(os.path.dirname(args.save_train), "prepare_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("[prepare] done (no outlier clipping).")

if __name__ == "__main__":
    main()
