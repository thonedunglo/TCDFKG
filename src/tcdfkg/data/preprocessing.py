import os, glob
from typing import List, Optional
import pandas as pd
import numpy as np


def load_timeseries_any(path_or_dir: str) -> pd.DataFrame:
    """
    - Nếu là file .csv/.parquet: cột 'timestamp' hoặc index datetime.
    - Nếu là thư mục: load tất cả .csv/.parquet, join theo timestamp.
      + CSV: yêu cầu có cột 'timestamp' + một cột giá trị; tên cột giá trị dùng tên file làm column.
    """
    p = path_or_dir
    if os.path.isdir(p):
        files = sorted(glob.glob(os.path.join(p, "*.parquet")) + glob.glob(os.path.join(p, "*.csv")))
        if not files:
            raise FileNotFoundError(f"No CSV/Parquet under dir: {p}")
        dfs = [load_timeseries_any(f) for f in files]
        # join theo index thời gian (outer)
        df = pd.concat(dfs, axis=1).sort_index()
        return df
    else:
        ext = os.path.splitext(p)[1].lower()
        if ext == ".csv":
            df = pd.read_csv(p)
        elif ext == ".parquet":
            df = pd.read_parquet(p)
        else:
            raise ValueError(f"Unsupported format: {ext}")
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp")
        # nếu chỉ còn 1 cột giá trị không tên → đặt tên theo file
        if df.shape[1] == 1 and df.columns.tolist() == [0]:
            colname = os.path.splitext(os.path.basename(p))[0]
            df.columns = [colname]
        return df


def resample_df(df: pd.DataFrame, rule: str, how: str = "mean") -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be DatetimeIndex for resampling.")
    if how == "mean":
        return df.resample(rule).mean()
    elif how == "median":
        return df.resample(rule).median()
    elif how == "sum":
        return df.resample(rule).sum()
    else:
        raise ValueError(f"Unsupported resample how={how}")


def align_columns(df: pd.DataFrame) -> pd.DataFrame:
    # bỏ cột trùng tên, sắp xếp cột
    df = df.loc[:, ~df.columns.duplicated()].copy()
    df = df.sort_index()
    df = df.reindex(sorted(df.columns), axis=1)
    return df


def impute_df(df: pd.DataFrame, method: str = "ffill_bfill") -> pd.DataFrame:
    if method == "ffill":
        return df.ffill()
    elif method == "bfill":
        return df.bfill()
    elif method == "ffill_bfill":
        return df.ffill().bfill()
    elif method == "mean":
        return df.fillna(df.mean())
    else:
        raise ValueError(f"Unsupported impute method: {method}")


def clip_outliers_iqr(df: pd.DataFrame, whisker: float = 1.5) -> pd.DataFrame:
    """
    Cắt giá trị vượt ngoài [Q1 - whisker*IQR, Q3 + whisker*IQR] theo từng cột.
    """
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    iqr = q3 - q1
    low = q1 - whisker * iqr
    high = q3 + whisker * iqr
    return df.clip(lower=low, upper=high, axis=1)
