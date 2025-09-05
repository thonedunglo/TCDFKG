from __future__ import annotations
from typing import Optional, Sequence
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .utils import ensure_dir


def plot_series_pred_anomaly(
    truth: pd.DataFrame,
    preds: pd.DataFrame,
    labels: Optional[pd.DataFrame],
    var_name: str,
    out_path: str,
    title: Optional[str] = None,
) -> None:
    """
    Vẽ 1 biến: giá trị thật vs dự báo; đánh dấu bất thường bằng nền/marker.
    - truth: DataFrame ground-truth (có thể là df test hoặc full chuỗi), index là time
    - preds: DataFrame dự báo (cùng index slice)
    - labels: DataFrame {0/1} per-feature; nếu None, chỉ vẽ truth/preds
    - var_name: tên cột cần vẽ
    """
    if var_name not in truth.columns:
        raise KeyError(f"{var_name} not in truth columns")
    if var_name not in preds.columns:
        raise KeyError(f"{var_name} not in preds columns")
    y = truth[var_name]
    yhat = preds[var_name]
    # align index
    y, yhat = y.align(yhat, join="inner")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(y.index, y.values, label="truth", linewidth=1.2)
    ax.plot(yhat.index, yhat.values, label="pred", linewidth=1.2)

    if labels is not None and var_name in labels.columns:
        lab = labels[var_name].reindex(y.index).fillna(0).values.astype(bool)
        if lab.any():
            ax.scatter(y.index[lab], y.values[lab], s=18, marker="o", label="anomaly", zorder=3)

    ax.set_title(title or var_name)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, linewidth=0.3, alpha=0.6)
    ensure_dir(out_path)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
