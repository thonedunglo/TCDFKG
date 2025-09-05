import numpy as np
from typing import Tuple

def make_windows(X: np.ndarray, w: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tạo tập (windows, targets) cho one-step forecasting.
    Input:
      X: (N,T)  (cột là thời gian)
      w: window length
    Output:
      Xw: (B,N,w), Yt: (B,N)
    """
    N, T = X.shape
    assert T > w, "T must be > window length"
    xs, ys = [], []
    for t in range(w, T):
        xs.append(X[:, t - w:t])
        ys.append(X[:, t])
    return np.stack(xs), np.stack(ys)
