import numpy as np
from tcdfkg.causality.tcdf import TCDF
from tcdfkg.causality.delay import estimate_lags_tcdf_saliency

def _gen_data(T=300, lag=3, noise=0.05):
    t = np.arange(T)
    x1 = np.sin(0.05*t)
    x2 = np.roll(x1, lag) + noise*np.random.randn(T)  # x1 -> x2 with lag
    x1[:lag] = 0.0  # cold start
    X = np.vstack([x1, x2])  # N=2
    # z-score
    X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True)+1e-8)
    return X

def test_tcdf_saliency_delay_close():
    lag_true = 3
    X = _gen_data(T=320, lag=lag_true)
    model = TCDF(num_series=2, window=64, dilations=(1,2,4,8), kernel_size=3, hid_ch=32, dropout=0.0, device="cpu")
    model.fit(X, epochs=8, lr=3e-3, batch_size=64, verbose=False)
    d_hat = estimate_lags_tcdf_saliency(model, X, max_lag=16)
    assert (0,1) in d_hat
    assert abs(d_hat[(0,1)] - lag_true) <= 2  # dung sai nhỏ cho toy
