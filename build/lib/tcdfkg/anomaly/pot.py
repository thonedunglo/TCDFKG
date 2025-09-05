import numpy as np
from scipy.stats import genpareto as gpd

class PotPerFeature:
    """
    POT theo từng feature:
    - Chọn u_i = quantile q0 (mặc định 0.95) trên scores_val[:,i]
    - Fit GPD cho exceedances (x - u_i)
    - Tính ngưỡng τ_i để đạt tail prob ~ (1 - gamma)
    Fallback: nếu exceedances < min_exceed => τ_i = quantile gamma
    """
    def __init__(self, q0:float=0.95, gamma:float=0.99, min_exceed:int=50):
        self.q0 = q0
        self.gamma = gamma
        self.min_exceed = min_exceed
        self.taus = None

    def fit(self, scores_val: np.ndarray) -> dict:
        T, N = scores_val.shape
        taus = {}
        for i in range(N):
            x = scores_val[:, i]
            u = np.quantile(x, self.q0)
            exc = x[x > u] - u
            if exc.size >= self.min_exceed:
                # fit GPD (ξ, β)
                c, loc, scale = gpd.fit(exc, floc=0)  # loc=0
                # tail quantile so that P(X>τ)=1-gamma:
                # For exceedance Y~GPD(c, scale), quantile at p = (gamma - q0)/(1 - q0)
                p = (self.gamma - self.q0) / (1 - self.q0 + 1e-12)
                p = np.clip(p, 1e-6, 1-1e-6)
                q = gpd.ppf(p, c, loc=0, scale=scale)
                tau = float(u + max(0.0, q))
            else:
                tau = float(np.quantile(x, self.gamma))
            taus[i] = tau
        self.taus = taus
        return taus

    def predict(self, scores: np.ndarray, taus: dict=None) -> np.ndarray:
        th = self.taus if taus is None else taus
        assert th is not None, "Call fit() first or pass taus"
        T, N = scores.shape
        y = (scores > np.array([th[i] for i in range(N)])[None, :]).astype(np.int32)
        return y
