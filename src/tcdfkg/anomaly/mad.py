import numpy as np

class MadNormalizer:
    """
    a_{i,t} = (|ŷ_i - x_i| - med_i) / (MAD_i + eps)
    """
    def __init__(self, eps:float=1e-6):
        self.eps = eps
        self.med = None
        self.mad = None

    def fit(self, errors_val: np.ndarray):
        # errors_val: (T_val, N)
        self.med = np.median(errors_val, axis=0)
        self.mad = np.median(np.abs(errors_val - self.med[None,:]), axis=0)

    def transform(self, errors: np.ndarray) -> np.ndarray:
        assert self.med is not None and self.mad is not None, "Call fit() first"
        return (errors - self.med[None,:]) / (self.mad[None,:] + self.eps)
