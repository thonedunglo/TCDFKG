import numpy as np

def mse(yhat: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((yhat - y) ** 2))

def mae(yhat: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.abs(yhat - y)))

def rmse(yhat: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(np.mean((yhat - y) ** 2)))
