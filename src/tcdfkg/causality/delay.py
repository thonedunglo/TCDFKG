import numpy as np, torch

@torch.no_grad()
def _prep_windows(X, w, device):
    N, T = X.shape
    wins = [X[:, t-w:t] for t in range(w, T)]
    return torch.tensor(np.stack(wins), dtype=torch.float32, device=device)  # (B,N,w)

def estimate_lags_tcdf_saliency(model, X: np.ndarray, max_lag: int = None, batch_size: int = 128):
    """
    Ước lượng trễ kiểu TCDF: chọn τ làm chỉ số thời gian w-1-τ có |∂ŷ_j/∂x_i| lớn nhất (trung bình theo batch).
    - model: TCDF đã fit
    - X: (N,T) đã chuẩn hoá
    Trả về: dict {(i,j): tau_best}
    """
    device = next(model.parameters()).device
    N, T = X.shape; w = model.w
    tau_max = min(max_lag if max_lag is not None else (w-1), w-1)
    Xw = _prep_windows(X, w, device)  # (B,N,w)
    B = Xw.shape[0]
    S_att = model.get_attention()
    d_hat = {}

    for j in range(N):
        sal_ij = np.zeros((N, tau_max+1), dtype=np.float64)
        for s in range(0, B, batch_size):
            xb = Xw[s:s+batch_size].clone().detach().requires_grad_(True)
            yhat = model(xb)            # (b,N)
            y = yhat[:, j].sum()
            model.zero_grad(set_to_none=True)
            y.backward()
            g = xb.grad.detach().abs().cpu().numpy()  # (b,N,w)
            for tau in range(tau_max+1):
                idx = w-1-tau
                sal_ij[:, tau] += g[:, :, idx].mean(axis=0)
        for i in range(N):
            if i == j or S_att[i, j] <= 0: continue
            d_hat[(i, j)] = int(np.argmax(sal_ij[i, :]))
    return d_hat
