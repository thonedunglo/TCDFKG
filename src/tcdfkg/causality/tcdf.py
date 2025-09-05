import numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F

class CausalConv1d(nn.Conv1d):
    def __init__(self, in_ch, out_ch, ksz, dilation=1, **kw):
        padding = (ksz - 1) * dilation
        super().__init__(in_ch, out_ch, ksz, padding=padding, dilation=dilation, **kw)
        self._pad = padding
    def forward(self, x):
        y = super().forward(x)
        return y[..., :-self._pad] if self._pad else y

class TCNBlock(nn.Module):
    def __init__(self, ch, ksz=3, dilation=1, drop=0.0):
        super().__init__()
        self.c1 = CausalConv1d(ch, ch, ksz, dilation=dilation)
        self.c2 = CausalConv1d(ch, ch, ksz, dilation=dilation)
        self.n1 = nn.BatchNorm1d(ch); self.n2 = nn.BatchNorm1d(ch)
        self.do = nn.Dropout(drop)
    def forward(self, x):
        y = self.do(self.n1(F.relu(self.c1(x))))
        y = self.n2(F.relu(self.c2(y)))
        return x + y

class TemporalCNN(nn.Module):
    def __init__(self, in_ch, hid_ch=64, out_ch=1, dil=(1,2,4,8,16), ksz=3, drop=0.1):
        super().__init__()
        self.inp = nn.Conv1d(in_ch, hid_ch, 1)
        self.blocks = nn.ModuleList([TCNBlock(hid_ch, ksz, d, drop) for d in dil])
        self.head = nn.Conv1d(hid_ch, out_ch, 1)
    def forward(self, x):
        h = F.relu(self.inp(x))
        for b in self.blocks: h = b(h)
        return self.head(h)  # (B,1,T)

class TCDF(nn.Module):
    """
    TCDF-style: attention nguồn→đích để chọn ứng viên, CNN nhân quả giãn nở dự báo X_j.
    Sau train, dùng softmax theo cột để lấy s_{i->j} và dùng saliency theo thời gian để ước lượng trễ.
    """
    def __init__(self, num_series, window, dilations=(1,2,4,8,16), kernel_size=3, hid_ch=64, dropout=0.1, device="cpu"):
        super().__init__()
        self.N, self.w = num_series, window
        self.register_buffer("diag_mask", (~torch.eye(self.N, dtype=torch.bool)).float())
        self.att_logits = nn.Parameter(0.01 * torch.randn(self.N, self.N))  # i->j
        self.cnn = TemporalCNN(self.N, hid_ch, 1, tuple(dilations), kernel_size, dropout)
        self.bias_j = nn.Parameter(torch.zeros(self.N))
        self.device = torch.device(device)
        self.to(self.device)

    def att_softmax(self):
        logits = self.att_logits + torch.log(self.diag_mask + 1e-8)  # -inf trên đường chéo
        return torch.softmax(logits, dim=0)  # chuẩn hóa theo i (cha) cho mỗi j

    def forward(self, xw: torch.Tensor):
        # xw: (B,N,w) -> y_hat: (B,N)
        B, N, w = xw.shape
        assert N == self.N and w == self.w
        S = self.att_softmax().to(xw.device)  # (N,N)
        outs = []
        for j in range(self.N):
            s_col = S[:, j].view(1, N, 1)   # (1,N,1)
            xs = xw * s_col
            yseq = self.cnn(xs)             # (B,1,w)
            outs.append(yseq[:, 0, -1] + self.bias_j[j])
        return torch.stack(outs, dim=1)      # (B,N)

    def fit(self, X: np.ndarray, epochs=20, lr=1e-3, batch_size=64, verbose=True):
        # X: (N,T)
        N, T = X.shape; assert N == self.N
        num_windows = T - self.w
        if num_windows <= 0:
            raise ValueError(
                f"TCDF.fit needs at least one window but got X.shape={X.shape}, "
                f"window={self.w} -> {num_windows} windows"
            )
        windows, targets = [], []
        for t in range(self.w, T):
            windows.append(X[:, t-self.w:t]); targets.append(X[:, t])
        Xw = torch.tensor(np.stack(windows), dtype=torch.float32, device=self.device)  # (B,N,w)
        Yt = torch.tensor(np.stack(targets), dtype=torch.float32, device=self.device)  # (B,N)
        opt = torch.optim.Adam(self.parameters(), lr=lr); mse = nn.MSELoss()
        for ep in range(epochs):
            self.train(); loss_sum = 0.0
            idx = torch.randperm(Xw.shape[0])
            for s in range(0, len(idx), batch_size):
                sel = idx[s:s+batch_size]
                xb, yb = Xw[sel], Yt[sel]
                opt.zero_grad(); yhat = self.forward(xb)
                loss = mse(yhat, yb); loss.backward(); opt.step()
                loss_sum += loss.item() * xb.size(0)
            if verbose: print(f"[TCDF] epoch {ep+1:02d} | loss={loss_sum/len(idx):.6f}")

    def get_attention(self) -> np.ndarray:
        with torch.no_grad(): return self.att_softmax().cpu().numpy()
