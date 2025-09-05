import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
from .temporal_encoder import TempEnc
from .edge_bias_attn import EdgeBiasAttention

class Detector(nn.Module):
    """
    X_(t-w:t-1) -> TempEnc -> (Q,K,V) -> EdgeBiasAttention -> Residual -> Head -> \hat x_t
    """
    def __init__(self, enc_cfg:Dict, attn_cfg:Dict, tcg:dict, device="cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.enc = TempEnc(window=enc_cfg["window"],
                           channels=enc_cfg.get("channels", 128),
                           kernels=tuple(enc_cfg.get("kernels", [2,3,5,6])))
        c = enc_cfg.get("channels", 128)
        self.Wq = nn.Linear(c, c, bias=False)
        self.Wk = nn.Linear(c, c, bias=False)
        self.Wv = nn.Linear(c, c, bias=False)
        self.attn = EdgeBiasAttention(tcg, dmax=attn_cfg.get("dmax", 32),
                                      mlp_hidden=attn_cfg.get("bias_mlp_hidden", 16),
                                      device=device)
        self.norm = nn.LayerNorm(c)
        self.head = nn.Linear(c, 1)  # dùng chung cho mọi nút
        self.to(self.device)

    def forward(self, x):  # x: (B,N,w)
        H = self.enc(x)                 # (B,N,C)
        Q = self.Wq(H); K = self.Wk(H); V = self.Wv(H)  # (B,N,C)
        Hm = self.attn(Q, K, V)         # (B,N,C)
        Ht = self.norm(H + Hm)          # residual + LN
        y_hat = self.head(Ht).squeeze(-1)  # (B,N)
        return y_hat

    def fit(self, dl_train, dl_val, lr=1e-3, epochs=50, early_stop=7):
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        mse = nn.MSELoss()
        best, best_state, patience = float("inf"), None, early_stop
        for ep in range(1, epochs+1):
            self.train(); tr_loss = 0.0; n=0
            for xb, yb in dl_train:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                yhat = self.forward(xb)
                loss = mse(yhat, yb)
                loss.backward(); opt.step()
                tr_loss += loss.item() * xb.size(0); n += xb.size(0)
            # val
            self.eval(); va_loss = 0.0; m=0
            with torch.no_grad():
                for xb, yb in dl_val:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    yhat = self.forward(xb)
                    loss = mse(yhat, yb)
                    va_loss += loss.item()*xb.size(0); m += xb.size(0)
            tr, va = tr_loss/max(1,n), va_loss/max(1,m)
            print(f"[Detector] epoch {ep:02d} | train MSE {tr:.6f} | val MSE {va:.6f}")
            if va+1e-9 < best:
                best, best_state, patience = va, {k:v.cpu() for k,v in self.state_dict().items()}, early_stop
            else:
                patience -= 1
                if patience <= 0: break
        if best_state: self.load_state_dict(best_state)
        return {"best_val_mse": best}
