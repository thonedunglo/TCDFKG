import torch
import torch.nn as nn

class EdgeBiasAttention(nn.Module):
    """
    Attention theo TCG: logit(i->j) = Q_j·K_i + MLP(e_ij), e_ij=[S_norm, d_norm]
    - TCG: {'edges': [{'src','dst','lag','score'}, ...]}
    - dmax: độ trễ chuẩn hoá. d_norm = min(lag,dmax)/dmax
    """
    def __init__(self, tcg:dict, dmax:int=32, mlp_hidden:int=16, device="cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.dmax = max(1, int(dmax))
        self.edges = tcg.get("edges", [])
        # chuẩn hoá S
        S = [e["score"] for e in self.edges] or [0.0]
        smin, smax = float(min(S)), float(max(S))
        self._s_rng = (smin, smax)
        # adj_list[j] = list(i)
        self.adj = {}
        self.eij = {}  # (i,j) -> tensor([S_norm, d_norm])
        for e in self.edges:
            i, j = int(e["src"]), int(e["dst"])
            s = float(e["score"])
            lag = int(e.get("lag", 0))
            s_norm = 0.0 if smax == smin else (s - smin) / (smax - smin + 1e-8)
            d_norm = min(lag, self.dmax) / self.dmax
            self.adj.setdefault(j, []).append(i)
            self.eij[(i, j)] = torch.tensor([s_norm, d_norm], dtype=torch.float32)
        self.bias_mlp = nn.Sequential(
            nn.Linear(2, mlp_hidden), nn.ReLU(),
            nn.Linear(mlp_hidden, 1)
        )
        self.to(self.device)

    def forward(self, Q:torch.Tensor, K:torch.Tensor, V:torch.Tensor):
        """
        Q,K,V: (B,N,C) -> H_msg: (B,N,C)
        """
        B, N, C = Q.shape
        H_msg = torch.zeros_like(Q)
        for j, parents in self.adj.items():
            if not parents: 
                continue
            # (B,1,C) and (B,P,C)
            qj = Q[:, j:j+1, :]                     # (B,1,C)
            ki = K[:, parents, :]                   # (B,P,C)
            vi = V[:, parents, :]                   # (B,P,C)
            # dot: (B,P)
            logits = torch.einsum("bic,bjc->bij", qj, ki).squeeze(1)  # (B,P)
            # add bias from e_ij
            bias_list = []
            for i in parents:
                bias_list.append(self.bias_mlp(self.eij[(i, j)].to(Q.device)).view(1,1))
            bias = torch.cat(bias_list, dim=1)                     # (1,P)
            logits = logits + bias                                  # broadcast to (B,P)
            alpha = torch.softmax(logits, dim=1)                    # (B,P)
            H_msg[:, j, :] = torch.einsum("bp,bpc->bc", alpha, vi)  # (B,C)
        return H_msg
