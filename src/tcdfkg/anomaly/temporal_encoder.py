import torch
import torch.nn as nn
import torch.nn.functional as F

class _CausalConv1d(nn.Conv1d):
    def __init__(self, in_ch, out_ch, ksz, dilation=1):
        pad = (ksz - 1) * dilation
        super().__init__(in_ch, out_ch, ksz, padding=pad, dilation=dilation)
        self._pad = pad
    def forward(self, x):  # x: (B,C,T)
        y = super().forward(x)
        return y[..., :-self._pad] if self._pad else y

class _InceptionBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernels=(2,3,5,6)):
        super().__init__()
        self.branches = nn.ModuleList([_CausalConv1d(in_ch, out_ch, k) for k in kernels])
        self.bn = nn.BatchNorm1d(out_ch * len(kernels))
    def forward(self, x):  # (B,1,T)
        ys = [F.relu(b(x)) for b in self.branches]   # list of (B,out_ch,T)
        y = torch.cat(ys, dim=1)                     # (B,out_ch*B,T)
        return self.bn(y)

class TempEnc(nn.Module):
    """
    Mã hoá (B,N,w) -> (B,N,C_enc).
    Mỗi series được xử lý độc lập (group by series), dùng inception causal conv,
    sau đó pooling nhẹ theo thời gian + linear để ra embedding theo nút.
    """
    def __init__(self, window:int, channels:int=128, kernels=(2,3,5,6)):
        super().__init__()
        self.w = window
        self.kernels = kernels
        bch = channels // len(kernels)
        bch = max(8, bch)
        self.inception = _InceptionBlock(1, bch, kernels=kernels)  # (B*N, bch*len(k), T)
        self.proj = nn.Conv1d(bch*len(kernels), channels, kernel_size=1)
        self.head = nn.AdaptiveAvgPool1d(1)  # gộp thời gian -> 1
    def forward(self, x):  # x: (B,N,w)
        B, N, w = x.shape
        assert w == self.w, f"window mismatch: {w}!={self.w}"
        x = x.reshape(B*N, 1, w)
        h = self.inception(x)                # (B*N, K*bch, w)
        h = F.relu(self.proj(h))             # (B*N, C_enc, w)
        h = self.head(h).squeeze(-1)         # (B*N, C_enc)
        h = h.reshape(B, N, -1)              # (B,N,C_enc)
        return h
