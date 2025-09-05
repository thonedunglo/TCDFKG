import argparse, os, json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from tcdfkg.utils.config import load_cfg
from tcdfkg.utils.seed import set_seed
from tcdfkg.data.loaders import load_mts
from tcdfkg.causality.tcg import load_tcg
from tcdfkg.anomaly.detector import Detector
from tcdfkg.anomaly.mad import MadNormalizer
from tcdfkg.anomaly.pot import PotPerFeature

class WindowDataset(Dataset):
    def __init__(self, X: np.ndarray, w: int, start:int, end:int):
        """
        X: (N,T) z-scored; windows [t-w:t] -> predict X[:,t]
        range t in [start, end)
        """
        self.X, self.w = X, w
        self.ts = list(range(max(w, start), end))
    def __len__(self): return len(self.ts)
    def __getitem__(self, idx):
        t = self.ts[idx]
        xw = self.X[:, t-self.w:t]     # (N,w)
        yt = self.X[:, t]              # (N,)
        return torch.from_numpy(xw).float(), torch.from_numpy(yt).float()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_yaml", required=True)
    ap.add_argument("--model_anom_yaml", required=True)
    ap.add_argument("--training_yaml", required=True)
    ap.add_argument("--tcg_path", default="artifacts/tcg.json")
    ap.add_argument("--save_dir", default="artifacts")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    cfg = load_cfg(args.dataset_yaml, args.model_anom_yaml, args.save_dir)
    # Load data
    df = load_mts(cfg.dataset["mts"]["train"])
    X = df.to_numpy(dtype=float).T  # (N,T)
    # split train/val (nếu chưa có file riêng)
    val_ratio = 0.1
    T = X.shape[1]
    cut = int(T*(1 - val_ratio))
    # z-score theo train
    mu = X[:, :cut].mean(axis=1, keepdims=True)
    sd = X[:, :cut].std(axis=1, keepdims=True) + 1e-8
    Xn = (X - mu) / sd

    # windows & loaders
    w = cfg.model_causal.get("encoder", cfg.model_causal).get("window", 256)  # fallback nếu cấu hình gộp
    manom = cfg.model_causal if "encoder" in cfg.model_causal else None
    enc_cfg = cfg.model_causal["encoder"] if "encoder" in cfg.model_causal else cfg.model_causal
    attn_cfg = cfg.model_causal["attention"] if "attention" in cfg.model_causal else {"dmax": 32, "bias_mlp_hidden": 16}

    ds_tr = WindowDataset(Xn, w=w, start=0, end=cut)
    ds_va = WindowDataset(Xn, w=w, start=cut, end=T)
    dl_tr = DataLoader(ds_tr, batch_size=cfg.model_causal.get("training", {}).get("batch_size", 64), shuffle=True)
    dl_va = DataLoader(ds_va, batch_size=256, shuffle=False)

    # load TCG
    tcg = load_tcg(args.tcg_path)
    # init model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    det = Detector(enc_cfg=enc_cfg, attn_cfg=attn_cfg, tcg=tcg, device=str(device))
    # train
    lr = cfg.model_causal.get("training", {}).get("lr", 1e-3)
    epochs = cfg.model_causal.get("training", {}).get("epochs", 50)
    early = cfg.model_causal.get("training", {}).get("early_stop", 7)
    det.fit(dl_tr, dl_va, lr=lr, epochs=epochs, early_stop=early)

    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_path = os.path.join(args.save_dir, "detector.ckpt")
    torch.save(det.state_dict(), ckpt_path)
    # build errors on val for thresholds
    det.eval()
    errs = []
    with torch.no_grad():
        for xb, yb in dl_va:
            xb, yb = xb.to(device), yb.to(device)
            yhat = det(xb)
            errs.append((yhat - yb).abs().cpu().numpy())
    errors_val = np.concatenate(errs, axis=0)  # (T_val, N)
    # MAD + POT
    norm = MadNormalizer(eps=cfg.model_causal.get("threshold", {}).get("eps_mad", 1e-6))
    norm.fit(errors_val)
    pot = PotPerFeature(
        q0=cfg.model_causal.get("threshold", {}).get("pot_q0", 0.95),
        gamma=cfg.model_causal.get("threshold", {}).get("pot_gamma", 0.99),
        min_exceed=cfg.model_causal.get("threshold", {}).get("pot_min_exceed", 50),
    )
    a_val = norm.transform(errors_val)
    taus = pot.fit(a_val)

    # save artefacts
    np.savez(os.path.join(args.save_dir, "normalizer.npz"), mu=mu, sd=sd, med=norm.med, mad=norm.mad)
    with open(os.path.join(args.save_dir, "thresholds.json"), "w") as f:
        json.dump({str(k): float(v) for k,v in taus.items()}, f, indent=2)
    print(f"[OK] Saved: {ckpt_path}, normalizer.npz, thresholds.json")

if __name__ == "__main__":
    main()
