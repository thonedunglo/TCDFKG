import argparse, os, json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from tcdfkg.utils.config import load_cfg
from tcdfkg.data.loaders import load_mts
from tcdfkg.causality.tcg import load_tcg
from tcdfkg.anomaly.detector import Detector
from tcdfkg.anomaly.mad import MadNormalizer
from tcdfkg.anomaly.pot import PotPerFeature
from tcdfkg.eval.metrics_anom import evaluate_anomaly

class WindowDataset(Dataset):
    def __init__(self, Xn: np.ndarray, w: int, start:int, end:int):
        self.X, self.w = Xn, w
        self.ts = list(range(max(w, start), end))
    def __len__(self): return len(self.ts)
    def __getitem__(self, idx):
        t = self.ts[idx]
        xw = self.X[:, t-self.w:t]
        yt = self.X[:, t]
        return torch.from_numpy(xw).float(), torch.from_numpy(yt).float()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_yaml", required=True)
    ap.add_argument("--model_anom_yaml", required=True)
    ap.add_argument("--training_yaml", required=True)
    ap.add_argument("--tcg_path", default="artifacts/tcg.json")
    ap.add_argument("--save_dir", default="artifacts")
    ap.add_argument("--use_test_file", action="store_true", help="dùng đường dẫn test nếu có trong dataset.yaml")
    args = ap.parse_args()

    cfg = load_cfg(args.dataset_yaml, args.model_anom_yaml, args.save_dir)
    # load data
    if args.use_test_file and "test" in cfg.dataset.get("mts", {}):
        df = load_mts(cfg.dataset["mts"]["test"])
    else:
        df = load_mts(cfg.dataset["mts"]["train"])   # fallback: dùng đuôi của train làm test
    X = df.to_numpy(dtype=float).T
    # load normalizer
    nz = np.load(os.path.join(args.save_dir, "normalizer.npz"))
    mu, sd = nz["mu"], nz["sd"]
    Xn = (X - mu) / sd

    w = cfg.model_causal.get("encoder", cfg.model_causal).get("window", 256)
    ds = WindowDataset(Xn, w=w, start=0, end=X.shape[1])
    dl = DataLoader(ds, batch_size=256, shuffle=False)

    # load TCG & model
    tcg = load_tcg(args.tcg_path)
    enc_cfg = cfg.model_causal["encoder"] if "encoder" in cfg.model_causal else cfg.model_causal
    attn_cfg = cfg.model_causal["attention"] if "attention" in cfg.model_causal else {"dmax": 32, "bias_mlp_hidden": 16}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    det = Detector(enc_cfg=enc_cfg, attn_cfg=attn_cfg, tcg=tcg, device=str(device))
    ckpt = torch.load(os.path.join(args.save_dir, "detector.ckpt"), map_location=device)
    det.load_state_dict(ckpt); det.eval()

    # predict
    preds, gts = [], []
    with torch.no_grad():
        for xb, yb in dl:
            xb = xb.to(device); yb = yb.to(device)
            yhat = det(xb)
            preds.append(yhat.cpu().numpy()); gts.append(yb.cpu().numpy())
    Yhat = np.concatenate(preds, axis=0)  # (T_eff, N)
    Y = np.concatenate(gts, axis=0)

    # anomaly scores
    errors = np.abs(Yhat - Y)
    norm = MadNormalizer(eps=cfg.model_causal.get("threshold", {}).get("eps_mad", 1e-6))
    norm.med, norm.mad = nz["med"], nz["mad"]
    A = norm.transform(errors)

    with open(os.path.join(args.save_dir, "thresholds.json")) as f:
        taus = {int(k): float(v) for k,v in json.load(f).items()}
    pot = PotPerFeature()
    Ypred = pot.predict(A, taus=taus)  # (T_eff, N)

    # nếu bạn có y_true, nạp và tính metrics; ở đây minh hoạ giả định không có nhãn.
    # Lưu outputs
    pd.DataFrame(Yhat, columns=df.columns[:Yhat.shape[1]]).to_parquet(os.path.join(args.save_dir, "preds.parquet"))
    pd.DataFrame(Ypred, columns=df.columns[:Ypred.shape[1]]).to_parquet(os.path.join(args.save_dir, "anomaly_labels.parquet"))
    print("[OK] Saved preds.parquet & anomaly_labels.parquet")

if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    main()
