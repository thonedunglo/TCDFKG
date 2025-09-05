import argparse, os, numpy as np
from src.tcdfkg.utils.config import load_cfg
from src.tcdfkg.utils.seed import set_seed
from src.tcdfkg.data.loaders import load_mts
from src.tcdfkg.causality.tcdf import TCDF
from src.tcdfkg.causality.delay import estimate_lags_tcdf_saliency
from src.tcdfkg.kg.verification import KGVerifier
from src.tcdfkg.causality.scorer import fuse_scores
from src.tcdfkg.causality.tcg import save_tcg

def _infer_types_from_columns(cols):
    return {i: str(c) for i, c in enumerate(cols)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_yaml", required=True)
    ap.add_argument("--model_causal_yaml", required=True)
    ap.add_argument("--save_dir", default="./artifacts")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    cfg = load_cfg(args.dataset_yaml, args.model_causal_yaml, args.save_dir, device=args.device)

    # 1) Load train data
    df = load_mts(cfg.dataset["mts"]["train"])
    X = df.to_numpy(dtype=float).T  # (N,T)
    # z-score
    X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)

    # 2) Train TCDF + attention candidates
    tcdf_cfg = cfg.model_causal["tcdf"]
    model = TCDF(num_series=X.shape[0],
                 window=tcdf_cfg["window"],
                 dilations=tcdf_cfg["dilations"],
                 kernel_size=tcdf_cfg["kernel_size"],
                 hid_ch=tcdf_cfg["channels"],
                 dropout=tcdf_cfg["dropout"],
                 device=cfg.device)
    model.fit(X, epochs=tcdf_cfg["epochs"], lr=tcdf_cfg["lr"], batch_size=tcdf_cfg["batch_size"], verbose=True)
    S_att = model.get_attention()
    np.save(os.path.join(cfg.save_dir, "S_att.npy"), S_att)

    # 3) Delay estimation (TCDF saliency-based)
    d_hat = estimate_lags_tcdf_saliency(model, X, max_lag=tcdf_cfg["max_lag"])

    # 4) Build candidate list by tau_att
    tau_att = float(tcdf_cfg["tau_att"])
    edges = []
    N = X.shape[0]
    for j in range(N):
        for i in range(N):
            if i == j: continue
            s_ij = float(S_att[i, j])
            if s_ij >= tau_att:
                edges.append({"src": i, "dst": j, "lag": int(d_hat.get((i,j), 0)), "s_att": s_ij})

    # 5) KG verify
    types = _infer_types_from_columns(df.columns)
    meta = {"candidates": [(e["src"], e["dst"]) for e in edges], "type": types}
    ver = KGVerifier(cfg.dataset["kg"]["schema_path"],
                     alpha=cfg.model_causal["verify"]["alpha_struct"],
                     beta=cfg.model_causal["verify"]["beta_type"],
                     delta=cfg.model_causal["verify"]["delta_rule"],
                     kappa=cfg.model_causal["verify"]["kappa_dist"])
    s_kg = ver.score(meta)

    # 6) Fuse and save TCG
    tcg = fuse_scores(edges, s_kg,
                      lambda_mix=cfg.model_causal["verify"]["lambda_mix"],
                      theta_edge=cfg.model_causal["select"]["theta_edge"])
    os.makedirs(cfg.save_dir, exist_ok=True)
    out_path = os.path.join(cfg.save_dir, "tcg.json")
    save_tcg(tcg, out_path)
    print(f"[OK] Saved TCG with {len(tcg['edges'])} edges to {out_path}")

if __name__ == "__main__":
    main()
