"""
Orchestrator: prepare -> build_tcg -> train_detector -> evaluate -> propagate
Dùng subprocess để tái sử dụng các CLI có sẵn.
"""
import argparse, os, subprocess, sys

def run(cmd: list):
    print("[run]", " ".join(cmd))
    p = subprocess.run(cmd)
    if p.returncode != 0:
        sys.exit(p.returncode)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_yaml", default="configs/dataset.yaml")
    ap.add_argument("--model_causal_yaml", default="configs/model_causal.yaml")
    ap.add_argument("--model_anom_yaml",   default="configs/model_anom.yaml")
    ap.add_argument("--training_yaml",     default="configs/training.yaml")
    ap.add_argument("--save_dir", default="artifacts")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip_prepare", action="store_true")
    ap.add_argument("--skip_propagate", action="store_true")
    args = ap.parse_args()

    env = os.environ.copy()
    # đảm bảo package được tìm thấy
    env["PYTHONPATH"] = f"src:{env.get('PYTHONPATH','')}"

    if not args.skip_prepare:
        run([sys.executable, "-m", "scripts.prepare_data",
             "--dataset_yaml", args.dataset_yaml])

    run([sys.executable, "-m", "scripts.build_tcg",
         "--dataset_yaml", args.dataset_yaml,
         "--model_causal_yaml", args.model_causal_yaml,
         "--save_dir", args.save_dir,
         "--device", args.device,
         "--seed", str(args.seed)],
        )

    run([sys.executable, "-m", "scripts.train_detector",
         "--dataset_yaml", args.dataset_yaml,
         "--model_anom_yaml", args.model_anom_yaml,
         "--training_yaml", args.training_yaml,
         "--tcg_path", os.path.join(args.save_dir, "tcg.json"),
         "--save_dir", args.save_dir,
         "--seed", str(args.seed)],
        )

    run([sys.executable, "-m", "scripts.evaluate",
         "--dataset_yaml", args.dataset_yaml,
         "--model_anom_yaml", args.model_anom_yaml,
         "--training_yaml", args.training_yaml,
         "--tcg_path", os.path.join(args.save_dir, "tcg.json"),
         "--save_dir", args.save_dir],
        )

    if not args.skip_propagate:
        run([sys.executable, "-m", "scripts.propagate",
             "--dataset_yaml", args.dataset_yaml,
             "--tcg_path", os.path.join(args.save_dir, "tcg.json"),
             "--labels_path", os.path.join(args.save_dir, "anomaly_labels.parquet"),
             "--time_index", "-1",
             "--lam", "0.1", "--L", "3", "--max_delta", "64",
             "--save_path", os.path.join(args.save_dir, "propagation.json")],
            )

if __name__ == "__main__":
    main()
