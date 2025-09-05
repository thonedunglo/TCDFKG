# TCDFKG — Temporal Causal Graph + KG Verification + Anomaly + Propagation

Hệ thống 3 khối chính:
1) **TCG Builder**: TCDF attention → ước lượng trễ kiểu TCDF (saliency) → KG verification → hợp nhất điểm → `artifacts/tcg.json`.
2) **Anomaly Detection**: TempEnc + Edge-bias Attention (bias từ TCG) → MAD + POT → `detector.ckpt`, `thresholds.json`, `preds.parquet`, `anomaly_labels.parquet`.
3) **Propagation**: lan truyền bất thường theo TCG (suy giảm theo trễ + noisy-OR) → `propagation.json`.

> Lưu ý: **không clip outlier** ở bước chuẩn bị dữ liệu để giữ nguyên các bất thường.

---

## 0) Yêu cầu hệ thống

- Python ≥ 3.9  
- Thư viện chính: `numpy`, `pandas`, `pyyaml`, `scikit-learn`, `networkx`, `tqdm`, `matplotlib`, `scipy`, `torch`  
- Đọc/ghi Parquet: `pyarrow`  
- (Tùy chọn) GPU: cài PyTorch phù hợp CUDA.

Cài đặt (gợi ý):

```bash
# tại gốc repo
pip install -e .
pip install pyarrow
# pip install torch --index-url https://download.pytorch.org/whl/cpu   # nếu cần bản CPU
```

Thiết lập PYTHONPATH (cần cho mọi lệnh CLI):

```bash
export PYTHONPATH=src:$PYTHONPATH
```

---

## 1) Cấu trúc thư mục

```
configs/
  dataset.yaml
  model_causal.yaml
  model_anom.yaml
  training.yaml
data/
  raw/           # input thô (csv/parquet)
  processed/     # output từ prepare_data.py
  kg/
    extracted_schema.json
artifacts/       # outputs (tcg.json, ckpt, thresholds, preds, labels, propagation, plots, ...)
scripts/
  prepare_data.py
  build_tcg.py
  train_detector.py
  evaluate.py
  propagate.py
  eval_causal.py
  run_pipeline.py
  visualize.py
src/tcdfkg/
  ... (code)
```

---

## 2) Cấu hình

### `configs/dataset.yaml` (ví dụ)

```yaml
data_dir: ./data
raw_dir: ./data/raw
processed_dir: ./data/processed
mts:
  train: ./data/processed/train.parquet
  val:   ./data/processed/val.parquet
  test:  ./data/processed/test.parquet
kg:
  schema_path: ./data/kg/extracted_schema.json
time:
  freq: "1min"        # nếu muốn resample
split:
  val_ratio: 0.1
  test_ratio: 0.1
```

### `configs/model_causal.yaml` (ví dụ)

```yaml
tcdf:
  window: 256
  dilations: [1,2,4,8,16]
  kernel_size: 3
  channels: 64
  dropout: 0.1
  epochs: 20
  lr: 1e-3
  batch_size: 64
  tau_att: 0.02
  max_lag: 32

verify:
  lambda_mix: 0.5
  alpha_struct: 0.5
  beta_type: 0.3
  delta_rule: 0.2
  kappa_dist: 0.7

select:
  theta_edge: 0.15
```

### `configs/model_anom.yaml` (ví dụ)

```yaml
encoder:
  window: 256
  channels: 128
  kernels: [2,3,5,6]
attention:
  dmax: 32
  bias_mlp_hidden: 16
training:
  lr: 1e-3
  batch_size: 64
  epochs: 50
  early_stop: 7
threshold:
  pot_q0: 0.95
  pot_gamma: 0.99
  pot_min_exceed: 50
  eps_mad: 1e-6
```

### `configs/training.yaml` (ví dụ)

```yaml
device: "cpu"     # hoặc "cuda:0"
save_dir: "./artifacts"
log_dir: "./runs"
seed: 42
```

---

## 3) Chuẩn bị dữ liệu

Script **không cắt outlier**, chỉ resample/align/impute/split.

```bash
python -m scripts.prepare_data \
  --dataset_yaml configs/dataset.yaml \
  --input data/raw \
  --freq 1min \
  --val_ratio 0.1 --test_ratio 0.1 \
  --impute ffill_bfill
```

Kết quả:
- `data/processed/train.parquet`, `val.parquet`, `test.parquet`
- `data/processed/prepare_meta.json`

> Yêu cầu: các DataFrame có index thời gian (`timestamp`) và cột là biến đo (numeric).

---

## 4) Chạy pipeline 3 khối (lần lượt)

### 4.1 Build TCG

```bash
python -m scripts.build_tcg \
  --dataset_yaml configs/dataset.yaml \
  --model_causal_yaml configs/model_causal.yaml \
  --save_dir artifacts \
  --device cpu \
  --seed 42
```

Kết quả: `artifacts/tcg.json`

> (Tuỳ chọn) Lưu ma trận attention để vẽ heatmap:
> thêm vào `scripts/build_tcg.py` ngay sau `S_att = model.get_attention()`:
> ```python
> import numpy as np, os
> np.save(os.path.join(cfg.save_dir, "S_att.npy"), S_att)
> ```

### 4.2 Train detector + fit MAD/POT

```bash
python -m scripts.train_detector \
  --dataset_yaml configs/dataset.yaml \
  --model_anom_yaml configs/model_anom.yaml \
  --training_yaml configs/training.yaml \
  --tcg_path artifacts/tcg.json \
  --save_dir artifacts \
  --seed 42
```

Kết quả: `artifacts/detector.ckpt`, `artifacts/normalizer.npz`, `artifacts/thresholds.json`

### 4.3 Evaluate → dự báo & nhãn bất thường

```bash
python -m scripts.evaluate \
  --dataset_yaml configs/dataset.yaml \
  --model_anom_yaml configs/model_anom.yaml \
  --training_yaml configs/training.yaml \
  --tcg_path artifacts/tcg.json \
  --save_dir artifacts
```

Kết quả: `artifacts/preds.parquet`, `artifacts/anomaly_labels.parquet`

### 4.4 Propagation

```bash
python -m scripts.propagate \
  --dataset_yaml configs/dataset.yaml \
  --tcg_path artifacts/tcg.json \
  --labels_path artifacts/anomaly_labels.parquet \
  --time_index -1 \
  --lam 0.1 --L 3 --max_delta 64 \
  --save_path artifacts/propagation.json
```

Kết quả: `artifacts/propagation.json`

---

## 5) Visualization (tuỳ chọn)

### Vẽ TCG (top-200 cạnh)

```bash
python -m scripts.visualize tcg \
  --tcg_path artifacts/tcg.json \
  --dataset_yaml configs/dataset.yaml \
  --out artifacts/plots/tcg.png \
  --top_edges 200 \
  --layout kamada_kawai
```

### Heatmap attention hoặc adjacency

```bash
# nếu đã lưu S_att.npy:
python -m scripts.visualize heatmap \
  --satt_path artifacts/S_att.npy \
  --dataset_yaml configs/dataset.yaml \
  --out artifacts/plots/attention.png

# nếu chưa có S_att: dùng adjacency (score) suy từ TCG
python -m scripts.visualize heatmap \
  --tcg_path artifacts/tcg.json \
  --dataset_yaml configs/dataset.yaml \
  --out artifacts/plots/adjacency.png
```

### Một biến: truth vs pred + dấu bất thường

```bash
python -m scripts.visualize anomaly \
  --truth_path data/processed/test.parquet \
  --pred_path artifacts/preds.parquet \
  --labels_path artifacts/anomaly_labels.parquet \
  --var <TEN_COT> \
  --out artifacts/plots/anomaly_<TEN_COT>.png
```

### Propagation timeline

```bash
python -m scripts.visualize propagation \
  --prop_path artifacts/propagation.json \
  --dataset_yaml configs/dataset.yaml \
  --out artifacts/plots/propagation.png \
  --topk 10
```

---

## 6) Đánh giá đồ thị nhân quả (nếu có ground-truth)

Chuẩn bị ground-truth cạnh (JSON/CSV):
- JSON: `{"edges":[{"src":i,"dst":j, "lag": <optional>}, ...]}`
- CSV: 2 cột `src,dst`

Chạy:

```bash
python -m scripts.eval_causal \
  --pred_tcg artifacts/tcg.json \
  --truth_path data/kg/ground_truth.json \
  --score_threshold 0.5 \
  --tau_tol 2 \
  --save_path artifacts/metrics_causal.json
```

Kết quả: `artifacts/metrics_causal.json` chứa Precision / Recall / F1 / AP / SHD.

---

## 7) Orchestrator (one-shot)

Gom tất cả bước thành **một lệnh**:

```bash
python -m scripts.run_pipeline \
  --dataset_yaml configs/dataset.yaml \
  --model_causal_yaml configs/model_causal.yaml \
  --model_anom_yaml configs/model_anom.yaml \
  --training_yaml configs/training.yaml \
  --save_dir artifacts \
  --device cpu \
  --seed 42
```

Tham số:
- `--skip_prepare`: bỏ qua bước prepare nếu bạn đã có `data/processed/*.parquet`
- `--skip_propagate`: bỏ qua bước propagate

---

## 8) Tips & Troubleshooting

- **ModuleNotFoundError: tcdfkg** → chắc chắn đã `export PYTHONPATH=src:$PYTHONPATH`.
- **pandas.read_parquet lỗi** → cài `pyarrow`.
- **Thiếu Torch** → cài đúng bản (CPU hoặc GPU theo môi trường).
- **Thiết bị**: có GPU thì chỉnh `configs/training.yaml: device: "cuda:0"` (và cài torch CUDA).
- **Lưu ý dữ liệu**: index thời gian (`timestamp`) và numeric columns; script sẽ `z-score` theo **train**.
- **Attention heatmap**: muốn đúng S_att từ TCDF, hãy lưu `artifacts/S_att.npy` trong `build_tcg.py`.

---

## 9) Kiểm thử nhanh

```bash
export PYTHONPATH=src:$PYTHONPATH
pytest -q
```

Bộ test tối thiểu kiểm tra: KG verification, ước lượng trễ TCDF (toy), edge-bias attention, và POT.

---

## 10) Ghi chú thiết kế

- **Ước lượng trễ**: dùng **saliency theo thời gian** trên CNN nhân quả giãn nở (TCDF-style).
- **Edge-bias Attention**: logit = `Q_j·K_i + MLP([S_norm, d_norm])`, bias từ TCG.
- **Anomaly thresholding**: MAD chuẩn hoá lỗi dự báo (per-feature) + POT (per-feature).
- **Propagation**: xác suất lan truyền \(p_{j,t+\Delta} = 1 - \prod_i (1 - p_{i,t} \cdot S_{ij} \cdot e^{-\lambda d_{ij}})\), tối đa L bậc.

---

## 11) Giấy phép

(Điền thông tin license của bạn tại đây, nếu cần)

```
MIT / Apache-2.0 / Proprietary
```
