# ==== Config ====
PY      ?= python
DS      ?= configs/dataset.yaml
MC      ?= configs/model_causal.yaml
MA      ?= configs/model_anom.yaml
TR      ?= configs/training.yaml
SAVE    ?= artifacts
TCG     ?= $(SAVE)/tcg.json
SEED    ?= 42
DEVICE  ?= cpu   # hoặc cuda:0

# ==== Targets ====
.PHONY: help tcg detector evaluate anomaly clean-anomaly all

help:
	@echo "Targets:"
	@echo "  tcg         - build Temporal Causal Graph (TCDF attention + KG verify)"
	@echo "  detector    - train Edge-bias detector + fit MAD/POT thresholds"
	@echo "  evaluate    - run inference on test and save preds & anomaly labels"
	@echo "  anomaly     - detector + evaluate"
	@echo "  clean-anomaly - remove detector/threshold/prediction artifacts"
	@echo "  all         - tcg + anomaly"

tcg:
	$(PY) -m scripts.build_tcg \
	 --dataset_yaml $(DS) \
	 --model_causal_yaml $(MC) \
	 --save_dir $(SAVE) \
	 --device $(DEVICE) \
	 --seed $(SEED)

detector: tcg
	$(PY) -m scripts.train_detector \
	 --dataset_yaml $(DS) \
	 --model_anom_yaml $(MA) \
	 --training_yaml $(TR) \
	 --tcg_path $(TCG) \
	 --save_dir $(SAVE) \
	 --seed $(SEED)

evaluate: detector
	$(PY) -m scripts.evaluate \
	 --dataset_yaml $(DS) \
	 --model_anom_yaml $(MA) \
	 --training_yaml $(TR) \
	 --tcg_path $(TCG) \
	 --save_dir $(SAVE)

anomaly: detector evaluate

clean-anomaly:
	rm -f $(SAVE)/detector.ckpt $(SAVE)/normalizer.npz $(SAVE)/thresholds.json \
	      $(SAVE)/preds.parquet $(SAVE)/anomaly_labels.parquet

all: tcg anomaly
