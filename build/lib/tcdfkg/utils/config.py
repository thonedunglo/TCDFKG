import os, yaml
from dataclasses import dataclass
from typing import Any, Dict

def _read_yaml(p: str) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

@dataclass
class Cfg:
    dataset: Dict[str, Any]
    model_causal: Dict[str, Any]
    save_dir: str
    device: str = "cuda"

def load_cfg(dataset_yaml: str, model_causal_yaml: str, save_dir: str, device: str = "cpu") -> Cfg:
    ds = _read_yaml(dataset_yaml)
    mc = _read_yaml(model_causal_yaml)
    os.makedirs(save_dir, exist_ok=True)
    return Cfg(dataset=ds, model_causal=mc, save_dir=save_dir, device=device)
