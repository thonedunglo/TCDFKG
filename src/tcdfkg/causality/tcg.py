import json
from typing import Dict, Any

def save_tcg(tcg: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(tcg, f, ensure_ascii=False, indent=2)

def load_tcg(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
