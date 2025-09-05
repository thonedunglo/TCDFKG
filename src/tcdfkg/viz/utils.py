from __future__ import annotations
import os
from typing import Optional

def ensure_dir(p: str) -> None:
    d = os.path.dirname(p)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def default_figsize(n_items: int, base=(8, 5)) -> tuple[int, int]:
    w, h = base
    if n_items > 40:
        w = int(w * 1.5)
    if n_items > 80:
        w = int(w * 1.8)
    return (w, h)
