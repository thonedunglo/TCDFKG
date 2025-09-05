import os, random, numpy as np
import pytest

try:
    import torch
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

@pytest.fixture(autouse=True)
def _seed_everything():
    random.seed(123)
    np.random.seed(123)
    if HAS_TORCH:
        torch.manual_seed(123)  # type: ignore
