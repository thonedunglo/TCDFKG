import os, random, numpy as np
import pytest
import sys  # ensure project src is on path

# add src and build/lib directories for package discovery
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "build", "lib")))  #
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))  #

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
