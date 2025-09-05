import numpy as np
import pytest
from tcdfkg.causality.tcdf import TCDF

def test_fit_raises_when_no_windows():
    X = np.zeros((2, 5))
    model = TCDF(num_series=2, window=10, dilations=(1,), kernel_size=3, hid_ch=4, dropout=0.0, device="cpu")
    with pytest.raises(ValueError) as exc:
        model.fit(X, epochs=1, lr=1e-3, batch_size=2, verbose=False)
    msg = str(exc.value)
    assert "window" in msg and "X.shape" in msg
