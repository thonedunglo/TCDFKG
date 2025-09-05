import numpy as np
from tcdfkg.anomaly.pot import PotPerFeature

def test_pot_threshold_flags_tail():
    rng = np.random.default_rng(123)
    # feature 0: normal(0,1) with rare big tail
    base = rng.normal(0,1, size=1000)
    tail = np.array([6,7,5,8,9])  # heavy exceedances
    x0 = np.concatenate([base, tail])
    # feature 1: lighter
    x1 = rng.normal(0,1, size=x0.shape[0])

    val = np.stack([x0, x1], axis=1)  # (T, N=2)
    pot = PotPerFeature(q0=0.95, gamma=0.99, min_exceed=20)
    taus = pot.fit(val)
    assert taus[0] > np.quantile(x0, 0.95), "τ0 should be above 95th quantile due to heavy tail"
    # predict: last few heavy points should be flagged
    Y = pot.predict(val, taus)
    assert Y[-1,0] == 1 and Y[-2,0] == 1
