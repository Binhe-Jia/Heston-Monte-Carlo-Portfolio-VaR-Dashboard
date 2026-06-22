from __future__ import annotations

import numpy as np
import pandas as pd


def synthetic_returns(
    assets: int = 4,
    observations: int = 900,
    seed: int = 7,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    names = [f"ASSET{i + 1}" for i in range(assets)]
    base_corr = np.full((assets, assets), 0.35)
    np.fill_diagonal(base_corr, 1.0)
    chol = np.linalg.cholesky(base_corr)

    vol_state = np.full(assets, 0.012)
    rows = []
    for _ in range(observations):
        vol_state = 0.94 * vol_state + 0.06 * np.abs(rng.normal(0.012, 0.006, assets))
        shocks = rng.standard_normal(assets) @ chol.T
        rows.append(0.00035 + vol_state * shocks)

    index = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=observations)
    return pd.DataFrame(rows, index=index, columns=names)
