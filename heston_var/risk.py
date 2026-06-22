from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RiskResult:
    model: str
    confidence: float
    var: float
    expected_shortfall: float
    mean_pnl: float
    volatility: float
    observations: int

    @property
    def var_percent_of_capital(self) -> float:
        return self.var


def risk_from_pnl(pnl: np.ndarray, confidence: float, model: str) -> RiskResult:
    values = np.asarray(pnl, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("PnL sample is empty.")
    if not 0.5 < confidence < 1.0:
        raise ValueError("Confidence must be between 0.5 and 1.0.")

    losses = -values
    var = float(np.quantile(losses, confidence))
    tail = losses[losses >= var]
    expected_shortfall = float(tail.mean()) if tail.size else var
    return RiskResult(
        model=model,
        confidence=confidence,
        var=var,
        expected_shortfall=expected_shortfall,
        mean_pnl=float(values.mean()),
        volatility=float(values.std(ddof=1)) if values.size > 1 else 0.0,
        observations=int(values.size),
    )
