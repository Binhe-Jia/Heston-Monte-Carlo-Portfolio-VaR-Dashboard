from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import chi2
from scipy.stats import norm


@dataclass(frozen=True)
class BacktestResult:
    confidence: float
    observations: int
    breaches: int
    expected_breaches: float
    breach_rate: float
    kupiec_pvalue: float
    christoffersen_pvalue: float


def breach_backtest(realized_pnl: pd.Series, var_forecast: pd.Series, confidence: float) -> BacktestResult:
    aligned = pd.concat([realized_pnl.rename("pnl"), var_forecast.rename("var")], axis=1).dropna()
    if aligned.empty:
        raise ValueError("No overlapping PnL and VaR observations.")

    breaches = (-aligned["pnl"] > aligned["var"]).astype(int).to_numpy()
    count = int(breaches.sum())
    n = int(breaches.size)
    alpha = 1.0 - confidence
    return BacktestResult(
        confidence=confidence,
        observations=n,
        breaches=count,
        expected_breaches=n * alpha,
        breach_rate=count / n,
        kupiec_pvalue=_kupiec_pvalue(count, n, alpha),
        christoffersen_pvalue=_christoffersen_pvalue(breaches),
    )


def rolling_historical_var(
    pnl: pd.Series,
    confidence: float = 0.99,
    window: int = 252,
) -> pd.Series:
    if window < 30:
        raise ValueError("window must be at least 30.")
    losses = -pnl
    return losses.rolling(window).quantile(confidence).shift(1).rename("historical_var")


def rolling_gaussian_var(
    pnl: pd.Series,
    confidence: float = 0.99,
    window: int = 252,
) -> pd.Series:
    if window < 30:
        raise ValueError("window must be at least 30.")
    z = norm.ppf(confidence)
    mean = pnl.rolling(window).mean().shift(1)
    sigma = pnl.rolling(window).std(ddof=1).shift(1)
    return (-(mean - z * sigma)).rename("gaussian_var")


def backtest_standard_models(
    realized_pnl: pd.Series,
    confidence: float = 0.99,
    window: int = 252,
) -> dict[str, BacktestResult]:
    forecasts = {
        "Historical": rolling_historical_var(realized_pnl, confidence, window),
        "Gaussian": rolling_gaussian_var(realized_pnl, confidence, window),
    }
    return {
        name: breach_backtest(realized_pnl, forecast, confidence)
        for name, forecast in forecasts.items()
    }


def _kupiec_pvalue(breaches: int, observations: int, alpha: float) -> float:
    if observations == 0:
        return float("nan")
    phat = breaches / observations
    if breaches == 0 or breaches == observations:
        phat = np.clip(phat, 1e-12, 1 - 1e-12)
    likelihood_null = (1 - alpha) ** (observations - breaches) * alpha**breaches
    likelihood_alt = (1 - phat) ** (observations - breaches) * phat**breaches
    lr = -2.0 * np.log(max(likelihood_null, 1e-300) / max(likelihood_alt, 1e-300))
    return float(1.0 - chi2.cdf(lr, df=1))


def _christoffersen_pvalue(breaches: np.ndarray) -> float:
    if breaches.size < 2:
        return float("nan")

    previous = breaches[:-1]
    current = breaches[1:]
    n00 = int(((previous == 0) & (current == 0)).sum())
    n01 = int(((previous == 0) & (current == 1)).sum())
    n10 = int(((previous == 1) & (current == 0)).sum())
    n11 = int(((previous == 1) & (current == 1)).sum())

    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    pi01 = n01 / max(n00 + n01, 1)
    pi11 = n11 / max(n10 + n11, 1)

    unrestricted = _bernoulli_likelihood(n00, n01, pi01) * _bernoulli_likelihood(n10, n11, pi11)
    restricted = _bernoulli_likelihood(n00 + n10, n01 + n11, pi)
    lr = -2.0 * np.log(max(restricted, 1e-300) / max(unrestricted, 1e-300))
    return float(1.0 - chi2.cdf(lr, df=1))


def _bernoulli_likelihood(non_events: int, events: int, probability: float) -> float:
    p = float(np.clip(probability, 1e-12, 1 - 1e-12))
    return (1 - p) ** non_events * p**events
