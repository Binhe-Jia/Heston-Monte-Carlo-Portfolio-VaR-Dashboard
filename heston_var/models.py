from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from heston_var.portfolio import Portfolio
from heston_var.risk import RiskResult, risk_from_pnl


def portfolio_horizon_pnl(portfolio: Portfolio, returns: pd.DataFrame, horizon_days: int = 1) -> pd.Series:
    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive.")
    if horizon_days == 1:
        return portfolio.historical_pnl(returns)

    aligned = portfolio.align_returns(returns)
    if len(aligned) < horizon_days:
        raise ValueError(f"Need at least {horizon_days} return observations for this horizon.")
    portfolio_returns = pd.Series(aligned.to_numpy() @ portfolio.weights, index=aligned.index)
    horizon_returns = (1.0 + portfolio_returns).rolling(horizon_days).apply(np.prod, raw=True) - 1.0
    return (horizon_returns.dropna() * portfolio.initial_capital).rename("pnl")


def historical_var(
    portfolio: Portfolio,
    returns: pd.DataFrame,
    confidence: float,
    horizon_days: int = 1,
) -> RiskResult:
    pnl = portfolio_horizon_pnl(portfolio, returns, horizon_days).to_numpy()
    return risk_from_pnl(pnl, confidence, "Historical")


def gaussian_var(
    portfolio: Portfolio,
    returns: pd.DataFrame,
    confidence: float,
    horizon_days: int = 1,
) -> RiskResult:
    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive.")
    pnl = portfolio.historical_pnl(returns).to_numpy()
    if pnl.size < 2:
        raise ValueError("Gaussian VaR needs at least two PnL observations.")

    mu = float(pnl.mean()) * horizon_days
    sigma = float(pnl.std(ddof=1)) * horizon_days**0.5
    z = norm.ppf(confidence)
    var = -(mu - z * sigma)
    expected_shortfall = -(mu - sigma * norm.pdf(z) / (1.0 - confidence))
    return RiskResult(
        model="Gaussian",
        confidence=confidence,
        var=float(var),
        expected_shortfall=float(expected_shortfall),
        mean_pnl=mu,
        volatility=sigma,
        observations=int(pnl.size),
    )


def ewma_var(
    portfolio: Portfolio,
    returns: pd.DataFrame,
    confidence: float,
    decay: float = 0.94,
    horizon_days: int = 1,
) -> RiskResult:
    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive.")
    pnl = portfolio.historical_pnl(returns).to_numpy()
    if pnl.size < 2:
        raise ValueError("EWMA VaR needs at least two PnL observations.")
    if not 0.0 < decay < 1.0:
        raise ValueError("EWMA decay must be between 0 and 1.")

    demeaned = pnl - pnl.mean()
    weights = (1.0 - decay) * decay ** np.arange(pnl.size - 1, -1, -1)
    weights = weights / weights.sum()
    variance = float(np.sum(weights * demeaned**2))
    sigma = variance**0.5 * horizon_days**0.5
    mu = float(pnl.mean()) * horizon_days
    z = norm.ppf(confidence)
    var = -(mu - z * sigma)
    expected_shortfall = -(mu - sigma * norm.pdf(z) / (1.0 - confidence))
    return RiskResult(
        model="EWMA",
        confidence=confidence,
        var=float(var),
        expected_shortfall=float(expected_shortfall),
        mean_pnl=mu,
        volatility=float(sigma),
        observations=int(pnl.size),
    )
