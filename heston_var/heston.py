from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


TRADING_DAYS = 252


@dataclass(frozen=True)
class HestonParams:
    mu: float
    kappa: float
    theta: float
    xi: float
    rho: float
    v0: float

    def sanitized(self) -> "HestonParams":
        theta = max(float(self.theta), 1e-8)
        v0 = max(float(self.v0), 1e-8)
        return HestonParams(
            mu=float(self.mu),
            kappa=max(float(self.kappa), 1e-4),
            theta=theta,
            xi=max(float(self.xi), 1e-4),
            rho=float(np.clip(self.rho, -0.98, 0.98)),
            v0=v0,
        )


def estimate_heston_params_from_returns(
    returns: pd.Series,
    trading_days: int = TRADING_DAYS,
    ewma_decay: float = 0.94,
) -> HestonParams:
    """Estimate pragmatic Heston parameters from historical returns.

    This is intentionally a real-world return calibration, not an option-implied
    risk-neutral calibration. It uses EWMA variance as a latent variance proxy.
    """

    r = pd.Series(returns).dropna().astype(float)
    if r.size < 30:
        raise ValueError("At least 30 return observations are required for Heston estimation.")

    variance_proxy = _ewma_variance_proxy(r.to_numpy(), ewma_decay)
    dt = 1.0 / trading_days

    mu = float(r.mean() * trading_days)
    theta = float(np.mean(variance_proxy) * trading_days)
    v0 = float(variance_proxy[-1] * trading_days)

    x = variance_proxy[:-1] * trading_days
    y = variance_proxy[1:] * trading_days
    slope, intercept = _linear_fit(x, y)
    slope = float(np.clip(slope, 1e-4, 0.9999))
    kappa = float(max((1.0 - slope) / dt, 1e-4))
    if intercept > 0 and kappa > 1e-4:
        theta = float(max(intercept / (1.0 - slope), 1e-8))

    dv = np.diff(variance_proxy * trading_days)
    variance_level = np.sqrt(np.maximum(x, 1e-8) * dt)
    xi_samples = dv / variance_level
    xi = float(np.nanstd(xi_samples, ddof=1))
    if not np.isfinite(xi) or xi <= 0:
        xi = max(theta**0.5, 1e-4)

    price_shock = (r.to_numpy()[1:] - r.mean()) / np.sqrt(np.maximum(variance_proxy[1:], 1e-10))
    vol_shock = xi_samples / max(xi, 1e-8)
    rho = _safe_corr(price_shock, vol_shock)

    return HestonParams(mu=mu, kappa=kappa, theta=theta, xi=xi, rho=rho, v0=v0).sanitized()


def estimate_heston_params_frame(returns: pd.DataFrame) -> dict[str, HestonParams]:
    return {column: estimate_heston_params_from_returns(returns[column]) for column in returns.columns}


def _ewma_variance_proxy(returns: np.ndarray, decay: float) -> np.ndarray:
    if not 0.0 < decay < 1.0:
        raise ValueError("EWMA decay must be between 0 and 1.")
    variance = np.empty_like(returns, dtype=float)
    seed_window = returns[: min(30, returns.size)]
    variance[0] = max(float(np.var(seed_window, ddof=1)), 1e-8)
    for i in range(1, returns.size):
        variance[i] = decay * variance[i - 1] + (1.0 - decay) * returns[i - 1] ** 2
    return np.maximum(variance, 1e-10)


def _linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if np.std(x) < 1e-12:
        return 0.9, float(np.mean(y) * 0.1)
    slope, intercept = np.polyfit(x, y, deg=1)
    return float(slope), float(intercept)


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or np.std(x[mask]) < 1e-12 or np.std(y[mask]) < 1e-12:
        return -0.3
    return float(np.clip(np.corrcoef(x[mask], y[mask])[0, 1], -0.98, 0.98))
