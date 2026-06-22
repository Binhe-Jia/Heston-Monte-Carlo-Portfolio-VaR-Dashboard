from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from heston_var.heston import HestonParams, TRADING_DAYS, estimate_heston_params_frame
from heston_var.portfolio import Portfolio
from heston_var.risk import RiskResult, risk_from_pnl


@dataclass(frozen=True)
class SimulationDiagnostics:
    paths: int
    horizon_days: int
    assets: int
    elapsed_seconds: float
    backend: str
    antithetic: bool


@dataclass(frozen=True)
class SimulationResult:
    pnl: np.ndarray
    risk: RiskResult
    diagnostics: SimulationDiagnostics
    params: dict[str, HestonParams]


def heston_portfolio_var(
    portfolio: Portfolio,
    returns: pd.DataFrame,
    confidence: float = 0.99,
    paths: int = 20_000,
    horizon_days: int = 1,
    seed: int | None = 42,
    backend: str = "auto",
    antithetic: bool = True,
) -> SimulationResult:
    aligned = portfolio.align_returns(returns)
    params = estimate_heston_params_frame(aligned)
    start = time.perf_counter()
    pnl, backend_used = simulate_heston_portfolio_pnl(
        portfolio=portfolio,
        returns=aligned,
        params=params,
        paths=paths,
        horizon_days=horizon_days,
        seed=seed,
        backend=backend,
        antithetic=antithetic,
    )
    elapsed = time.perf_counter() - start
    risk = risk_from_pnl(pnl, confidence, "Heston MC")
    return SimulationResult(
        pnl=pnl,
        risk=risk,
        diagnostics=SimulationDiagnostics(
            paths=paths,
            horizon_days=horizon_days,
            assets=len(portfolio.tickers),
            elapsed_seconds=elapsed,
            backend=backend_used,
            antithetic=antithetic,
        ),
        params=params,
    )


def simulate_heston_portfolio_pnl(
    portfolio: Portfolio,
    returns: pd.DataFrame,
    params: dict[str, HestonParams],
    paths: int,
    horizon_days: int,
    seed: int | None,
    trading_days: int = TRADING_DAYS,
    backend: str = "auto",
    antithetic: bool = True,
) -> tuple[np.ndarray, str]:
    if paths <= 0:
        raise ValueError("paths must be positive.")
    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive.")
    if backend not in {"auto", "python", "rust"}:
        raise ValueError("backend must be one of: auto, python, rust.")

    aligned = portfolio.align_returns(returns)
    tickers = list(portfolio.tickers)
    correlation = _nearest_correlation(aligned.corr().to_numpy())

    if backend in {"auto", "rust"}:
        try:
            pnl = _simulate_heston_portfolio_pnl_rust(
                portfolio=portfolio,
                tickers=tickers,
                params=params,
                correlation=correlation,
                paths=paths,
                horizon_days=horizon_days,
                seed=seed,
                antithetic=antithetic,
            )
            return pnl, "rust-pyo3"
        except ImportError as exc:
            if backend == "rust":
                raise RuntimeError("Rust backend requested but heston_var_rust is not installed.") from exc
        except AttributeError as exc:
            if backend == "rust":
                raise RuntimeError("Installed heston_var_rust lacks the portfolio simulator.") from exc
        except Exception as exc:
            if backend == "rust":
                raise RuntimeError(f"Rust backend failed: {exc}") from exc

    chol = np.linalg.cholesky(correlation)
    rng = np.random.default_rng(seed)

    s_rel = np.ones((paths, len(tickers)), dtype=float)
    variances = np.array([params[t].sanitized().v0 for t in tickers], dtype=float)
    variances = np.broadcast_to(variances, s_rel.shape).copy()
    dt = 1.0 / trading_days

    mu = np.array([params[t].sanitized().mu for t in tickers], dtype=float)
    kappa = np.array([params[t].sanitized().kappa for t in tickers], dtype=float)
    theta = np.array([params[t].sanitized().theta for t in tickers], dtype=float)
    xi = np.array([params[t].sanitized().xi for t in tickers], dtype=float)
    rho = np.array([params[t].sanitized().rho for t in tickers], dtype=float)

    if antithetic:
        pair_paths = (paths + 1) // 2
        s_plus = np.ones((pair_paths, len(tickers)), dtype=float)
        s_minus = np.ones((pair_paths, len(tickers)), dtype=float)
        initial_variances = np.array([params[t].sanitized().v0 for t in tickers], dtype=float)
        variances_plus = np.broadcast_to(initial_variances, s_plus.shape).copy()
        variances_minus = np.broadcast_to(initial_variances, s_minus.shape).copy()

        for _ in range(horizon_days):
            independent_price = rng.standard_normal((pair_paths, len(tickers)))
            z_price = independent_price @ chol.T
            z_extra = rng.standard_normal((pair_paths, len(tickers)))
            z_var = rho * z_price + np.sqrt(np.maximum(1.0 - rho**2, 1e-8)) * z_extra

            v_plus = np.maximum(variances_plus, 0.0)
            variances_plus = (
                variances_plus
                + kappa * (theta - v_plus) * dt
                + xi * np.sqrt(v_plus * dt) * z_var
            )
            variances_plus = np.maximum(variances_plus, 1e-10)
            log_return_plus = (mu - 0.5 * v_plus) * dt + np.sqrt(v_plus * dt) * z_price
            s_plus *= np.exp(log_return_plus)

            v_minus = np.maximum(variances_minus, 0.0)
            variances_minus = (
                variances_minus
                + kappa * (theta - v_minus) * dt
                - xi * np.sqrt(v_minus * dt) * z_var
            )
            variances_minus = np.maximum(variances_minus, 1e-10)
            log_return_minus = (mu - 0.5 * v_minus) * dt - np.sqrt(v_minus * dt) * z_price
            s_minus *= np.exp(log_return_minus)

        paired = np.empty((pair_paths * 2, len(tickers)), dtype=float)
        paired[0::2] = s_plus
        paired[1::2] = s_minus
        portfolio_returns = (paired[:paths] - 1.0) @ portfolio.weights
        return portfolio_returns * portfolio.initial_capital, "python-numpy"

    for _ in range(horizon_days):
        independent_price = rng.standard_normal((paths, len(tickers)))
        z_price = independent_price @ chol.T
        z_extra = rng.standard_normal((paths, len(tickers)))
        z_var = rho * z_price + np.sqrt(np.maximum(1.0 - rho**2, 1e-8)) * z_extra

        v_pos = np.maximum(variances, 0.0)
        variances = (
            variances
            + kappa * (theta - v_pos) * dt
            + xi * np.sqrt(v_pos * dt) * z_var
        )
        variances = np.maximum(variances, 1e-10)
        log_return = (mu - 0.5 * v_pos) * dt + np.sqrt(v_pos * dt) * z_price
        s_rel *= np.exp(log_return)

    portfolio_returns = (s_rel - 1.0) @ portfolio.weights
    return portfolio_returns * portfolio.initial_capital, "python-numpy"


def _simulate_heston_portfolio_pnl_rust(
    portfolio: Portfolio,
    tickers: list[str],
    params: dict[str, HestonParams],
    correlation: np.ndarray,
    paths: int,
    horizon_days: int,
    seed: int | None,
    antithetic: bool,
) -> np.ndarray:
    import heston_var_rust

    sanitized = [params[t].sanitized() for t in tickers]
    seed_u64 = int(seed if seed is not None else np.random.SeedSequence().entropy) & ((1 << 64) - 1)
    args = (
        [p.mu for p in sanitized],
        [p.kappa for p in sanitized],
        [p.theta for p in sanitized],
        [p.xi for p in sanitized],
        [p.rho for p in sanitized],
        [p.v0 for p in sanitized],
        portfolio.weights.astype(float).tolist(),
        np.asarray(correlation, dtype=float).ravel(order="C").tolist(),
        float(portfolio.initial_capital),
        int(paths),
        int(horizon_days),
        seed_u64,
    )
    try:
        pnl = heston_var_rust.simulate_heston_portfolio_pnl(*args, bool(antithetic))
    except TypeError as exc:
        if "positional arguments" not in str(exc):
            raise
        if antithetic:
            raise RuntimeError(
                "The loaded Rust extension is an older build without antithetic support. "
                "Restart the notebook kernel after reinstalling the latest wheel, or run with "
                "antithetic=False."
            ) from exc
        pnl = heston_var_rust.simulate_heston_portfolio_pnl(*args)
    return np.asarray(pnl, dtype=float)


def _nearest_correlation(matrix: np.ndarray) -> np.ndarray:
    corr = np.asarray(matrix, dtype=float)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr = (corr + corr.T) / 2.0
    np.fill_diagonal(corr, 1.0)
    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals = np.maximum(eigvals, 1e-8)
    corr = (eigvecs * eigvals) @ eigvecs.T
    d = np.sqrt(np.diag(corr))
    corr = corr / np.outer(d, d)
    np.fill_diagonal(corr, 1.0)
    return corr
