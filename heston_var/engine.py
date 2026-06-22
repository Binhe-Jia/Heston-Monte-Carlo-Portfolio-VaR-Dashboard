from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from heston_var.models import ewma_var, gaussian_var, historical_var
from heston_var.portfolio import Portfolio
from heston_var.risk import RiskResult
from heston_var.simulation import SimulationDiagnostics, heston_portfolio_var


@dataclass(frozen=True)
class PortfolioVaRReport:
    risk_results: tuple[RiskResult, ...]
    heston_diagnostics: SimulationDiagnostics

    def to_frame(self) -> pd.DataFrame:
        rows = [
            {
                "model": result.model,
                "confidence": result.confidence,
                "var": result.var,
                "expected_shortfall": result.expected_shortfall,
                "mean_pnl": result.mean_pnl,
                "volatility": result.volatility,
                "observations": result.observations,
            }
            for result in self.risk_results
        ]
        return pd.DataFrame(rows)


class PortfolioVaREngine:
    def __init__(
        self,
        confidence: float = 0.99,
        paths: int = 20_000,
        horizon_days: int = 1,
        seed: int | None = 42,
        backend: str = "auto",
        antithetic: bool = True,
    ) -> None:
        self.confidence = confidence
        self.paths = paths
        self.horizon_days = horizon_days
        self.seed = seed
        self.backend = backend
        self.antithetic = antithetic

    def run(self, portfolio: Portfolio, returns: pd.DataFrame) -> PortfolioVaRReport:
        aligned = portfolio.align_returns(returns)
        heston = heston_portfolio_var(
            portfolio=portfolio,
            returns=aligned,
            confidence=self.confidence,
            paths=self.paths,
            horizon_days=self.horizon_days,
            seed=self.seed,
            backend=self.backend,
            antithetic=self.antithetic,
        )
        results = (
            historical_var(portfolio, aligned, self.confidence, horizon_days=self.horizon_days),
            gaussian_var(portfolio, aligned, self.confidence, horizon_days=self.horizon_days),
            ewma_var(portfolio, aligned, self.confidence, horizon_days=self.horizon_days),
            heston.risk,
        )
        return PortfolioVaRReport(risk_results=results, heston_diagnostics=heston.diagnostics)
