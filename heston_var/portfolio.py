from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


TICKER_COLUMN_ALIASES = ("ticker", "symbol", "asset", "security", "ric", "bbg")
WEIGHT_COLUMN_ALIASES = ("weight", "weights", "target_weight", "portfolio_weight", "allocation", "alloc")
VALUE_COLUMN_ALIASES = ("market_value", "value", "notional", "position_value", "mv", "amount")
QUANTITY_COLUMN_ALIASES = ("quantity", "qty", "shares", "units", "position")
PRICE_COLUMN_ALIASES = ("price", "last_price", "close", "market_price")


@dataclass(frozen=True)
class Portfolio:
    tickers: tuple[str, ...]
    weights: np.ndarray
    initial_capital: float = 1_000_000.0

    def __post_init__(self) -> None:
        weights = np.asarray(self.weights, dtype=float)
        if len(self.tickers) != len(weights):
            raise ValueError("Tickers and weights must have the same length.")
        if not np.isfinite(weights).all():
            raise ValueError("Weights must be finite.")
        total = weights.sum()
        if abs(total) < 1e-12:
            raise ValueError("Weights must not sum to zero.")
        object.__setattr__(self, "weights", weights / total)
        object.__setattr__(self, "tickers", tuple(t.upper() for t in self.tickers))

    @classmethod
    def equal_weight(cls, tickers: Sequence[str], initial_capital: float = 1_000_000.0) -> "Portfolio":
        if not tickers:
            raise ValueError("At least one ticker is required.")
        weights = np.full(len(tickers), 1.0 / len(tickers))
        return cls(tuple(tickers), weights, initial_capital)

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        initial_capital: float = 1_000_000.0,
        ticker_col: str | None = None,
        weight_col: str | None = None,
        value_col: str | None = None,
        quantity_col: str | None = None,
        price_col: str | None = None,
    ) -> "Portfolio":
        frame = pd.read_csv(path)
        tickers, weights = parse_portfolio_frame(
            frame,
            ticker_col=ticker_col,
            weight_col=weight_col,
            value_col=value_col,
            quantity_col=quantity_col,
            price_col=price_col,
        )
        return cls(tickers, weights, initial_capital)

    def align_returns(self, returns: pd.DataFrame) -> pd.DataFrame:
        missing = [ticker for ticker in self.tickers if ticker not in returns.columns]
        if missing:
            raise ValueError(f"Missing returns for portfolio tickers: {missing}")
        return returns.loc[:, list(self.tickers)].dropna(how="any")

    def historical_pnl(self, returns: pd.DataFrame) -> pd.Series:
        aligned = self.align_returns(returns)
        portfolio_returns = aligned.to_numpy() @ self.weights
        return pd.Series(portfolio_returns * self.initial_capital, index=aligned.index, name="pnl")


def parse_portfolio_frame(
    frame: pd.DataFrame,
    ticker_col: str | None = None,
    weight_col: str | None = None,
    value_col: str | None = None,
    quantity_col: str | None = None,
    price_col: str | None = None,
) -> tuple[tuple[str, ...], np.ndarray]:
    if frame.empty:
        raise ValueError("Portfolio file is empty.")

    normalized_columns = {_normalize_column(column): column for column in frame.columns}
    ticker_col = ticker_col or _find_column(normalized_columns, TICKER_COLUMN_ALIASES)
    if ticker_col is None:
        raise ValueError(
            "Could not find a ticker column. Use one of "
            f"{TICKER_COLUMN_ALIASES}, or pass ticker_col explicitly."
        )

    weight_col = weight_col or _find_column(normalized_columns, WEIGHT_COLUMN_ALIASES)
    value_col = value_col or _find_column(normalized_columns, VALUE_COLUMN_ALIASES)
    quantity_col = quantity_col or _find_column(normalized_columns, QUANTITY_COLUMN_ALIASES)
    price_col = price_col or _find_column(normalized_columns, PRICE_COLUMN_ALIASES)

    data = frame.copy()
    tickers = data[ticker_col].astype(str).str.strip()
    valid_tickers = tickers.ne("") & tickers.str.lower().ne("nan")
    data = data.loc[valid_tickers].copy()
    tickers = tickers.loc[valid_tickers].str.upper()
    if data.empty:
        raise ValueError("Portfolio file contains no valid tickers.")

    if weight_col is not None:
        weights = _numeric_column(data, weight_col)
    elif value_col is not None:
        weights = _numeric_column(data, value_col)
    elif quantity_col is not None and price_col is not None:
        weights = _numeric_column(data, quantity_col) * _numeric_column(data, price_col)
    elif quantity_col is not None:
        weights = _numeric_column(data, quantity_col)
    else:
        raise ValueError(
            "Could not infer portfolio exposure. Provide a weight column, market value column, "
            "or quantity/shares column. Supported aliases include "
            f"weights={WEIGHT_COLUMN_ALIASES}, values={VALUE_COLUMN_ALIASES}, "
            f"quantities={QUANTITY_COLUMN_ALIASES}, prices={PRICE_COLUMN_ALIASES}."
        )

    parsed = pd.DataFrame({"ticker": tickers.to_numpy(), "weight": weights.to_numpy()}).dropna()
    parsed = parsed[np.isfinite(parsed["weight"].to_numpy(dtype=float))]
    parsed = parsed[parsed["weight"].astype(float).abs() > 0.0]
    if parsed.empty:
        raise ValueError("Portfolio file contains no non-zero numeric exposures.")

    grouped = parsed.groupby("ticker", sort=False)["weight"].sum()
    if abs(float(grouped.sum())) < 1e-12:
        raise ValueError("Portfolio exposures sum to zero after parsing.")
    return tuple(grouped.index.astype(str)), grouped.to_numpy(dtype=float)


def _normalize_column(column: object) -> str:
    return str(column).strip().lower().replace(" ", "_").replace("-", "_")


def _find_column(normalized_columns: dict[str, str], aliases: Sequence[str]) -> str | None:
    for alias in aliases:
        found = normalized_columns.get(_normalize_column(alias))
        if found is not None:
            return found
    return None


def _numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")
