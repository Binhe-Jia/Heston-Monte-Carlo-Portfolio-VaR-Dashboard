from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class MarketDataConfig:
    start: str = "2020-01-01"
    end: str | None = None
    cache_dir: Path = Path("data_cache")
    use_cache: bool = True


class YahooQueryMarketData:
    """Loads adjusted close data through yahooquery and caches the result."""

    def __init__(self, config: MarketDataConfig | None = None) -> None:
        self.config = config or MarketDataConfig()

    def adjusted_close(self, tickers: Iterable[str]) -> pd.DataFrame:
        symbols = tuple(dict.fromkeys(t.upper() for t in tickers))
        if not symbols:
            raise ValueError("At least one ticker is required.")

        cache_path = self._cache_path(symbols)
        if self.config.use_cache and cache_path.exists():
            cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            if all(symbol in cached.columns for symbol in symbols):
                return cached.loc[:, list(symbols)]

        try:
            from yahooquery import Ticker
        except ImportError as exc:
            raise RuntimeError("yahooquery is required for live market data.") from exc

        history = Ticker(list(symbols), asynchronous=True).history(
            start=self.config.start,
            end=self.config.end,
            adj_ohlc=True,
        )
        prices = _history_to_adjusted_close(history, symbols)
        prices = _clean_price_frame(prices, symbols)

        if prices.empty:
            raise RuntimeError("No usable price data was returned by yahooquery.")

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        prices.to_csv(cache_path)
        return prices

    def _cache_path(self, symbols: tuple[str, ...]) -> Path:
        end = self.config.end or "latest"
        name = "_".join(symbols)
        return self.config.cache_dir / f"{name}_{self.config.start}_{end}.csv"


def _history_to_adjusted_close(history: pd.DataFrame, symbols: tuple[str, ...]) -> pd.DataFrame:
    if isinstance(history, dict):
        errors = {k: v for k, v in history.items() if isinstance(v, str)}
        raise RuntimeError(f"Yahoo query failed: {errors}")

    frame = history.copy()
    if frame.empty:
        return pd.DataFrame()

    close_col = "adjclose" if "adjclose" in frame.columns else "close"

    if isinstance(frame.index, pd.MultiIndex):
        names = list(frame.index.names)
        symbol_level = names.index("symbol") if "symbol" in names else 0
        date_level = names.index("date") if "date" in names else 1
        closes = frame[close_col].unstack(symbol_level)
        closes.index = pd.to_datetime(closes.index)
        closes = closes.rename_axis(None, axis=1)
        if closes.index.name != "date":
            closes.index.name = "date"
        return closes.loc[:, [s for s in symbols if s in closes.columns]]

    if "symbol" in frame.columns:
        closes = frame.pivot_table(values=close_col, index="date", columns="symbol", aggfunc="last")
        closes.index = pd.to_datetime(closes.index)
        return closes.loc[:, [s for s in symbols if s in closes.columns]]

    if len(symbols) == 1 and close_col in frame:
        closes = frame[[close_col]].rename(columns={close_col: symbols[0]})
        closes.index = pd.to_datetime(closes.index)
        return closes

    raise RuntimeError("Could not parse yahooquery history response.")


def _clean_price_frame(prices: pd.DataFrame, symbols: tuple[str, ...]) -> pd.DataFrame:
    if prices.empty:
        return prices

    available = prices.loc[:, [symbol for symbol in symbols if symbol in prices.columns]]
    available = available.sort_index().dropna(how="all")
    available = available.ffill()

    # Mixed ETF/crypto portfolios often start on a weekend or holiday. Drop
    # leading rows until every retained ticker has an observed or forward-filled
    # price instead of dropping the entire ETF columns.
    available = available.dropna(axis=0, how="any")
    return available


def returns_from_prices(prices: pd.DataFrame) -> pd.DataFrame:
    returns = prices.sort_index().pct_change(fill_method=None).dropna(how="all")
    return returns.replace([float("inf"), float("-inf")], pd.NA).dropna(axis=1, how="any")
