from __future__ import annotations

import base64
import io
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html

from heston_var.data import MarketDataConfig, YahooQueryMarketData, returns_from_prices
from heston_var.demo import synthetic_returns
from heston_var.models import ewma_var, gaussian_var, historical_var, portfolio_horizon_pnl
from heston_var.portfolio import Portfolio, parse_portfolio_frame
from heston_var.simulation import heston_portfolio_var


def _project_root() -> Path:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parents[1]


PROJECT_ROOT = _project_root()
DEFAULT_PORTFOLIO = PROJECT_ROOT / "configs" / "large_multi_market_portfolio.csv"
RISK_HORIZONS = ((1, "1-Day"), (21, "1-Month"), (252, "1-Year"))


@dataclass(frozen=True)
class DashboardData:
    portfolio: Portfolio
    returns: pd.DataFrame
    holdings: pd.DataFrame
    source_label: str


def create_app() -> Dash:
    app = Dash(__name__, title="Portfolio VaR Impact Dashboard")
    app.index_string = _index_string()
    app.layout = html.Div(
        className="page",
        children=[
            html.Div(
                className="topbar",
                children=[
                    html.Div(
                        children=[
                            html.H1("Portfolio Risk Impact Dashboard"),
                            html.P("A plain-English view of downside loss, tail risk, and model confidence."),
                        ]
                    ),
                    html.Div(
                        className="status-pill",
                        children="Heston Monte Carlo VaR / ES",
                    ),
                ],
            ),
            html.Div(
                className="control-band",
                children=[
                    html.Div(
                        className="control-group",
                        children=[
                            html.Label("Portfolio Source"),
                            dcc.RadioItems(
                                id="data-source",
                                options=[
                                    {"label": "Offline Demo", "value": "demo"},
                                    {"label": "Large Stress CSV", "value": "stress"},
                                    {"label": "Custom CSV Path", "value": "custom"},
                                    {"label": "Upload CSV", "value": "upload"},
                                ],
                                value="demo",
                                inline=True,
                                className="radio-row",
                            ),
                        ],
                    ),
                    html.Div(
                        className="control-grid",
                        children=[
                            html.Div(
                                children=[
                                    html.Label("Custom CSV Path"),
                                    dcc.Input(
                                        id="portfolio-path",
                                        value="",
                                        type="text",
                                        debounce=True,
                                        placeholder="Choose Custom CSV Path, then paste a local CSV path here",
                                    ),
                                ]
                            ),
                            html.Div(
                                children=[
                                    html.Label("Start Date"),
                                    dcc.Input(id="start-date", value="2021-01-01", type="text", debounce=True),
                                ]
                            ),
                            html.Div(
                                children=[
                                    html.Label("Capital"),
                                    dcc.Input(id="capital", value=1_000_000, type="number", min=1),
                                ]
                            ),
                            html.Div(
                                children=[
                                    html.Label("MC Paths"),
                                    dcc.Input(id="paths", value=50_000, type="number", min=1_000, step=1_000),
                                ]
                            ),
                            html.Div(
                                children=[
                                    html.Label("Confidence"),
                                    dcc.Dropdown(
                                        id="confidence",
                                        options=[
                                            {"label": "95%", "value": 0.95},
                                            {"label": "99%", "value": 0.99},
                                        ],
                                        value=0.99,
                                        clearable=False,
                                    ),
                                ]
                            ),
                            html.Div(
                                children=[
                                    html.Label("Risk Horizon"),
                                    dcc.Dropdown(
                                        id="horizon-days",
                                        options=[
                                            {"label": "1 Day", "value": 1},
                                            {"label": "1 Month", "value": 21},
                                            {"label": "1 Year", "value": 252},
                                        ],
                                        value=1,
                                        clearable=False,
                                    ),
                                ]
                            ),
                            html.Div(
                                children=[
                                    html.Label("Backend"),
                                    dcc.Dropdown(
                                        id="backend",
                                        options=[
                                            {"label": "Auto", "value": "auto"},
                                            {"label": "Rust", "value": "rust"},
                                            {"label": "Python", "value": "python"},
                                        ],
                                        value="auto",
                                        clearable=False,
                                    ),
                                ]
                            ),
                        ],
                    ),
                    dcc.Upload(
                        id="upload-csv",
                        className="upload-box",
                        children=html.Div(["Drop a portfolio CSV here or click to select"]),
                        multiple=False,
                    ),
                    html.Div(id="upload-status", className="upload-status"),
                    html.Button("Run Risk Analysis", id="run-button", n_clicks=0, className="run-button"),
                    html.Div(id="run-message", className="run-message"),
                ],
            ),
            dcc.Loading(
                type="circle",
                children=html.Div(
                    className="dashboard-body",
                    children=[
                        dcc.Tabs(
                            className="tabs",
                            children=[
                                dcc.Tab(
                                    label="Executive Summary",
                                    children=[
                                        html.Div(id="impact-summary", className="summary-grid"),
                                        html.Div(
                                            className="chart-grid",
                                            children=[
                                                _panel("Model Comparison", dcc.Graph(id="model-chart", config={"displayModeBar": False})),
                                                _panel("Portfolio Exposure", dcc.Graph(id="exposure-chart", config={"displayModeBar": False})),
                                                _panel(
                                                    "Tail Loss Distribution",
                                                    html.Div(
                                                        children=[
                                                            dcc.Graph(
                                                                id="pnl-chart",
                                                                className="tail-chart",
                                                                config={"displayModeBar": False},
                                                            ),
                                                            html.Div(
                                                                id="var-footnote",
                                                                className="chart-footnote",
                                                                children="Note: the red dashed line marks Heston VaR.",
                                                            ),
                                                        ]
                                                    ),
                                                ),
                                                _panel("Business Interpretation", html.Div(id="plain-language", className="plain-language")),
                                            ],
                                        ),
                                    ],
                                ),
                                dcc.Tab(
                                    label="Model Details",
                                    children=[
                                        html.Div(
                                            className="table-panel",
                                            children=[
                                                html.H2("Model Results"),
                                                html.Div(id="result-table"),
                                            ],
                                        ),
                                        html.Div(
                                            className="table-panel",
                                            children=[
                                                html.H2("VaR / ES by Horizon"),
                                                dcc.Graph(id="horizon-chart", config={"displayModeBar": False}),
                                                html.Div(id="horizon-table"),
                                            ],
                                        ),
                                    ],
                                ),
                                dcc.Tab(
                                    label="Model Diagnostics",
                                    children=[
                                        html.Div(id="diagnostic-cards", className="summary-grid"),
                                        html.Div(
                                            className="table-panel",
                                            children=[
                                                html.H2("Distribution and Model Fit Diagnostics"),
                                                html.Div(id="diagnostic-table"),
                                                html.Div(id="diagnostic-note", className="diagnostic-note"),
                                            ],
                                        ),
                                    ],
                                ),
                                dcc.Tab(
                                    label="Portfolio Data",
                                    children=[
                                        html.Div(
                                            className="table-panel",
                                            children=[
                                                html.H2("Portfolio Inputs"),
                                                html.Div(id="data-quality", className="data-quality"),
                                                html.Div(id="holdings-table"),
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
            ),
        ],
    )

    @app.callback(
        Output("data-source", "value"),
        Output("upload-status", "children"),
        Input("upload-csv", "contents"),
        State("upload-csv", "filename"),
        State("data-source", "value"),
        prevent_initial_call=True,
    )
    def select_upload_source(upload_contents: str | None, upload_filename: str | None, current_source: str):
        if upload_contents:
            return "upload", f"Uploaded file ready: {upload_filename or 'portfolio CSV'}"
        return current_source, ""

    @app.callback(
        Output("impact-summary", "children"),
        Output("model-chart", "figure"),
        Output("exposure-chart", "figure"),
        Output("pnl-chart", "figure"),
        Output("var-footnote", "children"),
        Output("plain-language", "children"),
        Output("result-table", "children"),
        Output("horizon-chart", "figure"),
        Output("horizon-table", "children"),
        Output("diagnostic-cards", "children"),
        Output("diagnostic-table", "children"),
        Output("diagnostic-note", "children"),
        Output("holdings-table", "children"),
        Output("data-quality", "children"),
        Output("run-message", "children"),
        Input("run-button", "n_clicks"),
        State("data-source", "value"),
        State("portfolio-path", "value"),
        State("upload-csv", "contents"),
        State("upload-csv", "filename"),
        State("start-date", "value"),
        State("capital", "value"),
        State("paths", "value"),
        State("confidence", "value"),
        State("horizon-days", "value"),
        State("backend", "value"),
    )
    def run_dashboard(
        _n_clicks: int,
        data_source: str,
        portfolio_path: str,
        upload_contents: str | None,
        upload_filename: str | None,
        start_date: str,
        capital: float,
        paths: int,
        confidence: float,
        horizon_days: int,
        backend: str,
    ):
        try:
            horizon_days = int(horizon_days or 1)
            horizon_label = _horizon_label(horizon_days)
            data = _load_dashboard_data(
                data_source=data_source,
                portfolio_path=portfolio_path,
                upload_contents=upload_contents,
                upload_filename=upload_filename,
                start_date=start_date,
                capital=float(capital or 1_000_000),
            )
            heston_simulation = heston_portfolio_var(
                data.portfolio,
                data.returns,
                confidence=float(confidence),
                paths=int(paths or 50_000),
                horizon_days=horizon_days,
                seed=42,
                backend=backend,
                antithetic=True,
            )
            aligned_returns = data.portfolio.align_returns(data.returns)
            risk_results = (
                historical_var(data.portfolio, aligned_returns, float(confidence), horizon_days=horizon_days),
                gaussian_var(data.portfolio, aligned_returns, float(confidence), horizon_days=horizon_days),
                ewma_var(data.portfolio, aligned_returns, float(confidence), horizon_days=horizon_days),
                heston_simulation.risk,
            )

            table = _risk_results_to_frame(risk_results)
            heston_row = table.loc[table["model"].eq("Heston MC")].iloc[0]
            historical_row = table.loc[table["model"].eq("Historical")].iloc[0]
            historical_pnl = portfolio_horizon_pnl(data.portfolio, data.returns, horizon_days).to_numpy()
            horizon_table = _horizon_comparison_frame(
                portfolio=data.portfolio,
                returns=aligned_returns,
                confidence=float(confidence),
                paths=int(paths or 50_000),
                backend=backend,
                selected_horizon_days=horizon_days,
                selected_heston_result=heston_simulation.risk,
            )
            diagnostics = _diagnostics_frame(table, historical_pnl, heston_simulation.pnl, horizon_label)

            summary = _summary_cards(
                heston_var=float(heston_row["var"]),
                heston_es=float(heston_row["expected_shortfall"]),
                historical_var=float(historical_row["var"]),
                capital=float(capital or 1_000_000),
                backend=heston_simulation.diagnostics.backend,
                assets=len(data.portfolio.tickers),
                horizon_label=horizon_label,
            )
            message = (
                f"Loaded {len(data.portfolio.tickers)} assets from {data.source_label}. "
                f"Used {heston_simulation.diagnostics.backend} with {heston_simulation.diagnostics.paths:,} paths "
                f"for a {horizon_label.lower()} horizon."
            )
            return (
                summary,
                _model_figure(table),
                _exposure_figure(data.holdings),
                _pnl_figure(heston_simulation.pnl, historical_pnl, float(heston_row["var"]), horizon_label),
                f"Note: the red dashed line marks Heston VaR, the estimated {horizon_label.lower()} loss threshold at the selected confidence level.",
                _plain_language(table, float(capital or 1_000_000), horizon_label),
                _result_table(table),
                _horizon_figure(horizon_table),
                _horizon_table(horizon_table),
                _diagnostic_cards(diagnostics, horizon_label),
                _diagnostic_table(diagnostics),
                _diagnostic_note(horizon_label),
                _holdings_table(data.holdings),
                _data_quality_summary(data),
                message,
            )
        except Exception as exc:
            empty = _empty_figure("Run the analysis to populate this view.")
            return (
                [],
                empty,
                empty,
                empty,
                "",
                html.Div(className="error-box", children=str(exc)),
                html.Div(),
                empty,
                html.Div(),
                [],
                html.Div(),
                "",
                html.Div(),
                html.Div(),
                f"Could not run analysis: {exc}",
            )

    return app


def _load_dashboard_data(
    data_source: str,
    portfolio_path: str,
    upload_contents: str | None,
    upload_filename: str | None,
    start_date: str,
    capital: float,
) -> DashboardData:
    if data_source == "demo":
        returns = synthetic_returns(assets=4, observations=900, seed=7)
        portfolio = Portfolio.equal_weight(tuple(returns.columns), initial_capital=capital)
        holdings = pd.DataFrame(
            {"ticker": portfolio.tickers, "weight": portfolio.weights, "asset_class": "Synthetic", "market": "Demo"}
        )
        return DashboardData(portfolio, returns, holdings, "offline demo")

    if data_source == "upload":
        if not upload_contents:
            raise ValueError("Choose a CSV file before running uploaded portfolio analysis.")
        try:
            holdings = _uploaded_csv_to_frame(upload_contents)
            tickers, weights = parse_portfolio_frame(holdings)
        except Exception as exc:
            raise ValueError(
                "Could not read the uploaded portfolio CSV. It must include a ticker/symbol "
                "column and a weight, allocation, market_value, shares, or quantity column. "
                f"Details: {exc}"
            ) from exc
        portfolio = Portfolio(tickers, weights, initial_capital=capital)
        source_label = upload_filename or "uploaded CSV"
    else:
        if data_source == "stress":
            csv_path = DEFAULT_PORTFOLIO
        else:
            portfolio_path = _clean_path_input(portfolio_path)
            if not portfolio_path:
                raise ValueError("Paste a custom CSV path, or choose Large Stress CSV / Upload CSV.")
            csv_path = Path(portfolio_path)
            if not csv_path.exists():
                raise ValueError(f"Custom CSV path does not exist: {csv_path}")
        try:
            holdings = pd.read_csv(csv_path)
            portfolio = Portfolio.from_csv(csv_path, initial_capital=capital)
        except Exception as exc:
            raise ValueError(
                "Could not read the portfolio CSV. It must include a ticker/symbol column "
                "and a weight, allocation, market_value, shares, or quantity column. "
                f"Details: {exc}"
            ) from exc
        source_label = csv_path.name

    try:
        loader = YahooQueryMarketData(MarketDataConfig(start=start_date))
        prices = loader.adjusted_close(portfolio.tickers)
        returns = returns_from_prices(prices)
    except Exception as exc:
        raise ValueError(
            "The portfolio CSV loaded, but market data could not be prepared. Check that the "
            "tickers are Yahoo-compatible and that the selected start date has available data. "
            f"Details: {exc}"
        ) from exc
    holdings = _normalize_holdings_for_display(holdings, portfolio)
    return DashboardData(portfolio, returns, holdings, source_label)


def _uploaded_csv_to_frame(contents: str) -> pd.DataFrame:
    _content_type, content_string = contents.split(",", 1)
    decoded = base64.b64decode(content_string)
    return pd.read_csv(io.StringIO(decoded.decode("utf-8-sig")))


def _clean_path_input(path_value: str | None) -> str:
    if path_value is None:
        return ""
    return str(path_value).strip().strip('"').strip("'")


def _normalize_holdings_for_display(holdings: pd.DataFrame, portfolio: Portfolio) -> pd.DataFrame:
    display = pd.DataFrame({"ticker": portfolio.tickers, "weight": portfolio.weights})
    for column in ("asset_class", "market", "sleeve"):
        if column in holdings.columns:
            meta = holdings.copy()
            ticker_col = "ticker" if "ticker" in meta.columns else "symbol" if "symbol" in meta.columns else None
            if ticker_col:
                meta[ticker_col] = meta[ticker_col].astype(str).str.upper()
                display = display.merge(
                    meta[[ticker_col, column]].drop_duplicates(),
                    left_on="ticker",
                    right_on=ticker_col,
                    how="left",
                )
                if ticker_col != "ticker":
                    display = display.drop(columns=[ticker_col])
    display["asset_class"] = display.get("asset_class", pd.Series(index=display.index, dtype=object)).fillna("Unclassified")
    display["market"] = display.get("market", pd.Series(index=display.index, dtype=object)).fillna("Unclassified")
    display["sleeve"] = display.get("sleeve", pd.Series(index=display.index, dtype=object)).fillna("Portfolio")
    return display


def _summary_cards(
    heston_var: float,
    heston_es: float,
    historical_var: float,
    capital: float,
    backend: str,
    assets: int,
    horizon_label: str,
) -> list[html.Div]:
    difference = heston_var - historical_var
    return [
        _metric_card(f"{horizon_label} VaR", _money(heston_var), f"{heston_var / capital:.2%} of portfolio capital"),
        _metric_card("Expected Shortfall", _money(heston_es), "Average loss in the worst simulated outcomes"),
        _metric_card("Historical Gap", _money(difference), "Heston minus Historical VaR"),
        _metric_card("Assets Covered", f"{assets}", f"Simulation backend: {backend}"),
    ]


def _risk_results_to_frame(risk_results) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": result.model,
                "confidence": result.confidence,
                "var": result.var,
                "expected_shortfall": result.expected_shortfall,
                "mean_pnl": result.mean_pnl,
                "volatility": result.volatility,
                "observations": result.observations,
            }
            for result in risk_results
        ]
    )


def _horizon_comparison_frame(
    portfolio: Portfolio,
    returns: pd.DataFrame,
    confidence: float,
    paths: int,
    backend: str,
    selected_horizon_days: int,
    selected_heston_result,
) -> pd.DataFrame:
    rows = []
    for horizon_days, horizon_label in RISK_HORIZONS:
        try:
            heston_result = (
                selected_heston_result
                if horizon_days == selected_horizon_days
                else heston_portfolio_var(
                    portfolio,
                    returns,
                    confidence=confidence,
                    paths=paths,
                    horizon_days=horizon_days,
                    seed=42 + horizon_days,
                    backend=backend,
                    antithetic=True,
                ).risk
            )
            results = (
                historical_var(portfolio, returns, confidence, horizon_days=horizon_days),
                gaussian_var(portfolio, returns, confidence, horizon_days=horizon_days),
                ewma_var(portfolio, returns, confidence, horizon_days=horizon_days),
                heston_result,
            )
            for result in results:
                rows.append(
                    {
                        "horizon": horizon_label,
                        "horizon_days": horizon_days,
                        "model": result.model,
                        "var": result.var,
                        "expected_shortfall": result.expected_shortfall,
                        "observations": result.observations,
                        "note": _horizon_model_note(result.model, horizon_days),
                    }
                )
        except Exception as exc:
            rows.append(
                {
                    "horizon": horizon_label,
                    "horizon_days": horizon_days,
                    "model": "Unavailable",
                    "var": np.nan,
                    "expected_shortfall": np.nan,
                    "observations": 0,
                    "note": str(exc),
                }
            )
    return pd.DataFrame(rows)


def _diagnostics_frame(
    table: pd.DataFrame,
    historical_pnl: np.ndarray,
    heston_pnl: np.ndarray,
    horizon_label: str,
) -> pd.DataFrame:
    historical_var_value = float(table.loc[table["model"].eq("Historical"), "var"].iloc[0])
    stats_by_model = {
        "Historical": _distribution_stats(historical_pnl),
        "Heston MC": _distribution_stats(heston_pnl),
    }
    rows = []
    for _, row in table.iterrows():
        model = row["model"]
        stats = stats_by_model.get(model, {})
        gap = float(row["var"]) - historical_var_value
        gap_pct = gap / max(historical_var_value, 1.0)
        rows.append(
            {
                "model": model,
                "horizon": horizon_label,
                "var": float(row["var"]),
                "expected_shortfall": float(row["expected_shortfall"]),
                "gap_vs_historical": gap,
                "gap_pct": gap_pct,
                "observations": int(row["observations"]),
                "skewness": stats.get("skewness", np.nan),
                "excess_kurtosis": stats.get("excess_kurtosis", np.nan),
                "diagnostic": _diagnostic_text(model, stats, gap_pct, horizon_label),
            }
        )
    return pd.DataFrame(rows)


def _horizon_model_note(model: str, horizon_days: int) -> str:
    if model == "Historical":
        if horizon_days == 1:
            return "Uses realized daily portfolio P&L."
        return "Uses rolling compounded returns; windows overlap and can reflect market regimes."
    if model == "Gaussian":
        if horizon_days == 1:
            return "Bell-curve benchmark using daily mean and volatility."
        return "Scales daily risk by time; assumes bell-shaped returns."
    if model == "EWMA":
        return "Weights recent volatility more heavily than older observations."
    if model == "Heston MC":
        return "Simulates stochastic volatility paths for this selected horizon."
    return ""


def _distribution_stats(values: np.ndarray) -> dict[str, float]:
    series = pd.Series(np.asarray(values, dtype=float)).replace([np.inf, -np.inf], np.nan).dropna()
    if len(series) < 3:
        return {"observations": float(len(series)), "skewness": np.nan, "excess_kurtosis": np.nan}
    return {
        "observations": float(len(series)),
        "skewness": float(series.skew()),
        "excess_kurtosis": float(series.kurt()),
    }


def _diagnostic_text(model: str, stats: dict[str, float], gap_pct: float, horizon_label: str) -> str:
    if model == "Gaussian":
        return "Bell-curve benchmark; treat with caution when historical skew or fat tails are visible."
    if model == "EWMA":
        return "Recent-volatility benchmark; often rises after volatility spikes and can be conservative."
    if model == "Historical":
        skewness = stats.get("skewness", np.nan)
        kurtosis = stats.get("excess_kurtosis", np.nan)
        if abs(skewness) > 0.75 or kurtosis > 1.5:
            return f"{horizon_label} historical outcomes look non-normal; overlapping windows and market regimes matter."
        return "Historical outcomes are reasonably close to a bell-shaped reference for this horizon."
    if abs(gap_pct) < 0.05:
        return "Close to historical VaR for the selected horizon."
    if gap_pct < 0:
        return "Lower than historical VaR; useful, but review whether Heston calibration is too optimistic."
    return "Above historical VaR; Heston is more conservative than historical simulation here."


def _metric_card(title: str, value: str, subtitle: str) -> html.Div:
    return html.Div(className="metric-card", children=[html.Div(className="metric-title", children=title), html.Div(className="metric-value", children=value), html.Div(className="metric-subtitle", children=subtitle)])


def _model_figure(table: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        table,
        x="model",
        y=["var", "expected_shortfall"],
        barmode="group",
        labels={"value": "Dollars", "model": "", "variable": "Measure"},
        color_discrete_map={"var": "#2563eb", "expected_shortfall": "#f97316"},
    )
    fig.update_layout(template="plotly_white", margin=dict(l=30, r=20, t=25, b=30), legend_title_text="")
    return fig


def _horizon_figure(table: pd.DataFrame) -> go.Figure:
    chart = table.loc[table["model"].ne("Unavailable") & table["var"].notna()].copy()
    fig = px.bar(
        chart,
        x="horizon",
        y="var",
        color="model",
        barmode="group",
        labels={"var": "VaR dollars", "horizon": "", "model": ""},
        color_discrete_sequence=["#2563eb", "#16a34a", "#f97316", "#7c3aed"],
    )
    fig.update_layout(template="plotly_white", margin=dict(l=30, r=20, t=25, b=35), legend_title_text="")
    return fig


def _horizon_table(table: pd.DataFrame) -> html.Table:
    display = table.copy()
    for column in ("var", "expected_shortfall"):
        display[column] = display[column].map(lambda value: "n/a" if pd.isna(value) else _money(float(value)))
    display["observations"] = display["observations"].map(lambda value: f"{int(value):,}" if pd.notna(value) else "n/a")
    columns = ["horizon", "model", "var", "expected_shortfall", "observations", "note"]
    return html.Table(
        className="result-table",
        children=[
            html.Thead(html.Tr([html.Th(column.replace("_", " ").title()) for column in columns])),
            html.Tbody([html.Tr([html.Td(row[column]) for column in columns]) for _, row in display.iterrows()]),
        ],
    )


def _diagnostic_cards(diagnostics: pd.DataFrame, horizon_label: str) -> list[html.Div]:
    historical = diagnostics.loc[diagnostics["model"].eq("Historical")].iloc[0]
    heston = diagnostics.loc[diagnostics["model"].eq("Heston MC")].iloc[0]
    skewness = float(historical["skewness"]) if pd.notna(historical["skewness"]) else 0.0
    kurtosis = float(historical["excess_kurtosis"]) if pd.notna(historical["excess_kurtosis"]) else 0.0
    shape = "Non-normal" if abs(skewness) > 0.75 or kurtosis > 1.5 else "Stable"
    gap = float(heston["gap_vs_historical"])
    return [
        _metric_card("Historical Shape", shape, f"Skew {skewness:.2f}, excess kurtosis {kurtosis:.2f}"),
        _metric_card("Heston Gap", _money(gap), "Heston VaR minus Historical VaR"),
        _metric_card("Heston Gap %", f"{float(heston['gap_pct']):.1%}", "Negative means lower than historical"),
        _metric_card("Diagnostic Horizon", horizon_label, "Main distribution checks use the selected horizon"),
    ]


def _diagnostic_table(diagnostics: pd.DataFrame) -> html.Table:
    display = diagnostics.copy()
    for column in ("var", "expected_shortfall", "gap_vs_historical"):
        display[column] = display[column].map(_money)
    display["gap_pct"] = display["gap_pct"].map(lambda value: f"{value:.1%}")
    for column in ("skewness", "excess_kurtosis"):
        display[column] = display[column].map(lambda value: "n/a" if pd.isna(value) else f"{value:.2f}")
    columns = [
        "model",
        "horizon",
        "var",
        "expected_shortfall",
        "gap_vs_historical",
        "gap_pct",
        "skewness",
        "excess_kurtosis",
        "diagnostic",
    ]
    return html.Table(
        className="result-table",
        children=[
            html.Thead(html.Tr([html.Th(column.replace("_", " ").title()) for column in columns])),
            html.Tbody([html.Tr([html.Td(row[column]) for column in columns]) for _, row in display.iterrows()]),
        ],
    )


def _diagnostic_note(horizon_label: str) -> str:
    return (
        f"Note: {horizon_label} historical P&L uses rolling compounded returns. "
        "For one-month and one-year horizons these windows overlap, so the shape can be skewed, clustered, "
        "or regime-driven rather than normally distributed."
    )


def _exposure_figure(holdings: pd.DataFrame) -> go.Figure:
    grouped = holdings.groupby("asset_class", dropna=False)["weight"].sum().reset_index()
    fig = px.pie(grouped, names="asset_class", values="weight", hole=0.48)
    fig.update_traces(textposition="inside", texttemplate="%{label}<br>%{percent}")
    fig.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=20, b=20), showlegend=False)
    return fig


def _pnl_figure(
    simulated_pnl: np.ndarray,
    historical_pnl: np.ndarray,
    var_value: float,
    horizon_label: str,
) -> go.Figure:
    fig = go.Figure()
    historical_x, historical_y, bin_width = _histogram_percentages(historical_pnl, simulated_pnl)
    simulated_x, simulated_y, _ = _histogram_percentages(simulated_pnl, historical_pnl)
    fig.add_trace(
        go.Bar(
            x=historical_x,
            y=historical_y,
            width=bin_width,
            name="Historical P&L",
            opacity=0.55,
            hovertemplate="P&L bin: %{x:$,.0f}<br>Share: %{y:.1%}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=simulated_x,
            y=simulated_y,
            width=bin_width,
            name="Heston MC P&L",
            opacity=0.55,
            hovertemplate="P&L bin: %{x:$,.0f}<br>Share: %{y:.1%}<extra></extra>",
        )
    )
    fig.add_vline(x=-var_value, line_width=2, line_dash="dash", line_color="#dc2626")
    fig.update_layout(
        template="plotly_white",
        barmode="overlay",
        margin=dict(l=30, r=20, t=25, b=30),
        xaxis_title=f"{horizon_label} portfolio profit / loss",
        yaxis_title="Share of outcomes",
        yaxis_tickformat=".0%",
        legend_title_text="",
    )
    return fig


def _histogram_percentages(values: np.ndarray, reference_values: np.ndarray, bins: int = 70) -> tuple[np.ndarray, np.ndarray, float]:
    sample = pd.Series(np.asarray(values, dtype=float)).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    reference = pd.Series(np.asarray(reference_values, dtype=float)).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    combined = np.concatenate([sample, reference])
    if sample.size == 0 or combined.size == 0:
        return np.array([]), np.array([]), 0.0
    low = float(np.min(combined))
    high = float(np.max(combined))
    if not np.isfinite(low) or not np.isfinite(high) or low == high:
        low, high = low - 0.5, high + 0.5
    edges = np.linspace(low, high, bins + 1)
    counts, _ = np.histogram(sample, bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2.0
    shares = counts / max(sample.size, 1)
    return centers, shares, float(edges[1] - edges[0])


def _plain_language(table: pd.DataFrame, capital: float, horizon_label: str) -> html.Div:
    heston = table.loc[table["model"].eq("Heston MC")].iloc[0]
    historical = table.loc[table["model"].eq("Historical")].iloc[0]
    heston_var = float(heston["var"])
    heston_es = float(heston["expected_shortfall"])
    gap = heston_var - float(historical["var"])
    stance = "close to" if abs(gap) / max(float(historical["var"]), 1.0) < 0.05 else "above" if gap > 0 else "below"
    return html.Div(
        children=[
            html.P(f"At this confidence level, the Heston model estimates a {horizon_label.lower()} downside loss of about {_money(heston_var)}."),
            html.P(f"If losses move into the tail, the average severe-loss outcome is about {_money(heston_es)}."),
            html.P(f"Relative to historical simulation, Heston is {stance} the historical benchmark. The gap is {_money(gap)}, or {gap / capital:.2%} of capital."),
            html.P("For a non-technical audience, read VaR as the loss threshold and Expected Shortfall as the more severe tail-loss estimate."),
        ]
    )


def _horizon_label(horizon_days: int) -> str:
    if horizon_days == 1:
        return "1-Day"
    if horizon_days == 21:
        return "1-Month"
    if horizon_days == 252:
        return "1-Year"
    return f"{horizon_days}-Day"


def _result_table(table: pd.DataFrame) -> html.Table:
    display = table.copy()
    for column in ("var", "expected_shortfall", "mean_pnl", "volatility"):
        display[column] = display[column].map(_money)
    display["confidence"] = display["confidence"].map(lambda x: f"{x:.0%}")
    columns = ["model", "confidence", "var", "expected_shortfall", "volatility", "observations"]
    return html.Table(
        className="result-table",
        children=[
            html.Thead(html.Tr([html.Th(column.replace("_", " ").title()) for column in columns])),
            html.Tbody([html.Tr([html.Td(row[column]) for column in columns]) for _, row in display.iterrows()]),
        ],
    )


def _holdings_table(holdings: pd.DataFrame) -> html.Table:
    display = holdings.copy()
    display["weight"] = display["weight"].map(lambda value: f"{value:.2%}")
    columns = [column for column in ("ticker", "weight", "asset_class", "market", "sleeve") if column in display.columns]
    return html.Table(
        className="result-table",
        children=[
            html.Thead(html.Tr([html.Th(column.replace("_", " ").title()) for column in columns])),
            html.Tbody([html.Tr([html.Td(row[column]) for column in columns]) for _, row in display.iterrows()]),
        ],
    )


def _data_quality_summary(data: DashboardData) -> html.Div:
    start = data.returns.index.min().date() if len(data.returns.index) else "n/a"
    end = data.returns.index.max().date() if len(data.returns.index) else "n/a"
    return html.Div(
        className="data-quality-grid",
        children=[
            _metric_card("Data Source", data.source_label, "Portfolio input used for this run"),
            _metric_card("Return Rows", f"{len(data.returns):,}", f"Date range: {start} to {end}"),
            _metric_card("Ticker Count", f"{len(data.portfolio.tickers)}", "Assets included after parsing"),
            _metric_card("Weight Check", f"{data.holdings['weight'].sum():.2%}", "Normalized portfolio weight"),
        ],
    )


def _panel(title: str, child) -> html.Div:
    return html.Div(className="panel", children=[html.H2(title), child])


def _empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper")
    fig.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=20, b=20))
    return fig


def _money(value: float) -> str:
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value):,.0f}"


def _index_string() -> str:
    return (
        """<!DOCTYPE html>
        <html>
          <head>
            {%metas%}
            <title>{%title%}</title>
            {%favicon%}
            {%css%}
            <style>
        body { margin: 0; background: #f6f7f9; color: #172033; font-family: Arial, Helvetica, sans-serif; }
        .page { min-height: 100vh; }
        .topbar { display: flex; justify-content: space-between; align-items: center; padding: 22px 28px; background: #ffffff; border-bottom: 1px solid #dfe3ea; }
        h1 { margin: 0; font-size: 28px; font-weight: 700; letter-spacing: 0; }
        h2 { margin: 0 0 12px 0; font-size: 17px; font-weight: 700; letter-spacing: 0; }
        p { line-height: 1.45; }
        .topbar p { margin: 6px 0 0; color: #5b6678; }
        .status-pill { padding: 8px 12px; border: 1px solid #c9d3e4; background: #eef4ff; color: #24466f; border-radius: 6px; font-weight: 700; }
        .control-band { padding: 18px 28px; background: #ffffff; border-bottom: 1px solid #dfe3ea; }
        .control-group { margin-bottom: 14px; }
        label { display: block; font-size: 12px; color: #516070; font-weight: 700; margin-bottom: 6px; }
        .radio-row label { margin-right: 18px; font-size: 14px; color: #1f2937; font-weight: 500; }
        .control-grid { display: grid; grid-template-columns: minmax(260px, 2fr) repeat(6, minmax(115px, 1fr)); gap: 12px; align-items: end; }
        input, .Select-control { width: 100%; min-height: 38px; border: 1px solid #cfd6e3; border-radius: 6px; box-sizing: border-box; }
        .upload-box { margin-top: 12px; padding: 12px; border: 1px dashed #9aa7bb; border-radius: 6px; color: #475569; background: #fbfcfe; text-align: center; }
        .upload-status { margin-top: 8px; color: #166534; font-size: 13px; font-weight: 700; }
        .run-button { margin-top: 14px; min-height: 40px; padding: 0 18px; border: 0; border-radius: 6px; background: #1d4ed8; color: white; font-weight: 700; cursor: pointer; }
        .run-message { display: inline-block; margin-left: 14px; color: #475569; }
        .dashboard-body { padding: 22px 28px 34px; }
        .tabs { margin-bottom: 16px; }
        .summary-grid { display: grid; grid-template-columns: repeat(4, minmax(160px, 1fr)); gap: 14px; margin-bottom: 18px; }
        .metric-card, .panel, .table-panel { background: white; border: 1px solid #dfe3ea; border-radius: 8px; }
        .metric-card { padding: 16px; }
        .metric-title { color: #5b6678; font-size: 13px; font-weight: 700; }
        .metric-value { margin-top: 8px; font-size: 28px; font-weight: 800; color: #101827; }
        .metric-subtitle { margin-top: 6px; color: #64748b; font-size: 13px; }
        .chart-grid { display: grid; grid-template-columns: repeat(2, minmax(320px, 1fr)); gap: 16px; }
        .panel { padding: 16px; min-height: 360px; overflow: hidden; }
        .tail-chart { height: 315px; }
        .chart-footnote { color: #64748b; font-size: 13px; line-height: 1.35; padding: 8px 6px 2px; }
        .plain-language { color: #263244; font-size: 15px; }
        .table-panel { margin-top: 16px; padding: 16px; }
        .data-quality-grid { display: grid; grid-template-columns: repeat(4, minmax(160px, 1fr)); gap: 14px; margin-bottom: 16px; }
        .result-table { width: 100%; border-collapse: collapse; font-size: 14px; }
        th, td { padding: 10px 12px; border-bottom: 1px solid #e5eaf2; text-align: left; }
        th { color: #475569; background: #f8fafc; }
        .error-box { color: #991b1b; background: #fee2e2; border: 1px solid #fecaca; padding: 12px; border-radius: 6px; }
        @media (max-width: 1100px) {
          .control-grid, .summary-grid, .chart-grid { grid-template-columns: 1fr; }
          .topbar { align-items: flex-start; gap: 12px; flex-direction: column; }
        }
            </style>
          </head>
          <body>
            {%app_entry%}
            <footer>
              {%config%}
              {%scripts%}
              {%renderer%}
            </footer>
          </body>
        </html>
        """
    )


def main() -> None:
    app = create_app()
    app.run(debug=False, host="0.0.0.0", port=8050)


if __name__ == "__main__":
    main()
