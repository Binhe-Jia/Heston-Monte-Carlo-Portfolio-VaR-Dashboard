# Heston Monte Carlo Portfolio VaR Dashboard

This repository contains a Python Dash dashboard and research notebook for portfolio VaR and Expected Shortfall analysis.

The dashboard compares:

- Historical simulation
- Gaussian VaR
- EWMA VaR
- Heston Monte Carlo VaR

It supports 1-day, 1-month, and 1-year risk horizons, flexible CSV portfolio uploads, model diagnostics, and visual comparison of historical versus simulated P&L distributions.

## Repository Contents

```text
app.py                                      # Dash web app entry point
heston_var/                                 # VaR engine and dashboard code
configs/                                    # Example portfolio CSV datasets
notebooks/heston_var_research_workflow.ipynb # Research notebook
requirements.txt                            # Python dependencies
```

## Quick Start

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the dashboard:

```powershell
python app.py
```

Open:

```text
http://127.0.0.1:8050
```

## Portfolio CSV Format

The dashboard accepts CSV files with flexible column names.

Ticker columns can be named:

```text
ticker, symbol, asset, security, ric, bbg
```

Exposure columns can be named:

```text
weight, allocation, market_value, notional, shares, quantity
```

If using `shares` or `quantity`, include a `price` column when you want market-value weighting.

Example datasets are in `configs/`.

## Notes

- Live market data is loaded through `yahooquery`, so dashboard runs using live data require internet access.
- The dashboard works without the Rust backend and falls back to the Python/NumPy simulation path.
- The Windows executable build artifacts are intentionally excluded from this upload folder.
