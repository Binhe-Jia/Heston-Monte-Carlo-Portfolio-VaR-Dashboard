# Heston Monte Carlo Portfolio VaR Dashboard

This repository contains a Python Dash dashboard, research notebook, sample datasets, and optional Rust acceleration for portfolio VaR and Expected Shortfall analysis.

The dashboard compares:

- Historical simulation
- Gaussian VaR
- EWMA VaR
- Heston Monte Carlo VaR

It supports 1-day, 1-month, and 1-year risk horizons, flexible CSV portfolio uploads, model diagnostics, and visual comparison of historical versus simulated P&L distributions. The Heston Monte Carlo path can run through the Python/NumPy backend or the Rust/PyO3 backend when installed.

## Repository Contents

```text
app.py                                      # Dash web app entry point
heston_var/                                 # VaR engine and dashboard code
rust_engine/                                # Optional Rust/PyO3 Monte Carlo backend
configs/                                    # Example portfolio CSV datasets
notebooks/heston_var_research_workflow.ipynb # Research notebook
requirements.txt                            # Python dependencies
requirements-rust.txt                       # Optional Rust build helper dependency
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

## Optional Rust Acceleration

The dashboard works without Rust. By default, `backend="auto"` uses the Rust backend when the extension is installed and falls back to the Python/NumPy simulation path otherwise.

To build the Rust backend locally, install Rust and Cargo first:

```powershell
rustc --version
cargo --version
```

Then install `maturin` and build the extension:

```powershell
python -m pip install -r requirements-rust.txt
cd rust_engine
python -m maturin develop --release
cd ..
```

You can confirm that Python sees the Rust backend with:

```powershell
python -c "from heston_var.rust_backend import rust_backend_available, rust_backend_info; print(rust_backend_available()); print(rust_backend_info())"
```

If this prints `True`, the dashboard's `Auto` backend will use Rust for Heston Monte Carlo VaR. If it prints `False`, the dashboard still runs with the Python backend.

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
- The Rust backend source is included for reproducibility and speed benchmarking, but compiled artifacts such as `target/` and `.whl` files should not be committed.
- The Windows executable build artifacts are intentionally excluded from this upload folder.
