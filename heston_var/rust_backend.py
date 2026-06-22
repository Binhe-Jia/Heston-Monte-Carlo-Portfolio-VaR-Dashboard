from __future__ import annotations

import importlib.util


def rust_backend_available() -> bool:
    return importlib.util.find_spec("heston_var_rust") is not None


def rust_backend_info() -> str:
    if not rust_backend_available():
        return "heston_var_rust is not installed"
    import heston_var_rust

    if hasattr(heston_var_rust, "backend_info"):
        return str(heston_var_rust.backend_info())
    return "heston_var_rust is installed, but this is an older build without backend_info"
