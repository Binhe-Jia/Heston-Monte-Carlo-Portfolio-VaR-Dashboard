from __future__ import annotations

from heston_var.dashboard import create_app


app = create_app()
server = app.server


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=8050)
