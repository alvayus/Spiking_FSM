"""Entry point of the web version (GitHub Pages).

The page runs this package in the browser with Pyodide and calls :func:`handle`
instead of the local server: the same routes, minus the ones that need the
local disk or a PyNN simulator.
"""

from __future__ import annotations

import json
import traceback

from . import server

WEB_REASON = "Not available in the web version: run the application locally to use SpiNNaker."


def handle(method: str, path: str, body_json: str = "{}") -> str:
    """Answer one API call, as the local server would, with a JSON string."""
    fn = server.ROUTES.get((method, path))
    if fn is None or path.startswith("/api/fs/"):
        return json.dumps(dict(ok=False, errors=[f"{path} is not available in the web version."]))
    try:
        body = json.loads(body_json or "{}")
        if path == "/api/simulate":
            body["params"] = dict(body.get("params") or {}, save=False)   # no results folder
        out = fn(body)
        if path == "/api/info":
            out["web"] = True
            for b in out["backends"]:
                if b["id"] != "ideal":
                    b.update(available=False, reason=WEB_REASON)
        return json.dumps(out, ensure_ascii=False)
    except Exception as exc:  # report errors to the UI, like the server
        traceback.print_exc()
        return json.dumps(dict(ok=False, errors=[f"{type(exc).__name__}: {exc}"]))
