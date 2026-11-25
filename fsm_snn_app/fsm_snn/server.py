"""Local web server for the visual application (standard library only)."""

from __future__ import annotations

import json
import mimetypes
import os
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import __version__, files
from .backends import available_backends
from .builder import ARCHITECTURES, DEFAULT_ARCHITECTURE, INIT_STEP, build
from .codegen import generate_code
from .energy import estimate as estimate_energy
from .pipeline import export_script, request_cancel, run
from .spec import normalise, notes, template_simple, validate

STATIC = os.path.join(os.path.dirname(__file__), "static")


def api_info(_):
    """Simulators available, timing limits for the inputs and the machine shown on first start."""
    return dict(backends=available_backends(), default_spec=template_simple(),
                timing=dict(ready_step=INIT_STEP + 2),
                architectures=ARCHITECTURES, default_architecture=DEFAULT_ARCHITECTURE)


def api_check(body):
    """Live validation of the machine being edited (Design tab)."""
    s = normalise(body.get("spec"))
    errors, warnings = validate(s)
    return dict(errors=errors, warnings=warnings, notes=notes(s))


def api_network(body):
    """Neuron-level netlist of the machine, for the Network tab."""
    s = normalise(body.get("spec"))
    errors, _ = validate(s)
    if errors:
        return dict(ok=False, errors=errors)
    net = build(s)
    return dict(ok=True, net=net.to_dict(), counts=net.counts())


def api_energy(body):
    """Energy per spike of every neuron (Energy tab, before a simulation)."""
    s = normalise(body.get("spec"))
    errors, _ = validate(s)
    if errors:
        return dict(ok=False, errors=errors)
    return dict(ok=True, **estimate_energy(build(s)))


def api_code(body):
    """Readable PyNN code of the network (Network tab)."""
    return generate_code(body.get("spec"), body.get("params"))


def api_simulate(body):
    return run(body.get("spec"), body.get("params"))


def api_cancel(body):
    """Cancel a running simulation (it stops at its next safe point)."""
    return request_cancel(body.get("job"))


def api_script(body):
    return export_script(body.get("spec"), body.get("params"))


ROUTES = {
    ("GET", "/api/info"): api_info,
    ("POST", "/api/check"): api_check,
    ("POST", "/api/network"): api_network,
    ("POST", "/api/code"): api_code,
    ("POST", "/api/energy"): api_energy,
    ("POST", "/api/simulate"): api_simulate,
    ("POST", "/api/cancel"): api_cancel,
    ("POST", "/api/script"): api_script,
    ("POST", "/api/fs/list"): files.api_list,
    ("POST", "/api/fs/read"): files.api_read,
    ("POST", "/api/fs/write"): files.api_write,
}


class Handler(BaseHTTPRequestHandler):
    server_version = "SpikingFSMBuilder/" + __version__

    def log_message(self, fmt, *args):  # quieter console
        if "/api/" in (str(args[0]) if args else ""):
            super().log_message(fmt, *args)

    def _send(self, code, payload, ctype="application/json; charset=utf-8"):
        data = payload if isinstance(payload, bytes) else json.dumps(
            payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _same_origin(self):
        """The file routes only answer the application's own page, not other web sites."""
        origin = self.headers.get("Origin")
        return origin is None or origin.split("://", 1)[-1] == self.headers.get("Host")

    def _handle(self, method):
        path = self.path.split("?", 1)[0]
        fn = ROUTES.get((method, path))
        if fn and path.startswith("/api/fs/") and not self._same_origin():
            self._send(403, dict(ok=False, errors=["Forbidden"]))
            return
        if fn:
            try:
                body = {}
                if method == "POST":
                    n = int(self.headers.get("Content-Length") or 0)
                    body = json.loads(self.rfile.read(n) or b"{}")
                self._send(200, fn(body))
            except Exception as exc:  # report errors to the UI
                traceback.print_exc()
                self._send(500, dict(ok=False, errors=[f"{type(exc).__name__}: {exc}"]))
            return
        if method == "GET":
            rel = "index.html" if path in ("/", "/index.html") else path.lstrip("/")
            if rel.startswith("static/"):
                rel = rel[len("static/"):]
            full = os.path.normpath(os.path.join(STATIC, rel))
            if full.startswith(STATIC) and os.path.isfile(full):
                ctype = mimetypes.guess_type(full)[0] or "application/octet-stream"
                if ctype.startswith("text/") or ctype.endswith("javascript"):
                    ctype += "; charset=utf-8"
                with open(full, "rb") as f:
                    self._send(200, f.read(), ctype)
                return
        self._send(404, dict(ok=False, errors=["Not found"]))

    def do_GET(self):
        self._handle("GET")

    def do_POST(self):
        self._handle("POST")


def serve(host="127.0.0.1", port=8765):
    # Import the PyNN back-ends now, on the main thread and only once, so that
    # /api/info answers at once and any import side effects happen at start-up.
    available_backends()
    httpd = ThreadingHTTPServer((host, port), Handler)
    return httpd
