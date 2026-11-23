#!/usr/bin/env python3
"""Start the visual FSM -> SNN application.

    python run_app.py            # opens http://127.0.0.1:8765 in the browser
    python run_app.py --port 9000 --no-browser
"""

import argparse
import os
import sys
import threading
import webbrowser

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fsm_snn.server import serve  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="FSM -> SNN visual application (PyNN)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    httpd = serve(args.host, args.port)
    url = f"http://{args.host}:{args.port}/"
    print(f"FSM -> SNN application running at {url}  (Ctrl+C to quit)")
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        httpd.server_close()


if __name__ == "__main__":
    main()
