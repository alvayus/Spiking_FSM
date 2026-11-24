#!/usr/bin/env python3
"""Build the web version of the Spiking FSM builder (a static site for GitHub Pages).

    python build_web.py              # writes _site/
    python build_web.py --out DIR

The site is the same page as the local application. web.js loads Pyodide (Python
compiled to WebAssembly) and runs the fsm_snn package in the browser, so Design,
Network (with its PyNN code) and Simulate with the Ideal simulator work without a
server. Files are downloaded instead of saved, and SpiNNaker is not available.
"""

import argparse
import json
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(HERE, "fsm_snn")
STATIC = os.path.join(PKG, "static")
DESIGNS = os.path.join(HERE, "designs")
PYODIDE_JS = "https://cdn.jsdelivr.net/pyodide/v0.27.7/full/pyodide.js"


def build(out):
    if os.path.isdir(out):
        shutil.rmtree(out)
    shutil.copytree(STATIC, os.path.join(out, "static"), ignore=shutil.ignore_patterns("index.html"))

    # the page, with Pyodide and web.js loaded before the application scripts
    with open(os.path.join(STATIC, "index.html"), encoding="utf-8") as f:
        html = f.read()
    hook = '<script src="static/util.js"></script>'
    assert hook in html, "index.html no longer loads static/util.js"
    html = html.replace(hook, f'<script src="{PYODIDE_JS}"></script>\n'
                              f'<script src="static/web.js"></script>\n{hook}')
    with open(os.path.join(out, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)

    # the Python package, file by file, and the list that web.js reads
    py = os.path.join(out, "py", "fsm_snn")
    os.makedirs(py)
    files = sorted(f for f in os.listdir(PKG) if f.endswith(".py"))
    for f in files:
        shutil.copy2(os.path.join(PKG, f), py)
    with open(os.path.join(out, "py", "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(files, f)

    # the example machines, offered in the Recent menu
    d = os.path.join(out, "designs")
    os.makedirs(d)
    examples = []
    for f in sorted(os.listdir(DESIGNS)) if os.path.isdir(DESIGNS) else []:
        if f.endswith(".json"):
            shutil.copy2(os.path.join(DESIGNS, f), d)
            with open(os.path.join(DESIGNS, f), encoding="utf-8") as fh:
                examples.append(dict(file=f, name=json.load(fh).get("name") or f[:-5]))
    with open(os.path.join(d, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False)

    open(os.path.join(out, ".nojekyll"), "w").close()     # serve the files as they are
    print(f"Web version written to {out} ({len(files)} Python modules, {len(examples)} examples)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build the web version (GitHub Pages)")
    ap.add_argument("--out", default=os.path.join(HERE, "_site"))
    build(os.path.abspath(ap.parse_args().out))
