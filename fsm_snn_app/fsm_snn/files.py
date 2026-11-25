"""Files on the local disk: Open, Save and the recent files, and images of the network.

The server reads and writes the files itself, so saving over the file that was opened
works in any browser (including embedded ones without the File System Access API).
Machines are ``.json`` files; images of the network (``.svg``, ``.png``) can also be
written. Nothing else is listed, read or written.
"""

from __future__ import annotations

import base64
import os
import string

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DESIGNS = os.path.join(APP_DIR, "designs")
WRITABLE = (".json", ".svg", ".png", ".md", ".csv")


def _places() -> list[dict]:
    out = [dict(name="Designs", path=DESIGNS), dict(name="Home", path=os.path.expanduser("~"))]
    if os.name == "nt":
        out += [dict(name=f"{c}:", path=f"{c}:\\") for c in string.ascii_uppercase if os.path.isdir(f"{c}:\\")]
    else:
        out.append(dict(name="/", path="/"))
    return out


def _checked_path(path, exts=(".json",)) -> str:
    if not path or not isinstance(path, str):
        raise ValueError("No file given.")
    full = os.path.abspath(os.path.expanduser(path))
    if not full.lower().endswith(tuple(exts)):
        raise ValueError(f"Only {', '.join(exts)} files can be used here.")
    return full


def api_list(body):
    """Folders and files of one type (.json by default) of a folder (designs/ by default)."""
    ext = body.get("ext") if body.get("ext") in WRITABLE else ".json"
    d = os.path.abspath(os.path.expanduser(body.get("dir") or DESIGNS))
    if not os.path.isdir(d):
        d = DESIGNS if os.path.isdir(DESIGNS) else os.path.expanduser("~")
    try:
        names = os.listdir(d)
    except OSError as exc:
        return dict(ok=False, dir=d, errors=[f"Cannot read {d}: {exc.strerror or exc}"], places=_places())
    folders, files = [], []
    for n in names:
        if n.startswith(".") or n.startswith("$"):
            continue
        full = os.path.join(d, n)
        try:
            if os.path.isdir(full):
                folders.append(dict(name=n, dir=True))
            elif n.lower().endswith(ext) and os.path.isfile(full):
                files.append(dict(name=n, dir=False, mtime=os.path.getmtime(full)))
        except OSError:
            continue
    key = lambda e: e["name"].lower()
    parent = os.path.dirname(d)
    return dict(ok=True, dir=d, sep=os.sep, parent=parent if parent != d else None,
                entries=sorted(folders, key=key) + sorted(files, key=key), places=_places())


def api_read(body):
    path = _checked_path(body.get("path"))
    if not os.path.isfile(path):
        return dict(ok=False, missing=True, errors=[f"{os.path.basename(path)} no longer exists."])
    with open(path, encoding="utf-8") as f:
        return dict(ok=True, path=path, name=os.path.basename(path), text=f.read())


def api_write(body):
    """Write a machine, an SVG image, a Markdown or CSV summary (``text``) or a PNG image (``base64``).
    A new file (Save as) does not replace an existing one unless asked."""
    path = _checked_path(body.get("path"), WRITABLE)
    if os.path.exists(path) and not body.get("overwrite"):
        return dict(ok=False, exists=True, path=path, name=os.path.basename(path))
    folder = os.path.dirname(path)
    if not os.path.isdir(folder):
        return dict(ok=False, errors=[f"The folder {folder} does not exist."])
    tmp = path + ".tmp"
    if path.lower().endswith(".png"):
        with open(tmp, "wb") as f:
            f.write(base64.b64decode(body.get("base64") or ""))
    else:
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            f.write(body.get("text") or "")
    os.replace(tmp, path)                     # never leaves a half-written file
    return dict(ok=True, path=path, name=os.path.basename(path))
