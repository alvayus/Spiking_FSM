"""End-to-end pipeline used by the UI and the CLI:
spec -> network -> input spike trains -> simulation -> analysis -> saved results.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import random
import re
import time

from . import pynn_runner
from .analysis import analyse
from .backends import SpiNNakerUnavailable, simulate
from .builder import INIT_STEP, build, summary
from .inputs import generate
from .spec import normalise, validate

# Simulations the user asked to cancel (job ids sent by the page with the parameters).
CANCELLED = set()


def request_cancel(job) -> dict:
    """Ask the simulation ``job`` to stop at its next safe point (see run_pynn)."""
    if job:
        CANCELLED.add(str(job))
    return dict(ok=True)


RESULTS_DIR = os.environ.get("FSM_SNN_RESULTS", os.path.join(os.getcwd(), "results"))


def _sim_defaults(p: dict | None) -> dict:
    p = dict(p or {})
    p.setdefault("backend", "ideal")
    p.setdefault("dt", 1.0)
    p.setdefault("mode", "random")
    p.setdefault("n_events", 60)
    p.setdefault("min_gap", 5)
    p.setdefault("max_gap", 15)
    p.setdefault("p_input", 0.4)
    # a new seed on every run unless one is given (tests, reproducing a saved run);
    # the seed used is returned and saved with the results
    if p.get("seed") is None:
        p["seed"] = random.SystemRandom().randrange(1, 2 ** 31)
    p.setdefault("manual", {})
    p.setdefault("weights", None)
    p.setdefault("neuron_params", None)
    p.setdefault("save", True)
    p.setdefault("time_scale_factor", None)      # SpiNNaker: None = the smallest supported
    return p


def make_inputs(spec, p):
    return generate(spec, mode=p["mode"], n_events=int(p["n_events"]),
                    min_gap=int(p["min_gap"]), max_gap=int(p["max_gap"]),
                    p_input=float(p["p_input"]),
                    seed=int(p["seed"]), manual=p.get("manual"), dt=float(p["dt"]))


def run(spec: dict, params: dict | None = None) -> dict:
    s = normalise(spec)
    p = _sim_defaults(params)
    errors, warnings = validate(s)
    if errors:
        return dict(ok=False, errors=errors, warnings=warnings, invalid_spec=True)
    tsf = p["time_scale_factor"]
    if p["backend"] != "ideal" and tsf and int(tsf) < pynn_runner.time_scale_factor(float(p["dt"])):
        return dict(ok=False, errors=[f"With a {float(p['dt']):g} ms time step SpiNNaker needs a time scale factor "
                                      f"of at least {pynn_runner.time_scale_factor(float(p['dt']))}."])
    net = build(s)
    inp = make_inputs(s, p)
    last = max([e for e, _ in inp["events"]] + [INIT_STEP])
    duration = last + 30
    t0 = time.time()
    job = str(p.get("job") or "")
    try:
        spikes = simulate(net, inp["trains"], backend=p["backend"], dt=float(p["dt"]),
                          duration_steps=duration, neuron_params=p["neuron_params"],
                          weights=p["weights"], should_stop=lambda: bool(job) and job in CANCELLED,
                          time_scale=p["time_scale_factor"])
    except pynn_runner.SimulationCancelled:
        return dict(ok=False, cancelled=True, errors=["Simulation cancelled."])
    except SpiNNakerUnavailable as e:
        return dict(ok=False, errors=[str(e)])
    except Exception as e:                    # noqa: BLE001 - report simulator failures to the UI
        msg = str(e).splitlines()[0] if str(e) else ""
        return dict(ok=False, errors=[f"The {p['backend']} simulation failed: {type(e).__name__}"
                                      + (f": {msg}" if msg else "")])
    finally:
        CANCELLED.discard(job)
    elapsed = time.time() - t0
    res = analyse(s, net, spikes, inp["events"], inp["trains"], duration, dt=float(p["dt"]))
    res["metrics"]["wall_time_s"] = round(elapsed, 3)
    res.pop("reference")
    notes = []
    if inp.get("dropped"):
        notes.append(f"{inp['dropped']} random spike time(s) were left out because they fell less than "
                     f"{max(int(p['min_gap']), 1)} steps (the minimum spacing) from one of the typed spikes.")
    out = dict(ok=True, warnings=warnings + res.pop("warnings"), notes=notes, summary=summary(s),
               duration=duration, dt=float(p["dt"]), backend=p["backend"], seed=int(p["seed"]), mode=p["mode"],
               events=[[e, sorted(ev)] for e, ev in inp["events"]],
               manual_steps=inp.get("manual_steps", []),
               **res)
    if p.get("save"):
        out["saved_to"] = save_results(s, p, inp, spikes, out, net)
    return out


# --------------------------------------------------------------------------
def _slug(text):
    import unicodedata
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_")[:40] or "fsm"


def save_results(s, p, inp, spikes, out, net) -> str:
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    folder = os.path.join(RESULTS_DIR, f"{stamp}_{_slug(s['name'])}_{p['backend']}")
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "fsm.json"), "w", encoding="utf-8") as f:
        json.dump(s, f, ensure_ascii=False, indent=2)
    with open(os.path.join(folder, "simulation.json"), "w", encoding="utf-8") as f:
        json.dump(dict(params={k: v for k, v in p.items() if k != "save"},
                       trains=inp["trains"], events=out["events"],
                       metrics=out["metrics"], summary=out["summary"],
                       spikes={net.neurons[i]["label"] + f" #{i}": v
                               for i, v in spikes.items()}),
                  f, ensure_ascii=False, indent=1)
    _save_raster_png(out, os.path.join(folder, "raster.png"))
    return folder


def _save_raster_png(out, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    # same layout as the figures of the paper: inputs at the bottom, then OP, a
    # separator, the states and Output on top; time in ms
    rows, dt = out["rows"], out["dt"]
    colors = {"input": "#6c8ebf", "op": "#d79b00", "state": "#82b366", "output": "#9673a6"}
    fig, ax = plt.subplots(figsize=(12, 0.32 * len(rows) + 1.2))
    ax.hlines(range(len(rows)), 0, out["duration"] * dt, linestyles="dotted",
              linewidth=0.6, color="indigo", alpha=0.3)
    n_src = sum(r["kind"] in ("input", "op") for r in rows)
    if 0 < n_src < len(rows):
        ax.axhline(n_src - 0.5, linewidth=0.4, color="indigo")
    for k, row in enumerate(rows):
        if row.get("expected"):
            ax.scatter([t * dt for t in row["expected"]], [k] * len(row["expected"]),
                       marker="|", s=90, color="#dddddd", linewidths=3)
        ax.scatter([t * dt for t in row["spikes"]], [k] * len(row["spikes"]), marker="|",
                   s=50, color=colors.get(row["kind"], "k"))
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r["label"] for r in rows])
    ax.set_xlabel("Time (ms)")
    ax.set_xlim(0, out["duration"] * dt)
    ax.set_ylim(-1, len(rows))
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------


def export_script(spec: dict, params: dict | None = None) -> dict:
    """Stand-alone script of the Simulate tab: the readable network of the Network tab,
    the input spikes, the run, the comparison with the reference and the saved results."""
    from .codegen import generate_code          # local import: codegen imports this module
    return generate_code(spec, params, simulation=True)
