"""End-to-end pipeline used by the UI, the scaling study and the CLI:
spec -> network -> input spike trains -> simulation -> analysis -> saved results.
"""

from __future__ import annotations

import datetime as _dt
import inspect
import json
import os
import re
import time

from . import pynn_runner
from .analysis import analyse, synaptic_event_count
from .backends import simulate
from .builder import INIT_STEP, OUTPUT_LATENCY, build, summary
from .inputs import generate
from .reference import run_reference
from .spec import normalise, sequence_for_n, template_counter, template_sequence, validate

RESULTS_DIR = os.environ.get("FSM_SNN_RESULTS", os.path.join(os.getcwd(), "results"))


def _sim_defaults(p: dict | None) -> dict:
    p = dict(p or {})
    p.setdefault("backend", "ideal")
    p.setdefault("dt", 1.0)
    p.setdefault("mode", "planted")
    p.setdefault("n_events", 60)
    p.setdefault("min_gap", 5)
    p.setdefault("max_gap", 15)
    p.setdefault("p_input", 0.4)
    p.setdefault("planted", 3)
    p.setdefault("seed", 1)
    p.setdefault("manual", {})
    p.setdefault("weights", None)
    p.setdefault("neuron_params", None)
    p.setdefault("save", True)
    return p


def make_inputs(spec, p):
    return generate(spec, mode=p["mode"], n_events=int(p["n_events"]),
                    min_gap=int(p["min_gap"]), max_gap=int(p["max_gap"]),
                    p_input=float(p["p_input"]), planted=int(p["planted"]),
                    seed=int(p["seed"]), manual=p.get("manual"), dt=float(p["dt"]))


def run(spec: dict, params: dict | None = None) -> dict:
    s = normalise(spec)
    p = _sim_defaults(params)
    errors, warnings = validate(s)
    if errors:
        return dict(ok=False, errors=errors, warnings=warnings)
    net = build(s)
    inp = make_inputs(s, p)
    last = max([e for e, _ in inp["events"]] + [INIT_STEP])
    duration = last + 30
    t0 = time.time()
    spikes = simulate(net, inp["trains"], backend=p["backend"], dt=float(p["dt"]),
                      duration_steps=duration, neuron_params=p["neuron_params"],
                      weights=p["weights"], seed=int(p["seed"]))
    elapsed = time.time() - t0
    res = analyse(s, net, spikes, inp["events"], inp["trains"], duration)
    syn_ev = synaptic_event_count(net, spikes, inp["trains"])
    res["metrics"]["synaptic_events"] = syn_ev
    res["metrics"]["synaptic_events_per_event"] = (syn_ev / len(inp["events"])
                                                   if inp["events"] else None)
    res["metrics"]["wall_time_s"] = round(elapsed, 3)
    out = dict(ok=True, warnings=warnings + res.pop("warnings"), summary=summary(s),
               duration=duration, dt=float(p["dt"]), backend=p["backend"],
               events=[[e, sorted(ev)] for e, ev in inp["events"]],
               planted=inp["planted"], **res)
    out["reference"] = dict(detections=res["reference"]["detections"])
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
    rows = out["rows"]
    colors = {"input": "#3b6fb6", "op": "#e08a1e", "state": "#3c9a5f", "output": "#7b3fa6"}
    fig, ax = plt.subplots(figsize=(12, 0.45 * len(rows) + 1.2))
    for k, row in enumerate(reversed(rows)):
        if row.get("expected"):
            ax.scatter(row["expected"], [k] * len(row["expected"]), marker="|", s=90,
                       color="#cccccc", linewidths=3)
        ax.scatter(row["spikes"], [k] * len(row["spikes"]), marker="|", s=60,
                   color=colors.get(row["kind"], "k"))
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r["label"] for r in reversed(rows)])
    ax.set_xlabel(f"Time (steps of {out['dt']} ms)")
    ax.set_xlim(0, out["duration"])
    ax.set_title("Obtained (colour) vs. expected (grey)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------
def scaling(family: str = "counter", n_min: int = 2, n_max: int = 10,
            simulate_runs: bool = True, params: dict | None = None) -> dict:
    """Scaling study: resources, latency and (optionally) correctness vs n."""
    p = _sim_defaults(params)
    p["save"] = False
    rows = []
    for n in range(int(n_min), int(n_max) + 1):
        sp = template_counter(n) if family == "counter" else template_sequence(sequence_for_n(n))
        c = summary(sp)
        row = dict(n=n, name=sp["name"], states=c["states"], transitions=c["transitions"],
                   neurons=c["neurons"], synapses=c["synapses"],
                   excitatory=c["excitatory"], inhibitory=c["inhibitory"],
                   output_latency=c["output_latency"], state_latency=c["state_latency"],
                   min_event_gap=c["min_event_gap"])
        if simulate_runs:
            r = run(sp, p)
            m = r["metrics"]
            row.update(state_agreement=m["state_agreement"], tp=m["tp"], fp=m["fp"],
                       fn=m["fn"], synaptic_events_per_event=m["synaptic_events_per_event"],
                       wall_time_s=m["wall_time_s"])
        rows.append(row)
    return dict(ok=True, family=family, rows=rows)


# --------------------------------------------------------------------------
SCRIPT_TEMPLATE = '''#!/usr/bin/env python3
"""Stand-alone PyNN script generated by the FSM -> SNN application.

FSM: {name}
Generated: {date}

Run it on the machine that has PyNN and the chosen back-end installed, e.g.

    python {filename} --backend spinnaker

Results (spikes, metrics and a raster plot) are written to ./results_{slug}/
Neurons: {neurons}   Synapses: {synapses}   Time step: {dt} ms
"""

import argparse
import json
import os

# ---------------------------------------------------------------- data
FSM = json.loads({fsm_json!r})
NETLIST = json.loads({net_json!r})
TRAINS = json.loads({trains_json!r})          # input name -> spike steps
EXPECTED_DETECTIONS = {expected!r}            # event steps that complete a pattern
OUTPUT_LATENCY = {out_lat}                    # steps from event to output spike
DT = {dt}
DURATION_STEPS = {duration}
WEIGHTS = {weights!r}
NEURON_PARAMS = {nparams!r}

# ---------------------------------------------------------------- PyNN runner
{runner_src}

# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="{backend}", choices=sorted(BACKEND_MODULES))
    ap.add_argument("--dt", type=float, default=DT)
    args = ap.parse_args()
    spikes = run_pynn(NETLIST, TRAINS, backend=args.backend, dt=args.dt,
                      duration_steps=DURATION_STEPS, neuron_params=NEURON_PARAMS,
                      weights=WEIGHTS)
    out_idx = NETLIST["output"]
    out = spikes.get(out_idx, []) if out_idx is not None else []
    first = None
    init_latch = [v for k, v in NETLIST["latches"].items()
                  if any(st["id"] == k and st["initial"] for st in FSM["states"])]
    if init_latch:
        a = spikes.get(init_latch[0][1], [])
        first = a[0] - (NETLIST["init_step"] + 2) if a else None
    offset = first or 0
    expected = [e + OUTPUT_LATENCY + offset for e in EXPECTED_DETECTIONS]
    left = list(out)
    tp = 0
    for t in expected:
        hit = next((x for x in left if abs(x - t) <= 1), None)
        if hit is not None:
            left.remove(hit)
            tp += 1
    print("Expected detections:", len(expected), " TP:", tp,
          " FN:", len(expected) - tp, " FP:", len(left), " offset:", offset)
    folder = "results_{slug}"
    os.makedirs(folder, exist_ok=True)
    labels = [n["label"] for n in NETLIST["neurons"]]
    with open(os.path.join(folder, "spikes_" + args.backend + ".json"), "w") as f:
        json.dump({{labels[i] + " #" + str(i): v for i, v in spikes.items()}}, f, indent=1)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        rows = [(name, TRAINS.get(name, [])) for name in FSM["inputs"]]
        for st in FSM["states"]:
            _, a, b = NETLIST["latches"][st["id"]]
            rows.append((st["name"], sorted(set(spikes.get(a, [])) | set(spikes.get(b, [])))))
        if out_idx is not None:
            rows.append(("Output", out))
        fig, ax = plt.subplots(figsize=(12, 0.45 * len(rows) + 1))
        for k, (name, sp) in enumerate(reversed(rows)):
            ax.scatter(sp, [k] * len(sp), marker="|", s=60)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r[0] for r in reversed(rows)])
        ax.set_xlabel("Time (steps of %s ms)" % args.dt)
        fig.tight_layout()
        fig.savefig(os.path.join(folder, "raster_" + args.backend + ".png"), dpi=130)
    except Exception as exc:  # plotting is optional
        print("Plot skipped:", exc)
    print("Results saved in", folder)


if __name__ == "__main__":
    main()
'''


def export_script(spec: dict, params: dict | None = None) -> dict:
    s = normalise(spec)
    p = _sim_defaults(params)
    errors, _ = validate(s)
    if errors:
        return dict(ok=False, errors=errors)
    net = build(s)
    inp = make_inputs(s, p)
    ref = run_reference(s, inp["events"])
    last = max([e for e, _ in inp["events"]] + [INIT_STEP])
    d = net.to_dict()
    d["init_step"] = INIT_STEP
    backend = p["backend"] if p["backend"] != "ideal" else "spinnaker"
    slug = _slug(s["name"])
    filename = f"run_{slug}.py"
    runner_src = inspect.getsource(pynn_runner)
    c = net.counts()
    text = SCRIPT_TEMPLATE.format(
        name=s["name"], date=_dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        filename=filename, slug=slug, neurons=c["neurons"], synapses=c["synapses"],
        dt=float(p["dt"]), fsm_json=json.dumps(s, ensure_ascii=False),
        net_json=json.dumps(d, ensure_ascii=False), trains_json=json.dumps(inp["trains"]),
        expected=ref["detections"], out_lat=OUTPUT_LATENCY, duration=last + 30,
        weights=p["weights"], nparams=p["neuron_params"], runner_src=runner_src,
        backend=backend)
    return dict(ok=True, filename=filename, text=text)
