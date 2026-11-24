#!/usr/bin/env python3
"""Repeat the experiments of the paper with the Spiking FSM builder and compare.

For every run saved in results/*.pickle, the network that the application
generates for the same machine (fsm_snn_app/designs/) is simulated with exactly
the same input spikes. Three things are compared, event by event:

  * paper:    the spikes recorded on SpiNNaker for the paper (the pickle);
  * app:      the network of the application, on SpiNNaker (or the Ideal simulator);
  * expected: the FSM itself (the reference automaton of the application).

A state is read from the spikes of its latch ring. For each input event, the
active states some steps after it (the latency of each network, chosen as the one
that agrees best and reported) are compared with the states the FSM is in after
the event. A figure per run and a summary are written to
fsm_snn_app/designs/paper_comparison/.

    python app_vs_paper.py                    # app on SpiNNaker
    python app_vs_paper.py --backend ideal    # app on the Ideal simulator
"""

import argparse
import glob
import json
import os
import pickle
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.join(HERE, "..", "fsm_snn_app")
sys.path.insert(0, APP)

from fsm_snn import backends                                # noqa: E402
from fsm_snn.analysis import _fsm_intervals, _state_spikes  # noqa: E402
from fsm_snn.builder import MIN_EVENT_GAP, build            # noqa: E402
from fsm_snn.inputs import events_from_trains               # noqa: E402
from fsm_snn.reference import run_reference                 # noqa: E402
from fsm_snn.spec import normalise                          # noqa: E402

OUT = os.path.join(APP, "designs", "paper_comparison")

# experiment (pickle prefix) -> design, duration (ms), description
EXPERIMENTS = [
    ("simple_case_only_forward", "simple_FSM_forward.json", 100, "Simple FSM, forward transitions only (SG1 every 3-6 ms)"),
    ("simple_case_complete", "simple_FSM.json", 100, "Simple FSM, forward (SG1) and backward (SG2) transitions, every 3-6 ms"),
    ("simple_case_max_freq_3ms", "simple_FSM_forward.json", 100, "Simple FSM, maximum frequency: SG1 every 3 ms"),
    ("simple_case_max_freq_2ms", "simple_FSM_forward.json", 100, "Simple FSM, maximum frequency: SG1 every 2 ms"),
    ("simple_case_max_freq_1ms", "simple_FSM_forward.json", 100, "Simple FSM, maximum frequency: SG1 every 1 ms"),
    ("complex_case", "pattern_detector.json", 500, "Pattern detector, inputs A-D 1-49 ms apart"),
    ("complex_case_v2_max_freq", "pattern_detector.json", 500, "Pattern detector, maximum frequency (1-5 ms apart)"),
]
PAPER_STATES = ["S0", "S1", "S2"]          # states with a latch in both networks


def load_paper(prefix, path):
    """Input spike trains (ms) and spikes of the paper's latch rings (and detections)."""
    with open(path, "rb") as f:
        d = pickle.load(f)
    ints = lambda xs: sorted(int(round(float(x))) for x in xs)
    if prefix.startswith("simple"):
        trains = {"SGf": ints(d[0])}
        if len(d[1]):
            trains["SGb"] = ints(d[1])
        latches, detections = d[2], None
    else:
        trains = dict(zip("ABCD", (ints(t) for t in d[0])))
        latches, detections = d[2], ints(d[4][2].spiketrains[2])     # T2: pattern detected
    states = {}
    for name, seg in zip(PAPER_STATES, latches):
        ring = seg.spiketrains[1:3]                                  # N2 <-> N3, as in the app
        states[name] = set(int(round(float(t))) for st in ring for t in st)
    return trains, states, detections


def agreement(states, events, after, latency, duration, names):
    """Fraction of events after which exactly the expected states (among names)
    are active `latency` steps later. Also returned per event (None = not evaluable)."""
    ok = []
    for (e, _), exp in zip(events, after):
        t = e + latency
        if t >= duration:
            ok.append(None)
            continue
        got = {n for n in names if t in states.get(n, ())}
        ok.append(got == {n for n in exp if n in names})
    vals = [x for x in ok if x is not None]
    return (sum(vals) / len(vals) if vals else None), ok


def best_latency(states, events, after, duration, names):
    best = None
    for L in range(1, 13):
        a, per = agreement(states, events, after, L, duration, names)
        if a is not None and (best is None or a > best[1]):
            best = (L, a, per)
    return best


def spaced_mask(events):
    """Events at least MIN_EVENT_GAP steps from the previous and the next one."""
    steps = [e for e, _ in events]
    out = []
    for i, e in enumerate(steps):
        prev_ok = i == 0 or e - steps[i - 1] >= MIN_EVENT_GAP
        next_ok = i == len(steps) - 1 or steps[i + 1] - e >= MIN_EVENT_GAP
        out.append(prev_ok and next_ok)
    return out


def frac(per, mask=None):
    vals = [x for i, x in enumerate(per) if x is not None and (mask is None or mask[i])]
    return (sum(vals) / len(vals), len(vals)) if vals else (None, 0)


def plot(path, title, trains, panels, duration):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    inputs = list(trains)
    fig, axes = plt.subplots(len(panels), 1, figsize=(9, 1.0 + 1.25 * len(panels)), sharex=True)
    for ax, (ptitle, states, bars) in zip(axes, panels):
        rows = inputs + PAPER_STATES_ALL
        for k, name in enumerate(inputs):
            ax.vlines(trains[name], k - .3, k + .3, color="#6c8ebf", lw=.8)
        for j, name in enumerate(PAPER_STATES_ALL):
            y = len(inputs) + j
            if bars is not None:
                for a, b in bars.get(name, []):
                    ax.add_patch(plt.Rectangle((a, y - .3), b - a, .6, color="#82b366", lw=0))
            else:
                ax.vlines(sorted(states.get(name, ())), y - .3, y + .3, color="#82b366", lw=.6)
        ax.axhline(len(inputs) - .5, color="#4b3f72", lw=.5)
        ax.set_yticks(range(len(rows)), rows, fontsize=7)
        ax.set_ylim(-.7, len(rows) - .3)
        ax.set_title(ptitle, fontsize=8, loc="left")
        ax.tick_params(axis="x", labelsize=7)
        for k in range(len(rows)):
            ax.axhline(k, color="#c3b5d6", lw=.4, ls=":", zorder=0)
    axes[-1].set_xlim(0, duration)
    axes[-1].set_xlabel("Time (ms)", fontsize=8)
    fig.suptitle(title, fontsize=9, x=.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


PAPER_STATES_ALL = PAPER_STATES


def main():
    global PAPER_STATES_ALL
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default="spinnaker", choices=["spinnaker", "ideal"])
    ap.add_argument("--only", help="run only the experiments whose name starts with this")
    args = ap.parse_args()
    if args.backend == "spinnaker":
        ok, why = backends.spinnaker_reachable()
        if not ok:
            sys.exit(why)
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for prefix, design, duration, desc in EXPERIMENTS:
        if args.only and not prefix.startswith(args.only):
            continue
        with open(os.path.join(APP, "designs", design), encoding="utf-8") as f:
            spec = normalise(json.load(f))
        net = build(spec)
        names = {x["id"]: x["name"] for x in spec["states"]}
        PAPER_STATES_ALL = [names[x["id"]] for x in spec["states"]]
        files = [p for p in glob.glob(os.path.join(HERE, "results", prefix + "_*.pickle"))
                 if re.fullmatch(re.escape(prefix) + r"_\d+", os.path.basename(p)[:-7])]
        for path in sorted(files, key=lambda p: int(re.findall(r"_(\d+)\.pickle$", p)[0])):
            run = os.path.basename(path)[:-7]
            trains, paper_states, paper_det = load_paper(prefix, path)
            events = events_from_trains(trains)
            ref = run_reference(spec, events)
            after = [[names[s] for s in a] for a in ref["active_after"]]
            t0 = time.time()
            spikes = backends.simulate(net, trains, backend=args.backend, dt=1.0, duration_steps=duration)
            app_states = {names[sid]: set(_state_spikes(net, spikes, sid)) for sid in names}
            fsm = {names[k]: v for k, v in _fsm_intervals(spec, events, ref, duration).items()}
            # compare the states that both networks have (S0-S2)
            pL, pa, pper = best_latency(paper_states, events, after, duration, PAPER_STATES)
            aL, aa, aper = best_latency(app_states, events, after, duration, PAPER_STATES)
            mask = spaced_mask(events)
            same = [x == y if x is not None and y is not None else None for x, y in zip(pper, aper)]
            row = dict(experiment=prefix, run=run, events=len(events), spaced=sum(mask),
                       paper_latency=pL, paper=frac(pper)[0], paper_spaced=frac(pper, mask)[0],
                       app_latency=aL, app=frac(aper)[0], app_spaced=frac(aper, mask)[0],
                       same=frac(same)[0])
            if paper_det is not None:            # pattern detector: count of detections
                t_det = [t for t in ref["fired"] for t in t if t == "t4"]
                det_sid = next(x["id"] for x in spec["states"] if x["name"] == "Detected")
                app_det = sum(1 for t in sorted(_state_spikes(net, spikes, det_sid))
                              if t - 1 not in set(_state_spikes(net, spikes, det_sid)))
                row.update(det_expected=len(t_det), det_paper=len(paper_det), det_app=app_det)
            rows.append(row)
            pct = lambda x: "—" if x is None else f"{100 * x:.0f} %"
            print(f"{run:32s} events {len(events):3d}  paper {pct(row['paper'])} (L={pL})  "
                  f"app {pct(row['app'])} (L={aL})  [{time.time() - t0:.0f} s]", flush=True)
            plot(os.path.join(OUT, run + ".png"),
                 f"{run}: {desc}",
                 trains,
                 [(f"Paper, SpiNNaker (recorded): {pct(row['paper'])} of the events right (latency {pL} ms)", paper_states, None),
                  (f"Spiking FSM builder, {args.backend}: {pct(row['app'])} of the events right (latency {aL} ms)", app_states, None),
                  ("Expected: ideal FSM, without synaptic delays", None, fsm)],
                 duration)
            with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8") as f:
                json.dump(dict(backend=args.backend, runs=rows), f, indent=1)


if __name__ == "__main__":
    main()
