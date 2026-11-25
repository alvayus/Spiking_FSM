"""Readable PyNN code of the network that implements a machine.

The code spells the network out: one population per gate or latch, named after
its role, and one projection per synapse with a comment that says what it does.
It is generated from the same netlist (``builder.build``), so it is the exact
network that the application simulates.

* Network tab (``generate_code``): the network, the input spikes and the run.
* Simulate tab (``generate_code(..., simulation=True)``): the same code plus the
  comparison with the reference automaton and the saved results, so the script
  needs nothing but PyNN (and, optionally, matplotlib for the plot).
"""

from __future__ import annotations

import datetime as _dt
import re

from .analysis import _expected_state_steps
from .builder import INIT_STEP, build
from .reference import run_reference
from .pynn_runner import DEFAULT_NEURON_PARAMS, DEFAULT_WEIGHTS
from .spec import normalise, option_to_text, validate


def _ident(text: str) -> str:
    t = re.sub(r"[^0-9A-Za-z_]+", "_", text).strip("_") or "x"
    return ("n_" + t) if t[0].isdigit() else t


def generate_code(spec: dict, params: dict | None = None, simulation: bool = False) -> dict:
    from .pipeline import _sim_defaults, make_inputs     # local import: avoids a cycle
    s = normalise(spec)
    errors, _ = validate(s)
    if errors:
        return dict(ok=False, errors=errors)
    p = _sim_defaults(params)
    net = build(s)
    inp = make_inputs(s, p)
    st_by_id = {st["id"]: st for st in s["states"]}
    trans = {t["id"]: t for t in s["transitions"]}

    # ---- names: one population per role ------------------------------------------
    used = set()

    def unique(name):
        base, k = _ident(name), 2
        name = base
        while name in used:
            name = f"{base}_{k}"
            k += 1
        used.add(name)
        return name

    src_var = {}
    for i, src in enumerate(net.sources):
        src_var[i] = unique("in_" + src["label"]) if src["kind"] == "input" else unique("init")
    where = {}            # neuron index -> (population variable, index inside it)
    pops = []             # (variable, size, comment)
    latch_of = {}
    for sid, (n1, n2, n3) in net.latches.items():
        var = unique("latch_" + st_by_id[sid]["name"])
        for j, n in enumerate((n1, n2, n3)):
            where[n] = (var, j)
        latch_of[sid] = var
    bypass = net.architecture == "parallel"
    sname = lambda sid: st_by_id[sid]["name"] if sid in st_by_id else sid
    term_info = {n: (tid, k) for tid, k, n in net.terms}
    for i, n in enumerate(net.neurons):
        if i in where:
            continue
        kind, label = n["kind"], n["label"]
        if kind == "op":
            var, com = unique("OP"), "input-driven OR of every input (operation signal)"
        elif kind == "delay_line":
            var = unique("delay_" + label.replace(" +", "_plus"))
            com = f"delay line: {label.rsplit(' +', 1)[0]} repeated {n['k']} step{'s' if n['k'] > 1 else ''} later"
        elif kind == "delay":
            var, com = unique("delay"), "delay neuron shared by the AND gates"
        elif kind == "not_input":
            name = label[4:]
            var, com = unique("NOT_" + name), f"NOT gate of input {name}"
        elif kind == "not_state":
            name = label[4:]
            var, com = unique("NOT_" + name), f"NOT gate of state {name}"
        elif kind == "relay":
            var, com = unique("copy_" + label.replace(" #", "_")), f"copy of input {label.split(' #')[0]} (1-step bypass vetoes)"
        elif kind == "no_event":
            var = unique("not_OP_" + str(n["g"]) + (("_" + label.split("#")[1]) if "#" in label else ""))
            com = f"NOT OP(t) AND OP(t-{n['g']}): no input spikes now, some {n['g']} step{'s' if n['g'] > 1 else ''} before (cancels the bypasses)"
        elif kind == "enter_or":
            var, com = unique(label.replace(" ", "_")), f"OR of the AND gates that set {sname(n['sid'])}"
        elif kind == "leave_or":
            var, com = unique(label.replace(" ", "_")), f"OR of the AND gates that reset {sname(n['sid'])}"
        elif kind == "term":
            tid, k = term_info[i]
            t = trans[tid]
            a, b = st_by_id[t["from"]]["name"], st_by_id[t["to"]]["name"]
            via = n.get("via")
            var = unique(f"AND_{a}_{b}" + (f"_{k + 1}" if len(t["options"]) > 1 else "")
                         + (f"_{ {'latch': 'latch', 'bypass 1': 'b1', 'bypass 2': 'b2'}[via]}" if via else ""))
            com = f"AND gate of {a} -> {b}: {option_to_text(n.get('cond', t['options'][k]))}" + (f" ({via})" if via else "")
        else:
            var, com = unique("output"), "output neuron"
        where[i] = (var, 0)
        pops.append((var, 1, com))

    def ref(pre):
        return (src_var[pre[1]], 0) if pre[0] == "src" else where[pre[1]]

    def name_of(pre):
        if pre[0] == "src":
            return net.sources[pre[1]]["label"]
        return net.neurons[pre[1]]["label"]

    # ---- one comment per synapse, grouped by the element it reaches -----------------
    def explain(pre, post, rec):
        pk = "src" if pre[0] == "src" else net.neurons[pre[1]]["kind"]
        src_kind = net.sources[pre[1]]["kind"] if pre[0] == "src" else None
        qk = net.neurons[post]["kind"]
        pn, qn = name_of(pre), net.neurons[post]["label"]
        if qk == "op":
            return "OP", f"input {pn} drives OP"
        if qk == "relay":
            return "NOT gates and delay neuron", f"input {pn} drives its copy"
        if qk == "no_event":
            return "NOT gates and delay neuron", ("OP of earlier input spikes excites it" if rec == "exc"
                                                  else "OP of input spikes now inhibits it")
        if qk in ("enter_or", "leave_or"):
            return "OR gates of the states (bypasses)", f"{pn} fired"
        if qk == "delay_line":
            return "Delay lines", f"{pn} -> {qn} (one step later)"
        if qk == "delay":
            return "NOT gates and delay neuron", "OP -> delay"
        if qk in ("not_input", "not_state"):
            if rec == "exc":
                return "NOT gates and delay neuron", f"OP excites {qn}"
            if qk == "not_input":
                return "NOT gates and delay neuron", f"input {pn} inhibits {qn}"
            return "NOT gates and delay neuron", f"latch {pn.split('.')[0]} inhibits {qn} (state active)"
        if qk == "term":
            if pk == "delay":
                return "AND gates (transition terms)", f"delay excites {qn}"
            if pk == "not_state":
                return "AND gates (transition terms)", f"{pn}: source state {pn[4:]} must be active"
            if pk == "not_input":
                return "AND gates (transition terms)", f"{pn}: {pn[4:].split(' #')[0]} must spike"
            if pk == "relay":
                return "AND gates (transition terms)", f"{pn}: {pn.split(' #')[0]} must not spike"
            if pk == "no_event":
                return "AND gates (transition terms)", f"{pn}: cancels the bypass when there are no input spikes now"
            if pk == "term":
                return "AND gates (transition terms)", (f"bypass 1: {pn} sets the source state" if rec == "exc"
                                                        else f"{pn} changed the source state 1 step earlier")
            if pk in ("enter_or", "leave_or"):
                return "AND gates (transition terms)", (f"bypass 2: {pn}" if rec == "exc" else f"{pn} 2 steps earlier")
            return "AND gates (transition terms)", f"input {pn}: {pn} must not spike"
        if qk == "latch":
            st = qn.split(".")[0]
            if src_kind == "init":
                return "Latches", f"{st} is active at start"
            if pk == "term":
                return "Latches", (f"{pn} sets {st}" if rec == "exc" else f"{pn} resets {st}")
            if pk == "latch":
                return "Latches", "internal connections of the latch"
            return "Latches", ""
        return "Output", f"{pn} -> output"

    sections = {}
    for pre, post, rec, delay, mult in net.conns:
        sec, com = explain(pre, post, rec)
        sections.setdefault(sec, []).append((pre, post, rec, delay, mult, com))

    lines = []
    w = lines.append
    slug = _ident(s["name"]).lower()
    c = net.counts()
    if simulation:
        w("#!/usr/bin/env python3")
        w(f'"""Simulation of the spiking FSM "{s["name"]}" with PyNN, stand-alone.')
    else:
        w(f'"""PyNN implementation of the spiking FSM "{s["name"]}".')
    w("")
    w(f"Generated by the Spiking FSM builder on {_dt.datetime.now():%Y-%m-%d %H:%M}.")
    w(f"{c['neurons']} neurons and {c['synapses']} synapses ({c['excitatory']} excitatory, "
      f"{c['inhibitory']} inhibitory).")
    w("")
    w("Every state is a latch of three neurons: index 0 is the set neuron (N1) and")
    w("indices 1 and 2 are the ring neurons (N2, N3), which keep spiking while the state")
    w("is active. Every synapse has weight 1 and a delay of one time step (longer delays go")
    w("through delay lines of relay neurons). Input spike times come from the settings")
    w("of the Simulate tab.")
    if bypass:
        w("")
        w("Parallel implementation: bypasses pass the transitions fired by the input spikes")
        w("of the 2 previous steps on to the next ones, so input spikes may come on")
        w("consecutive steps. Each term has up to three AND gates (bypass 1, bypass 2 and")
        w("latch), and at most one of them fires.")
    if simulation:
        w("")
        w("The script builds the network, runs it, compares the states (and Output) with")
        w("the reference automaton computed by the application, and saves the spikes and")
        w(f"a raster plot in ./results_{slug}/. It only needs PyNN with sPyNNaker")
        w("(matplotlib is optional, for the plot). Run it on the computer connected to the")
        w(f"SpiNNaker board:  python run_{slug}.py")
    w('"""')
    w("")
    w("import pyNN.spiNNaker as sim")
    w("")
    w(f"DT = {float(p['dt'])}                         # time step (ms)")
    wts = dict(DEFAULT_WEIGHTS)
    wts.update(p.get("weights") or {})
    w(f"W_EXC, W_INH = {wts['exc']}, {wts['inh']}       # synaptic weights (nA)")
    nparams = dict(DEFAULT_NEURON_PARAMS)
    nparams.update(p.get("neuron_params") or {})
    w("NEURON_PARAMS = {")
    for k, v in nparams.items():
        w(f"    {k!r}: {v},")
    w("}")
    last = max([e for e, _ in inp["events"]] + [INIT_STEP])
    w(f"DURATION = {last + 30}                     # simulated time steps")
    w("")
    w("sim.setup(timestep=DT, min_delay=DT)")
    w("cell = sim.IF_curr_exp(**NEURON_PARAMS)")
    w("")
    w("")
    w("def connect(pre, post, receptor, conns):")
    w('    """conns: list of (index in pre, index in post, delay in time steps)."""')
    w('    weight = W_EXC if receptor == "excitatory" else W_INH')
    w("    return sim.Projection(pre, post,")
    w("                          sim.FromListConnector([(i, j, weight, d * DT) for i, j, d in conns]),")
    w("                          synapse_type=sim.StaticSynapse(), receptor_type=receptor)")
    w("")
    w("")
    w("# ------------------------------------------------------------ spike sources")
    for i, src in enumerate(net.sources):
        if src["kind"] == "input":
            times = [float(x) for x in inp["trains"].get(src["input"], [])]
            w(f"{src_var[i]} = sim.Population(1, sim.SpikeSourceArray(spike_times=[[t * DT for t in {times!r}]]), label={src['label']!r})")
        else:
            w(f"{src_var[i]} = sim.Population(1, sim.SpikeSourceArray(spike_times=[[{float(INIT_STEP)} * DT]]), label='init')"
              "   # sets the states active at start")
    w("")
    w("# ------------------------------------------------------------ neurons")
    for var, size, com in pops:
        w(f"{var} = sim.Population({size}, cell, label={var!r})   # {com}")
    for sid, var in latch_of.items():
        st = st_by_id[sid]
        extra = ", active at start" if st["initial"] else ""
        w(f"{var} = sim.Population(3, cell, label={var!r})   # latch of state {st['name']}: [N1 set, N2 ring, N3 ring]{extra}")
    w("")
    order = ["OP", "Delay lines", "NOT gates and delay neuron", "AND gates (transition terms)", "OR gates of the states (bypasses)",
             "Latches", "Output"]
    for sec in order:
        if sec not in sections:
            continue
        w(f"# ------------------------------------------------------------ {sec.lower()}")
        latch_internal = {}
        for pre, post, rec, delay, mult, com in sections[sec]:
            (pv, pi), (qv, qi) = ref(pre), where[post]
            receptor = "excitatory" if rec == "exc" else "inhibitory"
            if com == "internal connections of the latch":
                latch_internal.setdefault((pv, receptor), []).append((pi, qi, delay))
                continue
            w(f"connect({pv}, {qv}, {receptor!r}, [({pi}, {qi}, {delay})])   # {com}")
        for (var, receptor), conns in latch_internal.items():
            what = "N1 -> N2 (set), N2 <-> N3 (ring)" if receptor == "excitatory" else "the ring inhibits N1 (no second set)"
            w(f"connect({var}, {var}, {receptor!r}, {conns!r})   # {what}")
        w("")
    w("# ------------------------------------------------------------ run")
    w("# neurons whose spikes are each state: the ring of its latch (N2, N3)")
    w("STATE_NEURONS = {")
    for st in s["states"]:
        w(f"    {st['name']!r}: [" + ", ".join(f"({where[n][0]}, {where[n][1]})" for n in net.states[st["id"]]) + "],")
    w("}")
    w("recorded = {id(pop): pop for ns in STATE_NEURONS.values() for pop, _ in ns}")
    w("for pop in recorded.values():")
    w('    pop.record("spikes")')
    if simulation and net.output is not None:
        w(f'{where[net.output][0]}.record("spikes")')
    w("sim.run(DURATION * DT)")
    w("")
    w('trains = {k: pop.get_data("spikes").segments[0].spiketrains for k, pop in recorded.items()}')
    w("active_steps = {}")
    w("for name, ns in STATE_NEURONS.items():")
    w("    active_steps[name] = sorted({int(round(float(t) / DT)) for pop, j in ns for t in trains[id(pop)][j].magnitude})")
    w('    print(f"{name}: active at steps {active_steps[name]}")')
    if simulation and net.output is not None:
        w(f'output_steps = sorted(int(round(float(t) / DT)) for t in {where[net.output][0]}.get_data("spikes").segments[0].spiketrains[0].magnitude)')
    w("sim.end()")
    if simulation:
        _comparison(w, s, net, inp, last + 30, slug)
    text = "\n".join(lines) + "\n"
    return dict(ok=True, filename=(f"run_{slug}.py" if simulation else f"pynn_{slug}.py"), text=text)

def _comparison(w, s, net, inp, duration, slug):
    """Simulate tab: compare with the reference automaton and save the results."""
    events = inp["events"]
    ref = run_reference(s, events)
    exp = _expected_state_steps(s, events, ref, duration, net.architecture)
    initial = next((st["name"] for st in s["states"] if st["initial"]), None)
    out = net.output is not None
    w("")
    w("# ------------------------------------------------------------ comparison with the reference")
    w("# Computed by the application from the reference automaton: the steps in which each")
    w("# state must spike ([first, last] ranges) and the steps of the expected Output spikes.")
    w(f"EVENTS = {[[e, sorted(ev)] for e, ev in events]!r}   # [step, inputs that spike]")
    w("EXPECTED = {")
    for st in s["states"]:
        w(f"    {st['name']!r}: {_ranges(exp[st['id']])!r},")
    w("}")
    if out:
        w(f"EXPECTED_OUTPUT = {[e + net.timing['output_latency'] for e in ref['detections']]!r}")
    w(f"INITIAL_STATE, READY_STEP = {initial!r}, {net.timing['ready_step']}")
    w("")
    w("# a simulator may shift every spike by a constant number of steps: measure it on the")
    w("# initial state, which must become active at READY_STEP")
    w("first = active_steps.get(INITIAL_STATE) or [READY_STEP]")
    w("offset = first[0] - READY_STEP")
    w("missing, extra = {}, {}")
    w("for name, rs in EXPECTED.items():")
    w("    want = {t + offset for a, b in rs for t in range(a, b + 1) if t + offset < DURATION}")
    w("    got = set(active_steps[name])")
    w("    missing[name], extra[name] = sorted(want - got), sorted(got - want)")
    if out:
        w("want = {t + offset for t in EXPECTED_OUTPUT if t + offset < DURATION}")
        w('missing["Output"], extra["Output"] = sorted(want - set(output_steps)), sorted(set(output_steps) - want)')
    w("n_miss = sum(len(v) for v in missing.values())")
    w("n_extra = sum(len(v) for v in extra.values())")
    w('print(f"Time offset: {offset} steps")')
    w("for name in missing:")
    w("    if missing[name] or extra[name]:")
    w('        print(f"{name}: missing spikes at steps {missing[name]}, extra spikes at steps {extra[name]}")')
    w('print("The network behaves exactly like the reference automaton." if n_miss + n_extra == 0')
    w('      else f"Differences with the reference: {n_miss} missing, {n_extra} extra spikes.")')
    w("")
    w("# ------------------------------------------------------------ save the results")
    w("import json")
    w("import os")
    w("")
    w(f'folder = "results_{slug}"')
    w("os.makedirs(folder, exist_ok=True)")
    w('with open(os.path.join(folder, "results.json"), "w") as f:')
    w("    json.dump(dict(dt=DT, duration=DURATION, offset=offset, events=EVENTS, active_steps=active_steps,")
    w("                   missing=missing, extra=extra" + (", output_steps=output_steps" if out else "") + "), f, indent=1)")
    w("try:")
    w("    import matplotlib")
    w('    matplotlib.use("Agg")')
    w("    import matplotlib.pyplot as plt")
    w("    inputs = {}")
    w("    for step, names in EVENTS:")
    w("        for n in names:")
    w("            inputs.setdefault(n, []).append(step)")
    w(f"    rows = [(n, inputs.get(n, []), None) for n in {s['inputs']!r}]")
    w("    rows += [(n, active_steps[n], EXPECTED[n]) for n in EXPECTED]")
    if out:
        w('    rows.append(("Output", output_steps, [[t, t] for t in EXPECTED_OUTPUT]))')
    w("    fig, ax = plt.subplots(figsize=(12, 0.45 * len(rows) + 1))")
    w("    for k, (name, steps, want) in enumerate(rows):")
    w("        for a, b in want or []:                  # expected activity, in grey")
    w("            ax.fill_between([(a + offset - .5) * DT, (b + offset + .5) * DT], k - .4, k + .4, color=\"#e2e8f0\")")
    w("        bad = set(extra.get(name, []))")
    w("        ax.vlines([t * DT for t in steps], k - .3, k + .3, colors=[\"#c0392b\" if t in bad else \"#334155\" for t in steps], lw=1)")
    w("        ax.vlines([t * DT for t in missing.get(name, [])], k - .3, k + .3, colors=\"#e67e22\", lw=2)")
    w("    ax.set_yticks(range(len(rows)), [r[0] for r in rows])")
    w("    ax.set_ylim(-.5, len(rows) - .5)")
    w("    ax.set_xlim(0, DURATION * DT)")
    w('    ax.set_xlabel("Time (ms)")')
    w('    ax.set_title("Grey: expected. Red: extra spikes. Orange: missing spikes.", fontsize=9, loc="left")')
    w("    fig.tight_layout()")
    w('    fig.savefig(os.path.join(folder, "raster.png"), dpi=130)')
    w("except ImportError:")
    w('    print("matplotlib is not installed: the plot is not saved")')
    w('print("Results saved in", folder)')


def _ranges(steps):
    out = []
    for t in sorted(steps):
        if out and t == out[-1][1] + 1:
            out[-1][1] = t
        else:
            out.append([t, t])
    return out

