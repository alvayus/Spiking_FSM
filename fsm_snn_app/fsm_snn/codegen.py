"""Readable PyNN code of the network that implements a machine.

Unlike the stand-in script exported from the Simulate tab (which loads the netlist
from JSON), this code spells the network out: one population per gate or latch,
named after its role, and one projection per synapse with a comment that says
what it does. It is generated from the same netlist (``builder.build``), so it is
the exact network that the application simulates.
"""

from __future__ import annotations

import datetime as _dt
import re

from .builder import INIT_STEP, build
from .pynn_runner import DEFAULT_NEURON_PARAMS, DEFAULT_WEIGHTS
from .spec import normalise, option_to_text, validate


def _ident(text: str) -> str:
    t = re.sub(r"[^0-9A-Za-z_]+", "_", text).strip("_") or "x"
    return ("n_" + t) if t[0].isdigit() else t


def generate_code(spec: dict, params: dict | None = None) -> dict:
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
    term_info = {n: (tid, k) for tid, k, n in net.terms}
    for i, n in enumerate(net.neurons):
        if i in where:
            continue
        kind, label = n["kind"], n["label"]
        if kind == "op":
            var, com = unique("OP"), "input-driven OR of every input (operation signal)"
        elif kind == "delay":
            var, com = unique("delay"), "delay neuron shared by the AND gates"
        elif kind == "not_input":
            name = label[4:]
            var, com = unique("NOT_" + name), f"NOT gate of input {name}"
        elif kind == "not_state":
            name = label[4:]
            var, com = unique("NOT_" + name), f"NOT gate of state {name}"
        elif kind == "term":
            tid, k = term_info[i]
            t = trans[tid]
            a, b = st_by_id[t["from"]]["name"], st_by_id[t["to"]]["name"]
            var = unique(f"AND_{a}_{b}" + (f"_{k + 1}" if len(t["options"]) > 1 else ""))
            com = f"AND gate of {a} -> {b}: {option_to_text(t['options'][k])}"
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
        if qk == "delay":
            return "NOT gates and delay neuron", "OP -> delay"
        if qk in ("not_input", "not_state"):
            if rec == "exc":
                return "NOT gates and delay neuron", f"OP excites {qn}"
            if qk == "not_input":
                return "NOT gates and delay neuron", f"input {pn} inhibits {qn}"
            return "NOT gates and delay neuron", f"ring of latch {pn.split('.')[0]} inhibits {qn} (state active)"
        if qk == "term":
            if pk == "delay":
                return "AND gates (transition terms)", f"delay excites {qn}"
            if pk == "not_state":
                return "AND gates (transition terms)", f"{pn}: source state {pn[4:]} must be active"
            if pk == "not_input":
                return "AND gates (transition terms)", f"{pn}: {pn[4:]} must spike"
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
    for pre, post, rec, delay in net.conns:
        sec, com = explain(pre, post, rec)
        sections.setdefault(sec, []).append((pre, post, rec, delay, com))

    lines = []
    w = lines.append
    slug = _ident(s["name"]).lower()
    c = net.counts()
    w(f'"""PyNN implementation of the spiking FSM "{s["name"]}".')
    w("")
    w(f"Generated by the Spiking FSM builder on {_dt.datetime.now():%Y-%m-%d %H:%M}.")
    w(f"{c['neurons']} neurons and {c['synapses']} synapses ({c['excitatory']} excitatory, "
      f"{c['inhibitory']} inhibitory).")
    w("")
    w("Every state is a latch of three neurons: index 0 is the set neuron (N1) and")
    w("indices 1 and 2 are the ring neurons (N2, N3), which keep spiking while the state")
    w("is active. Input spike times come from the settings of the Simulate tab.")
    w('"""')
    w("")
    w("import pyNN.spiNNaker as sim      # or: import pyNN.nest as sim / import pyNN.brian2 as sim")
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
    order = ["OP", "NOT gates and delay neuron", "AND gates (transition terms)", "Latches", "Output"]
    for sec in order:
        if sec not in sections:
            continue
        w(f"# ------------------------------------------------------------ {sec.lower()}")
        latch_internal = {}
        for pre, post, rec, delay, com in sections[sec]:
            (pv, pi), (qv, qi) = ref(pre), where[post]
            receptor = "excitatory" if rec == "exc" else "inhibitory"
            if com == "internal connections of the latch":
                latch_internal.setdefault((pv, receptor), []).append((pi, qi, delay))
                continue
            w(f"connect({pv}, {qv}, {receptor!r}, [({pi}, {qi}, {delay})])   # {com}")
        for (var, receptor), conns in latch_internal.items():
            what = ("N1 -> N2 (set), N2 <-> N3 (ring)" if receptor == "excitatory"
                    else "N1 inhibits itself, the ring inhibits N1")
            w(f"connect({var}, {var}, {receptor!r}, {conns!r})   # {what}")
        w("")
    w("# ------------------------------------------------------------ run")
    w("latches = {" + ", ".join(f"{st_by_id[sid]['name']!r}: {var}" for sid, var in latch_of.items()) + "}")
    w("for pop in latches.values():")
    w('    pop.record("spikes")')
    w("sim.run(DURATION * DT)")
    w("")
    w("active_steps = {}")
    w("for name, pop in latches.items():")
    w('    trains = pop.get_data("spikes").segments[0].spiketrains')
    w("    ring = trains[1:]                    # the state is active while N2 / N3 spike")
    w("    active_steps[name] = sorted({int(round(float(t) / DT)) for tr in ring for t in tr.magnitude})")
    w('    print(f"{name}: active at steps {active_steps[name]}")')
    w("sim.end()")
    text = "\n".join(lines) + "\n"
    return dict(ok=True, filename=f"pynn_{slug}.py", text=text)
