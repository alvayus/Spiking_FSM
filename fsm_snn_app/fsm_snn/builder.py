"""Specification -> spiking network (netlist) using the spiking gates of Fig. 1.

Building blocks (Fig. 1 of the manuscript):

* NOT gate: one neuron excited by the operation signal OP and inhibited by the
  input.  Fires iff OP is present and the input is not.
* AND gate: a delay neuron (excited by OP), one NOT neuron per input, and an
  output neuron excited by the delay neuron and inhibited by every NOT neuron.
  Here the delay neuron and the NOT neurons are shared by all transitions, so
  each product term costs a single output neuron.  A literal that must *not*
  spike inhibits the term neuron directly (it is its own negation).
* Input-driven OR gate (introduced in [3]): one neuron excited by every input.
  It generates OP (union of all input spikes) and the machine output.
* Latch: three neurons.  Set -> N1 -> N2 <-> N3 (ring that keeps the state);
  Reset inhibits N2 and N3.  The state is "active" while the ring fires (every
  time step, alternating between N2 and N3).  Setting a state that is already
  active is harmless: both ring neurons may then fire on every step until the
  next reset.

Each transition option (a product term) becomes one AND gate whose inputs are
the required inputs, the forbidden inputs (negated) and the source state.

Two implementations (``spec["architecture"]``):

* ``pipelined``: the paradigm of the manuscript.  A transition is evaluated from
  the state stored in the latches, so an input event must wait until the
  previous transition has been written: events at least ``3`` steps apart.
* ``parallel``: the same network plus bypasses, as in the pattern detector of
  the manuscript.  The transitions fired by the one or two previous events are
  forwarded to the transitions of the next event, which therefore do not wait
  for the latches: events may come on consecutive steps.

Timing (in time steps tau, relative to an input event at step e):

====================  =======================================================
step e                input spikes
e + 1                 OP (input-driven OR of all inputs)
e + 2                 delay neuron, NOT(inputs), NOT(states) - state sampled
                      from the latch spikes (N1, N2, N3) at e + 1
e + 3                 transition (product term) neurons
e + 4                 machine output; reset of the source states (N2, N3);
                      set of the destination states (N1)
e + 5                 destination states active (ring)
====================  =======================================================

The latches hold the state after an event from step e + 4 on, which an event
at e + 3 samples: hence the minimum spacing of 3 steps of ``pipelined``.

Bypasses of ``parallel``, for an event at e' = e + g after an event at e:

* g = 2: the transitions of e (step e + 3) reach NOT(state) of e' (step
  e' + 2 = e + 4): entering a state inhibits its NOT gate, leaving it excites
  the NOT gate, overriding the latch (which still shows the state before e).
* g = 1: the transitions of e reach the transitions of e' directly (step
  e' + 3 = e + 4): entering the source state excites them, leaving it inhibits
  them.  A "no event" neuron (fires at e + 3 when an event at e is not
  followed by one at e + 1) cancels this excitation when there is no event at
  e', so it never fires a transition by itself.

Signals of more recent events always outweigh older ones (the weights below are
chosen for it), so the network follows the machine whatever the spacing.  All
weights are whole multiples of the excitatory weight, and every neuron is
designed so that the excitation minus the inhibition it receives is either at
least one excitatory weight (fires) or at most zero (silent): the behaviour does
not depend on the exact threshold of the neuron model.

All synaptic delays are chosen automatically so that every signal arrives at
the step required by the pipeline above.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .spec import ON, OFF, normalise

T_OP, T_GATE, T_TERM, T_OUT, T_SET, T_RESET = 1, 2, 3, 4, 4, 4
OUTPUT_LATENCY = T_OUT
STATE_LATENCY = T_SET + 1
INIT_STEP = 0           # step of the spike that sets the initial state(s): active from step 2
FIRST_EVENT_STEP = 10   # earliest input event (as in the manuscript, 10 ms)
INH_RATIO = 2           # inhibitory / excitatory weight (DEFAULT_WEIGHTS of pynn_runner)

ARCHITECTURES = {
    "pipelined": dict(label="Pipelined", min_event_gap=3,
                      desc="The paradigm of the manuscript: each transition reads the state "
                           "stored in the latches. Input events at least 3 steps apart."),
    "parallel": dict(label="Parallel (bypasses)", min_event_gap=1,
                     desc="Bypasses forward the transitions of the previous events, as in the "
                          "pattern detector of the manuscript. Input events at any spacing."),
}
DEFAULT_ARCHITECTURE = "parallel"


def architecture(spec: dict) -> str:
    a = (spec or {}).get("architecture")
    return a if a in ARCHITECTURES else DEFAULT_ARCHITECTURE


def min_event_gap(spec: dict) -> int:
    return ARCHITECTURES[architecture(spec)]["min_event_gap"]


@dataclass
class Signal:
    """A spike source and the step (relative to an event) at which it fires."""
    ref: tuple  # ("src", index) or ("n", index)
    t: int


@dataclass
class Netlist:
    neurons: list = field(default_factory=list)   # dicts: label, kind
    sources: list = field(default_factory=list)   # dicts: label, kind, input
    conns: list = field(default_factory=list)     # (pre_ref, post, receptor, delay, weight)
    latches: dict = field(default_factory=dict)   # state id -> (n1, n2, n3)
    terms: list = field(default_factory=list)     # (transition id, option idx, neuron)
    output: int | None = None
    op: int | None = None
    input_sources: dict = field(default_factory=dict)  # input name -> source index
    init_source: int | None = None
    architecture: str = DEFAULT_ARCHITECTURE

    # ---------------------------------------------------------------
    def add_neuron(self, label, kind):
        self.neurons.append(dict(label=label, kind=kind))
        return len(self.neurons) - 1

    def add_source(self, label, kind, input_name=None):
        self.sources.append(dict(label=label, kind=kind, input=input_name))
        return len(self.sources) - 1

    def connect(self, pre_ref, post, receptor, delay, weight=1):
        """weight: multiple of the excitatory or inhibitory weight (a whole number)."""
        if delay < 1:
            raise ValueError(f"Invalid delay ({delay}) towards {self.neurons[post]['label']}")
        self.conns.append((tuple(pre_ref), int(post), receptor, int(delay), int(weight)))

    def connect_at(self, sig: Signal, post, receptor, arrive_at, weight=1):
        self.connect(sig.ref, post, receptor, arrive_at - sig.t, weight)

    # ---------------------------------------------------------------
    def counts(self) -> dict:
        exc = sum(1 for c in self.conns if c[2] == "exc")
        inh = sum(1 for c in self.conns if c[2] == "inh")
        return dict(neurons=len(self.neurons), synapses=len(self.conns),
                    excitatory=exc, inhibitory=inh, sources=len(self.sources),
                    max_delay=max((c[3] for c in self.conns), default=0))

    def to_dict(self) -> dict:
        return dict(neurons=self.neurons, sources=self.sources,
                    conns=[[list(c[0]), c[1], c[2], c[3], c[4]] for c in self.conns],
                    latches={k: list(v) for k, v in self.latches.items()},
                    terms=[list(t) for t in self.terms], output=self.output,
                    op=self.op, input_sources=self.input_sources,
                    init_source=self.init_source, architecture=self.architecture)


def _inh(units: int) -> int:
    """Inhibitory weight multiple that cancels at least ``units`` excitatory weights."""
    return max(1, math.ceil(units / INH_RATIO))


def build(spec: dict) -> Netlist:
    """Build the spiking network that implements ``spec``."""
    s = normalise(spec)
    arch = architecture(s)
    parallel = arch == "parallel"
    net = Netlist(architecture=arch)
    states = s["states"]
    st_by_id = {st["id"]: st for st in states}
    trans = [t for t in s["transitions"] if t["from"] in st_by_id and t["to"] in st_by_id]

    # ---- sources -------------------------------------------------------
    for name in s["inputs"]:
        net.input_sources[name] = net.add_source(name, "input", name)
    net.init_source = net.add_source("init", "init")
    inp = {name: Signal(("src", idx), 0) for name, idx in net.input_sources.items()}

    # ---- OP: input-driven OR of every input (fires at e+1) -------------
    net.op = net.add_neuron("OP", "op")
    for name in s["inputs"]:
        net.connect_at(inp[name], net.op, "exc", T_OP)
    op = Signal(("n", net.op), T_OP)

    # ---- latches (one per state) ----------------------------------------
    for st in states:
        n1 = net.add_neuron(f"{st['name']}.set", "latch")
        n2 = net.add_neuron(f"{st['name']}.a", "latch")
        n3 = net.add_neuron(f"{st['name']}.b", "latch")
        net.latches[st["id"]] = (n1, n2, n3)
        net.connect(("n", n1), n2, "exc", 1)   # set -> ring
        net.connect(("n", n2), n3, "exc", 1)   # ring
        net.connect(("n", n3), n2, "exc", 1)   # ring
        if st["initial"]:
            net.connect(("src", net.init_source), n1, "exc", 1)

    if not trans:
        return net

    # ---- term neurons first (the bypasses connect them to each other) -------
    terms = []           # (transition, option index, option, neuron)
    for t in trans:
        for k, opt in enumerate(t["options"]):
            name = f"{st_by_id[t['from']]['name']}→{st_by_id[t['to']]['name']}"
            if len(t["options"]) > 1:
                name += f" ({k + 1})"
            terms.append((t, k, opt, net.add_neuron(name, "term")))
    entering = {sid: [n for t, _, _, n in terms if t["to"] == sid] for sid in st_by_id}
    leaving = {sid: [n for t, _, _, n in terms if t["from"] == sid and t["to"] != sid]
               for sid in st_by_id}

    # ---- shared delay neuron (fires at e+2) ------------------------------
    d = net.add_neuron("Delay", "delay")
    net.connect_at(op, d, "exc", T_GATE)
    delay_sig = Signal(("n", d), T_GATE)

    # ---- shared NOT gates of inputs used positively -----------------------
    not_inp = {}
    used_on = sorted({i for t in trans for o in t["options"] for i, v in o.items() if v == ON},
                     key=s["inputs"].index)
    for name in used_on:
        n = net.add_neuron(f"NOT {name}", "not_input")
        net.connect_at(op, n, "exc", T_GATE)
        net.connect_at(inp[name], n, "inh", T_GATE)
        not_inp[name] = Signal(("n", n), T_GATE)

    # ---- "no event" neuron of the bypasses: an event at e not followed by one at e+1
    no_next = None
    if parallel:
        n = net.add_neuron("No event at t+1", "no_event")
        net.connect(("n", net.op), n, "exc", 2)      # OP of the event at e (e+1) -> e+3
        net.connect(("n", net.op), n, "inh", 1)      # OP of an event at e+1 (e+2) -> e+3
        no_next = n

    # ---- shared NOT gates of source states ---------------------------------
    not_st = {}
    for sid in dict.fromkeys(t["from"] for t in trans):
        n = net.add_neuron(f"NOT {st_by_id[sid]['name']}", "not_state")
        net.connect_at(op, n, "exc", T_GATE)
        # latch spikes (set or ring) at e+1 arrive at e+2: up to three of them
        for x in net.latches[sid]:
            net.connect(("n", x), n, "inh", 1)
        if parallel:
            # bypass, event two steps earlier: its transitions (at e+1) arrive at e+2.
            # Leaving outweighs the latch (3 spikes); entering outweighs every leaving.
            w_leave = 3 * INH_RATIO
            for x in leaving[sid]:
                net.connect(("n", x), n, "exc", 1, w_leave)
            w_enter = _inh(1 + w_leave * len(leaving[sid]))
            for x in entering[sid]:
                net.connect(("n", x), n, "inh", 1, w_enter)
        not_st[sid] = Signal(("n", n), T_GATE)

    # ---- product terms (AND gates) -----------------------------------------
    for t, k, opt, n in terms:
        sid = t["from"]
        net.connect_at(delay_sig, n, "exc", T_TERM)
        net.connect_at(not_st[sid], n, "inh", T_TERM)
        w_veto = 1
        if parallel:
            # bypass, event one step earlier: its transitions (at e+2) arrive at e+3.
            # Entering the source state outweighs NOT(state) and every leaving one.
            nl, ne = len(leaving[sid]), len(entering[sid])
            w_enter = 1 + INH_RATIO * (1 + nl)
            for x in entering[sid]:
                net.connect(("n", x), n, "exc", 1, w_enter)
            for x in leaving[sid]:
                net.connect(("n", x), n, "inh", 1, 1)
            if ne:
                net.connect(("n", no_next), n, "inh", 1, _inh(w_enter * ne))
            w_veto = _inh(1 + w_enter * ne)       # a condition that fails always wins
        for i, v in opt.items():
            if v == ON:
                net.connect_at(not_inp[i], n, "inh", T_TERM, w_veto)
            elif v == OFF:
                net.connect_at(inp[i], n, "inh", T_TERM, w_veto)
        net.terms.append((t["id"], k, n))

    # ---- set destinations / reset sources (both at e+4) ----------------------
    for t, k, opt, n in terms:
        sig = Signal(("n", n), T_TERM)
        net.connect_at(sig, net.latches[t["to"]][0], "exc", T_SET)
        if t["to"] != t["from"]:
            _, a_src, b_src = net.latches[t["from"]]
            net.connect_at(sig, a_src, "inh", T_RESET)
            net.connect_at(sig, b_src, "inh", T_RESET)

    # ---- machine output: OR of transitions entering accepting states --------
    acc = [n for t, _, _, n in terms if st_by_id[t["to"]]["accepting"]]
    if acc:
        net.output = net.add_neuron("Output", "output")
        for n in acc:
            net.connect_at(Signal(("n", n), T_TERM), net.output, "exc", T_OUT)
    return net


def summary(spec: dict) -> dict:
    """Resource counts and timing figures (Simulate tab)."""
    s = normalise(spec)
    net = build(s)
    c = net.counts()
    c.update(states=len(s["states"]), transitions=len(s["transitions"]),
             options=sum(len(t["options"]) for t in s["transitions"]),
             literals=sum(len(o) for t in s["transitions"] for o in t["options"]),
             architecture=architecture(s), min_event_gap=min_event_gap(s),
             output_latency=OUTPUT_LATENCY, state_latency=STATE_LATENCY)
    return c
