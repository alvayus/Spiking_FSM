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
  N1 inhibits itself and is inhibited by N2 and N3 (no double set); Reset
  inhibits N2 and N3.  The state is "active" while the ring fires (every time
  step, alternating between N2 and N3).

Each transition option (a product term) becomes one AND gate whose inputs are
the required inputs, the forbidden inputs (negated) and the source state.

Timing (in time steps tau, relative to an input event at step e):

====================  =======================================================
step e                input spikes
e + 1                 OP (input-driven OR of all inputs)
e + 2                 delay neuron, NOT(inputs), NOT(states) - state sampled
                      from the latch ring spikes at e + 1
e + 3                 transition (product term) neurons
e + 4                 machine output; reset of "hold" source states;
                      ("discard" states are reset by OP at e + 3)
e + 5                 set of destination latches (N1)
e + 6                 destination states active (ring)
====================  =======================================================

Hence two consecutive input events must be at least ``MIN_EVENT_GAP = 5``
steps apart; the output appears ``OUTPUT_LATENCY = 4`` steps after the event
that completes a pattern, and a destination state is active
``STATE_LATENCY = 6`` steps after its triggering event.

All synaptic delays are chosen automatically so that every signal arrives at
the step required by the pipeline above.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .spec import ON, OFF, normalise

T_OP, T_GATE, T_TERM, T_OUT, T_SET = 1, 2, 3, 4, 5
T_RESET_DISCARD, T_RESET_HOLD = 3, 4
MIN_EVENT_GAP = 5
OUTPUT_LATENCY = T_OUT
STATE_LATENCY = T_SET + 1
INIT_STEP = 5           # step of the spike that sets the initial state(s)
FIRST_EVENT_STEP = 10   # earliest input event (as in the manuscript, 10 ms)


@dataclass
class Signal:
    """A spike source and the step (relative to an event) at which it fires."""
    ref: tuple  # ("src", index) or ("n", index)
    t: int


@dataclass
class Netlist:
    neurons: list = field(default_factory=list)   # dicts: label, kind
    sources: list = field(default_factory=list)   # dicts: label, kind, input
    conns: list = field(default_factory=list)     # (pre_ref, post, receptor, delay)
    latches: dict = field(default_factory=dict)   # state id -> (n1, n2, n3)
    terms: list = field(default_factory=list)     # (transition id, option idx, neuron)
    output: int | None = None
    op: int | None = None
    input_sources: dict = field(default_factory=dict)  # input name -> source index
    init_source: int | None = None

    # ---------------------------------------------------------------
    def add_neuron(self, label, kind):
        self.neurons.append(dict(label=label, kind=kind))
        return len(self.neurons) - 1

    def add_source(self, label, kind, input_name=None):
        self.sources.append(dict(label=label, kind=kind, input=input_name))
        return len(self.sources) - 1

    def connect(self, pre_ref, post, receptor, delay):
        if delay < 1:
            raise ValueError(f"Invalid delay ({delay}) towards {self.neurons[post]['label']}")
        self.conns.append((tuple(pre_ref), int(post), receptor, int(delay)))

    def connect_at(self, sig: Signal, post, receptor, arrive_at):
        self.connect(sig.ref, post, receptor, arrive_at - sig.t)

    # ---------------------------------------------------------------
    def counts(self) -> dict:
        exc = sum(1 for c in self.conns if c[2] == "exc")
        inh = sum(1 for c in self.conns if c[2] == "inh")
        return dict(neurons=len(self.neurons), synapses=len(self.conns),
                    excitatory=exc, inhibitory=inh, sources=len(self.sources),
                    max_delay=max((c[3] for c in self.conns), default=0))

    def to_dict(self) -> dict:
        return dict(neurons=self.neurons, sources=self.sources,
                    conns=[[list(c[0]), c[1], c[2], c[3]] for c in self.conns],
                    latches={k: list(v) for k, v in self.latches.items()},
                    terms=[list(t) for t in self.terms], output=self.output,
                    op=self.op, input_sources=self.input_sources,
                    init_source=self.init_source)


def build(spec: dict) -> Netlist:
    """Build the spiking network that implements ``spec``."""
    s = normalise(spec)
    net = Netlist()
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
        net.connect(("n", n1), n1, "inh", 1)   # self-inhibition
        net.connect(("n", n1), n2, "exc", 1)   # set -> ring
        net.connect(("n", n2), n3, "exc", 1)   # ring
        net.connect(("n", n3), n2, "exc", 1)   # ring
        net.connect(("n", n2), n1, "inh", 1)   # no double set
        net.connect(("n", n3), n1, "inh", 1)
        if st["initial"]:
            net.connect(("src", net.init_source), n1, "exc", 1)

    needs_term = bool(trans)
    # ---- shared delay neuron (fires at e+2) ------------------------------
    delay_sig = None
    if needs_term:
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

    # ---- shared NOT gates of source states ---------------------------------
    not_st = {}
    for sid in dict.fromkeys(t["from"] for t in trans):
        n = net.add_neuron(f"NOT {st_by_id[sid]['name']}", "not_state")
        net.connect_at(op, n, "exc", T_GATE)
        _, a, b = net.latches[sid]
        # latch ring spikes at e+1 (any step) arrive at e+2
        net.connect(("n", a), n, "inh", 1)
        net.connect(("n", b), n, "inh", 1)
        not_st[sid] = Signal(("n", n), T_GATE)

    # ---- product terms (AND gates) -----------------------------------------
    term_sig = {}
    for t in trans:
        for k, opt in enumerate(t["options"]):
            name = f"{st_by_id[t['from']]['name']}→{st_by_id[t['to']]['name']}"
            if len(t["options"]) > 1:
                name += f" ({k + 1})"
            n = net.add_neuron(name, "term")
            net.connect_at(delay_sig, n, "exc", T_TERM)
            net.connect_at(not_st[t["from"]], n, "inh", T_TERM)
            for i, v in opt.items():
                if v == ON:
                    net.connect_at(not_inp[i], n, "inh", T_TERM)
                elif v == OFF:
                    net.connect_at(inp[i], n, "inh", T_TERM)
            net.terms.append((t["id"], k, n))
            term_sig[(t["id"], k)] = Signal(("n", n), T_TERM)

    # ---- set destinations / reset sources -----------------------------------
    for t in trans:
        n1_dst = net.latches[t["to"]][0]
        src = st_by_id[t["from"]]
        _, a_src, b_src = net.latches[t["from"]]
        for k in range(len(t["options"])):
            sig = term_sig[(t["id"], k)]
            net.connect_at(sig, n1_dst, "exc", T_SET)
            if (src["on_mismatch"] == "hold" and not src["always_active"]
                    and t["to"] != t["from"]):
                net.connect_at(sig, a_src, "inh", T_RESET_HOLD)
                net.connect_at(sig, b_src, "inh", T_RESET_HOLD)

    for st in states:
        if st["on_mismatch"] == "discard" and not st["always_active"]:
            _, a, b = net.latches[st["id"]]
            net.connect_at(op, a, "inh", T_RESET_DISCARD)
            net.connect_at(op, b, "inh", T_RESET_DISCARD)

    # ---- machine output: OR of transitions entering accepting states --------
    acc_terms = [term_sig[(t["id"], k)] for t in trans if st_by_id[t["to"]]["accepting"]
                 for k in range(len(t["options"]))]
    if acc_terms:
        net.output = net.add_neuron("Output", "output")
        for sig in acc_terms:
            net.connect_at(sig, net.output, "exc", T_OUT)
    return net


def summary(spec: dict) -> dict:
    """Resource counts and timing figures (Simulate tab and scaling study)."""
    s = normalise(spec)
    net = build(s)
    c = net.counts()
    c.update(states=len(s["states"]), transitions=len(s["transitions"]),
             options=sum(len(t["options"]) for t in s["transitions"]),
             literals=sum(len(o) for t in s["transitions"] for o in t["options"]),
             min_event_gap=MIN_EVENT_GAP, output_latency=OUTPUT_LATENCY,
             state_latency=STATE_LATENCY)
    return c
