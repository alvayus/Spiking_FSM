"""Specification -> spiking network (netlist) using the spiking gates of Fig. 1.

Boolean paradigm: every synapse has the same weight (1) and a neuron fires when
an input spike reaches it, unless an inhibitory spike reaches it at the same
time.  The networks are built so that whenever a neuron is inhibited, it never
receives more excitatory spikes than inhibitory ones: the behaviour is the same
whether inhibition cancels excitation spike for spike (LIF neurons with equal
weights, as on SpiNNaker) or vetoes it outright.

Building blocks (Fig. 1 of the manuscript):

* NOT gate: one neuron excited by a driving signal and inhibited by the input.
* AND gate: a delay neuron, one NOT neuron per input and an output neuron
  excited by the delay neuron and inhibited by every NOT neuron.  A literal that
  must *not* spike inhibits the output neuron directly.
* Input-driven OR gate: one neuron excited by every input (OP: an input event).
* Latch: Set -> N1 -> ring N2 <-> N3, Reset inhibits N2 and N3.

Two implementations (``spec["architecture"]``):

* ``pipelined``: the paradigm of the manuscript, with one latch per state.  A
  transition reads the state stored in the latches, so an input event must wait
  until the previous transition has been written: events at least 3 steps
  apart.

  ====================  =====================================================
  step e                input spikes
  e + 1                 OP
  e + 2                 delay neuron, NOT(inputs), NOT(states) - latch spikes
                        (N1, N2, N3) of e + 1
  e + 3                 transition (product term) neurons
  e + 4                 output; reset of the source (N2, N3); set of the
                        destination (N1)
  e + 5                 destination active (ring)
  ====================  =====================================================

* ``parallel``: a synchronous machine.  The conditions of the transitions are
  computed from the inputs on their own (they do not depend on the state), and
  the state is recomputed on every time step by one neuron per state ("hold":
  the state stays) and one per transition ("enter": the transition fires), which
  excite each other one step later.  A clock (a two-neuron ring, like a latch)
  drives the gates that say "this transition does not fire" on every step, also
  on steps without input spikes.  Every step processes the input spikes of one
  step, so input spikes may come on consecutive steps.

  ====================  =====================================================
  step e                input spikes
  e + 1                 OP
  e + 2                 delay neuron, NOT(inputs)
  e + 3                 condition terms (one AND gate per term, no state)
  e + 4                 "leaving" OR of each state; "no transition" NOT gates
                        (driven by the clock)
  e + 5                 "must leave" NOT gates of each state
  e + 6                 hold and enter neurons: the state after the event
  e + 7                 output
  ====================  =====================================================

  The hold neuron of a state and the enter neurons of the transitions into it
  are the state: it is active while any of them fires.  A neuron excited by k
  of them is inhibited by k copies of its "does not fire" gate, so that it is
  never excited more than inhibited.

All synaptic delays are chosen automatically so that every signal arrives at
the step required above.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .spec import ON, OFF, normalise

# pipelined
T_OP, T_GATE, T_TERM, T_OUT, T_SET, T_RESET = 1, 2, 3, 4, 4, 4
# parallel (synchronous)
P_OP, P_GATE, P_TERM, P_COND, P_LEAVE, P_STATE, P_OUT = 1, 2, 3, 4, 5, 6, 7
P_START = 4             # first step of the hold neurons of the initial states
INIT_STEP = 0           # step of the initialisation spike
FIRST_EVENT_STEP = 10   # earliest input event of the random trains (as in the manuscript, 10 ms)

ARCHITECTURES = {
    "pipelined": dict(label="Pipelined", min_event_gap=3,
                      desc="Each transition reads the state from the latches. "
                           "Input spikes at least 3 steps apart."),
    "parallel": dict(label="Parallel (synchronous)", min_event_gap=1,
                     desc="The state is recomputed on every time step from conditions computed "
                          "apart. Input spikes at any spacing."),
}
DEFAULT_ARCHITECTURE = "parallel"
# timing of each implementation (steps after an input event)
TIMING = {
    "pipelined": dict(state_latency=T_SET + 1, output_latency=T_OUT, ready_step=INIT_STEP + 2),
    "parallel": dict(state_latency=P_STATE, output_latency=P_OUT, ready_step=P_START),
}


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
    conns: list = field(default_factory=list)     # (pre_ref, post, receptor, delay, weight = 1)
    latches: dict = field(default_factory=dict)   # pipelined: state id -> (n1, n2, n3)
    states: dict = field(default_factory=dict)    # state id -> neurons whose spikes are the state
    terms: list = field(default_factory=list)     # (transition id, option idx, neuron)
    enters: list = field(default_factory=list)    # parallel: (transition id, neuron)
    output: int | None = None
    op: int | None = None
    input_sources: dict = field(default_factory=dict)  # input name -> source index
    init_source: int | None = None
    architecture: str = DEFAULT_ARCHITECTURE

    @property
    def timing(self) -> dict:
        return TIMING[self.architecture]

    # ---------------------------------------------------------------
    def add_neuron(self, label, kind, **extra):
        self.neurons.append(dict(label=label, kind=kind, **extra))
        return len(self.neurons) - 1

    def add_source(self, label, kind, input_name=None):
        self.sources.append(dict(label=label, kind=kind, input=input_name))
        return len(self.sources) - 1

    def connect(self, pre_ref, post, receptor, delay):
        if delay < 1:
            raise ValueError(f"Invalid delay ({delay}) towards {self.neurons[post]['label']}")
        self.conns.append((tuple(pre_ref), int(post), receptor, int(delay), 1))

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
                    conns=[[list(c[0]), c[1], c[2], c[3], c[4]] for c in self.conns],
                    latches={k: list(v) for k, v in self.latches.items()},
                    states={k: list(v) for k, v in self.states.items()},
                    terms=[list(t) for t in self.terms], enters=[list(t) for t in self.enters],
                    output=self.output, op=self.op, input_sources=self.input_sources,
                    init_source=self.init_source, architecture=self.architecture,
                    init_step=INIT_STEP)


def build(spec: dict) -> Netlist:
    """Build the spiking network that implements ``spec``."""
    s = normalise(spec)
    arch = architecture(s)
    net = Netlist(architecture=arch)
    for name in s["inputs"]:
        net.input_sources[name] = net.add_source(name, "input", name)
    net.init_source = net.add_source("init", "init")
    net.op = net.add_neuron("OP", "op")
    for name in s["inputs"]:
        net.connect(("src", net.input_sources[name]), net.op, "exc", T_OP)
    (_build_parallel if arch == "parallel" else _build_pipelined)(s, net)
    return net


def _input_gates(s, net, trans, t_gate):
    """Delay neuron and NOT gates of the inputs used positively (fire at e + t_gate)."""
    op = Signal(("n", net.op), T_OP)
    inp = {name: Signal(("src", idx), 0) for name, idx in net.input_sources.items()}
    d = net.add_neuron("Delay", "delay")
    net.connect_at(op, d, "exc", t_gate)
    not_inp = {}
    used_on = sorted({i for t in trans for o in t["options"] for i, v in o.items() if v == ON},
                     key=s["inputs"].index)
    for name in used_on:
        n = net.add_neuron(f"NOT {name}", "not_input")
        net.connect_at(op, n, "exc", t_gate)
        net.connect_at(inp[name], n, "inh", t_gate)
        not_inp[name] = Signal(("n", n), t_gate)
    return Signal(("n", d), t_gate), not_inp, inp


def _term_name(st_by_id, t, k):
    name = f"{st_by_id[t['from']]['name']}→{st_by_id[t['to']]['name']}"
    return name + (f" ({k + 1})" if len(t["options"]) > 1 else "")


# ------------------------------------------------------------------------- pipelined
def _build_pipelined(s, net):
    states = s["states"]
    st_by_id = {st["id"]: st for st in states}
    trans = [t for t in s["transitions"] if t["from"] in st_by_id and t["to"] in st_by_id]
    for st in states:
        n1 = net.add_neuron(f"{st['name']}.set", "latch")
        n2 = net.add_neuron(f"{st['name']}.a", "latch")
        n3 = net.add_neuron(f"{st['name']}.b", "latch")
        net.latches[st["id"]] = (n1, n2, n3)
        net.states[st["id"]] = (n2, n3)
        net.connect(("n", n1), n2, "exc", 1)   # set -> ring
        net.connect(("n", n2), n3, "exc", 1)   # ring
        net.connect(("n", n3), n2, "exc", 1)   # ring
        if st["initial"]:
            net.connect(("src", net.init_source), n1, "exc", 1)
    if not trans:
        return
    delay_sig, not_inp, inp = _input_gates(s, net, trans, T_GATE)
    op = Signal(("n", net.op), T_OP)
    not_st = {}
    for sid in dict.fromkeys(t["from"] for t in trans):
        n = net.add_neuron(f"NOT {st_by_id[sid]['name']}", "not_state")
        net.connect_at(op, n, "exc", T_GATE)
        for x in net.latches[sid]:             # latch spikes (N1, N2, N3) at e+1 arrive at e+2
            net.connect(("n", x), n, "inh", 1)
        not_st[sid] = Signal(("n", n), T_GATE)
    terms = []
    for t in trans:
        for k, opt in enumerate(t["options"]):
            n = net.add_neuron(_term_name(st_by_id, t, k), "term")
            net.connect_at(delay_sig, n, "exc", T_TERM)
            net.connect_at(not_st[t["from"]], n, "inh", T_TERM)
            for i, v in opt.items():
                if v == ON:
                    net.connect_at(not_inp[i], n, "inh", T_TERM)
                elif v == OFF:
                    net.connect_at(inp[i], n, "inh", T_TERM)
            net.terms.append((t["id"], k, n))
            terms.append((t, n))
    for t, n in terms:
        sig = Signal(("n", n), T_TERM)
        net.connect_at(sig, net.latches[t["to"]][0], "exc", T_SET)
        if t["to"] != t["from"]:
            _, a_src, b_src = net.latches[t["from"]]
            net.connect_at(sig, a_src, "inh", T_RESET)
            net.connect_at(sig, b_src, "inh", T_RESET)
    acc = [n for t, n in terms if st_by_id[t["to"]]["accepting"]]
    if acc:
        net.output = net.add_neuron("Output", "output")
        for n in acc:
            net.connect_at(Signal(("n", n), T_TERM), net.output, "exc", T_OUT)


# ------------------------------------------------------------------------- parallel
def _build_parallel(s, net):
    states = s["states"]
    st_by_id = {st["id"]: st for st in states}
    trans = [t for t in s["transitions"] if t["from"] in st_by_id and t["to"] in st_by_id]
    name = lambda sid: st_by_id[sid]["name"]

    # ---- clock: a ring that fires on every step from step 1
    c1 = net.add_neuron("Clock.a", "clock")
    c2 = net.add_neuron("Clock.b", "clock")
    net.connect(("src", net.init_source), c1, "exc", 1)
    net.connect(("n", c1), c2, "exc", 1)
    net.connect(("n", c2), c1, "exc", 1)

    # ---- condition terms: event and term holds (no state), at e+3
    opts = {}                                   # transition id -> [term neurons]
    if trans:
        delay_sig, not_inp, inp = _input_gates(s, net, trans, P_GATE)
        for t in trans:
            opts[t["id"]] = []
            for k, opt in enumerate(t["options"]):
                n = net.add_neuron(_term_name(st_by_id, t, k), "term")
                net.connect_at(delay_sig, n, "exc", P_TERM)
                for i, v in opt.items():
                    if v == ON:
                        net.connect_at(not_inp[i], n, "inh", P_TERM)
                    elif v == OFF:
                        net.connect_at(inp[i], n, "inh", P_TERM)
                net.terms.append((t["id"], k, n))
                opts[t["id"]].append(n)

    # ---- state neurons: hold of each state, enter of each transition between two states
    hold = {st["id"]: net.add_neuron(f"{st['name']} (hold)", "hold", sid=st["id"]) for st in states}
    enter = {}
    for t in trans:
        if t["to"] != t["from"]:
            enter[t["id"]] = net.add_neuron(f"{name(t['from'])}→{name(t['to'])}", "enter", tid=t["id"])
            net.enters.append((t["id"], enter[t["id"]]))
    into = {sid: [hold[sid]] + [enter[t["id"]] for t in trans if t["id"] in enter and t["to"] == sid]
            for sid in st_by_id}
    for sid, ns in into.items():
        net.states[sid] = tuple(ns)
    for st in states:
        if st["initial"]:
            net.connect(("src", net.init_source), hold[st["id"]], "exc", P_START - INIT_STEP)

    # every neuron of a state excites, one step later, the hold of the state and the
    # enter neurons of the transitions that leave it
    for sid, ns in into.items():
        targets = [hold[sid]] + [enter[t["id"]] for t in trans if t["id"] in enter and t["from"] == sid]
        for x in ns:
            for y in targets:
                net.connect(("n", x), y, "exc", 1)

    # ---- "transition does not fire": clock and no term of the transition (at e+4),
    #      one copy per neuron of the source state (it inhibits the enter neuron)
    for t in trans:
        if t["id"] not in enter:
            continue
        k = len(into[t["from"]])
        for c in range(k):
            n = net.add_neuron(f"NOT {name(t['from'])}→{name(t['to'])}" + (f" #{c + 1}" if k > 1 else ""),
                               "no_trans", tid=t["id"])
            net.connect(("n", c1), n, "exc", 1)
            net.connect(("n", c2), n, "exc", 1)
            for x in opts[t["id"]]:
                net.connect(("n", x), n, "inh", P_COND - P_TERM)
            net.connect(("n", n), enter[t["id"]], "inh", P_STATE - P_COND)

    # ---- "must leave": some term leaving the state and no self-loop term (at e+5),
    #      one copy per neuron of the state (it inhibits the hold neuron)
    for st in states:
        sid = st["id"]
        leave = [x for t in trans if t["from"] == sid and t["to"] != sid for x in opts[t["id"]]]
        if not leave:
            continue
        self_terms = [x for t in trans if t["from"] == sid and t["to"] == sid for x in opts[t["id"]]]
        l_or = net.add_neuron(f"{st['name']} leaves", "leave", sid=sid)
        for x in leave:
            net.connect(("n", x), l_or, "exc", P_COND - P_TERM)
        k = len(into[sid])
        for c in range(k):
            n = net.add_neuron(f"NOT {st['name']} stays" + (f" #{c + 1}" if k > 1 else ""), "no_stay", sid=sid)
            net.connect(("n", l_or), n, "exc", P_LEAVE - P_COND)
            for x in self_terms:
                net.connect(("n", x), n, "inh", P_LEAVE - P_TERM)
            net.connect(("n", n), hold[sid], "inh", P_STATE - P_LEAVE)

    # ---- output: a transition into an accepting state fires (at e+7)
    acc = [t for t in trans if st_by_id[t["to"]]["accepting"]]
    if acc:
        net.output = net.add_neuron("Output", "output")
        for t in acc:
            if t["id"] in enter:
                net.connect(("n", enter[t["id"]]), net.output, "exc", P_OUT - P_STATE)
            else:                               # self-loop of an accepting state: its own detector
                d = net.add_neuron(f"{name(t['from'])}→{name(t['to'])}", "enter", tid=t["id"])
                net.enters.append((t["id"], d))
                for x in into[t["from"]]:
                    net.connect(("n", x), d, "exc", 1)
                k = len(into[t["from"]])
                for c in range(k):
                    n = net.add_neuron(f"NOT {name(t['from'])}→{name(t['to'])}" + (f" #{c + 1}" if k > 1 else ""),
                                       "no_trans", tid=t["id"])
                    net.connect(("n", c1), n, "exc", 1)
                    net.connect(("n", c2), n, "exc", 1)
                    for x in opts[t["id"]]:
                        net.connect(("n", x), n, "inh", P_COND - P_TERM)
                    net.connect(("n", n), d, "inh", P_STATE - P_COND)
                net.connect(("n", d), net.output, "exc", P_OUT - P_STATE)


def summary(spec: dict) -> dict:
    """Resource counts and timing figures (Simulate tab)."""
    s = normalise(spec)
    net = build(s)
    c = net.counts()
    c.update(states=len(s["states"]), transitions=len(s["transitions"]),
             options=sum(len(t["options"]) for t in s["transitions"]),
             literals=sum(len(o) for t in s["transitions"] for o in t["options"]),
             architecture=architecture(s), min_event_gap=min_event_gap(s), **net.timing)
    return c
