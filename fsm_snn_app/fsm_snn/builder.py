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

* ``parallel``: the same paradigm with bypasses, so that input spikes may come on
  consecutive steps.  It stays asynchronous: there is no clock, neurons only
  spike in response to input spikes, and the state is kept by the latches.

  ====================  =====================================================
  step e                input spikes
  e + 1                 OP; copies of the inputs (for the 1-step bypass)
  e + 2                 delay neuron, NOT(inputs), NOT(states) - latch spikes
                        of e + 1 -, "no input spikes now" gates
  e + 3                 transition (term) neurons
  e + 4                 output; OR gates "the state was entered" / "was left";
                        first reset of the source (N2, N3)
  e + 5                 set of the destination (N1, from "entered", guarded by
                        its ring); second reset (from "left")
  e + 6                 destination active (ring)
  ====================  =====================================================

  NOT(state) reads the latch and the "entered" gate, so for input spikes at e' it
  knows the state after the spikes up to e' - 3.  The spikes at e' - 1 and
  e' - 2 reach the term neurons through bypasses.  Each term has up to three
  neurons, one per case, and at most one of them fires:

  * via 1 step: the source state was entered by the spikes at e' - 1 (the term
    neurons of those spikes excite it directly);
  * via 2 steps: entered at e' - 2 ("entered" gate) and no change at e' - 1;
  * via latch: no change at e' - 2 and e' - 1, and NOT(state) says active.

  "No input spikes now" gates cancel the bypasses when there are no input spikes
  at e'.  The 1-step bypass cannot be gathered by an OR gate (there is no time),
  so a term neuron that may receive k of its spikes is inhibited by k copies of
  each gate that can veto it, where k is the largest number of transitions
  terms into its source state that can fire together (found by exploring the
  sets of states that can be active at the same time).

Every synapse has a delay of one time step (some platforms do not implement
longer delays). Where a signal has to arrive later, it goes through a delay line:
relay neurons that each repeat the spikes one step later, as in the manuscript.
A delay line belongs to the neuron (or input) it repeats and is shared by all
the synapses that need it; the delay neuron of the AND gates is OP + 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .spec import ON, OFF, normalise

# pipelined
T_OP, T_GATE, T_TERM, T_OUT, T_SET, T_RESET = 1, 2, 3, 4, 4, 4
# parallel (bypasses): OR gates at e+4, set at e+5, ring from e+6
P_OR, P_SET = 4, 5
INIT_STEP = 0           # step of the initialisation spike
FIRST_EVENT_STEP = 10   # earliest input event of the random trains (as in the manuscript, 10 ms)

ARCHITECTURES = {
    "pipelined": dict(label="Pipelined", min_event_gap=3,
                      desc="Each transition reads the state from the latches. "
                           "Input spikes at least 3 steps apart."),
    "parallel": dict(label="Parallel (bypasses)", min_event_gap=1,
                     desc="Bypasses pass the transitions of the latest input spikes on to the next "
                          "ones, without waiting for the latches. Input spikes at any spacing."),
}
DEFAULT_ARCHITECTURE = "parallel"
# timing of each implementation (steps after an input event)
TIMING = {
    "pipelined": dict(state_latency=T_SET + 1, output_latency=T_OUT, ready_step=INIT_STEP + 2),
    "parallel": dict(state_latency=P_SET + 1, output_latency=T_OUT, ready_step=INIT_STEP + 2),
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
                    terms=[list(t) for t in self.terms],
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
    _unit_delays(net)
    return net


def _unit_delays(net):
    """Replace every synapse longer than one step by a delay line of relay neurons
    (pre -> +1 -> +2 -> ...) and a one-step synapse from its last neuron."""
    taps = {}                                   # (pre ref, k) -> neuron that repeats pre k steps later
    incoming = {}
    for pre, post, rec, d, w in net.conns:
        incoming.setdefault(post, []).append((pre, rec, d))
    for i, n in enumerate(net.neurons):         # an existing pure relay of pre (the delay neuron: OP + 1)
        if n["kind"] == "delay" and len(incoming.get(i, [])) == 1:
            pre, rec, d = incoming[i][0]
            if rec == "exc" and d == 1:
                taps[(tuple(pre), 1)] = i

    def label(pre):
        return net.sources[pre[1]]["label"] if pre[0] == "src" else net.neurons[pre[1]]["label"]

    def tap(pre, k):
        if k == 0:
            return pre
        key = (pre, k)
        if key not in taps:
            prev = tap(pre, k - 1)
            n = net.add_neuron(f"{label(pre)} +{k}", "delay_line", base=list(pre), k=k)
            net.conns.append((tuple(prev) if isinstance(prev, tuple) else ("n", prev), n, "exc", 1, 1))
            taps[key] = n
        return ("n", taps[key])

    conns, net.conns = net.conns, []
    for pre, post, rec, d, w in conns:
        if d > 1:
            pre = tap(tuple(pre), d - 1)
        net.conns.append((tuple(pre), post, rec, 1, w))
    # the delay lines were appended to net.conns while rewriting; keep them
    seen = set()
    for c in net.conns:
        key = (c[0], c[1], c[2])
        assert key not in seen, f"duplicate synapse {key}"
        seen.add(key)


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
VIA = ("latch", "1 step", "2 steps")
MAX_EXPLORE = 4096          # sets of active states explored to count the 1-step bypass spikes


def _bypass_copies(s, trans, entering):
    """For each state, the largest number of terms entering it that can fire for the
    same input spikes, over the sets of states that can be active together. Falls
    back to the number of terms entering it when the machine is too large."""
    from itertools import product
    from .spec import transition_holds
    inputs = s["inputs"]
    fallback = {sid: len(v) for sid, v in entering.items()}
    if len(inputs) > 10:
        return fallback
    events = [{i for i, on in zip(inputs, bits) if on} for bits in product((0, 1), repeat=len(inputs))]
    events = [e for e in events if e]
    holds = lambda opt, ev: transition_holds(dict(options=[opt]), ev)
    start = frozenset(x["id"] for x in s["states"] if x["initial"])
    seen, todo = {start}, [start]
    best = {sid: 0 for sid in entering}
    while todo:
        if len(seen) > MAX_EXPLORE:
            return fallback
        act = todo.pop()
        for ev in events:
            fired = [(t, k) for t in trans if t["from"] in act for k, o in enumerate(t["options"]) if holds(o, ev)]
            for sid in entering:
                best[sid] = max(best[sid], sum(1 for t, _ in fired if t["to"] == sid))
            tf = {t["id"] for t, _ in fired}
            left = {t["from"] for t in trans if t["id"] in tf and t["to"] != t["from"]}
            nxt = frozenset((act - left) | {t["to"] for t in trans if t["id"] in tf})
            if nxt not in seen:
                seen.add(nxt)
                todo.append(nxt)
    return best


def _disjoint(options, order):
    """The same condition with terms that can never hold together (a term that would
    overlap an earlier one is split by the literals it lacks), so that at most one term
    of a transition fires: fewer spikes reach the 1-step bypass."""
    flip = {ON: OFF, OFF: ON}

    def sharp(a, b):                       # a AND NOT b, as disjoint terms
        if any(i in a and a[i] != v for i, v in b.items()):
            return [a]                     # already disjoint
        out, acc = [], dict(a)
        for i, v in b.items():
            if i not in a:
                out.append({**acc, i: flip[v]})
                acc[i] = v
        return out                         # [] when a lies inside b
    done = []
    for o in sorted(options, key=len):
        pieces = [dict(o)]
        for r in done:
            pieces = [c for p in pieces for c in sharp(p, r)]
        done += pieces
    return [dict(sorted(o.items(), key=lambda kv: order.index(kv[0]))) for o in done] or [{}]


def _build_parallel(s, net):
    states = s["states"]
    st_by_id = {st["id"]: st for st in states}
    trans = [dict(t, options=_disjoint(t["options"], s["inputs"])) for t in s["transitions"]
             if t["from"] in st_by_id and t["to"] in st_by_id]
    N = lambda i: ("n", i)
    src = lambda name: ("src", net.input_sources[name])
    op = net.op
    # latches: N1 -> ring, N1 guarded by the ring (a state that is already active is not set again)
    for st in states:
        n1 = net.add_neuron(f"{st['name']}.set", "latch")
        a = net.add_neuron(f"{st['name']}.a", "latch")
        b = net.add_neuron(f"{st['name']}.b", "latch")
        net.latches[st["id"]] = (n1, a, b)
        net.states[st["id"]] = (a, b)
        net.connect(N(n1), a, "exc", 1)
        net.connect(N(a), b, "exc", 1)
        net.connect(N(b), a, "exc", 1)
        net.connect(N(a), n1, "inh", 1)
        net.connect(N(b), n1, "inh", 1)
        if st["initial"]:
            net.connect(("src", net.init_source), n1, "exc", 1)
    if not trans:
        return
    name = lambda sid: st_by_id[sid]["name"]
    entering = {sid: [(t, k) for t in trans for k in range(len(t["options"])) if t["to"] == sid] for sid in st_by_id}
    leaving = {sid: [(t, k) for t in trans for k in range(len(t["options"])) if t["from"] == sid and t["to"] != sid]
               for sid in st_by_id}
    kcopies = _bypass_copies(s, trans, entering)           # copies for the 1-step bypass
    # copies of each gate that can veto a "via 1 step" term neuron: as many as the spikes
    # its source state may receive from the 1-step bypass
    need = lambda lit: max([kcopies[t["from"]] for t in trans if entering[t["from"]]
                            for o in t["options"] if lit in o.items()] + [1])
    K = max([kcopies[t["from"]] for t in trans if entering[t["from"]]] + [1])

    delay = net.add_neuron("Delay", "delay")
    net.connect(N(op), delay, "exc", T_GATE - T_OP)
    used_on = sorted({i for t in trans for o in t["options"] for i, v in o.items() if v == ON}, key=s["inputs"].index)
    used_off = sorted({i for t in trans for o in t["options"] for i, v in o.items() if v == OFF}, key=s["inputs"].index)
    not_inp, relay = {}, {}
    for i in used_on:                                           # NOT gates of the inputs (e+2)
        not_inp[i] = []
        for c in range(need((i, ON))):
            n = net.add_neuron(f"NOT {i}" + (f" #{c + 1}" if c else ""), "not_input")
            net.connect(N(op), n, "exc", T_GATE - T_OP)
            net.connect(src(i), n, "inh", T_GATE)
            not_inp[i].append(n)
    for i in used_off:                                          # copies of the inputs (e+1), besides the input
        relay[i] = [None]
        for c in range(1, need((i, OFF))):
            n = net.add_neuron(f"{i} #{c + 1}", "relay")
            net.connect(src(i), n, "exc", 1)
            relay[i].append(n)
    # "no input spikes now" (e'+2): there were input spikes at e'-g and there are none at e'
    nospike = {}
    for g, copies in ((1, K), (2, 1)):
        nospike[g] = []
        for c in range(copies):
            n = net.add_neuron(f"No spikes, {g} step{'s' if g > 1 else ''} after" + (f" #{c + 1}" if c else ""),
                               "no_event", g=g)
            net.connect(N(op), n, "exc", 1 + g)
            net.connect(N(op), n, "inh", 1)
            nospike[g].append(n)
    not_st = {}
    for sid in dict.fromkeys(t["from"] for t in trans):         # NOT of the state (e+2), see below
        n = net.add_neuron(f"NOT {name(sid)}", "not_state")
        net.connect(N(op), n, "exc", T_GATE - T_OP)
        for x in net.latches[sid]:
            net.connect(N(x), n, "inh", 1)
        not_st[sid] = n
    # term neurons: one per case that can happen (no bypass into a state nobody enters)
    terms = {}
    for t in trans:
        for k in range(len(t["options"])):
            vias = VIA if entering[t["from"]] else VIA[:1]
            base = f"{name(t['from'])}→{name(t['to'])}" + (f" ({k + 1})" if len(t["options"]) > 1 else "")
            terms[(t["id"], k)] = {v: net.add_neuron(base, "term", via=v, tid=t["id"], cond=t["options"][k])
                                   for v in vias}
            for n in terms[(t["id"], k)].values():
                net.terms.append((t["id"], k, n))
    raw = lambda opts: [n for t, k in opts for n in terms[(t["id"], k)].values()]
    E, Lv = {}, {}
    for sid in st_by_id:                                        # OR gates (e+4)
        if entering[sid]:
            E[sid] = net.add_neuron(f"{name(sid)} entered", "enter_or", sid=sid)
            for n in raw(entering[sid]):
                net.connect(N(n), E[sid], "exc", P_OR - T_TERM)
            if sid in not_st:                                   # entered by the spikes at e'-3
                net.connect(N(E[sid]), not_st[sid], "inh", 1)   # (e'+1 -> e'+2), before the latch shows it
        if leaving[sid]:
            Lv[sid] = net.add_neuron(f"{name(sid)} left", "leave_or", sid=sid)
            for n in raw(leaving[sid]):
                net.connect(N(n), Lv[sid], "exc", P_OR - T_TERM)
    for t in trans:
        sid = t["from"]
        change1 = raw(entering[sid]) + raw(leaving[sid])        # any change at e'-1 (they fire at e'+2)
        for k, opt in enumerate(t["options"]):
            T = terms[(t["id"], k)]

            def veto(n, copies=1):
                for i, v in opt.items():
                    for c in range(copies):
                        if v == ON:
                            net.connect(N(not_inp[i][c]), n, "inh", T_TERM - T_GATE)
                        elif v == OFF and c == 0:
                            net.connect(src(i), n, "inh", T_TERM)
                        elif v == OFF:
                            net.connect(N(relay[i][c]), n, "inh", T_TERM - 1)
            # via latch: no change at e'-1 and e'-2, and NOT(state) says the state is active
            n = T["latch"]
            net.connect(N(delay), n, "exc", T_TERM - T_GATE)
            net.connect(N(not_st[sid]), n, "inh", T_TERM - T_GATE)
            for x in change1:
                net.connect(N(x), n, "inh", 1)
            for X in (E, Lv):
                if sid in X:
                    net.connect(N(X[sid]), n, "inh", 1)         # spikes at e'-2: OR at e'+2
            veto(n)
            if not entering[sid]:
                continue
            # via 1 step: entered by the spikes at e'-1 (their terms fire at e'+2)
            n, ke = T["1 step"], max(kcopies[sid], 1)
            for x in raw(entering[sid]):
                net.connect(N(x), n, "exc", 1)
            for c in range(ke):
                net.connect(N(nospike[1][c]), n, "inh", 1)
            veto(n, ke)
            # via 2 steps: entered at e'-2, no change at e'-1
            n = T["2 steps"]
            net.connect(N(E[sid]), n, "exc", 1)
            net.connect(N(nospike[2][0]), n, "inh", 1)
            for x in change1:
                net.connect(N(x), n, "inh", 1)
            veto(n)
    # set (from "entered", e+5) and reset (terms at e+4, "left" at e+5)
    for sid in st_by_id:
        n1, a, b = net.latches[sid]
        if sid in E:
            net.connect(N(E[sid]), n1, "exc", P_SET - P_OR)
        for x in raw(leaving[sid]):
            net.connect(N(x), a, "inh", T_RESET - T_TERM)
            net.connect(N(x), b, "inh", T_RESET - T_TERM)
            # a set still pending from the previous spikes (entered, then left at once)
            net.connect(N(x), n1, "inh", T_RESET - T_TERM)
        if sid in Lv:
            net.connect(N(Lv[sid]), a, "inh", 1)
            net.connect(N(Lv[sid]), b, "inh", 1)
    acc = [n for t in trans if st_by_id[t["to"]]["accepting"] for k in range(len(t["options"]))
           for n in terms[(t["id"], k)].values()]
    if acc:
        net.output = net.add_neuron("Output", "output")
        for n in acc:
            net.connect(N(n), net.output, "exc", T_OUT - T_TERM)


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
