"""Specification -> spiking network (netlist) using the spiking gates of Fig. 1.

Boolean paradigm (spiking ALU, Ayuso-Martinez et al., 2025): every synapse has the same
weight (1) and the same delay (one time step), and a neuron fires when an excitatory spike
reaches it unless an inhibitory spike reaches it at the same time.  The networks follow the
rules found when these gates were implemented on Dynap-SE:

* No overexcitation: a neuron never receives two excitatory spikes in the same time step
  (on Dynap-SE such a neuron stops firing).  A neuron has a single excitatory synapse, or
  several whose spikes can never coincide.  Inhibition does not have this limit: a neuron
  excited by one synapse and inhibited by several is a NOR gate.
* OR gates are therefore built with inhibition (Fig. 1b of that work): a NOR neuron excited
  by OP and inhibited by the inputs of the OR, and an output neuron excited by OP one step
  later and inhibited by the NOR.  OP itself is an "alternative OR" (Fig. 6): one relay
  neuron per input, inhibited by the inputs before it, so at most one relay excites OP.
* No constant spike source: the gates only spike in response to input spikes, and the
  state is kept by the latches.  Paths that meet at a neuron arrive in the same step.

Building blocks:

* Input relays and OP: relay i is excited by input i and inhibited by inputs 1 .. i - 1;
  OP is excited by every relay (at most one fires).
* NOT gate: one neuron excited by OP and inhibited by the input.
* AND gate: the delay neuron OP + 1 excites the output neuron, which every NOT gate of its
  literals inhibits; a literal that must *not* spike inhibits it directly.
* Inhibition-based OR: NOR (excited by OP + 2, inhibited by the inputs of the OR) and output
  (excited by OP + 3, inhibited by the NOR).  The set neuron N1 of each latch is the output
  of the OR of the AND gates entering its state, and Output that of the accepting ones.
* Latch: N1 -> ring N2 <-> N3; the ring inhibits N1 (a state that is already active is not
  set again, so N2 never gets N1 and N3 together); the AND gates leaving the state inhibit
  N2 and N3 (reset).

Timing, in steps after input spikes at step e (both implementations):

  ====================  =====================================================
  e + 1                 input relays
  e + 2                 OP
  e + 3                 OP + 1 (delay neuron), NOT(inputs), NOT(states)
  e + 4                 AND gates (transition terms)
  e + 5                 NOR gates of the ORs; reset of the source (N2, N3)
  e + 6                 set of the destination (N1) and Output
  e + 7                 destination active (ring)
  ====================  =====================================================

NOT(state) reads the latch spikes of e + 2 and the AND gates that entered the state 3
steps earlier, so it knows the state after the input spikes up to e - 3.

* ``pipelined``: the paradigm of the manuscript, every AND gate reads NOT(state): input
  spikes at least 3 steps apart.
* ``parallel``: the same paradigm with bypasses, so that input spikes may come closer.
  Each term has up to three AND gates, one per case, and at most one of them fires:

  * latch: no change of the source state at e' - 1 and e' - 2, and NOT(state) says it is
    active;
  * bypass 2: it was entered at e' - 2 and nothing changed at e' - 1.  Excited by OP + 1 and
    inhibited by the NOR of its set (fires when it was *not* entered) and by
    OP(t) AND NOT OP(t - 2) (there were no input spikes at e' - 2);
  * bypass 1: it was entered at e' - 1; excited directly by the AND gates that enter it,
    and inhibited by NOT OP(t) AND OP(t - 1) when there are no input spikes now.  There is
    no time for an OR here, so this case exists only for states that a single AND gate can
    enter at a time (always, in deterministic machines).  Otherwise input spikes must be at
    least 2 steps apart.

Every synapse has a delay of one time step (some platforms do not implement longer delays).
Where a signal has to arrive later, it goes through a delay line: relay neurons that each
repeat the spikes one step later, as in the manuscript.  A delay line belongs to the neuron
(or input) it repeats and is shared by all the synapses that need it; the delay neuron of
the AND gates is OP + 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .spec import ON, OFF, normalise

# steps after the input spikes (both implementations)
T_REL, T_OP, T_GATE, T_TERM, T_NOR, T_SET = 1, 2, 3, 4, 5, 6
T_RESET = T_TERM + 1        # the AND gates leaving a state inhibit its ring
T_OUT = T_SET               # Output is an inhibition-based OR, like N1
INIT_STEP = 0           # step of the initialisation spike
FIRST_EVENT_STEP = 10   # earliest input event of the random trains (as in the manuscript, 10 ms)

ARCHITECTURES = {
    "pipelined": dict(label="Pipelined", min_event_gap=3,
                      desc="Each transition reads the state from the latches. "
                           "Input spikes at least 3 steps apart."),
    "parallel": dict(label="Parallel (bypasses)", min_event_gap=1,
                     desc="Bypasses pass the transitions of the latest input spikes on to the next "
                          "ones, without waiting for the latches. Input spikes at any spacing "
                          "(2 steps when several transitions can enter a state at once)."),
}
DEFAULT_ARCHITECTURE = "parallel"
_TIMING = dict(state_latency=T_SET + 1, output_latency=T_OUT, stop_latency=T_RESET - 1,
               ready_step=INIT_STEP + 2)
TIMING = {"pipelined": dict(_TIMING), "parallel": dict(_TIMING)}


def architecture(spec: dict) -> str:
    a = (spec or {}).get("architecture")
    return a if a in ARCHITECTURES else DEFAULT_ARCHITECTURE


def min_event_gap(spec: dict) -> int:
    arch = architecture(spec)
    if arch == "parallel":
        return 1 if not _shared_entries(normalise(spec)) else 2
    return ARCHITECTURES[arch]["min_event_gap"]


@dataclass
class Netlist:
    neurons: list = field(default_factory=list)   # dicts: label, kind
    sources: list = field(default_factory=list)   # dicts: label, kind, input
    conns: list = field(default_factory=list)     # (pre_ref, post, receptor, delay); every weight is 1
    latches: dict = field(default_factory=dict)   # state id -> (n1, n2, n3)
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
        self.conns.append((tuple(pre_ref), int(post), receptor, int(delay)))

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
    # OP: alternative OR of the inputs (relay i inhibited by the inputs before it)
    net.op = net.add_neuron("OP", "op")
    for k, name in enumerate(s["inputs"]):
        r = net.add_neuron(f"{name} relay", "relay", input=name)
        net.connect(("src", net.input_sources[name]), r, "exc", T_REL)
        for prev in s["inputs"][:k]:
            net.connect(("src", net.input_sources[prev]), r, "inh", T_REL)
        net.connect(("n", r), net.op, "exc", T_OP - T_REL)
    (_build_parallel if arch == "parallel" else _build_pipelined)(s, net)
    _unit_delays(net)
    return net


def _unit_delays(net):
    """Replace every synapse longer than one step by a delay line of relay neurons
    (pre -> +1 -> +2 -> ...) and a one-step synapse from its last neuron."""
    taps = {}                                   # (pre ref, k) -> neuron that repeats pre k steps later
    incoming = {}
    for pre, post, rec, d in net.conns:
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
            net.conns.append((prev, n, "exc", 1))
            taps[key] = n
        return ("n", taps[key])

    conns, net.conns = net.conns, []
    for pre, post, rec, d in conns:
        if d > 1:
            pre = tap(tuple(pre), d - 1)
        net.conns.append((tuple(pre), post, rec, 1))
    # the delay lines were appended to net.conns while rewriting; keep them
    seen = set()
    for c in net.conns:
        key = (c[0], c[1], c[2])
        assert key not in seen, f"duplicate synapse {key}"
        seen.add(key)


# ------------------------------------------------------------------------- shared parts
def _transitions(s, disjoint=False):
    st = {x["id"] for x in s["states"]}
    trans = [t for t in s["transitions"] if t["from"] in st and t["to"] in st]
    if disjoint:
        trans = [dict(t, options=_disjoint(t["options"], s["inputs"])) for t in trans]
    return trans


def _latches(s, net):
    """N1 -> ring N2 <-> N3; the ring inhibits N1, so a state that is already active is not
    set again (N2 never receives N1 and N3 in the same step)."""
    N = lambda i: ("n", i)
    for st in s["states"]:
        n1 = net.add_neuron(f"{st['name']}.N1", "latch")
        a = net.add_neuron(f"{st['name']}.N2", "latch")
        b = net.add_neuron(f"{st['name']}.N3", "latch")
        net.latches[st["id"]] = (n1, a, b)
        net.states[st["id"]] = (a, b)
        net.connect(N(n1), a, "exc", 1)
        net.connect(N(a), b, "exc", 1)
        net.connect(N(b), a, "exc", 1)
        net.connect(N(a), n1, "inh", 1)
        net.connect(N(b), n1, "inh", 1)
        if st["initial"]:
            net.connect(("src", net.init_source), n1, "exc", 1)


def _gates(s, net, trans):
    """Delay neuron OP + 1, NOT gates of the inputs used positively and NOT gates of the
    source states (all at e + T_GATE)."""
    N = lambda i: ("n", i)
    op = net.op
    delay = net.add_neuron("OP +1", "delay")
    net.connect(N(op), delay, "exc", T_GATE - T_OP)
    used_on = sorted({i for t in trans for o in t["options"] for i, v in o.items() if v == ON},
                     key=s["inputs"].index)
    not_inp = {}
    for i in used_on:
        n = net.add_neuron(f"NOT {i}", "not_input")
        net.connect(N(op), n, "exc", T_GATE - T_OP)
        net.connect(("src", net.input_sources[i]), n, "inh", T_GATE)
        not_inp[i] = n
    not_st = {}
    for sid in dict.fromkeys(t["from"] for t in trans):
        n = net.add_neuron(f"NOT {_name(s, sid)}", "not_state", sid=sid)
        net.connect(N(op), n, "exc", T_GATE - T_OP)
        for x in net.latches[sid]:             # latch spikes of e + T_GATE - 1
            net.connect(N(x), n, "inh", 1)
        not_st[sid] = n
    return delay, not_inp, not_st


def _veto(s, net, n, opt, not_inp):
    """Inhibition of an AND gate by its literals: the NOT gate of a positive literal, the
    input itself for a negated one."""
    for i, v in opt.items():
        if v == ON:
            net.connect(("n", not_inp[i]), n, "inh", T_TERM - T_GATE)
        elif v == OFF:
            net.connect(("src", net.input_sources[i]), n, "inh", T_TERM)


def _inhibition_or(net, inputs, out, label, **extra):
    """Inhibition-based OR of the AND gates ``inputs`` (they fire at e + T_TERM): a NOR
    neuron (OP + 2, inhibited by them) and the output ``out`` (OP + 3, inhibited by the
    NOR), which fires at e + T_SET."""
    N = lambda i: ("n", i)
    nor = net.add_neuron(label, "nor", **extra)
    net.connect(N(net.op), nor, "exc", T_NOR - T_OP)
    for x in inputs:
        net.connect(N(x), nor, "inh", T_NOR - T_TERM)
    net.connect(N(net.op), out, "exc", T_SET - T_OP)
    net.connect(N(nor), out, "inh", T_SET - T_NOR)
    return nor


def _term_relays(s, net, trans, gates_of):
    """The AND gates of each term repeated one step later by a single neuron (they are the
    cases of the same term, so they never fire together: no overexcitation)."""
    rel = {}

    def get(t, k):
        if (t["id"], k) not in rel:
            n = net.add_neuron(_term_name(s, t, k) + " +1", "term_relay", tid=t["id"], k=k)
            for x in gates_of(t, k):
                net.connect(("n", x), n, "exc", 1)
            rel[(t["id"], k)] = n
        return rel[(t["id"], k)]
    return get


def _finish(s, net, trans, gates_of, not_st, relay):
    """Set (inhibition-based OR into N1), reset, NOT(state) and Output."""
    N = lambda i: ("n", i)
    nors = {}
    for sid in [x["id"] for x in s["states"]]:
        n1, a, b = net.latches[sid]
        ent = [n for t in trans if t["to"] == sid for k in range(len(t["options"])) for n in gates_of(t, k)]
        lev = [n for t in trans if t["from"] == sid and t["to"] != sid
               for k in range(len(t["options"])) for n in gates_of(t, k)]
        if ent:
            nors[sid] = _inhibition_or(net, ent, n1, f"NOR set {_name(s, sid)}", sid=sid, role="set")
            if sid in not_st:                  # entered 3 steps before: not in the latch yet
                for t in trans:
                    if t["to"] == sid:
                        for k in range(len(t["options"])):
                            net.connect(N(relay(t, k)), not_st[sid], "inh", 1)
        for x in lev:
            net.connect(N(x), a, "inh", T_RESET - T_TERM)
            net.connect(N(x), b, "inh", T_RESET - T_TERM)
    acc = [n for t in trans if _state(s, t["to"])["accepting"] for k in range(len(t["options"])) for n in gates_of(t, k)]
    if acc:
        net.output = net.add_neuron("Output", "output")
        _inhibition_or(net, acc, net.output, "NOR Output", role="output")
    return nors


def _name(s, sid):
    return _state(s, sid)["name"]


def _state(s, sid):
    return next(x for x in s["states"] if x["id"] == sid)


def _term_name(s, t, k):
    name = f"{_name(s, t['from'])}→{_name(s, t['to'])}"
    return name + (f" ({k + 1})" if len(t["options"]) > 1 else "")


# ------------------------------------------------------------------------- pipelined
def _build_pipelined(s, net):
    trans = _transitions(s)
    _latches(s, net)
    if not trans:
        return
    delay, not_inp, not_st = _gates(s, net, trans)
    terms = {}
    for t in trans:
        terms[t["id"]] = []
        for k, opt in enumerate(t["options"]):
            n = net.add_neuron(_term_name(s, t, k), "term", tid=t["id"], cond=opt)
            net.connect(("n", delay), n, "exc", T_TERM - T_GATE)
            net.connect(("n", not_st[t["from"]]), n, "inh", T_TERM - T_GATE)
            _veto(s, net, n, opt, not_inp)
            net.terms.append((t["id"], k, n))
            terms[t["id"]].append(n)
    gates_of = lambda t, k: [terms[t["id"]][k]]
    _finish(s, net, trans, gates_of, not_st, _term_relays(s, net, trans, gates_of))


# ------------------------------------------------------------------------- parallel
VIA = ("latch", "bypass 1", "bypass 2")
MAX_EXPLORE = 4096          # sets of active states explored to find the states entered by several terms


def _entries(s, trans):
    """For each state, the largest number of terms entering it that can fire for the
    same input spikes, over the sets of states that can be active together. Falls
    back to the number of terms entering it when the machine is too large."""
    from itertools import product
    from .spec import transition_holds
    inputs = s["inputs"]
    entering = {x["id"]: [(t, k) for t in trans for k in range(len(t["options"])) if t["to"] == x["id"]]
                for x in s["states"]}
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


def _shared_entries(s) -> set:
    """States that several terms can enter for the same input spikes (and that some
    transition leaves): the 1-step bypass out of them would overexcite its AND gates."""
    trans = _transitions(s, disjoint=True)
    best = _entries(s, trans)
    sources = {t["from"] for t in trans}
    return {sid for sid, k in best.items() if k > 1 and sid in sources}


def _disjoint(options, order):
    """The same condition with terms that can never hold together (a term that would
    overlap an earlier one is split by the literals it lacks), so that at most one term
    of a transition fires for the same input spikes."""
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
    trans = _transitions(s, disjoint=True)
    _latches(s, net)
    if not trans:
        return
    N = lambda i: ("n", i)
    op = net.op
    shared = _shared_entries(s)
    delay, not_inp, not_st = _gates(s, net, trans)
    entered = {sid for sid in (t["to"] for t in trans)}
    # input spikes now and none 1 step before / some 1 step before and none now (e' + T_GATE)
    no_now = net.add_neuron("NOT OP(t) AND OP(t-1)", "no_event", g=1)
    net.connect(N(op), no_now, "exc", T_GATE - T_OP + 1)
    net.connect(N(op), no_now, "inh", T_GATE - T_OP)
    no_prev2 = net.add_neuron("OP(t) AND NOT OP(t-2)", "no_prev", g=2)
    net.connect(N(op), no_prev2, "exc", T_GATE - T_OP)
    net.connect(N(op), no_prev2, "inh", T_GATE - T_OP + 2)
    # AND gates: one per case that can happen
    terms = {}
    for t in trans:
        sid = t["from"]
        vias = ["latch"]
        if sid in entered:
            vias += (["bypass 1"] if sid not in shared else []) + ["bypass 2"]
        base = f"{_name(s, sid)}→{_name(s, t['to'])}"
        for k, opt in enumerate(t["options"]):
            label = base + (f" ({k + 1})" if len(t["options"]) > 1 else "")
            terms[(t["id"], k)] = {v: net.add_neuron(label, "term", via=v, tid=t["id"], cond=opt) for v in vias}
            for n in terms[(t["id"], k)].values():
                net.terms.append((t["id"], k, n))
    gates_of = lambda t, k: list(terms[(t["id"], k)].values())
    relay = _term_relays(s, net, trans, gates_of)
    tk = lambda pred: [(t, k) for t in trans if pred(t) for k in range(len(t["options"]))]
    ent = {x["id"]: [n for t, k in tk(lambda t: t["to"] == x["id"]) for n in gates_of(t, k)] for x in s["states"]}
    lev = {x["id"]: [n for t, k in tk(lambda t: t["from"] == x["id"] and t["to"] != x["id"]) for n in gates_of(t, k)]
           for x in s["states"]}
    nors = _finish(s, net, trans, gates_of, not_st, relay)
    for t in trans:
        sid = t["from"]
        change = ent[sid] + lev[sid]                       # any change of the source state
        change2 = [relay(u, k) for u, k in tk(lambda u: u["to"] == sid or (u["from"] == sid and u["to"] != sid))]
        for k, opt in enumerate(t["options"]):
            T = terms[(t["id"], k)]
            # latch: no change at e'-1 and e'-2, and NOT(state) says the state is active
            n = T["latch"]
            net.connect(N(delay), n, "exc", T_TERM - T_GATE)
            net.connect(N(not_st[sid]), n, "inh", T_TERM - T_GATE)
            for x in change:
                net.connect(N(x), n, "inh", 1)             # changes at e'-1
            for x in change2:
                net.connect(N(x), n, "inh", 1)             # changes at e'-2 (+1 of those AND gates)
            _veto(s, net, n, opt, not_inp)
            if "bypass 2" in T:                             # entered at e'-2, no change at e'-1
                n = T["bypass 2"]
                net.connect(N(delay), n, "exc", T_TERM - T_GATE)
                net.connect(N(nors[sid]), n, "inh", T_TERM + 2 - T_NOR)
                net.connect(N(no_prev2), n, "inh", T_TERM - T_GATE)
                for x in change:
                    net.connect(N(x), n, "inh", 1)
                _veto(s, net, n, opt, not_inp)
            if "bypass 1" in T:                             # entered at e'-1 (by a single AND gate)
                n = T["bypass 1"]
                for x in ent[sid]:
                    net.connect(N(x), n, "exc", 1)
                net.connect(N(no_now), n, "inh", T_TERM - T_GATE)
                _veto(s, net, n, opt, not_inp)
    # a set still pending from the previous input spikes (entered at e'-1, left at e')
    for sid in net.latches:
        for x in lev[sid]:
            net.connect(N(x), net.latches[sid][0], "inh", T_SET - T_TERM - 1)


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
