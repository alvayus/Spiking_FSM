"""Specification -> spiking network (netlist) with the spiking components of the papers.

Spiking Boolean computation (FSM manuscript, Sec. 2.3; spiking ALU, Ayuso-Martinez et al.,
2025): every synapse has the same weight (1) and the same delay (one time step), and a
neuron fires once when an excitatory spike reaches it, unless an inhibitory spike reaches it
at the same time.  The networks follow the rules found when the gates were implemented on
Dynap-SE:

* No overexcitation: a neuron never receives two excitatory spikes in the same time step
  (on Dynap-SE such a neuron stops firing).  A neuron has a single excitatory synapse, or
  several whose spikes can never coincide (as the purple neurons of the ripple-carry adder
  of the ALU).  Inhibition has no limit: a neuron excited by one synapse and inhibited by
  several is a NOR gate.
* No constant spike source: the gates only spike when input spikes arrive (OP), and the
  state is kept by the latches.  The paths that meet at a neuron arrive in the same step.

Components (Fig. 1 of the manuscript, Figs. 1, 2 and 6 of the ALU):

* OP, the alternative OR of the inputs: one input neuron per input, inhibited by the inputs
  before it, so at most one excites OP (the first input neuron is a plain relay).
* NOT gate: excited by OP, inhibited by the input.
* AND gate (inhibition-based): an output neuron excited by the delay neuron (OP + 1) and
  inhibited by the NOT gate of each input.  As in the complex case of the manuscript, a
  positive input of the term may excite the output neuron itself instead (the input,
  delayed to arrive with OP + 1), so that input needs no NOT gate.  A negated input
  inhibits it directly, and the NOT gate of the source state (S̄) requires the state.
* OR gate (inhibition-based): a NOR gate excited by OP + 2 and inhibited by the inputs, and
  an output neuron excited by OP + 3 and inhibited by the NOR gate.  When its inputs can
  never fire together, the OR is a single neuron excited by all of them.
* Latch (Fig. 1d): Set -> N1 -> ring N2 <-> N3; N1 inhibits itself and the ring inhibits
  N1 (a state that is already active is not set again); Reset inhibits N2 and N3.  N1 is
  the output of the OR gate of the AND gates that enter the state.

Two set modes, chosen for each machine by exploring the sets of states that can be active
together:

* direct: no state can be entered by two AND gates for the same input spikes, nor left
  and entered at once (always, in deterministic machines).  N1 is the OR of the AND gates
  that enter the state, excited by them directly.
* inhibition-based: otherwise, N1 is the output of an inhibition-based OR gate, one step
  later.

Timing, in steps after input spikes at step e:

  ====================  =====================================================
  e + 1                 input neurons of the alternative OR
  e + 2                 OP
  e + 3                 delay neuron (OP + 1), NOT gates of the inputs and states
  e + 4                 AND gates (transitions)
  e + 5                 reset of the source (N2, N3); direct: set (N1), Output;
                        inhibition-based: NOR gates
  e + 6                 direct: destination active (ring);
                        inhibition-based: set (N1), Output
  e + 7                 inhibition-based: destination active (ring)
  ====================  =====================================================

* ``pipelined``: the paradigm of the simple case of the manuscript: every AND gate reads
  the NOT gate of its source state, which knows the state after the input spikes up to
  e - 3: input spikes at least 3 steps apart.
* ``parallel``: the bypasses of the complex case, so that input spikes may come closer.
  The NOT gate of the source state knows the latch and the AND gates that entered the state
  2 steps before; a delay neuron of the Reset says the state was left 2 steps before (and
  not entered at the same time). Each term has up to two AND gates, one per case, and at
  most one of them fires:

  * latch: the NOT gate says the state is active and it was not left at e' - 2 or e' - 1;
  * bypass t-1: the state was entered at e' - 1, excited directly by the AND gate that
    entered it. There is no time for an OR gate here: where several AND gates can enter the
    state at once, there is one bypass per group of them that never fire together, with
    priority between groups as in the Alt. OR, so only one fires.

  A state that every input event leaves (its outgoing conditions cover every event) is told
  so by OP, delayed, instead of by every AND gate that leaves it; a state that is active
  from the start and is never left needs no Set, Reset, NOT gate or bypass. ``build`` keeps
  the cheaper network, with or without OP telling the leaves.

Every synapse has a delay of one time step (some platforms do not implement longer delays).
Where a signal has to arrive later, it goes through a delay line: delay neurons that each
repeat the spikes one step later, as in the manuscript, shared by all the synapses that need
them; the delay neuron of the AND gates is OP + 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .spec import ON, OFF, normalise

# steps after the input spikes
T_REL, T_OP, T_GATE, T_TERM = 1, 2, 3, 4
T_RESET = T_TERM + 1        # the AND gates leaving a state inhibit its ring
INIT_STEP = 0           # step of the initialisation spike
FIRST_EVENT_STEP = 10   # earliest input event of the random trains (as in the manuscript, 10 ms)

ARCHITECTURES = {
    "pipelined": dict(label="Pipelined", min_event_gap=3,
                      desc="Simple case of the manuscript: each transition reads the state from the "
                           "latches. Input spikes at least 3 steps apart."),
    "parallel": dict(label="Parallel (bypasses)", min_event_gap=1,
                     desc="Complex case of the manuscript: bypasses pass the transitions of the latest "
                          "input spikes on to the next ones. Input spikes at any spacing."),
}
DEFAULT_ARCHITECTURE = "parallel"


def timing(direct: bool, direct_output: bool) -> dict:
    t_set = T_TERM + (1 if direct else 2)
    return dict(state_latency=t_set + 1, output_latency=T_TERM + (1 if direct_output else 2),
                stop_latency=T_RESET - 1, ready_step=INIT_STEP + 1, set_step=t_set)


TIMING = {a: timing(True, True) for a in ARCHITECTURES}      # deterministic machines (most common)


def architecture(spec: dict) -> str:
    a = (spec or {}).get("architecture")
    return a if a in ARCHITECTURES else DEFAULT_ARCHITECTURE


def min_event_gap(spec: dict) -> int:
    s = normalise(spec)
    arch = architecture(s)
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
    timing: dict = field(default_factory=lambda: dict(TIMING[DEFAULT_ARCHITECTURE]))
    set_mode: str = "direct"
    perm: list = field(default_factory=list)      # states that are never left (no Set, Reset or loops)

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
                    init_step=INIT_STEP, timing=self.timing, set_mode=self.set_mode, perm=self.perm)


def build(spec: dict) -> Netlist:
    """Build the spiking network that implements ``spec``: the cheapest (fewest neurons, then
    fewest synapses) of the variants with and without OP telling that a state was left."""
    s = normalise(spec)
    nets = [_build(s, use_op) for use_op in (False, True)]
    return min(nets, key=lambda n: (len(n.neurons), len(n.conns)))


def _build(s, use_op):
    arch = architecture(s)
    net = Netlist(architecture=arch)
    for name in s["inputs"]:
        net.input_sources[name] = net.add_source(name, "input", name)
    net.init_source = net.add_source("init", "init")
    # OP: alternative OR of the inputs (input neuron i inhibited by the inputs before it)
    net.op = net.add_neuron("OP", "op")
    for k, name in enumerate(s["inputs"]):
        r = net.add_neuron(f"OP·{name}", "relay", input=name)
        net.connect(("src", net.input_sources[name]), r, "exc", T_REL)
        for prev in s["inputs"][:k]:
            net.connect(("src", net.input_sources[prev]), r, "inh", T_REL)
        net.connect(("n", r), net.op, "exc", T_OP - T_REL)
    _Builder(s, net, arch == "parallel", use_op).run()
    _unit_delays(net)
    return net


def _unit_delays(net):
    """Replace every synapse longer than one step by a delay line of delay neurons
    (pre -> +1 -> +2 -> ...) and a one-step synapse from its last neuron."""
    taps = {}                                   # (pre ref, k) -> neuron that repeats pre k steps later
    incoming = {}
    for pre, post, rec, d in net.conns:
        incoming.setdefault(post, []).append((pre, rec, d))
    # existing pure relays of pre: the delay neuron (OP + 1), the first input neuron of OP
    # and, with a single input, OP itself (that input 2 steps later)
    for i, n in enumerate(net.neurons):
        if n["kind"] in ("delay", "relay") and len(incoming.get(i, [])) == 1:
            pre, rec, d = incoming[i][0]
            if rec == "exc" and d == 1:
                taps[(tuple(pre), 1)] = i
                if n["kind"] == "relay":
                    n["tap"] = 1                 # it is also the input one step later
    if len(net.input_sources) == 1 and len(incoming.get(net.op, [])) == 1:
        src = ("src", next(iter(net.input_sources.values())))
        if taps.get((src, 1)) == incoming[net.op][0][0][1]:
            taps[(src, 2)] = net.op

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
    seen = set()
    for c in net.conns:
        key = (c[0], c[1], c[2])
        assert key not in seen, f"duplicate synapse {key}"
        seen.add(key)


# ------------------------------------------------------------------------- analysis of the machine
def _transitions(s, disjoint=False):
    st = {x["id"] for x in s["states"]}
    trans = [t for t in s["transitions"] if t["from"] in st and t["to"] in st]
    if disjoint:
        trans = [dict(t, options=_disjoint(t["options"], s["inputs"])) for t in trans]
    return trans


def _lits(o):
    return {i: v for i, v in o.items() if v in (ON, OFF)}


def _together(lits, n_inputs):
    """Whether these terms (dicts of literals) can hold for the same input spikes: no input
    both on and off, and not all the inputs forced off (some input always spikes)."""
    seen = {}
    for d in lits:
        for i, v in d.items():
            if seen.setdefault(i, v) != v:
                return False
    return not (len(seen) >= n_inputs and all(v == OFF for v in seen.values()))


def _moves(terms, bit, n_inputs):
    """Every (left, entered) pair of state masks that the input spikes can produce from
    these terms (from, to, literals). Exact: the inputs are decided one at a time (the one
    most read first), each term keeps only its undecided literals, a term that can no longer
    change the masks is dropped, and the same remaining problem is solved only once."""
    out, memo = set(), set()

    def go(und, left, entered, assigned, anyon):
        while True:                     # fire the decided terms, drop those that change nothing
            fired = [(f, g) for f, g, l in und if not l]
            for f, g in fired:
                left |= 0 if f == g else bit[f]
                entered |= bit[g]
            und = [(f, g, l) for f, g, l in und if l and not ((bit[f] & left or f == g) and bit[g] & entered)]
            if not fired or not und:
                break
        if not und:
            if anyon or assigned < n_inputs:                      # a real event
                out.add((left, entered))
            return
        reads = {}
        for _, _, l in und:
            for i, _ in l:
                reads[i] = reads.get(i, 0) + 1
        key = (frozenset((f, g, l) for f, g, l in und), left, entered, anyon, min(n_inputs - assigned, len(reads) + 1))
        if key in memo:
            return
        memo.add(key)
        var = max(reads, key=reads.get)
        for on in (True, False):
            want = ON if on else OFF
            nxt = []
            for f, g, l in und:
                d = dict(l)
                if var in d:
                    if d.pop(var) != want:
                        continue                                  # contradicted: never fires
                    nxt.append((f, g, frozenset(d.items())))
                else:
                    nxt.append((f, g, l))
            go(nxt, left, entered, assigned + 1, anyon or on)

    go([(f, g, frozenset(_lits(o).items())) for f, g, o in terms], 0, 0, 0, False)
    return out


def _covers(options, n_inputs):
    """Whether these terms (dicts of literals) cover every input event: whatever inputs
    spike (at least one), one of them holds. Exact, deciding one input at a time."""
    def go(opts, assigned, anyon):
        if any(not o for o in opts):
            return True
        if not opts:
            return not (anyon or assigned < n_inputs)      # an event no term covers?
        var = next(iter(opts[0]))
        for on in (True, False):
            want = ON if on else OFF
            nxt = [{i: v for i, v in o.items() if i != var} for o in opts if o.get(var, want) == want]
            if not go(nxt, assigned + 1, anyon or on):
                return False
        return True
    return go([_lits(o) for o in options], 0, False)


def _modes(s, trans) -> dict:
    """Explore, exactly, the sets of states that can be active together and every input
    event: for each state, whether no term, one or several (2) can enter it or leave it for
    the same input spikes, whether it can be left and entered at once, and the same count for
    the terms that enter accepting states."""
    ids = [x["id"] for x in s["states"]]
    n = len(s["inputs"])
    bit = {sid: 1 << k for k, sid in enumerate(ids)}
    acc = {x["id"] for x in s["states"] if x["accepting"]}
    terms = [(t["from"], t["to"], _lits(o)) for t in trans for o in t["options"]]
    out = dict(ent={sid: 0 for sid in ids}, lev={sid: 0 for sid in ids}, acc=0, both={sid: False for sid in ids})
    reent = 0                           # states that can be entered while already active
    coact = set()                       # pairs of source states that can be active together

    def most(ts):                       # 0, 1 or 2 (several) of these terms at once
        ts = [l for l in ts if _together([l], n)]
        if not ts:
            return 0
        return 2 if any(_together([a, b], n) for k, a in enumerate(ts) for b in ts[k + 1:]) else 1

    moves = {}                          # active source states (mask) -> {(left, entered)}
    can_enter = {}                      # active source states (mask) -> states some term can enter
    start = sum(bit[x["id"]] for x in s["states"] if x["initial"])
    seen, todo = {start}, [start]
    while todo:
        act = todo.pop()
        src = sum({bit[f] for f, _, _ in terms if act & bit[f]})
        if src not in moves:            # the counts depend only on the active source states
            ts = [t for t in terms if src & bit[t[0]]]
            for sid in ids:
                into = [l for f, g, l in ts if g == sid]
                away = [l for f, g, l in ts if f == sid and g != sid]
                out["ent"][sid] = max(out["ent"][sid], most(into))
                out["lev"][sid] = max(out["lev"][sid], most(away))
                out["both"][sid] |= any(_together([a, b], n) for a in into for b in away)
            out["acc"] = max(out["acc"], most([l for f, g, l in ts if g in acc]))
            moves[src] = _moves([(f, g, l) for f, g, l in ts], bit, n)
            can_enter[src] = sum({bit[g] for f, g, l in ts if _together([l], n)})
        reent |= act & can_enter[src]
        on = [sid for sid in ids if src & bit[sid]]
        coact.update((a, b) for a in on for b in on)
        for left, entered in moves[src]:
            nxt = (act & ~left) | entered
            if nxt not in seen:
                seen.add(nxt)
                todo.append(nxt)
    always = start
    for x in seen:
        always &= x
    out["perm"] = {sid for sid in ids if always & bit[sid]}       # active from the start, never left
    var = [sid for sid in ids if sid not in out["perm"]]
    out["direct"] = all(out["ent"][sid] <= 1 and not out["both"][sid] for sid in var)
    out["reent"] = {sid: bool(reent & bit[sid]) for sid in ids}
    out["coact"] = coact
    return out


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


# ------------------------------------------------------------------------- the network


class _Builder:
    def __init__(self, s, net, parallel, use_op=True):
        self.s, self.net, self.parallel = s, net, parallel
        # terms split so that at most one term of a transition fires at a time
        self.trans = _transitions(s, disjoint=True)
        self.m = _modes(s, self.trans)
        # a state that is active from the start and can never be left (the exploration finds
        # it in every reachable set) needs no Set, Reset, NOT gate or bypass: its loops change
        # nothing (they are kept only to tell Output, in accepting states), and the terms that
        # enter it only reset their own source state
        self.perm = self.m["perm"]
        net.perm = sorted(self.perm)
        acc = {x["id"] for x in s["states"] if x["accepting"]}
        self.trans = [t for t in self.trans if not (t["from"] == t["to"] and t["from"] in self.perm - acc)]
        # states that every input event leaves (their outgoing conditions cover all events):
        # OP, delayed, tells that they were left, instead of every AND gate that leaves them
        n_in = len(s["inputs"])
        self.total = {sid: use_op and sid not in self.perm and _covers(
            [o for t in self.trans if t["from"] == sid and t["to"] != sid for o in t["options"]], n_in)
            for sid in (x["id"] for x in s["states"])}
        self.direct = self.m["direct"]
        self.direct_out = self.m["acc"] <= 1
        net.set_mode = "direct" if self.direct else "inhibition"
        net.timing = timing(self.direct, self.direct_out)
        self.t_set = net.timing["set_step"]
        self._delay = None
        self._not_inp = {}
        self.names = {x["id"]: x["name"] for x in s["states"]}

    # -- helpers ------------------------------------------------------------------
    def N(self, i):
        return ("n", i)

    def con(self, pre, post, rec, d):
        self.net.connect(pre, post, rec, d)

    def delay(self):
        """Delay neuron of the AND gates (OP + 1, e + T_GATE)."""
        if self._delay is None:
            self._delay = self.net.add_neuron("OP +1", "delay")
            self.con(self.N(self.net.op), self._delay, "exc", T_GATE - T_OP)
        return self._delay

    def not_input(self, i):
        """NOT gate of an input (e + T_GATE): excited by OP, inhibited by the input."""
        if i not in self._not_inp:
            n = self.net.add_neuron(f"NOT {i}", "not_input")
            self.con(self.N(self.net.op), n, "exc", T_GATE - T_OP)
            self.con(("src", self.net.input_sources[i]), n, "inh", T_GATE)
            self._not_inp[i] = n
        return self._not_inp[i]

    def driver(self, opt):
        """The positive input that excites an AND gate (delayed to arrive with OP + 1), or None."""
        return next((i for i, v in opt.items() if v == ON), None)

    def excite(self, n, opt, use_driver=True):
        """Excitation of an AND gate (e + T_TERM): its first positive input, or OP + 1.
        Returns the literal used, which then needs no NOT gate."""
        d = self.driver(opt) if use_driver else None
        if d is not None:
            self.con(("src", self.net.input_sources[d]), n, "exc", T_TERM)
        else:
            self.con(self.N(self.delay()), n, "exc", T_TERM - T_GATE)
        return d

    def veto(self, n, opt, skip=None):
        """Inhibition of an AND gate by its inputs: NOT gate of a positive input, the input
        itself for a negated one."""
        for i, v in opt.items():
            if i == skip:
                continue
            if v == ON:
                self.con(self.N(self.not_input(i)), n, "inh", T_TERM - T_GATE)
            elif v == OFF:
                self.con(("src", self.net.input_sources[i]), n, "inh", T_TERM)

    def inhibition_or(self, inputs, out, label, **extra):
        """Inhibition-based OR gate of AND gates (they fire at e + T_TERM): NOR gate (OP + 2,
        inhibited by them) and output ``out`` (OP + 3, inhibited by the NOR gate)."""
        nor = self.net.add_neuron(label, "nor", **extra)
        self.con(self.N(self.net.op), nor, "exc", T_TERM + 1 - T_OP)
        for x in inputs:
            self.con(self.N(x), nor, "inh", 1)
        self.con(self.N(self.net.op), out, "exc", T_TERM + 2 - T_OP)
        self.con(self.N(nor), out, "inh", 1)
        return nor

    # -- network --------------------------------------------------------------------
    def run(self):
        s, net = self.s, self.net
        for st in s["states"]:                                   # latches (Fig. 1d)
            n1 = net.add_neuron(f"{st['name']}.N1", "latch")
            a = net.add_neuron(f"{st['name']}.N2", "latch")
            b = net.add_neuron(f"{st['name']}.N3", "latch")
            net.latches[st["id"]] = (n1, a, b)
            net.states[st["id"]] = (a, b)
            self.con(self.N(n1), a, "exc", 1)
            self.con(self.N(a), b, "exc", 1)
            self.con(self.N(b), a, "exc", 1)
            self.con(self.N(n1), n1, "inh", 1)
            self.con(self.N(a), n1, "inh", 1)
            self.con(self.N(b), n1, "inh", 1)
            if st["initial"]:                                    # active at start: straight into the ring
                self.con(("src", net.init_source), a, "exc", 1)
        if not self.trans:
            return
        self.not_st = {}
        for sid in dict.fromkeys(t["from"] for t in self.trans if t["from"] not in self.perm):  # S̄
            n = net.add_neuron(f"NOT {self.names[sid]}", "not_state", sid=sid)
            self.con(self.N(net.op), n, "exc", T_GATE - T_OP)
            for x in net.latches[sid]:
                self.con(self.N(x), n, "inh", 1)
            self.not_st[sid] = n
        self.terms = {}
        (self.parallel_gates if self.parallel else self.pipelined_gates)()
        self.finish()

    def term_label(self, t, k):
        name = f"{self.names[t['from']]}→{self.names[t['to']]}"
        return name + (f" ({k + 1})" if len(t["options"]) > 1 else "")

    def gates(self, t, k=None):
        ks = range(len(t["options"])) if k is None else [k]
        return [n for kk in ks for n in self.terms[(t["id"], kk)].values()]

    def ent(self, sid):
        """The AND gates that set the state (none for a state that is never left)."""
        return [] if sid in self.perm else [n for t in self.trans if t["to"] == sid for n in self.gates(t)]

    def lev(self, sid):
        """The AND gates that reset the state (none for a state that is never left)."""
        return [] if sid in self.perm else [n for t in self.trans if t["from"] == sid and t["to"] != sid for n in self.gates(t)]

    def pipelined_gates(self):
        for t in self.trans:
            for k, opt in enumerate(t["options"]):
                n = self.net.add_neuron(self.term_label(t, k), "term", tid=t["id"], cond=opt)
                d = self.excite(n, opt)
                if t["from"] in self.not_st:
                    self.con(self.N(self.not_st[t["from"]]), n, "inh", T_TERM - T_GATE)
                self.veto(n, opt, skip=d)
                self.net.terms.append((t["id"], k, n))
                self.terms[(t["id"], k)] = {None: n}

    def finish(self):
        """Set (N1), reset (N2, N3) and Output."""
        net = self.net
        self.nors = {}
        self._relay = {}
        for sid in self.names:
            n1, a, b = net.latches[sid]
            ent = self.ent(sid)
            if ent and self.direct:                              # OR gate of exclusive inputs: N1 itself
                for x in ent:
                    self.con(self.N(x), n1, "exc", self.t_set - T_TERM)
            elif ent:                                            # inhibition-based OR gate
                self.nors[sid] = self.inhibition_or(ent, n1, f"NOR set {self.names[sid]}", sid=sid, role="set")
                if sid in self.not_st:                           # entered 3 steps before: not in the latch yet
                    for t in self.trans:
                        if t["to"] == sid:
                            for k in range(len(t["options"])):
                                self.con(self.N(self.relay(t, k)), self.not_st[sid], "inh", 1)
            if self.total[sid]:                                  # reset: any input spikes leave it
                self.con(self.N(net.op), a, "inh", T_RESET - T_OP)
                self.con(self.N(net.op), b, "inh", T_RESET - T_OP)
            else:
                for x in self.lev(sid):                          # reset
                    self.con(self.N(x), a, "inh", T_RESET - T_TERM)
                    self.con(self.N(x), b, "inh", T_RESET - T_TERM)
        acc = [n for t in self.trans if any(x["id"] == t["to"] and x["accepting"] for x in self.s["states"])
               for n in self.gates(t)]
        if acc:
            net.output = net.add_neuron("Output", "output")
            if self.direct_out:
                for x in acc:
                    self.con(self.N(x), net.output, "exc", 1)
            else:
                self.inhibition_or(acc, net.output, "NOR Output", role="output")
        if self.parallel:
            self.finish_parallel()

    def relay(self, t, k):
        """The AND gates of a term one step later (they never fire together)."""
        key = (t["id"], k)
        if key not in self._relay:
            n = self.net.add_neuron(self.term_label(t, k) + " +1", "term_relay", tid=t["id"], k=k)
            for x in self.gates(t, k):
                self.con(self.N(x), n, "exc", 1)
            self._relay[key] = n
        return self._relay[key]

    def reset_relay(self, sid):
        """Parallel: the AND gates that leave a state, one step later, when the state was not
        entered at the same time (entering wins): one neuron when they never fire together,
        else one per term. It is inhibited by the AND gates that can enter the state for the
        same input spikes."""
        if not self.lev(sid):
            return []
        if self.total[sid]:          # input spikes 2 steps before, which did not enter the state
            key = ("reset", sid, 0)
            if key not in self._relay:
                n = self.net.add_neuron(f"Reset {self.names[sid]} +1", "reset_relay", sid=sid)
                self.con(self.N(self.net.op), n, "exc", T_RESET - T_OP)
                for x in self.ent(sid):
                    self.con(self.N(x), n, "inh", 1)
                self._relay[key] = n
            return [self._relay[key]]
        cond = lambda t, k: _lits(t["options"][k])
        into = [(u, j) for u in self.trans if u["to"] == sid for j in range(len(u["options"]))]
        away = [(t, k) for t in self.trans if t["from"] == sid and t["to"] != sid for k in range(len(t["options"]))]
        if self.m["lev"][sid] > 1 and not self.m["both"][sid]:   # the +1 of each term will do
            return [self.relay(t, k) for t, k in away]
        groups = [away] if self.m["lev"][sid] <= 1 else [[x] for x in away]
        out = []
        for g in groups:
            key = ("reset", sid, len(out))
            if key not in self._relay:
                one = len(groups) > 1
                label = f"Reset {self.names[sid]} +1" + (f" ({self.term_label(*g[0])})" if one else "")
                n = self.net.add_neuron(label, "reset_relay", sid=sid, **(dict(tid=g[0][0]["id"], k=g[0][1]) if one else {}))
                for t, k in g:
                    for x in self.gates(t, k):
                        self.con(self.N(x), n, "exc", 1)
                if self.m["both"][sid]:
                    n_in = len(self.s["inputs"])
                    for u, j in into:
                        if any(_together([cond(u, j), cond(t, k)], n_in) for t, k in g):
                            for x in self.gates(u, j):
                                self.con(self.N(x), n, "inh", 1)
                self._relay[key] = n
            out.append(self._relay[key])
        return out

    def coincide(self, a, b):
        """Whether two terms (transition, option) can fire for the same input spikes."""
        (u, j), (w, i) = a, b
        return ((u["from"], w["from"]) in self.m["coact"]
                and _together([_lits(u["options"][j]), _lits(w["options"][i])], len(self.s["inputs"])))

    def entry_groups(self, sid):
        """The terms that enter a state, in groups whose terms never fire together (one
        bypass t-1 per group; with a single group, one bypass)."""
        groups = []
        if sid in self.perm:
            return groups
        for x in [(u, j) for u in self.trans if u["to"] == sid for j in range(len(u["options"]))]:
            g = next((g for g in groups if not any(self.coincide(x, y) for y in g)), None)
            if g is None:
                groups.append([x])
            else:
                g.append(x)
        return groups

    def parallel_gates(self):
        net, op = self.net, self.net.op
        self.groups = {sid: self.entry_groups(sid) for sid in self.names}
        for t in self.trans:
            ng = len(self.groups[t["from"]])
            vias = ["latch"] + [f"bypass 1" + (f"/{g + 1}" if g else "") for g in range(ng)]
            for k, opt in enumerate(t["options"]):
                self.terms[(t["id"], k)] = {v: net.add_neuron(self.term_label(t, k), "term", tid=t["id"], cond=opt,
                                                              **(dict(via="bypass 1", grp=vias.index(v) - 1) if v != "latch" else dict(via="latch")))
                                            for v in vias}
                for n in self.terms[(t["id"], k)].values():
                    net.terms.append((t["id"], k, n))
        self._no_now = None
        if any(self.groups[t["from"]] for t in self.trans):
            # ¬OP(t) ∧ OP(t−1): input spikes 1 step before and none now (e' + T_GATE)
            self._no_now = net.add_neuron("NOT OP(t) AND OP(t-1)", "no_event", g=1)
            self.con(self.N(op), self._no_now, "exc", T_GATE - T_OP + 1)
            self.con(self.N(op), self._no_now, "inh", T_GATE - T_OP)

    def finish_parallel(self):
        """Parallel: the cases of each term, at most one of which fires.

        * The NOT gate of the state reads the latch and the AND gates that entered the state
          2 steps before (and, with an Inh.-based OR, the +1 of those of 3 steps before).
        * latch: the NOT gate says the state is active and it was not left 2 or 1 steps before
          (the Resets +1, the AND gates that leave it).
        * bypass t-1: the state was entered 1 step before (excited by the AND gates that
          entered it). Where several of those AND gates can fire together, there is one
          bypass per group of them that never fire together, and a group is inhibited by the
          groups before it (priority, as in the Alt. OR), so only one fires. If the state
          can also be entered while it is active, one of the two
          cases must yield: the bypass, inhibited by the latch, or the latch case, inhibited by
          the AND gates that enter the state (the only choice where a state can be left and
          entered at once); the cheaper one is used."""
        net = self.net
        ent_relays = lambda sid: [self.relay(u, k) for u in self.trans if u["to"] == sid for k in range(len(u["options"]))]
        for sid, n in self.not_st.items():                       # entered 2 steps before
            for x in self.ent(sid):
                self.con(self.N(x), n, "inh", 1)
        for sid in dict.fromkeys(t["from"] for t in self.trans):
            outs = [t for t in self.trans if t["from"] == sid]
            groups = self.groups[sid]
            b1 = bool(groups)
            latch_guard = []                                     # inhibits bypass t-1: the state was active
            entry_guard = []                                     # inhibits the latch case: entered 1 step before
            # every input event leaves the state (its outgoing conditions cover all of them):
            # it is active after input spikes only if they entered it, so the latch case never
            # fires 1 step after input spikes, and OP of then (+3) inhibits it instead of the AND
            # gates that left or entered the state
            total = b1 and self.total[sid]
            if b1 and not total and (self.m["reent"][sid] or self.m["both"][sid]):
                newly = list(net.latches[sid]) + ([] if self.direct else ent_relays(sid))
                if self.m["both"][sid] or len(self.ent(sid)) < len(newly):
                    entry_guard = self.ent(sid)
                else:
                    latch_guard = newly
            # left 1 step before (only when input spikes can come 1 step apart) or 2 steps before
            change = ([] if total else self.lev(sid) + entry_guard) + self.reset_relay(sid)
            for t in outs:
                for k, opt in enumerate(t["options"]):
                    T = self.terms[(t["id"], k)]
                    n = T["latch"]
                    d = self.excite(n, opt)
                    if sid in self.not_st:
                        self.con(self.N(self.not_st[sid]), n, "inh", T_TERM - T_GATE)
                    for x in change:
                        self.con(self.N(x), n, "inh", 1)
                    if total:                                    # input spikes 1 step before
                        self.con(self.N(net.op), n, "inh", T_TERM + 1 - T_OP)
                    self.veto(n, opt, skip=d)
                    for g, grp in enumerate(groups):
                        n = T["bypass 1" + (f"/{g + 1}" if g else "")]
                        for u, j in grp:                             # the AND gates that entered it
                            for x in self.gates(u, j):
                                self.con(self.N(x), n, "exc", 1)
                        for y in [y for h in groups[:g] for y in h if any(self.coincide(x, y) for x in grp)]:
                            for x in self.gates(*y):                 # priority: an earlier group fired
                                self.con(self.N(x), n, "inh", 1)
                        self.con(self.N(self._no_now), n, "inh", T_TERM - T_GATE)
                        for x in latch_guard:
                            self.con(self.N(x), n, "inh", 1)
                        self.veto(n, opt)
        if not self.direct:      # a set still pending from the previous input spikes (entered, then left)
            for sid in self.names:
                if self.total[sid]:
                    self.con(self.N(net.op), net.latches[sid][0], "inh", self.t_set - T_OP - 1)
                    continue
                for x in self.lev(sid):
                    self.con(self.N(x), net.latches[sid][0], "inh", self.t_set - T_TERM - 1)


def summary(spec: dict) -> dict:
    """Resource counts and timing figures (Simulate tab)."""
    s = normalise(spec)
    net = build(s)
    c = net.counts()
    c.update(states=len(s["states"]), transitions=len(s["transitions"]),
             options=sum(len(t["options"]) for t in s["transitions"]),
             literals=sum(len(o) for t in s["transitions"] for o in t["options"]),
             architecture=architecture(s), min_event_gap=min_event_gap(s), set_mode=net.set_mode,
             **{k: v for k, v in net.timing.items() if k != "set_step"})
    return c
