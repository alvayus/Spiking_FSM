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
  Each term has up to three AND gates, one per case, and at most one of them fires:

  * latch: no change of the source state at e' - 1 and e' - 2, and its NOT gate says it
    is active;
  * bypass t-2: the source state was entered at e' - 2 and nothing changed at e' - 1;
  * bypass t-1: it was entered at e' - 1, excited directly by the AND gate that entered it
    (there is no time for an OR gate here, so in inhibition-based mode this case only
    exists out of states that a single AND gate can enter at a time; otherwise input
    spikes must be at least 2 steps apart).

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
                          "input spikes on to the next ones. Input spikes at any spacing (2 steps when "
                          "several transitions can enter a state at once)."),
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
    if arch == "parallel":
        m = _modes(s, _transitions(s, disjoint=True))
        return 2 if (not m["direct"] and m["shared"]) else 1
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
                    init_step=INIT_STEP, timing=self.timing, set_mode=self.set_mode)


def build(spec: dict) -> Netlist:
    """Build the spiking network that implements ``spec``."""
    s = normalise(spec)
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
    _Builder(s, net, arch == "parallel").run()
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

    def most(ts):                       # 0, 1 or 2 (several) of these terms at once
        ts = [l for l in ts if _together([l], n)]
        if not ts:
            return 0
        return 2 if any(_together([a, b], n) for k, a in enumerate(ts) for b in ts[k + 1:]) else 1

    moves = {}                          # active source states (mask) -> {(left, entered)}
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
        for left, entered in moves[src]:
            nxt = (act & ~left) | entered
            if nxt not in seen:
                seen.add(nxt)
                todo.append(nxt)
    sources = {t["from"] for t in trans}
    out["direct"] = all(out["ent"][sid] <= 1 and not out["both"][sid] for sid in ids)
    out["shared"] = {sid for sid in ids if out["ent"][sid] > 1 and sid in sources}
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
VIA = ("latch", "bypass 1", "bypass 2")


class _Builder:
    def __init__(self, s, net, parallel):
        self.s, self.net, self.parallel = s, net, parallel
        # terms split so that at most one term of a transition fires at a time
        self.trans = _transitions(s, disjoint=True)
        self.m = _modes(s, self.trans)
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
        for sid in dict.fromkeys(t["from"] for t in self.trans):  # S̄: NOT gate of the state
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
        return [n for t in self.trans if t["to"] == sid for n in self.gates(t)]

    def lev(self, sid):
        return [n for t in self.trans if t["from"] == sid and t["to"] != sid for n in self.gates(t)]

    def pipelined_gates(self):
        for t in self.trans:
            for k, opt in enumerate(t["options"]):
                n = self.net.add_neuron(self.term_label(t, k), "term", tid=t["id"], cond=opt)
                d = self.excite(n, opt)
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
            for x in self.lev(sid):                              # reset
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
        """The AND gates that leave a state, one step later (one neuron when they never fire
        together, else one per term)."""
        if self.m["lev"][sid] <= 1:
            key = ("reset", sid)
            if key not in self._relay:
                n = self.net.add_neuron(f"Reset {self.names[sid]} +1", "reset_relay", sid=sid)
                for x in self.lev(sid):
                    self.con(self.N(x), n, "exc", 1)
                self._relay[key] = n
            return [self._relay[key]]
        return [self.relay(t, k) for t in self.trans if t["from"] == sid and t["to"] != sid
                for k in range(len(t["options"]))]

    def parallel_gates(self):
        net, op = self.net, self.net.op
        entered = {t["to"] for t in self.trans}
        b1_ok = lambda sid: self.direct or sid not in self.m["shared"]
        for t in self.trans:
            sid = t["from"]
            # the NOT gate of the state also reads the AND gates of 2 steps before, so there is no
            # bypass t-2, unless the state can be left and entered at once (entering wins there)
            vias = ["latch"] + ((["bypass 1"] if b1_ok(sid) else []) + (["bypass 2"] if self.m["both"][sid] else [])
                                if sid in entered else [])
            for k, opt in enumerate(t["options"]):
                self.terms[(t["id"], k)] = {v: net.add_neuron(self.term_label(t, k), "term", via=v, tid=t["id"], cond=opt)
                                            for v in vias}
                for n in self.terms[(t["id"], k)].values():
                    net.terms.append((t["id"], k, n))
        # ¬OP(t) ∧ OP(t−1): input spikes 1 step before and none now (e' + T_GATE)
        no_now = net.add_neuron("NOT OP(t) AND OP(t-1)", "no_event", g=1)
        self.con(self.N(op), no_now, "exc", T_GATE - T_OP + 1)
        self.con(self.N(op), no_now, "inh", T_GATE - T_OP)
        prev2 = None
        if any(self.m["both"][sid] for sid in entered):   # OP(t) ∧ ¬OP(t−2), for bypass t−2
            prev2 = net.add_neuron("OP(t) AND NOT OP(t-2)", "no_prev", g=2)
            self.con(self.N(op), prev2, "exc", T_GATE - T_OP)
            self.con(self.N(op), prev2, "inh", T_GATE - T_OP + 2)
        self._parallel = (no_now, prev2)

    def finish_parallel(self):
        """Parallel: the cases of each term. A state that cannot be left and entered at once
        ("simple") is read by its NOT gate up to the AND gates of 2 steps before; one that can
        keeps the bypass t-2 and treats any change of the state alike."""
        no_now, prev2 = self._parallel
        net = self.net
        simple = lambda sid: not self.m["both"][sid]
        ent_relays = lambda sid: [self.relay(u, k) for u in self.trans if u["to"] == sid for k in range(len(u["options"]))]
        lev_relays = lambda sid: self.reset_relay(sid) if self.lev(sid) else []
        for sid, n in self.not_st.items():
            if simple(sid):                  # entered 2 steps before
                for x in self.ent(sid):
                    self.con(self.N(x), n, "inh", 1)
        for t in self.trans:
            sid = t["from"]
            if simple(sid):
                # a state entered at e'-1 while active fires the latch case, not bypass t-1
                change1 = self.lev(sid)                                 # left at e' - 1
                change2 = lev_relays(sid)                               # left at e' - 2
            else:
                change1 = self.ent(sid) + self.lev(sid)                 # any change at e' - 1
                change2 = ent_relays(sid) + [self.relay(u, k) for u in self.trans if u["from"] == sid and u["to"] != sid
                                             for k in range(len(u["options"]))]
            for k, opt in enumerate(t["options"]):
                T = self.terms[(t["id"], k)]
                n = T["latch"]                                         # no change at e'-1, e'-2; S̄ says active
                d = self.excite(n, opt)
                self.con(self.N(self.not_st[sid]), n, "inh", T_TERM - T_GATE)
                for x in change1 + change2:
                    self.con(self.N(x), n, "inh", 1)
                self.veto(n, opt, skip=d)
                if "bypass 2" in T:                                    # entered at e'-2, no change at e'-1
                    n = T["bypass 2"]                                  # OP + 1 and not "not entered" (NOR)
                    d = self.excite(n, opt)
                    self.con(self.N(self.nors[sid]), n, "inh", 1)
                    self.con(self.N(prev2), n, "inh", T_TERM - T_GATE)
                    for x in change1:
                        self.con(self.N(x), n, "inh", 1)
                    self.veto(n, opt, skip=d)
                if "bypass 1" in T:                                    # entered at e'-1
                    n = T["bypass 1"]
                    for x in self.ent(sid):
                        self.con(self.N(x), n, "exc", 1)
                    self.con(self.N(no_now), n, "inh", T_TERM - T_GATE)
                    if simple(sid):                                    # newly entered: not active after e'-2
                        for x in net.latches[sid] + (tuple(ent_relays(sid)) if not self.direct else ()):
                            self.con(self.N(x), n, "inh", 1)
                    self.veto(n, opt)
        if not self.direct:      # a set still pending from the previous input spikes (entered, then left)
            for sid in self.names:
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
