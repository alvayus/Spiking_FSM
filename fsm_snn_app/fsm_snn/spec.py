"""FSM specification: data model, validation and built-in machines.

A specification is a plain JSON-compatible dict, which is what the visual
editor sends and what is saved to disk::

    {
      "name": "My FSM",
      "inputs": ["A", "B"],
      "states": [
        {"id": "s0", "name": "S0", "x": 120, "y": 200,
         "initial": true, "accepting": false}
      ],
      "transitions": [
        {"id": "t0", "from": "s0", "to": "s1",
         "options": [{"A": "on", "B": "off"}]}
      ]
    }

A transition condition is a Boolean expression over the inputs in disjunctive
normal form: a transition has one or more *options* (terms OR-ed together);
each option says, for each input, whether it must spike ("on", literal X),
must not spike ("off", literal ¬X), or does not matter (input absent).
The editor shows them with the operators ¬ (NOT), ∧ (AND) and ∨ (OR).

Every state is a spiking latch: it stays active until one of its outgoing
transitions fires. ``normalise`` therefore forces ``on_mismatch = "hold"`` and
``always_active = False`` on every state; the builder and the reference
automaton still understand the other modes, but the application no longer
exposes them.
"""

from __future__ import annotations

import copy
import math
import re

ON, OFF = "on", "off"


# --------------------------------------------------------------------------
# Normalisation and validation
# --------------------------------------------------------------------------

def normalise(spec: dict) -> dict:
    """Return a cleaned deep copy of ``spec`` with defaults filled in."""
    s = copy.deepcopy(spec or {})
    s.setdefault("name", "FSM")
    s["inputs"] = [str(i).strip() for i in s.get("inputs", []) if str(i).strip()]
    states = []
    for k, st in enumerate(s.get("states", [])):
        st = dict(st)
        st.setdefault("id", f"s{k}")
        st["id"] = str(st["id"])
        st.setdefault("name", st["id"].upper())
        st["initial"] = bool(st.get("initial", False))
        st["accepting"] = bool(st.get("accepting", False))
        # all states are held latches (see module docstring)
        st["always_active"] = False
        st["on_mismatch"] = "hold"
        states.append(st)
    s["states"] = states
    trans = []
    for k, t in enumerate(s.get("transitions", [])):
        t = dict(t)
        t.setdefault("id", f"t{k}")
        t["id"] = str(t["id"])
        opts = []
        for opt in t.get("options", [{}]):
            clean = {i: v for i, v in (opt or {}).items()
                     if i in s["inputs"] and v in (ON, OFF)}
            opts.append(clean)
        t["options"] = opts or [{}]
        trans.append(t)
    s["transitions"] = trans
    return s


def validate(spec: dict) -> tuple[list[str], list[str]]:
    """Return (errors, warnings), ready to show in the UI."""
    s = normalise(spec)
    errors, warnings = [], []
    ids = [st["id"] for st in s["states"]]
    names = {st["id"]: st["name"] for st in s["states"]}
    if not s["inputs"]:
        errors.append("Add at least one input signal.")
    if len(set(s["inputs"])) != len(s["inputs"]):
        errors.append("Two input signals have the same name.")
    if not s["states"]:
        errors.append("Add at least one state.")
    if len(set(ids)) != len(ids):
        errors.append("Two states have the same identifier.")
    if s["states"] and not any(st["initial"] for st in s["states"]):
        errors.append("Mark at least one state as active at start.")
    for t in s["transitions"]:
        if t["from"] not in ids or t["to"] not in ids:
            errors.append(f"Transition {t['id']} points to a state that does not exist.")
            continue
        label = f"{names[t['from']]} → {names[t['to']]}"
        for opt in t["options"]:
            # a term that negates every input can never fire: every event has a spike
            if s["inputs"] and all(opt.get(i) == OFF for i in s["inputs"]):
                warnings.append(
                    f"{label}: the term {option_to_text(opt)} can never fire, because "
                    "transitions are only evaluated when at least one input spikes.")
    reach = reachable_states(s)
    for st in s["states"]:
        if st["id"] not in reach:
            warnings.append(f"State {st['name']} is not reachable from the initial state.")
    return errors, warnings


def _join(xs, word) -> str:
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + f" {word} " + xs[-1]


def notes(spec: dict) -> list[str]:
    """Informative remarks on how the network will interpret the machine.

    Transitions are evaluated only on input events (time steps in which at least
    one input spikes, which is when OP fires). A term without positive literals,
    such as ``¬A ∧ ¬B``, therefore fires on every event in which A and B stay
    silent, that is, whenever some *other* input spikes.
    """
    s = normalise(spec)
    names = {st["id"]: st["name"] for st in s["states"]}
    out = []
    for t in s["transitions"]:
        if t["from"] not in names or t["to"] not in names:
            continue
        label = f"{names[t['from']]} → {names[t['to']]}"
        for opt in t["options"]:
            if any(v == ON for v in opt.values()):
                continue
            others = [i for i in s["inputs"] if i not in opt]
            if not others:
                continue                      # reported as a warning by validate()
            if not opt:
                out.append(f"{label}: the condition is always true, so it fires on every "
                           "input event (whenever any input spikes).")
            else:
                negs = [i for i in s["inputs"] if i in opt]
                out.append(
                    f"{label}: {option_to_text(opt)} fires on any input event in which "
                    f"{_join(negs, 'and')} "
                    f"{'do' if len(negs) > 1 else 'does'} not spike, i.e. when "
                    f"{_join(others, 'or')} {'spike' if len(others) > 1 else 'spikes'}. "
                    "Without any input spike, no transition is evaluated.")
    # several transitions from the same state may hold at once: all destinations activate.
    # Only reported when a single input spike triggers both: transitions that need
    # different inputs (e.g. A and B) would only collide if both spiked in the same step.
    for st in s["states"]:
        outs = [t for t in s["transitions"] if t["from"] == st["id"] and t["to"] in names]
        for a in range(len(outs)):
            for b in range(a + 1, len(outs)):
                if outs[a]["to"] == outs[b]["to"]:
                    continue
                shared = _shared_triggers(outs[a]["options"], outs[b]["options"], s["inputs"])
                if shared:
                    out.append(
                        f"From {st['name']}, a spike of {_join(shared, 'or')} activates both "
                        f"transitions, to {names[outs[a]['to']]} and {names[outs[b]['to']]}: "
                        "both states would become active (non-deterministic behaviour).")
    return out


def _shared_triggers(opts_a, opts_b, inputs) -> list[str]:
    """Inputs whose spike alone (no other input spiking) satisfies both conditions."""
    return [x for x in inputs
            if any(option_holds(o, {x}) for o in opts_a) and any(option_holds(o, {x}) for o in opts_b)]


def reachable_states(spec: dict) -> set[str]:
    s = normalise(spec)
    seen = {st["id"] for st in s["states"] if st["initial"]}
    frontier = list(seen)
    while frontier:
        cur = frontier.pop()
        for t in s["transitions"]:
            if t["from"] == cur and t["to"] not in seen:
                seen.add(t["to"])
                frontier.append(t["to"])
    return seen


NOT, AND, OR = "¬", "∧", "∨"


def option_to_text(opt: dict) -> str:
    """One term as a conjunction of literals, e.g. ``C ∧ ¬B``."""
    if not opt:
        return "any event"
    return f" {AND} ".join(i if v == ON else NOT + i for i, v in opt.items())



def option_holds(opt: dict, event: set[str]) -> bool:
    for i, v in opt.items():
        if v == ON and i not in event:
            return False
        if v == OFF and i in event:
            return False
    return True


def transition_holds(t: dict, event: set[str]) -> bool:
    return any(option_holds(o, event) for o in t["options"])


# --------------------------------------------------------------------------
# Templates
# --------------------------------------------------------------------------

def _circle_layout(n, cx=420, cy=270, r=None):
    r = r or max(150, min(230, 40 * n))
    pos = []
    for k in range(n):
        a = -math.pi / 2 + 2 * math.pi * k / max(n, 1)
        pos.append((round(cx + r * math.cos(a)), round(cy + r * math.sin(a))))
    return pos


def _line_layout(n, x0=110, y=270, dx=None):
    dx = dx or max(110, min(170, 900 // max(n, 1)))
    return [(x0 + k * dx, y + (40 if k % 2 else -40) * (n > 6)) for k in range(n)]


def template_simple() -> dict:
    """Simple deterministic FSM of the manuscript (Sec. 3.2, Fig. 2)."""
    pos = _circle_layout(3, r=160)
    states = [dict(id=f"s{k}", name=f"S{k}", x=pos[k][0], y=pos[k][1],
                   initial=(k == 0), accepting=False) for k in range(3)]
    tr = [
        dict(id="t0", **{"from": "s0", "to": "s1"}, options=[{"SGf": ON}]),
        dict(id="t1", **{"from": "s1", "to": "s2"}, options=[{"SGf": ON}]),
        dict(id="t2", **{"from": "s2", "to": "s0"}, options=[{"SGf": ON}]),
        dict(id="t3", **{"from": "s2", "to": "s1"}, options=[{"SGb": ON}]),
    ]
    return normalise(dict(name="Simple FSM of the paper", inputs=["SGf", "SGb"],
                          states=states, transitions=tr))


def template_detector() -> dict:
    """Pattern detector of the manuscript: A -> (C+D)·B' -> B·C (Sec. 3.3)."""
    pos = _line_layout(4, x0=130, dx=200)
    names = ["S0", "S1", "S2", "Detected"]
    states = []
    for k in range(4):
        states.append(dict(id=f"s{k}", name=names[k], x=pos[k][0], y=pos[k][1],
                           initial=(k == 0), accepting=(k == 3)))
    tr = [
        dict(id="t0", **{"from": "s0", "to": "s1"}, options=[{"A": ON}]),
        dict(id="t1", **{"from": "s1", "to": "s2"},
             options=[{"C": ON, "B": OFF}, {"D": ON, "B": OFF}]),
        dict(id="t2", **{"from": "s2", "to": "s3"}, options=[{"B": ON, "C": ON}]),
    ]
    return normalise(dict(name="Pattern detector of the paper",
                          inputs=["A", "B", "C", "D"], states=states, transitions=tr))


def template_counter(n: int = 4) -> dict:
    """n-state cyclic counter driven by a single 'Step' input."""
    n = max(2, int(n))
    pos = _circle_layout(n)
    states = [dict(id=f"s{k}", name=f"C{k}", x=pos[k][0], y=pos[k][1],
                   initial=(k == 0), accepting=False) for k in range(n)]
    tr = [dict(id=f"t{k}", **{"from": f"s{k}", "to": f"s{(k + 1) % n}"},
               options=[{"Step": ON}]) for k in range(n)]
    return normalise(dict(name=f"{n}-state cyclic counter", inputs=["Step"],
                          states=states, transitions=tr))


def template_sequence(sequence: str = "ABBA") -> dict:
    """Detector of a sequence of events over inputs A and B, e.g. 'ABBA'.

    Each symbol means "this input spikes and the other one does not".
    The detector has len(sequence)+1 states; the last one is the output.
    """
    seq = re.sub(r"[^AB]", "", str(sequence).upper()) or "AB"
    n = len(seq)
    pos = _line_layout(n + 1)
    states = []
    for k in range(n + 1):
        states.append(dict(id=f"s{k}", name=("Start" if k == 0 else
                                            "Detected" if k == n else f"P{k}"),
                           x=pos[k][0], y=pos[k][1], initial=(k == 0),
                           accepting=(k == n)))
    other = {"A": "B", "B": "A"}
    tr = [dict(id=f"t{k}", **{"from": f"s{k}", "to": f"s{k + 1}"},
               options=[{c: ON, other[c]: OFF}]) for k, c in enumerate(seq)]
    return normalise(dict(name=f"Detector of sequence {seq}", inputs=["A", "B"],
                          states=states, transitions=tr))


