"""Generation of input spike trains.

As in the manuscript, inputs are spike trains with random inter-spike
intervals.  Spikes of different inputs that share a time step form one input
*event* (e.g. "B and C spike together").  Three modes are offered:

* ``random``: event times with random intervals in [min_gap, max_gap]; each
  input takes part in each event with probability ``p_input`` (at least one
  input per event).
* ``planted``: like ``random``, plus a number of complete target patterns
  (sequences of events that drive the machine into an accepting state),
  inserted at random positions.
* ``manual``: spike times typed by the user for each input (in steps or ms).
"""

from __future__ import annotations

import random
import re

from .builder import FIRST_EVENT_STEP, MIN_EVENT_GAP
from .spec import ON, OFF, normalise


def events_from_trains(trains: dict[str, list[int]]) -> list[tuple[int, set]]:
    by_step = {}
    for name, steps in trains.items():
        for t in steps:
            by_step.setdefault(int(t), set()).add(name)
    return sorted(by_step.items())


def trains_from_events(inputs, events) -> dict[str, list[int]]:
    trains = {i: [] for i in inputs}
    for step, ev in events:
        for i in ev:
            trains[i].append(int(step))
    return trains


def _random_event(inputs, rng, p):
    ev = {i for i in inputs if rng.random() < p}
    if not ev:
        ev = {rng.choice(inputs)}
    return ev


def _event_for_option(inputs, opt, rng, p):
    """An input event that satisfies a transition option."""
    ev = {i for i, v in opt.items() if v == ON}
    for i in inputs:
        if i not in opt and rng.random() < p * 0.5:
            ev.add(i)
    if not ev:  # option made only of prohibitions: pick an allowed input
        allowed = [i for i in inputs if opt.get(i) != OFF]
        if allowed:
            ev.add(rng.choice(allowed))
    return ev


def accepting_paths(spec: dict, max_len: int = 12) -> list[list[tuple[str, int]]]:
    """Shortest transition paths from an initial state to each accepting state."""
    s = normalise(spec)
    acc = {x["id"] for x in s["states"] if x["accepting"]}
    start = [x["id"] for x in s["states"] if x["initial"]]
    paths, seen = [], set(start)
    frontier = [(sid, []) for sid in start]
    while frontier:
        nxt = []
        for sid, path in frontier:
            for t in s["transitions"]:
                if t["from"] != sid or len(path) >= max_len:
                    continue
                for k in range(len(t["options"])):
                    p2 = path + [(t["id"], k)]
                    if t["to"] in acc:
                        paths.append(p2)
                    if t["to"] not in seen:
                        seen.add(t["to"])
                        nxt.append((t["to"], p2))
        frontier = nxt
    return paths


def generate(spec: dict, mode: str = "random", n_events: int = 60,
             min_gap: int = MIN_EVENT_GAP, max_gap: int = 15, p_input: float = 0.4,
             planted: int = 3, seed: int = 1, manual: dict | None = None,
             manual_unit_ms: bool = True, dt: float = 1.0) -> dict:
    """Return dict(trains={input: [steps]}, events=[(step, set)], planted=[steps])."""
    s = normalise(spec)
    inputs = s["inputs"]
    rng = random.Random(seed)
    min_gap = max(int(min_gap), MIN_EVENT_GAP)
    max_gap = max(int(max_gap), min_gap)
    planted_marks = []

    if mode == "manual":
        trains = {}
        for name in inputs:
            txt = (manual or {}).get(name, "")
            vals = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", str(txt))]
            steps = sorted({int(round(v / dt)) if manual_unit_ms else int(v) for v in vals})
            trains[name] = [x for x in steps if x >= 1]
        return dict(trains=trains, events=events_from_trains(trains), planted=[])

    # Build a list of event "blocks": single random events or planted patterns.
    blocks = [[_random_event(inputs, rng, p_input)] for _ in range(int(n_events))]
    if mode == "planted":
        paths = accepting_paths(s)
        tmap = {t["id"]: t for t in s["transitions"]}
        if paths:
            for _ in range(int(planted)):
                path = rng.choice(paths)
                blk = [_event_for_option(inputs, tmap[tid]["options"][k], rng, p_input)
                       for tid, k in path]
                pos = rng.randrange(len(blocks) + 1)
                blocks.insert(pos, ("planted", blk))
    events, step = [], FIRST_EVENT_STEP
    for blk in blocks:
        is_planted = isinstance(blk, tuple)
        evs = blk[1] if is_planted else blk
        if is_planted:
            planted_marks.append(step)
        for k, ev in enumerate(evs):
            events.append((step, ev))
            # planted patterns use the minimum gap so they stay contiguous
            step += min_gap if (is_planted and k < len(evs) - 1) else rng.randint(min_gap, max_gap)
    trains = trains_from_events(inputs, events)
    return dict(trains=trains, events=events, planted=planted_marks)


def check_gaps(events) -> list[int]:
    """Steps of events that come too close to the previous one."""
    bad = []
    for (a, _), (b, _) in zip(events, events[1:]):
        if b - a < MIN_EVENT_GAP:
            bad.append(b)
    return bad
