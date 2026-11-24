"""Generation of input spike trains.

As in the manuscript, inputs are spike trains with random inter-spike
intervals.  Spikes of different inputs that share a time step form one input
*event* (e.g. "B and C spike together").  Three modes are offered:

* ``random``: event times with random intervals in [min_gap, max_gap]; each
  input takes part in each event with probability ``p_input`` (at least one
  input per event).
* ``manual``: spike times typed by the user for each input (in steps or ms).
* ``mixed``: a random sequence with the typed spikes inserted on top of it.
  Random events closer than min_gap steps to a typed one are dropped.

The spacing is the user's choice: the pipelined network only guarantees events
from a minimum spacing on (the analysis reports closer ones), the parallel one
accepts any spacing.
"""

from __future__ import annotations

import random
import re

from .builder import FIRST_EVENT_STEP
from .spec import normalise


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


def generate(spec: dict, mode: str = "random", n_events: int = 60,
             min_gap: int = 5, max_gap: int = 15, p_input: float = 0.4,
             seed: int = 1, manual: dict | None = None,
             manual_unit_ms: bool = True, dt: float = 1.0) -> dict:
    """Return dict(trains={input: [steps]}, events=[(step, set)])."""
    s = normalise(spec)
    inputs = s["inputs"]
    rng = random.Random(seed)
    min_gap = max(int(min_gap), 1)
    max_gap = max(int(max_gap), min_gap)

    if mode == "manual":
        trains = _manual_trains(inputs, manual, dt, manual_unit_ms)
        return dict(trains=trains, events=events_from_trains(trains))

    events, step = [], FIRST_EVENT_STEP
    for _ in range(int(n_events)):
        events.append((step, _random_event(inputs, rng, p_input)))
        step += rng.randint(min_gap, max_gap)
    if mode != "mixed":
        return dict(trains=trains_from_events(inputs, events), events=events)

    # mixed: the typed spikes are kept as they are; random events that fall less
    # than min_gap steps from one of them are dropped so they do not collide
    fixed = events_from_trains(_manual_trains(inputs, manual, dt, manual_unit_ms))
    fixed_steps = [t for t, _ in fixed]
    kept = [(t, ev) for t, ev in events
            if all(abs(t - m) >= min_gap for m in fixed_steps)]
    events = sorted(kept + fixed, key=lambda e: e[0])
    return dict(trains=trains_from_events(inputs, events), events=events,
                manual_steps=fixed_steps, dropped=int(n_events) - len(kept))


def _manual_trains(inputs, manual, dt, unit_ms=True) -> dict[str, list[int]]:
    """Spike times typed by the user (ms, or steps) -> {input: [steps]}."""
    trains = {}
    for name in inputs:
        txt = (manual or {}).get(name, "")
        vals = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", str(txt))]
        steps = sorted({int(round(v / dt)) if unit_ms else int(v) for v in vals})
        trains[name] = [x for x in steps if x >= 1]
    return trains


def check_gaps(events, min_gap: int) -> list[int]:
    """Steps of events that come closer than ``min_gap`` steps to the previous one."""
    bad = []
    for (a, _), (b, _) in zip(events, events[1:]):
        if b - a < min_gap:
            bad.append(b)
    return bad
