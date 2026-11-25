"""Software reference automaton used to validate the spiking implementation.

It processes the same input events as the network, with the same semantics
(see spec.py), and returns the expected active states and detections.
"""

from __future__ import annotations

from .spec import normalise, transition_holds


def run_reference(spec: dict, events: list[tuple[int, set]]) -> dict:
    """``events``: list of (step, set of input names), sorted by step.

    Returns a dict with:
      * ``active_before``: for each event, the set of active state ids just
        before the event is processed (what the network samples);
      * ``active_after``: the set after the event;
      * ``detections``: steps of the events that complete a pattern
        (fire a transition into an accepting state);
      * ``fired``: list of fired transition ids per event.
    """
    s = normalise(spec)
    st = {x["id"]: x for x in s["states"]}
    trans = [t for t in s["transitions"] if t["from"] in st and t["to"] in st]
    active = {x["id"] for x in s["states"] if x["initial"]}
    before, after, detections, fired_all = [], [], [], []
    for step, ev in events:
        ev = set(ev)
        before.append(sorted(active))
        fired = [t for t in trans if t["from"] in active and transition_holds(t, ev)]
        # a state stays active unless a transition leaves it; destinations become active
        left = {t["from"] for t in fired if t["to"] != t["from"]}
        new = (active - left) | {t["to"] for t in fired}
        if any(st[t["to"]]["accepting"] for t in fired):
            detections.append(step)
        fired_all.append([t["id"] for t in fired])
        active = new
        after.append(sorted(active))
    return dict(active_before=before, active_after=after,
                detections=detections, fired=fired_all)
