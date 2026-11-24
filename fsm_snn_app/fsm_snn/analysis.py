"""Decoding of simulation results and automatic comparison with the reference."""

from __future__ import annotations

from .builder import (INIT_STEP, OUTPUT_LATENCY, STATE_LATENCY, T_TERM, Netlist,
                      architecture, min_event_gap)
from .inputs import check_gaps
from .reference import run_reference
from .spec import normalise


def _state_spikes(net: Netlist, spikes: dict, sid: str) -> list[int]:
    _, a, b = net.latches[sid]
    return sorted(set(spikes.get(a, [])) | set(spikes.get(b, [])))


def _expected_state_steps(s, events, ref, end_step):
    """Expected ring activity (one spike per step while active) for each state."""
    st = {x["id"]: x for x in s["states"]}
    active_from = {sid: INIT_STEP + 2 for sid in st if st[sid]["initial"]}
    steps = {sid: [] for sid in st}

    def close(sid, last):
        start = active_from.pop(sid, None)
        if start is not None and last >= start:
            steps[sid].extend(range(start, last + 1))

    tmap = {t["id"]: t for t in s["transitions"]}
    for (e, _), before, after, fired in zip(events, ref["active_before"],
                                            ref["active_after"], ref["fired"]):
        b, a = set(before), set(after)
        left = {tmap[t]["from"] for t in fired if tmap[t]["to"] != tmap[t]["from"]}
        for sid in b:
            x = st[sid]
            if x["always_active"]:
                continue
            if x["on_mismatch"] == "discard":
                close(sid, e + 2)
                if sid in a:
                    active_from[sid] = e + STATE_LATENCY
            elif sid not in a or sid in left:
                close(sid, e + 3)
                if sid in a:
                    active_from[sid] = e + STATE_LATENCY
        for sid in a - b:
            active_from.setdefault(sid, e + STATE_LATENCY)
    for sid in list(active_from):
        close(sid, end_step)
    return steps


def _fsm_intervals(s, events, ref, end_step):
    """States of the abstract machine, without synaptic delays: it starts in its
    initial states and changes state at the very step of each input event.
    Returns {state id: [[first step, end step (excluded)], ...]}."""
    out = {x["id"]: [] for x in s["states"]}
    since = {x["id"]: 0 for x in s["states"] if x["initial"]}
    for (e, _), after in zip(events, ref["active_after"]):
        a = set(after)
        for sid in [x for x in since if x not in a]:
            if e > since[sid]:
                out[sid].append([since[sid], e])
            del since[sid]
        for sid in a:
            since.setdefault(sid, e)
    for sid, t in since.items():
        if end_step > t:
            out[sid].append([t, end_step])
    return out


def analyse(spec: dict, net: Netlist, spikes: dict, events: list, trains: dict,
            duration_steps: int, tolerance: int = 1, dt: float = 1.0) -> dict:
    s = normalise(spec)
    names = {x["id"]: x["name"] for x in s["states"]}
    ref = run_reference(s, events)
    warnings = []

    # ---- calibrate a constant offset of the back-end (ideal: 0) ---------------
    offset = 0
    init_states = [x["id"] for x in s["states"] if x["initial"]]
    if init_states:
        first = _state_spikes(net, spikes, init_states[0])
        if first:
            offset = first[0] - (INIT_STEP + 2)
        else:
            warnings.append("The initial state never became active: check the weights and neuron parameters.")
    if abs(offset) > 3:
        warnings.append(f"Unusual simulator time offset ({offset} steps).")

    # ---- state agreement at every event ----------------------------------------
    # The transitions that the network fires for each event (its term neurons, at
    # e + T_TERM) give the states it is in, whatever the spacing of the events: with
    # the bypasses, a state that lasts a single event may never reach its latch ring.
    state_spk = {sid: set(_state_spikes(net, spikes, sid)) for sid in names}
    tmap = {t["id"]: t for t in s["transitions"]}
    term_spk = {}
    for tid, _, n in net.terms:
        for t in spikes.get(n, []):
            term_spk.setdefault(t, set()).add(tid)
    active = {x["id"] for x in s["states"] if x["initial"]}
    per_event = []
    agree = 0
    for (e, ev), exp in zip(events, ref["active_before"]):
        got = sorted(active)
        ok = got == exp
        agree += ok
        per_event.append(dict(step=e, inputs=sorted(ev), expected=[names[x] for x in exp],
                              obtained=[names[x] for x in got], ok=ok))
        fired = [tmap[t] for t in term_spk.get(e + T_TERM + offset, ()) if t in tmap]
        left = {t["from"] for t in fired if t["to"] != t["from"]}
        active = (active - left) | {t["to"] for t in fired}
    final_ok = None
    if events:
        t_final = events[-1][0] + STATE_LATENCY + 1 + offset
        if t_final < duration_steps:
            got = sorted(sid for sid in names if t_final in state_spk[sid])
            final_ok = got == ref["active_after"][-1]

    # ---- detections --------------------------------------------------------------
    out_spk = sorted(spikes.get(net.output, [])) if net.output is not None else []
    # the simulation records steps 0 .. duration_steps-1: later detections cannot be evaluated
    expected_out = [e + OUTPUT_LATENCY + offset for e in ref["detections"]]
    ref_det = [e for e, t in zip(ref["detections"], expected_out) if t < duration_steps]
    expected_out = [t for t in expected_out if t < duration_steps]
    unmatched = list(out_spk)
    tp, latencies, fn_steps = 0, [], []
    for e, t_exp in zip(ref_det, expected_out):
        hit = next((t for t in unmatched if abs(t - t_exp) <= tolerance), None)
        if hit is None:
            fn_steps.append(e)
        else:
            unmatched.remove(hit)
            tp += 1
            latencies.append(hit - e)
    fp, fn = len(unmatched), len(fn_steps)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None

    ms = lambda steps: f"{steps * dt:g} ms"
    early = [e for e, _ in events if e <= INIT_STEP + 2]
    if early:
        warnings.append(f"{len(early)} input spike time(s) arrive before the initial states are set "
                        f"(step {INIT_STEP + 2}, {ms(INIT_STEP + 2)} with a {dt:g} ms time step); "
                        "they are not evaluated correctly.")
    gap = min_event_gap(s)
    gaps = check_gaps(events, gap)
    if gaps:
        warnings.append(f"{len(gaps)} input spike times arrive less than {gap} steps "
                        f"({ms(gap)} with a {dt:g} ms time step) after the previous one; "
                        f"the {architecture(s)} network only guarantees input spikes from that spacing on "
                        "(the parallel implementation accepts any spacing).")

    # ---- raster for the UI --------------------------------------------------------
    rows = []
    for name in s["inputs"]:
        rows.append(dict(label=name, kind="input", spikes=sorted(trains.get(name, []))))
    if net.op is not None:
        rows.append(dict(label="OP", kind="op", spikes=sorted(spikes.get(net.op, []))))
    exp_states = _expected_state_steps(s, events, ref, duration_steps)
    # fsm: the abstract machine (no delays), drawn in its own plot ("Show expected");
    # the comparison (expected=) uses the reference shifted by the network latency
    fsm = _fsm_intervals(s, [ev for ev in events if ev[0] < duration_steps], ref, duration_steps)
    for x in s["states"]:
        rows.append(dict(label=x["name"], kind="state", spikes=sorted(state_spk[x["id"]]),
                         expected=[t + offset for t in exp_states[x["id"]] if t + offset < duration_steps],
                         fsm=fsm[x["id"]]))
    if net.output is not None:
        rows.append(dict(label="Output", kind="output", spikes=out_spk,
                         expected=expected_out))
    # idx: position in the netlist, so the page can name each neuron as the Network tab does
    all_rows = [dict(label=n["label"], kind=n["kind"], idx=i, spikes=sorted(spikes.get(i, [])))
                for i, n in enumerate(net.neurons)]

    n_ev = len(events)
    metrics = dict(
        events=n_ev, state_agreement=(agree / n_ev if n_ev else None),
        final_state_ok=final_ok, expected_detections=len(ref_det),
        tp=tp, fp=fp, fn=fn, precision=precision, recall=recall,
        mean_latency=(sum(latencies) / len(latencies) if latencies else None),
        offset=offset, total_spikes=sum(len(v) for v in spikes.values()),
    )
    return dict(metrics=metrics, rows=rows, all_rows=all_rows, per_event=per_event,
                warnings=warnings, reference=ref)

