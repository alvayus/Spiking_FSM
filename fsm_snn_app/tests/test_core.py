"""Core tests: the generated SNN (ideal back-end) must match the reference
automaton on every template, for random inputs."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fsm_snn import spec as S  # noqa: E402
from fsm_snn.analysis import analyse  # noqa: E402
from fsm_snn.backends import simulate  # noqa: E402
from fsm_snn.builder import build, min_event_gap, summary  # noqa: E402
from fsm_snn.inputs import generate  # noqa: E402


def run(spec, arch="parallel", **kw):
    spec = dict(spec, architecture=arch)
    net = build(spec)
    inp = generate(spec, **kw)
    last = max([e for e, _ in inp["events"]] + [10])
    dur = last + 30
    spikes = simulate(net, inp["trains"], "ideal", duration_steps=dur)
    return analyse(spec, net, spikes, inp["events"], inp["trains"], dur)


def overexcited(net, trains, spikes):
    """(neuron, step) pairs in which a neuron receives two or more excitatory spikes."""
    d = net.to_dict()
    src = {k: (trains.get(x["input"], []) if x["kind"] == "input" else [d["init_step"]])
           for k, x in enumerate(d["sources"])}
    count = {}
    for pre, post, rec, delay in d["conns"]:
        if rec == "exc":
            for t in (src[pre[1]] if pre[0] == "src" else spikes.get(pre[1], [])):
                count[(post, t + delay)] = count.get((post, t + delay), 0) + 1
    return [k for k, v in count.items() if v > 1]


def check(res):
    m = res["metrics"]
    assert m["state_agreement"] == 1.0, [e for e in res["per_event"] if not e["ok"]][:3]
    assert m["final_state_ok"] in (True, None)
    assert m["fp"] == 0 and m["fn"] == 0, m
    return m


def test_templates_random():
    specs = [S.template_simple(), S.template_detector(), S.template_counter(5),
             S.template_sequence("ABBA"), S.template_sequence("AAB")]
    for sp in specs:
        errs, _ = S.validate(sp)
        assert not errs, errs
        for arch in ("pipelined", "parallel"):
            lo = min_event_gap(dict(sp, architecture=arch))
            for seed in range(5):
                check(run(sp, arch, mode="random", n_events=80, seed=seed, min_gap=lo, max_gap=9, p_input=0.5))
                m = check(run(sp, arch, mode="random", n_events=300, seed=seed, min_gap=lo, p_input=0.4))
                if any(x["accepting"] for x in sp["states"]):
                    assert m["expected_detections"] >= 1, (sp["name"], seed)


def test_min_gap_events_back_to_back():
    sp = S.template_detector()
    check(run(sp, "pipelined", mode="random", n_events=200, min_gap=3, max_gap=3, seed=3, p_input=0.5))
    check(run(sp, "parallel", mode="random", n_events=200, min_gap=1, max_gap=1, seed=3, p_input=0.5))


def test_counts_paper_simple():
    c = summary(dict(S.template_simple(), architecture="pipelined"))
    assert c["states"] == 3 and c["neurons"] > 0
    # SGf and SGb together leave S2 for S0 and S1 (non-deterministic): the OR gates that set
    # the states are inhibition-based. 3 latches (9) + 2 input neurons of the alternative OR
    # + OP + 3 S̄ + 4 AND gates + 3 NOR gates + 4 delay neurons of the transitions
    # + delay neurons SGf +2 +3, SGb +1 +2 +3, OP +1 +2 +3 (SGf +1 is its input neuron of OP)
    assert c["set_mode"] == "inhibition"
    assert c["neurons"] == 9 + 2 + 1 + 3 + 4 + 3 + 4 + 8
    # deterministic (S2 -> S0 on SGf and not SGb): Set excited directly by the AND gates
    det = S.template_simple()
    det["transitions"][2]["options"] = [{"SGf": "on", "SGb": "off"}]
    d = summary(dict(det, architecture="pipelined"))
    assert d["set_mode"] == "direct"
    # 9 + 2 + OP + 3 S̄ + 4 AND gates (excited by their input, no NOT gates of the inputs)
    # + delay neurons SGf +2 +3, SGb +1 +2 +3
    assert d["neurons"] == 9 + 2 + 1 + 3 + 4 + 5
    assert c["max_delay"] == 1
    assert summary(dict(S.template_simple(), architecture="parallel"))["neurons"] > c["neurons"]
    # SGf and SGb together leave S2 for S0 and S1: two terms can enter S1 at once, so the
    # terms out of S1 have one bypass t-1 per group of them (priority): spikes on any step
    assert min_event_gap(dict(S.template_simple(), architecture="parallel")) == 1
    assert min_event_gap(dict(S.template_detector(), architecture="parallel")) == 1


def test_no_overexcitation():
    """No neuron ever receives two excitatory spikes in the same step (Dynap-SE rule)."""
    for sp in [S.template_simple(), S.template_detector(), S.template_sequence("ABBA")]:
        for arch in ("pipelined", "parallel"):
            spec = dict(sp, architecture=arch)
            net = build(spec)
            for seed in range(4):
                inp = generate(spec, mode="random", n_events=120, seed=seed, p_input=0.6,
                               min_gap=min_event_gap(spec), max_gap=6)
                dur = inp["events"][-1][0] + 30
                spikes = simulate(net, inp["trains"], "ideal", duration_steps=dur)
                assert not overexcited(net, inp["trains"], spikes), (sp["name"], arch, seed)


def test_manual_inputs():
    sp = S.template_counter(3)
    res = run(sp, mode="manual", manual={"Step": "10, 20, 30, 40"})
    m = check(res)
    assert m["events"] == 4


def test_mixed_inputs():
    """Typed spikes are inserted on top of the random sequence and always kept;
    random events too close to them are dropped, so spacing stays valid."""
    from fsm_snn.inputs import check_gaps
    MIN_EVENT_GAP = 5                       # default minimum spacing of the random events
    sp = S.template_simple()
    for seed in range(5):
        kw = dict(n_events=40, seed=seed, p_input=0.5, manual={"SGf": "52, 120", "SGb": "52, 200"})
        inp = generate(sp, mode="mixed", **kw)
        rnd = generate(sp, mode="random", **{k: v for k, v in kw.items() if k != "manual"})
        steps = [t for t, _ in inp["events"]]
        assert {52, 120, 200} <= set(steps)
        assert dict(inp["events"])[52] == {"SGf", "SGb"}       # typed together -> one event
        assert not check_gaps(inp["events"], MIN_EVENT_GAP)
        kept = [e for e in rnd["events"] if all(abs(e[0] - m) >= MIN_EVENT_GAP for m in (52, 120, 200))]
        assert inp["dropped"] == len(rnd["events"]) - len(kept)
        assert len(inp["events"]) == len(kept) + 3
        check(run(sp, mode="mixed", **kw))


def test_cost_linear_in_states():
    for arch in ["pipelined", "parallel"]:
        prev, steps = None, set()
        for n in range(2, 11):
            c = summary(dict(S.template_counter(n), architecture=arch))
            if prev:
                steps.add(c["neurons"] - prev["neurons"])
            prev = c
        assert len(steps) == 1, (arch, steps)       # the same cost for every extra state
    assert steps                                    # pipelined: 3 latch + NOT state + term


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
