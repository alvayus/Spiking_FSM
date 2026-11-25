"""Core tests: the generated SNN (ideal back-end) must match the reference
automaton on every template, for random inputs."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fsm_snn import spec as S  # noqa: E402
from fsm_snn.analysis import analyse  # noqa: E402
from fsm_snn.backends import simulate  # noqa: E402
from fsm_snn.builder import build, summary  # noqa: E402
from fsm_snn.inputs import generate  # noqa: E402


def run(spec, arch="parallel", **kw):
    spec = dict(spec, architecture=arch)
    net = build(spec)
    inp = generate(spec, **kw)
    last = max([e for e, _ in inp["events"]] + [10])
    dur = last + 30
    spikes = simulate(net, inp["trains"], "ideal", duration_steps=dur)
    return analyse(spec, net, spikes, inp["events"], inp["trains"], dur)


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
        for arch, lo in [("pipelined", 3), ("parallel", 1)]:
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
    # 3 latches (9) + OP + delay + NOT SGf + NOT SGb + 3 NOT states + 4 terms + delay lines SGf +1, SGb +1
    assert c["neurons"] == 9 + 1 + 1 + 2 + 3 + 4 + 2
    assert c["max_delay"] == 1
    # parallel (synchronous): clock (2) + OP + delay + NOT SGf + NOT SGb + 4 terms + 3 hold
    # + 3 enter + 3 "no transition" (one per neuron of the source state, 1 each) + 3 leave
    # + "no stay" copies (one per neuron of the state: S0 1+1, S1 1+2, S2 1+1)
    assert summary(dict(S.template_simple(), architecture="parallel"))["neurons"] > c["neurons"]


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
