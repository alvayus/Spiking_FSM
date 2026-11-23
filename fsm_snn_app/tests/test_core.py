"""Core tests: the generated SNN (ideal back-end) must match the reference
automaton on every template, for random and planted inputs."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fsm_snn import spec as S  # noqa: E402
from fsm_snn.analysis import analyse  # noqa: E402
from fsm_snn.backends import simulate  # noqa: E402
from fsm_snn.builder import build, summary  # noqa: E402
from fsm_snn.inputs import generate  # noqa: E402


def run(spec, **kw):
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


def test_templates_random_and_planted():
    specs = [S.template_simple(), S.template_detector(), S.template_counter(5),
             S.template_sequence("ABBA"), S.template_sequence("AAB")]
    for sp in specs:
        errs, _ = S.validate(sp)
        assert not errs, errs
        for seed in range(5):
            check(run(sp, mode="random", n_events=80, seed=seed, max_gap=9, p_input=0.5))
            m = check(run(sp, mode="planted", n_events=60, planted=5, seed=seed, p_input=0.4))
            if any(x["accepting"] for x in sp["states"]):
                assert m["expected_detections"] >= 1


def test_min_gap_events_back_to_back():
    sp = S.template_detector()
    check(run(sp, mode="random", n_events=200, min_gap=5, max_gap=5, seed=3, p_input=0.5))


def test_counts_paper_simple():
    c = summary(S.template_simple())
    assert c["states"] == 3 and c["neurons"] > 0
    # 3 latches (9) + OP + delay + NOT SGf + NOT SGb + 3 NOT states + 4 terms
    assert c["neurons"] == 9 + 1 + 1 + 2 + 3 + 4


def test_manual_inputs():
    sp = S.template_counter(3)
    res = run(sp, mode="manual", manual={"Step": "10, 20, 30, 40"})
    m = check(res)
    assert m["events"] == 4


def test_scaling_linear():
    prev = None
    for n in range(2, 11):
        c = summary(S.template_counter(n))
        if prev:
            assert c["neurons"] - prev["neurons"] == 5  # 3 latch + NOT state + term
        prev = c


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
