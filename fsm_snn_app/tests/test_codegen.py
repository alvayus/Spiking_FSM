"""The readable PyNN code (Network tab) must build exactly the simulated network."""
import contextlib, io, os, sys
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "stub_pynn"))
from fsm_snn import spec as S
from fsm_snn.builder import INIT_STEP, build
from fsm_snn.codegen import generate_code
from fsm_snn.pipeline import _sim_defaults, make_inputs
from fsm_snn.pynn_runner import run_pynn


def check(sp, params):
    r = generate_code(sp, params)
    assert r["ok"], r
    ns = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(r["text"], r["filename"], "exec"), ns)       # runs on the PyNN stand-in
    s = S.normalise(sp)
    net = build(s).to_dict()
    net["init_step"] = INIT_STEP
    p = _sim_defaults(params)
    inp = make_inputs(s, p)
    last = max([e for e, _ in inp["events"]] + [INIT_STEP])
    spikes = run_pynn(net, inp["trains"], backend="spinnaker", duration_steps=last + 30)
    for st in s["states"]:
        _, a, b = net["latches"][st["id"]]
        expected = sorted(set(spikes.get(a, [])) | set(spikes.get(b, [])))
        assert ns["active_steps"][st["name"]] == expected, st["name"]


def test_code_matches_runner():
    for sp in [S.template_simple(), S.template_detector(), S.template_counter(4), S.template_sequence("ABBA")]:
        check(sp, dict(mode="random", n_events=40, seed=3))


if __name__ == "__main__":
    test_code_matches_runner(); print("ok")
