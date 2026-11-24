"""Exercise the PyNN code path and the exported script with a stub PyNN."""
import os, subprocess, sys
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "stub_pynn"))
from fsm_snn import spec as S
from fsm_snn.pipeline import run, export_script
from fsm_snn import backends
from fsm_snn.backends import available_backends

# the stub stands in for a board: skip the reachability check of the real one
backends.spinnaker_reachable = lambda *a, **k: (True, "")

def test_backends_listed():
    av = {b["id"]: b["available"] for b in available_backends()}
    assert av["ideal"] and av["spinnaker"] and set(av) == {"ideal", "spinnaker"}

def test_pynn_matches_ideal():
    for sp in [S.template_detector(), S.template_simple(), S.template_sequence("ABBA")]:
        p = dict(mode="random", n_events=150, seed=4, save=False)
        a = run(sp, dict(p, backend="ideal"))
        b = run(sp, dict(p, backend="spinnaker"))
        assert a["metrics"]["state_agreement"] == b["metrics"]["state_agreement"] == 1.0
        assert [r["spikes"] for r in a["all_rows"]] == [r["spikes"] for r in b["all_rows"]]

def test_exported_script_runs(tmp="/tmp/fsm_export"):
    os.makedirs(tmp, exist_ok=True)
    r = export_script(S.template_detector(), dict(mode="random", n_events=150, seed=2))
    path = os.path.join(tmp, r["filename"]); open(path, "w", encoding="utf-8").write(r["text"])
    env = dict(os.environ, PYTHONPATH=os.path.join(HERE, "stub_pynn"))
    out = subprocess.run([sys.executable, path, "--backend", "spinnaker"], cwd=tmp, env=env,
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert "FN: 0" in out.stdout and "FP: 0" in out.stdout, out.stdout
    print(out.stdout)

if __name__ == "__main__":
    test_backends_listed(); test_pynn_matches_ideal(); test_exported_script_runs(); print("ok")
