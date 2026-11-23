"""Simulation back-ends: built-in ideal simulator and PyNN (via pynn_runner)."""

from __future__ import annotations

import importlib.util

from . import pynn_runner
from .builder import INIT_STEP, Netlist
from .pynn_runner import DEFAULT_WEIGHTS

BACKENDS = {
    "ideal": "Ideal (built-in, no PyNN): discrete-time Boolean neurons",
    "spinnaker": "SpiNNaker (pyNN.spiNNaker / sPyNNaker)",
    "nest": "NEST (pyNN.nest)",
    "brian2": "Brian 2 (pyNN.brian2)",
}


def available_backends() -> list[dict]:
    out = []
    for key, label in BACKENDS.items():
        if key == "ideal":
            ok, why = True, ""
        else:
            mod = pynn_runner.BACKEND_MODULES[key]
            try:
                ok = importlib.util.find_spec("pyNN") is not None and \
                    importlib.util.find_spec(mod) is not None
            except (ImportError, ValueError):
                ok = False
            why = "" if ok else f"{mod} is not installed on this computer"
        out.append(dict(id=key, label=label, available=ok, reason=why))
    return out


def simulate_ideal(net: Netlist | dict, trains: dict, duration_steps: int | None = None,
                   weights: dict | None = None) -> dict:
    """Discrete-time Boolean simulation.

    A neuron fires at step t when the excitatory input arriving at t exceeds
    the inhibitory input arriving at t by at least half an excitatory weight.
    With the parameters of Table 1 (tau_m = tau_syn = 0.1 ms) and a 1 ms time
    step, the LIF neurons forget their state within one step, so this is the
    behaviour the gates are designed for.
    """
    d = net.to_dict() if isinstance(net, Netlist) else net
    w = dict(DEFAULT_WEIGHTS)
    w.update(weights or {})
    theta = 0.5 * w["exc"]
    n = len(d["neurons"])
    out_src, out_n = {}, {}
    for pre, post, rec, delay in d["conns"]:
        tgt = out_src if pre[0] == "src" else out_n
        tgt.setdefault(int(pre[1]), []).append((post, rec, delay))
    src_spikes = {}
    for k, src in enumerate(d["sources"]):
        steps = trains.get(src["input"], []) if src["kind"] == "input" else [INIT_STEP]
        for t in steps:
            src_spikes.setdefault(int(t), []).append(k)
    last = max(list(src_spikes) + [INIT_STEP])
    T = duration_steps if duration_steps is not None else last + 40
    exc, inh = {}, {}
    spikes = {i: [] for i in range(n)}

    def schedule(lst, t):
        for post, rec, delay in lst:
            tab = exc if rec == "exc" else inh
            row = tab.setdefault(t + delay, {})
            row[post] = row.get(post, 0.0) + (w["exc"] if rec == "exc" else w["inh"])

    for t in range(T + 1):
        e_row, i_row = exc.pop(t, {}), inh.pop(t, {})
        fired = [j for j, v in e_row.items() if v - i_row.get(j, 0.0) >= theta]
        for j in fired:
            spikes[j].append(t)
        for k in src_spikes.get(t, []):
            schedule(out_src.get(k, []), t)
        for j in fired:
            schedule(out_n.get(j, []), t)
    return spikes


def simulate(net: Netlist, trains: dict, backend: str = "ideal", dt: float = 1.0,
             duration_steps: int | None = None, neuron_params: dict | None = None,
             weights: dict | None = None, seed: int | None = None) -> dict:
    if backend == "ideal":
        return simulate_ideal(net, trains, duration_steps, weights)
    d = net.to_dict()
    d["init_step"] = INIT_STEP
    return pynn_runner.run_pynn(d, trains, backend=backend, dt=dt,
                                duration_steps=duration_steps,
                                neuron_params=neuron_params, weights=weights, seed=seed)
