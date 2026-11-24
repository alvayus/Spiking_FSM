"""Simulation back-ends: built-in ideal simulator and PyNN (via pynn_runner)."""

from __future__ import annotations

import functools
import importlib
import importlib.util
import os
import socket
import subprocess
import sys

from . import pynn_runner
from .builder import INIT_STEP, Netlist
from .pynn_runner import DEFAULT_WEIGHTS

BACKENDS = {
    "ideal": "Ideal (built-in, no PyNN): discrete-time Boolean neurons",
    "spinnaker": "SpiNNaker (pyNN.spiNNaker / sPyNNaker)",
}


@functools.lru_cache(maxsize=None)
def _check_backend(key: str) -> tuple[bool, str]:
    """Import the PyNN module of a back-end once: only a real import tells
    whether the simulator can be used (e.g. sPyNNaker installed but broken)."""
    mod = pynn_runner.BACKEND_MODULES[key]
    if importlib.util.find_spec("pyNN") is None:
        return False, f"PyNN is not installed in this Python ({sys.executable})"
    try:
        importlib.import_module(mod)
        return True, ""
    except Exception as e:                       # noqa: BLE001 - any failure means unusable
        msg = str(e).splitlines()[0] if str(e) else type(e).__name__
        return False, f"{mod} cannot be imported: {msg[:120]}"


class SpiNNakerUnavailable(RuntimeError):
    """The SpiNNaker machine configured for sPyNNaker cannot be reached."""


def _spynnaker_machine() -> dict:
    """[Machine] section of the user's sPyNNaker configuration (home, then cwd)."""
    import configparser
    cfg = configparser.ConfigParser()
    cfg.read([os.path.expanduser("~/.spynnaker.cfg"), ".spynnaker.cfg"], encoding="utf-8")
    sec = cfg["Machine"] if cfg.has_section("Machine") else {}
    val = lambda k: (sec.get(k) or "").strip()
    none = lambda v: v if v and v.lower() != "none" else ""
    return dict(host=none(val("machineName")), spalloc=none(val("spalloc_server")),
                spalloc_port=int(none(val("spalloc_port")) or 22244),
                virtual=val("virtual_board").lower() == "true")


def spinnaker_reachable(timeout_s: float = 1.0) -> tuple[bool, str]:
    """Quick check before a SpiNNaker run, which otherwise takes about 40 s of
    retries to fail when the board is off or not connected."""
    m = _spynnaker_machine()
    if m["virtual"]:
        return True, ""
    if m["spalloc"]:
        try:
            with socket.create_connection((m["spalloc"], m["spalloc_port"]), timeout=timeout_s):
                return True, ""
        except OSError:
            return False, (f"Cannot reach the spalloc server {m['spalloc']}:{m['spalloc_port']} "
                           "set in .spynnaker.cfg.")
    if not m["host"]:
        return False, ("No SpiNNaker machine is configured: set machineName in the [Machine] "
                       "section of ~/.spynnaker.cfg.")
    # SpiNNaker boards answer ICMP echo; a reply line always contains "TTL="
    win = os.name == "nt"
    cmd = ["ping", "-n" if win else "-c", "1", "-w" if win else "-W",
           str(int(timeout_s * 1000)) if win else str(max(1, int(timeout_s))), m["host"]]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s + 4,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if out.returncode == 0 and "ttl=" in out.stdout.lower():
            return True, ""
    except (OSError, subprocess.SubprocessError):
        return True, ""          # cannot ping from here: let sPyNNaker try
    return False, (f"The SpiNNaker board at {m['host']} (machineName in ~/.spynnaker.cfg) does not "
                   "answer. Check that it is switched on and connected to this computer.")


def available_backends() -> list[dict]:
    out = []
    for key, label in BACKENDS.items():
        ok, why = (True, "") if key == "ideal" else _check_backend(key)
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
             weights: dict | None = None) -> dict:
    if backend == "ideal":
        return simulate_ideal(net, trains, duration_steps, weights)
    if backend == "spinnaker":
        ok, why = spinnaker_reachable()
        if not ok:
            raise SpiNNakerUnavailable(why)
    d = net.to_dict()
    d["init_step"] = INIT_STEP
    return pynn_runner.run_pynn(d, trains, backend=backend, dt=dt,
                                duration_steps=duration_steps,
                                neuron_params=neuron_params, weights=weights)
