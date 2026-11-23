"""Minimal stand-in for the PyNN API used by pynn_runner (tests only).
It records the calls and simulates with the Boolean rule, so the whole PyNN
code path (populations, projections, delays, receptors, get_data) is exercised."""
_state = {}

class _Celltype:
    def __init__(self, **kw): self.params = kw

class IF_curr_exp(_Celltype):
    REQUIRED = {"cm","tau_m","tau_refrac","tau_syn_E","tau_syn_I","v_rest","v_reset","v_thresh"}
    def __init__(self, **kw):
        unknown = set(kw) - self.REQUIRED - {"i_offset"}
        assert not unknown, unknown
        super().__init__(**kw)

class SpikeSourceArray(_Celltype):
    def __init__(self, spike_times): self.spike_times = spike_times

class StaticSynapse:
    def __init__(self, **kw): pass

class FromListConnector:
    def __init__(self, conn_list): self.conn_list = conn_list

class _Train(list):
    def __init__(self, times, idx):
        super().__init__(times); self.annotations = {"source_index": idx}
    @property
    def magnitude(self): return list(self)

class _Seg:
    def __init__(self, trains): self.spiketrains = trains

class _Block:
    def __init__(self, trains): self.segments = [_Seg(trains)]

class Population:
    def __init__(self, size, celltype, label=None):
        self.size, self.celltype, self.label = size, celltype, label
        self.recorded = False
        _state["pops"].append(self)
    def record(self, what):
        assert what == "spikes"; self.recorded = True
    def get_data(self, what):
        sp = _state["result"].get(id(self), {})
        return _Block([_Train([t * _state["dt"] for t in sp.get(i, [])], i) for i in range(self.size)])

class Projection:
    def __init__(self, pre, post, connector, synapse_type=None, receptor_type="excitatory"):
        assert receptor_type in ("excitatory", "inhibitory")
        for (i, j, w, d) in connector.conn_list:
            assert w > 0, "weights must be positive magnitudes"
            assert abs(d / _state["dt"] - round(d / _state["dt"])) < 1e-9 and d >= _state["dt"]
        _state["projs"].append((pre, post, connector, receptor_type))

def setup(timestep=1.0, min_delay=1.0, **kw):
    _state.clear(); _state.update(dt=timestep, pops=[], projs=[], result={})

def run(duration):
    dt = _state["dt"]; steps = int(round(duration / dt))
    pops = [p for p in _state["pops"] if isinstance(p.celltype, IF_curr_exp)]
    srcs = [p for p in _state["pops"] if isinstance(p.celltype, SpikeSourceArray)]
    out = {}
    for pr, po, c, r in _state["projs"]:
        for i, j, w, d in c.conn_list:
            out.setdefault((id(pr), i), []).append((id(po), j, w if r == "excitatory" else -w, int(round(d / dt))))
    arrivals = {}
    def sched(key, t):
        for po, j, w, d in out.get(key, []):
            arrivals.setdefault(t + d, {}).setdefault((po, j), 0.0)
            arrivals[t + d][(po, j)] += w
    src_sp = {}
    for s in srcs:
        for i, times in enumerate(s.celltype.spike_times):
            for t in times: src_sp.setdefault(int(round(t / dt)), []).append((id(s), i))
    w_exc = min(w for lst in out.values() for (_, _, w, _) in lst if w > 0)
    res = {id(p): {} for p in pops}
    for t in range(steps + 1):
        fired = [k for k, v in arrivals.pop(t, {}).items() if v >= 0.5 * w_exc]
        for (po, j) in fired: res[po].setdefault(j, []).append(t)
        for key in src_sp.get(t, []): sched(key, t)
        for key in fired: sched(key, t)
    _state["result"] = res

def end(): pass
