"""PyNN execution of a netlist.  Self-contained on purpose: this file's source
is also embedded in the stand-alone scripts exported by the application, so
it must not import anything from the fsm_snn package.
"""

# Neuron parameters of Table 1 of the manuscript.
DEFAULT_NEURON_PARAMS = {
    "cm": 0.1,          # nF
    "tau_m": 0.1,       # ms
    "tau_refrac": 0.0,  # ms
    "tau_syn_E": 0.1,   # ms
    "tau_syn_I": 0.1,   # ms
    "v_rest": -65.0,    # mV
    "v_reset": -65.0,   # mV
    "v_thresh": -64.91, # mV
}

# Synaptic weights (nA), all 1 as in the manuscript: the gates work in a Boolean
# way, so a spike makes the receiving neuron fire unless it is inhibited.
# Inhibitory weights are positive magnitudes applied
# through the inhibitory receptor (PyNN convention).  Replace with the values
# of the repository if they differ.
DEFAULT_WEIGHTS = {"exc": 1.0, "inh": 1.0}

BACKEND_MODULES = {
    "spinnaker": "pyNN.spiNNaker",
}


def import_backend(backend):
    import importlib
    return importlib.import_module(BACKEND_MODULES[backend])


def run_pynn(netlist, trains, backend="spinnaker", dt=1.0, duration_steps=None,
             neuron_params=None, weights=None):
    """Simulate ``netlist`` (dict form, see Netlist.to_dict) with PyNN.

    ``trains``: {input name: [steps]}.  Returns {neuron index: [steps]}.
    """
    sim = import_backend(backend)
    params = dict(DEFAULT_NEURON_PARAMS)
    params.update(neuron_params or {})
    w = dict(DEFAULT_WEIGHTS)
    w.update(weights or {})

    neurons = netlist["neurons"]
    sources = netlist["sources"]
    conns = netlist["conns"]
    init_step = netlist.get("init_step", 0)

    last = max([s for tr in trains.values() for s in tr] + [init_step])
    if duration_steps is None:
        duration_steps = last + 40

    # The network itself is deterministic: randomness only enters through the
    # input spike trains, which are generated (with a seed) before this call.
    sim.setup(timestep=dt, min_delay=dt)
    try:
        # Spike sources: one per input plus the initialisation source.
        spike_times = []
        for src in sources:
            if src["kind"] == "input":
                steps = trains.get(src["input"], [])
            else:
                steps = [init_step]
            spike_times.append([float(s) * dt for s in sorted(steps)])
        src_pop = sim.Population(len(sources), sim.SpikeSourceArray(spike_times=spike_times),
                                 label="inputs")
        pop = sim.Population(len(neurons), sim.IF_curr_exp(**params), label="fsm")
        pop.record("spikes")

        groups = {}
        for c in conns:
            pre, post, receptor, delay = c[:4]
            mult = c[4] if len(c) > 4 else 1     # whole multiple of the weight
            key = (pre[0], receptor)
            weight = mult * (w["exc"] if receptor == "exc" else w["inh"])
            groups.setdefault(key, []).append((int(pre[1]), int(post), weight, float(delay) * dt))
        for (pre_kind, receptor), lst in groups.items():
            pre_pop = src_pop if pre_kind == "src" else pop
            sim.Projection(pre_pop, pop, sim.FromListConnector(lst),
                           synapse_type=sim.StaticSynapse(),
                           receptor_type="excitatory" if receptor == "exc" else "inhibitory")

        sim.run(float(duration_steps) * dt)
        data = pop.get_data("spikes")
        spikes = {}
        for k, train in enumerate(data.segments[0].spiketrains):
            idx = int(train.annotations.get("source_index", k))
            spikes[idx] = sorted(int(round(float(t) / dt)) for t in train.magnitude)
    finally:
        try:                 # leave the simulator ready for the next run
            sim.end()
        except Exception:
            pass
    return spikes
