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

# Table 1 is for a time step of 1 ms. The gates rely on the neurons forgetting their
# input within one step (tau = 0.1 ms << 1 ms), so with another time step the time
# constants (and the capacitance, which keeps R = tau_m / cm) are scaled by it: the
# dynamics, counted in time steps, are then the same for any time step.
SCALED_WITH_DT = ("cm", "tau_m", "tau_refrac", "tau_syn_E", "tau_syn_I")


def neuron_params_for(dt, neuron_params=None):
    """Neuron parameters for a time step ``dt`` (ms): Table 1 scaled by dt, then any
    parameter given explicitly in ``neuron_params`` (used as given)."""
    params = {k: (v * dt if k in SCALED_WITH_DT else v) for k, v in DEFAULT_NEURON_PARAMS.items()}
    params.update(neuron_params or {})
    return params


def time_scale_factor(dt):
    """SpiNNaker cannot tick faster than 1 ms of wall-clock time: below 1 ms the
    simulation runs that many times slower than real time."""
    import math
    return max(1, math.ceil(1.0 / dt - 1e-9))


class SimulationCancelled(Exception):
    """The user cancelled the simulation; the simulator was closed cleanly."""


# Long simulations run in pieces, so that a cancellation takes effect between them.
RUN_CHUNK_STEPS = 2000

BACKEND_MODULES = {
    "spinnaker": "pyNN.spiNNaker",
}


def import_backend(backend):
    import importlib
    return importlib.import_module(BACKEND_MODULES[backend])


def run_pynn(netlist, trains, backend="spinnaker", dt=1.0, duration_steps=None,
             neuron_params=None, weights=None, should_stop=None, time_scale=None):
    """Simulate ``netlist`` (dict form, see Netlist.to_dict) with PyNN.

    ``trains``: {input name: [steps]}.  Returns {neuron index: [steps]}.
    ``should_stop``: called at safe points (before the setup, before the run,
    between pieces of the run and before reading the recordings); when it returns
    True the simulator is closed with ``sim.end()`` and SimulationCancelled is
    raised. The simulator is never killed: loading the network on the board, which
    happens inside the first ``sim.run``, cannot be interrupted.
    """
    stop = should_stop or (lambda: False)
    if stop():
        raise SimulationCancelled()
    sim = import_backend(backend)
    params = neuron_params_for(dt, neuron_params)
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
    # time_scale: the user's factor (at least the smallest one SpiNNaker supports)
    tsf = max(time_scale_factor(dt), int(time_scale or 1))
    sim.setup(timestep=dt, min_delay=dt, time_scale_factor=tsf)
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

        if stop():
            raise SimulationCancelled()
        done = 0
        while done < duration_steps:
            n = min(RUN_CHUNK_STEPS, duration_steps - done)
            sim.run(float(n) * dt)
            done += n
            if stop():
                raise SimulationCancelled()
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
