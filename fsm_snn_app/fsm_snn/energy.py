"""Dynamic energy of the network on DYNAP-SE (Dynap-SE1), the Energy tab.

The estimate follows the spiking ALU work (Ayuso-Martinez et al., Advanced
Intelligent Systems 7, 2400524, 2025, Eq. 3, taken from N. Risi's PhD thesis,
2022), with the energy of each event given by Moradi et al. (IEEE TBioCAS 12,
106, 2018):

    P = sum_n r_n (E_spike + E_en + N_cores (E_br + E_rt) + N_cam-match E_pulse)

r_n is the firing rate of neuron n, N_cores the number of cores that receive its
spikes and N_cam-match the number of synapses (CAM entries) that receive them.
Over a simulation, the energy is the same sum with the number of spikes of each
neuron instead of its rate, and the mean power is that energy over the simulated
time. As in that work, only the dynamic energy is counted: the static power of
the chip and the spike generators (input and initialisation sources, outside the
neurons of the chip) are left out.
"""

from __future__ import annotations

import math

# energy of each event (pJ)
E_SPIKE, E_EN, E_BR, E_RT, E_PULSE = 883, 883, 6840, 360, 324
EVENTS = [
    dict(key="E_spike", pj=E_SPIKE, what="generation of a spike"),
    dict(key="E_en", pj=E_EN, what="encoding of the spike and its destinations"),
    dict(key="E_br", pj=E_BR, what="broadcast of the spike to a target core"),
    dict(key="E_rt", pj=E_RT, what="routing of the spike to a target core"),
    dict(key="E_pulse", pj=E_PULSE, what="pulse extension at each synapse that receives it"),
]
# Dynap-SE1 board: 4 chips of 4 cores of 256 neurons; 64 CAM entries (input synapses) per neuron
NEURONS_PER_CORE, CORES_PER_CHIP, CHIPS_PER_BOARD, CAM_PER_NEURON = 256, 4, 4, 64


def estimate(net, spikes: dict | None = None, duration_steps: int | None = None, dt: float = 1.0) -> dict:
    """Energy per spike and resources of every neuron of ``net`` (a Netlist or its dict).

    The neurons are placed on the cores in order. With ``spikes`` ({neuron index:
    steps}), also the energy of that simulation and its mean power."""
    d = net if isinstance(net, dict) else net.to_dict()
    n_neurons = len(d["neurons"])
    targets = [[] for _ in range(n_neurons)]
    fan_in = [0] * n_neurons
    for pre, post, _rec, _delay in d["conns"]:
        fan_in[post] += 1
        if pre[0] == "n":
            targets[int(pre[1])].append(post)
    neurons = []
    for i, n in enumerate(d["neurons"]):
        cam = len(targets[i])
        cores = len({p // NEURONS_PER_CORE for p in targets[i]})
        parts = dict(generation=E_SPIKE + E_EN, communication=cores * (E_BR + E_RT), pulses=cam * E_PULSE)
        row = dict(idx=i, label=n["label"], kind=n["kind"], fan_in=fan_in[i], fan_out=cam, cores=cores,
                   per_spike=sum(parts.values()), parts=parts)
        if spikes is not None:
            k = len(spikes.get(i, spikes.get(str(i), [])))
            row.update(spikes=k, energy=k * row["per_spike"])
        neurons.append(row)

    cores = max(1, math.ceil(n_neurons / NEURONS_PER_CORE))
    resources = dict(neurons=n_neurons, synapses=len(d["conns"]), cores=cores,
                     chips=math.ceil(cores / CORES_PER_CHIP),
                     fits_board=cores <= CORES_PER_CHIP * CHIPS_PER_BOARD,
                     max_fan_in=max(fan_in, default=0),
                     over_cam=[i for i, f in enumerate(fan_in) if f > CAM_PER_NEURON])
    out = dict(neurons=neurons, resources=resources, events=EVENTS,
               limits=dict(neurons_per_core=NEURONS_PER_CORE, cores_per_chip=CORES_PER_CHIP,
                           chips_per_board=CHIPS_PER_BOARD, cam_per_neuron=CAM_PER_NEURON))
    if spikes is not None:
        time_ms = float(duration_steps or 0) * float(dt)
        energy = sum(r["energy"] for r in neurons)
        out["run"] = dict(
            energy_pj=energy, time_ms=time_ms, dt=float(dt),
            power_w=energy * 1e-12 / (time_ms * 1e-3) if time_ms > 0 else None,
            spikes=sum(r["spikes"] for r in neurons),
            parts={p: sum(r["spikes"] * r["parts"][p] for r in neurons)
                   for p in ("generation", "communication", "pulses")})
    return out
