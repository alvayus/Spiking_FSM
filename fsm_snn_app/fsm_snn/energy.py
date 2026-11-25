"""Theoretical dynamic energy of the network, the Power tab.

The estimate follows the energy model of the spiking ALU work (Ayuso-Martinez et
al., Advanced Intelligent Systems 7, 2400524, 2025, Eq. 3, from N. Risi's PhD
thesis, 2022), with the energy of each event given by Moradi et al. (IEEE TBioCAS
12, 106, 2018):

    P = sum_n r_n (E_spike + E_en + N_cores (E_br + E_rt) + N_cam-match E_pulse)

r_n is the firing rate of neuron n and N_cam-match the number of synapses that
receive its spikes. As in that work, the whole network shares one core, so a spike
with at least one target is broadcast and routed once (N_cores = 1). Over a
simulation, the energy is the same sum with the number of spikes of each neuron
instead of its rate, and the mean power is that energy over the simulated time.

It is a theoretical model: it only uses the spikes of the network, so it does not
depend on the simulator that produced them. Only the dynamic energy of the neurons
is counted; the spike generators (input and initialisation sources) and the static
power are left out.
"""

from __future__ import annotations

# energy of each event (pJ)
E_SPIKE, E_EN, E_BR, E_RT, E_PULSE = 883, 883, 6840, 360, 324
PARTS = ("generation", "communication", "pulses")


def estimate(net, spikes: dict | None = None, duration_steps: int | None = None, dt: float = 1.0) -> dict:
    """Energy per spike of every neuron of ``net`` (a Netlist or its dict). With
    ``spikes`` ({neuron index: steps}), also the energy of that simulation and its
    mean power."""
    d = net if isinstance(net, dict) else net.to_dict()
    fan_out = [0] * len(d["neurons"])
    fan_in = [0] * len(d["neurons"])
    for pre, post, _rec, _delay in d["conns"]:
        fan_in[post] += 1
        if pre[0] == "n":
            fan_out[int(pre[1])] += 1
    neurons = []
    for i, n in enumerate(d["neurons"]):
        cam = fan_out[i]
        parts = dict(generation=E_SPIKE + E_EN, communication=(E_BR + E_RT) if cam else 0, pulses=cam * E_PULSE)
        row = dict(idx=i, label=n["label"], kind=n["kind"], fan_in=fan_in[i], fan_out=cam,
                   per_spike=sum(parts.values()), parts=parts)
        if spikes is not None:
            k = len(spikes.get(i, spikes.get(str(i), [])))
            row.update(spikes=k, energy=k * row["per_spike"])
        neurons.append(row)

    out = dict(neurons=neurons, synapses=len(d["conns"]))
    if spikes is not None:
        time_ms = float(duration_steps or 0) * float(dt)
        energy = sum(r["energy"] for r in neurons)
        out["run"] = dict(
            energy_pj=energy, time_ms=time_ms, dt=float(dt),
            power_w=energy * 1e-12 / (time_ms * 1e-3) if time_ms > 0 else None,
            spikes=sum(r["spikes"] for r in neurons),
            parts={p: sum(r["spikes"] * r["parts"][p] for r in neurons) for p in PARTS})
    return out
