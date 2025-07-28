import os
import pickle

import pyNN.spiNNaker as sim
from matplotlib import pyplot as plt

# Neuron parameters
global_params = {"min_delay": 1.0, "sim_time": 100.0}
neuron_params = {"cm": 0.1, "tau_m": 0.1, "tau_refrac": 0.0, "tau_syn_E": 0.1, "tau_syn_I": 0.1, "v_rest": -65.0, "v_reset": -65.0, "v_thresh": -64.91}


if __name__ == '__main__':
    # --- Simulation ---
    sim.setup(global_params["min_delay"])

    # --- Predefined objects ---
    std_conn = sim.StaticSynapse(weight=1.0, delay=global_params["min_delay"])  # Standard connection

    # -- Network architecture --
    # - Spike injectors -
    start_time = [10]
    start_src = sim.Population(1, sim.SpikeSourceArray(spike_times=start_time))
    sg1_times = [20, 30, 40, 50, 52, 54, 56, 57, 58, 59, 60, 70, 80]
    sg1_src = sim.Population(1, sim.SpikeSourceArray(spike_times=sg1_times))
    sg2_times = []
    #sg2_src = sim.Population(1, sim.SpikeSourceArray(spike_times=sg2_times))

    # - Populations -
    latch_array = []
    and_array = []

    for i in range(3):
        latch_pop = sim.Population(3, sim.IF_curr_exp(**neuron_params), initial_values={'v': neuron_params["v_rest"]}, label="switch")
        latch_array.append(latch_pop)

    for i in range(4):
        and_pop = sim.Population(3, sim.IF_curr_exp(**neuron_params), initial_values={'v': neuron_params["v_rest"]}, label="and")
        and_array.append(and_pop)

    # - Connections -
    # Latch
    sim.Projection(start_src, sim.PopulationView(latch_array[0], [0]), sim.OneToOneConnector(), std_conn)  # Starting at S0

    for i in range(3):
        # Interconnections
        sim.Projection(sim.PopulationView(latch_array[i], [0]), sim.PopulationView(latch_array[i], [1]), sim.OneToOneConnector(), std_conn)
        sim.Projection(sim.PopulationView(latch_array[i], [1, 2]), sim.PopulationView(latch_array[i], [0]), sim.AllToAllConnector(), std_conn, receptor_type="inhibitory")

        # Recurrence
        sim.Projection(sim.PopulationView(latch_array[i], [0]), sim.PopulationView(latch_array[i], [0]), sim.OneToOneConnector(), std_conn, receptor_type="inhibitory")
        sim.Projection(sim.PopulationView(latch_array[i], [1]), sim.PopulationView(latch_array[i], [2]), sim.OneToOneConnector(), std_conn)
        sim.Projection(sim.PopulationView(latch_array[i], [2]), sim.PopulationView(latch_array[i], [1]), sim.OneToOneConnector(), std_conn)

    # Forward transitions
    for i in range(3):
        # Delay neuron
        sim.Projection(sg1_src, sim.PopulationView(and_array[i], [0]), sim.OneToOneConnector(), std_conn)

        # NOT neuron
        sim.Projection(sg1_src, sim.PopulationView(and_array[i], [1]), sim.OneToOneConnector(), std_conn)
        sim.Projection(latch_array[i], sim.PopulationView(and_array[i], [1]), sim.AllToAllConnector(), std_conn, receptor_type="inhibitory")

        # Output neuron
        sim.Projection(sim.PopulationView(and_array[i], [0]), sim.PopulationView(and_array[i], [2]), sim.OneToOneConnector(), std_conn)
        sim.Projection(sim.PopulationView(and_array[i], [1]), sim.PopulationView(and_array[i], [2]), sim.OneToOneConnector(), std_conn, receptor_type="inhibitory")

        sim.Projection(sim.PopulationView(and_array[i], [2]), sim.PopulationView(latch_array[(i + 1) % 3], [0]), sim.OneToOneConnector(), std_conn)  # Next state
        sim.Projection(sim.PopulationView(and_array[i], [2]), sim.PopulationView(latch_array[i], [1]), sim.OneToOneConnector(), std_conn, receptor_type="inhibitory")
        sim.Projection(sim.PopulationView(and_array[i], [2]), sim.PopulationView(latch_array[i], [2]), sim.OneToOneConnector(), std_conn, receptor_type="inhibitory")

    '''# Backward transition
    # Delay neuron
    sim.Projection(sg2_src, sim.PopulationView(and_array[3], [0]), sim.OneToOneConnector(), std_conn)

    # NOT neuron
    sim.Projection(sg2_src, sim.PopulationView(and_array[3], [1]), sim.OneToOneConnector(), std_conn)
    sim.Projection(latch_array[2], sim.PopulationView(and_array[3], [1]), sim.AllToAllConnector(), std_conn, receptor_type="inhibitory")

    # Output neuron
    sim.Projection(sim.PopulationView(and_array[3], [0]), sim.PopulationView(and_array[3], [2]), sim.OneToOneConnector(), std_conn)
    sim.Projection(sim.PopulationView(and_array[3], [1]), sim.PopulationView(and_array[3], [2]), sim.OneToOneConnector(), std_conn, receptor_type="inhibitory")

    sim.Projection(sim.PopulationView(and_array[3], [2]), sim.PopulationView(latch_array[2], [0]), sim.OneToOneConnector(), std_conn)  # Previous state
    sim.Projection(sim.PopulationView(and_array[3], [2]), sim.PopulationView(latch_array[3], [1]), sim.OneToOneConnector(), std_conn, receptor_type="inhibitory")
    sim.Projection(sim.PopulationView(and_array[3], [2]), sim.PopulationView(latch_array[3], [2]), sim.OneToOneConnector(), std_conn, receptor_type="inhibitory")
    '''

    # -- Recording --
    for i in range(3):
        latch_array[i].record(["spikes"])
    for i in range(4):
        and_array[i].record(["spikes"])

    # -- Run simulation --
    sim.run(global_params["sim_time"])

    # -- Get data from the simulation --
    latch_data = [latch_array[i].get_data().segments[0] for i in range(3)]
    and_data = [and_array[i].get_data().segments[0] for i in range(4)]

    # - End simulation -
    sim.end()

    # --- Saving test ---
    save_array = [sg1_times, sg2_times, latch_data, and_data]
    test_name = os.path.basename(__file__).split('.')[0]

    cwd = os.getcwd()
    if not os.path.exists(cwd + "/results/"):
        os.mkdir(cwd + "/results/")

    i = 1
    while os.path.exists(cwd + "/results/" + test_name + "_" + str(i) + ".pickle"):
        i += 1

    filename = test_name + "_" + str(i)

    with open("results/" + filename + '.pickle', 'wb') as handle:
        pickle.dump(save_array, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # --- Saving plot ---
    plt.rcParams['figure.dpi'] = 400
    plt.rcParams['font.size'] = '4'
    plt.rcParams["figure.figsize"] = (4, 1.2)

    show_and_gates = False

    # Inputs
    row = 0
    plt.plot(start_time, [row] * len(start_time), 'o', markersize=0.5, color='blue')
    row += 1
    plt.plot(sg1_times, [row] * len(sg1_times), 'o', markersize=0.5, color='blue')
    row += 1
    plt.plot(sg2_times, [row] * len(sg2_times), 'o', markersize=0.5, color='blue')
    row += 1

    # Latches
    for segment in latch_data:
        n_tmp = len(segment.spiketrains)
        for i in range(n_tmp):
            plt.plot(segment.spiketrains[i], [row] * len(segment.spiketrains[i]), 'o', markersize=0.5, color='green')
            row += 1

    # AND gates
    if show_and_gates:
        for segment in and_data:
            n_tmp = len(segment.spiketrains)
            for i in range(n_tmp):
                plt.plot(segment.spiketrains[i], [row] * len(segment.spiketrains[i]), 'o', markersize=0.5, color='orange')
                row += 1

    plt.xlim([0, global_params["sim_time"]])
    plt.ylim([-1, row])
    plt.yticks([0], labels=[" "])
    plt.xlabel('Time (ms)')
    plt.ylabel('Input')

    plt.tight_layout()
    plt.savefig("results/" + filename + '.png', transparent=False, facecolor='white', edgecolor='black')
    plt.show()
