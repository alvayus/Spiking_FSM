import os
import pickle

import numpy as np
import pyNN.spiNNaker as sim
from matplotlib import pyplot as plt

# Neuron parameters
global_params = {"min_delay": 1.0, "sim_time": 100.0}
neuron_params = {"cm": 0.1, "tau_m": 0.1, "tau_refrac": 0.0, "tau_syn_E": 0.1, "tau_syn_I": 0.1, "v_rest": -65.0, "v_reset": -65.0, "v_thresh": -64.91}


def generate_spike_times(time_min, time_max, min_step, max_step):
    res = []

    random_step = np.random.randint(min_step, max_step)
    number = time_min + random_step
    while number < time_max:
        res.append(number)

        random_step = np.random.randint(min_step, max_step)
        number = number + random_step

    return res


if __name__ == '__main__':
    for reps in range(1):
        # --- Simulation ---
        sim.setup(global_params["min_delay"])

        # --- Predefined objects ---
        std_conn = sim.StaticSynapse(weight=1.0, delay=global_params["min_delay"])  # Standard connection

        # -- Network architecture --
        # - Spike injectors -
        #times = [generate_spike_times(10, global_params["sim_time"] - 10, 1, 5) for i in range(2)]
        times = [[20, 21, 25, 30],
                 [35, 40],
                 [45, 50],
                 [55, 60]]
        A_pop = sim.Population(1, sim.SpikeSourceArray(spike_times=times[0]))
        B_pop = sim.Population(1, sim.SpikeSourceArray(spike_times=times[1]))
        C_pop = sim.Population(1, sim.SpikeSourceArray(spike_times=times[2]))
        D_pop = sim.Population(1, sim.SpikeSourceArray(spike_times=times[3]))

        # - Populations -
        delayer_straight_array = [sim.Population(2, sim.IF_curr_exp(**neuron_params), initial_values={'v': neuron_params["v_rest"]}) for i in range(4)]
        delayer_alt_array = [sim.Population(4, sim.IF_curr_exp(**neuron_params), initial_values={'v': neuron_params["v_rest"]}) for i in range(4)]

        # - Connections -
        # Straight path
        sim.Projection(A_pop, sim.PopulationView(delayer_straight_array[0], [0]), sim.OneToOneConnector(), std_conn)
        sim.Projection(B_pop, sim.PopulationView(delayer_straight_array[1], [0]), sim.OneToOneConnector(), std_conn)
        sim.Projection(C_pop, sim.PopulationView(delayer_straight_array[2], [0]), sim.OneToOneConnector(), std_conn)
        sim.Projection(D_pop, sim.PopulationView(delayer_straight_array[3], [0]), sim.OneToOneConnector(), std_conn)

        for i in range(4):
            # Autoinhibition
            sim.Projection(sim.PopulationView(delayer_straight_array[i], [0]), sim.PopulationView(delayer_straight_array[i], [0]), sim.OneToOneConnector(), std_conn, receptor_type="inhibitory")

            # Straight output
            sim.Projection(sim.PopulationView(delayer_straight_array[i], [0]), sim.PopulationView(delayer_straight_array[i], [1]), sim.OneToOneConnector(), std_conn)

            # Alternative path inhibition
            sim.Projection(sim.PopulationView(delayer_straight_array[i], [0]), sim.PopulationView(delayer_alt_array[i], [1, 2, 3]), sim.AllToAllConnector(), std_conn, receptor_type="inhibitory")

            # Inhibiting other inputs
            for j in range(4):
                if j != i:
                    sim.Projection(sim.PopulationView(delayer_straight_array[i], [0]), sim.PopulationView(delayer_straight_array[j], [0]), sim.OneToOneConnector(), std_conn, receptor_type="inhibitory")

        # Alternative path (+1 delay)
        sim.Projection(A_pop, sim.PopulationView(delayer_alt_array[0], [0]), sim.OneToOneConnector(), std_conn)
        sim.Projection(B_pop, sim.PopulationView(delayer_alt_array[1], [0]), sim.OneToOneConnector(), std_conn)
        sim.Projection(C_pop, sim.PopulationView(delayer_alt_array[2], [0]), sim.OneToOneConnector(), std_conn)
        sim.Projection(D_pop, sim.PopulationView(delayer_alt_array[3], [0]), sim.OneToOneConnector(), std_conn)

        for i in range(4):
            # Interconnections
            sim.Projection(sim.PopulationView(delayer_alt_array[i], [0]), sim.PopulationView(delayer_alt_array[i], [1]), sim.OneToOneConnector(), std_conn)
            sim.Projection(sim.PopulationView(delayer_alt_array[i], [1, 2]), sim.PopulationView(delayer_alt_array[i], [0]), sim.AllToAllConnector(), std_conn, receptor_type="inhibitory")

            # Recurrence
            sim.Projection(sim.PopulationView(delayer_alt_array[i], [0]), sim.PopulationView(delayer_alt_array[i], [0]), sim.OneToOneConnector(), std_conn, receptor_type="inhibitory")
            sim.Projection(sim.PopulationView(delayer_alt_array[i], [1]), sim.PopulationView(delayer_alt_array[i], [2]), sim.OneToOneConnector(), std_conn)
            sim.Projection(sim.PopulationView(delayer_alt_array[i], [2]), sim.PopulationView(delayer_alt_array[i], [1]), sim.OneToOneConnector(), std_conn)

            # Delay neuron
            sim.Projection(sim.PopulationView(delayer_alt_array[i], [0, 1, 2]), sim.PopulationView(delayer_alt_array[i], [3]), sim.AllToAllConnector(), std_conn)
            sim.Projection(sim.PopulationView(delayer_alt_array[i], [3]), sim.PopulationView(delayer_alt_array[i], [1, 2]), sim.AllToAllConnector(), std_conn, receptor_type="inhibitory")

            # Inhibiting other inputs
            for j in range(4):
                if j != i:
                    sim.Projection(sim.PopulationView(delayer_alt_array[i], [3]), sim.PopulationView(delayer_alt_array[j], [3]), sim.OneToOneConnector(), std_conn, receptor_type="inhibitory")

            # Output neuron
            sim.Projection(sim.PopulationView(delayer_alt_array[i], [3]), sim.PopulationView(delayer_straight_array[i], [1]), sim.OneToOneConnector(), std_conn)

        # -- Recording --
        for i in range(len(delayer_straight_array)):
            delayer_straight_array[i].record(["spikes"])
        for i in range(len(delayer_alt_array)):
            delayer_alt_array[i].record(["spikes"])

        # -- Run simulation --
        sim.run(global_params["sim_time"])

        # -- Get data from the simulation --
        delayer_straight_data = [delayer_straight_array[i].get_data().segments[0] for i in range(len(delayer_straight_array))]
        delayer_alt_data = [delayer_alt_array[i].get_data().segments[0] for i in range(len(delayer_alt_array))]

        # - End simulation -
        sim.end()

        # --- Saving test ---
        save_array = [times, delayer_straight_data, delayer_alt_data]
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
        plt.rcParams["figure.figsize"] = (4, 1.0)

        show_lines = True
        show_internal = False

        if show_lines:
            plt.hlines(range(0, 50), 0, global_params["sim_time"], linestyles='dotted', linewidth=0.25, color='indigo', alpha=0.3)
        lines = []

        # Inputs
        row = 0

        colors = ["hotpink", "olivedrab", "chocolate", "indianred"]
        for i in range(len(times)):
            plt.plot(times[i], [row] * len(times[i]), '|', markersize=2, color=colors[i])
            row += 1
        lines.append(row - 0.5)

        # Data
        for i in range(len(delayer_straight_data)):
            if show_internal:
                # Straight path data (Neuron 0)
                plt.plot(delayer_straight_data[i].spiketrains[0], [row] * len(delayer_straight_data[i].spiketrains[0]), '|', markersize=2, color='teal')
                row += 1

                # Alternative path data
                plt.plot(delayer_alt_data[i].spiketrains[0], [row] * len(delayer_alt_data[i].spiketrains[0]), '|', markersize=2, color='goldenrod')
                plt.plot(delayer_alt_data[i].spiketrains[1], [row] * len(delayer_alt_data[i].spiketrains[1]), '|', markersize=2, color='goldenrod')
                plt.plot(delayer_alt_data[i].spiketrains[2], [row] * len(delayer_alt_data[i].spiketrains[2]), '|', markersize=2, color='goldenrod')
                row += 1

                plt.plot(delayer_alt_data[i].spiketrains[3], [row] * len(delayer_alt_data[i].spiketrains[3]), '|', markersize=2, color='goldenrod')
                row += 1

            # Output neuron
            plt.plot(delayer_straight_data[i].spiketrains[1], [row] * len(delayer_straight_data[i].spiketrains[1]), '|', markersize=2, color='darkviolet')
            row += 1

        if show_lines:
            plt.hlines(lines, xmin=0, xmax=global_params["sim_time"], linewidth=0.1, color="indigo")
        plt.xlim([0, global_params["sim_time"]])
        plt.ylim([-1, row])
        #plt.yticks(range(0, 10), labels=["Start", "A", "B", "C", "D", "OP", "S0", "S1", "S2", "Output"])
        plt.xlabel('Time (ms)')

        plt.tight_layout()
        plt.savefig("results/" + filename + '.png', transparent=False, facecolor='white', edgecolor='black')
        plt.show()
        plt.close()