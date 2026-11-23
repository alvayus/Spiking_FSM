import os
import pickle

import numpy as np
import pyNN.spiNNaker as sim
from matplotlib import pyplot as plt

# Neuron parameters
global_params = {"min_delay": 1.0, "sim_time": 500.0}
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
    for reps in range(10):
        # --- Simulation ---
        sim.setup(global_params["min_delay"])

        # --- Predefined objects ---
        std_conn = sim.StaticSynapse(weight=1.0, delay=global_params["min_delay"])  # Standard connection

        # -- Network architecture --
        # - Spike injectors -
        start_time = [10]
        start_src = sim.Population(1, sim.SpikeSourceArray(spike_times=start_time))
        times = [generate_spike_times(10, global_params["sim_time"] - 10, 1, 50) for i in range(4)]
        '''times = [[20, 50],
                 [30, 70],
                 [30, 70],
                 [40, 60]]'''
        A_pop = sim.Population(1, sim.SpikeSourceArray(spike_times=times[0]))
        B_pop = sim.Population(1, sim.SpikeSourceArray(spike_times=times[1]))
        C_pop = sim.Population(1, sim.SpikeSourceArray(spike_times=times[2]))
        D_pop = sim.Population(1, sim.SpikeSourceArray(spike_times=times[3]))

        # - Populations -
        op_pop = sim.Population(7, sim.IF_curr_exp(**neuron_params), initial_values={'v': neuron_params["v_rest"]})
        input_data_array = [sim.Population(4, sim.IF_curr_exp(**neuron_params), initial_values={'v': neuron_params["v_rest"]}) for i in range(4)]
        cd_nor_pop = sim.Population(1, sim.IF_curr_exp(**neuron_params), initial_values={'v': neuron_params["v_rest"]})
        latch_array = [sim.Population(4, sim.IF_curr_exp(**neuron_params), initial_values={'v': neuron_params["v_rest"]}) for i in range(3)]
        transition_array = [sim.Population(2, sim.IF_curr_exp(**neuron_params), initial_values={'v': neuron_params["v_rest"]}),
                            sim.Population(3, sim.IF_curr_exp(**neuron_params), initial_values={'v': neuron_params["v_rest"]}),
                            sim.Population(3, sim.IF_curr_exp(**neuron_params), initial_values={'v': neuron_params["v_rest"]})]

        # - Connections -
        # OP signal
        sim.Projection(A_pop, sim.PopulationView(op_pop, [0]), sim.OneToOneConnector(), std_conn)
        for i in range(1, 4):
            sim.Projection(A_pop, sim.PopulationView(op_pop, [i]), sim.OneToOneConnector(), std_conn, receptor_type="inhibitory")
        sim.Projection(B_pop, sim.PopulationView(op_pop, [1]), sim.OneToOneConnector(), std_conn)
        for i in range(2, 4):
            sim.Projection(B_pop, sim.PopulationView(op_pop, [i]), sim.OneToOneConnector(), std_conn, receptor_type="inhibitory")
        sim.Projection(C_pop, sim.PopulationView(op_pop, [2]), sim.OneToOneConnector(), std_conn)
        for i in range(3, 4):
            sim.Projection(A_pop, sim.PopulationView(op_pop, [i]), sim.OneToOneConnector(), std_conn, receptor_type="inhibitory")
        sim.Projection(D_pop, sim.PopulationView(op_pop, [3]), sim.OneToOneConnector(), std_conn)

        sim.Projection(sim.PopulationView(op_pop, [0, 1, 2, 3]), sim.PopulationView(op_pop, [4]), sim.AllToAllConnector(), std_conn)
        sim.Projection(sim.PopulationView(op_pop, [4]), sim.PopulationView(op_pop, [5]), sim.OneToOneConnector(), std_conn)
        sim.Projection(sim.PopulationView(op_pop, [5]), sim.PopulationView(op_pop, [6]), sim.OneToOneConnector(), std_conn)

        # ... Disabling transitions after a spike is received, until new state is calculated ...
        for i in range(1, 3):
            sim.Projection(sim.PopulationView(op_pop, [5]), sim.PopulationView(latch_array[i], [0]), sim.OneToOneConnector(), std_conn, receptor_type="inhibitory")

        # ... Resetting latches before setting them ...
        for i in range(1, 3):
            sim.Projection(sim.PopulationView(op_pop, [5]), sim.PopulationView(latch_array[i], [1, 2]), sim.AllToAllConnector(), std_conn, receptor_type="inhibitory")

        # Input data
        sim.Projection(A_pop, sim.PopulationView(input_data_array[0], [0]), sim.OneToOneConnector(), std_conn)
        sim.Projection(B_pop, sim.PopulationView(input_data_array[1], [0]), sim.OneToOneConnector(), std_conn)
        sim.Projection(C_pop, sim.PopulationView(input_data_array[2], [0]), sim.OneToOneConnector(), std_conn)
        sim.Projection(D_pop, sim.PopulationView(input_data_array[3], [0]), sim.OneToOneConnector(), std_conn)

        for i in range(4):
            sim.Projection(sim.PopulationView(input_data_array[i], [0]), sim.PopulationView(input_data_array[i], [1]), sim.OneToOneConnector(), std_conn)
            sim.Projection(sim.PopulationView(input_data_array[i], [1]), sim.PopulationView(input_data_array[i], [2]), sim.OneToOneConnector(), std_conn)

            sim.Projection(sim.PopulationView(op_pop, [4]), sim.PopulationView(input_data_array[i], [3]), sim.OneToOneConnector(), std_conn)
            sim.Projection(sim.PopulationView(input_data_array[i], [1]), sim.PopulationView(input_data_array[i], [3]), sim.OneToOneConnector(), std_conn, receptor_type="inhibitory")

        # NOT (C + D)
        sim.Projection(sim.PopulationView(op_pop, [4]), cd_nor_pop, sim.OneToOneConnector(), std_conn)
        sim.Projection(sim.PopulationView(input_data_array[2], [1]), cd_nor_pop, sim.OneToOneConnector(), std_conn, receptor_type="inhibitory")
        sim.Projection(sim.PopulationView(input_data_array[3], [1]), cd_nor_pop, sim.OneToOneConnector(), std_conn, receptor_type="inhibitory")

        # Latches
        sim.Projection(start_src, sim.PopulationView(latch_array[0], [0]), sim.OneToOneConnector(), std_conn)  # Starting at S0

        for i in range(3):
            # Interconnections
            sim.Projection(sim.PopulationView(latch_array[i], [0]), sim.PopulationView(latch_array[i], [1]), sim.OneToOneConnector(), std_conn)

            # Recurrence
            sim.Projection(sim.PopulationView(latch_array[i], [1]), sim.PopulationView(latch_array[i], [2]), sim.OneToOneConnector(), std_conn)
            sim.Projection(sim.PopulationView(latch_array[i], [2]), sim.PopulationView(latch_array[i], [1]), sim.OneToOneConnector(), std_conn)

            # Negated output
            sim.Projection(sim.PopulationView(op_pop, [5]), sim.PopulationView(latch_array[i], [3]), sim.OneToOneConnector(), std_conn)
            sim.Projection(sim.PopulationView(latch_array[i], [0, 1, 2]), sim.PopulationView(latch_array[i], [3]), sim.AllToAllConnector(), std_conn, receptor_type="inhibitory")

        # Transition 0
        sim.Projection(sim.PopulationView(input_data_array[0], [2]), sim.PopulationView(transition_array[0], [0]), sim.OneToOneConnector(), std_conn)
        sim.Projection(sim.PopulationView(input_data_array[0], [3]), sim.PopulationView(transition_array[0], [1]), sim.OneToOneConnector(), std_conn)

        sim.Projection(sim.PopulationView(op_pop, [6]), sim.PopulationView(latch_array[1], [0]), sim.OneToOneConnector(), std_conn)
        sim.Projection(sim.PopulationView(latch_array[0], [3]), sim.PopulationView(latch_array[1], [0]), sim.OneToOneConnector(), std_conn, receptor_type="inhibitory")
        sim.Projection(sim.PopulationView(transition_array[0], [1]), sim.PopulationView(latch_array[1], [0]), sim.OneToOneConnector(), std_conn, receptor_type="inhibitory")

        sim.Projection(sim.PopulationView(transition_array[0], [0]), sim.PopulationView(latch_array[1], [3]), sim.OneToOneConnector(), std_conn, receptor_type="inhibitory")

        # Transition 1
        sim.Projection(sim.PopulationView(input_data_array[1], [2]), sim.PopulationView(transition_array[1], [0]), sim.OneToOneConnector(), std_conn)

        sim.Projection(cd_nor_pop, sim.PopulationView(transition_array[1], [1]), sim.OneToOneConnector(), std_conn)

        sim.Projection(sim.PopulationView(op_pop, [5]), sim.PopulationView(transition_array[1], [2]), sim.OneToOneConnector(), std_conn)
        sim.Projection(sim.PopulationView(input_data_array[1], [2]), sim.PopulationView(transition_array[1], [2]), sim.OneToOneConnector(), std_conn, receptor_type="inhibitory")
        sim.Projection(cd_nor_pop, sim.PopulationView(transition_array[1], [2]), sim.OneToOneConnector(), std_conn, receptor_type="inhibitory")

        sim.Projection(sim.PopulationView(op_pop, [6]), sim.PopulationView(latch_array[2], [0]), sim.OneToOneConnector(), std_conn)
        sim.Projection(sim.PopulationView(latch_array[1], [3]), sim.PopulationView(latch_array[2], [0]), sim.OneToOneConnector(), std_conn, receptor_type="inhibitory")
        sim.Projection(sim.PopulationView(transition_array[1], [0]), sim.PopulationView(latch_array[2], [0]), sim.OneToOneConnector(), std_conn, receptor_type="inhibitory")
        sim.Projection(sim.PopulationView(transition_array[1], [1]), sim.PopulationView(latch_array[2], [0]), sim.OneToOneConnector(), std_conn, receptor_type="inhibitory")

        sim.Projection(sim.PopulationView(transition_array[1], [2]), sim.PopulationView(latch_array[2], [3]), sim.OneToOneConnector(), std_conn, receptor_type="inhibitory")

        # Transition 2
        sim.Projection(sim.PopulationView(input_data_array[1], [3]), sim.PopulationView(transition_array[2], [0]), sim.OneToOneConnector(), std_conn)
        sim.Projection(sim.PopulationView(input_data_array[2], [3]), sim.PopulationView(transition_array[2], [1]), sim.OneToOneConnector(), std_conn)

        sim.Projection(sim.PopulationView(op_pop, [6]), sim.PopulationView(transition_array[2], [2]), sim.OneToOneConnector(), std_conn)
        sim.Projection(sim.PopulationView(latch_array[2], [3]), sim.PopulationView(transition_array[2], [2]), sim.OneToOneConnector(), std_conn, receptor_type="inhibitory")
        sim.Projection(sim.PopulationView(transition_array[2], [0]), sim.PopulationView(transition_array[2], [2]), sim.OneToOneConnector(), std_conn, receptor_type="inhibitory")
        sim.Projection(sim.PopulationView(transition_array[2], [1]), sim.PopulationView(transition_array[2], [2]), sim.OneToOneConnector(), std_conn, receptor_type="inhibitory")

        # -- Recording --
        op_pop.record(["spikes"])
        for i in range(len(input_data_array)):
            input_data_array[i].record(["spikes"])
        for i in range(len(latch_array)):
            latch_array[i].record(["spikes"])
        for i in range(len(transition_array)):
            transition_array[i].record(["spikes"])

        # -- Run simulation --
        sim.run(global_params["sim_time"])

        # -- Get data from the simulation --
        op_data = op_pop.get_data().segments[0]
        latch_data = [latch_array[i].get_data().segments[0] for i in range(len(latch_array))]
        input_data = [input_data_array[i].get_data().segments[0] for i in range(len(input_data_array))]
        transition_data = [transition_array[i].get_data().segments[0] for i in range(len(transition_array))]

        # - End simulation -
        sim.end()

        # --- Saving test ---
        save_array = [times, op_data, latch_data, input_data, transition_data]
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

        reduce_states = True
        show_lines = True
        show_transitions = False

        if show_lines and reduce_states:
            plt.hlines(range(0, 50), 0, global_params["sim_time"], linestyles='dotted', linewidth=0.25, color='indigo', alpha=0.3)
        lines = []

        # Inputs
        row = 0
        plt.plot(start_time, [row] * len(start_time), '|', markersize=2, color='blue')
        row += 1

        colors = ["hotpink", "olivedrab", "chocolate", "indianred"]
        for i in range(len(times)):
            plt.plot(times[i], [row] * len(times[i]), '|', markersize=2, color=colors[i])
            row += 1
        plt.plot(op_data.spiketrains[4], [row] * len(op_data.spiketrains[4]), '|', markersize=2, color="goldenrod")
        lines.append(row + 0.5)
        row += 1

        # Latches
        if reduce_states:
            for i in range(len(latch_data)):
                plt.plot(latch_data[i].spiketrains[0], [row] * len(latch_data[i].spiketrains[0]), '|', markersize=2, color='darkgreen')
                plt.plot(latch_data[i].spiketrains[1], [row] * len(latch_data[i].spiketrains[1]), '|', markersize=2, color='darkgreen')
                plt.plot(latch_data[i].spiketrains[2], [row] * len(latch_data[i].spiketrains[2]), '|', markersize=2, color='darkgreen')
                row += 1
        '''else:
            for segment in latch_data:
                n_tmp = len(segment.spiketrains)
                for i in range(n_tmp):
                    plt.plot(segment.spiketrains[i], [row] * len(segment.spiketrains[i]), '|', markersize=2, color='darkgreen')
                    row += 1'''

        # Transitions
        if show_transitions:
            for segment in transition_data:
                n_tmp = len(segment.spiketrains)
                for i in range(n_tmp):
                    plt.plot(segment.spiketrains[i], [row] * len(segment.spiketrains[i]), '|', markersize=2, color='darkviolet')
                    row += 1
                lines.append(row - 0.5)
        else:
            plt.plot(transition_data[2].spiketrains[2], [row] * len(transition_data[2].spiketrains[2]), '|', markersize=2, color='darkviolet')
            row += 1

        if show_lines:
            plt.hlines(lines, xmin=0, xmax=global_params["sim_time"], linewidth=0.1, color="indigo")
        plt.xlim([0, global_params["sim_time"]])
        plt.ylim([-1, row])
        if reduce_states:
            plt.yticks(range(0, 10), labels=["Start", "A", "B", "C", "D", "OP", "S0", "S1", "S2", "Output"])
        plt.xlabel('Time (ms)')
        #plt.ylabel('Input')

        plt.tight_layout()
        plt.savefig("results/" + filename + '.png', transparent=False, facecolor='white', edgecolor='black')
        #plt.show()
        plt.close()