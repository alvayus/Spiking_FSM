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
    for reps in range(10):
        # --- Simulation ---
        sim.setup(global_params["min_delay"])

        # --- Predefined objects ---
        std_conn = sim.StaticSynapse(weight=1.0, delay=global_params["min_delay"])  # Standard connection

        # -- Network architecture --
        # - Spike injectors -
        start_time = [10]
        start_src = sim.Population(1, sim.SpikeSourceArray(spike_times=start_time))
        sg1_times = generate_spike_times(10, 90, 3, 7)
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
    
        sim.Projection(sim.PopulationView(and_array[3], [2]), sim.PopulationView(latch_array[1], [0]), sim.OneToOneConnector(), std_conn)  # Previous state
        sim.Projection(sim.PopulationView(and_array[3], [2]), sim.PopulationView(latch_array[2], [1]), sim.OneToOneConnector(), std_conn, receptor_type="inhibitory")
        sim.Projection(sim.PopulationView(and_array[3], [2]), sim.PopulationView(latch_array[2], [2]), sim.OneToOneConnector(), std_conn, receptor_type="inhibitory")
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
        plt.rcParams["figure.figsize"] = (4, 1.0)

        reduce_states = True
        show_and_gates = False
        show_lines = True

        if show_lines and reduce_states:
            plt.hlines(range(0, 6), 0, global_params["sim_time"], linestyles='dotted', linewidth=0.25, color='indigo', alpha=0.3)
        lines = []

        # Inputs
        row = 0
        plt.plot(start_time, [row] * len(start_time), '|', markersize=2, color='blue')
        row += 1
        plt.plot(sg1_times, [row] * len(sg1_times), '|', markersize=2, color='goldenrod')
        row += 1
        plt.plot(sg2_times, [row] * len(sg2_times), '|', markersize=2, color='hotpink')
        lines.append(row + 0.5)
        row += 1

        # Latches
        if reduce_states:
            for segment in latch_data:
                for i in range(0, len(latch_data), 3):
                    plt.plot(segment.spiketrains[i], [row] * len(segment.spiketrains[i]), '|', markersize=2, color='darkgreen')
                    plt.plot(segment.spiketrains[i+1], [row] * len(segment.spiketrains[i+1]), '|', markersize=2, color='darkgreen')
                    plt.plot(segment.spiketrains[i+2], [row] * len(segment.spiketrains[i+2]), '|', markersize=2, color='darkgreen')
                row += 1
        else:
            for segment in latch_data:
                n_tmp = len(segment.spiketrains)
                for i in range(n_tmp):
                    plt.plot(segment.spiketrains[i], [row] * len(segment.spiketrains[i]), '|', markersize=2, color='darkgreen')
                    row += 1

        # AND gates
        if show_and_gates:
            for segment in and_data:
                n_tmp = len(segment.spiketrains)
                for i in range(n_tmp):
                    plt.plot(segment.spiketrains[i], [row] * len(segment.spiketrains[i]), '|', markersize=2, color='navajowhite')
                    row += 1

        if show_lines:
            plt.hlines(lines, xmin=0, xmax=global_params["sim_time"], linewidth=0.1, color="indigo")
        plt.xlim([0, global_params["sim_time"]])
        plt.ylim([-1, row])
        if reduce_states:
            plt.yticks(range(0, 6), labels=["Start", "SG1", "SG2", "S0", "S1", "S2"])
        plt.xlabel('Time (ms)')
        #plt.ylabel('Input')

        plt.tight_layout()
        plt.savefig("results/" + filename + '.png', transparent=False, facecolor='white', edgecolor='black')
        plt.close()