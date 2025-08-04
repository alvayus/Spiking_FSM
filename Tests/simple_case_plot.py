import os
import pickle
from tkinter import Tk
from tkinter.filedialog import askopenfilename

import numpy as np
import pyNN.spiNNaker as sim
from matplotlib import pyplot as plt

if __name__ == '__main__':
    # --- Data load ---
    window = Tk()
    window.attributes('-topmost', True)
    window.attributes('-alpha', 0)
    filename = askopenfilename(defaultextension=".pkl", filetypes=[("Pickle files", ["*.pkl", "*.pickle"])])

    if filename != "":
        with open(filename, "rb") as f:
            sg1_times, sg2_times, latch_data, and_data = pickle.load(f)

        # --- Plot ---
        plt.rcParams['figure.dpi'] = 400
        plt.rcParams['font.size'] = '4'
        plt.rcParams["figure.figsize"] = (4, 1.0)

        reduce_states = True
        show_lines = True
        show_transitions = False
        t_min = int(input("t_min: "))
        t_max = int(input("t_max: "))

        if show_lines and reduce_states:
            plt.hlines(range(0, 50), t_min, t_max, linestyles='dotted', linewidth=0.25, color='indigo', alpha=0.3)
        lines = []

        # Inputs
        row = 0

        plt.plot(sg1_times, [row] * len(sg1_times), '|', markersize=2, color='goldenrod')
        row += 1
        plt.plot(sg2_times, [row] * len(sg2_times), '|', markersize=2, color='hotpink')
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
            for segment in and_data:
                n_tmp = len(segment.spiketrains)
                for i in range(n_tmp):
                    plt.plot(segment.spiketrains[i], [row] * len(segment.spiketrains[i]), '|', markersize=2,
                             color='navajowhite')
                    row += 1

        if show_lines:
            plt.hlines(lines, xmin=t_min, xmax=t_max, linewidth=0.1, color="indigo")
        plt.xlim([t_min, t_max])
        plt.ylim([-1, row])
        if reduce_states:
            plt.yticks(range(0, 5), labels=["SG1", "SG2", "S0", "S1", "S2"])
        plt.xlabel('Time (ms)')

        plt.tight_layout()
        plt.savefig("results/" + os.path.splitext(os.path.basename(filename))[0] + "_from_" + str(t_min) + "_to_" + str(t_max) + '.png', transparent=False, facecolor='white', edgecolor='black')
        plt.show()
        plt.close()