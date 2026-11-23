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
            times, op_data, latch_data, input_data, transition_data = pickle.load(f)

        # --- Plot ---
        plt.rcParams['figure.dpi'] = 400
        plt.rcParams['font.size'] = '4'
        plt.rcParams["figure.figsize"] = (4, 1.2)

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

        #colors = ["hotpink", "olivedrab", "chocolate", "indianred"]
        for i in range(len(times)):
            plt.plot(times[i], [row] * len(times[i]), '|', markersize=2, color=[108/255, 142/255, 191/255])
            row += 1
        plt.plot(op_data.spiketrains[4], [row] * len(op_data.spiketrains[4]), '|', markersize=2, color=[215/255, 155/255, 0/255])
        lines.append(row + 0.5)
        row += 1

        # Latches
        if reduce_states:
            for i in range(len(latch_data)):
                plt.plot(latch_data[i].spiketrains[0], [row] * len(latch_data[i].spiketrains[0]), '|', markersize=2, color=[130/255, 179/255, 102/255])
                plt.plot(latch_data[i].spiketrains[1], [row] * len(latch_data[i].spiketrains[1]), '|', markersize=2, color=[130/255, 179/255, 102/255])
                plt.plot(latch_data[i].spiketrains[2], [row] * len(latch_data[i].spiketrains[2]), '|', markersize=2, color=[130/255, 179/255, 102/255])
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
            plt.plot(transition_data[2].spiketrains[2], [row] * len(transition_data[2].spiketrains[2]), '|', markersize=2, color=[150/255, 115/255, 166/255])
            row += 1

        if show_lines:
            plt.hlines(lines, xmin=t_min, xmax=t_max, linewidth=0.1, color="indigo")
        plt.xlim([t_min, t_max])
        plt.ylim([-1, row])
        if reduce_states:
            plt.yticks(range(0, 9), labels=["A", "B", "C", "D", "OP", "S0", "S1", "S2", "Output"])
        plt.xlabel('Time (ms)')

        plt.tight_layout()
        plt.savefig("results/" + os.path.splitext(os.path.basename(filename))[0] + "_from_" + str(t_min) + "_to_" + str(t_max) + '.png', transparent=False, facecolor='white', edgecolor='black')
        plt.show()
        plt.close()