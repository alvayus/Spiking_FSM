import os
import pickle
from tkinter import Tk
from tkinter.filedialog import askopenfilename

import numpy as np
import pyNN.spiNNaker as sim
from matplotlib import pyplot as plt


def evaluate_fsm(A, B, C, D, T_min, T_max):
    signal_times = sorted(set(t for t in A + B + C + D if T_min <= t <= T_max))
    A_set, B_set, C_set, D_set = set(A), set(B), set(C), set(D)

    T0_times = []
    T1_times = []
    T2_times = []

    S0_times = list(range(T_min, T_max + 1))
    S1_times = set()
    S2_times = set()

    # Track when S1 and S2 became active and last evaluated time
    S1_start = None
    S2_start = None

    for i, t in enumerate(signal_times):
        # Step 1: Check T0 immediately (A * S0)
        if t in A_set:
            T0_times.append(t)
            # If S1 was active from previous trigger, extend until now
            if S1_start is not None:
                S1_times.update(range(S1_start, t))
            S1_start = t  # Activate/Reset S1 at this event

        # Step 2: Evaluate T1 if S1 active and we're at next event after S1_start
        if S1_start is not None and t > S1_start:
            S1_times.update(range(S1_start, t))
            if (t in C_set or t in D_set) and (t not in B_set):
                T1_times.append(t)
                # If S2 was active from previous trigger, extend until now
                if S2_start is not None:
                    S2_times.update(range(S2_start, t))
                S2_start = t  # Activate/Reset S2
            S1_start = None  # Clear S1 pending flag after evaluation

        # Step 3: Evaluate T2 if S2 active and we're at next event after S2_start
        if S2_start is not None and t > S2_start:
            S2_times.update(range(S2_start, t))
            if (t in B_set) and (t in C_set):
                T2_times.append(t)
            S2_start = None  # Clear S2 pending flag after evaluation

    # After looping, extend any lingering active states to T_max
    if S1_start is not None:
        S1_times.update(range(S1_start, T_max + 1))
    if S2_start is not None:
        S2_times.update(range(S2_start, T_max + 1))

    return {
        "T0": sorted(T0_times),
        "T1": sorted(T1_times),
        "T2": sorted(T2_times),
        "S0": S0_times,
        "S1": sorted(S1_times),
        "S2": sorted(S2_times),
    }

if __name__ == '__main__':
    # --- Data load ---
    window = Tk()
    window.attributes('-topmost', True)
    window.attributes('-alpha', 0)
    filename = askopenfilename(defaultextension=".pkl", filetypes=[("Pickle files", ["*.pkl", "*.pickle"])])

    if filename != "":
        with open(filename, "rb") as f:
            times, _, _, _, _ = pickle.load(f)

        # --- Plot ---
        plt.rcParams['figure.dpi'] = 400
        plt.rcParams['font.size'] = '4'
        plt.rcParams["figure.figsize"] = (4, 1.0)

        show_lines = True
        t_min = int(input("t_min: "))
        t_max = int(input("t_max: "))

        output = evaluate_fsm(times[0], times[1], times[2], times[3], t_min, t_max)
        T0_times = output["T0"]
        T1_times = output["T1"]
        T2_times = output["T2"]
        S0_times = output["S0"]
        S1_times = output["S1"]
        S2_times = output["S2"]

        if show_lines:
            plt.hlines(range(0, 50), t_min, t_max, linestyles='dotted', linewidth=0.25, color='indigo', alpha=0.3)
        lines = []

        # Inputs
        row = 0

        for i in range(len(times)):
            plt.plot(times[i], [row] * len(times[i]), '|', markersize=2, color="goldenrod")
            row += 1
        for i in range(len(times)):
            plt.plot(times[i], [row] * len(times[i]), '|', markersize=2, color="goldenrod")
        lines.append(row + 0.5)
        row += 1

        # Latches
        plt.plot(S0_times, [row] * len(S0_times), '|', markersize=2, color="goldenrod")
        row += 1
        plt.plot(S1_times, [row] * len(S1_times), '|', markersize=2, color="goldenrod")
        row += 1
        plt.plot(S2_times, [row] * len(S2_times), '|', markersize=2, color="goldenrod")
        row += 1

        # Transitions
        plt.plot(T2_times, [row] * len(T2_times), '|', markersize=2, color="goldenrod")
        row += 1

        if show_lines:
            plt.hlines(lines, xmin=t_min, xmax=t_max, linewidth=0.1, color="indigo")
        plt.xlim([t_min, t_max])
        plt.ylim([-1, row])
        plt.yticks(range(0, 9), labels=["A", "B", "C", "D", "OP", "S0", "S1", "S2", "Output"])
        plt.xlabel('Time (ms)')

        plt.tight_layout()
        plt.savefig("results/" + os.path.splitext(os.path.basename(filename))[0] + "_theoretical_from_" + str(t_min) + "_to_" + str(t_max) + '.png', transparent=False, facecolor='white', edgecolor='black')
        plt.show()
        plt.close()