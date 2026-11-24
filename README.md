# Simple_FSM: spiking finite state machines

This repository has the code for implementing finite state machines (FSMs) as spiking neural networks (SNNs). The networks are built from spiking NOT/AND/OR gates and latches and run on SpiNNaker through PyNN.

| Folder | Contents |
| --- | --- |
| [`experiments/`](experiments) | PyNN/SpiNNaker experiments of the original submission: the simple FSM, the complex-case pattern detector, maximum-frequency tests and plots. |
| [`fsm_snn_app/`](fsm_snn_app) | **Spiking FSM builder**: a web application to design an FSM visually, generate its spiking network automatically and simulate it. |

## Spiking FSM builder

Draw a state machine and write its transition conditions with Boolean operators. The application builds the corresponding SNN, simulates it on SpiNNaker (PyNN) or with a built-in ideal simulator, and validates it against a reference automaton.

### Launching the web application

The application only needs Python ≥ 3.9 and its standard library. From the root of the repository:

```bash
cd fsm_snn_app
python run_app.py                # serves http://127.0.0.1:8765 and opens it in the browser
python run_app.py --port 9000 --no-browser
```

On Windows, if `python` is not on the PATH, use the launcher: `py run_app.py`.

Stop the server with Ctrl+C. Simulation results are saved in a `results/` folder inside the folder the application is launched from (here, `fsm_snn_app/results/`).

Optional dependencies (see [`requirements-optional.txt`](fsm_snn_app/requirements-optional.txt)):

| To… | Install |
| --- | --- |
| Simulate on SpiNNaker | `sPyNNaker` and `PyNN` (with the usual `.spynnaker.cfg` pointing to the board). The file pins the versions used for the experiments of the paper (Python 3.10, sPyNNaker 7.3.0, PyNN 0.12.4) |
| Save the `raster.png` plot with the results | `matplotlib` |

SpiNNaker is only offered when the application runs with a Python that has sPyNNaker installed. For example, with a virtual environment at the root of the repository:

```bash
py -3.10 -m venv .venv
.venv\Scripts\python -m pip install -r fsm_snn_app/requirements-optional.txt
cd fsm_snn_app
..\.venv\Scripts\python run_app.py
```

Without PyNN the application works the same with the **Ideal** simulator. The Simulate tab can also download a **stand-alone PyNN script** to run on the computer connected to the SpiNN-3 board:

```bash
python run_<machine_name>.py --backend spinnaker
```

### Web version (GitHub Pages)

The application is also published at **https://alvayus.github.io/Spiking_FSM/**, with nothing to install. The same Python package runs in the browser with [Pyodide](https://pyodide.org) (Python compiled to WebAssembly), so Design, Network (with its PyNN code) and Simulate with the **Ideal** simulator work as in the local application. The differences:

- SpiNNaker is not available: run the application locally for that.
- **Open…** reads a file from your computer, **Save** downloads the machine, and the recent files are copies kept in the browser. The **▾** menu also offers the example machines of `designs/`.
- Simulation results are not saved to a `results/` folder.
- Python takes a few seconds to load (longer on the first visit, while it downloads).

The site is built by [`build_web.py`](fsm_snn_app/build_web.py) and published by the [GitHub Pages workflow](.github/workflows/pages.yml) on every push to `main` that changes `fsm_snn_app/`. To try the build locally:

```bash
cd fsm_snn_app
python build_web.py              # writes _site/
python -m http.server 8770       # then open http://127.0.0.1:8770/_site/
```

### Usage

1. **Design.**
   - Add states by double-clicking the canvas or with **Add state**. **New** starts an empty machine.
   - Drag from the edge of a state to another to create a transition.
   - Select a transition and write its condition with **¬** (NOT), **∧** (AND) and **∨** (OR), for example `(C ∨ D) ∧ ¬B`. The buttons under the field insert the operators. You can also type `!`, `&` and `|`, or `NOT`, `AND` and `OR`.
   - The condition is also shown as a sum of products (terms joined by ∨), which can be edited input by input.
   - Mark which states are active at start. Several states may be active at the same time, for example in non-deterministic machines.
   - The **Summary** panel shows the number of states and transitions, and checks the machine live (errors, warnings and notes on how it will behave).
   - **Open…** (Ctrl + O) shows a file dialog that browses the disk (it starts in `designs/`), and **▾** next to it reopens a recent file. **Save** (Ctrl + S) writes over the file that was opened or saved last, whose name is shown next to the buttons (with • while there are unsaved changes); **Save as…** (Ctrl + Shift + S) chooses a new file and asks before replacing an existing one. The local server reads and writes the files ([`fsm_snn/files.py`](fsm_snn_app/fsm_snn/files.py)), so this works in any browser; only folders and `.json` files are listed, and only the application's own page may use it.
   - The canvas works like draw.io: drag the background to select several states (Shift / Ctrl + click adds to the selection), Space + drag, right drag or scroll to move, Ctrl + scroll (or pinch) to zoom, arrows to nudge, Ctrl + C / X / V / D to copy, cut, paste and duplicate, and **Grid** to snap states to the grid. The full list is under *Navigation and shortcuts* in the help panel.
2. **Network.**
   - Shows the generated network neuron by neuron and synapse by synapse, with the colours of the paper: inputs blue, OP orange, NOT gates red, delay neuron yellow, AND gates (transition terms) purple and latches green.
   - The columns follow the order in which the neurons spike after an input event, and each state has its own row with its NOT gate, the AND gates of the transitions that leave it and its latch. Synapses are drawn as orthogonal wires, each with its own port and lane so that they do not overlap. Excitatory synapses end in an arrow and inhibitory synapses in an open circle that touches the target neuron.
   - The panel shows the number of neurons (plus the spike sources: inputs and initialisation) and of synapses (excitatory and inhibitory).
   - **Show** filters the drawing by neuron type (spike sources, OP, delay, NOT gates, AND gates, latches) and by synapse type; its coloured checkboxes double as the legend.
   - Hover over a neuron to highlight its synapses; click it to list its incoming and outgoing synapses and to see when it first spikes after an input event (e.g. *t* + 6 for a ring neuron), with that path highlighted.
   - Double-click any name of the diagram (column headings, state boxes, the text inside or next to a neuron) to rename it; only the names change. The names are saved with the machine (`net_labels` in the JSON file) and **Original names** restores them all.
   - **Save image** saves the whole diagram, with the current names and the neuron types shown, as SVG (vector) or PNG, in the folder chosen in the same file dialog as the machines.
   - **PyNN code** shows the network as readable PyNN code (one population per gate or latch, one projection per synapse, with comments), generated from the same netlist that is simulated, and lets you copy or download it.
3. **Simulate.**
   - Choose the simulator and how the input spikes are generated: **Random** (a new random sequence of input events on every run; its seed is saved with the results), **Manual** (spike times typed by hand) or **Random + manual** (typed spikes inserted on top of a random sequence; random events closer than 5 steps to a typed one are left out).
   - Press **Simulate**. The activity of the network is compared with a reference automaton; if they differ, the missing and extra spikes of each state are listed and highlighted in the plot (missing spikes in orange, extra spikes in red).
   - The activity raster follows the layout of the figures of the paper (inputs at the bottom, then OP, the states and Output on top). With SpiNNaker, **Show expected** draws the activity of the reference in grey behind it.
   - Every run is saved in `results/<date>_<name>_<simulator>/`: `fsm.json`, `simulation.json` with spikes and metrics, and `raster.png`.

### State behaviour

Every state is a **spiking latch**: once active, it stays active until one of its outgoing transitions fires. An input event that fires none of its transitions leaves it active.

Transitions are evaluated only on **input events**, the time steps in which at least one input spikes (when the OP neuron fires). A term without positive literals, such as `¬A ∧ ¬B`, therefore fires on every event in which A and B stay silent, that is, whenever another input spikes. It never fires when no input spikes at all.

Several states may be active at start, and if two transitions leaving the same state hold at the same time, both destinations become active (non-determinism). The Design tab shows a note about each of these situations.

The spikes of the latch neurons of every state are recorded, so the state of the machine can be read directly from the simulation (also on SpiNNaker).

### Network construction ([`fsm_snn/builder.py`](fsm_snn_app/fsm_snn/builder.py))

A transition condition is stored as a sum of products: each term is an AND of literals (`X` or `¬X`), and the terms are OR-ed together.

| Element | Neurons | Implementation |
| --- | --- | --- |
| OP | 1 | Input-driven OR: spikes whenever any input spikes |
| Delay | 1 (shared) | Yellow neuron of the AND gate |
| NOT of each input used as a positive literal | 1 per input | Excited by OP, inhibited by the input |
| NOT of each source state | 1 per state | Excited by OP, inhibited by the latch ring |
| Transition term (AND) | 1 per term | Excited by the delay neuron, inhibited by the NOT gates and by the inputs of the negated literals |
| Latch | 3 per state | Set → N1 → ring N2 ↔ N3. N1 inhibits itself and is inhibited by N2 and N3; Reset inhibits N2 and N3 |
| Output (internal, optional) | 1 | OR of the transitions entering states flagged as `accepting` in the JSON specification. The editor does not create it; the built-in sequence detectors used by the tests use it. |

Timing, in time steps after an input event at time step *t*:

| Step | What happens |
| --- | --- |
| *t* + 1 | OP |
| *t* + 2 | NOT gates and delay neuron (state sampled at *t* + 1) |
| *t* + 3 | Transitions |
| *t* + 4 | Output and reset of the source state |
| *t* + 5 | Set of the destination |
| *t* + 6 | Destination active |

Events must therefore be at least 5 steps apart. Synaptic delays are computed automatically so that every signal arrives at the step it is needed. For any specification the cost is linear: 3·|S| latch neurons, plus one neuron per transition term, plus the shared NOT gates.

### Code structure (`fsm_snn_app/`)

| File | Role |
| --- | --- |
| `run_app.py` | Starts the local web server |
| `fsm_snn/spec.py` | Machine format, validation, notes and the built-in machines (default machine and tests) |
| `fsm_snn/builder.py` | Machine → netlist of neurons and synapses (the spiking gates and latches) |
| `fsm_snn/codegen.py` | Netlist → readable PyNN code (Network tab) |
| `fsm_snn/files.py` | Design files on disk: folder listing, open and save (Open / Save / recent files) |
| `fsm_snn/web.py`, `static/web.js`, `build_web.py` | Web version: the API run in the browser by Pyodide, and the build of the static site |
| `fsm_snn/inputs.py`, `reference.py`, `analysis.py` | Input spike trains, reference automaton and comparison of the simulation with it |
| `fsm_snn/backends.py`, `pynn_runner.py`, `pipeline.py` | Ideal simulator, PyNN execution and the end-to-end run and stand-alone script |
| `fsm_snn/server.py` | HTTP API used by the web page |
| `fsm_snn/static/` | Web page: `util.js` (helpers), `expr.js` (Boolean conditions), `viewport.js` (pan and zoom), `editor.js` (Design tab), `network.js` (Network tab), `simulate.js` (Simulate tab), `app.js` (start-up) |

### Validation

[`fsm_snn/reference.py`](fsm_snn_app/fsm_snn/reference.py) implements the same semantics as a software automaton. After each simulation, the set of active states of the network is compared with the reference event by event, and detections are matched within ±1 step. A constant time offset introduced by a simulator is calibrated automatically from the activation of the initial state.

```bash
cd fsm_snn_app
python tests/test_core.py        # built-in machines with random, manual and mixed inputs
python tests/test_pynn_path.py   # PyNN code path and exported script (with a PyNN stand-in)
python tests/test_codegen.py     # the readable PyNN code builds exactly the simulated network
```

### Known limitations

* **Synaptic weights.** The default weights (1.0 nA excitatory, 2.0 nA inhibitory) and the parameters of Table 1 are in [`fsm_snn/pynn_runner.py`](fsm_snn_app/fsm_snn/pynn_runner.py). They have not been copied from the experiments in `experiments/`: adjust them in that file before using SpiNNaker.
* **Time step.** The Ideal simulator gives the same result for any time step (it only changes the time scale in ms). SpiNNaker accepts any whole number of microseconds; below 1 ms it runs slower than real time. Only the 1 ms step has been used in the experiments.
* **No bypass.** The generic builder does not implement the parallel-transition mechanism of the paper's detector, which allows events 1τ apart. Correctness is guaranteed for events at least 5τ apart. The gate library is isolated in `builder.py` so it can be replaced with the exact implementation used in `experiments/`.
* **Validation without hardware.** The PyNN code path has only been verified with a stand-in that reproduces the PyNN API. The first run on SpiNNaker should confirm a 100 % state agreement.
