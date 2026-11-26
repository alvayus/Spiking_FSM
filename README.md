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

Without PyNN the application works the same with the **Ideal** simulator: it uses no specific neuron model, only Boolean units that fire when they receive an input spike, after the synaptic delays of the network. SpiNNaker uses PyNN's `IF_curr_exp` neuron model. **PyNN code** in the Simulate tab gives a **stand-alone script** with everything needed: the network written out as in the Network tab (one population per gate or latch, one projection per synapse), the input spikes, the run, the comparison with the reference automaton and the saved results. It only needs PyNN with sPyNNaker (matplotlib is optional, for the plot); run it on the computer connected to the SpiNN-3 board:

```bash
python run_<machine_name>.py
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

The page also works on tablets and phones: below 900 px wide the panels go under the canvas and the page scrolls, and on touch screens two fingers move and zoom the canvases.

1. **Design.**
   - Add states by double-clicking the canvas or with **Add state**. **New** starts an empty machine.
   - Drag from the edge of a state to another to create a transition.
   - Select a transition and write its condition with **¬** (NOT), **∧** (AND) and **∨** (OR), for example `(C ∨ D) ∧ ¬B`. The buttons under the field insert the operators. You can also type `!`, `&` and `|`, or `NOT`, `AND` and `OR`.
   - The condition is also shown as a sum of products (terms joined by ∨), which can be edited input by input.
   - Mark which states are active at start. Several states may be active at the same time, for example in non-deterministic machines.
   - The name of the design and its file are shown at the top right of every tab.
   - The **Summary** panel shows the number of states and transitions, and checks the machine live (errors, warnings and notes on how it will behave).
   - The file buttons are in the Design tab's toolbar. **Open…** (Ctrl + O) shows a file dialog that browses the disk (it starts in `designs/`), and **▾** next to it reopens a recent file. **Save** (Ctrl + S) writes over the file that was opened or saved last, whose name is shown next to the buttons (with • while there are unsaved changes); **Save as…** (Ctrl + Shift + S) chooses a new file and asks before replacing an existing one. The local server reads and writes the files ([`fsm_snn/files.py`](fsm_snn_app/fsm_snn/files.py)), so this works in any browser; only folders and `.json` files are listed, and only the application's own page may use it.
   - The canvas works like draw.io: drag the background to select several states (Shift / Ctrl + click adds to the selection), Space + drag, right drag or scroll to move, Ctrl + scroll (or pinch) to zoom, arrows to nudge, Ctrl + C / X / V / D to copy, cut, paste and duplicate, and **Grid** to snap states to the grid. The full list is under *Navigation and shortcuts* in the help panel.
2. **Network.**
   - Shows the generated network neuron by neuron and synapse by synapse, with the colours of the papers: inputs blue, the alternative OR gate (OP) orange, NOT and NOR gates red (also the NOT gates of the states, S̄), delay neurons yellow, AND gates and bypass guards purple, and latches green.
   - Each column is a step after the input spikes at *t* (*t*, *t* + 1, *t* + 2...): every neuron in it fires at that step, a step has one sub-column per kind of neuron (named under the step), and light bands and dotted lines separate the steps. Each state has its own row with its NOT gate, the AND gates of the transitions that leave it and its latch (N1 at its step, the ring one step later). Synapses are drawn as orthogonal wires; every signal (a neuron and the kind of its synapses) has its own lane and its own point on the neuron it reaches, so that synapses of different neurons never share a line, and a channel widens when it needs more lanes; when too many reach the same side of a neuron, the outer ones enter (or leave) it through its diagonals. Excitatory synapses end in an arrow and inhibitory synapses in an open circle that touches the target neuron.
   - Neurons carry the names of the papers: OP and the input neurons of the alternative OR gate, the NOT gates Ā and S̄0, the AND gates of the transitions T0, T1... (as in the manuscript), the NOR gates of the OR gates, the delay neurons (A + 3, OP + 1...) and the Set (N1) and ring of each latch. Every synapse also says what it does (Set, Reset, first positive input, the source state is not active...), in its tooltip and in the panels.
   - The panel shows the number of neurons (without the spike sources: inputs and initialisation, as in the papers) and of synapses (excitatory and inhibitory).
   - **Show** filters the drawing by neuron type (spike sources, alternative OR gate, delay neurons, NOT gates, bypass guards, AND gates, NOR gates, latches) and by synapse type; its coloured checkboxes double as the legend.
   - Hover over a neuron to highlight its synapses; click it to list its incoming and outgoing synapses and to see when it first spikes after an input event (e.g. *t* + 6 for a ring neuron), with that path highlighted.
   - Ctrl or Shift + click adds neurons and synapses to the selection (or removes them), and a click on the box of a state or a term selects all its neurons. Selecting a neuron also selects its synapses, so what is highlighted is what is selected (the latency path of a single neuron is drawn dashed and is not part of the selection). **Summary** then shows how many neurons and synapses are selected next to the totals of the network, and the selection panel lists them. Esc clears the selection.
   - Double-click any name of the diagram (column headings, state boxes, the text inside or next to a neuron) to rename it; only the names change. The names are saved with the machine (`net_labels` in the JSON file) and **Original names** restores them all.
   - **Save image** saves the whole diagram, with the current names and the neuron types shown, as SVG (vector) or PNG, in the folder chosen in the same file dialog as the machines.
   - **PyNN code** shows the network as readable PyNN code (one population per gate or latch, one projection per synapse, with comments), generated from the same netlist that is simulated, and lets you copy or download it.
3. **Simulate.**
   - Choose the simulator and how the input spikes are generated: **Random** (a new random sequence of input events on every run; its seed is saved with the results), **Manual** (spike times typed by hand) or **Random + manual** (typed spikes inserted on top of a random sequence; random events closer than the minimum spacing to a typed one are left out). The spacing between random events is the user's choice; the generated network guarantees correct behaviour from a minimum spacing on (3 steps for the pipelined network, 1 for the parallel one), and closer events are reported after the run.
   - Press **Simulate**. The activity of the network is compared with a reference automaton; if they differ, the missing and extra spikes of each state are listed and highlighted in the plot (missing spikes in orange, extra spikes in red).
   - The activity raster follows the layout of the figures of the paper (inputs at the bottom, then OP, the states and Output on top). With either simulator, **Show expected** adds below it the plot of the FSM itself (input signals and states), which changes state at the very step of each input event (no synaptic delays). **Save image** saves the plots shown as SVG or PNG. Extra and missing spikes are found by comparing the network with that reference shifted by its processing time (states and Output 5 steps after the event, or 6 when their OR gates are inhibition-based).
   - Every run is saved in `results/<date>_<name>_<simulator>/`: `fsm.json`, `simulation.json` with spikes and metrics, and `raster.png`.
4. **Power analysis.**
   - Estimates the theoretical dynamic energy of the spikes of the last simulation on a possible implementation of the design on the Dynap-SE neuromorphic platform. The formula comes from works on Dynap-SE: previous works showed that spiking logic gates with this same architecture can be implemented there (with other neuron parameters that make the neurons behave as Boolean gates), so it applies to these designs even though they are simulated with the Ideal simulator or SpiNNaker.
   - It uses the energy model used for the spiking ALU (Ayuso-Martinez et al., *Advanced Intelligent Systems*, 2025, Eq. 3, from Risi, 2022) and the energy of each event measured on Dynap-SE by Moradi et al. (2018): P = Σ<sub>n</sub> r<sub>n</sub> [E<sub>spike</sub> + E<sub>en</sub> + N<sub>cores</sub> (E<sub>br</sub> + E<sub>rt</sub>) + N<sub>cam-match</sub> E<sub>pulse</sub>], with 883, 883, 6840, 360 and 324 pJ. Over a simulation, the number of spikes of each neuron replaces its rate, and N<sub>cores</sub> = 1 for a neuron with output synapses. The estimate depends only on the spikes, not on the simulator that produced them.
   - Shows the energy and mean power over the simulated time, where the energy goes (by kind of event and by type of neuron), and the energy per spike and spikes of every neuron. Before a simulation, it shows the energy per spike of every neuron. **Save PDF** prints the tab as it is shown, with the model, and the print dialog saves it as a PDF.
   - Only the dynamic energy of the neurons is counted: the spike generators (inputs and initialisation) and the static power are left out, as in that work.

### State behaviour

Every state is a **spiking latch**: once active, it stays active until one of its outgoing transitions fires. An input event that fires none of its transitions leaves it active.

Transitions are evaluated only on **input events**, the time steps in which at least one input spikes (when the OP neuron fires). A term without positive literals, such as `¬A ∧ ¬B`, therefore fires on every event in which A and B stay silent, that is, whenever another input spikes. It never fires when no input spikes at all.

Several states may be active at start, and if two transitions leaving the same state hold at the same time, both destinations become active (non-determinism). The Design tab shows a note about each of these situations.

The spikes of the latch neurons of every state are recorded, so the state of the machine can be read directly from the simulation (also on SpiNNaker).

### Network construction ([`fsm_snn/builder.py`](fsm_snn_app/fsm_snn/builder.py))

A transition condition is stored as a sum of products: each term is an AND of literals (`X` or `¬X`), and the terms are OR-ed together.

The network is built with the spiking components of the manuscript (Sec. 2.3, Fig. 1) and of the spiking ALU on Dynap-SE (Ayuso-Martinez et al., *Advanced Intelligent Systems*, 2025), and follows the spiking Boolean computing paradigm:

* **Boolean neurons.** Every synapse has weight 1 and a delay of one time step; a neuron fires once when an excitatory spike reaches it, unless an inhibitory spike reaches it at the same time.
* **No overexcitation.** A neuron never receives two excitatory spikes in the same time step: on Dynap-SE such a neuron stops firing. Every neuron has a single excitatory synapse, or several whose spikes can never coincide (as the purple neurons of the ripple-carry adder of the ALU). Inhibition has no such limit: a neuron excited by one synapse and inhibited by several behaves as a NOR gate, the basis of the inhibition-based gates.
* **No constant spike source.** The gates only operate when input spikes arrive (OP), and the state is kept by the latches. The paths that meet at a neuron arrive in the same time step.

| Component (name in the papers) | Neurons | Implementation |
| --- | --- | --- |
| OP, alternative OR gate of the inputs (ALU Fig. 6, complex case of the manuscript) | 1 + 1 per input | One input neuron per input, excited by it and inhibited by the inputs before it, so at most one excites OP. The first one is also the input one step later |
| Delay neurons (yellow) | shared | OP + 1, the delay neuron of the AND gates (only for AND gates without a positive input); chains that repeat an input or OP one step later each |
| NOT gate of an input, Ā (Fig. 1a) | 1 per input that needs it | Excited by OP, inhibited by the input |
| NOT gate of a state, S̄ (complex case) | 1 per source state (none for a state that is never left) | Excited by OP, inhibited by the latch (N1, N2, N3) and, in parallel, by the AND gates that entered the state 2 steps before |
| AND gate of a transition (Fig. 1c) | 1 per term (parallel: latch + bypasses) | A NOR neuron excited by the first positive input of the term (delayed to arrive with OP + 1, as A·S0 in the complex case) or, without one, by OP + 1. When the term negates every input before that one, the input neuron of the Alt. OR (which already excludes them) is that input one step later, if the network is cheaper so; inhibited by the NOT gates of its other positive inputs, by its negated inputs and by S̄ |
| OR gate (Fig. 1b) | 0 or 1 NOR per state | The Set of each state is the OR of the AND gates that enter it. When they can never fire together, N1 itself is the OR (excited by all of them). Otherwise it is inhibition-based: a NOR gate (excited by OP + 2, inhibited by them) inhibits N1, which OP + 3 excites |
| Latch (Fig. 1d) | 3 per state | Set → N1 → ring N2 ↔ N3; N1 inhibits itself and the ring inhibits N1, so a state that is already active is not set again (N2 never receives N1 and N3 together); Reset (the AND gates that leave the state) inhibits N2 and N3. A state active at start is set by the initialisation spike (N1), like any other Set |
| Output, pattern found (internal, optional) | 1 (+ 1 NOR) | OR gate of the transitions entering states flagged as `accepting` in the JSON specification. The editor does not create it; the built-in sequence detectors used by the tests use it |

The OR gates of a machine are direct (**direct** set) when no state can be entered by two AND gates for the same input spikes, nor left and entered at once, which holds for every deterministic machine; otherwise they are inhibition-based. The builder finds out by exploring, exactly and with no size limit, every set of states that can be active together and every combination of input spikes (only the inputs that some condition reads are decided, one at a time), and the Network tab says which one it used. The terms of each condition are split so that they never hold together.

Timing, in time steps after an input event at time step *t*:

| Step | What happens |
| --- | --- |
| *t* + 1 | Input neurons of the alternative OR gate |
| *t* + 2 | OP |
| *t* + 3 | NOT gates (Ā, S̄) and delay neuron |
| *t* + 4 | AND gates (transitions) |
| *t* + 5 | Reset of the source state; direct: Set of the destination (N1: it is active) and Output; inhibition-based: NOR gates |
| *t* + 6 | Direct: ring; inhibition-based: Set (N1: active) and Output |
| *t* + 7 | Inhibition-based: ring |

A state is active while any of its three latch neurons spikes: N1 (the step it is set), then the ring N2 ↔ N3.

Two implementations, chosen in the Network tab and saved with the machine (`"architecture"` in the JSON):

* **Pipelined**, the simple case of the manuscript: each AND gate reads the NOT gate of its source state, which knows the state after the input spikes up to *t* − 3. Correct behaviour is guaranteed for events at least **3 steps** apart, the 3τ of the manuscript (closer events can be simulated, and are reported).
* **Parallel** (default), the **bypasses** of the complex case of the manuscript, so that input spikes may come closer, even on consecutive steps. It stays asynchronous: there is no clock, neurons only spike in response to input spikes, and the state is kept by the latches. An AND gate fires 4 steps after the input spikes it evaluates, but the latch only shows the changes of 3 or more steps before. The NOT gate of the source state also reads the AND gates that entered it 2 steps before, a delay neuron of the Reset says it was left 2 steps before, and the AND gates that left it 1 step before inhibit the term directly. A state entered 1 step before must instead *allow* the term, which no inhibition can do, so each term has up to two AND gates, one per case, and at most one of them fires (the Network tab draws them in a box): *latch*, the NOT gate of the state says it is active and it was not left 1 or 2 steps before; and *bypass t−1*, the state was just entered, excited by the AND gate that entered it. The bypass guard ¬OP(*t*) ∧ OP(*t* − 1) cancels the bypasses when there are no input spikes now. Where the exploration finds that a state can be entered while it is already active, one of the two cases is inhibited by the other's evidence (the cheaper choice); where it can be left and entered at once, entering wins. There is no time for an OR gate in the bypass: where several AND gates can enter a state at once, each term has one bypass per group of them that never fire together, and each group is inhibited by the groups before it (the priority of the Alt. OR), so only one fires and input spikes may still come on consecutive steps. Two more savings come from the exploration: a state that every input event leaves is told so by OP (delayed), instead of by every AND gate that leaves it (the builder keeps whichever network is cheaper), and a state that is active from the start and never left needs no Set, Reset, NOT gate or bypass. [`docs/network_generation.pdf`](docs/network_generation.pdf) (section 4) explains how the bypasses work, with an example.

Since no neuron ever receives two excitatory spikes at once, the result is the same whether inhibition cancels excitation spike for spike or vetoes it: the Ideal simulator and SpiNNaker give the same spikes, and the network keeps working on a platform where overexcitation stops a neuron. Where a signal has to arrive later, it goes through a chain of delay neurons that each repeat the spikes one step later (as in the manuscript), shared by all the synapses that need it, since some platforms do not implement longer delays. The cost is linear in the size of the machine. [`docs/network_generation.pdf`](docs/network_generation.pdf) summarises how the network of any machine is built: steps, layers and generic blocks.

### Code structure (`fsm_snn_app/`)

| File | Role |
| --- | --- |
| `run_app.py` | Starts the local web server |
| `fsm_snn/spec.py` | Machine format, validation, notes and the built-in machines (default machine and tests) |
| `fsm_snn/builder.py` | Machine → netlist of neurons and synapses (the spiking gates and latches) |
| `fsm_snn/codegen.py` | Netlist → readable PyNN code (Network tab) and stand-alone simulation script (Simulate tab) |
| `fsm_snn/files.py` | Design files on disk: folder listing, open and save (Open / Save / recent files) |
| `fsm_snn/web.py`, `static/web.js`, `build_web.py` | Web version: the API run in the browser by Pyodide, and the build of the static site |
| `fsm_snn/inputs.py`, `reference.py`, `analysis.py` | Input spike trains, reference automaton and comparison of the simulation with it |
| `fsm_snn/backends.py`, `pynn_runner.py`, `pipeline.py` | Ideal simulator, PyNN execution (time step, time scale factor, cancellation) and the end-to-end run |
| `fsm_snn/energy.py` | Theoretical dynamic energy and power of the spikes (Power analysis tab) |
| `fsm_snn/server.py` | HTTP API used by the web page |
| `fsm_snn/static/` | Web page: `util.js` (helpers), `expr.js` (Boolean conditions), `viewport.js` (pan and zoom), `editor.js` (Design tab), `network.js` (Network tab), `simulate.js` (Simulate tab), `energy.js` (Power analysis tab), `app.js` (start-up) |

### Validation

[`fsm_snn/reference.py`](fsm_snn_app/fsm_snn/reference.py) implements the same semantics as a software automaton. After each simulation, the set of active states of the network is compared with the reference event by event, and detections are matched within ±1 step. A constant time offset introduced by a simulator is calibrated automatically from the activation of the initial state.

```bash
cd fsm_snn_app
python tests/test_core.py        # built-in machines with random, manual and mixed inputs
python tests/test_pynn_path.py   # PyNN code path and exported script (with a PyNN stand-in)
python tests/test_codegen.py     # the readable PyNN code builds exactly the simulated network
```

### Known limitations

* **Synaptic weights.** Every synapse has weight 1.0 nA (excitatory and inhibitory, as in `experiments/`), with the neuron parameters of Table 1, in [`fsm_snn/pynn_runner.py`](fsm_snn_app/fsm_snn/pynn_runner.py).
* **Time step.** The gates need each neuron to forget its input within one step (Table 1, τ = 0.1 ms, is for a 1 ms step), so for SpiNNaker the time constants and the capacitance of Table 1 are scaled with the time step: the network then gives the same spikes, counted in steps, for any step (checked with 0.1, 0.5, 2 and 5 ms). Below 1 ms SpiNNaker runs slower than real time (`time_scale_factor`). The Ideal simulator gives the same result for any time step.
* **Cancelling.** Any simulation that lasts more than a moment can be cancelled from the page (the Ideal simulator checks every 1000 steps). A SpiNNaker simulation is never killed (that could leave the board unusable): it stops at its next safe point (before the run, between pieces of 2000 steps, or before reading the recordings) and closes sPyNNaker with `sim.end()`. Loading the network on the board, inside the first piece of the run, cannot be interrupted.
* **Implementations.** The pipelined network needs input spikes at least 3τ apart; the parallel one (bypasses) accepts them 1τ apart for any machine, at the cost of more neurons and synapses: for example 35 neurons and 132 synapses for the simple FSM (pipelined: 23 and 60) and 72 and 247 for the pattern detector (pipelined: 54 and 132), delay lines included and spike sources excluded. Both have been checked against the reference automaton with random machines and random trains on the Ideal simulator, and spike for spike against SpiNNaker.
* **Validation on SpiNNaker.** The networks of the designs of the repository give on SpiNNaker exactly the same spikes as on the Ideal simulator, with a 100 % state agreement with the reference automaton.
