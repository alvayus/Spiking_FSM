"""FSM -> SNN generator based on the spiking Boolean computing paradigm.

Builds spiking finite state machines from the spiking NOT/AND/OR gates and
latches of Ayuso-Martinez et al. (Fig. 1 of the manuscript), simulates them
with PyNN (SpiNNaker, NEST, Brian2) or with a built-in ideal simulator, and
validates the result against a software reference automaton.
"""

__version__ = "0.1.0"
