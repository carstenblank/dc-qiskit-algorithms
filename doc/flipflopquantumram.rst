Flip-Flop Quantum RAM
======================

This module implements the state preparation scheme called FFQRAM see https://arxiv.org/abs/1901.02362.

Each DB has entries that are created by controlled rotations. The final step is a measurement to cancel out
the wrong branch. This makes the algorithm probabilistic in its nature.

.. automodule:: dc_qiskit_algorithms.FlipFlopQuantumRam
    :members:
    :undoc-members:
    :show-inheritance:
