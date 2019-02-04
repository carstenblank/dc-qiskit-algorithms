# Copyright 2018 Carsten Blank
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""
DraperAdder
===========

.. currentmodule:: dc_qiskit_algorithms.DraperAdder

This is the legendary Draper adder (arXiv:quant-ph/0008033).

.. autosummary::
   :nosignatures:

   draper_adder

More details:

draper_adder
############

.. autofunction:: draper_adder

"""
from typing import Optional, Tuple # pylint: disable=unused-argument

import qiskit
from qiskit import QuantumCircuit # pylint: disable=unused-argument
from qiskit.circuit.measure import measure
from qiskit.extensions import standard

from . import quantum_fourier_transform as qft


def draper_adder(input_a, input_b, length=None, with_barriers=False, with_measurement=True, pre_circuit=None):
    # type: (int, int, Optional[int], bool, bool, Optional[QuantumCircuit]) -> Tuple[QuantumCircuit, float]
    """
    The Draper adder (arXiv:quant-ph/0008033), provide a and b and make sure to define a size of
    a register that can hold a or b.
    A new circuit is created with registers a and b of necessary size. Set to False if you need
    to work with the results afterwards.
    A pre-circuit may be provided, but it must use quantum registers with the name 'a', 'b' and
    correct size size.

    :param input_a: integer a
    :param input_b: integer b
    :param length: size of qubit registers
    :param with_barriers: includes barriers between QFT - adder - Qft_dg. Defaults to False
    :param with_measurement: includes measurements if True (default)
    :param pre_circuit: A circuit to be used before the addition is done. Will cause an error if not compatible.
    :return: tuple of the circuit and the length of the register
    """
    a_01s = "{0:b}".format(input_a)
    b_01s = "{0:b}".format(input_b)
    length = max(len(a_01s), len(b_01s), length if length is not None else 0)
    a_01s = a_01s.zfill(length)
    b_01s = b_01s.zfill(length)

    a = qiskit.QuantumRegister(len(a_01s), "a")
    b = qiskit.QuantumRegister(len(b_01s), "b")
    qc = qiskit.QuantumCircuit(a, b, name='draper adder')

    # Will only be used if measurements are taken
    c_a = qiskit.ClassicalRegister(len(a_01s), "c_a")
    c_b = qiskit.ClassicalRegister(len(b_01s), "c_b")
    if with_measurement:
        qc.add_register(c_a)
        qc.add_register(c_b)

    if pre_circuit:
        qc += pre_circuit

    if with_barriers:
        standard.barrier(qc, a, b)

    for i, c in enumerate(a_01s):
        if c == '1':
            standard.x(qc, a[i])

    for i, c in enumerate(b_01s):
        if c == '1':
            standard.x(qc, b[i])

    if with_barriers:
        standard.barrier(qc, a, b)

    qft.qft(qc, a)

    if with_barriers:
        standard.barrier(qc, a, b)

    for b_index in reversed(range(b.size)):
        theta_index = 1
        for a_index in reversed(range(b_index + 1)):
            standard.cu1(qc, qft.get_theta(theta_index), b[b_index], a[a_index])
            theta_index += 1

    if with_barriers:
        standard.barrier(qc, a, b)

    qft.qft_dg(qc, a)

    if with_barriers:
        standard.barrier(qc, a, b)

    if with_measurement:
        measure(qc, a, c_a)
        measure(qc, b, c_b)

    return qc, 2**length
