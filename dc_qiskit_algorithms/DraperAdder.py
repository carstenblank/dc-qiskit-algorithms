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
from typing import Optional, Tuple

import qiskit
from qiskit import QuantumCircuit
from qiskit.extensions import standard

from . import Qft as qft


def draper_adder(input_a, input_b, length = None):
    # type: (int, int, Optional[int]) -> Tuple[QuantumCircuit, float]
    """
    The Draper adder (arXiv:quant-ph/0008033), provide a and b and make sure to define a size of
    a register that can hold a or b
    :param input_a: integer a
    :param input_b: integer b
    :param length: size of qubit registers
    :return: tuple of the circuit and the length of the register
    """
    a_01s = "{0:b}".format(input_a)
    b_01s = "{0:b}".format(input_b)
    length = max(len(a_01s), len(b_01s), length if length is not None else 0)
    a_01s = a_01s.zfill(length)
    b_01s = b_01s.zfill(length)

    a = qiskit.QuantumRegister(len(a_01s), "a")
    b = qiskit.QuantumRegister(len(b_01s), "b")
    c_a = qiskit.ClassicalRegister(len(a_01s), "c_a")
    c_b = qiskit.ClassicalRegister(len(b_01s), "c_b")
    qc = qiskit.QuantumCircuit(a, b, c_a, c_b, name='draper adder')

    standard.barrier(qc, a, b)

    for i, c in enumerate(a_01s):
        if c == '1':
            standard.x(qc, a[i])

    for i, c in enumerate(b_01s):
        if c == '1':
            standard.x(qc, b[i])

    qft.qft(qc, a)

    for b_index in reversed(range(b.size)):
        theta_index = 1
        for a_index in reversed(range(b_index + 1)):
            standard.cu1(qc, qft.get_theta(theta_index), b[b_index], a[a_index])
            theta_index += 1

    qft.qft_dg(qc, a)

    standard.barrier(qc, a, b)

    qc.measure(a, c_a)
    qc.measure(b, c_b)

    return qc, 2**length
