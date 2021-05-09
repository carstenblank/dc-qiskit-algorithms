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

   add_draper

More details:

draper_adder
############

.. autofunction:: add_draper

"""
from typing import Optional, Tuple, List, Union

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Gate, Instruction, Qubit
from qiskit.extensions import XGate, CU1Gate

from dc_qiskit_algorithms.Qft import QuantumFourierTransformGate
from . import Qft as qft


class DraperAdderGate(Gate):

    def __init__(self, input_a, input_b, length=None):
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
        length = DraperAdderGate.compute_length(input_a, input_b, length)
        a_01s = a_01s.zfill(length)
        b_01s = b_01s.zfill(length)
        super().__init__("add_draper", num_qubits=2*length, params=[input_a, input_b])
        self.a_01s = a_01s
        self.b_01s = b_01s

    def _define(self):
        rule = []  # type: List[Tuple[Gate, list, list]]

        q = QuantumRegister(len(self.a_01s) + len(self.b_01s), "q")
        qc = QuantumCircuit(q, name=self.name)

        a = q[0:len(self.a_01s)]
        b = q[len(self.a_01s):]

        for i, c in enumerate(self.a_01s):
            if c == '1':
                rule.append((XGate(), [a[i]], []))

        for i, c in enumerate(self.b_01s):
            if c == '1':
                rule.append((XGate(), [b[i]], []))

        rule.append((QuantumFourierTransformGate(len(a)), a, []))

        for b_index in reversed(range(len(b))):
            theta_index = 1
            for a_index in reversed(range(b_index + 1)):
                rule.append((CU1Gate(qft.get_theta(theta_index)), [b[b_index], a[a_index]], []))
                theta_index += 1

        rule.append((QuantumFourierTransformGate(len(a)).inverse(), a, []))

        qc._data = rule.copy()
        self.definition = qc

    def inverse(self):
        return super().inverse()

    @staticmethod
    def compute_length(input_a, input_b, length=None):
        a_01s = "{0:b}".format(input_a)
        b_01s = "{0:b}".format(input_b)
        length = max(len(a_01s), len(b_01s), length if length is not None else 0)
        return length


def add_draper(self, input_a, input_b, qubits, length=None):
    # type: (QuantumCircuit, int, int, Union[QuantumRegister, List[Qubit]], Optional[int]) -> Instruction

    if isinstance(qubits, QuantumRegister):
        qubits = list(qubits)

    return self.append(DraperAdderGate(input_a, input_b, length), qubits, [])


QuantumCircuit.add_draper = add_draper
