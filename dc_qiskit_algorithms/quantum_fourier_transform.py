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
Qft
====

.. currentmodule:: dc_qiskit_algorithms.Qft

Here are a couple of functions that implement the quantum fourier transform:

.. autosummary::
   :nosignatures:

   get_theta
   qft
   qft_dg

See below for a description of the different functions to apply a quantum fourier transform

get_theta
##########

.. autofunction:: get_theta

qft
####

.. autofunction:: qft

qft_dg
#######

.. autofunction:: qft_dg

"""

import math
from typing import Tuple, List, Union  # pylint: disable=unused-argument

import qiskit.extensions.standard as standard
from qiskit import QuantumRegister, QuantumCircuit


def get_theta(k):
    # type: (int) -> float
    """
    As the conditional rotations are defined by a parameter k we have a
    convenience function for this: theta = (+/-) 2pi/2^(|k|)
    :param k: the integer for the angle
    :return: the angle
    """
    sign = -1 if k < 0 else 1
    lam = sign * 2 * math.pi * 2**(-abs(k))
    return lam


def qft(q_circuit, q_register):
    # type: (QuantumCircuit, Union[List[Tuple[QuantumRegister, int]], QuantumRegister]) -> QuantumCircuit
    """
    Applies the Quantum Fourier Transform to q
    :param q_circuit: the circuit to which the qft is applied
    :param q_register: the quantum register or list of quantum register/index tuples
    :return: the circuit with applied qft
    """
    q_list = []  # type: List[Tuple[QuantumRegister, int]]
    if isinstance(q_register, QuantumRegister):
        q_list = [q_register[i] for i in range(q_register.size)]
    else:
        q_list = q_register

    unused = q_list.copy()
    for q_regs in q_list:
        standard.h(q_circuit, q_regs)
        k = 2
        unused.remove(q_regs)
        for q_j in unused:
            standard.cu1(q_circuit, get_theta(k), q_j, q_regs)
            k = k + 1
    return q_circuit


def qft_dg(q_circuit, q_register):
    # type: (QuantumCircuit, Union[List[Tuple[QuantumRegister, int]], QuantumRegister]) -> QuantumCircuit
    """
        Applies the inverse Quantum Fourier Transform to q
        :param q_circuit: the circuit to which the qft_dag is applied
        :param q_register: the quantum register or list of quantum register/index tuples
        :return: the circuit with applied qft_dag
        """
    q_circuit_2 = QuantumCircuit(*q_circuit.qregs, *q_circuit.cregs)
    qft(q_circuit_2, q_register)
    new_data = [op.inverse() for op in reversed(q_circuit_2.data)]
    q_circuit_2.data = new_data
    q_circuit.extend(q_circuit_2)
    return q_circuit
