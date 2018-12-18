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
=======

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
###

.. autofunction:: qft

qft_dg
######

.. autofunction:: qft_dg

"""

import math
from typing import Tuple, List, Union

from qiskit import QuantumCircuit, QuantumRegister, Gate
import qiskit.extensions.standard as standard


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


def qft(qc, q):
    # type: (QuantumCircuit, Union[List[Tuple[QuantumRegister, int]], QuantumRegister]) -> QuantumCircuit
    """
    Applies the Quantum Fourier Transform to q
    :param qc: the circuit to which the qft is applied
    :param q: the quantum register or list of quantum register/index tuples
    :return: the circuit with applied qft
    """
    q_list = []  # type: List[Tuple[QuantumRegister, int]]
    if isinstance(q, QuantumRegister):
        q_list = [q[i] for i in range(q.size)]
    else:
        q_list = q

    unused = q_list.copy()
    for qr in q_list:
        standard.h(qc, qr)
        k = 2
        unused.remove(qr)
        for qj in reversed(unused):
            standard.cu1(qc, get_theta(k), qj, qr)
            k = k + 1
    return qc


def qft_dg(qc, q):
    # type: (QuantumCircuit, Union[List[Tuple[QuantumRegister, int]], QuantumRegister]) -> QuantumCircuit
    """
        Applies the inverse Quantum Fourier Transform to q
        :param qc: the circuit to which the qft_dag is applied
        :param q: the quantum register or list of quantum register/index tuples
        :return: the circuit with applied qft_dag
        """
    qc2 = QuantumCircuit(*qc.get_qregs().values(), *qc.get_cregs().values())
    qft(qc2, q)
    new_data = [op.inverse() for op in reversed(qc2.data)]
    qc2.data = new_data
    qc.extend(qc2)
    return qc
