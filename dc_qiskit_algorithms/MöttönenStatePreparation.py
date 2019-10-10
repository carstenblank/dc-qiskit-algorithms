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
MöttönenStatePrep
==================

.. currentmodule:: dc_qiskit_algorithms.MöttönenStatePrep

This module implements the state preparation scheme defined by Möttönen et. al.
(http://dl.acm.org/citation.cfm?id=2011670.2011675)

.. autosummary::
   :nosignatures:

   get_alpha_z
   get_alpha_y
   state_prep_möttönen
   state_prep_möttönen_dg
   MöttönenStatePrep

See below the meaning and usage of the functions and the class

get_alpha_z
############

.. autofunction:: get_alpha_z

get_alpha_y
############

.. autofunction:: get_alpha_y

get_alpha_y
############

.. autofunction:: state_prep_möttönen

get_alpha_y
############

.. autofunction:: state_prep_möttönen_dg

MöttönenStatePrep
##################

.. autoclass:: MöttönenStatePrep

"""

import math
from typing import List, Tuple, Union, Optional

import numpy
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.circuit import InstructionSet, CompositeGate, Gate
from scipy import sparse

from .UniformRotation import uniry, unirz


def get_alpha_z(omega, n, k):
    # type: (sparse.dok_matrix, int, int) -> sparse.dok_matrix
    """
    Computes the rotation angles alpha for the z-rotations
    :param omega: the input phase
    :param n: total number of qubits
    :param k: current qubit
    :return: a sparse vector
    """
    alpha_z_k = sparse.dok_matrix((2 ** (n - k), 1), dtype=numpy.float64)

    for (i, _), om in omega.items():
        i += 1
        j = int(numpy.ceil(i * 2 ** (-k)))
        s_condition = 2 ** (k - 1) * (2 * j - 1)
        s_i = 1.0 if i > s_condition else -1.0
        alpha_z_k[j - 1, 0] = alpha_z_k[j - 1, 0] + s_i * om / 2 ** (k - 1)

    return alpha_z_k


def get_alpha_y(a, n, k):
    # type: (sparse.dok_matrix, int, int) -> sparse.dok_matrix
    """
    Computes the rotation angles alpha for the y-rotations
    :param a: the input absolute values
    :param n: total number of qubits
    :param k: current qubit
    :return: a sparse vector
    """
    alpha = sparse.dok_matrix((2**(n - k), 1), dtype=numpy.float64)

    numerator = sparse.dok_matrix((2 ** (n - k), 1), dtype=numpy.float64)
    denominator = sparse.dok_matrix((2 ** (n - k), 1), dtype=numpy.float64)

    for (i, _), e in a.items():
        j = int(math.ceil((i + 1) / 2**k))
        l = (i + 1) - (2*j - 1)*2**(k-1)
        is_part_numerator = 1 <= l <= 2**(k-1)

        if is_part_numerator:
            numerator[j - 1, 0] += e*e
        denominator[j - 1, 0] += e*e

    for (j, _), e in numerator.items():
        numerator[j, 0] = numpy.sqrt(e)
    for (j, _), e in denominator.items():
        denominator[j, 0] = 1/numpy.sqrt(e)

    pre_alpha= numerator.multiply(denominator)  # type: sparse.csr_matrix
    for (j, _), e in pre_alpha.todok().items():
        alpha[j, 0] = 2*numpy.arcsin(e)

    return alpha


# noinspection NonAsciiCharacters
class MöttönenStatePreparationGate(Gate):
    """Uniform rotation Y gate (Möttönen)."""

    def __init__(self, vector):
        # type: (sparse.dok_matrix) -> None
        """
        Create the composite gate for the Möttönen state preparation scheme with an input vector, which registers/qubits
        to apply it to, and the circuit (if any)
        :param vector: the input complex sparse vector
        """
        num_qubits = int(math.log2(vector.shape[0]))
        super().__init__("state_prep_möttönen", num_qubits=num_qubits, params=[])
        self.vector = vector  # type: sparse.dok_matrix

    def _define(self):
        a = sparse.dok_matrix(self.vector.get_shape())  # type: sparse.dok_matrix
        omega = sparse.dok_matrix(self.vector.get_shape())  # type: sparse.dok_matrix

        for (i, j), v in self.vector.items():
            a[i, j] = numpy.absolute(v)
            omega[i, j] = numpy.angle(v)

        qubits = list(QuantumRegister(self.num_qubits, "qubits"))  # type: List[Qubit]

        rule = []  # type: List[Tuple[Gate, List[Qubit], List[Clbit]]]

        rule.extend(self.apply_rot_z(omega, qubits))
        rule.extend(self.apply_rot_y(a, qubits))

        inverse_rule = []
        for inst, qargs, cargs in reversed(rule):
            inverse_rule.append((inst.inverse(), qargs, cargs))

        self._definition = inverse_rule

    @staticmethod
    def apply_rot_y(a, qubits):
        # type: (sparse.dok_matrix, List[Qubit]) -> List[Tuple[Gate, List[Qubit], List[Clbit]]]
        """
        Applies the cascade of y-uniform rotations to the qubits
        :param a: the sparse absolute value vector
        :param qubits: qubits to which the scheme are applied
        :return: None
        """
        rule = []  # type: List[Tuple[Gate, List[Qubit], List[Clbit]]]
        num_qubits = int(math.log2(a.shape[0]))
        for k in range(1, num_qubits + 1):
            alpha_y_k = get_alpha_y(a, num_qubits, k)  # type: sparse.dok_matrix
            control = qubits[k:]
            target = qubits[k - 1]
            rule.append((UniformRotationGate(gate=lambda theta: RYGate(-theta), alpha=alpha_y_k), control + [target], []))

        return rule

    @staticmethod
    def apply_rot_z(omega, qubits):
        # type: (sparse.dok_matrix, List[Qubit]) -> List[Tuple[Gate, List[Qubit], List[Clbit]]]
        """
        Applies the cascade of z-uniform rotations to the qubits
        :param omega: the sparse phase vector
        :param qubits: qubits to which the scheme are applied
        :return: None
        """
        rule = []  # type: List[Tuple[Gate, List[Qubit], List[Clbit]]]
        num_qubits = int(math.log2(omega.shape[0]))
        for k in range(1, num_qubits + 1):
            alpha_z_k = get_alpha_z(omega, num_qubits, k)
            control = qubits[k:]
            target = qubits[k - 1]
            # if len(alpha_z_k) != 0:
            rule.append((UniformRotationGate(gate=lambda theta: RZGate(-theta), alpha=alpha_z_k), control + [target], []))

        return rule

def state_prep_möttönen(self, a, qubits):
    # type: (QuantumCircuit, Union[List[float], sparse.dok_matrix], Union[List[Qubit], QuantumRegister]) -> Instruction
    """
    Convenience function to encapsulate the composite gate of the state preparation
    :param self: Quantum circuit to apply this to
    :param a: the input vector
    :param qubits: the qubits to be transformed
    :return: gate just added
    """
    if isinstance(qubits, QuantumRegister):
        qubits = list(qubits)

    if isinstance(a, sparse.dok_matrix):
        return self.append(MöttönenStatePreparationGate(a), qubits, [])
    else:
        return self.append(MöttönenStatePreparationGate(sparse.dok_matrix([a]).transpose()), qubits)


def state_prep_möttönen_dg(self, a, qubits):
    # type: (QuantumCircuit, Union[List[float], sparse.dok_matrix], Union[List[Qubit], QuantumRegister]) -> Instruction
    """
        Convenience function to encapsulate the composite gate of the dagger of the state preparation
        :param self: Composite Gate or Quantum circuit to apply this to
        :param a: the input vector
        :param qubits: the qubits to be transformed
        :return: gate or instruction set
        """
    return state_prep_möttönen(self, a, qubits).inverse()


QuantumCircuit.state_prep_möttönen = state_prep_möttönen
QuantumCircuit.state_prep_möttönen_dg = state_prep_möttönen_dg
