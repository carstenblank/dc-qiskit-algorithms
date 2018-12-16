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


import math
from typing import List, Tuple, Union

import numpy
from qiskit import CompositeGate, QuantumRegister, QuantumCircuit, InstructionSet
from scipy import sparse

from .UniformRotation import uniry, unirz


def get_alpha_z(omega, n: int, k: int) -> sparse.dok_matrix:
    alpha_z_k = sparse.dok_matrix((2 ** (n - k), 1), dtype=numpy.float64)

    for (i, _), om in omega.items():
        i += 1
        j = int(numpy.ceil(i * 2 ** (-k)))
        s_condition = 2 ** (k - 1) * (2 * j - 1)
        s_i = 1.0 if i > s_condition else -1.0
        alpha_z_k[j - 1, 0] = alpha_z_k[j - 1, 0] + s_i * om / 2 ** (k - 1)

    return alpha_z_k


def get_alpha_y(a: sparse.dok_matrix, n: int, k: int) -> sparse.dok_matrix:
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


class MöttönenStatePrep(CompositeGate):
    """Uniform rotation Y gate (Möttönen)."""

    def __init__(self, vector: sparse.dok_matrix, qubits: List[Tuple[QuantumRegister, int]], circ=None):
        """Create new cu1 gate."""
        super().__init__("state_prep_möttönen", [], qubits, circ)
        a = sparse.dok_matrix(vector.get_shape())  # type: sparse.dok_matrix
        omega = sparse.dok_matrix(vector.get_shape())  # type: sparse.dok_matrix
        for (i, j), v in vector.items():
            a[i, j] = numpy.absolute(v)
            omega[i, j] = numpy.angle(v)
        self.apply_rot_z(omega, qubits)
        self.apply_rot_y(a, qubits)
        self.inverse()

    def apply_rot_y(self, a: sparse.dok_matrix, qubits: List[Tuple[QuantumRegister, int]]):
        n = int(math.log2(a.get_shape()[0]))
        for k in range(1, n + 1):
            alpha_y_k = get_alpha_y(a, n, k)  # type: sparse.dok_matrix
            control = qubits[k:]
            target = qubits[k - 1]
            uniry(self, alpha_y_k, control, target)

    def apply_rot_z(self, omega: sparse.dok_matrix, qubits: List[Tuple[QuantumRegister, int]]):
        n = int(math.log2(omega.get_shape()[0]))
        for k in range(1, n + 1):
            alpha_z_k = get_alpha_z(omega, n, k)
            control = qubits[k:]
            target = qubits[k - 1]
            if len(alpha_z_k) != 0:
                unirz(self, alpha_z_k, control, target)


def state_prep_möttönen(self, a: Union[List[float], sparse.dok_matrix],
                        qubits: Union[List[Tuple[QuantumRegister, int]], QuantumRegister]):
    if isinstance(qubits, QuantumRegister):
        instructions = InstructionSet()
        qb = [(qubits, j) for j in range(qubits.size)]
        instructions.add(self.state_prep_möttönen(a, qb))
        return instructions

    for qb in qubits:
        self._check_qubit(qb)
    if isinstance(a, sparse.dok_matrix):
        return self._attach(MöttönenStatePrep(a, qubits, self))
    else:
        return self._attach(MöttönenStatePrep(sparse.dok_matrix([a]).transpose(), qubits, self))


def state_prep_möttönen_dg(self, a: List[float], qubits: Union[List[Tuple[QuantumRegister, int]], QuantumRegister]):
    return self.state_prep_möttönen_dg(a, qubits).inverse()


QuantumCircuit.state_prep_möttönen = state_prep_möttönen
QuantumCircuit.state_prep_möttönen_dg = state_prep_möttönen_dg
CompositeGate.state_prep_möttönen = state_prep_möttönen
CompositeGate.state_prep_möttönen_dg = state_prep_möttönen_dg
