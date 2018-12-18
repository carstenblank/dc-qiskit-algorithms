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

.. currentmodule:: dc_qiskit_algorithms.UniformRotation

This is the legendary Draper adder (arXiv:quant-ph/0008033).

.. autosummary::
   :nosignatures:

   draper_adder

More details:

draper_adder
############

.. autofunction:: draper_adder

"""
import logging
from itertools import tee
from typing import List, Tuple, Union, Callable

import numpy as np
from qiskit import CompositeGate, QuantumRegister, QuantumCircuit, InstructionSet
from qiskit._register import Register
from qiskit.extensions.standard import ry, rz, cx
from scipy import sparse

log = logging.getLogger('UniformRotation')


def binary_codes(number_qubits: int) -> List[int]:
    N = int(2**number_qubits)
    return list(range(N))


def gray_code(number: int) -> int:
    return (number >> 1) ^ number


def matrix_M_entry(row: int, col: int) -> float:
    # b_and_g = bcodes[row] & gcodes[col]
    b_and_g = row & gray_code(col)
    sum_of_ones = 0
    while b_and_g > 0:
        if b_and_g & 0b1:
            sum_of_ones += 1
        b_and_g = b_and_g >> 1
    return (-1)**sum_of_ones


def compute_theta(alpha: sparse.dok_matrix) -> sparse.dok_matrix:
    k = np.log2(alpha.shape[0])
    factor = 2**(-k)

    theta = sparse.dok_matrix(alpha.shape, dtype=np.float64)  # type: sparse.dok_matrix
    for row in range(alpha.shape[0]):
        # Use transpose of M:
        entry = sum([matrix_M_entry(col, row) * a for (col, _), a in alpha.items()])
        entry *= factor
        if abs(entry) > 1e-6:
            theta[row, 0] = entry

    return theta


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


class UniformRotationGate(CompositeGate):
    """Uniform rotation Y gate (Möttönen)."""

    def __init__(self, gate: Callable[[CompositeGate, float, Tuple[QuantumRegister,int]], InstructionSet],
                 alpha: sparse.dok_matrix, control_qubits: List[Tuple[QuantumRegister, int]],
                 tgt: Tuple[QuantumRegister, int], circ=None):
        """Create new cu1 gate."""
        super().__init__("uni_rot_" + str(gate), [], control_qubits + [tgt], circ)

        theta = compute_theta(alpha)  # type: sparse.dok_matrix

        gray_code_rank = len(control_qubits)
        if gray_code_rank == 0:
            gate(self, theta[0, 0], tgt)
            return

        from sympy.combinatorics.graycode import GrayCode
        gc = GrayCode(gray_code_rank)  # type: GrayCode

        current_gray = gc.current
        for i in range(gc.selections):
            gate(self, theta[i, 0], tgt)
            next_gray = gc.next(i + 1).current

            control_index = int(np.log2(int(current_gray, 2) ^ int(next_gray, 2)))
            cx(self, control_qubits[control_index], tgt)

            current_gray = next_gray


def uni_rot(self, gate: Callable[[CompositeGate, float, Tuple[QuantumRegister,int]], InstructionSet],
            alpha: Union[List[float], sparse.dok_matrix],
            control_qubits: Union[List[Tuple[QuantumRegister, int]],QuantumRegister],
            tgt: Union[Tuple[QuantumRegister, int], QuantumRegister]):

    if isinstance(control_qubits, QuantumRegister):
        instructions = InstructionSet()
        ctrs = [(control_qubits, j) for j in range(control_qubits.size)]
        if isinstance(tgt, QuantumRegister):
            for j in range(tgt.size):
                instructions.add(uni_rot(self, gate, alpha, ctrs, (tgt, j)))
        else:
            instructions.add(uni_rot(self, gate, alpha, ctrs, tgt))
        return instructions

    if not isinstance(alpha, sparse.dok_matrix):
        alpha = sparse.dok_matrix([alpha]).transpose()

    self._check_qubit(tgt)
    for qb in control_qubits:
        self._check_qubit(qb)
    return self._attach(UniformRotationGate(gate, alpha, control_qubits, tgt, self))


def uni_rot_dg(self, gate: Callable[[CompositeGate, float, Tuple[QuantumRegister,int]], InstructionSet],
               alpha: Union[List[float], sparse.dok_matrix],
               control_qubits: Union[List[Tuple[QuantumRegister, int]],QuantumRegister],
               tgt: Union[Tuple[QuantumRegister, int], QuantumRegister]):
    return uni_rot(self, gate, alpha, control_qubits, tgt).inverse()


def unirz(self, alpha: Union[List[float], sparse.dok_matrix],
          control_qubits: Union[List[Tuple[QuantumRegister, int]],QuantumRegister],
          tgt: Union[Tuple[QuantumRegister, int], QuantumRegister]):

    def rz_möttönen2ibm(cg: CompositeGate, theta: float, qreg: Tuple[QuantumRegister, int]) -> InstructionSet:
        return rz(cg, -theta, qreg)

    return uni_rot(self, rz_möttönen2ibm, alpha, control_qubits, tgt)


def unirz_dg(self, alpha: Union[List[float], sparse.dok_matrix],
             control_qubits: Union[List[Tuple[QuantumRegister, int]],QuantumRegister],
          tgt: Union[Tuple[QuantumRegister, int], QuantumRegister]):
    return unirz(self, alpha, control_qubits, tgt).inverse()


def uniry(self, alpha: Union[List[float], sparse.dok_matrix],
          control_qubits: Union[List[Tuple[QuantumRegister, int]],QuantumRegister],
          tgt: Union[Tuple[QuantumRegister, int], QuantumRegister]):

    def ry_möttönen2ibm(cg: CompositeGate, theta: float, qreg: Tuple[QuantumRegister, int]) -> InstructionSet:
        return ry(cg, -theta, qreg)

    return uni_rot(self, ry_möttönen2ibm, alpha, control_qubits, tgt)


def uniry_dg(self, alpha: Union[List[float], sparse.dok_matrix],
             control_qubits: Union[List[Tuple[QuantumRegister, int]],QuantumRegister],
          tgt: Union[Tuple[QuantumRegister, int], QuantumRegister]):
    return self.uniry(alpha, control_qubits, tgt).inverse()


def cnry(self, theta: float, control_qubits: Union[List[Tuple[QuantumRegister, int]],QuantumRegister],
         tgt: Union[Tuple[QuantumRegister, int], QuantumRegister]):
    length = 2**len(control_qubits)
    alpha = sparse.dok_matrix((length, 1), dtype=np.float64)
    alpha[-1] = theta
    return self.uniry(alpha, control_qubits, tgt)


def cnry_dg(self, theta: float, control_qubits: Union[List[Tuple[QuantumRegister, int]],QuantumRegister],
                 tgt: Union[Tuple[QuantumRegister, int], QuantumRegister]):
    return self.cnry(theta, control_qubits, tgt).inverse()


class MultiControlledXGate(CompositeGate):
    """Multi-Controlled X-Gate (via Möttönen)."""

    def __init__(self, conditial_case: int, control_qubits: Union[List[Tuple[Register, int]],QuantumRegister],
         tgt: Union[Tuple[Register, int], QuantumRegister], circ=None):

        super().__init__("ccx_uni_rot", [conditial_case], control_qubits + [tgt], circ)

        length = 2 ** len(control_qubits)
        alpha = sparse.dok_matrix((length, 1), dtype=np.float64)
        alpha[conditial_case] = np.pi
        from qiskit.extensions.standard import h
        h(self, tgt)
        uniry(self, alpha, control_qubits, tgt)
        h(self, tgt)

    def __repr__(self):
        return "{}({}) {};".format(self.name, self.param, ["{}[{}]".format(q.name, i) for q, i in self.arg])


def ccx(self, conditial_case: int, control_qubits: Union[List[Tuple[Register, int]],QuantumRegister],
         tgt: Union[Tuple[Register, int], QuantumRegister]):

    if isinstance(control_qubits, QuantumRegister):
        instructions = InstructionSet()
        ctrs = [(control_qubits, j) for j in range(control_qubits.size)]
        if isinstance(tgt, QuantumRegister):
            for j in range(tgt.size):
                instructions.add(ccx(self, conditial_case, control_qubits, (tgt, j)))
        else:
            instructions.add(ccx(self, conditial_case, control_qubits, tgt))
        return instructions

    self._check_qubit(tgt)
    for qb in control_qubits:
        self._check_qubit(qb)
    return self._attach(MultiControlledXGate(conditial_case, control_qubits, tgt, self))


def ccx_dg(self, conditial_case: int, control_qubits: Union[List[Tuple[QuantumRegister, int]],QuantumRegister],
         tgt: Union[Tuple[QuantumRegister, int], QuantumRegister]):
    return self.ccx(conditial_case, control_qubits, tgt).inverse()


QuantumCircuit.uniry = uniry
QuantumCircuit.uniry_dg = uniry_dg
CompositeGate.uniry = uniry
CompositeGate.uniry_dg = uniry_dg

QuantumCircuit.unirz = unirz
QuantumCircuit.unirz_dg = unirz_dg
CompositeGate.unirz = unirz
CompositeGate.unirz_dg = unirz_dg

QuantumCircuit.cnry = cnry
QuantumCircuit.cnry_dg = cnry_dg
CompositeGate.cnry = cnry
CompositeGate.cnry_dg = cnry_dg