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
UniformRotation
================

.. currentmodule:: dc_qiskit_algorithms.UniformRotation

This module implements the uniform rotation gate defined by Möttönen et. al.
(10.1103/PhysRevLett.93.130502).

There are many convenience functions also being monkey patched.

.. autosummary::
   :nosignatures:

   binary_codes
   gray_code
   matrix_M_entry
   compute_theta
   pairwise
   UniformRotationGate
   uni_rot
   uni_rot_dg
   unirz
   unirz_dg
   uniry
   uniry_dg
   MultiControlledXGate
   ccx
   ccx_dg

Here are the details:

binary_codes
#############

.. autofunction:: binary_codes

gray_code
##########

.. autofunction:: gray_code

matrix_M_entry
###############

.. autofunction:: matrix_M_entry

compute_theta
##############

.. autofunction:: compute_theta

pairwise
#########

.. autofunction:: pairwise

UniformRotationGate
####################

.. autoclass:: UniformRotationGate

uni_rot
########

.. autofunction:: uni_rot

uni_rot_dg
############

.. autofunction:: uni_rot_dg

unirz
######

.. autofunction:: unirz

unirz_dg
#########

.. autofunction:: unirz_dg

uniry
######

.. autofunction:: uniry

uniry_dg
#########

.. autofunction:: uniry_dg

MultiControlledXGate
#####################

.. autoclass:: MultiControlledXGate

ccx
#####

.. autofunction:: ccx

ccx_dg
#######

.. autofunction:: ccx_dg

"""
import logging
from itertools import tee
from typing import List, Tuple, Union, Callable, Iterable

import numpy as np
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.circuit import Gate, Instruction, Qubit, Clbit
from qiskit.extensions import CnotGate, HGate, standard, RZGate, RYGate
from scipy import sparse

log = logging.getLogger('UniformRotation')


# noinspection PyPep8Naming
def binary_codes(number_qubits):
    # type: (int) -> List[int]
    """
    Convenience function to get a list of numbers from 0 to 2**number_qubits - 1
    :param number_qubits: exponent
    :return: list of numbers
    """
    N = int(2**number_qubits)
    return list(range(N))


def gray_code(number):
    # type: (int) -> int
    """
    Cyclic Gray Code of number
    :param number: input number
    :return: Gray Code
    """
    return (number >> 1) ^ number


# noinspection PyPep8Naming
def matrix_M_entry(row, col):
    # type: (int, int) -> float
    """
    The matrix for the angle computation
    :param row: row number (one based!)
    :param col: column number (one based!)
    :return: matrix entry
    """
    b_and_g = row & gray_code(col)
    sum_of_ones = 0
    while b_and_g > 0:
        if b_and_g & 0b1:
            sum_of_ones += 1
        b_and_g = b_and_g >> 1
    return (-1)**sum_of_ones


def compute_theta(alpha):
    # type: (sparse.dok_matrix) -> sparse.dok_matrix
    """
    Compute the rotational angles from alpha
    :param alpha: the input uniform rotation angles
    :return: the single qubit rotation angles
    """
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
    # type: (Iterable) -> Iterable[Tuple]
    """
    Calculates pairwise consecutive pairs of an iterable
    s -> (s0,s1), (s1,s2), (s2, s3), ...
    :param iterable: any iterable
    :return: an iterable of tuples
    """
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


class UniformRotationGate(Gate):
    """Uniform rotation gate (Möttönen)."""

    def __init__(self, gate, alpha):
        # type: (Callable[[float], Gate], sparse.dok_matrix) -> None
        """
        Create new uniform rotation gate.
        :param gate: A single qubit rotation gate (typically rx, ry, rz)
        :param alpha: The conditional rotation angles
        """
        number_of_control_qubits = int(np.ceil(np.log2(alpha.shape[0])))
        super().__init__("uni_rot_" + gate(0).name, num_qubits=number_of_control_qubits + 1, params=[])
        self.alpha = alpha  # type: sparse.dok_matrix
        self.gate = gate  # type: Callable[[float], Gate]

    def _define(self):

        q = QuantumRegister(self.num_qubits, "q")
        rule = []  # type: List[Tuple[Gate, List[Qubit], List[Clbit]]]

        theta = compute_theta(self.alpha)  # type: sparse.dok_matrix

        gray_code_rank = self.num_qubits - 1
        if gray_code_rank == 0:
            rule.append((self.gate(theta[0, 0]), [q[0]], []))
            self._definition = rule.copy()
            return

        from sympy.combinatorics.graycode import GrayCode
        gc = GrayCode(gray_code_rank)  # type: GrayCode

        current_gray = gc.current
        for i in range(gc.selections):
            rule.append((self.gate(theta[i, 0]), [q[-1]], []))
            next_gray = gc.next(i + 1).current

            control_index = int(np.log2(int(current_gray, 2) ^ int(next_gray, 2)))
            rule.append((CnotGate(), [q[control_index], q[-1]], []))

            current_gray = next_gray

        self._definition = rule.copy()


def uni_rot(self, rotation_gate, alpha, control_qubits, tgt):
    # type: (QuantumCircuit, Callable[[float], Gate], Union[List[float], sparse.dok_matrix], Union[List[Qubit],QuantumRegister], Union[Qubit, QuantumRegister]) -> Instruction
    """
    Apply a generic uniform rotation with rotation gate.
    :param self: either a composite gate or a circuit
    :param rotation_gate: A single qubit rotation gate
    :param alpha: conditional rotation angles
    :param control_qubits: control qubits
    :param tgt: target
    :return: applied composite gate or circuit
    """
    if not isinstance(alpha, sparse.dok_matrix):
        alpha = sparse.dok_matrix([alpha]).transpose()

    if isinstance(tgt, QuantumRegister):
        tgt = tgt[0]
    if isinstance(control_qubits, QuantumRegister):
        control_qubits = list(control_qubits)

    return self.append(UniformRotationGate(rotation_gate, alpha), control_qubits + [tgt], [])


def uni_rot_dg(self, rotation_gate, alpha, control_qubits, tgt):
    # type: (QuantumCircuit, Callable[[float], Gate], Union[List[float], sparse.dok_matrix], Union[List[Qubit],QuantumRegister], Union[Qubit, QuantumRegister]) -> Instruction
    """
    Apply the dagger (inverse) of a generic uniform rotation with rotation gate.
    :param self: either a composite gate or a circuit
    :param rotation_gate: A single qubit rotation gate
    :param alpha: conditional rotation angles
    :param control_qubits: control qubits
    :param tgt: target
    :return: applied composite gate or circuit
    """
    return uni_rot(self, rotation_gate, alpha, control_qubits, tgt).inverse()


def unirz(self, alpha, control_qubits, tgt):
    # type: (QuantumCircuit, Union[List[float], sparse.dok_matrix], Union[List[Qubit],QuantumRegister], Union[Qubit, QuantumRegister]) -> Instruction
    """
    Apply a uniform rotation around z.
    :param self: either a composite gate or a circuit
    :param alpha: conditional rotation angles
    :param control_qubits: control qubits
    :param tgt: target
    :return: applied composite gate or circuit
    """
    return uni_rot(self, RZGate, alpha, control_qubits, tgt)


def unirz_dg(self, alpha, control_qubits, tgt):
    # type: (QuantumCircuit, Union[List[float], sparse.dok_matrix], Union[List[Qubit],QuantumRegister], Union[Qubit, QuantumRegister]) -> Instruction
    """
    Apply dagger (inverse) of a uniform rotation around z.
    :param self: either a composite gate or a circuit
    :param alpha: conditional rotation angles
    :param control_qubits: control qubits
    :param tgt: target
    :return: applied composite gate or circuit
    """
    return unirz(self, alpha, control_qubits, tgt).inverse()


def uniry(self, alpha, control_qubits, tgt):
    # type: (QuantumCircuit, Union[List[float], sparse.dok_matrix], Union[List[Qubit],QuantumRegister], Union[Qubit, QuantumRegister]) -> Instruction
    """
    Apply a uniform rotation around y.
    :param self: either a composite gate or a circuit
    :param alpha: conditional rotation angles
    :param control_qubits: control qubits
    :param tgt: target
    :return: applied composite gate or circuit
    """
    return uni_rot(self, RYGate, alpha, control_qubits, tgt)


def uniry_dg(self, alpha, control_qubits, tgt):
    # type: (QuantumCircuit, Union[List[float], sparse.dok_matrix], Union[List[Qubit],QuantumRegister], Union[Qubit, QuantumRegister]) -> Instruction
    """
    Apply the dagger (inverse) of a uniform rotation around y.
    :param self: either a composite gate or a circuit
    :param alpha: conditional rotation angles
    :param control_qubits: control qubits
    :param tgt: target
    :return: applied composite gate or circuit
    """
    return uniry(self, alpha, control_qubits, tgt).inverse()


def cnry(self, theta, control_qubits, tgt):
    # type: (QuantumCircuit, float, Union[List[Qubit],QuantumRegister], Union[Qubit, QuantumRegister]) -> Instruction
    """
    Apply a multiple controlled y rotation on the target qubit.
    :param self: either a composite gate or a circuit
    :param theta: rotation angle
    :param control_qubits: control qubits
    :param tgt: target
    :return: applied composite gate or circuit
    """
    length = 2**len(control_qubits)
    alpha = sparse.dok_matrix((length, 1), dtype=np.float64)
    alpha[-1] = theta
    return uniry(self, alpha, control_qubits, tgt)


def cnry_dg(self, theta, control_qubits, tgt):
    # type: (QuantumCircuit, float, Union[List[Qubit],QuantumRegister], Union[Qubit, QuantumRegister]) -> Instruction
    """
    Apply the dagger (inverse) of a multiple controlled y rotation on the target qubit.
    :param self: either a composite gate or a circuit
    :param theta: rotation angle
    :param control_qubits: control qubits
    :param tgt: target
    :return: applied composite gate or circuit
    """
    return cnry(self, theta, control_qubits, tgt).inverse()


class MultiControlledXGate(Gate):
    """Multi-Controlled X-Gate (via Möttönen)."""

    def __init__(self, conditional_case, control_qubits):
        # type: (int, int) -> None
        """
        Create a new multi-controlled X gate according to the conditional (binary) case
        :param conditional_case: binary representation of 0/1 control case
        :param control_qubits: control qubits
        """
        super().__init__("ccx_uni_rot", params=[conditional_case], num_qubits=control_qubits + 1)
        self.conditional_case = conditional_case
        self.control_qubits = control_qubits

    def _define(self):
        q = QuantumRegister(self.num_qubits, "q")
        rule = []  # type: List[Tuple[Gate, List[Qubit], List[Clbit]]]

        length = 2 ** self.control_qubits
        alpha = sparse.dok_matrix((length, 1), dtype=np.float64)
        alpha[self.conditional_case] = np.pi
        rule.append((HGate(), [q[-1]], []))
        rule.append((UniformRotationGate(standard.RYGate, alpha), list(q), []))
        rule.append((HGate(), [q[-1]], []))

        self._definition = rule.copy()


def ccx_uni_rot(self, conditional_case, control_qubits, tgt):
    # type: (QuantumCircuit, int, Union[List[Qubit], QuantumRegister], Union[Qubit, QuantumRegister]) -> Instruction
    """
    Apply a multi-controlled X gate depending on conditional binary representation
    :param self: either a composite gate or a circuit
    :param conditional_case: the controlled case (1 or 0) in binary
    :param control_qubits: control qubits
    :param tgt: target
    :return: applied composite gate or circuit
    """
    if isinstance(tgt, QuantumRegister):
        tgt = tgt[0]
    if isinstance(control_qubits, QuantumRegister):
        control_qubits = list(control_qubits)

    return self.append(MultiControlledXGate(conditional_case, len(control_qubits)), control_qubits + [tgt])


def ccx_uni_rot_dg(self, conditional_case, control_qubits, tgt):
    # type: (QuantumCircuit, int, Union[List[Qubit],QuantumRegister], Union[Qubit, QuantumRegister]) -> Instruction
    """
    Apply the dagger (inverse) a multi-controlled X gate depending on conditional binary representation
    :param self: either a composite gate or a circuit
    :param conditional_case: the controlled case (1 or 0) in binary
    :param control_qubits: control qubits
    :param tgt: target
    :return: applied composite gate or circuit
    """
    return ccx_uni_rot(self, conditional_case, control_qubits, tgt).inverse()


QuantumCircuit.uniry = uniry
QuantumCircuit.uniry_dg = uniry_dg

QuantumCircuit.unirz = unirz
QuantumCircuit.unirz_dg = unirz_dg

QuantumCircuit.cnry = cnry
QuantumCircuit.cnry_dg = cnry_dg

QuantumCircuit.ccx_uni_rot = ccx_uni_rot
QuantumCircuit.ccx_uni_rot_dg = ccx_uni_rot_dg