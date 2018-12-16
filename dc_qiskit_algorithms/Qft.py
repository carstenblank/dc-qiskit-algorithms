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
from typing import Tuple, List

from qiskit import QuantumCircuit, QuantumRegister, Gate
import qiskit.extensions.standard as standard


def get_theta(k: int) -> float:
    sign = -1 if k < 0 else 1
    lam = sign * 2 * math.pi * 2**(-abs(k))
    return lam


def qft_reg(qc: QuantumCircuit, q: QuantumRegister):
    q_list = [q[i] for i in range(q.size)]  # type: List[Tuple[QuantumRegister, int]]
    return qft(qc, q_list)


def qft(qc: QuantumCircuit, q: List[Tuple[QuantumRegister, int]]):
    unused = q.copy()
    for qr in q:
        standard.h(qc, qr)
        # print("H on %s[%d]" % qr)
        k = 2
        unused.remove(qr)
        for qj in reversed(unused):
            standard.cu1(qc, get_theta(k), qj, qr)
            # print("cR_%d on %s[%d] controlled by %s[%d]" % (k, qr[0].name, qr[1], qj[0].name, qj[1]))
            k = k + 1
    # print("qft done")
    return qc


def qft_dg_reg(qc: QuantumCircuit, q: QuantumRegister):
    q_list= [q[i] for i in range(q.size)]  # type: List[Tuple[QuantumRegister, int]]
    return qft_dg(qc, q_list)


def qft_dg(qc: QuantumCircuit, q: List[Tuple[QuantumRegister, int]]):
    qc2 = QuantumCircuit(*qc.get_qregs().values(), *qc.get_cregs().values())
    qft(qc2, q)
    new_data = [op.inverse() for op in reversed(qc2.data)]
    qc2.data = new_data
    qc.extend(qc2)
    return qc
