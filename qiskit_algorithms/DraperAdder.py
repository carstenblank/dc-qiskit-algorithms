# -*- coding: utf-8 -*-

# Copyright 2018, Carsten Blank.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

from typing import Optional, Tuple

import qiskit
from qiskit import QuantumCircuit
from qiskit.extensions import standard

from . import Qft as qft


def draper_adder(input_a: int, input_b: int, length: Optional[int] = None) -> Tuple[QuantumCircuit, float]:
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

    qft.qft_reg(qc, a)

    for b_index in reversed(range(b.size)):
        theta_index = 1
        for a_index in reversed(range(b_index + 1)):
            standard.cu1(qc, qft.get_theta(theta_index), b[b_index], a[a_index])
            theta_index += 1

    # standard.cu1(qc, qft.get_theta(1), b[1], a[1])
    # standard.cu1(qc, qft.get_theta(2), b[1], a[0])
    #
    # standard.cu1(qc, qft.get_theta(1), b[0], a[0])

    qft.qft_dg_reg(qc, a)

    standard.barrier(qc, a, b)

    qc.measure(a, c_a)
    qc.measure(b, c_b)

    return qc, 2**length
