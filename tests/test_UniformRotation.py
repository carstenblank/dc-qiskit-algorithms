# -*- coding: utf-8 -*-

# Copyright 2018, Carsten Blank.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

import unittest

import numpy
from ddt import ddt, unpack, data
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, Aer, execute, Result
from qiskit.extensions.standard import x, h

from qiskit_algorithms.UniformRotation import ccx


@ddt
class MultipleControlledNotGateTest(unittest.TestCase):

    @unpack
    @data(
        {'vector': [-0.1, 0.2, -0.3, 0.4, -0.5, 0.6, -0.7, 0.8]}
    )
    def test_1(self, vector):
        q = QuantumRegister(4)
        c = ClassicalRegister(4)
        qc = QuantumCircuit(q, c)

        x(qc, q[1])
        x(qc, q[2])
        x(qc, q[3])
        ccx(qc, 7, [q[1], q[2], q[3]], q[0])
        qc.measure(q, c)

        backend = Aer.get_backend('qasm_simulator')
        job_sim = execute(qc, backend, shots=10000)
        sim_result: Result = job_sim.result()

        counts: dict = sim_result.get_counts(qc)
        self.assertIsNotNone(counts.keys())
        self.assertListEqual(list(counts.keys()), ['1111'])

    def test_2(self):
        q = QuantumRegister(4)
        c = ClassicalRegister(4)
        qc = QuantumCircuit(q, c)

        h(qc, q[0])
        h(qc, q[2])
        # State now ['0000', '0001', '0100', '0101']
        ccx(qc, 5, [q[0], q[1], q[2]], q[3])
        # State now ['0000', '0001', '0100', '1101']
        qc.measure(q, c)

        backend = Aer.get_backend('qasm_simulator')
        job_sim = execute(qc, backend, shots=10000)
        sim_result: Result = job_sim.result()

        counts: dict = sim_result.get_counts(qc)
        self.assertIsNotNone(counts.keys())
        self.assertListEqual(list(counts.keys()), ['0000', '0001', '0100', '1101'])


if __name__ == '__main__':
        unittest.main(verbosity=2)