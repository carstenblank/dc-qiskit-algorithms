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


import unittest

from ddt import ddt, unpack, data
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, Aer, execute, Result
from qiskit.extensions.standard import x, h

import defaults
from dc_qiskit_algorithms.UniformRotation import ccx


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
        sim_result = job_sim.result()  # type: Result

        counts = sim_result.get_counts(qc)  # type: dict
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

        counts = sim_result.get_counts(qc)  # type: dict
        self.assertIsNotNone(counts.keys())
        self.assertListEqual(list(counts.keys()), ['0000', '0001', '0100', '1101'])


if __name__ == '__main__':
        unittest.main(verbosity=2)