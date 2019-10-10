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
import logging
import unittest

import qiskit
from ddt import ddt, unpack, data
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute
import qiskit.extensions
from qiskit.result import Result

import dc_qiskit_algorithms

logging.basicConfig(format=logging.BASIC_FORMAT, level='INFO')
log = logging.getLogger('test_DraperAdder')


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

        qc.x(q[1])
        qc.x(q[2])
        qc.x(q[3])
        qc.ccx_uni_rot(7, [q[1], q[2], q[3]], q[0])
        qc.measure(q, c)

        backend = qiskit.BasicAer.get_backend('qasm_simulator')
        job_sim = execute(qc, backend, shots=10000)
        sim_result = job_sim.result()  # type: Result

        counts = sim_result.get_counts(qc)  # type: dict
        log.info(counts)
        self.assertIsNotNone(counts.keys())
        self.assertListEqual(list(counts.keys()), ['1111'])

    def test_2(self):
        q = QuantumRegister(4)
        c = ClassicalRegister(4)
        qc = QuantumCircuit(q, c)

        qc.h(q[0])
        qc.h(q[2])
        # State now ['0000', '0001', '0100', '0101']
        qc.ccx_uni_rot(5, [q[0], q[1], q[2]], q[3])
        # State now ['0000', '0001', '0100', '1101']
        qc.measure(q, c)

        backend = qiskit.BasicAer.get_backend('qasm_simulator')
        job_sim = execute(qc, backend, shots=10000)
        sim_result = job_sim.result()  # type: Result

        counts = sim_result.get_counts(qc)  # type: dict
        log.info(counts)
        self.assertIsNotNone(counts.keys())
        self.assertListEqual(list(sorted(counts.keys())), ['0000', '0001', '0100', '1101'])


if __name__ == '__main__':
        unittest.main(verbosity=2)