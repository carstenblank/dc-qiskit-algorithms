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
from typing import List

import numpy as np
import qiskit
from qiskit.providers import JobV1
from qiskit_aer import Aer, StatevectorSimulator, QasmSimulator
from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister
from ddt import ddt, data as test_data, unpack
from qiskit.result import Result

import dc_qiskit_algorithms.MottonenStatePreparation

logging.basicConfig(format=logging.BASIC_FORMAT, level='INFO')
log = logging.getLogger('test_DraperAdder')


# noinspection NonAsciiCharacters
@ddt
class MottonenStatePrepTests(unittest.TestCase):

    def execute_test(self, vector: List[float]):
        probability_vector = [np.absolute(e)**2 for e in vector]

        qubits = int(np.log2(len(vector)))
        reg = QuantumRegister(qubits, "reg")
        c = ClassicalRegister(qubits, "c")
        qc = QuantumCircuit(reg, c, name='state prep')
        qc.state_prep_möttönen(vector, reg)

        local_backend: StatevectorSimulator = Aer.get_backend('statevector_simulator')

        job: JobV1 = qiskit.execute(qc, backend=local_backend, shots=1)
        result: Result = job.result()

        # State vector
        result_state_vector = result.get_statevector('state prep')
        print(["{0:.2f}".format(e) for e in result_state_vector])
        # Try to find a global phase
        global_phase = set([np.angle(v) - np.angle(rv) for v, rv in zip(vector, result_state_vector)
                            if abs(v) > 1e-3 and abs(rv) > 1e-3])
        global_phase = global_phase.pop() or 0.0
        result_state_vector = np.exp(1.0j * global_phase) * result_state_vector
        for expected, actual in zip(vector, result_state_vector):
            self.assertAlmostEqual(actual.imag, 0.0, places=6)
            self.assertAlmostEqual(expected, actual.real, places=6)

        # Probability vector from state vector
        result_probability_vector = [np.absolute(e)**2 for e in result_state_vector]
        print(["{0:.3f}".format(e) for e in result_probability_vector])
        for expected, actual in zip(probability_vector, result_probability_vector):
            self.assertAlmostEqual(expected, actual, places=2)

        # Probability Vector by Measurement
        qc.measure(reg, c)
        local_qasm_backend: QasmSimulator = Aer.get_backend('qasm_simulator')
        shots = 2**12
        job: JobV1 = qiskit.execute(qc, backend=local_qasm_backend, shots=shots)
        result = job.result()  # type: Result
        counts = result.get_counts('state prep')
        measurement_probability_vector = [0.0 for e in result_state_vector]
        for binary, count in sorted(counts.items()):
            index = int(binary, 2)
            probability = float(count) / float(shots)
            print("%s (%d): %.3f" % (binary, index, probability))
            measurement_probability_vector[index] = probability

        print(["{0:.3f}".format(e) for e in measurement_probability_vector])
        for expected, actual in zip(probability_vector, measurement_probability_vector):
            self.assertAlmostEqual(expected, actual, delta=0.02)

    @unpack
    @test_data(
        {'vector': [-0.1, 0.2, -0.3, 0.4, -0.5, 0.6, -0.7, 0.8]},
        {'vector': [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
        {'vector': [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
        {'vector': [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
        {'vector': [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]},
        {'vector': [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]},
        {'vector': [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]},
        {'vector': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]},
        {'vector': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]}
    )
    def test_state_preparation(self, vector):
        vector = np.asarray(vector)
        vector = (1 / np.linalg.norm(vector)) * vector
        self.execute_test(list(vector))

    def test_instantiation(self):
        gate = dc_qiskit_algorithms.MottonenStatePreparationGate([1.0, 0.0])
        self.assertIsInstance(gate, dc_qiskit_algorithms.MottonenStatePreparationGate)


if __name__ == '__main__':
    unittest.main(verbosity=2)
