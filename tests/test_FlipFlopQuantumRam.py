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
from typing import List

import numpy as np
import qiskit
from ddt import ddt, data as test_data, unpack
from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister
from qiskit.circuit.measure import measure
from qiskit.providers import BaseBackend
from qiskit.result import Result

from dc_qiskit_algorithms.FlipFlopQuantumRam import FFQramDb, add_vector

@ddt
class FlipFlopQuantumRamnStatePrepTests(unittest.TestCase):

    def execute_test(self, vector: List[float]):
        probability_vector = [np.absolute(e)**2 for e in vector]
        print("Input Vector (state) & its measurement probability:")
        print(["{0:.3f}".format(e) for e in vector])
        print(["{0:.3f}".format(e) for e in probability_vector])

        db = FFQramDb()
        add_vector(db, vector)

        bus = QuantumRegister(db.bus_size(), "bus")
        reg = QuantumRegister(1, "reg")

        c_bus = ClassicalRegister(db.bus_size(), "c_bus")
        c_reg = ClassicalRegister(1, "c_reg")

        qc = QuantumCircuit(bus, reg, c_bus, c_reg, name='state prep')

        qc.h(bus)
        db.add_to_circuit(qc, bus, reg[0])

        local_backend = qiskit.Aer.get_backend('statevector_simulator')  # type: BaseBackend

        job = qiskit.execute(qc, backend=local_backend, shots=1)
        result = job.result()  # type: Result

        # State vector
        result_state_vector = result.get_statevector('state prep')
        print("Full simulated state vector (n+1!)")
        print(["{0:.2f}".format(e) for e in result_state_vector])

        correct_branch_state = np.asarray(result_state_vector)[8:]
        correct_branch_state = correct_branch_state / np.linalg.norm(correct_branch_state)

        print("State vector on the correct (1) branch:")
        print(["{0:.2f}".format(e) for e in correct_branch_state])

        positive_global_phase_all_almost_equal = all(abs(a - e) < 0.02 for a, e in zip(vector, correct_branch_state))
        negative_global_phase_all_almost_equal = all(abs(a + e) < 0.02 for a, e in zip(vector, correct_branch_state))

        self.assertTrue(positive_global_phase_all_almost_equal or negative_global_phase_all_almost_equal)

        # Probability Vector by Measurement
        measure(qc, bus, c_bus)
        measure(qc, reg, c_reg)

        local_qasm_backend = qiskit.Aer.get_backend('qasm_simulator')  # type: BaseBackend
        shots = 2**16
        job = qiskit.execute(qc, backend=local_qasm_backend, shots=shots)
        result = job.result()  # type: Result
        counts = result.get_counts('state prep')

        measurement_probability_vector = [0.0 for e in probability_vector]
        shot_post_measurement = sum(c for b, c in counts.items() if b.startswith("1 "))
        print("Probability to be on the correct (1) branch: %.4f" % (shot_post_measurement / shots))
        for binary, count in sorted(counts.items()):
            [reg, bus] = binary.split(' ')
            if reg == '1':
                index = int(bus, 2)
                probability = float(count) / float(shot_post_measurement)
                # print("%s (%d): %.3f" % (bus, index, probability))
                measurement_probability_vector[index] = probability

        print("Measurement Probability on the correct (1) branch:")
        print(["{0:.3f}".format(e) for e in measurement_probability_vector])
        for expected, actual in zip(probability_vector, measurement_probability_vector):
            self.assertAlmostEqual(expected, actual, delta=0.05)

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
        self.check_add_vector(vector)

        vector = np.asarray(vector)
        vector = (1 / np.linalg.norm(vector)) * vector
        self.execute_test(list(vector))

    def check_add_vector(self, vector):
        unit_vector = np.asarray(vector)
        l2_norm = np.linalg.norm(unit_vector)
        unit_vector = unit_vector / l2_norm

        labels = [i for i, v in enumerate(unit_vector) if abs(v) > 1e-6]
        db = FFQramDb()
        add_vector(db, vector)

        check_labels = [int.from_bytes(e.label, byteorder='big') for e in db]

        self.assertListEqual(labels, check_labels)


if __name__ == '__main__':
    unittest.main(verbosity=2)
