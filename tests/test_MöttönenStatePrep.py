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

import numpy
import qiskit
from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister, Result
from qiskit.backends import BaseBackend, BaseJob
from ddt import ddt, data as test_data, unpack

import defaults
from dc_qiskit_algorithms.MöttönenStatePrep import state_prep_möttönen


# noinspection NonAsciiCharacters
@ddt
class MöttönenStatePrepTests(unittest.TestCase):

    def execute_test(self, vector: List[float]):
        probability_vector = [numpy.absolute(e)**2 for e in vector]

        qubits = int(numpy.log2(len(vector)))
        reg = QuantumRegister(qubits, "reg")
        c = ClassicalRegister(qubits, "c")
        qc = QuantumCircuit(reg, c, name='state prep')
        state_prep_möttönen(qc, vector, reg)

        local_backend: BaseBackend = qiskit.Aer.get_backend('statevector_simulator')

        qobj = qiskit.compile([qc], backend=local_backend, shots=1)
        job: BaseJob = local_backend.run(qobj)
        result: Result = job.result()

        # State vector
        result_state_vector = result.get_statevector('state prep')
        print(["{0:.2f}".format(e) for e in result_state_vector])
        sign = 1.0
        if abs(vector[0] - result_state_vector[0].real) > 1e-6:
            sign = -1.0
        for expected, actual in zip(vector, result_state_vector):
            self.assertAlmostEqual(actual.imag, 0.0, places=6)
            self.assertAlmostEqual(expected, sign*actual.real, places=6)

        # Probability vector from state vector
        result_probability_vector = [numpy.absolute(e)**2 for e in result_state_vector]
        print(["{0:.3f}".format(e) for e in result_probability_vector])
        for expected, actual in zip(probability_vector, result_probability_vector):
            self.assertAlmostEqual(expected, actual, places=2)

        # Probability Vector by Measurement
        qc.measure(reg, c)
        local_qasm_backend: BaseBackend = qiskit.Aer.get_backend('qasm_simulator')
        from qiskit import transpiler
        shots = 2**18
        qobj = transpiler.compile([qc], backend=local_qasm_backend, shots=shots)
        job: BaseJob = local_qasm_backend.run(qobj)
        result: Result = job.result()
        counts = result.get_counts('state prep')
        measurement_probability_vector = [0.0 for e in result_state_vector]
        for binary, count in sorted(counts.items()):
            index = int(binary, 2)
            probability = float(count) / float(shots)
            print("%s (%d): %.3f" % (binary, index, probability))
            measurement_probability_vector[index] = probability

        print(["{0:.3f}".format(e) for e in measurement_probability_vector])
        for expected, actual in zip(probability_vector, measurement_probability_vector):
            self.assertAlmostEqual(expected, actual, places=2)

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
        vector = numpy.asarray(vector)
        vector = (1 / numpy.linalg.norm(vector)) * vector
        self.execute_test(list(vector))


if __name__ == '__main__':
        unittest.main(verbosity=2)