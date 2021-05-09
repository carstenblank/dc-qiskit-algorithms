# Copyright 2021 Carsten Blank
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

import numpy as np
import qiskit
from ddt import ddt, data as test_data, unpack
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.providers.aer import AerJob
from qiskit.providers.aer.backends.aerbackend import AerBackend
from qiskit.result import Result
from scipy import sparse
from scipy.sparse import linalg

from dc_qiskit_algorithms.ControlledMottonen import ControlledStatePreparationGate

logging.basicConfig(format=logging.BASIC_FORMAT, level='ERROR')
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


# noinspection NonAsciiCharacters
@ddt
class ControlledMottonenStatePrepTests(unittest.TestCase):
    @unpack
    @test_data(
        {'matrix': [
            [+0.1, +0.1],
            [+0.2, +0.1],
            [+0.3, +0.1],
            [+0.4, +0.1]
        ]}
    )
    def test_angle_matrix_y(self, matrix):
        matrix = sparse.dok_matrix(matrix)
        columns_norm = linalg.norm(matrix, axis=0)
        matrix = matrix.multiply(np.power(columns_norm, -1)).todense()

        abs_matrix = np.abs(matrix)

        log.info("Input Matrix (Y):\n" + str(abs_matrix))

        iso_gate = ControlledStatePreparationGate(matrix)

        angle_matrix = iso_gate._to_angle_matrix_y()
        log.info("Final Angle Matrix (Y):\n" + str(angle_matrix.todense()))

        # The recovery is done by using a pre factor of 1/2 given the definition of the R_y gate
        matrix_recovered = sparse.dok_matrix(matrix.shape)
        for col in range(angle_matrix.shape[1]):
            matrix_recovered[0, col] = np.cos(0.5 * angle_matrix[0, col]) * np.cos(0.5 * angle_matrix[1, col])
            matrix_recovered[1, col] = np.cos(0.5 * angle_matrix[0, col]) * np.sin(0.5 * angle_matrix[1, col])
            matrix_recovered[2, col] = np.sin(0.5 * angle_matrix[0, col]) * np.cos(0.5 * angle_matrix[2, col])
            matrix_recovered[3, col] = np.sin(0.5 * angle_matrix[0, col]) * np.sin(0.5 * angle_matrix[2, col])

        log.info("Recovered Matrix (Y):\n" + str(matrix_recovered.multiply(columns_norm).todense()))

        self.assertAlmostEqual(np.linalg.norm(abs_matrix - matrix_recovered), 0.0, delta=1e-13)

    @unpack
    @test_data(
        {'matrix': [
            [-0.1, +0.1],
            [+0.1, +0.1],
            [-0.1, +0.1],
            [+0.1, +0.1]
        ]}
    )
    def test_angle_matrix_z(self, matrix):
        matrix = sparse.dok_matrix(matrix)
        columns_norm = linalg.norm(matrix, axis=0)
        matrix = matrix.multiply(np.power(columns_norm, -1)).todense()

        phase_matrix = np.angle(matrix)

        log.info("Input Phase Matrix (Z):\n" + str(phase_matrix))

        iso_gate = ControlledStatePreparationGate(matrix)

        angle_matrix, global_phase = iso_gate._to_angle_matrix_z()
        angle_matrix = sparse.vstack([global_phase, angle_matrix]).todok()
        log.info("Final Angle Matrix (Z):\n" + str(angle_matrix.todense()))

        # The recovery is done by using a pre factor of 1/2 given the definition of the R_y gate
        matrix_recovered = sparse.dok_matrix(matrix.shape)
        for col in range(angle_matrix.shape[1]):
            matrix_recovered[0, col] = np.angle(np.exp(1.0j * angle_matrix[0, col]) * np.exp(-0.5j * angle_matrix[1, col]) * np.exp(-0.5j * angle_matrix[2, col]))
            matrix_recovered[1, col] = np.angle(np.exp(1.0j * angle_matrix[0, col]) * np.exp(-0.5j * angle_matrix[1, col]) * np.exp(+0.5j * angle_matrix[2, col]))
            matrix_recovered[2, col] = np.angle(np.exp(1.0j * angle_matrix[0, col]) * np.exp(+0.5j * angle_matrix[1, col]) * np.exp(-0.5j * angle_matrix[3, col]))
            matrix_recovered[3, col] = np.angle(np.exp(1.0j * angle_matrix[0, col]) * np.exp(+0.5j * angle_matrix[1, col]) * np.exp(+0.5j * angle_matrix[3, col]))

        global_phases = np.unique(phase_matrix - matrix_recovered.todense(), axis=0)
        self.assertEqual(global_phases.shape, (1, matrix_recovered.shape[1]))
        matrix_recovered_1 = global_phases + matrix_recovered

        log.info("Recovered Matrix (Z):\n" + str(matrix_recovered_1))

        self.assertAlmostEqual(np.linalg.norm(phase_matrix - matrix_recovered_1), 0.0, delta=1e-13)

    @unpack
    @test_data(
        {'matrix': [
            [+0.1, +0.1],
            [+0.2, +0.1],
            [+0.3, +0.1],
            [+0.4, +0.1]
        ]},
        {'matrix': [
            [-0.1, +0.1],
            [+0.1, -0.1],
            [+0.1, +0.1],
            [+0.1, -0.1]
        ]},
        {'matrix': [
            [+0.5, +0.1, +0.5, +0.1],
            [+0.1, +0.1, +0.5, +0.0],
            [+0.3, +0.1, +0.1, +0.1],
            [+0.1, +0.1, +0.0, +0.0]
        ]},
        {'matrix': np.asarray([
            [+0.5, +0.1, +0.1, +0.0],
            [+0.1, -0.4, +0.1, +0.0],
            [+0.2, +0.4, +0.1, +1.0],
            [+0.1, +0.1, +0.7, +0.0]
        ])}
    )
    def test_define(self, matrix):
        log.info("STARTING TEST")
        matrix = sparse.dok_matrix(matrix)
        columns_norm = linalg.norm(matrix, axis=0)
        matrix_normed: np.matrix = matrix.multiply(np.power(columns_norm, -1)).todense()

        target_qubits = int(np.ceil(np.log2(matrix.shape[0])))
        control_qubits = int(np.ceil(np.log2(matrix.shape[1])))

        # We can compute the expected state vector by assuming we use an equal superposition (hadamard) on the
        # control qubits. We need to not only use the normed matrix, but also factor in the Hadamard factors 2^{-n/2}
        theoretical_state_vector: np.ndarray = np.asarray(matrix_normed.ravel(order='F')).reshape(-1) * np.power(2, -control_qubits / 2)
        log.info(f"Expected State: {theoretical_state_vector.tolist()}")

        ctrl_qb = QuantumRegister(control_qubits, name='ctrl')
        tgt_qb = QuantumRegister(target_qubits, name='tgt')
        qc = QuantumCircuit(tgt_qb, ctrl_qb)

        # The numbering is LSB on the left / MSB on the right. This creates unexpected results if not taken into account
        qc.h(ctrl_qb)
        qc.append(ControlledStatePreparationGate(matrix_normed), list(ctrl_qb) + list(reversed(tgt_qb)))

        # The the resulting state vector from the state vector simulator
        backend: AerBackend = qiskit.Aer.get_backend('statevector_simulator')
        job: AerJob = qiskit.execute(qc, backend)
        result: Result = job.result()
        vector: np.ndarray = result.get_statevector()

        # Computing the test:
        # The extraced state from the simulation is allowed to be off by a common (global) phase
        # If this is the case, taking the angle difference and correcting it, should give the same vector
        correction = np.angle(theoretical_state_vector[0]) - np.angle(vector[0])
        vector_phase_corrected = vector * np.exp(1.0j * correction)
        log.info(f"Actual State: {theoretical_state_vector.tolist()}")

        diff = vector_phase_corrected - theoretical_state_vector
        self.assertAlmostEqual(np.linalg.norm(diff), 0.0, places=13)

        if log.level == logging.DEBUG:
            basic_qc = qiskit.transpile(qc, optimization_level=0, basis_gates=['uni_rot_rz', 'uni_rot_ry', 'state_prep_möttönen', 'h'])
            log.debug(f"\n{basic_qc.draw(fold=-1)}")
            basic_qc = qiskit.transpile(qc, optimization_level=0, basis_gates=['rz', 'cp', 'cx', 'ry', 'p', 'h'])
            log.debug(f"\n{basic_qc.draw(fold=-1)}")
            basic_qc = qiskit.transpile(qc, optimization_level=3, basis_gates=['u3', 'u2', 'u1', 'cx'])
            log.debug(f"\n{basic_qc.draw(fold=-1)}")

            log.debug('Theoretical result:')
            log.debug(np.round(theoretical_state_vector, decimals=4).tolist())

            log.debug('Absolute:')
            log.debug(np.round(np.abs(vector), decimals=4).tolist())
            log.debug(np.round(np.abs(theoretical_state_vector), decimals=4).tolist())

            log.debug('Angle:')
            corrected_angle_vector = correction + np.angle(vector)
            corrected_angle_vector = np.fmod(corrected_angle_vector, 2*np.pi)
            log.debug(np.round(corrected_angle_vector, decimals=4).tolist())
            log.debug(np.round(np.angle(theoretical_state_vector), decimals=4).tolist())

            log.debug('TEST:')
            angle_diff = corrected_angle_vector - np.angle(theoretical_state_vector)
            abs_diff = np.abs(vector) - np.abs(theoretical_state_vector)

            log.debug(np.round(abs_diff, decimals=4).tolist())
            log.debug(np.round(angle_diff, decimals=4).tolist())

            log.debug('Real:')
            log.debug(np.round(np.real(vector * np.exp(1.0j * correction)), decimals=4).tolist())

            log.debug('Difference:')
            log.debug(np.round(diff, decimals=4).tolist())

        log.info("ENDING TEST")
