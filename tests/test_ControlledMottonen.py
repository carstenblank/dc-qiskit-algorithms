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
from qiskit_aer import AerJob
from qiskit_aer.backends.aerbackend import AerBackend
from qiskit.result import Result
from scipy import sparse
from scipy.sparse import linalg

from dc_qiskit_algorithms.ControlledStatePreparation import ControlledStatePreparationGate

logging.basicConfig(format=logging.BASIC_FORMAT, level='INFO')
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


@ddt
class AngleMatrixYTests(unittest.TestCase):
    @unpack
    @test_data(
        {'matrix': np.sqrt([
            [0.1, 0.2, 0.3, 0.4],
            [0.25, 0.25, 0.25, 0.25]
        ])}
    )
    def test_angle_matrix_y(self, matrix):
        matrix = sparse.dok_matrix(matrix)

        abs_matrix = np.abs(matrix)

        log.info("Input Matrix (Y):\n" + str(abs_matrix))

        iso_gate = ControlledStatePreparationGate(matrix)

        angle_matrix = iso_gate._to_angle_matrix_y()
        log.info("Final Angle Matrix (Y):\n" + str(angle_matrix.todense()))

        # The recovery is done by using a pre factor of 1/2 given the definition of the R_y gate
        matrix_recovered = sparse.dok_matrix(matrix.shape)
        for row in range(angle_matrix.shape[0]):
            matrix_recovered[row, 0] = np.cos(0.5 * angle_matrix[row, 0]) * np.cos(0.5 * angle_matrix[row, 1])
            matrix_recovered[row, 1] = np.cos(0.5 * angle_matrix[row, 0]) * np.sin(0.5 * angle_matrix[row, 1])
            matrix_recovered[row, 2] = np.sin(0.5 * angle_matrix[row, 0]) * np.cos(0.5 * angle_matrix[row, 2])
            matrix_recovered[row, 3] = np.sin(0.5 * angle_matrix[row, 0]) * np.sin(0.5 * angle_matrix[row, 2])

        log.info("Recovered Matrix (Y):\n" + str(matrix_recovered.todense()))

        self.assertAlmostEqual(linalg.norm(abs_matrix - matrix_recovered), 0.0, delta=1e-13)


@ddt
class AngleMatrixZTests(unittest.TestCase):
    @unpack
    @test_data(
        {
            'matrix': np.sqrt([
                [-0.25 + 0j, +0.25 + 0j, -0.25 + 0j, +0.25 + 0j],
                [+0.25 + 0j, +0.25 + 0j, +0.25 + 0j, +0.25 + 0j]
            ])
        },
        {
            'matrix': np.sqrt([
                [+0.50 + 0j, +0.30 + 0j, +0.20 + 0j, +0.00 + 0j],
                [+0.50 + 0j, -0.25 + 0j, +0.25 + 0j, +0.25 + 0j]
            ])
        }
    )
    def test_angle_matrix_z(self, matrix):

        phase_matrix = np.angle(matrix)
        matrix = sparse.dok_matrix(matrix)

        log.info("Input Phase Matrix (Z):\n" + str(phase_matrix))

        iso_gate = ControlledStatePreparationGate(matrix)

        angle_matrix, global_phase = iso_gate._to_angle_matrix_z()
        angle_matrix = sparse.hstack([global_phase, angle_matrix]).todok()
        log.info("Final Angle Matrix (Z):\n" + str(angle_matrix.todense()))

        # The recovery is done by using a pre factor of 1/2 given the definition of the R_y gate
        matrix_recovered = sparse.dok_matrix(matrix.shape)
        for row in range(angle_matrix.shape[0]):
            matrix_recovered[row, 0] = np.angle(np.exp(1.0j * angle_matrix[row, 0]) * np.exp(-0.5j * angle_matrix[row, 1]) * np.exp(-0.5j * angle_matrix[row, 2]))
            matrix_recovered[row, 1] = np.angle(np.exp(1.0j * angle_matrix[row, 0]) * np.exp(-0.5j * angle_matrix[row, 1]) * np.exp(+0.5j * angle_matrix[row, 2]))
            matrix_recovered[row, 2] = np.angle(np.exp(1.0j * angle_matrix[row, 0]) * np.exp(+0.5j * angle_matrix[row, 1]) * np.exp(-0.5j * angle_matrix[row, 3]))
            matrix_recovered[row, 3] = np.angle(np.exp(1.0j * angle_matrix[row, 0]) * np.exp(+0.5j * angle_matrix[row, 1]) * np.exp(+0.5j * angle_matrix[row, 3]))

        global_phases = np.unique(np.round(phase_matrix - matrix_recovered.todense(), decimals=13), axis=1)
        self.assertEqual(global_phases.shape, (matrix_recovered.shape[0], 1))
        matrix_recovered_1 = global_phases + matrix_recovered

        log.info("Recovered Matrix (Z):\n" + str(matrix_recovered_1))

        self.assertAlmostEqual(np.linalg.norm(phase_matrix - matrix_recovered_1), 0.0, delta=1e-13)


@ddt
class ControlledMottonenStatePrepTests(unittest.TestCase):
    @unpack
    @test_data(
        {
            'matrix': np.sqrt([
                [+0.5, +0.5],
                [0.5 * np.exp(1j * np.pi), 0.5 * np.exp(1j * 0)]
            ]),
            'debug_circuit': False
        },
        {
            'matrix': np.sqrt([
                [+0.5, +0.5],
                [0.5 * np.exp(1j * np.pi), 0.5 * np.exp(1j * 0)]
            ]),
            'debug_circuit': True
        },
        {
            'matrix': np.sqrt([
                [+0.5, +0.5],
                [+0.6, +0.4],
                [+0.7, +0.3],
                [+0.9, +0.1]
            ]),
            'debug_circuit': False
        },
        {
            'matrix': np.sqrt([
                [+0.5, +0.5],
                [+0.6, +0.4],
                [+0.7, +0.3],
                [+0.9, +0.1]
            ]),
            'debug_circuit': True
        },
        {
            'matrix': np.sqrt([
                [+0.50, +0.30, +0.20, +0.00],
                [+0.25, +0.25, +0.25, +0.25]
            ]),
            'debug_circuit': False
        },
        {
            'matrix': np.sqrt([
                [+0.50, +0.30, +0.20, +0.00],
                [+0.25, +0.25, +0.25, +0.25]
            ]),
            'debug_circuit': True
        },
        {
            'matrix': np.sqrt([
                [+0.50, +0.30, +0.20, +0.00],
                [+0.25, +0.25, +0.25, +0.25],
                [+0.10, +0.10, +0.40, +0.40],
                [+0.40, +0.10, +0.30, +0.20],
                [+0.50, +0.30, +0.20, +0.00],
                [+0.25, +0.25, +0.25, +0.25],
                [+0.10, +0.10, +0.40, +0.40],
                [+0.40, +0.10, +0.30, +0.20]
            ]),
            'debug_circuit': True
        },
        {
            'matrix': np.sqrt([
                [+0.50, +0.30, +0.20, +0.00],
                [+0.25, +0.25, +0.25, +0.25],
                [+0.10, +0.10, +0.40, +0.40],
                [+0.40, +0.10, +0.30, +0.20],
                [+0.50, +0.30, +0.20, +0.00],
                [+0.25, +0.25, +0.25, +0.25],
                [+0.10, +0.10, +0.40, +0.40],
                [+0.40, +0.10, +0.30, +0.20]
            ]),
            'debug_circuit': False
        },
        {
            'matrix': np.sqrt([
                [0.50 * np.exp(1j * 0), 0.30 * np.exp(1j * 0),      0.20 * np.exp(1j * 0), 0.00 * np.exp(1j * 0)],
                [0.40 * np.exp(1j * 0), 0.10 * np.exp(-1j * np.pi), 0.25 * np.exp(1j * 0), 0.25 * np.exp(1j * 0)]
            ]),
            'debug_circuit': False
        },
        {
            'matrix': np.sqrt([
                [0.50 * np.exp(1j * 0), 0.30 * np.exp(1j * 0),      0.20 * np.exp(1j * 0), 0.00 * np.exp(1j * 0)],
                        [0.40 * np.exp(1j * 0), 0.10 * np.exp(-1j * np.pi), 0.25 * np.exp(1j * 0), 0.25 * np.exp(1j * 0)]
            ]),
            'debug_circuit': True
        }
    )
    def test_define(self, matrix, debug_circuit):
        log.info("STARTING TEST")

        control_qubits = int(np.ceil(np.log2(matrix.shape[0])))
        target_qubits = int(np.ceil(np.log2(matrix.shape[1])))

        # We can compute the expected state vector by assuming we use an equal superposition (hadamard) on the
        # control qubits (we need to do that a few lines down) so a factor in the Hadamard factors 2^{-n/2} is needed
        # Also, the ordering of the registers is extremely important here. A few lines below, we first add the control
        # and then the target qubits. That gives the canonical ordering such that we can do column first ravel (order=F)
        theoretical_state_vector: np.ndarray = np.asarray(
            matrix.ravel(order='F')
        ).reshape(-1) * np.power(2, -control_qubits / 2)
        log.info(f"Expected State: {theoretical_state_vector.tolist()}")

        # As mentioned above the ordering is important for this test!
        ctrl_qb = QuantumRegister(control_qubits, name='ctrl')
        tgt_qb = QuantumRegister(target_qubits, name='tgt')
        qc = QuantumCircuit(ctrl_qb, tgt_qb)

        # Here we do the equal superposition on the control register as assumed above.
        qc.h(ctrl_qb)
        qc.append(
            ControlledStatePreparationGate(sparse.dok_matrix(matrix)).set_debug_flag(debug_circuit),
            list(ctrl_qb) + list(tgt_qb)
        )
        drawing = ControlledStatePreparationGate(sparse.dok_matrix(matrix)) \
            .set_debug_flag(debug_circuit) \
            .definition \
            .draw(fold=-1)
        log.info(f'Circuit:\n{drawing}')

        # The the resulting state vector from the state vector simulator
        backend: AerBackend = qiskit.Aer.get_backend('statevector_simulator')
        job: AerJob = qiskit.execute(qc, backend)
        result: Result = job.result()
        vector: np.ndarray = np.asarray(result.get_statevector())

        # Computing the test:
        # The extracted state from the simulation is allowed to be off by a common (global) phase
        # If this is the case, taking the angle difference and correcting it, should give the same vector
        correction = np.angle(theoretical_state_vector[0]) - np.angle(vector[0])
        vector_phase_corrected = vector * np.exp(1.0j * correction)
        log.info(f"Actual State: {vector_phase_corrected.tolist()}")

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
