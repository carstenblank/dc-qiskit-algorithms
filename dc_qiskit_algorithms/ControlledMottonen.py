import math
from typing import List, Union

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Gate
from qiskit.circuit.library import RYGate, RZGate
from scipy import sparse
from scipy.sparse.linalg import norm

from dc_qiskit_algorithms import UniformRotationGate
from dc_qiskit_algorithms.MöttönenStatePreparation import get_alpha_y, get_alpha_z, MöttönenStatePreparationGate


class IsometryGate(Gate):

    def __init__(self, matrix) -> None:
        """
        matrix = A = (a_ij)
        with j being the control and
        :param: matrix
        """

        if isinstance(matrix, list) or isinstance(matrix, np.ndarray):
            matrix = sparse.dok_matrix(matrix)
        # The rows define the target states, i.e. the space for the controlled state preparation
        num_targets_qb = int(math.ceil(math.log2(matrix.shape[0])))
        # The columns define the number of control states. If there is one row, this is normal state preparation.
        num_controls_qb = int(math.ceil(math.log2(matrix.shape[1])))
        super().__init__("iso_matrix", num_qubits=num_controls_qb + num_targets_qb, params=[])
        self.matrix_P = matrix  # type: Union[sparse.dok_matrix]

        matrix_P_abs = sparse.dok_matrix(matrix.shape)
        matrix_P_angle = sparse.dok_matrix(matrix.shape)
        for (i, j), v in self.matrix_P.items():
            matrix_P_abs[i, j] = np.absolute(v)
            matrix_P_angle[i, j] = np.angle(v)
        self.matrix_P_abs = matrix_P_abs
        self.matrix_P_angle = matrix_P_angle

        self.num_targets_qb = num_targets_qb
        self.num_controls_qb = num_controls_qb

    def _to_angle_matrix_y(self):
        # type: () -> Union[sparse.dok_matrix]

        # First, for each column, the angles that lead to this state need to be computed.
        matrix_A = sparse.dok_matrix((self.matrix_P_abs.shape[0] - 1, self.matrix_P_abs.shape[1]))
        for col_no in range(self.matrix_P_abs.shape[1]):
            amplitudes_column: sparse.csc_matrix = self.matrix_P_abs.getcol(col_no)
            # The reversed is necessary as the "highest" qubit is the one with the least controls
            # imagine a circuit the where the highest qubits control the lower one. Yes this is all but numbering
            # so that this is why I need to add this comment.
            angle_column_list_y: List[sparse.dok_matrix] = [
                get_alpha_y(amplitudes_column.todok(), self.num_targets_qb, k)
                for k in reversed(range(1, self.num_targets_qb + 1))
            ]
            angles_column = sparse.vstack(angle_column_list_y)
            matrix_A[:, col_no] = angles_column

        return matrix_A

    def _to_angle_matrix_z(self):
        # type: () -> Union[sparse.dok_matrix]

        # First, for each column, the angles that lead to this state need to be computed.
        matrix_A = sparse.dok_matrix((self.matrix_P_angle.shape[0] - 1, self.matrix_P_angle.shape[1]))
        for col_no in range(self.matrix_P_angle.shape[1]):
            amplitudes_column: sparse.csc_matrix = self.matrix_P_angle.getcol(col_no)
            # The reversed is necessary as the "highest" qubit is the one with the least controls
            # imagine a circuit the where the highest qubits control the lower one. Yes this is all but numbering
            # so that this is why I need to add this comment.
            angle_column_list: List[sparse.dok_matrix] = [
                get_alpha_z(amplitudes_column.todok(), self.num_targets_qb, k)
                for k in reversed(range(1, self.num_targets_qb + 1))
            ]
            angles_column = sparse.vstack(angle_column_list)
            matrix_A[:, col_no] = angles_column

        # A global phase is to be expected and must be corrected.
        total_depth = int(np.ceil(np.log2(matrix_A.shape[0])))
        recovered_angles = sparse.dok_matrix((1, matrix_A.shape[1]), dtype=float)
        for col in range(matrix_A.shape[1]):
            row = 0
            evaluation = 1
            for depth in range(total_depth):
                evaluation *= np.exp(-0.5j * matrix_A[row, col])
                row += 2**depth
            recovered_angles[0, col] = np.angle(evaluation)
        global_phases = self.matrix_P_angle[0, :] - recovered_angles
        matrix_A = sparse.vstack([global_phases, matrix_A]).todok()
        return matrix_A

    @staticmethod
    def _chi(l):
        return 1 + sum([2**(i-1) for i in range(1, l)])

    def _define(self):
        y_angle_matrix = self._to_angle_matrix_y()
        z_angle_matrix = self._to_angle_matrix_z()

        no_z_rotations = abs(sparse.linalg.norm(z_angle_matrix)) < 1e-6

        control = QuantumRegister(self.num_controls_qb, "q^c")
        target = QuantumRegister(self.num_targets_qb, "q^t")
        qc_y = QuantumCircuit(control, target, name=self.name)
        qc_z = QuantumCircuit(control, target, name=self.name)

        for l in range(1, self.num_targets_qb + 1):
            qargs = list(control) + target[0:l]

            # If there are no z-rotations we save a lot of gates, so we want to rule that out
            if not no_z_rotations:
                angles_z: sparse.spmatrix = z_angle_matrix[1:, :][range(IsometryGate._chi(l) - 1, IsometryGate._chi(l + 1) - 1), :]
                angles_z = angles_z.reshape(-1, 1)
                gate_z = UniformRotationGate(gate=lambda a: RZGate(a), alpha=angles_z.todok())
                qc_z.append(gate_z, qargs)
                qc_z.barrier()

            angles_y: sparse.spmatrix = y_angle_matrix[range(IsometryGate._chi(l) - 1, IsometryGate._chi(l+1) - 1), :]
            angles_y = angles_y.reshape(-1, 1)
            gate_y = UniformRotationGate(gate=lambda a: RYGate(a), alpha=angles_y.todok())
            qc_y.append(gate_y, qargs)
            qc_y.barrier()

        if not no_z_rotations:
            # A relative phase correction is pretty intensive: a state preparation on the control
            global_phase_correction = MöttönenStatePreparationGate(
                sparse.dok_matrix(np.exp(1.0j * z_angle_matrix[0, :].T.toarray())),
                neglect_absolute_value=True
            )
            qc_z.append(global_phase_correction, qargs=control)
            qc_z.barrier()

        qc = qc_y + qc_z
        self._definition = qc
