import math
from typing import List, Union, Tuple

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Gate
from qiskit.circuit.library import RYGate, RZGate
from scipy import sparse
from scipy.sparse.linalg import norm

from dc_qiskit_algorithms import UniformRotationGate
from dc_qiskit_algorithms.MöttönenStatePreparation import get_alpha_y, get_alpha_z, MöttönenStatePreparationGate


def _chi(l):
    return 1 + sum([2 ** (i - 1) for i in range(1, l)])


class ControlledStatePreparationGate(Gate):

    def __init__(self, matrix: sparse.dok_matrix) -> None:
        """
        matrix = A = (a_ij)
        with j being the control and

        :param: matrix that has the columns as subspaces and rows as the states to create in them
        """

        if isinstance(matrix, list) or isinstance(matrix, np.ndarray):
            matrix = sparse.dok_matrix(matrix)

        # The rows define the target states, i.e. the space for the controlled state preparation
        num_targets_qb = int(math.ceil(math.log2(matrix.shape[0])))
        self.num_targets_qb = num_targets_qb
        # The columns define the number of control states. If there is one row, this is normal state preparation.
        num_controls_qb = int(math.ceil(math.log2(matrix.shape[1])))
        self.num_controls_qb = num_controls_qb

        super().__init__("iso_matrix", num_qubits=num_controls_qb + num_targets_qb, params=[])

        # Of course we save the matrix but we need to calculate the absolute value and the angle/phase
        # matrix for the core logic
        self.matrix = matrix  # type: Union[sparse.dok_matrix]
        matrix_abs = sparse.dok_matrix(matrix.shape)
        matrix_angle = sparse.dok_matrix(matrix.shape)
        for (i, j), v in self.matrix.items():
            matrix_abs[i, j] = np.absolute(v)
            matrix_angle[i, j] = np.angle(v)
        self.matrix_abs = matrix_abs
        self.matrix_angle = matrix_angle

    def _to_angle_matrix_y(self) -> Union[sparse.dok_matrix]:
        # First, for each column, the angles that lead to this state need to be computed.
        matrix_A = sparse.dok_matrix((self.matrix_abs.shape[0] - 1, self.matrix_abs.shape[1]))
        for col_no in range(self.matrix_abs.shape[1]):
            amplitudes_column: sparse.csc_matrix = self.matrix_abs.getcol(col_no)
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

    def _to_angle_matrix_z(self) -> Tuple[sparse.spmatrix, sparse.spmatrix]:
        # First, for each column, the angles that lead to this state need to be computed.
        matrix_A = sparse.dok_matrix((self.matrix_angle.shape[0] - 1, self.matrix_angle.shape[1]))
        for col_no in range(self.matrix_angle.shape[1]):
            amplitudes_column: sparse.csc_matrix = self.matrix_angle.getcol(col_no)
            # The reversed is necessary as the "highest" qubit is the one with the least controls
            # imagine a circuit the where the highest qubits control the lower one. Yes this is all but numbering
            # so that this is why I need to add this comment.
            angle_column_list: List[sparse.dok_matrix] = [
                get_alpha_z(amplitudes_column.todok(), self.num_targets_qb, k)
                for k in reversed(range(1, self.num_targets_qb + 1))
            ]
            angles_column = sparse.vstack(angle_column_list)
            matrix_A[:, col_no] = angles_column

        # A global phase is to be expected on each subspace and must be corrected jointly later.
        total_depth = int(np.ceil(np.log2(matrix_A.shape[0])))
        recovered_angles = sparse.dok_matrix((1, matrix_A.shape[1]), dtype=float)
        # Each row is a separate sub-space, and by the algorithm of Möttönen et al
        # a global phase is to be expected. So we calculate it by...
        for col in range(matrix_A.shape[1]):
            # ... going through each row and applying rz rotations essentially, but not on all
            # involved qubits, just for one branch as the global phase must exist, well, globally.
            row = 0
            evaluation = 1
            for depth in range(total_depth):
                evaluation *= np.exp(-0.5j * matrix_A[row, col])
                row += 2**depth
            # After calculating the amplitude of one branch, I take the angle/phase
            # This is still not the global phase, we will get that later...
            recovered_angles[0, col] = np.angle(evaluation)

        # ... exactly here we take the difference of the phase of each subspace and the angle
        # matrix. That is the global phase of that subspace!
        global_phases: sparse.spmatrix = self.matrix_angle[0, :] - recovered_angles
        return matrix_A, global_phases

    def _define(self):
        # The y angle matrix stands for the absolute value, while the z angles stand for phases
        # The difficulty lies in the "global" phases that must be later accounted for in each subspace
        y_angle_matrix = self._to_angle_matrix_y()
        z_angle_matrix, global_phases = self._to_angle_matrix_z()

        # As the subspace phase correction is a very expensive module, we only want to do it if the
        # z rotation matrix is non-zero!
        no_z_rotations = abs(sparse.linalg.norm(z_angle_matrix)) < 1e-6

        control = QuantumRegister(self.num_controls_qb, "q^c")
        target = QuantumRegister(self.num_targets_qb, "q^t")
        qc_y = QuantumCircuit(control, target, name=self.name)
        qc_z = QuantumCircuit(control, target, name=self.name)

        for l in range(1, self.num_targets_qb + 1):
            qargs = list(control) + target[0:l]

            # If there are no z-rotations we save a lot of gates, so we want to rule that out
            if not no_z_rotations:
                angles_z: sparse.spmatrix = z_angle_matrix[range(ControlledStatePreparationGate._chi(l) - 1, ControlledStatePreparationGate._chi(l + 1) - 1), :]
                angles_z = angles_z.reshape(-1, 1)
                gate_z = UniformRotationGate(gate=lambda a: RZGate(a), alpha=angles_z.todok())
                qc_z.append(gate_z, qargs)
                qc_z.barrier()

            # The uniform rotation for the y rotation will take care of the absolute value
            angles_y: sparse.spmatrix = y_angle_matrix[range(ControlledStatePreparationGate._chi(l) - 1, ControlledStatePreparationGate._chi(l + 1) - 1), :]
            angles_y = angles_y.reshape(-1, 1)
            gate_y = UniformRotationGate(gate=lambda a: RYGate(a), alpha=angles_y.todok())
            qc_y.append(gate_y, qargs)
            qc_y.barrier()

        if not no_z_rotations:
            # A relative phase correction is pretty intensive: a state preparation on the control
            global_phase_correction = MöttönenStatePreparationGate(
                sparse.dok_matrix(np.exp(1.0j * global_phases.T.toarray())),
                neglect_absolute_value=True
            )
            qc_z.append(global_phase_correction, qargs=control)
            qc_z.barrier()

        qc = qc_y + qc_z
        self._definition = qc
