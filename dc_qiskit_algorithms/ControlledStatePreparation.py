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

    _debug_flag: bool = False

    def __init__(self, matrix: sparse.dok_matrix) -> None:
        """
        The matrix has the following structure:

        A = (a_ij)

        with i being the control and j  being the target. Also, each row must be normalized, i.e.,

        \sum_j |a_ij|^2 = 1

        The matrix is the quantum equivalent of a stochastic matrix therefore.

        :param: matrix that has the columns as subspaces and rows as the states to create in them
        """

        if isinstance(matrix, list) or isinstance(matrix, np.ndarray):
            matrix = sparse.dok_matrix(matrix)

        # The columns define the target states, i.e. the space for the controlled state preparation
        num_targets_qb = int(math.ceil(math.log2(matrix.shape[1])))
        self.num_targets_qb = num_targets_qb
        # The rows define the number of control states. If there is one row, this is normal state preparation.
        num_controls_qb = int(math.ceil(math.log2(matrix.shape[0])))
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

    def set_debug_flag(self, flag: bool = False) -> 'ControlledStatePreparationGate':
        """
        If this is used, not the standard routine is used (Möttönen state preparation using uniform rotations)
        but a manual form of several controlled rotations. This makes the plotting and therefore the debugging a lot
        easier, at the expense of a lot of computational time and a strong increase of cx gates when later executed.
        Therefore it is really only for debugging!

        :param flag: false is the default, for true you get the multiple controlled rotations.
        :return: None
        """
        self._debug_flag = flag
        return self

    def _to_angle_matrix_y(self) -> Union[sparse.dok_matrix]:
        # First, for each column, the angles that lead to this state need to be computed.
        matrix_A = sparse.dok_matrix((self.matrix_abs.shape[0], self.matrix_abs.shape[1] - 1))
        for row_no in range(self.matrix_abs.shape[0]):
            amplitudes_row: sparse.csc_matrix = self.matrix_abs.getrow(row_no).T
            # The reversed is necessary as the "highest" qubit is the one with the least controls
            # imagine a circuit the where the highest qubits control the lower one. Yes this is all but numbering
            # so that this is why I need to add this comment.
            angle_row_list_y: List[sparse.dok_matrix] = [
                get_alpha_y(amplitudes_row.todok(), self.num_targets_qb, k)
                for k in reversed(range(1, self.num_targets_qb + 1))
            ]
            angles_row = sparse.vstack(angle_row_list_y)
            matrix_A[row_no, :] = angles_row.T

        return matrix_A

    def _to_angle_matrix_z(self) -> Tuple[sparse.spmatrix, sparse.spmatrix]:
        # First, for each column, the angles that lead to this state need to be computed.
        matrix_A = sparse.dok_matrix((self.matrix_angle.shape[0], self.matrix_angle.shape[1] - 1))
        for row_no in range(self.matrix_angle.shape[0]):
            amplitudes_row: sparse.csc_matrix = self.matrix_angle.getrow(row_no).T
            # The reversed is necessary as the "highest" qubit is the one with the least controls
            # imagine a circuit the where the highest qubits control the lower one. Yes this is all but numbering
            # so that this is why I need to add this comment.
            angle_row_list_z: List[sparse.dok_matrix] = [
                get_alpha_z(amplitudes_row.todok(), self.num_targets_qb, k)
                for k in reversed(range(1, self.num_targets_qb + 1))
            ]
            angles_row = sparse.vstack(angle_row_list_z)
            matrix_A[row_no, :] = angles_row.T

        # A global phase is to be expected on each subspace and must be corrected jointly later.
        total_depth = max(1, int(np.ceil(np.log2(matrix_A.shape[1]))))
        recovered_angles = sparse.dok_matrix((matrix_A.shape[0], 1), dtype=float)
        # Each row is a separate sub-space, and by the algorithm of Möttönen et al
        # a global phase is to be expected. So we calculate it by...
        for row in range(matrix_A.shape[0]):
            # ... going through each row and applying rz rotations essentially, but not on all
            # involved qubits, just for one branch as the global phase must exist, well, globally.
            col = 0
            evaluation = 1
            for depth in range(total_depth):
                evaluation *= np.exp(-0.5j * matrix_A[row, col])
                col += 2**depth
            # After calculating the amplitude of one branch, I take the angle/phase
            # This is still not the global phase, we will get that later...
            recovered_angles[row, 0] = np.angle(evaluation)

        # ... exactly here we take the difference of the phase of each subspace and the angle
        # matrix. That is the global phase of that subspace!
        global_phases: sparse.spmatrix = self.matrix_angle[:, 0] - recovered_angles
        return matrix_A, global_phases

    def _define(self):
        # The y angle matrix stands for the absolute value, while the z angles stand for phases
        # The difficulty lies in the "global" phases that must be later accounted for in each subspace
        y_angle_matrix = self._to_angle_matrix_y()
        z_angle_matrix, global_phases = self._to_angle_matrix_z()

        if self._debug_flag:
            qc_y, qc_z = self._create_debug_circuit(y_angle_matrix, z_angle_matrix, global_phases)
        else:
            qc_y, qc_z = self._create_production_circuit(y_angle_matrix, z_angle_matrix, global_phases)

        qc = qc_y + qc_z
        self._definition = qc

    def _create_production_circuit(self, y_angle_matrix, z_angle_matrix, global_phases):

        # As the subspace phase correction is a very expensive module, we only want to do it if the
        # z rotation matrix is non-zero!
        no_z_rotations = abs(sparse.linalg.norm(z_angle_matrix)) < 1e-6

        control = QuantumRegister(self.num_controls_qb, "q^c")
        target = QuantumRegister(self.num_targets_qb, "q^t")
        qc_y = QuantumCircuit(control, target, name=self.name)
        qc_z = QuantumCircuit(control, target, name=self.name)

        # We do a set of uniformly controlled operations from the control register on each target qubit.
        # iterating through the target means that a target qubit that has been handled (i.e. was a target of a
        # uniform rotation) becomes a controlling qubit.
        # Thus, when creating the qargs for the operation, there are going to be all qubits of the control register
        # plus those target register qubits that have been used plus the current target qubit as the target
        # of the uniform rotation
        for l in range(1, self.num_targets_qb + 1):
            # Using slicing to get the correct target register qubits, start at the last, go to the (l+1) last.
            # The ordering of the qubits in the register is a bit weird... explanation:
            # ctr = [ctr0, ctr1, ctr2, ctr3], tgt = tgt0, tgt1, tgt2, tgt3
            # l = 2: qargs = [ctr0, ctr1, ctr2, ctr,3, tgt3, tgt2]
            #                                                  ^==> target of operation
            # the Möttönen et al. scheme uses the idea to move a ket |a> to |0> via U. The inverse of this operation
            # thus will take |0> to the desired state |a>. The first step of constructing U is to cancel the first qubit
            # (tgt0) to be only having the ground state contributing in a product state, this is controlled
            # by [tgt1, tgt2, tgt3]. Then (tgt1) is targeted (controls [tgt2, tgt3]) until (tgt3) is reached
            # (no control). This is inverted, thus first (tgt3) until (tgt0) with corresponding controls.
            qargs = list(control) + target[-1:-1-l:-1]

            # If there are no z-rotations we save a lot of gates, so we want to rule that out
            if not no_z_rotations:
                # The corresponding rotational angles are given in the Z-angle matrix by the function chi selecting
                # on the columns according to the operational parameter l.
                angles_z: sparse.spmatrix = z_angle_matrix[:, range(_chi(l) - 1, _chi(l + 1) - 1)]
                # The column-first Fortran-style reshaping to create one angle vector
                angles_z = angles_z.reshape(-1, 1, order='F')
                # The negative of the angles is needed to implement the inverse (as described above)
                gate_z = UniformRotationGate(gate=lambda a: RZGate(a), alpha=angles_z.todok())  # FIXME: removing the -
                qc_z.append(gate_z, qargs)

            # The uniform rotation for the y rotation will take care of the absolute value
            # The corresponding rotational angles are given in the Z-angle matrix by the function chi selecting
            # on the columns according to the operational parameter l.
            angles_y: sparse.spmatrix = y_angle_matrix[:, range(_chi(l) - 1, _chi(l + 1) - 1)]
            # The column-first Fortran-style reshaping to create one angle vector
            angles_y = angles_y.reshape(-1, 1, order='F')
            # The negative of the angles is needed to implement the inverse (as described above)
            gate_y = UniformRotationGate(gate=lambda a: RYGate(a), alpha=angles_y.todok())  # FIXME: removing the -
            qc_y.append(gate_y, qargs)

        if not no_z_rotations:
            # A relative phase correction is pretty intensive: a state preparation on the control
            global_phase_correction = MöttönenStatePreparationGate(
                sparse.dok_matrix(np.exp(1.0j * global_phases.toarray())),
                neglect_absolute_value=True
            )
            qc_z.append(global_phase_correction, qargs=control)

        return qc_y, qc_z

    def _create_debug_circuit(self, y_angle_matrix, z_angle_matrix, global_phases):

        # As the subspace phase correction is a very expensive module, we only want to do it if the
        # z rotation matrix is non-zero!
        no_z_rotations = abs(sparse.linalg.norm(z_angle_matrix)) < 1e-6

        control = QuantumRegister(self.num_controls_qb, "q^c")
        target = QuantumRegister(self.num_targets_qb, "q^t")
        qc_y = QuantumCircuit(control, target, name=self.name)
        qc_z = QuantumCircuit(control, target, name=self.name)

        for row in range(y_angle_matrix.shape[0]):
            qc_y_row = QuantumCircuit(control, target, name=self.name)
            for (_, j), angle in y_angle_matrix.getrow(row).todok().items():
                num_extra_control = int(np.floor(np.log2(j + 1)))
                num_control = len(control) + num_extra_control
                val_control = row + 2 ** len(control) * int(j - (2 ** num_extra_control - 1))
                gate = RYGate(angle).control(num_ctrl_qubits=num_control, ctrl_state=val_control)
                qargs = list(control) + target[-1:-2 - j:-1]
                qc_y_row.append(gate, qargs)
            qc_y += qc_y_row
        if not no_z_rotations:
            for row in range(z_angle_matrix.shape[0]):
                for (_, j), angle in z_angle_matrix.getrow(row).todok().items():
                    num_extra_control = int(np.floor(np.log2(j + 1)))
                    num_control = len(control) + num_extra_control
                    val_control = row + 2 ** len(control) * int(j - (2 ** num_extra_control - 1))
                    gate = RZGate(angle).control(num_ctrl_qubits=num_control, ctrl_state=val_control)
                    qargs = list(control) + target[-1:-2 - j:-1]
                    qc_z.append(gate, qargs)

        if not no_z_rotations:
            # A relative phase correction is pretty intensive: a state preparation on the control
            global_phase_correction = MöttönenStatePreparationGate(
                sparse.dok_matrix(np.exp(1.0j * global_phases.toarray())),
                neglect_absolute_value=True
            )
            qc_z.append(global_phase_correction, qargs=control)

        qc_y.barrier()
        qc_z.barrier()
        return qc_y, qc_z
