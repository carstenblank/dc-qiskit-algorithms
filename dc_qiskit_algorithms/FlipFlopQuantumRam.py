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
r"""
FlipFlopQuantumRam
====================

.. currentmodule:: dc_qiskit_algorithms.FlipFlopQuantumRam

This module implements the state preparation scheme called FFQRAM see https://arxiv.org/abs/1901.02362.

.. autosummary::
   :nosignatures:

   FFQramEntry
   FFQramDb
   add_vector

Each DB has entries that are created by controlled rotations. The final step is a measurement to cancel out
the wrong branch. This makes the algorithm probabilistic in its nature.

FFQramEntry
#############

.. autoclass:: FFQramEntry

FFQramDb
##########

.. autoclass:: FFQramEntry


add_vector
###########

-- autofunction:: add_vector

"""

import math
from typing import Tuple, List, Union

from bitarray import bitarray
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.circuit.register import Register
from qiskit.extensions.standard.x import x

from .UniformRotation import cnry


class FFQramEntry(object):
    """
    An DB entry of the FF QRAM scheme
    """

    def __init__(self):
        """
        Creates an entry with binary data & label as well as an (optional) amplitude
        """
        self.probability_amplitude = 0.0  # type: float
        self.data = bytes()  # type: bytes
        self.label = bytes()  # type: bytes

    def get_bits(self):
        # type: (FFQramEntry) -> bitarray
        """
        Get the binary bit representation of data and label
        for state basis identification

        :return: a bit array
        """
        b = self.data + self.label
        ba = bitarray()
        ba.frombytes(b)
        ba = bitarray(ba.to01().lstrip('0'))
        ba.reverse()
        return ba

    def bus_size(self):
        # type: (FFQramEntry) -> int
        """
        Returns needed bus size for this entry

        :return: the length
        """
        return self.get_bits().length()

    def add_to_circuit(self, qc, bus, register):
        # type: (FFQramEntry, QuantumCircuit, Union[QuantumRegister, list], Tuple[QuantumRegister, int]) -> QuantumCircuit
        """
        This method adds the gates to encode this entry into the circuit
        :param qc: quantum circuit to apply the entry to
        :param bus: the registers for the bus
        :param register: the target register for the amplitude
        :return: the applied circuit
        """
        theta = math.asin(self.probability_amplitude)
        if theta == 0:
            return qc
        bus_register = []  # type: List[Tuple[QuantumRegister, int]]
        if isinstance(bus, QuantumRegister):
            bus_register = [(bus, i) for i in range(bus.size)]
        else:
            bus_register = bus

        ba = self.get_bits()
        for i in range(len(bus_register) - ba.length()):
            ba.append(False)

        for i, b in enumerate(ba):
            if not b: x(qc, bus_register[i])

        cnry(qc, theta, bus_register, register)

        for i, b in enumerate(ba):
            if not b: x(qc, bus_register[i])

        return qc

    def __str__(self):
        return "FFQramEntry(%.8f, %s)" % (self.probability_amplitude, self.get_bits().to01())

    @staticmethod
    def _count_set_bits(b):
        # type: (bytes) -> int
        """
        Returns the number of ones in the byte array
        :param b: the data
        :return: the count
        """
        ba = bitarray()
        ba.frombytes(b)
        return ba.count()


class FFQramDb(List[FFQramEntry]):
    """
    The DB object with methods to create circuits
    """

    def bus_size(self):
        # type: (FFQramDb) -> int
        """
        From all entries get the maximum needed bus size

        :return: the bus size for the DB
        """
        return max([e.bus_size() for e in self])

    def add_to_circuit(self, qc, bus, register):
        # type: (FFQramDb, QuantumCircuit, Union[QuantumRegister, list], Tuple[Union[QuantumRegister, Register], int]) -> None
        """
        Add the DB to the circuit.

        :param qc: the quantum circuit
        :param bus: the bus register
        :param register: the target register for the amplitudes
        :return: the circuit after DB being applied
        """
        if not isinstance(register[0], QuantumRegister):
            raise Exception("Register must be a QuantumRegister!")  # type: Tuple[QuantumRegister, int]
        reg = (register[0], register[1])
        for entry in self:
            entry.add_to_circuit(qc, bus, reg)

    def add_entry(self, pa, data, label):
        # type: (FFQramDb, float, bytes, bytes) -> None
        """
        Add an entry to the (classical representation of) the DB.

        :param pa: probability amplitude
        :param data: binary representation of data
        :param label: binary representation of the label
        """
        entry = FFQramEntry()
        entry.probability_amplitude = pa
        entry.data = data
        entry.label = label
        self.append(entry)

    def add_entry_int(self, pa, data, label):
        # type: (FFQramDb, float, int, int) -> None
        """
        Add an entry to the (classical representation of) the DB.

        :param pa: probability amplitude
        :param data: the integer value of the data
        :param label: the integer value of the label
        """
        data_bits = [d == '1' for d in "{0:b}".format(data)]
        label_bits = [d == '1' for d in "{0:b}".format(label)]
        data_bits.reverse()
        label_bits.reverse()
        data_bytes = bitarray(data_bits, endian='little').tobytes()
        label_bytes = bitarray(label_bits, endian='little').tobytes()
        self.add_entry(pa, data_bytes, label_bytes)


def add_vector(db, vec):
    # type: (FFQramDb, List[complex]) -> None
    """
    Add a vector to the DB. It makes sense to give an empty DB.

    :param db: The FFQRAM DB
    :param vec: the vector to be added
    """
    import numpy as np
    vector = np.asarray(vec)
    l2_norm = np.linalg.norm(vector)
    unit_vector = vector / l2_norm
    for i, v in enumerate(unit_vector):
        db.add_entry_int(v, 0, i)
