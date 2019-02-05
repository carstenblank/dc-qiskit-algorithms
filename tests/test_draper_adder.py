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
Tests for the Draper adder using a couple of examples.
"""

import unittest
import logging

import qiskit
from ddt import ddt, data as test_data, unpack
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.extensions import standard

from dc_qiskit_algorithms.draper_adder import draper_adder

logging.basicConfig(format=logging.BASIC_FORMAT, level='INFO')
LOG = logging.getLogger('test_DraperAdder')


@ddt
class DraperAdderTest(unittest.TestCase):
    """Test class for the systematic test of the Draper adder."""

    def _test(self, a_input, b_input, q_circuit, modulo):
        """
        Test the addition of a and b.

        :param a_input: an integer
        :param b_input: an integer
        :param q_circuit: the circuit to test
        :param modulo: must be used for testing
        :return: nothing
        """
        from qiskit import compile as q_compile

        LOG.info("Testing 'DraperAdder' with a=%d(%s), b=%d(%s).",
                 a_input, "{0:b}".format(a_input), b_input, "{0:b}".format(b_input))

        backend = qiskit.Aer.get_backend('qasm_simulator')
        qobj = q_compile([q_circuit], backend=backend, shots=8192)

        job = backend.run(qobj)
        result_list = [{'b': k[::-1].split(' ')[1], 'a': k[::-1].split(' ')[0], 'counts': v}
                       for k, v in job.result().get_counts().items()]

        LOG.info(result_list)

        self.assertEqual(len(result_list), 1)

        data = result_list[0]  # type: dict
        self.assertEqual(int(data['b'], 2), b_input, "Register b must be unchanged!")
        self.assertEqual(int(data['a'], 2), (a_input + b_input) % modulo, "Addition must be correctly performed!")

    @test_data(
        (0, 0), (0, 1), (0, 2), (0, 3),
        (1, 0), (1, 1), (1, 2), (1, 3),
        (2, 0), (2, 1), (2, 2), (2, 3),
        (3, 0), (3, 1), (3, 2), (3, 3),
        (7, 0), (7, 1), (7, 2),
        (7, 3), (7, 4), (7, 5),
        (7, 6), (7, 7)
    )
    @unpack
    def test_with_length_none(self, a_input, b_input):
        # type: (DraperAdderTest, int, int) -> None
        q_circuit, modulo = draper_adder(a_input, b_input, None)
        self._test(a_input, b_input, q_circuit, modulo)

    @test_data(
        (0, 0), (0, 1), (0, 2), (0, 3),
        (1, 0), (1, 1), (1, 2), (1, 3),
        (2, 0), (2, 1), (2, 2), (2, 3),
        (3, 0), (3, 1), (3, 2), (3, 3)
    )
    @unpack
    def test_with_two_bits(self, a_input, b_input):
        # type: (DraperAdderTest, int, int) -> None
        q_circuit, modulo = draper_adder(a_input, b_input, 2)
        self._test(a_input, b_input, q_circuit, modulo)

    @test_data((7, 0), (7, 1), (7, 2), (7, 3), (7, 4), (7, 5), (7, 6), (7, 7))
    @unpack
    def test_with_three_bits(self, a_input, b_input):
        # type: (DraperAdderTest, int, int) -> None
        q_circuit, modulo = draper_adder(a_input, b_input, 3)
        self._test(a_input, b_input, q_circuit, modulo)

    @test_data((7, 0), (7, 1), (7, 2), (7, 3), (7, 4), (7, 5), (7, 6), (7, 7))
    @unpack
    def test_with_three_bits_with_barriers(self, a_input, b_input):
        # type: (DraperAdderTest, int, int) -> None
        q_circuit, modulo = draper_adder(a_input, b_input, 3, with_barriers=True)
        self._test(a_input, b_input, q_circuit, modulo)

    @test_data((7, 0), (7, 1), (7, 2), (7, 3), (7, 4), (7, 5), (7, 6), (7, 7))
    @unpack
    def test_with_three_bits_with_pre_circuit(self, a_input, b_input):
        # type: (DraperAdderTest, int, int) -> None
        a_reg = QuantumRegister(3, "a")
        b_reg = QuantumRegister(3, "b")
        pre_q_circuit = QuantumCircuit(a_reg, b_reg, name="pre-circuit")
        standard.x(pre_q_circuit, a_reg)
        standard.x(pre_q_circuit, b_reg)
        standard.x(pre_q_circuit, a_reg)
        standard.x(pre_q_circuit, b_reg)
        q_circuit, modulo = draper_adder(a_input, b_input, 3, pre_circuit=pre_q_circuit)
        self._test(a_input, b_input, q_circuit, modulo)


if __name__ == '__main__':
    unittest.main(verbosity=2)
