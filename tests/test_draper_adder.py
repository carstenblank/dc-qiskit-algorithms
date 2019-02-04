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

from dc_qiskit_algorithms.draper_adder import draper_adder

logging.basicConfig(format=logging.BASIC_FORMAT, level='INFO')
LOG = logging.getLogger('test_DraperAdder')


@ddt
class DraperAdderTest(unittest.TestCase):
    """Test class for the systematic test of the Draper adder."""

    @test_data(
        (0, 0, None), (0, 1, None), (0, 2, None), (0, 3, None),
        (1, 0, None), (1, 1, None), (1, 2, None), (1, 3, None),
        (2, 0, None), (2, 1, None), (2, 2, None), (2, 3, None),
        (3, 0, None), (3, 1, None), (3, 2, None), (3, 3, None),
        (0, 0, 2), (0, 1, 2), (0, 2, 2), (0, 3, 2),
        (1, 0, 2), (1, 1, 2), (1, 2, 2), (1, 3, 2),
        (2, 0, 2), (2, 1, 2), (2, 2, 2), (2, 3, 2),
        (3, 0, 2), (3, 1, 2), (3, 2, 2), (3, 3, 2),
        (7, 0, None), (7, 1, None), (7, 2, None),
        (7, 3, None), (7, 4, None), (7, 5, None),
        (7, 6, None), (7, 7, None)
    )
    @unpack
    def test_two_bit_adder(self, a_input, b_input, length):
        # type: (DraperAdderTest, int, int, int) -> None
        """
        Test the addition of a and b. The length is used to override the
        automatic qubit size calculation.

        :param a_input: an integer
        :param b_input: an integer
        :param length: a fixed qubit register size for both inputs.
        :return: nothing
        """
        LOG.info("Testing 'DraperAdder' with a=%d(%s), b=%d(%s).",
                 a_input, "{0:b}".format(a_input), b_input, "{0:b}".format(b_input))
        q_circuit, modulo = draper_adder(a_input, b_input, length)

        from qiskit import compile as q_compile

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


if __name__ == '__main__':
    unittest.main(verbosity=2)
