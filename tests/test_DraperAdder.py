# -*- coding: utf-8 -*-

# Copyright 2018, Carsten Blank.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

import unittest
import logging

import qiskit
from ddt import ddt, data as test_data, unpack

from qiskit_algorithms.DraperAdder import draper_adder

logging.basicConfig(format=logging.BASIC_FORMAT, level='INFO')
log = logging.getLogger('test_DraperAdder')


@ddt
class DraperAdderTwoBitTest(unittest.TestCase):

    @test_data(
        (0, 0, None), (0, 1, None), (0, 2, None), (0, 3, None),
        (1, 0, None), (1, 1, None), (1, 2, None), (1, 3, None),
        (2, 0, None), (2, 1, None), (2, 2, None), (2, 3, None),
        (3, 0, None), (3, 1, None), (3, 2, None), (3, 3, None),
        (0, 0, 2), (0, 1, 2), (0, 2, 2), (0, 3, 2),
        (1, 0, 2), (1, 1, 2), (1, 2, 2), (1, 3, 2),
        (2, 0, 2), (2, 1, 2), (2, 2, 2), (2, 3, 2),
        (3, 0, 2), (3, 1, 2), (3, 2, 2), (3, 3, 2)
    )
    # @test_data((7, 1, None), (7, 2, None), (7, 3, None))
    @unpack
    def test_two_bit_adder(self, a, b, length):
        log.info("Testing 'DraperAdder' with a=%d(%s), b=%d(%s).",
                 a, "{0:b}".format(a), b, "{0:b}".format(b))
        qc, modulo = draper_adder(a, b, length)

        from qiskit import transpiler

        backend = qiskit.Aer.get_backend('qasm_simulator')
        qobj = transpiler.compile([qc], backend=backend, shots=8192)

        job = backend.run(qobj)
        result_list = [{'b': k[::-1].split(' ')[1], 'a': k[::-1].split(' ')[0], 'counts': v}
         for k, v in job.result().get_counts().items()]

        log.info(result_list)

        self.assertEqual(len(result_list), 1)

        data: dict = result_list[0]
        self.assertEqual(int(data['b'], 2), b, "Register b must be unchanged!")
        self.assertEqual(int(data['a'], 2), (a + b) % modulo, "Addition must be correctly performed!")


if __name__ == '__main__':
        unittest.main(verbosity=2)
