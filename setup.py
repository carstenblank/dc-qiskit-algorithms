#!/usr/bin/env python

from setuptools import setup

setup(name='qiskit_algorithms',
      version='1.0',
      description='General algorithms with qiskit as basis',
      author='Carsten Blank',
      author_email='blank@data-cybernetics.com',
      url='https://data-cybernetics.com',
      packages=['qiskit_algorithms'],
      install_requires=['qiskit', 'numpy', 'scipy', 'bitstring', 'scikit_learn', 'ddt']
      )
