Data Cybernetics qiskit-algorithms
###################################

.. image:: https://img.shields.io/travis/com/carstenblank/dc-qiskit-algorithms/master.svg?style=for-the-badge
    :alt: Travis
    :target: https://travis-ci.com/carstenblank/dc-qiskit-algorithms

.. image:: https://img.shields.io/codecov/c/github/carstenblank/dc-qiskit-algorithms/master.svg?style=for-the-badge
    :alt: Codecov coverage
    :target: https://codecov.io/gh/carstenblank/dc-qiskit-algorithms

.. image:: https://img.shields.io/codacy/grade/f4132f03ce224f82bd3e8ba436b52af3.svg?style=for-the-badge
    :alt: Codacy grade
    :target: https://www.codacy.com/app/carstenblank/dc-qiskit-algorithms?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=carstenblank/dc-qiskit-algorithms&amp;utm_campaign=Badge_Grade

.. image:: https://img.shields.io/readthedocs/dc-qiskit-algorithms.svg?style=for-the-badge
    :alt: Read the Docs
    :target: https://dc-qiskit-algorithms.readthedocs.io

.. image:: https://img.shields.io/pypi/v/dc-qiskit-algorithms.svg?style=for-the-badge
    :alt: PyPI
    :target: https://pypi.org/project/dc-qiskit-algorithms

.. image:: https://img.shields.io/pypi/pyversions/dc-qiskit-algorithms.svg?style=for-the-badge
    :alt: PyPI - Python Version
    :target: https://pypi.org/project/dc-qiskit-algorithms

.. header-start-inclusion-marker-do-not-remove

`qiskit <https://qiskit.org/documentation/>`_ is an open-source compilation framework capable of targeting various
types of hardware and a high-performance quantum computer simulator with emulation capabilities, and various
compiler plug-ins.

This library sports some useful algorithms for quantum computers using qiskit as a basis.


Features
========

* Multi Qubit Quantum Fourier Transform

* Draper adder

* Uniform Rotations

* State Preparation

.. header-end-inclusion-marker-do-not-remove
.. installation-start-inclusion-marker-do-not-remove

Installation
============

This library requires Python version 3.5 and above, as well as qiskit.
Installation of this plugin, as well as all dependencies, can be done using pip:

.. code-block:: bash

    $ python -m pip install dc_qiskit_algorithms

To test that the algorithms are working correctly you can run

.. code-block:: bash

    $ make test

.. installation-end-inclusion-marker-do-not-remove
.. gettingstarted-start-inclusion-marker-do-not-remove

Getting started
===============

You can use the state preparation as follows

.. code-block:: python

    from dc_qiskit_algorithms.MottonenStatePrep import state_prep_möttönen

    vector = [-0.1, 0.2, -0.3, 0.4, -0.5, 0.6, -0.7, 0.8]
    vector = numpy.asarray(vector)
    vector = (1 / numpy.linalg.norm(vector)) * vector

    qubits = int(numpy.log2(len(vector)))
    reg = QuantumRegister(qubits, "reg")
    c = ClassicalRegister(qubits, "c")
    qc = QuantumCircuit(reg, c, name='state prep')
    state_prep_möttönen(qc, vector, reg)

After this, the quantum circuit is prepared in the given state. All complex vectors are possible!

.. gettingstarted-end-inclusion-marker-do-not-remove

Please refer to the `documentation of the dc qiskit algorithm Plugin <https://dc-qiskit-algorithms.readthedocs.io/>`_ .

.. howtocite-start-inclusion-marker-do-not-remove


Contributing
============

We welcome contributions - simply fork the repository of this plugin, and then make a
`pull request <https://help.github.com/articles/about-pull-requests/>`_ containing your contribution.
All contributers to this plugin will be listed as authors on the releases.

We also encourage bug reports, suggestions for new features and enhancements, and even links to cool projects or applications built on PennyLane.


Authors
=======

Carsten Blank

.. support-start-inclusion-marker-do-not-remove

Support
=======

- **Source Code:** https://github.com/carstenblank/qiskit-algorithms
- **Issue Tracker:** https://github.com/carstenblank/qiskit_algorithms/issues

If you are having issues, please let us know by posting the issue on our Github issue tracker.

.. support-end-inclusion-marker-do-not-remove
.. license-start-inclusion-marker-do-not-remove

License
=======

The data cybernetics qiskit algorithms plugin is **free** and **open source**, released under
the `Apache License, Version 2.0 <https://www.apache.org/licenses/LICENSE-2.0>`_.

.. license-end-inclusion-marker-do-not-remove
