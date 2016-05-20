.. image:: https://travis-ci.org/fabianp/pysofia.svg?branch=master
    :target: https://travis-ci.org/fabianp/pysofia

Maintainer wanted
=================
This project is looking for a maintaner, please contact me if you would like to adopt it.

PySofia
=============================

PySofia is a python wrapper around the methods present in the C++ sofia-ml library. These include Stochastic Gradient Descent implementations of some ranking algorithms, notably RankSVM.

Dependencies
------------

  - cython >= 0.17 (previous versions will not work)
  - numpy
  - sklearn => 0.15
  - six
  - enum34 (for Python versions before 3.4)
  - A C++ compiler (gcc will do)

You will not need to install sofia-ml since it is incuded in this distribution

Installation
============

    $ pip install -U git+https://github.com/fabianp/pysofia.git


Methods
=======

pysofia.train_svm

    Trains a model using stochastic gradient descent. See docstring for
    more details.

pysofia.compat.RankSVM implements an estimator following the conventions
used in scikit-learn.

Development
===========

Check out the latest version at github: https://github.com/fabianp/pysofia/

License
=======

Apache License 2.0

Authors
=======

PySofia is the work of Fabian Pedregosa. The sofia-ml library is written by D. Sculley.
