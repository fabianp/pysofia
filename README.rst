
PySofia
=============================

PySofia is a python wrapper around the methods present in the C++ sofia-ml library. These include Stochastic Gradient Descent implementations of some ranking algorithms, notably RankSVM.

Dependencies
------------

  - cython >= 0.17 (previous versions will not work)
  - numpy

Installation
============

    $ pip install -U pysofia


Methods
=======

pysofia.train_sgd

    Trains a model using stochastic gradient descent. See docstring for
    more details.

pysofia.compat.RankSVM implements an estimator following the conventions
used in scikit-learn.
