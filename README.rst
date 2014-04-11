
PySofia
=============================

PySofia is a python wrapper around the methods present in the C++ sofia-ml library. These include Stochastic Gradient Descent implementations of some ranking algorithms, notably RankSVM.

Dependencies
------------

  - cython >= 0.17 (previous versions will not work)
  - numpy
  - A C++ compiler (gcc will do)

You will not need to install sofia-ml since it is incuded in this distribution

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


Authors
=======

PySofia is the work of Fabian Pedregosa. The sofia-ml library is written by D. Sculley.