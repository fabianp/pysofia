import sys, tempfile
import numpy as np
from sklearn import datasets
import _sofia_ml

from enum import Enum

if sys.version_info[0] < 3:
    bstring = basestring
else:
    bstring = str

class learner_type(Enum):
    r"""
    define learner type
    """
    pegasos = 0
    margin_perceptron = 1
    passive_aggressive = 2
    logreg_pegasos = 3
    logreg = 4
    lms_regression = 5
    sgd_svm = 6
    romma = 7

class loop_type(Enum):
    r"""
    define loop type
    """
    rank = 'rank'
    roc = 'roc'
    combined_ranking = 'combined-ranking'
    stochastic = 'stochastic'
    balanced_stochastic = 'balanced-stochastic'

def svm_train(X, y, b, alpha, n_features, learner, loop,
              max_iter=100, step_probability=0.5):
    """
    Minimizes an expression of the form

        Loss(X, y, b) + 0.5 * alpha * (||w|| ** 2)

    where Loss is an Hinge loss defined on pairs of images

    Parameters
    ----------
    X : input data

    y : target labels

    b : blocks (aka query_id)

    alpha: float

    loop : {'rank', 'combined-ranking', 'roc', 'stochastic', 'balanced-stochastic'}

    Returns
    -------
    coef

    None
    """
    if isinstance(X, bstring):
        if n_features is None:
            n_features = X.shape[1] # the default in sofia-ml TODO: parse file to see
        w = _sofia_ml.train(X, n_features, alpha, max_iter, False,
                            learner.value, loop.value, step_probability)
    else:
        with tempfile.NamedTemporaryFile() as f:
            datasets.dump_svmlight_file(X, y, f.name, query_id=b)
            w = _sofia_ml.train(f.name, X.shape[1], alpha, max_iter, False,
                                learner.value, loop.value, step_probability)
    return w, None

def svm_predict(data, coef, blocks=None):
    # TODO: isn't query_id in data ???
    s_coef = ''
    for e in coef:
        s_coef += '%.5f ' % e
    s_coef = s_coef[:-1]
    if isinstance(data, bstring):
        return _sofia_ml.predict(data, s_coef, False)
    else:
        X = np.asarray(data)
        if blocks is None:
            blocks = np.ones(X.shape[0])
        with tempfile.NamedTemporaryFile() as f:
            y = np.ones(X.shape[0])
            datasets.dump_svmlight_file(X, y, f.name, query_id=blocks)
            prediction = _sofia_ml.predict(f.name, s_coef, False)
        return prediction
