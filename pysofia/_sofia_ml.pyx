# distutils: language = c++
# distutils: sources = minirank/src/sofia-ml-methods.cc ranking/src/{sf-weight-vector.cc,sf-data-set.cc}
from cython.operator cimport dereference as deref
from libcpp cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector

cimport numpy as np
import numpy as np

BUFFER_MB = 40 # default in sofia-ml

cdef extern from "src/sofia-ml-methods.h":
    cdef cppclass SfDataSet:
        SfDataSet(bool)
        SfDataSet(string, int, bool)
        SfDataSet(double *, double *, int, int, bool)

    cdef cppclass SfWeightVector:
        SfWeightVector(int)
        SfWeightVector(string)
        SfWeightVector(double *, int)
        string AsString()
        float ValueOf(int)

cdef extern from "src/sofia-ml-methods.h" namespace "sofia_ml":
    cdef enum LearnerType:
        PEGASOS, MARGIN_PERCEPTRON, PASSIVE_AGGRESSIVE, LOGREG_PEGASOS,
        LOGREG, LMS_REGRESSION, SGD_SVM, ROMMA

    cdef enum EtaType:
        BASIC_ETA
        PEGASOS_ETA
        CONSTANT

    void StochasticOuterLoop(SfDataSet, LearnerType, EtaType,
                           float, float, int, SfWeightVector*)

    void BalancedStochasticOuterLoop(SfDataSet, LearnerType, EtaType,
                           float, float, int, SfWeightVector*)

    void StochasticRocLoop(SfDataSet, LearnerType, EtaType,
                           float, float, int, SfWeightVector*)

    void BalancedStochasticOuterLoop(SfDataSet, LearnerType, EtaType,
                                     float, float, int, SfWeightVector*)

    void StochasticRankLoop(SfDataSet, LearnerType, EtaType,
          float, float, int, SfWeightVector*)

    void StochasticClassificationAndRankLoop(SfDataSet, LearnerType, EtaType,
        float, float, float, int, SfWeightVector*)

    void SvmPredictionsOnTestSet(SfDataSet, SfWeightVector, vector[float]*)

    void LogisticPredictionsOnTestSet(SfDataSet, SfWeightVector, vector[float]*)

def train(train_data, int n_features, float alpha, int max_iter, bool fit_intercept,
          learner, loop, eta, float step_probability):

    cdef SfDataSet *data = new SfDataSet(train_data, BUFFER_MB, fit_intercept)
    cdef SfWeightVector *w = new SfWeightVector(n_features)
    cdef float c = 0.0
    cdef int i

    if loop == 'rank':
        StochasticRankLoop(deref(data), learner, eta, alpha, c, max_iter, w)
    elif loop == 'roc':
        StochasticRocLoop(deref(data), learner, eta, alpha, c, max_iter, w)
    elif loop == 'combined-ranking':
        StochasticClassificationAndRankLoop(deref(data), learner, eta, alpha, c,
            step_probability, max_iter, w)
    elif loop == 'stochastic':
        StochasticOuterLoop(deref(data), learner, eta, alpha, c, max_iter, w)
    elif loop == 'balanced-stochastic':
        BalancedStochasticOuterLoop(deref(data), learner, eta, alpha, c, max_iter, w)
    else:
        raise NotImplementedError

    cdef np.ndarray[ndim=1, dtype=np.float64_t] coef = np.empty(n_features)
    for i in range(n_features):
        coef[i] = w.ValueOf(i)

    del data, w

    return coef

def train_fast(np.ndarray[np.float64_t, ndim=2] train_data,
               np.ndarray[np.float64_t, ndim=1] train_label,
               int n_samples, int n_features, float alpha, int max_iter,
               bool fit_intercept, learner, loop, eta, float step_probability):

    cdef SfDataSet *data = new SfDataSet(&train_data[0,0], &train_label[0],
                                    n_samples, n_features, fit_intercept)
    cdef SfWeightVector *w = new SfWeightVector(n_features)
    cdef float c = 0.0
    cdef int i

    if loop == 'rank':
        StochasticRankLoop(deref(data), learner, eta, alpha, c, max_iter, w)
    elif loop == 'roc':
        StochasticRocLoop(deref(data), learner, eta, alpha, c, max_iter, w)
    elif loop == 'combined-ranking':
        StochasticClassificationAndRankLoop(deref(data), learner, eta, alpha, c,
            step_probability, max_iter, w)
    elif loop == 'stochastic':
        StochasticOuterLoop(deref(data), learner, eta, alpha, c, max_iter, w)
    elif loop == 'balanced-stochastic':
        BalancedStochasticOuterLoop(deref(data), learner, eta, alpha, c, max_iter, w)
    else:
        raise NotImplementedError

    cdef np.ndarray[ndim=1, dtype=np.float64_t] coef = np.empty(n_features)
    for i in range(n_features):
        coef[i] = w.ValueOf(i)

    del data, w

    return coef

def predict(test_data, string coef, predict_type, bool fit_intercept):
    cdef SfDataSet *test_dataset = new SfDataSet(test_data, BUFFER_MB, fit_intercept)
    cdef SfWeightVector *w = new SfWeightVector(coef)
    cdef vector[float] *predictions = new vector[float]()

    if predict_type == 'linear':
        SvmPredictionsOnTestSet(deref(test_dataset), deref(w), predictions)
    elif predict_type == 'logistic':
        LogisticPredictionsOnTestSet(deref(test_dataset), deref(w), predictions)
    else:
        raise NotImplementedError

    cdef np.ndarray[ndim=1, dtype=np.float64_t] out = np.empty(predictions.size())
    for i in range(predictions.size()):
        out[i] = predictions.at(i)

    del test_dataset, w, predictions

    return out

def predict_fast(np.ndarray[np.float64_t, ndim=2] test_data,
                 np.ndarray[np.float64_t, ndim=1] test_label,
                 int n_samples, int n_features, string coef,
                 predict_type, bool fit_intercept):
    cdef SfDataSet *test_dataset = new SfDataSet(&test_data[0,0], &test_label[0],
                                            n_samples, n_features, fit_intercept)
    cdef SfWeightVector *w = new SfWeightVector(coef)
    cdef vector[float] *predictions = new vector[float]()

    if predict_type == 'linear':
        SvmPredictionsOnTestSet(deref(test_dataset), deref(w), predictions)
    elif predict_type == 'logistic':
        LogisticPredictionsOnTestSet(deref(test_dataset), deref(w), predictions)
    else:
        raise NotImplementedError

    cdef np.ndarray[ndim=1, dtype=np.float64_t] out = np.empty(predictions.size())
    for i in range(predictions.size()):
        out[i] = predictions.at(i)

    del test_dataset, w, predictions

    return out

def update_fast(np.ndarray[np.float64_t, ndim=2] train_data,
                np.ndarray[np.float64_t, ndim=1] train_label,
                np.ndarray[np.float64_t, ndim=1] prev_coef,
                int n_samples, int n_features, float alpha, int max_iter,
                bool fit_intercept, learner, loop, eta, float step_probability):

    cdef SfDataSet *data = new SfDataSet(&train_data[0,0], &train_label[0],
                                    n_samples, n_features, fit_intercept)
    cdef SfWeightVector *w = new SfWeightVector(&prev_coef[0], n_features)

    cdef float c = 0.0
    cdef int i

    if loop == 'rank':
        StochasticRankLoop(deref(data), learner, eta, alpha, c, max_iter, w)
    elif loop == 'roc':
        StochasticRocLoop(deref(data), learner, eta, alpha, c, max_iter, w)
    elif loop == 'combined-ranking':
        StochasticClassificationAndRankLoop(deref(data), learner, eta, alpha, c,
            step_probability, max_iter, w)
    elif loop == 'stochastic':
        StochasticOuterLoop(deref(data), learner, eta, alpha, c, max_iter, w)
    elif loop == 'balanced-stochastic':
        BalancedStochasticOuterLoop(deref(data), learner, eta, alpha, c, max_iter, w)
    else:
        raise NotImplementedError

    cdef np.ndarray[ndim=1, dtype=np.float64_t] coef = np.empty(n_features)
    for i in range(n_features):
        coef[i] = w.ValueOf(i)

    del data, w

    return coef
