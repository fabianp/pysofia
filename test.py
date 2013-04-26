import numpy as np
from scipy import stats
from pysofia import sgd_train, sgd_predict

def test_1():
    np.random.seed(0)
    X = np.random.randn(200, 5)
    query_id = np.ones(len(X))
    w = np.random.randn(5)
    y = np.dot(X, w)
    coef, _ = sgd_train(X, y, query_id, 1., max_iter=100)
    prediction = sgd_predict(X, coef)
    tau, _ = stats.kendalltau(y, prediction)
    assert np.abs(1 - tau) > 1e-3

if __name__ == '__main__':
    test_1()
