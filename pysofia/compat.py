import numpy as np
from scipy import stats
from sklearn import base, linear_model, cross_validation
from sklearn.externals import joblib
from .sofia_ml import sgd_train

class RankSVM(base.BaseEstimator):
    """
    RankSVM model using stochastic gradient descent.
    TODO: does this fit intercept ?
    """

    def __init__(self, alpha=1., model='rank', max_iter=1000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.model = model

    def fit(self, X, y, query_id=None):
        self.coef_, _ = sgd_train(X, y, query_id, self.alpha, max_iter=self.max_iter,
            model=self.model)
        return self

    def rank(self, X):
        order = np.argsort(X.dot(self.coef_))
        order_inv = np.zeros_like(order)
        order_inv[order] = np.arange(len(order))
        return order_inv

    # just so that GridSearchCV doesn't complain
    predict = rank

    def score(self, X, y):
        tau, _ = stats.kendalltau(X.dot(self.coef_), y)
        return np.abs(tau)

def _inner_fit(X, y, query_id, train, test, alpha):
    # aux method for joblib
    clf = RankSVM(alpha=alpha)
    if query_id is None:
        clf.fit(X[train], y[train])
    else:
        clf.fit(X[train], y[train], query_id[train])
    return clf.score(X[test], y[test])


class RankSVMCV(base.BaseEstimator):
    """
    Cross-validated RankSVM

    the cross-validation generator will be ShuffleSplit
    """
    def __init__(self, alphas=np.logspace(-1, 4, 5), cv=5, 
                    n_jobs=1, model='rank', max_iter=1000):
        self.alphas = alphas
        self.max_iter = max_iter
        self.model = model
        self.cv = cv
        self.n_jobs = n_jobs

    def fit(self, X, y, query_id=None):
        if hasattr(self.cv, '__iter__'):
            cv = self.cv
        else:
            cv = cross_validation.ShuffleSplit(len(y), n_iter=self.cv)
        mean_scores = []
        if query_id is not None:
            query_id = np.array(query_id)
            if not len(query_id) == len(y):
                raise ValueError('query_id of wrong shape')
        for a in self.alphas:
            scores = joblib.Parallel(n_jobs=self.n_jobs)(
                joblib.delayed(_inner_fit)
                (X, y, query_id, train, test, a) for train, test in cv)
            mean_scores.append(np.mean(scores))
        self.best_alpha_ = self.alphas[np.argmax(mean_scores)]
        self.estimator_ = RankSVM(self.best_alpha_)
        self.estimator_.fit(X, y, query_id)
        self.rank = self.estimator_.rank

    def score(self, X, y):
        return self.estimator_.score(X, y)

    def predict(self, X):
        return self.estimator_.predict(X)