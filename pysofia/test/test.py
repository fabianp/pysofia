import numpy as np
from pysofia import sofia_ml

n_samples, n_features = 500, 2000
X = np.random.randn(n_samples, n_features)
w = np.random.randn(n_features)
y = (X.dot(w) + np.random.randn(n_samples)).astype(np.int)

coef = sofia_ml.sgd_train(X, y, None, 0.01, n_features,
                          sofia_ml.learner_type.pegasos, sofia_ml.loop_type.rank)

