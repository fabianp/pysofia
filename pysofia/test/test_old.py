import numpy as np
from pysofia import sofia_ml
import menpo.io as mio

n_samples, n_features = 500, 2000
# X = np.random.randn(n_samples, n_features)
# w = np.random.randn(n_features)
# y = (X.dot(w) + np.random.randn(n_samples)).astype(np.int)
#
# mio.export_pickle([X, y], 'tmp.pkl', overwrite=False)

data = mio.import_pickle('tmp.pkl')
X = data[0]
y = data[1]

coef = sofia_ml.sgd_train(X, y, None, 0.01, n_features=n_features)

coef2 = sofia_ml.sgd_train(X, y, None, 0.01, n_features=n_features)

# prediction = sofia_ml.svm_predict(X, coef, sofia_ml.predict_type.logistic)

print "done"

