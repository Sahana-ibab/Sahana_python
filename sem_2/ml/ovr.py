# import numpy as np
#
# from sklearn.multiclass import OneVsRestClassifier
#
# from sklearn.svm import SVC
#
# X = np.array([
#
#     [10, 10],
#
#     [8, 10],
#
#     [-5, 5.5],
#
#     [-5.4, 5.5],
#
#     [-20, -20],
#
#     [-15, -20]
#
# ])
#
# y = np.array([0, 0, 1, 1, 2, 2])
#
# clf = OneVsRestClassifier(SVC()).fit(X, y)
#
# y_pred = clf.predict([[-19, -20], [9, 9], [-5, 5]])
# print(y_pred)

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
# define dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, n_classes=3, random_state=1)
# define model
model = LogisticRegression()
# define the ovr strategy
ovr = OneVsRestClassifier(model)
# fit model
ovr.fit(X, y)
# make predictions
yhat = ovr.predict(X)
print(yhat)
print(accuracy_score(y, yhat))