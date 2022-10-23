import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression


iris = datasets.load_iris()
X = iris['data']
y = iris['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

bag_clf = BaggingClassifier(
    LogisticRegression(multi_class = "multinomial", penalty = 'l2', solver= "lbfgs"), n_estimators = 500,
    max_samples = 100, bootstrap= True, n_jobs = -1
)
bag_clf.fit(X_train,y_train)
pred = bag_clf.predict(X_test)

from sklearn.metrics import classification_report

print(classification_report(y_test, pred))


bag_clf1 = BaggingClassifier(
    LogisticRegression(multi_class = "multinomial", penalty = 'l2', solver= "lbfgs"), n_estimators = 500,
    max_samples = 100, bootstrap= True, n_jobs = -1, oob_score = True
)
bag_clf1.fit(X_train,y_train)
pred1 = bag_clf1.predict(X_test)

from sklearn.metrics import classification_report

print(classification_report(y_test, pred1))