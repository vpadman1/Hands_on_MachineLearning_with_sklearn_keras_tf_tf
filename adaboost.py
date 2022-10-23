from random import Random
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

iris = datasets.load_iris()
X = iris['data']
y = iris['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

ada_clf = AdaBoostClassifier(
    RandomForestClassifier(max_depth = 16), n_estimators = 100, algorithm = "SAMME.R", learning_rate = 0.01
)
ada_clf.fit(X_train, y_train)

y_pred = ada_clf.predict(X_test)

print(classification_report(y_test, y_pred))



# def plot_learning_curves(model, x, y):
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
#     train_errors, val_errors = [],[]
#     for m in range(1, len(X_train)):
#         model.fit(X_train[:m], y_train[:m])
#         y_train_predict =  model.predict(X_train[:m])
#         y_test_predict  = model.predict(X_test[:m])
#         train_errors.append(accuracy_score(y_train[:m], y_train_predict))
#         val_errors.append(accuracy_score(y_test[:m], y_test_predict))
#         plt.plot(train_errors, "r-+", linewidth=2, label = "train")
#         plt.plot(val_errors, "b-", linewidth=3, label = "val")
#     plt.show()


# plot_learning_curves(ada_clf, X,y)