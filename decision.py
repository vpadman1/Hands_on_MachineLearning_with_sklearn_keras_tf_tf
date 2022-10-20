from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

iris = load_iris()
X = iris.data[:,(2,3)]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

tree_clf = DecisionTreeClassifier(max_depth = 2)
tree_clf.fit(X_train,y_train)

pred1 = tree_clf.predict(X_test)

print(classification_report(y_test, pred1))

print(tree_clf.predict([[5, 1.5]]))

# ----------------------------------------------------------------


tree_clf = DecisionTreeClassifier(max_depth = 2, criterion ="entropy")
tree_clf.fit(X_train,y_train)

pred2 = tree_clf.predict(X_test)

print(classification_report(y_test, pred2))

print(tree_clf.predict([[5, 1.5]]))

# ----------------------------------------------------------------

from sklearn.model_selection import cross_val_score

scores = cross_val_score(tree_clf, X_train, y_train, scoring= "accuracy", cv = 10)
print(np.mean(scores))

# ----------------------------------------------------------------

from sklearn.model_selection import GridSearchCV

param_grid = [
    {"max_depth":[1,2,3,4,5,6,7,8,9,10]}  
]

trees_clf = DecisionTreeClassifier()

grid_search = GridSearchCV(trees_clf, param_grid, cv = 5, scoring = "accuracy")

grid_search.fit(X_train, y_train)

print(grid_search.best_estimator_)

pred3 = grid_search.predict(X_test)

print(classification_report(y_test, pred3))

cvres = grid_search.cv_results_


for score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(score, params)

# ----------------------------------------------------------------