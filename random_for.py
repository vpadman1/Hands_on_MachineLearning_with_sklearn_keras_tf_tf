import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

iris = datasets.load_iris()
X = iris['data']
y = iris['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# params_grid= [
#     {"n_estimators":[50,100,200,300,500,700,800,1000],
#     "criterion":["entropy","gini"],
#     "max_leaf_nodes":[5,10,12,14,16,18,20]}
# ]

rnd_clf = RandomForestClassifier()

# grid_search = GridSearchCV( rnd_clf, params_grid, cv = 5, scoring = 'accuracy')

rnd_clf.fit(X_train,y_train)

y_pred = rnd_clf.predict(X_test)

print(classification_report(y_test, y_pred))

for name, score in zip(iris['feature_names'], rnd_clf.feature_importances_):
    print(name, score)