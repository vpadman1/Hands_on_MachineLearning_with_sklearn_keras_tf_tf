from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


iris = datasets.load_iris()
X = iris['data'][:,(2,3)]
y = iris['target']

# log_reg = LogisticRegression()
# log_reg.fit(X,y)

# X_new = np.linspace(0,3, 1000).reshape(-1,1)
# y_proba = log_reg.predict_proba(X_new)
# print(y_proba)
# plt.plot(X_new, y_proba[:,1], "g-", label = "Iris virginica")
# plt.plot(X_new, y_proba[:,0],"b--", label = "Not Iris virginica")
# plt.show()


softmax_reg = LogisticRegression(multi_class = "multinomial", solver = 'lbfgs', C = 10)
softmax_reg.fit(X,y)
pred = softmax_reg.predict([[5,2]])
print(pred)
print(softmax_reg.predict_proba([[5,2]]))