from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

m = 1000
x = 6 * np.random.rand(m,1) - 3
y = 0.5 * x**2 + x + 2 + np.random.randn(m, 1)

X_train,X_test,Y_train,Y_test = train_test_split(x,y, test_size = 0.2)

from copy import deepcopy

# prepare the data

poly_scaler = Pipeline([
    ("poly_features", PolynomialFeatures(degree = 90, include_bias = False)),
    ("std_scaler", StandardScaler())
])

X_train_poly_scaled = poly_scaler.fit_transform(X_train)
X_test_poly_scaled = poly_scaler.fit_transform(X_test)

sgd_reg = SGDRegressor(max_iter = 1, tol = -np.inf, warm_start = True, penalty = None, learning_rate = "constant", eta0 = 0.0005)
minimum_val_error = float("inf")
best_epoch = None
best_model = None
for epoch in range(1000):
    sgd_reg.fit(X_train_poly_scaled, Y_train)
    y_val_predict = sgd_reg.predict(X_test_poly_scaled)
    val_error = mean_squared_error(Y_test, y_val_predict)
    if val_error < minimum_val_error:
        minimum_val_error = val_error
        best_epoch = epoch
        best_model = deepcopy(sgd_reg)

print(minimum_val_error)
print(best_epoch)