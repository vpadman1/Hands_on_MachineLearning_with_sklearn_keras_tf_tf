from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

m = 100
x = 6 * np.random.rand(m,1) - 3
y = 0.5 * x**2 + x + 2 + np.random.randn(m, 1)

def plot_learning_curves(model, x, y):
    X_train,X_val,y_train,y_val = train_test_split(x,y, test_size = 0.2)
    train_errors, val_errors = [],[]
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict =  model.predict(X_train[:m])
        y_val_predict  = model.predict(X_val[:m])
        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
        val_errors.append(mean_squared_error(y_val[:m], y_val_predict))
        plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label = "train")
        plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label = "val")
    plt.show()

# ----------------------------------------------------------------

lin_reg = LinearRegression()
plot_learning_curves(lin_reg, x, y)
pred1 = lin_reg.predict([[1.5]])
print(pred1)

# ----------------------------------------------------------------

# Trying more complex model
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

polynomial_regression = Pipeline([
    ("poly_features",
    PolynomialFeatures(degree = 2, include_bias = False)), ("lin_reg", LinearRegression())
])
plot_learning_curves(polynomial_regression, x, y)
pred2 = polynomial_regression.predict([[1.5]])
print(pred2)

# ----------------------------------------------------------------

# Trying ridge model 

from sklearn.linear_model import Ridge
ridge_reg = Ridge(alpha = 1, solver= "cholesky")
ridge_reg.fit(x, y)
plot_learning_curves(ridge_reg, x, y)
pred3 = ridge_reg.predict([[1.5]])
print(pred3)

# ----------------------------------------------------------------

from sklearn.linear_model import SGDRegressor

sgd_reg = SGDRegressor()
sgd_reg.fit(x,y)
plot_learning_curves(sgd_reg, x, y)
pred4 = sgd_reg.predict([[1.5]])
print(pred4)

# ----------------------------------------------------------------

from sklearn.linear_model import Lasso

lasso_reg = Lasso(alpha = 0.1)
lasso_reg.fit(x,y)
plot_learning_curves(lasso_reg, x, y)
pred5 = lasso_reg.predict([[1.5]])
print(pred5)
