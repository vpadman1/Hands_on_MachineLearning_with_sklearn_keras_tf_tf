
# Importing libraries
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------
#actual data
x = 2 * np.random.rand(100,1)
y = 4 + 3 * x + np.random.randn(100,1)
plt.scatter(x,y)
plt.show()

# ----------------------------------------------------------------
#Lin reg
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x,y)
print(lin_reg.coef_, lin_reg.intercept_)

# ----------------------------------------------------------------
# trying poly
from sklearn.preprocessing import PolynomialFeatures

poly_features = PolynomialFeatures(degree= 2, include_bias= True)
x_poly =  poly_features.fit_transform(x)
lin_reg.fit(x_poly,y)
print(lin_reg.coef_, lin_reg.intercept_)