import numpy as np


# Franke function definition
def FrankeFunction(x, y):
    term1 = 0.75 * np.exp(-(0.25*(9*x - 2)**2) - 0.25*((9*y - 2)**2))
    term2 = 0.75 * np.exp(-((9*x + 1)**2) / 49.0 - 0.1*(9*y + 1))
    term3 = 0.5 * np.exp(-(9*x - 7)**2 / 4.0 - 0.25*((9*y - 3)**2))
    term4 = -0.2 * np.exp(-(9*x - 4)**2 - (9*y - 7)**2)
    return term1 + term2 + term3 + term4

# Generate data
np.random.seed(42)  # For reproducibility
n = 1000  # Number of points
x = np.random.rand(n)  # Random values for x in the range [0, 1]
y = np.random.rand(n)  # Random values for y in the range [0, 1]


z = FrankeFunction(x, y) + 0.1 * np.random.randn(n)  # Adding Gaussian noise


def create_polynomial_design_matrix(x, y, degree):
    X = np.ones(len(x))  # Start with a column of ones for the intercept term
    for i in range(1, degree + 1):
        for j in range(i + 1):
            X = np.column_stack((X, (x**(i-j)) * (y**j)))
    return X

# Create design matrix for polynomial degree 5
degree = 5

X = create_polynomial_design_matrix(x, y, degree)

from sklearn.model_selection import train_test_split

X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.3, random_state=42)


from sklearn.preprocessing import StandardScaler


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

reg = linear_model.Lasso(alpha=0.1)
reg.fit(X_train_scaled, z_train)

regRidge = linear_model.Ridge(alpha=0.1)
regRidge.fit(X_train_scaled, z_train)

regOLS = linear_model.LinearRegression()
regOLS.fit(X_train_scaled, z_train)

z_train_pred = reg.predict(X_train_scaled)
z_train_pred_ridge = regRidge.predict(X_train_scaled)
z_train_pred_ols = regOLS.predict(X_train_scaled)

z_test_pred= reg.predict(X_test_scaled)
z_test_pred_ridge= regRidge.predict(X_test_scaled)
z_test_pred_ols= regOLS.predict(X_test_scaled)

train_MSE = mean_squared_error(z_train, z_train_pred)
train_MSE_ridge = mean_squared_error(z_train, z_train_pred_ridge)
train_MSE_ols = mean_squared_error(z_train, z_train_pred_ols)

test_MSE = mean_squared_error(z_test, z_test_pred)
test_MSE_ridge = mean_squared_error(z_test, z_test_pred_ridge)
test_MSE_ols = mean_squared_error(z_test, z_test_pred_ols)

r2_train = r2_score(z_train, z_train_pred)
r2_train_ridge = r2_score(z_train, z_train_pred_ridge)
r2_train_ols = r2_score(z_train, z_train_pred_ols)

r2_test = r2_score(z_test, z_test_pred)
r2_test_ridge = r2_score(z_test, z_test_pred_ridge)
r2_test_ols = r2_score(z_test, z_test_pred_ols)

print("MSE and R2-score using Lasso regression:")
print("train_MSE:", train_MSE)
print("test_MSE:", test_MSE)
print("r2_train:", r2_train)
print("r2_test:", r2_test)

print("MSE and R2-score using Ridge regression:")
print("train_MSE:", train_MSE_ridge)
print("test_MSE:", test_MSE_ridge)
print("r2_train:", r2_train_ridge)
print("r2_test:", r2_test_ridge)

print("MSE and R2-score using OLS regression:")
print("train_MSE:", train_MSE_ols)
print("test_MSE:", test_MSE_ols)
print("r2_train:", r2_train_ols)
print("r2_test:", r2_test_ols)