import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

def FrankeFunction(x, y):
    term1 = 0.75 * np.exp(-(0.25*(9*x - 2)**2) - 0.25*((9*y - 2)**2))
    term2 = 0.75 * np.exp(-((9*x + 1)**2) / 49.0 - 0.1*(9*y + 1))
    term3 = 0.5 * np.exp(-(9*x - 7)**2 / 4.0 - 0.25*((9*y - 3)**2))
    term4 = -0.2 * np.exp(-(9*x - 4)**2 - (9*y - 7)**2)
    return term1 + term2 + term3 + term4

np.random.seed(42)
n = 1000
x = np.random.rand(n)
y = np.random.rand(n)

z = FrankeFunction(x, y) + 0.1 * np.random.randn(n)  # Adding noise

def create_polynomial_design_matrix(x, y, degree):
    X = np.ones(len(x))  # Start with a column of ones for the intercept term
    for i in range(1, degree + 1):
        for j in range(i + 1):
            X = np.column_stack((X, (x**(i-j)) * (y**j)))
    return X

# Polynomial degree
degree = 5
X = create_polynomial_design_matrix(x, y, degree)

X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def RidgeRegression(X_train, z_train, X_test, lmbda):
    I = np.eye(X_train.shape[1])  # Identity matrix
    beta_ridge = np.linalg.pinv(X_train.T @ X_train + lmbda * I) @ X_train.T @ z_train  # Ridge formula
    z_pred = X_test @ beta_ridge
    return z_pred, beta_ridge

lambdas = [0.01, 0.1, 1, 10, 100]

mse_train_ridge = []
mse_test_ridge = []
r2_train_ridge = []
r2_test_ridge = []

for lmbda in lambdas:
    z_train_pred_ridge, beta_ridge = RidgeRegression(X_train_scaled, z_train, X_train_scaled, lmbda)
    z_test_pred_ridge, _ = RidgeRegression(X_train_scaled, z_train, X_test_scaled, lmbda)

    mse_train_ridge.append(mean_squared_error(z_train, z_train_pred_ridge))
    mse_test_ridge.append(mean_squared_error(z_test, z_test_pred_ridge))

    r2_train_ridge.append(r2_score(z_train, z_train_pred_ridge))
    r2_test_ridge.append(r2_score(z_test, z_test_pred_ridge))

    print(f"Lambda = {lmbda}:")
    print(f"  Training MSE: {mse_train_ridge[-1]}")
    print(f"  Test MSE: {mse_test_ridge[-1]}")
    print(f"  Training R²: {r2_train_ridge[-1]}")
    print(f"  Test R²: {r2_test_ridge[-1]}")

plt.figure(figsize=(12,6))

# MSE
plt.subplot(1, 2, 1)
plt.plot(lambdas, mse_train_ridge, label="Train MSE", marker='o')
plt.plot(lambdas, mse_test_ridge, label="Test MSE", marker='o')
plt.xscale('log')
plt.xlabel('Lambda (log scale)')
plt.ylabel('Mean Squared Error')
plt.title('MSE vs Lambda')
plt.grid()
plt.legend()

# R²
plt.subplot(1, 2, 2)
plt.plot(lambdas, r2_train_ridge, label="Train R²", marker='o')
plt.plot(lambdas, r2_test_ridge, label="Test R²", marker='o')
plt.xscale('log')
plt.xlabel('Lambda')
plt.ylabel('R² Score')
plt.grid()
plt.title('R² vs Lambda')
plt.legend()

plt.tight_layout()
plt.show()
