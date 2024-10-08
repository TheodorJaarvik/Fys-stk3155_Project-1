import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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


X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.3, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def OLS(X_train, z_train, X_test):

    beta = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ z_train  # Solve for beta
    z_pred = X_test @ beta
    return z_pred, beta

def RidgeRegression(X_train, z_train, X_test, lmbda):
    I = np.eye(X_train.shape[1])  # Identity matrix
    beta_ridge = np.linalg.pinv(X_train.T @ X_train + lmbda * I) @ X_train.T @ z_train  # Ridge formula
    z_pred = X_test @ beta_ridge
    return z_pred, beta_ridge

reg = linear_model.Lasso(max_iter=1000000,alpha=0.01)
reg.fit(X_train_scaled, z_train)


z_train_pred = reg.predict(X_train_scaled)
z_train_pred_ridge, beta = RidgeRegression(X_train_scaled, z_train, X_train_scaled, 0.1)
z_train_pred_ols, beta = OLS(X_train_scaled, z_train, X_train_scaled)

z_test_pred= reg.predict(X_test_scaled)
z_test_pred_ridge, _= RidgeRegression(X_train_scaled, z_train, X_test_scaled, 0.1)
z_test_pred_ols, _= OLS(X_train_scaled, z_train, X_test_scaled)

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

#Find optimal alpha(lambda) for Lasso regression : This can also be done by cross-validation but im just using the same code as a) and b)

lambdas = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]

mse_train_lasso = []
mse_test_lasso = []
r2_train_lasso = []
r2_test_lasso = []
coefs=[]

for lmd in lambdas:

    reg2 = linear_model.Lasso(max_iter=1000000, alpha=lmd)
    reg2.fit(X_train_scaled, z_train)

    z_train_pred2 = reg2.predict(X_train_scaled)
    z_test_pred2 = reg2.predict(X_test_scaled)

    mse_train_lasso.append(mean_squared_error(z_train, z_train_pred2))
    mse_test_lasso.append(mean_squared_error(z_test, z_test_pred))
    r2_train_lasso.append(r2_score(z_train, z_train_pred2))
    r2_test_lasso.append(r2_score(z_test, z_test_pred2))
    coefs.append(reg2.coef_)

    print(f"Lambda: {lmd}")
    print(f"  Training MSE: {mse_train_lasso[-1]}")
    print(f"  Test MSE: {mse_test_lasso[-1]}")
    print(f"  Training R²: {r2_train_lasso[-1]}")
    print(f"  Test R²: {r2_test_lasso[-1]}")

coefs = np.array(coefs)

# Coefficient shrinking plot
plt.figure(figsize=(10, 6))
plt.plot(lambdas, coefs)
plt.xscale('log')
plt.xlabel('Lambda')
plt.ylabel('Coefficients')
plt.title('Coefficient Shrinking')
plt.axis('tight')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(lambdas, mse_train_lasso)
plt.xscale('log')
plt.xlabel('Lambda')
plt.ylabel('Train-MSE')
plt.title('MSE score')
plt.axis('tight')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(lambdas, r2_train_lasso)
plt.xscale('log')
plt.xlabel('Lambda')
plt.ylabel('Train-R2')
plt.title('R2-score')
plt.axis('tight')
plt.show()