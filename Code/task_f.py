import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.metrics import make_scorer, mean_squared_error


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

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

for i in range(5,11):

    kf = KFold(n_splits=i, shuffle=True, random_state=42)
    mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)

    regLasso = linear_model.Lasso(alpha=0.1)
    regRidge = linear_model.Ridge(alpha=0.1)
    regOLS = linear_model.LinearRegression()

    lassoScore = cross_val_score(regLasso, X_scaled, z, cv=kf, scoring=mse_scorer)
    ridgeScore = cross_val_score(regRidge, X_scaled, z, cv=kf, scoring=mse_scorer)
    olsScore = cross_val_score(regOLS, X_scaled, z, cv=kf, scoring=mse_scorer)

    lassoMSE = -np.mean(lassoScore)
    ridgeMSE = -np.mean(ridgeScore)
    olsMSE = -np.mean(olsScore)

    models = ['Lasso', 'Ridge', 'OLS']
    mse_values = [lassoMSE, ridgeMSE, olsMSE]

    plt.figure(figsize=(8, 5))
    plt.bar(models, mse_values, color=['blue', 'orange', 'green'])
    plt.ylabel('Mean Squared Error')
    plt.title(f'Cross-validated MSE for Lasso, Ridge, and OLS Regression with {i} folds')
    plt.show()

    print(f"{i} folds Lasso MSE: ", lassoMSE)
    print(f"{i} folds Rigde MSE: ", ridgeMSE)
    print(f"{i} folds OLS MSE: ", olsMSE)

    i = i + 1