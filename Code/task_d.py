import numpy as np

# Ordinary Least Squares (OLS) model
def ordinary_least_squares(X, y):
    """
    Computes the OLS estimator betâ = (X^T * X)^(-1) * X^T * y
    """
    # OLS Estimate of betâ
    beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
    return beta_hat

# Expected value of y (E(y))
def expected_value_y(X, beta):
    """
    Computes the expected value of y given by E(y) = X * beta
    """
    return X @ beta

# Variance of y
def variance_y(sigma2):
    """
    Returns the variance of y, which is equal to sigma^2
    """
    return sigma2

# Variance of betâ
def variance_beta_hat(X, sigma2):
    """
    Computes the variance of betâ, which is Var(betâ) = sigma^2 * (X^T * X)^(-1)
    """
    return sigma2 * np.linalg.inv(X.T @ X)

# Ridge Regression model
def ridge_regression(X, y, lambda_val):
    """
    Computes the Ridge regression estimator: betâ_ridge = (X^T * X + λI)^(-1) * X^T * y
    """
    identity_matrix = np.eye(X.shape[1])
    beta_ridge = np.linalg.inv(X.T @ X + lambda_val * identity_matrix) @ X.T @ y
    return beta_ridge

# Expectation of Ridge Regression Estimator
def expected_value_ridge(X, beta, lambda_val):
    """
    Computes the expectation of the Ridge regression estimator: E(betâ_ridge)
    """
    identity_matrix = np.eye(X.shape[1])
    return np.linalg.inv(X.T @ X + lambda_val * identity_matrix) @ X.T @ X @ beta

# Variance of Ridge Regression Estimator
def variance_beta_ridge(X, sigma2, lambda_val):
    """
    Computes the variance of the Ridge regression estimator: Var(betâ_ridge)
    """
    identity_matrix = np.eye(X.shape[1])
    return sigma2 * np.linalg.inv(X.T @ X + lambda_val * identity_matrix) @ X.T @ X @ np.linalg.inv(X.T @ X + lambda_val * identity_matrix)

if __name__ == "__main__":
    # Example usage
    X = np.array([[1, 2], [3, 4], [5, 6]])  # Design matrix
    y = np.array([1, 2, 3])  # Observed data
    beta = np.array([0.5, 0.5])  # True parameter vector
    sigma2 = 1  # Variance
    lambda_val = 0.1  # Ridge regularization parameter

    # OLS results
    beta_hat = ordinary_least_squares(X, y)
    expected_y = expected_value_y(X, beta)
    var_y = variance_y(sigma2)
    var_beta_hat = variance_beta_hat(X, sigma2)

    print("OLS Results:")
    print(f"betâ: {beta_hat}")
    print(f"E(y): {expected_y}")
    print(f"Var(y): {var_y}")
    print(f"Var(betâ): {var_beta_hat}")

    # Ridge Regression results
    beta_ridge = ridge_regression(X, y, lambda_val)
    expected_beta_ridge = expected_value_ridge(X, beta, lambda_val)
    var_beta_ridge = variance_beta_ridge(X, sigma2, lambda_val)

    print("\nRidge Regression Results:")
    print(f"betâ_ridge: {beta_ridge}")
    print(f"E(betâ_ridge): {expected_beta_ridge}")
    print(f"Var(betâ_ridge): {var_beta_ridge}")
