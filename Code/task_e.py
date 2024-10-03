import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample
from sklearn.model_selection import train_test_split

# Franke function definition
def FrankeFunction(x, y):
    term1 = 0.75 * np.exp(
        -(0.25 * (9 * x - 2) ** 2) - 0.25 * ((9 * y - 2) ** 2)
    )
    term2 = 0.75 * np.exp(
        -((9 * x + 1) ** 2) / 49.0 - 0.1 * (9 * y + 1)
    )
    term3 = 0.5 * np.exp(
        -(9 * x - 7) ** 2 / 4.0 - 0.25 * ((9 * y - 3) ** 2)
    )
    term4 = -0.2 * np.exp(
        -(9 * x - 4) ** 2 - (9 * y - 7) ** 2
    )
    return term1 + term2 + term3 + term4

# Generate data points
np.random.seed(2018)
step_size = 0.02
x_values = np.arange(0, 1, step_size)
y_values = np.arange(0, 1, step_size)
x_grid, y_grid = np.meshgrid(x_values, y_values)
x = x_grid.ravel()
y = y_grid.ravel()
z = FrankeFunction(x, y) +  np.random.normal(0, 0.1, x.shape)  # Adding Gaussian noise

# Combine x and y into a feature matrix
X = np.column_stack((x, y))

# Maximum degree for polynomial regression
maxdegree = 20
n_bootstraps = 100  # Number of bootstraps

# Initialize arrays for error, bias, and variance
error = np.zeros(maxdegree)
bias = np.zeros(maxdegree)
variance = np.zeros(maxdegree)
polydegree = np.arange(maxdegree)

# Split the data into training and testing sets
X_train, X_test, z_train, z_test = train_test_split(
    X, z, test_size=0.3
)

# Loop over polynomial degrees
for degree in range(maxdegree):
    model = make_pipeline(
        PolynomialFeatures(degree=degree), LinearRegression()
    )

    # Create an empty array to store bootstrap predictions
    z_pred = np.empty((z_test.shape[0], n_bootstraps))

    for i in range(n_bootstraps):
        # Resample the training data
        X_, z_ = resample(X_train, z_train)
        z_pred[:, i] = model.fit(X_, z_).predict(X_test).ravel()

    # Reshape z_test for broadcasting
    z_test_reshaped = z_test[:, np.newaxis]  # Shape (n_samples, 1)

    # Calculate bias, variance, and error
    error[degree] = np.mean((z_test_reshaped - z_pred) ** 2)
    mean_z_pred = np.mean(z_pred, axis=1)  # Shape (n_samples,)
    bias[degree] = np.mean((z_test - mean_z_pred) ** 2)
    variance[degree] = np.mean(np.var(z_pred, axis=1))

# Plot the results
plt.plot(polydegree, error, label='Error')
plt.plot(polydegree, bias, label='bias')
plt.plot(polydegree, variance, label='Variance')
plt.legend()
plt.show()
