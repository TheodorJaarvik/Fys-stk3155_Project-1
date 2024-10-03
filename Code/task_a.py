import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


# Franke function definition
def FrankeFunction(x, y):
    term1 = 0.75 * np.exp(-(0.25*(9*x - 2)**2) - 0.25*((9*y - 2)**2))
    term2 = 0.75 * np.exp(-((9*x + 1)**2) / 49.0 - 0.1*(9*y + 1))
    term3 = 0.5 * np.exp(-(9*x - 7)**2 / 4.0 - 0.25*((9*y - 3)**2))
    term4 = -0.2 * np.exp(-(9*x - 4)**2 - (9*y - 7)**2)
    return term1 + term2 + term3 + term4


# Define the fixed step size and generate grid points
step_size = 0.02  # Adjust this value for different step sizes
x_values = np.arange(0, 1, step_size)
y_values = np.arange(0, 1, step_size)
x_grid, y_grid = np.meshgrid(x_values, y_values)
x = x_grid.ravel()
y = y_grid.ravel()
n = len(x)

# z-values for the franke function
z = FrankeFunction(x, y) + 0.1 * np.random.normal(0, 1, x.shape)  # Adding Gaussian noise

# maximum degree
maxdegree = 6

# # list for mse and R2 scores
TestMSE = np.zeros(maxdegree)
TrainMSE = np.zeros(maxdegree)
TestR2 = np.zeros(maxdegree)
TrainR2 = np.zeros(maxdegree)
polyDegree = np.zeros(maxdegree)
betaValues = []


# function for design matrix
def create_design_matrix(x, y, degree):
        X = np.ones((len(x), 1)) # Column of ones for the intercept  
        for i in range(1, degree + 1):
            for j in range(i + 1):
                X = np.column_stack((X, (x**(i-j)) * (y**j)))
        return X

# creating a model for each polynomial degree
for i in range(maxdegree):

    # create a design matrix for the given polynomial degree
    X = create_design_matrix(x, y, i)

    # Split the data with scikit-learn: X_train & X_test will be our new design matricies
    # We use X_train to generate our beta values 
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.3, random_state=42)

    # scale the data: mean value = 0, SD = 1
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Optimal values for beta calculated with X_train and Z_train
    beta = np.linalg.pinv(X_train_scaled.T @ X_train_scaled) @ X_train_scaled.T @ z_train 
    betaValues.append(beta)
    
    # predictive model trained and tested on training data 
    model_train = X_train_scaled @ beta

    # predictive model trained on training data but tested on testing data
    model_test = X_test_scaled @ beta

    # add the mse values to array
    TrainMSE[i] = mean_squared_error(z_train, model_train)
    TestMSE[i] = mean_squared_error(z_test, model_test)
    TrainR2[i] = r2_score(z_train, model_train)
    TestR2[i] = r2_score(z_test, model_test)
    polyDegree[i] = i


# plotting MSE
fig = plt.figure(figsize=(20,6))
ax = fig.add_subplot(121)
ax.plot(polyDegree, TrainMSE, label='Train MSE')
ax.plot(polyDegree, TestMSE, label='Test MSE')
ax.set_xlabel('Polynomial degree')
ax.set_ylabel('MSE')
plt.legend()

# plotting R2
ax1 = fig.add_subplot(122)
ax1.plot(polyDegree, TrainR2, label='Train R2')
ax1.plot(polyDegree, TestR2, label='Test R2')
ax1.set_xlabel('Polynomial degree')
ax1.set_ylabel('R2')
plt.legend()

# plotting beta values
max_features = max([len(b) for b in betaValues])
beta_array = np.full((maxdegree, max_features), np.nan)

for i, beta in enumerate(betaValues):
    beta_array[i, :len(beta)] = beta

plt.figure(figsize=(10, 6))
for i in range(beta_array.shape[1]):
    plt.plot(polyDegree, beta_array[:, i], label=f'Beta {i}', marker='o')

plt.xlabel('Polynomial Degree')
plt.ylabel('Beta Values')
plt.title('Beta Values as a Function of Polynomial Degree')
plt.legend()
plt.grid(True)

plt.show()
