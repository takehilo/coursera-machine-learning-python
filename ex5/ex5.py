import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from functions import (
    linear_reg_cost_function, train_linear_reg, learning_curve,
    poly_features, feature_normalize, validation_curve
)

plt.ion()

# ########## Part1: Loading and Visualizing Data ##########
print('Loading and Visualizing Data ...\n')

data = sio.loadmat('ex5data1.mat')
X = data['X']
y = data['y']
X_val = data['Xval']
y_val = data['yval']
X_test = data['Xtest']
y_test = data['ytest']

m = X.shape[0]
m_val = X_val.shape[0]
m_test = X_test.shape[0]

plt.scatter(X, y, color='red', marker='x')
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.xlim(-50, 40)
plt.ylim(0, 40)
plt.show()

input('Program paused. Press enter to continue.\n')
plt.close()

# ########## Part2: Regularized Linear Regression Cost ##########
theta = np.array([1, 1])
j, _ = linear_reg_cost_function(np.hstack((np.ones((m, 1)), X)), y, theta, 1)

print(
    'Cost at theta = [1 ; 1]: {0:.6f} \n'.format(j),
    '(this value should be about 303.993192)')

input('Program paused. Press enter to continue.\n')

# ########## Part3: Regularized Linear Regression Gradient ##########
theta = np.array([1, 1])
j, grad = linear_reg_cost_function(
    np.hstack((np.ones((m, 1)), X)), y, theta, 1)

print(
    'Gradient at theta = [1 ; 1]: [{0:.6f}; {1:.6f}] \n'
    .format(grad[0], grad[1]),
    '(this value should be about [-15.303016; 598.250744])')

input('Program paused. Press enter to continue.\n')

# ########## Part4: Train Linear Regression ##########
lambda_ = 0
theta = train_linear_reg(np.hstack((np.ones((m, 1)), X)), y, lambda_)
y_predicted = np.hstack((np.ones((m, 1)), X)).dot(theta)

plt.scatter(X, y, color='red', marker='x')
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.xlim(-50, 40)
plt.ylim(-5, 40)
plt.plot(X, y_predicted)
plt.show()

input('Program paused. Press enter to continue.\n')
plt.close()

# ########## Part5: Linearing Curve for Linear Regression ##########
lambda_ = 0
error_train, error_val = learning_curve(
    np.hstack((np.ones((m, 1)), X)), y,
    np.hstack((np.ones((m_val, 1)), X_val)), y_val, lambda_)

plt.plot(np.arange(1, m + 1), error_train, label='Train')
plt.plot(np.arange(1, m + 1), error_val, label='Cross Validation')
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.xlim(0, m + 1)
plt.ylim(0, 150)
plt.legend()
plt.show()

print('# Training Examples\tTrain Error\tCross Validation Error')
for i in range(m):
    print('{0}\t{1}\t{2}'.format(i, error_train[i], error_val[i]))

input('Program paused. Press enter to continue.\n')
plt.close()

# ########## Part6: Feature Mapping for Polynomial Regression ##########
p = 8

X_poly = poly_features(X, p)
X_poly, mu, sigma = feature_normalize(X_poly)
X_poly = np.hstack((np.ones((m, 1)), X_poly))

X_poly_val = poly_features(X_val, p)
X_poly_val = (X_poly_val - mu) / sigma
X_poly_val = np.hstack((np.ones((m_val, 1)), X_poly_val))

X_poly_test = poly_features(X_test, p)
X_poly_test = (X_poly_test - mu) / sigma
X_poly_test = np.hstack((np.ones((m_test, 1)), X_poly_test))

print('Normalized Training Example 1:')
print(X_poly[0, :])

input('Program paused. Press enter to continue.\n')

# ########## Part7: Learning Curve for Polynomial Regression ##########
lambda_ = 0
theta = train_linear_reg(X_poly, y, lambda_)

X_plot = np.arange(X.min() - 15, X.max() + 25, 0.05).reshape(-1, 1)
m_plot = X_plot.shape[0]
X_poly_plot = poly_features(X_plot, p)
X_poly_plot = (X_poly_plot - mu) / sigma
X_poly_plot = np.hstack((np.ones((m_plot, 1)), X_poly_plot))

plt.scatter(X, y, color='red', marker='x')
plt.plot(X_plot, X_poly_plot.dot(theta))
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.xlim(-80, 80)
plt.ylim(-60, 40)
plt.title('Polynomial Regression Fit (lambda = {0})'.format(lambda_))
plt.show()

input('Program paused. Press enter to continue.\n')
plt.close()

error_train, error_val = learning_curve(X_poly, y, X_poly_val, y_val, lambda_)

plt.plot(np.arange(1, m + 1), error_train, label='Train')
plt.plot(np.arange(1, m + 1), error_val, label='Cross Validation')
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.xlim(0, m + 1)
plt.ylim(0, 100)
plt.title(
    'Polynomial Regression Learning Curve (lambda = {0})'.format(lambda_))
plt.legend()
plt.show()

print('# Training Examples\tTrain Error\tCross Validation Error')
for i in range(m):
    print('{0}\t{1}\t{2}'.format(i, error_train[i], error_val[i]))

input('Program paused. Press enter to continue.\n')
plt.close()

# ########## Part8: Validation for Selecting Lambda ##########
lambda_vec, error_train, error_val = validation_curve(
    X_poly, y, X_poly_val, y_val)

plt.plot(lambda_vec, error_train, label='Train')
plt.plot(lambda_vec, error_val, label='Cross Validation')
plt.xlabel('lambda')
plt.ylabel('Error')
plt.xlim(0, 10)
plt.ylim(0, 20)
plt.legend()
plt.show()

print('lambda\t\tTrain Error\tValidation Error')
for i in range(len(lambda_vec)):
    print(
        '{0}\t\t{1}\t{2}'.format(lambda_vec[i], error_train[i], error_val[i]))

input('Program paused. Press enter to continue.\n')
plt.close()
