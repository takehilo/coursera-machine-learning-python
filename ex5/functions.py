import numpy as np
from scipy.optimize import minimize


def linear_reg_cost_function(X, y, theta, lambda_):
    theta = theta.reshape(-1, 1)
    m = X.shape[0]
    error = X.dot(theta) - y

    cost = (1 / (2 * m)) * (error ** 2).sum()
    reg = (lambda_ / (2 * m)) * (theta[1:, 0] ** 2).sum()
    j = cost + reg

    grad = np.zeros(theta.size)
    grad[0] = (1 / m) * error.sum()
    grad[1:] = (1 / m) * (X[:, 1:].T.dot(error).ravel())\
        + (lambda_ / m) * theta[1:, 0]

    return (j, grad)


def train_linear_reg(X, y, lambda_):
    initial_theta = np.zeros(X.shape[1])

    result = minimize(
        lambda t: linear_reg_cost_function(X, y, t, lambda_),
        initial_theta, jac=True, method='BFGS', options={'maxiter': 100})

    theta = result.x
    return theta


def learning_curve(X, y, X_val, y_val, lambda_):
    error_train = []
    error_val = []
    m = X.shape[0]

    for i in range(1, m + 1):
        Xtrain = X[:i, :]
        ytrain = y[:i]
        theta = train_linear_reg(Xtrain, ytrain, lambda_)
        error_train.append(
            linear_reg_cost_function(Xtrain, ytrain, theta, lambda_)[0])
        error_val.append(
            linear_reg_cost_function(X_val, y_val, theta, lambda_)[0])

    return (np.array(error_train), np.array(error_val))


def poly_features(X, p):
    X_poly = np.copy(X)

    for i in range(2, p + 1):
        X_poly = np.hstack((X_poly, (X_poly[:, 0] ** i).reshape(-1, 1)))

    return X_poly


def feature_normalize(X):
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)

    return ((X - mu) / sigma, mu, sigma)


def validation_curve(X, y, X_val, y_val):
    lamba_vec = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
    error_train = []
    error_val = []

    for lambda_ in lamba_vec:
        theta = train_linear_reg(X, y, lambda_)
        error_train.append(
            linear_reg_cost_function(X, y, theta, lambda_)[0])
        error_val.append(
            linear_reg_cost_function(X_val, y_val, theta, lambda_)[0])

    return (lamba_vec, error_train, error_val)
