import numpy as np


def sigmoid(z):
    z = np.where(z > 36, 36, z)
    z = np.where(z < -709, -709, z)
    return 1 / (1 + np.exp(-z))


def cost_function(theta, X, y):
    theta = theta.reshape(-1, 1)
    m = X.shape[0]
    h = sigmoid(np.dot(X, theta))

    cost = (1 / m) * np.sum(-y * np.log(h) - (1 - y) * np.log(1 - h))
    grad = (1 / m) * np.dot(X.T, h - y)

    return (cost, grad.ravel())


def predict(theta, X):
    h = sigmoid(np.dot(X, theta))
    return (h >= 0.5) * 1


def map_feature(X1, X2):
    degree = 6
    out = np.ones((X1.shape[0], 1))

    for i in range(1, degree + 1):
        for j in range(i + 1):
            out = np.append(out, (X1 ** (i - j)) * (X2 ** j), axis=1)

    return out


def cost_function_reg(theta, X, y, lambda_):
    theta = theta.reshape(-1, 1)
    m = X.shape[0]
    h = sigmoid(X.dot(theta))

    cost = (1 / m) * np.sum(-y * np.log(h) - (1 - y) * np.log(1 - h))
    reg = (lambda_ / (2 * m)) * (theta[1:, 0] ** 2).sum()
    j = cost + reg

    grad = np.zeros(theta.size)
    grad[0] = (1 / m) * (h - y).sum()
    grad[1:] = (1 / m) * (X[:, 1:].T.dot(h - y).ravel())\
        + (lambda_ / m) * theta[1:, 0]

    return (j, grad)
