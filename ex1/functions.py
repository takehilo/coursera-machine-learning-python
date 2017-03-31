import numpy as np


def compute_cost(X, y, theta):
    m = y.size
    h = np.dot(X, theta)
    return 1 / (2 * m) * ((h - y) ** 2).sum()


def gradient_descent(X, y, theta, alpha, iterations):
    m = y.size

    for i in range(iterations):
        h = np.dot(X, theta)
        error = h - y
        theta -= (alpha / m) * np.dot(X.T, error)

    return theta


def feature_normalize(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    normX = (X - mu) / sigma
    return (normX, mu, sigma)


def gradient_descent_multi(X, y, theta, alpha, num_iters):
    m = y.size
    j_history = np.zeros(num_iters + 1)

    j0 = compute_cost(X, y, theta)

    for i in range(num_iters):
        h = np.dot(X, theta)
        error = h - y
        theta -= (alpha / m) * np.dot(X.T, error)
        j_history[i] = compute_cost(X, y, theta)

    np.insert(j_history, 0, j0)

    return (theta, j_history)


def normal_eqn(X, y):
    return np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)
