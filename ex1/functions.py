import numpy as np
import matplotlib.pyplot as plt


def plot_data(x, y):
    plt.plot(x, y, 'rx', markersize=10, label='Training data')
    plt.axis([4, 24, -5, 25])
    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')
    plt.legend()
    plt.show()


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
