import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cost_function(theta, X, y):
    m = X.shape[0]
    h = sigmoid(np.dot(X, theta))

    cost = (1 / m) * np.sum(-y * np.log(h) - (1 - y) * np.log(1 - h))
    grad = (1 / m) * np.dot(X.T, h - y)

    return (cost, grad)


def predict(theta, X):
    h = sigmoid(np.dot(X, theta))
    return (h >= 0.5) * 1


def plot_data_reg(X, y):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    ax.plot(X[pos, 0], X[pos, 1], marker='+', markersize=7,
            markerfacecolor='k', linestyle='None', label='y=1')
    ax.plot(X[neg, 0], X[neg, 1], marker='o', markersize=7,
            markerfacecolor='y', linestyle='None', label='y=0')
    ax.set_xlabel('Microchip Test 1')
    ax.set_ylabel('Microchip Test 2')
    ax.set_xlim(-1, 1.5)
    ax.set_ylim(-0.8, 1.2)
    ax.legend()
    plt.show()

    return ax


def map_feature(X1, X2):
    degree = 6
    out = np.ones((X1.shape[0], 1))

    for i in range(1, degree + 1):
        for j in range(i + 1):
            out = np.append(out, (X1 ** (i - j)) * (X2 ** j), axis=1)

    return out


def cost_function_reg(theta, X, y, lam):
    m = X.shape[0]
    h = sigmoid(np.dot(X, theta))
    tmp_theta = np.copy(theta)
    tmp_theta[0, 0] = 0

    cost = (1 / m) * np.sum(-y * np.log(h) - (1 - y) * np.log(1 - h)) +\
           (lam / (2 * m)) * np.sum(tmp_theta ** 2)

    grad = (1 / m) * np.dot(X.T, h - y) + (lam / m) * tmp_theta

    return (cost, grad)


def plot_decision_boundary_reg(theta, X, y):
    ax = plot_data_reg(X[:, 1:3], y)

    u = np.linspace(-1, 1.5, 50).reshape(-1, 1)
    v = np.linspace(-1, 1.5, 50).reshape(-1, 1)
    z = np.zeros((u.shape[0], v.shape[0]))

    for i in range(u.shape[0]):
        for j in range(v.shape[0]):
            z[i, j] = np.dot(
                map_feature(u[i].reshape(-1, 1), v[j].reshape(-1, 1)),
                theta
            )

    u, v = np.meshgrid(u, v)
    ax.contour(u, v, z.T, levels=[0])
