import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def display_data(X, example_width=None):
    m, n = X.shape

    if example_width is None:
        example_width = np.round(np.sqrt(n))

    example_height = n / example_width
    display_rows = np.floor(np.sqrt(m)).astype(np.int64)
    display_cols = np.ceil(m / display_rows).astype(np.int64)
    pad = 1

    display_array = -np.ones((
        pad + display_rows * (example_height + pad),
        pad + display_cols * (example_width + pad)
    ))

    curr_ex = 0
    for j in range(display_rows):
        for i in range(display_cols):
            if curr_ex > m:
                break

            max_val = np.max(np.abs(X[curr_ex, :]))
            start_row = pad + j * (example_height + pad)
            start_col = pad + i * (example_width + pad)
            display_array[
                start_row: start_row + example_height,
                start_col: start_col + example_width
            ] = X[curr_ex, :].reshape(example_height, example_width) / max_val

            curr_ex += 1

    plt.imshow(display_array.T, cmap='gray')
    plt.show()


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def lr_cost_function(theta, X, y, lambda_):
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


def one_vs_all(X_orig, y, num_labels, lambda_):
    all_theta = np.zeros((num_labels, X_orig.shape[1] + 1))
    X = np.hstack((np.ones((X_orig.shape[0], 1)), X_orig))

    for c in range(num_labels):
        result = minimize(
            lambda t: lr_cost_function(
                t, X, np.where(y == c, 1, 0), lambda_),
            all_theta[c, :], jac=True, method='BFGS', options={'maxiter': 100})

        all_theta[c, :] = result.x

    return all_theta


def predict_one_vs_all(all_theta, X_orig):
    X = np.hstack((np.ones((X_orig.shape[0], 1)), X_orig))
    return sigmoid(np.dot(X, all_theta.T)).argmax(axis=1).reshape(-1, 1)


def predict(Theta1, Theta2, X_orig):
    X = np.hstack((np.ones((X_orig.shape[0], 1)), X_orig))

    layer2 = sigmoid(np.dot(X, Theta1.T))
    layer2 = np.hstack((np.ones((X.shape[0], 1)), layer2))
    layer3 = sigmoid(np.dot(layer2, Theta2.T))

    pred = layer3.argmax(axis=1).reshape(-1, 1)

    # use octave compatible indexing
    pred += 1
    return pred
