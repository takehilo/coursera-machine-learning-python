import numpy as np
import matplotlib.pyplot as plt


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


def lr_cost_function(theta, X, y, lam):
    m = X.shape[0]
    h = sigmoid(np.dot(X, theta))
    tmp_theta = np.copy(theta)
    tmp_theta[0, 0] = 0

    cost = (1 / m) * np.sum(-y * np.log(h) - (1 - y) * np.log(1 - h)) +\
           (lam / (2 * m)) * np.sum(tmp_theta ** 2)

    grad = (1 / m) * np.dot(X.T, h - y) + (lam / m) * tmp_theta

    return (cost, grad)


def one_vs_all(X_orig, y, num_labels, lam):
    all_theta = np.zeros((num_labels, X_orig.shape[1] + 1))
    X = np.hstack((np.ones((X_orig.shape[0], 1)), X_orig))

    alpha = 0.99
    num_iters = 1000
    costs = np.zeros((num_labels, num_iters))

    # takes about 30 seconds
    for c in range(num_labels):
        for i in range(num_iters):
            costs[c, i], grad = lr_cost_function(
                all_theta[c, :].reshape(-1, 1), X, (y == c) * 1, lam
            )
            all_theta[c, :] -= alpha * grad.reshape(-1)

    return (costs, all_theta)


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
