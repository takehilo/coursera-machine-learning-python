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


def nn_cost_function(nn_params, input_layer_size, hidden_layer_size,
                     num_labels, X_orig, y, lam):
    Theta1 = nn_params[0:hidden_layer_size * (input_layer_size + 1)]\
                .reshape(hidden_layer_size, input_layer_size + 1)
    Theta2 = nn_params[Theta1.size:]\
                .reshape(num_labels, hidden_layer_size + 1)
    m = X_orig.shape[0]

    X = np.hstack((np.ones((m, 1)), X_orig))

    Y = np.zeros((m, num_labels))
    for i in range(m):
        # make Y compatible with octave indexing
        Y[i, y[i, 0] - 1] = 1

    # ## calculates cost ##
    layer2 = sigmoid(X.dot(Theta1.T))
    layer2 = np.hstack((np.ones((m, 1)), layer2))
    h = sigmoid(layer2.dot(Theta2.T))

    cost = (1 / m) * np.sum(-Y * np.log(h) - (1 - Y) * np.log(1 - h))
    reg = (lam / 2 / m) *\
          (np.sum(Theta1[:, 1:] ** 2) + np.sum(Theta2[:, 1:] ** 2))

    J = cost + reg
    # ## end: calculates cost ##

    # ## calculates grad ##
    delta1 = 0
    delta2 = 0

    for t in range(m):
        a1 = X[t, :].reshape(1, -1)
        z2 = a1.dot(Theta1.T)
        a2 = np.hstack(([[1]], sigmoid(z2)))
        z3 = a2.dot(Theta2.T)
        a3 = sigmoid(z3)

        d3 = a3 - Y[t, :].reshape(1, -1)
        d2 = d3.dot(Theta2) * sigmoid_gradient(np.hstack(([[1]], z2)))

        delta2 += d3.T.dot(a2)
        delta1 += d2[0, 1:].reshape(1, -1).T.dot(a1)

    delta1 = delta1 / m
    delta2 = delta2 / m

    delta1[:, 1:] += (lam / m) * Theta1[:, 1:]
    delta2[:, 1:] += (lam / m) * Theta2[:, 1:]

    grad = np.hstack((delta1.flatten(), delta2.flatten()))
    # ## end: calculates grad ##

    return (J, grad)


def sigmoid_gradient(z):
    g = sigmoid(z)
    return g * (1 - g)


def rand_initialize_weights(l_in, l_out):
    epsilon_init = 0.12
    return np.random.rand(l_out, l_in + 1) * 2 * epsilon_init - epsilon_init


def compute_numerical_gradient(J, theta):
    numgrad = np.zeros(theta.shape)
    perturb = np.zeros(theta.shape)
    e = 1e-4

    for p in range(theta.size):
        perturb[p] = e
        loss1 = J(theta - perturb)[0]
        loss2 = J(theta + perturb)[0]
        numgrad[p] = (loss2 - loss1) / (2 * e)
        perturb[p] = 0

    return numgrad


def check_nn_gradients(lam=0):
    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5

    Theta1 = debug_initialize_weights(hidden_layer_size, input_layer_size)
    Theta2 = debug_initialize_weights(num_labels, hidden_layer_size)
    X = debug_initialize_weights(m, input_layer_size - 1)
    y = (1 + np.mod(np.arange(m), num_labels).T).reshape(-1, 1)

    nn_params = np.hstack((Theta1.flatten(), Theta2.flatten()))

    def cost_func(nn_params):
        return nn_cost_function(
            nn_params, input_layer_size, hidden_layer_size,
            num_labels, X, y, lam
        )

    cost, grad = cost_func(nn_params)
    numgrad = compute_numerical_gradient(cost_func, nn_params)

    for n, g in zip(numgrad, grad):
        print(' {0:.6f} {1:.6f}'.format(n, g))

    print('The above two columns you get should be very similar.\n',
          '(Left-Your Numerical Gradient, Right-Analytical Gradient)\n')

    diff = np.linalg.norm(numgrad - grad) / np.linalg.norm(numgrad + grad)
    print(
        'If your backpropagation implementation is correct, then \n',
        'the relative difference will be small (less than 1e-9). \n',
        '\nRelative Difference: {0}\n'.format(diff)
    )


def debug_initialize_weights(fan_out, fan_in):
    params = np.sin(np.arange(fan_out * (fan_in + 1))) / 10
    return params.reshape(fan_out, fan_in + 1)


def predict(Theta1, Theta2, X_orig):
    X = np.hstack((np.ones((X_orig.shape[0], 1)), X_orig))

    layer2 = sigmoid(np.dot(X, Theta1.T))
    layer2 = np.hstack((np.ones((X.shape[0], 1)), layer2))
    layer3 = sigmoid(np.dot(layer2, Theta2.T))

    pred = layer3.argmax(axis=1).reshape(-1, 1)

    # use octave compatible indexing
    pred += 1
    pred = np.mod(pred, 10)
    return pred
