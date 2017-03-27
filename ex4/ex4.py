import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.optimize import minimize
from functions import (
    display_data, nn_cost_function, sigmoid_gradient, rand_initialize_weights,
    check_nn_gradients, predict
)

plt.ion()

input_layer_size = 400
hidden_layer_size = 25
num_labels = 10

# ########## Part1: Loading and Visualizing Data ##########
print('Loading and Visualizing Data ...\n')

data = sio.loadmat('ex4data1.mat')
X = data['X']
y = data['y']

# replace label 10 to 0
for i in range(y.size):
    if y[i] == 10:
        y[i] = 0

m = X.shape[0]

rand_indices = np.random.permutation(m)
sel = X[rand_indices[:100], :]

display_data(sel)

input('Program paused. Press enter to continue.\n')

# ########## Part2: Loading Parameters ##########
print('Loading Saved Neural Network Parameters ...\n')

data_params = sio.loadmat('ex4weights.mat')
Theta1 = data_params['Theta1']
Theta2 = data_params['Theta2']
nn_params = np.hstack((Theta1.flatten(), Theta2.flatten()))

# ########## Part3: Compute Cost (Feedforward) ##########
print('Feedforward Using Neural Network ...\n')

lam = 0

J, _ = nn_cost_function(
    nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lam
)

print(
    'Cost at parameters (loaded from ex4weights): {0:.6f}'.format(J),
    ' \n(this value should be about 0.287629)\n'
)

input('Program paused. Press enter to continue.\n')

# ########## Part4: Implement Regularization ##########
print('Checking Cost Function (w/ Regularization) ... \n')

lam = 1

J, _ = nn_cost_function(
    nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lam
)

print(
    'Cost at parameters (loaded from ex4weights): {0:.6f}'.format(J),
    ' \n(this value should be about 0.383770)\n'
)

input('Program paused. Press enter to continue.\n')

# ########## Part5: Sigmoid Gradient ##########
print('Evaluating sigmoid gradient...\n')

g = sigmoid_gradient(np.array([-1, -0.5, 0, 0.5, 1]))
print('Sigmoid gradient evaluated at [-1 -0.5 0 0.5 1]:')
print(g)

input('Program paused. Press enter to continue.\n')

# ########## Part6: Initializing Parameters ##########
print('Initializing Neural Network Parameters ...\n')

initial_theta1 = rand_initialize_weights(input_layer_size, hidden_layer_size)
initial_theta2 = rand_initialize_weights(hidden_layer_size, num_labels)

initial_nn_params = np.hstack((initial_theta1.flatten(),
                               initial_theta2.flatten()))

# ########## Part7: Implement Backpropagation ##########
print('Checking Backpropagation... \n')

check_nn_gradients()

input('Program paused. Press enter to continue.\n')

lam = 3
check_nn_gradients(lam)

debug_J, _ = nn_cost_function(
    nn_params, input_layer_size, hidden_layer_size,
    num_labels, X, y, lam
)

print(
    'Cost at (fixed) debugging parameters (w/ lambda = {0}): {1:.6f} '
    .format(lam, debug_J),
    '\n(for lambda = 3, this value should be about 0.576051)\n'
)

input('Program paused. Press enter to continue.\n')

# ########## Part8: Training NN ##########
print('Training Neural Network... \n')

lam = 1
result = minimize(
    nn_cost_function, initial_nn_params,
    args=(input_layer_size, hidden_layer_size, num_labels, X, y, lam),
    jac=True, method='TNC', options={'maxiter': 100}
)

Theta1 = result.x[0:hidden_layer_size * (input_layer_size + 1)]\
            .reshape(hidden_layer_size, input_layer_size + 1)
Theta2 = result.x[Theta1.size:]\
            .reshape(num_labels, hidden_layer_size + 1)

input('Program paused. Press enter to continue.\n')

# ########## Part9: Visualize Weights ##########
print('Visualizing Neural Network... \n')

display_data(Theta1[:, 1:])

input('Program paused. Press enter to continue.\n')

# ########## Part10: Implement Predict ##########
pred = predict(Theta1, Theta2, X)

print('Train Accuracy: {0:.1f}'.format(np.mean(pred == y) * 100))
