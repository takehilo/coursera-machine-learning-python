import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from functions import (
    display_data, lr_cost_function, one_vs_all, predict_one_vs_all
)

plt.ion()

input_layer_size = 400
num_labels = 10

# ########## Part1: Loading and Visualizing Data ##########
print('Loading and Visualizing Data ...\n')

data = sio.loadmat('ex3data1.mat')
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

# ########## Part2a: Vectorize Logistic Regression ##########
print('Testing lrCostFunction()')

theta_t = np.array([-2, -1, 1, 2]).reshape(4, 1)
X_t = np.hstack((np.ones((5, 1)), np.arange(1, 16).reshape(3, 5).T / 10))
y_t = np.array([1, 0, 1, 0, 1]).reshape(5, 1)
lambda_t = 3
J, grad = lr_cost_function(theta_t, X_t, y_t, lambda_t)

print('\nCost: {0:.6f}\n'.format(J))
print('Expected cost: 2.534819\n')
print('Gradients:')
print(' {0:.6f}'.format(grad[0, 0]))
print(' {0:.6f}'.format(grad[1, 0]))
print(' {0:.6f}'.format(grad[2, 0]))
print(' {0:.6f}'.format(grad[3, 0]))
print('Expected gradients:')
print(' 0.146561\n -0.548558\n 0.724722\n 1.398003\n')

input('Program paused. Press enter to continue.\n')

# ########## Part2b: One-vs-All Training ##########
print('Training One-vs-All Logistic Regression...\n')
lam = 0.1
costs, all_theta = one_vs_all(X, y, num_labels, lam)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(np.arange(costs.shape[1]), costs[0, :])
plt.show()

input('Program paused. Press enter to continue.\n')

# ########## Part3: Predict for One-vs-All ##########
pred = predict_one_vs_all(all_theta, X)

print('Train Accuracy: {0:.1f}'.format(np.mean(pred == y) * 100))

rp = np.random.permutation(m)

for i in range(m):
    print('Displaying Example Image\n')
    display_data(X[rp[i], :].reshape(1, -1))

    print('\Logistic Regression Prediction: {0}\n'.format(pred[rp[i], 0]))

    s = input('Paused - press enter to continue, q to exit:')
    if s == 'q':
        break

plt.close()