import numpy as np
import matplotlib.pyplot as plt
from functions import (
    plot_data_reg, map_feature, cost_function_reg, plot_decision_boundary_reg,
    predict
)

plt.ion()

data = np.loadtxt('ex2data2.txt', delimiter=',')
X = data[:, 0:2]
y = data[:, 2].reshape(-1, 1)

plot_data_reg(X, y)

input('Program paused. Press enter to continue.\n')

# ########## Part1: Regularized Logistic Regression ##########
X = map_feature(X[:, 0].reshape(-1, 1), X[:, 1].reshape(-1, 1))
initial_theta = np.zeros((X.shape[1], 1))

lam = 1
cost, grad = cost_function_reg(initial_theta, X, y, lam)

print('Cost at initial theta (zeros): {0:.3f}'.format(cost))
print('Expected cost (approx): 0.693')
print('Gradient at initial theta (zeros) - first five values only:')
print(' {0:.4f}'.format(grad[0, 0]))
print(' {0:.4f}'.format(grad[1, 0]))
print(' {0:.4f}'.format(grad[2, 0]))
print(' {0:.4f}'.format(grad[3, 0]))
print(' {0:.4f}'.format(grad[4, 0]))
print('Expected gradients (approx) - first five values only:')
print(' 0.0085\n 0.0188\n 0.0001\n 0.0503\n 0.0115\n')

input('Program paused. Press enter to continue.\n')

test_theta = np.ones((X.shape[1], 1))
cost, grad = cost_function_reg(test_theta, X, y, lam)

print('Cost at test theta: {0:.2f}'.format(cost))
print('Expected cost (approx): 2.13')
print('Gradient at test theta - first five values only:')
print(' {0:.4f}'.format(grad[0, 0]))
print(' {0:.4f}'.format(grad[1, 0]))
print(' {0:.4f}'.format(grad[2, 0]))
print(' {0:.4f}'.format(grad[3, 0]))
print(' {0:.4f}'.format(grad[4, 0]))
print('Expected gradients (approx) - first five values only:')
print(' 0.3460\n 0.0851\n 0.1185\n 0.1506\n 0.0159\n')

input('Program paused. Press enter to continue.\n')

# ########## Part2: Regularization and Accuracies ##########
theta = np.zeros((X.shape[1], 1))

lam = 1
alpha = 0.1
num_iters = 1000
costs = np.zeros(num_iters)

for i in range(num_iters):
    costs[i], grad = cost_function_reg(theta, X, y, lam)
    theta -= alpha * grad

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(np.arange(num_iters), costs)
plt.show()

input('Program paused. Press enter to continue.\n')

plot_decision_boundary_reg(theta, X, y)

p = predict(theta, X)

print('Train Accuracy: {0:.1f}'.format(np.mean(p == y) * 100))
print('Expected accuracy (with lambda = 1): 83.1 (approx)')

input('Program paused. Press enter to continue.\n')

plt.close()
