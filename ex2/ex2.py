import numpy as np
import matplotlib.pyplot as plt
from functions import (
    plot_data, sigmoid, cost_function, plot_decision_boundary, predict
)

plt.ion()

data = np.loadtxt('ex2data1.txt', delimiter=',')
X = data[:, 0:2]
y = data[:, 2].reshape(-1, 1)

# ########## Part1: Plotting ##########
print(
    'Plotting data with + indicating (y = 1) examples '
    'and o indicating (y = 0) examples.\n')

plot_data(X, y)

input('Program paused. Press enter to continue.\n')

# ########## Part2: Compute Cost and Gradient ##########
m, n = X.shape
X = np.hstack((np.ones((m, 1)), X))
initial_theta = np.zeros((n + 1, 1))
cost, grad = cost_function(initial_theta, X, y)

print('Cost at initial theta (zeros): {0:.3f}'.format(cost))
print('Expected cost (approx): 0.693\n')

print('Gradient at initial theta (zeros):')
print(' {0:.4f}'.format(grad[0, 0]))
print(' {0:.4f}'.format(grad[1, 0]))
print(' {0:.4f}'.format(grad[2, 0]))
print('Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628\n')

test_theta = np.array([-24, 0.2, 0.2]).reshape((3, 1))
cost, grad = cost_function(test_theta, X, y)

print('Cost at test theta: {0:.3f}'.format(cost))
print('Expected cost (approx): 0.218\n')

print('Gradient at test theta:')
print(' {0:.3f}'.format(grad[0, 0]))
print(' {0:.3f}'.format(grad[1, 0]))
print(' {0:.3f}'.format(grad[2, 0]))
print('Expected gradients (approx):\n 0.043\n 2.566\n 2.647\n')

input('Program paused. Press enter to continue.\n')

# ########## Part3: Optimizing ##########
alpha = 0.001
iteration = 1000000
theta = initial_theta
costs = np.zeros(iteration)

# takes about 30 seconds
for i in range(iteration):
    costs[i], grad = cost_function(theta, X, y)
    theta -= alpha * grad

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(np.arange(iteration), costs)
plt.show()

print('Cost at theta: {0:.3f}'.format(costs[-1]))
print('Expected cost (approx): 0.203\n')

print('theta:')
print(' {0:.3f}'.format(theta[0, 0]))
print(' {0:.3f}'.format(theta[1, 0]))
print(' {0:.3f}'.format(theta[2, 0]))
print('Expected theta (approx):')
print(' -25.161\n 0.206\n 0.201\n')

plot_decision_boundary(theta, X, y)

input('Program paused. Press enter to continue.\n')

# ########## Part4: Predict and Accuracies ##########
score_x = np.array([1, 45, 85]).reshape((3, 1))
prob = sigmoid(np.dot(theta.T, score_x))[0][0]
print('For a student with scores 45 and 85, we predict an admission '
      'probability of {0:.3f}'.format(prob))
print('Expected value: 0.775 +/- 0.002\n')

p = predict(theta, X)

print('Train Accuracy: {0:.1f}'.format(np.mean(p == y) * 100))
print('Expected accuracy (approx): 89.0\n')

plt.close()
