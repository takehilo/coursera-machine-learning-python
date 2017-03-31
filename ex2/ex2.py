import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from functions import (
    sigmoid, cost_function, predict
)

plt.ion()

data = np.loadtxt('ex2data1.txt', delimiter=',')
X = data[:, 0:2]
y = data[:, 2].reshape(-1, 1)

# ########## Part1: Plotting ##########
print(
    'Plotting data with + indicating (y = 1) examples '
    'and o indicating (y = 0) examples.\n')

pos = np.where(y == 1)[0]
neg = np.where(y == 0)[0]
plt.scatter(X[pos, 0], X[pos, 1], color='black', marker='+', label='Admitted')
plt.scatter(
    X[neg, 0], X[neg, 1], color='yellow', marker='o', edgecolors='black',
    label='Not admitted')
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.xlim([30, 100])
plt.ylim([30, 100])
plt.legend(scatterpoints=1)
plt.show()

input('Program paused. Press enter to continue.\n')
plt.close()

# ########## Part2: Compute Cost and Gradient ##########
m, n = X.shape
X = np.hstack((np.ones((m, 1)), X))
initial_theta = np.zeros(n + 1)
cost, grad = cost_function(initial_theta, X, y)

print('Cost at initial theta (zeros): {0:.3f}'.format(cost))
print('Expected cost (approx): 0.693\n')

print('Gradient at initial theta (zeros):')
print(' {0:.4f}'.format(grad[0]))
print(' {0:.4f}'.format(grad[1]))
print(' {0:.4f}'.format(grad[2]))
print('Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628\n')

test_theta = np.array([-24, 0.2, 0.2])
cost, grad = cost_function(test_theta, X, y)

print('Cost at test theta: {0:.3f}'.format(cost))
print('Expected cost (approx): 0.218\n')

print('Gradient at test theta:')
print(' {0:.3f}'.format(grad[0]))
print(' {0:.3f}'.format(grad[1]))
print(' {0:.3f}'.format(grad[2]))
print('Expected gradients (approx):\n 0.043\n 2.566\n 2.647\n')

input('Program paused. Press enter to continue.\n')

# ########## Part3: Optimizing ##########
result = minimize(
    lambda t: cost_function(t, X, y), initial_theta, jac=True,
    method='BFGS', options={'maxiter': 100})

theta = result.x
cost = result.fun

print('Cost at theta: {0:.3f}'.format(cost))
print('Expected cost (approx): 0.203\n')

print('theta:')
print(' {0:.3f}'.format(theta[0]))
print(' {0:.3f}'.format(theta[1]))
print(' {0:.3f}'.format(theta[2]))
print('Expected theta (approx):')
print(' -25.161\n 0.206\n 0.201\n')

pos = np.where(y == 1)[0]
neg = np.where(y == 0)[0]
plt.scatter(X[pos, 1], X[pos, 2], color='black', marker='+', label='Admitted')
plt.scatter(
    X[neg, 1], X[neg, 2], color='yellow', marker='o', edgecolors='black',
    label='Not admitted')

x1 = np.array([X[:, 1].min(), X[:, 1].max()])
x2 = (-1 / theta[2]) * (theta[1] * x1 + theta[0])
plt.plot(x1, x2)

plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.xlim([30, 100])
plt.ylim([30, 100])
plt.legend(scatterpoints=1)
plt.show()

input('Program paused. Press enter to continue.\n')
plt.close()

# ########## Part4: Predict and Accuracies ##########
score_x = np.array([1, 45, 85]).reshape((3, 1))
prob = sigmoid(np.dot(theta.reshape(-1, 1).T, score_x))[0, 0]
print('For a student with scores 45 and 85, we predict an admission '
      'probability of {0:.3f}'.format(prob))
print('Expected value: 0.775 +/- 0.002\n')

p = predict(theta.reshape(-1, 1), X)

print('Train Accuracy: {0:.1f}'.format(np.mean(p == y) * 100))
print('Expected accuracy (approx): 89.0\n')

plt.close()
