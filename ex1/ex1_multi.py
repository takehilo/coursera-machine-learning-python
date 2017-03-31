import numpy as np
from matplotlib import pyplot as plt
from functions import (
    feature_normalize, gradient_descent_multi, normal_eqn
)

plt.ion()

# ########## Part1: Feature Normalization ##########
print('Loading data ...\n')

data = np.loadtxt('ex1data2.txt', delimiter=',')
X = data[:, 0:2]
y = data[:, 2].reshape((-1, 1))
m = y.size

print('First 10 examples from the dataset:')
for i in range(10):
    print(
        'x = [{0:.0f}, {1:.0f}], y = {2:.0f}'
        .format(X[i, 0], X[i, 1], y[i, 0]))

input('\nProgram paused. Press enter to continue.\n')

print('Normalizing Features ...\n')

X, mu, sigma = feature_normalize(X)
X = np.hstack((np.ones((m, 1)), X))

# ########## Part2: Gradient Descent ##########
print('Running gradient descent ...\n')

alpha = 0.3
num_iters = 1000

theta = np.zeros((3, 1))
theta, j_history = gradient_descent_multi(X, y, theta, alpha, num_iters)

plt.plot(np.arange(num_iters + 1), j_history)
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
plt.xlim([0, 50])
plt.show()

print('Theta computed from gradient descent:')
print(theta, '\n')

X = np.array([1650, 3])
X = (X - mu) / sigma
X = np.insert(X, 0, 1).reshape((3, 1))

price = np.dot(theta.T, X)[0, 0]

print(
    'Predicted price of a 1650 sq-ft, 3 br house '
    '(using gradient descent): ${0:.0f}\n'
    .format(price)
)

input('Program paused. Press enter to continue.\n')
plt.close()

# ########## Part3: Normal Equations ##########
print('Solving with normal equations...\n')

data = np.loadtxt('ex1data2.txt', delimiter=',')
X = data[:, 0:2]
y = data[:, 2].reshape((-1, 1))
m = y.size
X = np.hstack((np.ones((m, 1)), X))

theta = normal_eqn(X, y)

print('Theta computed from the normal equations')
print(theta, '\n')

X = np.array([1, 1650, 3]).reshape(3, 1)
price = np.dot(theta.T, X)[0, 0]

print(
    'Predicted price of a 1650 sq-ft, 3 br house '
    '(using normal equations): ${0:.0f}\n'
    .format(price)
)

plt.close()
