import numpy as np
import matplotlib.pyplot as plt
from functions import (
    map_feature, cost_function_reg, predict
)

plt.ion()

data = np.loadtxt('ex2data2.txt', delimiter=',')
X = data[:, 0:2]
y = data[:, 2].reshape(-1, 1)

pos = np.where(y == 1)[0]
neg = np.where(y == 0)[0]
plt.scatter(X[pos, 0], X[pos, 1], color='black', marker='+', label='y=1')
plt.scatter(
    X[neg, 0], X[neg, 1], color='yellow', marker='o', edgecolors='black',
    label='y=0')
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')
plt.xlim([-1, 1.5])
plt.ylim([-0.8, 1.2])
plt.legend(scatterpoints=1)
plt.show()

input('Program paused. Press enter to continue.\n')
plt.close()

# ########## Part1: Regularized Logistic Regression ##########
X = map_feature(X[:, 0].reshape(-1, 1), X[:, 1].reshape(-1, 1))
initial_theta = np.zeros((X.shape[1], 1))

lam = 1
cost, grad = cost_function_reg(initial_theta, X, y, lam)

print('Cost at initial theta (zeros): {0:.3f}'.format(cost))
print('Expected cost (approx): 0.693\n')

print('Gradient at initial theta (zeros) - first five values only:')
for i in range(5):
    print(' {0:.4f}'.format(grad[i, 0]))
print('Expected gradients (approx) - first five values only:')
print(' 0.0085\n 0.0188\n 0.0001\n 0.0503\n 0.0115\n')

input('Program paused. Press enter to continue.\n')

test_theta = np.ones((X.shape[1], 1))
cost, grad = cost_function_reg(test_theta, X, y, lam)

print('Cost at test theta: {0:.2f}'.format(cost))
print('Expected cost (approx): 2.13\n')

print('Gradient at test theta - first five values only:')
for i in range(5):
    print(' {0:.4f}'.format(grad[i, 0]))
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

pos = np.where(y == 1)[0]
neg = np.where(y == 0)[0]
plt.scatter(X[pos, 1], X[pos, 2], color='black', marker='+', label='y=1')
plt.scatter(
    X[neg, 1], X[neg, 2], color='yellow', marker='o', edgecolors='black',
    label='y=0')

x1 = np.linspace(-1, 1.5, 50).reshape(-1, 1)
x2 = np.linspace(-1, 1.5, 50).reshape(-1, 1)
Z = np.zeros((x1.size, x2.size))

for i in range(x1.shape[0]):
    for j in range(x2.shape[0]):
        Z[i, j] = np.dot(
            map_feature(x1[i].reshape(-1, 1), x2[j].reshape(-1, 1)),
            theta
        )

x1, x2 = np.meshgrid(x1, x2)
CS = plt.contour(x1, x2, Z.T, levels=[0])

plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')
CS.collections[0].set_label('Decision Boundary')
plt.xlim([-1, 1.5])
plt.ylim([-0.8, 1.2])
plt.legend(scatterpoints=1)
plt.show()

p = predict(theta, X)

print('Train Accuracy: {0:.1f}'.format(np.mean(p == y) * 100))
print('Expected accuracy (with lambda = 1): 83.1 (approx)\n')

input('Program paused. Press enter to continue.\n')
plt.close()
