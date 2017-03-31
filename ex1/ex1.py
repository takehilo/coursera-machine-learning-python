import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from functions import (
    compute_cost,
    gradient_descent,
)

plt.ion()

# ########## Part2: Plotting ##########
print('Plotting Data ...\n')

data = np.loadtxt('ex1data1.txt', delimiter=',')
X = data[:, 0].reshape((-1, 1))
y = data[:, 1].reshape((-1, 1))
m = y.size

plt.scatter(X, y, color='red', marker='x')
plt.xlim([4, 24])
plt.ylim([-5, 25])
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.show()

input('Program paused. Press enter to continue.\n')
plt.close()

# ########## Part3: Gradient descent ##########
print('Running Gradient Descent ...\n')

X = np.hstack((np.ones((m, 1)), X))
theta = np.zeros((2, 1))

iterations = 1500
alpha = 0.01

print(compute_cost(X, y, theta))

theta = gradient_descent(X, y, theta, alpha, iterations)
print('Theta found by gradient descent:')
print(theta[0, 0], theta[1, 0], '\n')

plt.scatter(X[:, 1], y, color='red', marker='x', label='Training data')
plt.xlim([4, 24])
plt.ylim([-5, 25])
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.plot(X[:, 1], np.dot(X, theta), label='Linear regression')
plt.legend(loc='lower right', scatterpoints=1)
plt.show()

predict1 = np.dot(np.array([[1, 3.5]]), theta)[0, 0]
print(
    'For population = 35,000, we predict a profit of {0}'
    .format(predict1 * 10000)
)

predict2 = np.dot(np.array([[1, 7]]), theta)[0, 0]
print(
    'For population = 70,000, we predict a profit of {0}\n'
    .format(predict2 * 10000)
)

input('Program paused. Press enter to continue.\n')
plt.close()

# ########## Part4: Visualizing J(theta_0, theta_1) ##########
print('Visualizing J(theta_0, theta_1) ...\n')

theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)

J_vals = np.zeros((theta0_vals.size, theta1_vals.size))

for i in range(theta0_vals.size):
    for j in range(theta1_vals.size):
        t = np.array([theta0_vals[i], theta1_vals[j]]).reshape((2, 1))
        J_vals[i, j] = compute_cost(X, y, t)

theta0_vals, theta1_vals = np.meshgrid(theta0_vals, theta1_vals)

ax = plt.subplot(1, 1, 1, projection='3d')
ax.plot_surface(theta0_vals, theta1_vals, J_vals.T)
ax.set_xlabel('theta_0')
ax.set_ylabel('theta_1')
plt.show()

input('Program paused. Press enter to continue.\n')
plt.close()

plt.contour(theta0_vals, theta1_vals, J_vals.T, levels=np.logspace(-2, 3, 30))
plt.scatter(theta[0, 0], theta[1, 0], color='red', marker='x')
plt.xlabel('theta_0')
plt.ylabel('theta_1')
plt.show()

input('Program paused. Press enter to continue.\n')
plt.close()
