import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from functions import (
    plot_data,
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

plot_data(X, y)

input('Program paused. Press enter to continue.\n')

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

plt.plot(X[:, 1], np.dot(X, theta), '-', label='Linear regression')
plt.legend()
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

fig1 = plt.figure()
ax1 = fig1.add_subplot(1, 1, 1, projection='3d')
ax1.plot_surface(theta0_vals, theta1_vals, J_vals.T)
ax1.set_xlabel('theta_0')
ax1.set_ylabel('theta_1')
plt.show()

fig2 = plt.figure()
ax2 = fig2.add_subplot(1, 1, 1)
ax2.contour(theta0_vals, theta1_vals, J_vals.T, levels=np.logspace(-2, 3, 30))
ax2.plot(theta[0, 0], theta[1, 0], 'rx', markersize=10, linewidth=2)
ax2.set_xlabel('theta_0')
ax2.set_ylabel('theta_1')
plt.show()

input('Program paused. Press enter to continue.\n')

plt.close()
