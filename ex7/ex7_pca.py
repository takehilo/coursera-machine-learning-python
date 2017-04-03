import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from functions import (
    feature_normalize, pca
)

plt.ion()

# ########## Part1: Load Example Dataset ##########
print('Visualizing example dataset for PCA.\n')

data = sio.loadmat('ex7data1.mat')
X = data['X']

plt.scatter(X[:, 0], X[:, 1], color='black', marker='o')
plt.xlim([0.5, 6.5])
plt.ylim([2, 8])

input('Program paused. Press enter to continue.\n')
plt.close()

# ########## Part2: Principal Component Analysis ##########
print('Running PCA on example dataset.\n')

X_norm, mu, sigma = feature_normalize(X)
U, s, _ = pca(X_norm)

plt.scatter(X[:, 0], X[:, 1], color='black', marker='o')
plt.xlim([0.5, 6.5])
plt.ylim([2, 8])

plt.plot(
    [mu[0], (mu + 1.5 * s[0] * U[:, 0].T)[0]],
    [mu[1], (mu + 1.5 * s[0] * U[:, 0].T)[1]])
plt.plot(
    [mu[0], (mu + 1.5 * s[1] * U[:, 1].T)[0]],
    [mu[1], (mu + 1.5 * s[1] * U[:, 1].T)[1]])

print('Top eigenvector:')
print(' U[: ,0] = {0:.6f} {0:.6f}'.format(U[0, 0], U[1, 0]))
print('(you should expect to see -0.707107 -0.707107)\n')

input('Program paused. Press enter to continue.\n')
plt.close()
