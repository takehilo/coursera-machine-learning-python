import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.svm import SVC
from functions import (
    gaussian_kernel, dataset3_params
)

plt.ion()

# ########## Part1: Loading and Visualizing Data ##########
print('Loading and Visualizing Data ...\n')

data = sio.loadmat('ex6data1.mat')
X = data['X']  # 51 x 2 matrix
y = data['y']  # 51 x 1 matrix

pos = (y == 1).ravel()
neg = (y == 0).ravel()
plt.scatter(X[pos, 0], X[pos, 1], color='black', marker='+')
plt.scatter(
    X[neg, 0], X[neg, 1], color='yellow', edgecolors='black', marker='o')
plt.xlim(0, 4.5)
plt.ylim(1.5, 5)
plt.show()

input('Program paused. Press enter to continue.\n')
plt.close()

# ########## Part2: Training Linear SVM ##########
print('Training Linear SVM ...\n')

C = 1
svm = SVC(kernel='linear', C=C)
svm.fit(X, y.ravel())
weights = svm.coef_[0]
intercept = svm.intercept_[0]

xp = np.linspace(X.min(), X.max(), 100)
yp = - (weights[0] * xp + intercept) / weights[1]

pos = (y == 1).ravel()
neg = (y == 0).ravel()
plt.scatter(X[pos, 0], X[pos, 1], color='black', marker='+')
plt.scatter(
    X[neg, 0], X[neg, 1], color='yellow', edgecolors='black', marker='o')
plt.plot(xp, yp)
plt.xlim(0, 4.5)
plt.ylim(1.5, 5)
plt.show()

input('Program paused. Press enter to continue.\n')
plt.close()

# ########## Part3: Implementing Gaussian Kernel ##########
print('Evaluating the Gaussian Kernel ...\n')

x1 = np.array([1, 2, 1])
x2 = np.array([0, 4, -1])
sigma = 2
sim = gaussian_kernel(x1, x2, sigma)

print(
    'Gaussian Kernel between x1 = [1; 2; 1], x2 = [0; 4; -1], sigma = {0} :\n'
    .format(sigma),
    '\t{0:.6f}\n(for sigma = 2, this value should be about 0.324652)'
    .format(sim))

input('Program paused. Press enter to continue.\n')
plt.close()

# ########## Part4: Visualizing Dataset 2 ##########
data = sio.loadmat('ex6data2.mat')
X = data['X']  # 863 x 2 matrix
y = data['y']  # 863 x 1 matrix

pos = (y == 1).ravel()
neg = (y == 0).ravel()
plt.scatter(X[pos, 0], X[pos, 1], color='black', marker='+')
plt.scatter(
    X[neg, 0], X[neg, 1], color='yellow', edgecolors='black', marker='o')
plt.xlim(0, 1)
plt.ylim(0.4, 1)
plt.show()

input('Program paused. Press enter to continue.\n')
plt.close()

# ########## Part5: Training SVM with RBF Kernel (Dataset 2) ##########
print('Training SVM with RBF Kernel ...\n')

C = 30
sigma = 30

svm = SVC(kernel='rbf', C=C, gamma=sigma)
svm.fit(X, y.ravel())

x1 = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
x2 = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
x1, x2 = np.meshgrid(x1, x2)
yp = svm.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape)

pos = (y == 1).ravel()
neg = (y == 0).ravel()
plt.scatter(X[pos, 0], X[pos, 1], color='black', marker='+')
plt.scatter(
    X[neg, 0], X[neg, 1], color='yellow', edgecolors='black', marker='o')
plt.xlim(0, 1)
plt.ylim(0.4, 1)
plt.contour(x1, x2, yp)
plt.show()

input('Program paused. Press enter to continue.\n')
plt.close()

# ########## Part6: Visualizing Dataset 3 ##########
data = sio.loadmat('ex6data3.mat')
X = data['X']  # 211 x 2 matrix
y = data['y']  # 211 x 1 matrix
Xval = data['Xval']  # 200 x 2 matrix
yval = data['yval']  # 200 x 1 matrix

pos = (y == 1).ravel()
neg = (y == 0).ravel()
plt.scatter(X[pos, 0], X[pos, 1], color='black', marker='+')
plt.scatter(
    X[neg, 0], X[neg, 1], color='yellow', edgecolors='black', marker='o')
plt.xlim(-0.6, 0.3)
plt.ylim(-0.8, 0.6)
plt.show()

input('Program paused. Press enter to continue.\n')
plt.close()

# ########## Part7: Training SVM with RBF Kernel (Dataset 3) ##########
C, sigma = dataset3_params(X, y, Xval, yval)

svm = SVC(kernel='rbf', C=C, gamma=sigma)
svm.fit(X, y.ravel())

x1 = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
x2 = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
x1, x2 = np.meshgrid(x1, x2)
yp = svm.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape)

pos = (y == 1).ravel()
neg = (y == 0).ravel()
plt.scatter(X[pos, 0], X[pos, 1], color='black', marker='+')
plt.scatter(
    X[neg, 0], X[neg, 1], color='yellow', edgecolors='black', marker='o')
plt.xlim(-0.6, 0.3)
plt.ylim(-0.8, 0.6)
plt.contour(x1, x2, yp)
plt.show()

input('Program paused. Press enter to continue.\n')
plt.close()
