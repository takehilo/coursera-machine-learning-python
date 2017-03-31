import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def gaussian_kernel(x1, x2, sigma):
    return np.exp(- (np.linalg.norm(x1 - x2) ** 2).sum() / (2 * (sigma ** 2)))


def dataset3_params(X, y, Xval, yval):
    C_vec = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    sigma_vec = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    scores = np.zeros((len(C_vec), len(sigma_vec)))

    for i in range(len(C_vec)):
        for j in range(len(sigma_vec)):
            svm = SVC(kernel='rbf', C=C_vec[i], gamma=sigma_vec[j])
            svm.fit(X, y.ravel())
            scores[i, j] = accuracy_score(yval, svm.predict(Xval))

    max_c_index, max_s_index = np.unravel_index(scores.argmax(), scores.shape)
    return (C_vec[max_c_index], sigma_vec[max_s_index])
