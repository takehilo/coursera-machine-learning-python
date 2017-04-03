import numpy as np


def feature_normalize(X):
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    X_norm = (X - mu) / sigma
    return (X_norm, mu, sigma)


def pca(X):
    m = X.shape[0]
    Sigma = (1 / m) * X.T.dot(X)
    U, s, V = np.linalg.svd(Sigma)
    return (U, s, V)
