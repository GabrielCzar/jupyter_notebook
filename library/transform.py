import numpy as np

def standardize(X):
    X_std = np.copy(X)
    n_cols = X.shape[1]
    for i in range(n_cols):
        X_std[:, i] = (X[:, i] - np.mean(X[:, i])) / np.std(X[:, i])
    return X_std


def normalize(X):
    return [(X[i] - min(X)) / (max(X) - min(X)) for i in range(0, X.shape[1])] 