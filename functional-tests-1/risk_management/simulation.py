import numpy as np
from .covariance import chol_psd, near_psd


def simulate_normal(cov, n=100000, mean=None, seed=1234, fix_method=near_psd):
    cov = np.asarray(cov, dtype=float)
    try:
        root = chol_psd(cov)
    except np.linalg.LinAlgError:
        root = chol_psd(fix_method(cov))
    mean = np.zeros(len(cov)) if mean is None else np.asarray(mean, dtype=float)
    if mean.shape != (len(cov),):
        raise ValueError("Mean must match covariance dimension")
    return np.random.default_rng(seed).standard_normal((n, len(cov))) @ root.T + mean


def pca_root(cov, explained=1.0):
    if not 0 < explained <= 1:
        raise ValueError("Explained variance must be in (0, 1]")
    values, vectors = np.linalg.eigh(np.asarray(cov, dtype=float))
    values, vectors = values[::-1], vectors[:, ::-1]
    if values[-1] < -1e-8:
        raise ValueError("PCA input must be positive semidefinite")
    keep = np.count_nonzero(values >= 1e-8)
    if keep == 0:
        raise ValueError("No positive principal components")
    if explained < 1:
        keep = min(keep, np.searchsorted(np.cumsum(values) / values.sum(), explained) + 1)
    return vectors[:, :keep] * np.sqrt(values[:keep])


def simulate_pca(cov, n=100000, explained=1.0, mean=None, seed=1234):
    root = pca_root(cov, explained)
    mean = np.zeros(root.shape[0]) if mean is None else np.asarray(mean, dtype=float)
    if mean.shape != (root.shape[0],):
        raise ValueError("Mean must match covariance dimension")
    return np.random.default_rng(seed).standard_normal((n, root.shape[1])) @ root.T + mean
