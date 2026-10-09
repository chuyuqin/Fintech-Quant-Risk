import numpy as np


def missing_cov(x, skip_missing=True, correlation_matrix=False):
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError("Expected observations by variables")
    fn = np.corrcoef if correlation_matrix else np.cov
    if skip_missing:
        clean = x[~np.isnan(x).any(axis=1)]
        if len(clean) < 2:
            raise ValueError("At least two complete observations are required")
        return np.atleast_2d(fn(clean, rowvar=False))
    result = np.empty((x.shape[1], x.shape[1]))
    for i in range(x.shape[1]):
        for j in range(i + 1):
            pair = x[:, [i, j]]
            pair = pair[~np.isnan(pair).any(axis=1)]
            if len(pair) < 2:
                raise ValueError("At least two observations per pair are required")
            result[i, j] = result[j, i] = fn(pair, rowvar=False)[0, 1]
    return result


def ew_cov(x, lam=0.97):
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or len(x) == 0 or not 0 < lam <= 1:
        raise ValueError("Need observations and 0 < lambda <= 1")
    # Last row is newest. Normalize finite-history weights; no Bessel correction.
    weights = float(lam) ** np.arange(len(x) - 1, -1, -1)
    weights /= weights.sum()
    centered = x - weights @ x
    return centered.T @ (weights[:, None] * centered)


def correlation(cov):
    cov = np.asarray(cov, dtype=float)
    sd = np.sqrt(np.diag(cov))
    if np.any(sd <= 0):
        raise ValueError("Correlation needs positive variances")
    return cov / np.outer(sd, sd)


def _as_correlation(a):
    a = np.asarray(a, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or not np.allclose(a, a.T):
        raise ValueError("Expected a symmetric square matrix")
    sd = np.sqrt(np.diag(a))
    if np.any(sd <= 0):
        raise ValueError("Matrix repair needs positive diagonal entries")
    return a / np.outer(sd, sd), sd


def near_psd(a, epsilon=0.0):
    corr, sd = _as_correlation(a)
    values, vectors = np.linalg.eigh(corr)
    values = np.maximum(values, epsilon)
    root = vectors * np.sqrt(values)
    root /= np.sqrt(np.sum(root * root, axis=1))[:, None]
    return (root @ root.T) * np.outer(sd, sd)


def higham_psd(a, epsilon=1e-9, max_iter=100, tol=1e-9):
    """Higham/Dykstra alternating projections, identity weights as in the tests."""
    y, sd = _as_correlation(a)
    original = y.copy()
    correction = np.zeros_like(y)
    previous_norm = np.inf
    for _ in range(max_iter):
        r = y - correction
        values, vectors = np.linalg.eigh(r)
        projected = (vectors * np.maximum(values, 0)) @ vectors.T
        correction = projected - r
        y = projected.copy()
        np.fill_diagonal(y, 1.0)
        norm = np.sum((y - original) ** 2)
        if abs(norm - previous_norm) < tol and np.linalg.eigvalsh(y)[0] > -epsilon:
            return y * np.outer(sd, sd)
        previous_norm = norm
    raise RuntimeError("Higham projection did not converge")


def chol_psd(a, tolerance=1e-8):
    a = np.asarray(a, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or not np.allclose(a, a.T):
        raise ValueError("Expected a symmetric square covariance matrix")
    root = np.zeros_like(a)
    for j in range(len(a)):
        pivot = a[j, j] - root[j, :j] @ root[j, :j]
        if pivot < -tolerance:
            raise np.linalg.LinAlgError("Matrix is not positive semidefinite")
        root[j, j] = np.sqrt(max(pivot, 0.0))
        residual = a[j + 1:, j] - root[j + 1:, :j] @ root[j, :j]
        if root[j, j] > 0:
            root[j + 1:, j] = residual / root[j, j]
        elif np.any(np.abs(residual) > tolerance):
            raise np.linalg.LinAlgError("Nonzero residual at zero Cholesky pivot")
    return root
