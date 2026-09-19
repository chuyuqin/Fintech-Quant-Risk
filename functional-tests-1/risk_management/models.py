import numpy as np
from scipy import optimize, stats


def fit_normal(x):
    x = np.asarray(x, dtype=float)
    return dict(mu=x.mean(), sigma=x.std(ddof=1))


def _fit_t_residuals(y, design):
    # Standardize the problem for stable optimization, replacing Julia's globals.
    scale = np.std(y, ddof=1)
    if scale <= 0:
        raise ValueError("Need nonconstant observations")
    ys = y / scale
    beta = np.linalg.lstsq(design, ys, rcond=None)[0]
    residual = ys - design @ beta
    p = design.shape[1]

    def objective(theta):
        b = theta[:p]
        s, nu = np.exp(theta[p]), 2.0001 + np.exp(theta[p + 1])
        return -stats.t.logpdf(ys - design @ b, nu, loc=0, scale=s).sum()

    fits = []
    for nu_start in (5.0, 15.0, 50.0):
        s_start = np.std(residual) * np.sqrt((nu_start - 2) / nu_start)
        initial = np.r_[beta, np.log(s_start), np.log(nu_start - 2.0001)]
        fit = optimize.minimize(objective, initial, method="Nelder-Mead",
                                options={"maxiter":20000, "xatol":1e-10, "fatol":1e-10})
        if fit.success and np.isfinite(fit.fun):
            fits.append(fit)
    if not fits:
        raise RuntimeError("Student t likelihood optimization failed")
    best = min(fits, key=lambda f: f.fun)
    return best.x[:p] * scale, np.exp(best.x[p]) * scale, 2.0001 + np.exp(best.x[p + 1])


def fit_t(x):
    x = np.asarray(x, dtype=float).reshape(-1)
    beta, sigma, nu = _fit_t_residuals(x, np.ones((len(x), 1)))
    return dict(mu=beta[0], sigma=sigma, nu=nu)


def fit_regression_t(y, x):
    y, x = np.asarray(y, dtype=float), np.asarray(x, dtype=float)
    # Scaling predictors improves convergence without changing model coefficients.
    scales = np.std(x, axis=0, ddof=1)
    if np.any(scales == 0):
        raise ValueError("Predictors must be nonconstant")
    design = np.column_stack((np.ones(len(y)), x / scales))
    beta, sigma, nu = _fit_t_residuals(y, design)
    beta[1:] /= scales
    return dict(mu=0.0, sigma=sigma, nu=nu, beta=beta)


def aicc(model, x):
    x = np.asarray(x, dtype=float)
    k = 3
    if len(x) <= k + 1:
        raise ValueError("AICc requires n > k + 1")
    ll = stats.t.logpdf(x, model["nu"], loc=model["mu"], scale=model["sigma"]).sum()
    return -2 * ll + 2 * k + 2 * k * (k + 1) / (len(x) - k - 1)


def fit_nig_moments(x):
    x = np.asarray(x, dtype=float)
    m, v = x.mean(), x.var(ddof=1)
    s, k = stats.skew(x, bias=True), stats.kurtosis(x, fisher=True, bias=True)
    if k <= 0 or k <= 5 * s * s / 3:
        raise ValueError("Sample moments are outside the NIG feasible region")
    t = s * s / k
    rho2 = t / (3 - 4 * t)
    rho = np.sign(s) * np.sqrt(rho2)
    d = 3 * (1 + 4 * rho2) / k
    alpha = np.sqrt(d / (v * (1 - rho2) ** 2))
    beta = rho * alpha
    gamma = alpha * np.sqrt(1 - rho2)
    delta = d / gamma
    return dict(mu=m - delta * beta / gamma, alpha=alpha, beta=beta, delta=delta)


def fit_nig_mle(x):
    a, b, loc, scale = stats.norminvgauss.fit(np.asarray(x, dtype=float))
    return dict(mu=loc, alpha=a / scale, beta=b / scale, delta=scale)
