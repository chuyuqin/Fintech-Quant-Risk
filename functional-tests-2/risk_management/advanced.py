"""Risk, optimization, attribution and copula routines; no reference-file access."""

import numpy as np
import pandas as pd
from scipy import stats, optimize
from .covariance import higham_psd


def sample_risk(x, alpha=0.05):
    x = np.sort(np.asarray(x))
    q = (
        x[max(0, int(np.ceil(len(x) * alpha)) - 1)]
        + x[max(0, int(np.floor(len(x) * alpha)) - 1)]
    ) / 2
    return np.array([-q, -x[x <= q].mean()])


def distribution_risk(mu, sigma, nu=None, alpha=0.05):
    z = stats.norm.ppf(alpha) if nu is None else stats.t.ppf(alpha, nu)
    es = (
        stats.norm.pdf(z) / alpha
        if nu is None
        else (nu + z * z) / (nu - 1) * stats.t.pdf(z, nu) / alpha
    )
    return np.array([-mu - sigma * z, -mu + sigma * es])


def portfolio_risk(returns, values, names):
    pnl = np.asarray(returns) * np.asarray(values)
    rows = []
    for name, x, value in zip(
        list(names) + ["Total"],
        list(pnl.T) + [pnl.sum(axis=1)],
        list(values) + [sum(values)],
    ):
        v, e = sample_risk(x)
        rows.append([name, v, e, v / value, e / value])
    return pd.DataFrame(
        rows, columns=["Stock", "VaR95", "ES95", "VaR95_Pct", "ES95_Pct"]
    )


def risk_parity(cov, budget=None):
    cov = np.asarray(cov)
    n = len(cov)
    budget = np.ones(n) if budget is None else np.asarray(budget)

    def objective(w):
        c = w * (cov @ w) / np.sqrt(w @ cov @ w) / budget
        return 1e5 * np.sum((c - c.mean()) ** 2)

    fit = optimize.minimize(
        objective,
        np.ones(n) / n,
        bounds=[(0, 1)] * n,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
        method="SLSQP",
        options={"ftol": 1e-13, "maxiter": 2000},
    )
    if not fit.success:
        raise RuntimeError(fit.message)
    return fit.x


def max_sharpe(cov, means, rf, bounds=None):
    n = len(means)
    excess = np.asarray(means) - rf

    def objective(w):
        return -(w @ excess) / np.sqrt(w @ cov @ w)

    fit = optimize.minimize(
        objective,
        np.ones(n) / n,
        bounds=bounds or [(0, 1)] * n,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
        method="SLSQP",
        options={"ftol": 1e-13, "maxiter": 2000},
    )
    if not fit.success:
        raise RuntimeError(fit.message)
    return fit.x


def attribution(weights, returns, factors, beta):
    r = np.asarray(returns)
    f = np.asarray(factors)
    w = np.array(weights, dtype=float).copy()
    pr = []
    fw = []
    for row in r:
        fw.append(w @ beta)
        nextw = w * (1 + row)
        gross = nextw.sum()
        pr.append(gross - 1)
        w = nextw / gross
    pr = np.array(pr)
    fw = np.array(fw)
    residual = pr - (f * fw).sum(axis=1)
    total = np.expm1(np.log1p(pr).sum())
    ratio = lambda x: np.divide(
        np.log1p(x), x, out=np.ones_like(np.asarray(x)), where=np.asarray(x) != 0
    )
    k = ratio(pr) / ratio(total)
    y = np.column_stack((f * fw, residual))
    allr = np.column_stack((f, residual, pr))
    tr = np.expm1(np.log1p(allr).sum(axis=0))
    ar = np.r_[(y * k[:, None]).sum(axis=0), total]
    slopes = np.linalg.lstsq(np.column_stack((np.ones(len(pr)), pr)), y, rcond=None)[0][
        1
    ]
    vr = np.r_[slopes * pr.std(ddof=1), pr.std(ddof=1)]
    return pd.DataFrame(
        np.vstack((tr, ar, vr)), columns=list(factors.columns) + ["Alpha", "Portfolio"]
    ).assign(Value=["TotalReturn", "Return Attribution", "Vol Attribution"])


def kendall_correlation(x):
    n = x.shape[1]
    r = np.eye(n)
    for i in range(n):
        for j in range(i):
            r[i, j] = r[j, i] = np.sin(
                np.pi * stats.kendalltau(x[:, i], x[:, j]).statistic / 2
            )
    if np.linalg.eigvalsh(r).min() < -1e-8:
        r = higham_psd(r)
    r = (r + r.T) / 2
    if np.linalg.eigvalsh(r).min() < 1e-8:
        r = (r + np.eye(n) * 1e-8) / (1 + 1e-8)
    return r


def profile_nu(ll):
    grid = np.linspace(0.01, 0.49, 200)
    i = np.argmax([ll(1 / t) for t in grid])
    fine = np.linspace(grid[max(i - 1, 0)], grid[min(i + 1, 199)], 200)
    vals = np.array([ll(1 / t) for t in fine])
    j = vals.argmax()
    return 1 / fine[j], vals[j]


def fit_multivariate_t(x):
    mu = x.mean(axis=0)
    sd = x.std(axis=0, ddof=1)
    r = kendall_correlation(x)
    scale = lambda nu: np.outer(sd, sd) * r * (nu - 2) / nu
    nu, ll = profile_nu(
        lambda nu: stats.multivariate_t.logpdf(x, loc=mu, shape=scale(nu), df=nu).sum()
    )
    return mu, scale(nu), nu, ll


def copula_ll(u, r, nu=None):
    u = np.clip(u, 1e-12, 1 - 1e-12)
    if nu is None:
        z = stats.norm.ppf(u)
        return (
            stats.multivariate_normal.logpdf(z, cov=r).sum()
            - stats.norm.logpdf(z).sum()
        )
    z = stats.t.ppf(u, nu)
    return (
        stats.multivariate_t.logpdf(z, shape=r, df=nu).sum()
        - stats.t.logpdf(z, nu).sum()
    )


def simulate_copula(r, n=100000, nu=None, seed=13):
    rng = np.random.default_rng(seed)
    z = rng.multivariate_normal(np.zeros(len(r)), r, n)
    if nu is None:
        return stats.norm.cdf(z)
    z *= np.sqrt(nu / np.random.default_rng(seed + 1).chisquare(nu, n))[:, None]
    return stats.t.cdf(z, nu)


def tail_dependence(rho, nu):
    return 2 * stats.t.cdf(-np.sqrt((nu + 1) * (1 - rho) / (1 + rho)), nu + 1)
