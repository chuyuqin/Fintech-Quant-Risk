"""Rank-correlation copulas, likelihoods, and reproducible simulation."""
import numpy as np
from scipy import stats, optimize
from .core import information_criteria


def kendall_correlation(x):
    """Elliptical latent correlation R_ij=sin(pi*tau_ij/2)."""
    x = np.asarray(x)
    r = np.eye(x.shape[1])
    for i in range(x.shape[1]):
        for j in range(i):
            r[i,j] = r[j,i] = np.sin(np.pi/2*stats.kendalltau(x[:,i], x[:,j]).statistic)
    # Do not silently repair an estimator: the caller should decide if necessary.
    if np.linalg.eigvalsh(r)[0] <= 0:
        raise ValueError('Kendall correlation is not positive definite')
    return r


def copula_logpdf(u, r, nu=None):
    """One log density per observation, joint minus transformed marginal logs."""
    u = np.clip(np.asarray(u), 1e-12, 1-1e-12)
    if nu is None:
        z = stats.norm.ppf(u)
        return stats.multivariate_normal.logpdf(z, cov=r)-stats.norm.logpdf(z).sum(axis=1)
    z = stats.t.ppf(u, nu)
    return stats.multivariate_t.logpdf(z, shape=r, df=nu)-stats.t.logpdf(z, nu).sum(axis=1)


def fit_copulas(u, r):
    """Fit nu on [2,100], holding Kendall R fixed, as in course Week 5.

    Profile over a reciprocal-df grid, then refine the best interval. Include
    endpoints explicitly. The returned ICs are conditional on R/margins: k=0/1.
    """
    objective = lambda theta: -copula_logpdf(u, r, 1/theta).sum()
    grid = np.linspace(.01, .5, 100)
    values = np.array([objective(t) for t in grid])
    j = values.argmin()
    fit = optimize.minimize_scalar(objective, bounds=(grid[max(j-1,0)], grid[min(j+1,99)]), method='bounded', options={'xatol':1e-10})
    candidates = [(fit.fun, fit.x), (values[0],grid[0]), (values[-1],grid[-1])]
    loss, theta = min(candidates)
    nu = 1/theta
    lg, lt = copula_logpdf(u,r), copula_logpdf(u,r,nu)
    rows = []
    for name, ll, k, df in [('Gaussian',lg.sum(),0,None),('t',lt.sum(),1,nu)]:
        rows.append(dict(model=name,ll=ll,k=k,nu=df,**information_criteria(ll,k,len(u))))
    return nu, rows, lg, lt


def simulate_copula(r, n=100000, nu=None, seed=5452026):
    """Common Gaussian draws across copulas; shared chi-square scale for t."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n,len(r))) @ np.linalg.cholesky(r).T
    if nu is None:
        return stats.norm.cdf(z)
    # One scale shock per simulated day, shared by all assets, induces tail dependence.
    z /= np.sqrt(rng.chisquare(nu, size=n)/nu)[:,None]
    return stats.t.cdf(z,nu)


def tail_dependence(rho, nu=None):
    """Symmetric asymptotic tail dependence for nondegenerate elliptical copulas."""
    return 0.0 if nu is None else 2*stats.t.cdf(-np.sqrt((nu+1)*(1-rho)/(1+rho)),nu+1)
