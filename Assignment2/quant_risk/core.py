"""Distribution fitting, descriptive moments, and risk-measure conventions."""
import numpy as np
from scipy import stats


def moments(x):
    """Mean, unbiased sample variance, moment skewness, and excess kurtosis.

    Skewness and kurtosis use empirical central moments (bias=True), matching
    the descriptive moment convention in the course, rather than bias corrections.
    """
    x = np.asarray(x, dtype=float)
    return dict(mean=x.mean(), variance=x.var(ddof=1),
                skewness=stats.skew(x, bias=True),
                excess_kurtosis=stats.kurtosis(x, fisher=True, bias=True))


def exp_weights(n, lam):
    """Finite, normalized exponential weights; last observation is most recent."""
    if n < 1 or not 0 < lam <= 1:
        raise ValueError('Need n >= 1 and 0 < lambda <= 1')
    w = lam ** np.arange(n - 1, -1, -1, dtype=float)
    return w / w.sum()


def historical_risk(pnl, alpha=.05):
    """Loss-positive empirical VaR and integrated-quantile ES.

    VaR is the left inverse empirical CDF (ceil(n*alpha)-th sorted P&L).
    ES averages exactly alpha probability mass, fractionally weighting the
    boundary observation if necessary. Thus ties cannot inflate tail mass.
    At integer n*alpha with no ties this matches the course RiskStats.jl.
    A negative VaR is retained: even the selected bad quantile is a gain.
    """
    x = np.sort(np.asarray(pnl, dtype=float).reshape(-1))
    if len(x) == 0 or not np.all(np.isfinite(x)) or not 0 < alpha <= 1:
        raise ValueError('Need finite observations and 0 < alpha <= 1')
    mass = alpha * len(x)
    k = int(np.floor(mass))
    fraction = mass - k
    tail_sum = x[:k].sum() + (fraction * x[k] if k < len(x) else 0)
    return dict(VaR=-x[int(np.ceil(mass)) - 1], ES=-tail_sum / mass)


def normal_var(mean, sd, alpha=.05):
    """VaR of a normally distributed P&L, without clipping gains to zero."""
    return -mean - stats.norm.ppf(alpha) * sd


def information_criteria(ll, k, n):
    """Count estimated parameters once, including scale but no extra constant."""
    if n <= k + 1:
        raise ValueError('AICc requires n > k + 1')
    return dict(AICc=-2*ll + 2*k + 2*k*(k+1)/(n-k-1), BIC=-2*ll+k*np.log(n))


def fit_margin(x, family):
    """Univariate MLE with free location and scale: normal k=2, Student t k=3.

    Standardizing improves optimization on small daily-return scales. Several
    t starting degrees of freedom reduce dependence on a single local search.
    The t scale is NOT its standard deviation: sd=scale*sqrt(nu/(nu-2)).
    """
    x = np.asarray(x, dtype=float)
    center, scale = x.mean(), x.std(ddof=0)
    if scale <= 0:
        raise ValueError('Need nonconstant observations')
    if family == 'normal':
        pars = (center, scale)
        dist = stats.norm(*pars)
        info = dict(family=family, mu=center, scale=scale, nu=None, k=2)
    elif family == 't':
        y = (x-center)/scale
        trials = [stats.t.fit(y, df, loc=0, scale=1) for df in (3, 8, 30)]
        nu, loc, s = max(trials, key=lambda p: stats.t.logpdf(y, *p).sum())
        dist = stats.t(nu, loc=center+scale*loc, scale=scale*s)
        info = dict(family=family, mu=center+scale*loc, scale=scale*s, nu=nu, k=3)
    else:
        raise ValueError('family must be normal or t')
    info['ll'] = dist.logpdf(x).sum()
    info.update(information_criteria(info['ll'], info['k'], len(x)))
    return dist, info
