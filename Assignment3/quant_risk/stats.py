"""Conventions shared by the five analyses; returns are decimals throughout."""
import numpy as np
from scipy import stats

def moments(x):
    x=np.asarray(x);return dict(n=len(x),mean=float(x.mean()),variance=float(x.var(ddof=1)),sd=float(x.std(ddof=1)),skew=float(stats.skew(x,bias=True)),excess_kurtosis=float(stats.kurtosis(x,bias=True)))

def risk(x,alpha):
    # Left-inverse empirical quantile; ES integrates exactly alpha mass, using
    # fractional boundary weight when n*alpha is not an integer. No clipping.
    x=np.sort(np.asarray(x));mass=alpha*len(x);k=int(np.floor(mass));frac=mass-k
    var=-x[int(np.ceil(mass))-1]
    es=-(x[:k].sum()+(frac*x[k] if frac>1e-12 else 0))/mass
    return float(var),float(es)

def normal_risk(mu,sd,alpha):
    z=stats.norm.ppf(alpha);return float(-mu-sd*z),float(-mu+sd*stats.norm.pdf(z)/alpha)

def ols(y,X):
    y=np.asarray(y);X=np.column_stack([np.ones(len(y)),np.asarray(X)]);n,k=X.shape
    beta=np.linalg.lstsq(X,y,rcond=None)[0];resid=y-X@beta;sse=resid@resid
    se=np.sqrt(np.diag(np.linalg.inv(X.T@X))*sse/(n-k))
    ll=-n/2*(np.log(2*np.pi)+1+np.log(sse/n))
    # Gaussian regression likelihood estimates k mean coefficients AND variance.
    kp=k+1;aicc=-2*ll+2*kp+2*kp*(kp+1)/(n-kp-1)
    return dict(beta=beta,se=se,t=beta/se,r2=1-sse/np.sum((y-y.mean())**2),aicc=aicc,resid=resid,n=n)
