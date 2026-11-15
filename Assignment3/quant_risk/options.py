"""Own GBSM, analytic Greeks, quote inversion, and cash-dividend CRR tree.

The dividend tree is recombining ONLY between cash dividends. Every pre-dividend
node spawns its own shifted subtree; no interpolation or spot-PV approximation
is used. Batching those subtrees with NumPy changes storage, not the algorithm.
"""
import numpy as np
from scipy.special import ndtr
from scipy.optimize import brentq


def gbsm(S,K,T,r,b,sigma,call=True):
    S=np.asarray(S,dtype=float)
    if T<=0:return np.maximum((1 if call else -1)*(S-K),0)
    if sigma<=0:return np.exp(-r*T)*np.maximum((1 if call else -1)*(S*np.exp(b*T)-K),0)
    safe=np.maximum(S,1e-250)
    d1=(np.log(safe/K)+(b+sigma*sigma/2)*T)/(sigma*np.sqrt(T));d2=d1-sigma*np.sqrt(T)
    z=1 if call else -1
    value=z*(S*np.exp((b-r)*T)*ndtr(z*d1)-K*np.exp(-r*T)*ndtr(z*d2))
    return np.where(S<=0,0 if call else K*np.exp(-r*T),value)


def greeks(S,K,T,r,b,sigma,call=True):
    d1=(np.log(S/K)+(b+sigma*sigma/2)*T)/(sigma*np.sqrt(T));d2=d1-sigma*np.sqrt(T)
    phi=np.exp(-d1*d1/2)/np.sqrt(2*np.pi);z=1 if call else -1
    carry=np.exp((b-r)*T)
    return dict(delta=z*carry*ndtr(z*d1),gamma=carry*phi/(S*sigma*np.sqrt(T)),
                vega=S*carry*phi*np.sqrt(T),
                theta=-S*carry*phi*sigma/(2*np.sqrt(T))-z*(b-r)*S*carry*ndtr(z*d1)-z*r*K*np.exp(-r*T)*ndtr(z*d2))


def implied_vol(price,S,K,T,r,b,call):
    # Finite nonnegative volatility prices are bounded by discounted intrinsic
    # and the discounted stock (call) / strike (put). A zero-vol endpoint is valid.
    lower=float(gbsm(S,K,T,r,b,0,call)); upper=S*np.exp((b-r)*T) if call else K*np.exp(-r*T)
    if price<lower-1e-10:return np.nan,'below lower bound'
    if price>=upper:return np.nan,'at/above finite-vol upper bound'
    if abs(price-lower)<1e-10:return 0.,'zero-vol boundary'
    hi=1.
    while gbsm(S,K,T,r,b,hi,call)<price and hi<128:hi*=2
    return brentq(lambda v:float(gbsm(S,K,T,r,b,v,call))-price,1e-10,hi,xtol=1e-13),'ok'


def tree(S,K,T,r,sigma,N,call=True,american=True,dividends=()):
    """CRR with pre-dividend exercise and post-dividend spot max(S-D,0).

    Dividend dates MUST be exact grid dates. Limited liability clips negative
    ex-dividend spots to zero (material only for extreme unreachable tails here).
    Values at zero thereafter use an absorbing stock boundary. Dividends are
    stated in year fractions. This implementation supports batched root spots.
    """
    dt=T/N;u=np.exp(sigma*np.sqrt(dt));d=1/u;p=(np.exp(r*dt)-d)/(u-d);df=np.exp(-r*dt)
    if not 0<p<1:raise ValueError('CRR probabilities invalid')
    dates=[]
    for t,D in dividends:
        j=round(t/dt)
        if abs(j*dt-t)>1e-10 or not 0<j<N:raise ValueError('dividend not on interior grid')
        dates.append((j,D))
    z=1 if call else -1
    def payoff(s):return np.maximum(z*(s-K),0.)
    def subtree(roots,start,remaining):
        # Keep one independent recombining tree for every entering root.
        end,D=remaining[0] if remaining else (N,0.)
        m=end-start;updowns=u**(2*np.arange(m+1)-m)
        spots=np.asarray(roots)[...,None]*updowns
        if remaining:
            vals=subtree(np.maximum(spots-D,0.),end,remaining[1:])
            if american:vals=np.maximum(vals,payoff(spots))
        else:vals=payoff(spots)
        for j in range(m-1,-1,-1):
            vals=df*((1-p)*vals[...,:-1]+p*vals[...,1:])
            if american:
                spots=np.asarray(roots)[...,None]*u**(2*np.arange(j+1)-j)
                vals=np.maximum(vals,payoff(spots))
        return vals[...,0]
    return subtree(np.asarray(S,dtype=float),0,dates)
