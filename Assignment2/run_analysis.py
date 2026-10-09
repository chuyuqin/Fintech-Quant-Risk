"""Reproduce Assignment 2 tables, figures, and machine-readable results.

Run from any directory: python /path/to/Assignment2/run_analysis.py.
Reusable numerical methods are imported from quant_risk; this file assembles
the data, calculations, and output tables for the five problems.
"""
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', '/tmp/fintech545-mpl')
import json
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from quant_risk import (moments, exp_weights, historical_risk, normal_var,
                       fit_margin, near_psd, higham_psd, ew_cov,
                       kendall_correlation, fit_copulas, simulate_copula, tail_dependence)

BASE = Path(__file__).resolve().parent
OUT = BASE/'results'; FIG = BASE/'figures'
OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
SEED = 5452026
N = 100000
RESULTS = {}


def read(i):
    return pd.read_csv(BASE/f'data/problem{i}.csv')


def table(name, obj):
    df = obj if isinstance(obj,pd.DataFrame) else pd.DataFrame(obj)
    df.to_csv(OUT/f'{name}.csv',index=True)
    return df


def savefig(name):
    plt.tight_layout(); plt.savefig(FIG/f'{name}.png',dpi=180); plt.close()


def tail_counts(u, scale=1):
    return [dict(pair=f'X{i+1}-X{j+1}',lower=float(np.sum((u[:,i]<=.025)&(u[:,j]<=.025))*scale),
                 upper=float(np.sum((u[:,i]>=.975)&(u[:,j]>=.975))*scale))
            for i,j in [(0,1),(0,2),(1,2)]]


def problem1():
    x=read(1).drop(columns='Day'); names=x.columns
    observed=x.notna().astype(int)
    table('p1_overlap', observed.T@observed)
    complete=x.dropna().corr(); pair=x.corr()
    table('p1_complete_correlation',complete);table('p1_pairwise_correlation',pair)
    sd=x.std(ddof=1).to_numpy(); weights=np.array([-.4,-.3,-.2,-.1,1])
    rows=[]
    matrices={'Complete case':complete.to_numpy(),'Pairwise':pair.to_numpy(),
              'Rebonato-Jackel':near_psd(pair.to_numpy()),
              'Higham':higham_psd(pair.to_numpy(),epsilon=1e-12,tol=1e-14,max_iter=10000)}
    for label,r in matrices.items():
        eig=np.linalg.eigvalsh(r)
        try: np.linalg.cholesky(r); chol='success'
        except np.linalg.LinAlgError: chol='failure'
        rows.append(dict(matrix=label,min_eigenvalue=eig[0],frobenius_to_pair=np.linalg.norm(r-pair.to_numpy()),
                         tracking_variance=(weights*sd)@r@(weights*sd),cholesky=chol))
        table('p1_eigenvalues_'+label.lower().replace(' ','_'),pd.DataFrame({'eigenvalue':eig}))
        if label in ('Rebonato-Jackel','Higham'):table('p1_'+label.lower()+'_correlation',pd.DataFrame(r,index=names,columns=names))
    table('p1_summary',rows)
    delta=matrices['Higham']-pair.to_numpy()
    changes=[dict(pair=f'{names[i]}-{names[j]}',change=delta[i,j],observations=int((observed.T@observed).iloc[i,j])) for i in range(5) for j in range(i+1,5)]
    table('p1_higham_changes',pd.DataFrame(changes).sort_values('change',key=abs,ascending=False))
    RESULTS['p1']=dict(complete_rows=len(x.dropna()),summary=rows,estimator_gap=np.linalg.norm(complete-pair),changes=changes)


def problem2():
    raw=read(2).Price.pct_change().dropna().to_numpy(); r=raw-raw.mean()
    plt.figure(figsize=(8.6,3));plt.plot(np.arange(1,501),r,lw=.8);plt.axvline(460,color='firebrick',ls='--',label='Visual split: last 40 days');plt.xlabel('Return day');plt.ylabel('Demeaned arithmetic return');plt.legend(fontsize=8);savefig('p2_returns')
    table('p2_moments',[moments(r)])
    rows=[dict(method='Normal equal',sd=r.std(ddof=1),VaR=1e6*normal_var(0,r.std(ddof=1)))]
    ew=[]
    for lam in (.97,.94):
        w=exp_weights(len(r),lam)
        # Match the course ewCovar: weighted-mean centered, normalized weights,
        # no finite-sample Bessel correction. VaR mean remains zero as requested.
        sd=np.sqrt(ew_cov(r[:,None],lam)[0,0]); neff=1/(w@w)
        rows.append(dict(method=f'Normal EW {lam}',sd=sd,VaR=1e6*normal_var(0,sd)))
        ew.append(dict(lam=lam,n_eff=neff,half_life=np.log(.5)/np.log(lam),recent40_weight=w[-40:].sum(),sd=sd,sd_se=sd/np.sqrt(2*neff),VaR_se=1e6*(-stats.norm.ppf(.05))*sd/np.sqrt(2*neff)))
    dist,fit=fit_margin(r,'t')
    rows.extend([dict(method='Student t MLE',sd=dist.std(),VaR=-1e6*dist.ppf(.05)),dict(method='Historical',sd=np.nan,VaR=1e6*historical_risk(r)['VaR'])])
    table('p2_var',rows);table('p2_ew',ew);table('p2_t_fit',[fit])
    regimes=[dict(regime='Days 1-460',n=460,sd=r[:460].std(ddof=1)),dict(regime='Days 461-500',n=40,sd=r[460:].std(ddof=1))]
    table('p2_regimes',regimes)
    RESULTS['p2']=dict(moments=moments(r),raw_mean=raw.mean(),var=rows,ew=ew,t=fit,regimes=regimes)


def problem3():
    x=read(3)[['A','B']];table('p3_moments',pd.DataFrame({c:moments(x[c]) for c in x}).T)
    pnl={'$1m A':1e6*x.A,'$1m B':1e6*x.B,'$2m A':2e6*x.A,'$1m A + $1m B':1e6*(x.A+x.B)}
    fig,axes=plt.subplots(1,2,figsize=(8.6,3))
    for ax,key in zip(axes,['$2m A','$1m A + $1m B']):
        ax.hist(pnl[key]/1e6,bins=60);ax.set(title=key,xlabel='P&L ($million)',ylabel='Scenario count (log)');ax.set_yscale('log')
    savefig('p3_pnl')
    rows=[]
    for key,y in pnl.items():
        a=historical_risk(y,.05);b=historical_risk(y,.01)
        rows.append(dict(position=key,VaR5=a['VaR'],ES5=a['ES'],normal_VaR5=normal_var(y.mean(),y.std(ddof=1)),VaR1=b['VaR'],ES1=b['ES']))
    table('p3_risk',rows)
    counts=dict(A=int((x.A<-.2).sum()),B=int((x.B<-.2).sum()),either=int((x<-.2).any(axis=1).sum()),both=int((x<-.2).all(axis=1).sum()))
    RESULTS['p3']=dict(counts=counts,risk=rows)


def problem4():
    x=read(4);n=len(x);rank=x.rank().to_numpy()/(n+1)
    table('p4_moments',pd.DataFrame({c:moments(x[c]) for c in x}).T)
    sensitivity=[]
    for c in x:
        idx=int(np.argmax(np.abs(x[c]-x[c].median())))
        sensitivity.append(dict(asset=c,removed_row=idx+1,removed_return=x[c].iloc[idx],**moments(x[c].drop(index=idx))))
    table('p4_outlier_sensitivity',sensitivity)
    fig,axes=plt.subplots(1,3,figsize=(8.6,3))
    for ax,(i,j) in zip(axes,[(0,1),(0,2),(1,2)]):
        ax.scatter(rank[:,i],rank[:,j],s=3,alpha=.4);ax.set(xlabel=x.columns[i],ylabel=x.columns[j])
    savefig('p4_ranks')
    fits=[];margins=[]
    for c in x:
        options=[fit_margin(x[c],f) for f in ('normal','t')]
        selected=min(options,key=lambda p:p[1]['AICc'])
        margins.append(selected[0])
        for _,info in options: fits.append(dict(asset=c,selected=info is selected[1],**info))
    table('p4_margin_fits',fits)
    u=np.column_stack([d.cdf(x.iloc[:,j]) for j,d in enumerate(margins)])
    r=kendall_correlation(u);table('p4_kendall_R',pd.DataFrame(r,index=x.columns,columns=x.columns))
    nu,comparison,lg,lt=fit_copulas(u,r);table('p4_copula_fits',comparison)
    alltail=[dict(model='Observed ranks',**row) for row in tail_counts(rank)]
    risk=[]
    for name,df in [('Gaussian',None),('t',nu),('Historical',None)]:
        if name=='Historical':y=x.to_numpy()
        else:
            us=simulate_copula(r,N,df,SEED)
            y=np.column_stack([d.ppf(np.clip(us[:,j],1e-12,1-1e-12)) for j,d in enumerate(margins)])
            alltail.extend([dict(model=name,**row) for row in tail_counts(us,1000/N)])
        a=historical_risk(1e6*y.sum(axis=1),.05);b=historical_risk(1e6*y.sum(axis=1),.01)
        risk.append(dict(model=name,VaR5=a['VaR'],ES5=a['ES'],VaR1=b['VaR'],ES1=b['ES']))
    table('p4_risk',risk);table('p4_tail_counts',alltail)
    # Empirical ranks define 'own outer 5%' consistently with the prediction.
    central=((rank>.05)&(rank<.95)).all(axis=1)
    central_fitted=((u>.05)&(u<.95)).all(axis=1)
    diff=lt-lg
    table('p4_daily_loglik',pd.DataFrame(dict(day=np.arange(1,n+1),gaussian=lg,t=lt,difference=diff,central_rank=central,central_fitted=central_fitted)))
    i,j=max([(0,1),(0,2),(1,2)],key=lambda ij:r[ij])
    RESULTS['p4']=dict(nu=nu,comparison=comparison,risk=risk,tail=alltail,central_days=int(central.sum()),central_gain=diff[central].sum(),total_gain=diff.sum(),central_fraction=diff[central].sum()/diff.sum(),fitted_central_days=int(central_fitted.sum()),fitted_central_gain=diff[central_fitted].sum(),tail_pair=f'X{i+1}-X{j+1}',rho=r[i,j],tail_dependence=tail_dependence(r[i,j],nu),margin_fits=fits)


def problem5():
    r=read(5)[['MKT','A','B']].pct_change().dropna().to_numpy()
    m=r[:,0];stocks=r[:,1:];design=np.column_stack([np.ones(len(m)),m])
    coef=np.linalg.lstsq(design,stocks,rcond=None)[0];errors=stocks-design@coef
    # Use n-1 in BOTH sample and residual covariance, preserving the exact
    # sample covariance decomposition. n-2 regression residual SE is also shown.
    residual_cov=np.cov(errors,rowvar=False,ddof=1);market_var=m.var(ddof=1)
    beta=coef[1];stock_cov=np.cov(stocks,rowvar=False,ddof=1)
    reconstructed=np.outer(beta,beta)*market_var+residual_cov
    np.testing.assert_allclose(reconstructed,stock_cov,rtol=1e-12,atol=1e-15)
    regressions=[dict(asset=c,alpha=coef[0,j],beta=beta[j],residual_sd=np.sqrt(residual_cov[j,j]),regression_residual_se=np.sqrt(np.sum(errors[:,j]**2)/(len(m)-2))) for j,c in enumerate(['A','B'])]
    table('p5_regressions',regressions);table('p5_residual_cov',pd.DataFrame(residual_cov,index=['A','B'],columns=['A','B']))
    rng=np.random.default_rng(SEED+5);z=rng.standard_normal((N,3));market=z[:,0]*np.sqrt(market_var)
    rows=[]
    for name,cov in [('Full residual',residual_cov),('Independent residual',np.diag(np.diag(residual_cov)))]:
        eps=z[:,1:]@np.linalg.cholesky(cov).T
        # Alpha is reported but deliberately excluded: assignment assumes zero means.
        sim=market[:,None]*beta+eps
        total=np.outer(beta,beta)*market_var+cov
        for label,w in [('P1',np.array([1e6,1e6])),('P2',np.array([1e6,-1e6]))]:
            rows.append(dict(assumption=name,portfolio=label,simulated_VaR=historical_risk(sim@w)['VaR'],analytic_VaR=normal_var(0,np.sqrt(w@total@w)),delta_normal_VaR=normal_var(0,np.sqrt(w@stock_cov@w))))
    table('p5_risk',rows)
    RESULTS['p5']=dict(regressions=regressions,residual_correlation=np.corrcoef(errors,rowvar=False)[0,1],covariance_max_error=np.max(np.abs(reconstructed-stock_cov)),risk=rows)


def main():
    for f in [problem1,problem2,problem3,problem4,problem5]:
        f();print(f.__name__+' complete',flush=True)
    # JSON null, rather than nonstandard NaN, makes this usable outside Python.
    def clean(v):
        if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
        if isinstance(v,list):return [clean(x) for x in v]
        if isinstance(v,(float,np.floating)):return float(v) if np.isfinite(v) else None
        if isinstance(v,np.integer):return int(v)
        return v
    (OUT/'summary.json').write_text(json.dumps(clean(RESULTS),indent=2))

if __name__=='__main__':main()
