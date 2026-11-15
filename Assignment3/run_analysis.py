"""Reproduce every numerical table and figure, offline, from the supplied data.

All option values are per share. No option-pricing package is used. The Monte
Carlo assumes independent daily arithmetic returns; fitting permits a location,
but forecasting explicitly removes it as required by Problem 3.
"""
from pathlib import Path
import json,hashlib,platform,importlib.metadata,time
import numpy as np,pandas as pd
from scipy import stats,optimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from quant_risk.options import gbsm,greeks,implied_vol,tree
from quant_risk.stats import moments,risk,normal_risk,ols
B=Path(__file__).resolve().parent;D=B/'data';R=B/'results';F=B/'figures'
R.mkdir(exist_ok=True);F.mkdir(exist_ok=True)
plt.rcParams.update({'figure.dpi':140,'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
summary={}
def save(name,rows):
    df=pd.DataFrame(rows);df.to_csv(R/(name+'.csv'),index=False);return df

def p1():
    S=182.4;r=.0425;b=r-.006;T=35/365
    q=pd.read_csv(D/'problem1.csv');q['Mid']=(q.Bid+q.Ask)/2
    parity=[]
    for K,g in q.groupby('Strike'):
        c=g[g.Type=='Call'].iloc[0];p=g[g.Type=='Put'].iloc[0]
        gap=c.Mid-p.Mid-(S*np.exp((b-r)*T)-K*np.exp(-r*T));band=c.Ask-c.Bid+p.Ask-p.Bid
        parity.append(dict(Strike=K,gap=gap,band=band,flag=abs(gap)>band))
    save('p1_parity',parity);rows=[]
    for _,a in q.iterrows():
        row=a.to_dict();call=a.Type=='Call'
        for side in ['Bid','Mid','Ask']:
            iv,status=implied_vol(a[side],S,a.Strike,T,r,b,call);row['IV_'+side]=iv;row['status_'+side]=status
        if np.isfinite(row['IV_Mid']) and row['IV_Mid']>0:
            v=greeks(S,a.Strike,T,r,b,row['IV_Mid'],call)['vega']
            row.update(vega=v,vol_points_per_cent=1/v)
        rows.append(row)
    iv=save('p1_iv',rows)
    fig,ax=plt.subplots(figsize=(7.3,3.9))
    for typ,col in [('Call','#19749a'),('Put','#bd5b32')]:
        g=iv[iv.Type==typ];ax.plot(g.Strike,100*g.IV_Mid,'o-',label=typ+' mid',color=col)
        ax.fill_between(g.Strike,100*g.IV_Bid,100*g.IV_Ask,alpha=.14,color=col,label=typ+' bid-ask IV')
    ax.set(xlabel='Strike ($)',ylabel='Implied volatility (%)',title='Dirty chain: midpoint IV and quote uncertainty');ax.legend(ncol=2);fig.tight_layout();fig.savefig(F/'p1_iv.png');plt.close(fig)
    c=iv[(iv.Strike==175)&(iv.Type=='Call')].iloc[0];p=iv[(iv.Strike==175)&(iv.Type=='Put')].iloc[0]
    summary['p1']=dict(gap=parity[7]['gap'],vega_put=p.vega,vega_call=c.vega,iv_gap=c.IV_Mid-p.IV_Mid,linear_gap=parity[7]['gap']/p.vega,T=T)


def p2():
    S=48.5;K=50.;T=.5;r=.045;sig=.3;divs=[(.1,.25),(.4,1.5)];N=200
    prices=[]
    for call in [True,False]:
        typ='Call' if call else 'Put'
        for am in [False,True]:
            no=float(tree(S,K,T,r,sig,N,call,am));dv=float(tree(S,K,T,r,sig,N,call,am,divs))
            prices.append(dict(Type=typ,style='American' if am else 'European',no_div_tree=no,div_tree=dv,no_div_gbsm=float(gbsm(S,K,T,r,r,sig,call)) if not am else np.nan,pv_adjusted_gbsm=float(gbsm(S-sum(d*np.exp(-r*t) for t,d in divs),K,T,r,r,sig,call)) if not am else np.nan))
    save('p2_prices',prices);conv=[]
    for n in [50,55,100,105,150,155,200,205,300,305,400,405]:
        row=dict(N=n)
        for call in [True,False]:
            row['Call' if call else 'Put']=float(tree(S,K,T,r,sig,n,call,True,divs))
        conv.append(row);print('tree',n,flush=True)
    save('p2_convergence',conv);bumps=[]
    for call in [True,False]:
        base=float(tree(S,K,T,r,sig,N,call,True,divs))
        for h in [1,.1,.01,.001]:
            vm,vp=tree(np.array([S-h,S+h]),K,T,r,sig,N,call,True,divs)
            bumps.append(dict(Type='Call' if call else 'Put',bump=h,delta=(vp-vm)/(2*h),gamma=(vp+vm-2*base)/h**2))
    save('p2_greeks',bumps);sens=[]
    for call in [True,False]:
        for am in [False,True]:
            for idx in [0,1]:
                h=.001;up=divs.copy();dn=divs.copy();up[idx]=(divs[idx][0],divs[idx][1]+h);dn[idx]=(divs[idx][0],divs[idx][1]-h)
                v=(tree(S,K,T,r,sig,N,call,am,up)-tree(S,K,T,r,sig,N,call,am,dn))/(2*h)
                sens.append(dict(Type='Call' if call else 'Put',style='American' if am else 'European',dividend=idx+1,sensitivity=float(v),bump=h))
    save('p2_dividend_sensitivity',sens)
    # Immediately BEFORE t=.4, exercise for S>K iff intrinsic exceeds the
    # no-more-dividends continuation started at S-1.50. Use the SAME dt as N=200.
    def continuation(s):return float(tree(s-1.5,K,.1,r,sig,40,True,True))
    def gap(s):return s-K-continuation(s)
    boundary=optimize.brentq(gap,50.00001,100,xtol=1e-11)
    # Smooth BSM continuation gives a reference, not the submitted tree price.
    smooth=optimize.brentq(lambda s:s-K-gbsm(s-1.5,K,.1,r,r,sig,True),50.001,100)
    summary['p2']=dict(N=N,boundary=boundary,boundary_smooth=smooth,thresholds=[K*(1-np.exp(-r*(T-t))) for t,d in divs])
    fig,ax=plt.subplots(figsize=(7.2,3.0))
    for c in ['Call','Put']:ax.plot([v['N'] for v in conv],[v[c] for v in conv],'o-',label=c)
    ax.set(xlabel='N (dividends exactly on grid)',ylabel='American value ($)',title='Cash-dividend non-recombining tree convergence');ax.legend();fig.tight_layout();fig.savefig(F/'p2_convergence.png');plt.close(fig)


def p3():
    px=pd.read_csv(D/'problem3.csv');ret=px.ABC.pct_change().dropna().to_numpy();S=px.ABC.iloc[-1];book=pd.read_csv(D/'problem3_portfolios.csv');r=.0425
    save('p3_moments',[moments(ret)])
    mu,sd=stats.norm.fit(ret);nu,loc,scale=stats.t.fit(ret)
    fits=[]
    for name,k,ll,pars in [('Normal',2,stats.norm.logpdf(ret,mu,sd).sum(),dict(mu=mu,sd=sd)),('Student_t',3,stats.t.logpdf(ret,nu,loc,scale).sum(),dict(nu=nu,mu=loc,scale=scale,sd=scale*np.sqrt(nu/(nu-2))))]:
        fits.append(dict(model=name,k=k,loglik=ll,AICc=-2*ll+2*k+2*k*(k+1)/(len(ret)-k-1),**pars))
    save('p3_fits',fits)
    def value(g,spot,elapsed):
        v=np.zeros_like(np.asarray(spot,dtype=float))
        for _,a in g.iterrows():
            v+=a.Holding*(spot if a.Type=='Stock' else gbsm(spot,a.Strike,(a.DaysToExpiry-elapsed)/255,r,r,a.ImpliedVol,a.Type=='Call'))
        return v
    greekrows=[];groups=list(book.groupby('Portfolio',sort=False))
    for name,g in groups:
        row=dict(Portfolio=name,value=float(value(g,S,0)),delta=0.,gamma=0.,theta=0.)
        for _,a in g.iterrows():
            gg=dict(delta=1.,gamma=0.,theta=0.) if a.Type=='Stock' else greeks(S,a.Strike,a.DaysToExpiry/255,r,r,a.ImpliedVol,a.Type=='Call')
            for k in ['delta','gamma','theta']:row[k]+=a.Holding*gg[k]
        row['flat_spot_pnl']=float(value(g,S,10))-row['value'];greekrows.append(row)
    gr=save('p3_greeks',greekrows);dn=[]
    # Delta-normal uses sample n-1 sd, and sqrt(10) arithmetic horizon scaling.
    for a in greekrows:
        for alpha in [.05,.01]:
            v,e=normal_risk(0,abs(a['delta'])*S*ret.std(ddof=1)*np.sqrt(10),alpha)
            dn.append(dict(Portfolio=a['Portfolio'],alpha=alpha,VaR=v,ES=e))
    save('p3_delta_normal',dn)
    n=500000;rng=np.random.default_rng(5453026)
    # Common uniforms reduce noise when comparing fitted distributions.
    unif=rng.uniform(size=(n,10));models={'Normal':sd*stats.norm.ppf(unif),'Student_t':scale*stats.t.ppf(unif,nu)}
    rows=[];approx=[];monte=[]
    for model,draws in models.items():
        invalid=int(np.sum(draws<=-1))
        if invalid:raise ValueError('Arithmetic return <= -100%; define a boundary model')
        st=S*np.prod(1+draws,axis=1);ds=st-S
        for j,(name,g) in enumerate(groups):
            pnl=value(g,st,10)-greekrows[j]['value'];a=greekrows[j]
            ap=a['delta']*ds+.5*a['gamma']*ds**2+a['theta']*10/255
            for alpha in [.05,.01]:
                v,e=risk(pnl,alpha);va,ea=risk(ap,alpha)
                rows.append(dict(model=model,Portfolio=name,alpha=alpha,mean=pnl.mean(),VaR=v,ES=e,ES_VaR=e/v))
                approx.append(dict(model=model,Portfolio=name,alpha=alpha,mean=ap.mean(),VaR=va,ES=ea))
                # Ten disjoint MC batches diagnose simulation precision.
                batch=np.array([risk(x,alpha) for x in np.array_split(pnl,10)])
                monte.append(dict(model=model,Portfolio=name,alpha=alpha,mean_se=pnl.std(ddof=1)/np.sqrt(n),VaR_se=batch[:,0].std(ddof=1)/np.sqrt(10),ES_se=batch[:,1].std(ddof=1)/np.sqrt(10)))
    save('p3_full_revaluation',rows);save('p3_delta_gamma_theta',approx);save('p3_mc_uncertainty',monte)
    fig,axs=plt.subplots(2,2,figsize=(8.2,6.0));grid=np.linspace(65,140,600)
    for j,((name,g),ax) in enumerate(zip(groups,axs.flat)):
        a=greekrows[j];ax.plot(grid,value(g,grid,10)-a['value'],label='Full, 10 days elapsed');ax.plot(grid,a['delta']*(grid-S),'--',label='Delta only');ax.axhline(0,color='gray',lw=.5);ax.axvline(S,color='gray',lw=.5);ax.set(title=name,xlabel='ABC price in 10 days ($)',ylabel='Book P&L ($)')
    axs.flat[0].legend(fontsize=7);fig.tight_layout();fig.savefig(F/'p3_pnl.png');plt.close(fig)
    summary['p3']=dict(S=S,n_sim=n,seed=5453026,selected=min(fits,key=lambda a:a['AICc'])['model'],daily_sd=ret.std(ddof=1),nu=nu)


def tangency(mu,cov,rf,cap=None):
    ex=mu-rf
    if cap is None:
        x=np.linalg.solve(cov,ex);return x/x.sum()
    bounds=[(0,1)]*4
    if cap<1:bounds[3]=(0,cap)
    def objective(w):return -(w@ex)/np.sqrt(w@cov@w)
    result=optimize.minimize(objective,np.ones(4)/4 if cap==1 else np.array([.3,.3,.3,.1]),method='SLSQP',bounds=bounds,constraints=[{'type':'eq','fun':lambda w:w.sum()-1}],options={'ftol':1e-12,'maxiter':2000})
    if not result.success:raise RuntimeError(result.message)
    return result.x


def p4():
    df=pd.read_csv(D/'problem4.csv');names=['EQ','BOND','STK','PUTW'];x=df[names].to_numpy();n=len(x);rf=.03/12;mu=x.mean(axis=0);cov=np.cov(x,rowvar=False)
    rows=[]
    for j,c in enumerate(names):
        a=x[:,j];idx=np.argmin(a);z=np.delete(a,idx);m=moments(a)
        # Central sums give the share of squared/fourth deviations contributed
        # by the single worst observation, always using the full-sample mean.
        m.update(asset=c,sharpe=np.sqrt(12)*(a.mean()-rf)/a.std(ddof=1),mean_se=a.std(ddof=1)/np.sqrt(n),worst=a[idx],worst_month=df.Month.iloc[idx],variance_share_worst=(a[idx]-a.mean())**2/np.sum((a-a.mean())**2),fourth_share_worst=(a[idx]-a.mean())**4/np.sum((a-a.mean())**4),mean_without_worst=z.mean(),sd_without_worst=z.std(ddof=1),sharpe_without_worst=np.sqrt(12)*(z.mean()-rf)/z.std(ddof=1),kurt_without_worst=stats.kurtosis(z))
        rows.append(m)
    save('p4_assets',rows);weights=[];risks=[]
    ws={name:tangency(mu,cov,rf,cap) for name,cap in [('Unconstrained',None),('Long_only',1),('PUTW_cap20',.2)]}
    for name,w in ws.items():
        pm=float(w@mu);ps=float(np.sqrt(w@cov@w));sr=np.sqrt(12)*(pm-rf)/ps
        weights.append(dict(portfolio=name,**dict(zip(names,w)),mean=pm,sd=ps,sharpe=sr))
        for a in [.05,.01]:
            v,e=risk(x@w,a);vn,en=normal_risk(pm,ps,a)
            risks.append(dict(portfolio=name,alpha=a,historical_VaR=v,historical_ES=e,normal_VaR=vn,normal_ES=en))
    save('p4_weights',weights);save('p4_risk',risks)
    beta=[]
    for label,mask in [('All_months',np.ones(n,dtype=bool)),('EQ_excess_below_minus5pct',x[:,0]-rf<-.05)]:
        fit=ols(x[mask,3]-rf,x[mask,0]-rf)
        beta.append(dict(sample=label,n=fit['n'],alpha=fit['beta'][0],alpha_se=fit['se'][0],alpha_t=fit['t'][0],beta=fit['beta'][1],beta_se=fit['se'][1],R2=fit['r2']))
    save('p4_capm',beta)
    # Resample complete rows to preserve asset dependence. Report selection
    # instability, not a confidence interval for structural optimal weights.
    rng=np.random.default_rng(5454026);boots=[]
    for i in range(500):
        xb=x[rng.integers(0,n,n)];bm=xb.mean(0);bc=np.cov(xb,rowvar=False)
        for label,cap in [('Unconstrained',None),('Long_only',1),('PUTW_cap20',.2)]:
            w=tangency(bm,bc,rf,cap);boots.append(dict(draw=i,portfolio=label,**dict(zip(names,w))))
    boot=save('p4_bootstrap_draws',boots);intervals=[]
    for label,g in boot.groupby('portfolio',sort=False):
        for c in names:intervals.append(dict(portfolio=label,asset=c,p05=g[c].quantile(.05),median=g[c].median(),p95=g[c].quantile(.95)))
    save('p4_bootstrap_intervals',intervals)
    # Long-only minimum-variance frontier at each feasible target return.
    front=[]
    for target in np.linspace(mu.min(),mu.max(),100):
        res=optimize.minimize(lambda w:w@cov@w,np.ones(4)/4,method='SLSQP',bounds=[(0,1)]*4,constraints=[{'type':'eq','fun':lambda w:w.sum()-1},{'type':'eq','fun':lambda w:w@mu-target}],options={'ftol':1e-13,'maxiter':1000})
        if not res.success:raise RuntimeError('Frontier: '+res.message)
        front.append(dict(mean=target,sd=np.sqrt(res.fun)))
    fr=save('p4_frontier',front);fig,ax=plt.subplots(figsize=(7.2,4.2));gmv=fr.sd.idxmin()
    ax.plot(fr.sd.iloc[gmv:]*np.sqrt(12)*100,fr['mean'].iloc[gmv:]*12*100,label='Long-only efficient frontier')
    ax.plot(fr.sd.iloc[:gmv+1]*np.sqrt(12)*100,fr['mean'].iloc[:gmv+1]*12*100,':',color='gray',label='Inefficient lower branch')
    vol=np.linspace(0,max(np.sqrt(np.diag(cov)))*np.sqrt(12),100);sr=weights[1]['sharpe']
    ax.plot(vol*100,(.03+sr*vol)*100,'--',label='CML through long-only tangency')
    for j,c in enumerate(names):ax.scatter(np.sqrt(cov[j,j])*np.sqrt(12)*100,mu[j]*12*100);ax.annotate(c,(np.sqrt(cov[j,j])*np.sqrt(12)*100,mu[j]*12*100),xytext=(4,3),textcoords='offset points')
    w=weights[1];ax.scatter(w['sd']*np.sqrt(12)*100,w['mean']*12*100,marker='*',s=120,color='black',label='Long-only tangency')
    ax.set(xlabel='Annualized volatility (%)',ylabel='Annualized arithmetic mean (%)',title='Mean-variance frontier and capital market line');ax.legend(fontsize=7);fig.tight_layout();fig.savefig(F/'p4_frontier.png');plt.close(fig)
    summary['p4']=dict(rf_month=rf,bootstrap_n=500,crash_months=beta[1]['n'])


def load_factors():
    ff=pd.read_csv(D/'F-F_Research_Data_Factors_daily.CSV');mo=pd.read_csv(D/'F-F_Momentum_Factor_daily.CSV')
    for a in [ff,mo]:
        a.columns=a.columns.str.strip();a.Date=pd.to_datetime(a.Date.astype(str),format='%Y%m%d');a.loc[:,a.columns!='Date']=a.loc[:,a.columns!='Date']/100
    return ff.merge(mo,on='Date',validate='one_to_one')


def p5():
    fac=load_factors();fund=pd.read_csv(D/'problem5.csv',parse_dates=['Date']);df=fund.merge(fac,on='Date',how='inner',validate='one_to_one').sort_values('Date')
    assert len(df)==len(fund),'Missing factor dates'
    models={'CAPM':['Mkt-RF'],'FF3':['Mkt-RF','SMB','HML'],'Carhart4':['Mkt-RF','SMB','HML','Mom']}
    fits={};coef=[];metrics=[];fundstats=[];corr=[];premia=[]
    for c in ['FUND1','FUND2']:
        y=df[c]-df.RF;fs=moments(y);fs.update(fund=c,total_sd=df[c].std(ddof=1),sharpe=np.sqrt(255)*y.mean()/y.std(ddof=1),annual_realized_arithmetic=255*df[c].mean(),annual_realized_geometric=np.prod(1+df[c])**(255/len(df))-1);fundstats.append(fs)
        for name,cols in models.items():
            fit=ols(y,df[cols]);fits[c,name]=fit
            for k,b,se in zip(['alpha']+cols,fit['beta'],fit['se']):coef.append(dict(fund=c,model=name,term=k,coefficient=b,se=se))
            metrics.append(dict(fund=c,model=name,n=fit['n'],alpha_t=fit['t'][0],R2=fit['r2'],AICc=fit['aicc']))
        for f in ['SMB','HML','Mom']:corr.append(dict(fund=c,factor=f,correlation=np.corrcoef(fits[c,'CAPM']['resid'],df[f])[0,1]))
    save('p5_fund_stats',fundstats);save('p5_coefficients',coef);save('p5_model_metrics',metrics);save('p5_capm_residual_correlations',corr)
    # Most recent 10 calendar years END AT THE FILE'S LAST DATE, not the fund's
    # sample end. This deliberately separates expected from sample returns.
    end=fac.Date.max();start=end-pd.DateOffset(years=10);recent=fac[(fac.Date>start)&(fac.Date<=end)]
    factors=['Mkt-RF','SMB','HML','Mom'];means=recent[['RF']+factors].mean();sample=df[['RF']+factors].mean()
    save('p5_factor_premia',[dict(factor=k,recent10_daily=means[k],sample_daily=sample[k],recent10_annual=255*means[k],sample_annual=255*sample[k]) for k in ['RF']+factors])
    worst=fac.loc[fac.Mom.idxmin()];expected=[];stress=[];explain=[]
    for c in ['FUND1','FUND2']:
        fit=fits[c,'Carhart4'];base=means.RF+fit['beta'][1:]@means[factors]
        expected.append(dict(fund=c,without_alpha=255*base,with_alpha=255*(base+fit['beta'][0]),alpha_annual=255*fit['beta'][0]))
        pnl=1e7*fit['beta'][-1]*worst.Mom;daily_sd=df[c].std(ddof=1)
        stress.append(dict(fund=c,momentum_beta=fit['beta'][-1],momentum_shock=worst.Mom,pnl=pnl,loss=-pnl,loss_in_daily_sd=-pnl/(1e7*daily_sd),daily_total_sd=daily_sd))
        cap=fits[c,'CAPM']['beta'];full=fit['beta']
        # The OLS mean identity exactly decomposes the CAPM alpha change.
        parts={'market_beta_change':(full[1]-cap[1])*sample['Mkt-RF'],'SMB':full[2]*sample.SMB,'HML':full[3]*sample.HML,'Mom':full[4]*sample.Mom}
        explain.append(dict(fund=c,capm_alpha=cap[0],four_alpha=full[0],alpha_drop=cap[0]-full[0],**parts,sum_contributions=sum(parts.values())))
    save('p5_expected_returns',expected);save('p5_momentum_stress',stress);save('p5_alpha_decomposition',explain)
    summary['p5']=dict(n=len(df),sample_start=str(df.Date.min().date()),sample_end=str(df.Date.max().date()),recent_start_exclusive=str(start.date()),recent_end=str(end.date()),recent_n=len(recent),momentum_worst_date=str(worst.Date.date()),momentum_worst=worst.Mom,annual_days=255)

if __name__=='__main__':
    for fn in [p1,p2,p3,p4,p5]:
        print('Running',fn.__name__,flush=True);t=time.time();fn();print('Finished',round(time.time()-t,2),'s',flush=True)
    (R/'summary.json').write_text(json.dumps(summary,indent=2,default=lambda x:x.item()))
    versions={k:importlib.metadata.version(k) for k in ['numpy','scipy','pandas','matplotlib']};versions['python']=platform.python_version()
    (B/'environment_versions.json').write_text(json.dumps(versions,indent=2))
    (D/'SHA256SUMS.txt').write_text('\n'.join(hashlib.sha256(f.read_bytes()).hexdigest()+'  '+f.name for f in sorted(D.iterdir()) if f.suffix.lower()=='.csv')+'\n')
