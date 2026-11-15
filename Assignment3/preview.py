"""Pre-fit diagnostics required by the Predict parts; no IV/distribution/optimizer fitting."""
from pathlib import Path
import pandas as pd,numpy as np
from quant_risk.options import greeks
from quant_risk.stats import moments,ols
B=Path(__file__).resolve().parent;D=B/'data'
q=pd.read_csv(D/'problem1.csv');q['mid']=(q.Bid+q.Ask)/2
p=q.pivot(index='Strike',columns='Type',values=['mid','Bid','Ask']);T=35/365
p['gap']=p['mid','Call']-p['mid','Put']-(182.4*np.exp(-.006*T)-p.index*np.exp(-.0425*T));p['band']=p['Ask','Call']-p['Bid','Call']+p['Ask','Put']-p['Bid','Put']
print('P1 parity',p[['gap','band']].to_string());print(q.tail(8).to_string(index=False));print(q[q.Type=='Call'].tail(4))
S=pd.read_csv(D/'problem3.csv').ABC.iloc[-1];print('P3 spot',S)
book=pd.read_csv(D/'problem3_portfolios.csv');out=[]
for name,g in book.groupby('Portfolio',sort=False):
 v=dict(delta=0.,gamma=0.,theta=0.)
 for _,row in g.iterrows():
  gg=dict(delta=1.,gamma=0.,theta=0.) if row.Type=='Stock' else greeks(S,row.Strike,row.DaysToExpiry/255,.0425,.0425,row.ImpliedVol,row.Type=='Call')
  for k in v:v[k]+=row.Holding*gg[k]
 out.append(dict(Portfolio=name,**v))
print(pd.DataFrame(out).to_string(index=False))
x=pd.read_csv(D/'problem4.csv');rf=.03/12
for c in ['EQ','BOND','STK','PUTW']:
 a=x[c];m=moments(a);m.update(asset=c,sharpe=np.sqrt(12)*(a.mean()-rf)/a.std(ddof=1),worst=a.min(),worstmonth=x.Month[a.idxmin()],sd_without_worst=a.drop(a.idxmin()).std(ddof=1),mean_without_worst=a.drop(a.idxmin()).mean());print('P4',m)
f=pd.read_csv(D/'problem5.csv',parse_dates=['Date']);ff=pd.read_csv(D/'F-F_Research_Data_Factors_daily.CSV');ff.Date=pd.to_datetime(ff.Date.astype(str));ff.iloc[:,1:]=ff.iloc[:,1:]/100
mom=pd.read_csv(D/'F-F_Momentum_Factor_daily.CSV');mom.columns=mom.columns.str.strip();mom.Date=pd.to_datetime(mom.Date.astype(str));mom.iloc[:,1:]=mom.iloc[:,1:]/100
f=f.merge(ff,on='Date',validate='one_to_one').merge(mom,on='Date',validate='one_to_one');print('P5 columns',f.columns)
for c in ['FUND1','FUND2']:
 y=f[c]-f.RF;m=moments(y);o=ols(y,f[['Mkt-RF']]);print(c,m,'Sharpe',np.sqrt(255)*y.mean()/y.std(ddof=1),'CAPM',o['beta'],o['t'],'resid corr',[(k,np.corrcoef(o['resid'],f[k])[0,1]) for k in ['SMB','HML','Mom']])
