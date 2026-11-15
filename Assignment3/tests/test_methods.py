"""Independent small-tree reference, analytic checks, and accounting invariants."""
import unittest
from pathlib import Path
import numpy as np,pandas as pd
from scipy import stats
from quant_risk.options import gbsm,greeks,tree,implied_vol
from quant_risk.stats import risk,ols

class Methods(unittest.TestCase):
    def test_gbsm_parity_and_greeks(self):
        S,K,T,r,b,v=102,100,.7,.04,.025,.31
        c=gbsm(S,K,T,r,b,v,True);p=gbsm(S,K,T,r,b,v,False)
        self.assertAlmostEqual(c-p,S*np.exp((b-r)*T)-K*np.exp(-r*T),places=12)
        for call in [True,False]:
            g=greeks(S,K,T,r,b,v,call);h=.001;base=gbsm(S,K,T,r,b,v,call)
            vp,vm=gbsm(S+h,K,T,r,b,v,call),gbsm(S-h,K,T,r,b,v,call)
            self.assertAlmostEqual(g['delta'],(vp-vm)/(2*h),places=8)
            self.assertAlmostEqual(g['gamma'],(vp+vm-2*base)/h**2,places=6)
            self.assertAlmostEqual(g['theta'],(gbsm(S,K,T-h,r,b,v,call)-gbsm(S,K,T+h,r,b,v,call))/(2*h),places=4)
            self.assertAlmostEqual(g['vega'],(gbsm(S,K,T,r,b,v+1e-5,call)-gbsm(S,K,T,r,b,v-1e-5,call))/2e-5,places=6)
            self.assertAlmostEqual(implied_vol(base,S,K,T,r,b,call)[0],v,places=10)
        self.assertTrue(np.isnan(implied_vol(-1,S,K,T,r,b,True)[0]))

    def test_dividend_tree_against_full_binary_recursion(self):
        # Independent reference does NOT recombine or call the production tree.
        N=5;T=.5;r=.045;sig=.3;S=48.5;K=50;dt=T/N
        u=np.exp(sig*np.sqrt(dt));p=(np.exp(r*dt)-1/u)/(u-1/u);df=np.exp(-r*dt)
        for call in [True,False]:
            z=1 if call else -1
            for am in [False,True]:
                def ref(s,j):
                    intrinsic=max(z*(s-K),0)
                    if j==N:return intrinsic
                    post=max(s-{1:.25,4:1.5}.get(j,0),0)
                    hold=df*(p*ref(post*u,j+1)+(1-p)*ref(post/u,j+1))
                    # Both sides of the dividend event are exercisable. For a put,
                    # immediate post-dividend exercise can dominate pre-dividend exercise.
                    return max(intrinsic,max(z*(post-K),0),hold) if am else hold
                self.assertAlmostEqual(float(tree(S,K,T,r,sig,N,call,am,[(.1,.25),(.4,1.5)])),ref(S,0),places=12)

    def test_tree_limit_and_dividend_parity(self):
        args=(48.5,50,.5,.045,.3)
        for call in [True,False]:
            tr=float(tree(*args,800,call,False));bs=float(gbsm(48.5,50,.5,.045,.045,.3,call))
            self.assertLess(abs(tr-bs),.005)
        n=100;dv=[(.1,.25),(.4,1.5)]
        c=tree(*args,n,True,False,dv);p=tree(*args,n,False,False,dv)
        forward=48.5-sum(D*np.exp(-.045*t) for t,D in dv)-50*np.exp(-.045*.5)
        self.assertAlmostEqual(float(c-p),forward,places=10)
        for call in [True,False]:
            eu=float(tree(*args,n,call,False,dv));am=float(tree(*args,n,call,True,dv))
            self.assertGreaterEqual(am+1e-12,eu)
        self.assertRaises(ValueError,tree,*args,101,True,True,dv)

    def test_fractional_es_and_tail_ties(self):
        v,e=risk([-10,-2,3,8,10],.3);self.assertEqual(v,2);self.assertAlmostEqual(e,22/3)
        v,e=risk([-4,-4,1,2],.375);self.assertEqual(v,4);self.assertEqual(e,4)

    def test_saved_accounting_and_constraints(self):
        root=Path(__file__).resolve().parents[1]/'results'
        w=pd.read_csv(root/'p4_weights.csv')
        for _,a in w.iterrows():self.assertAlmostEqual(a[['EQ','BOND','STK','PUTW']].sum(),1,places=10)
        for name in ['Long_only','PUTW_cap20']:
            a=w[w.portfolio==name].iloc[0];self.assertGreaterEqual(a[['EQ','BOND','STK','PUTW']].min(),-1e-10)
        self.assertLessEqual(w[w.portfolio=='PUTW_cap20'].PUTW.iloc[0],.20000001)
        a=pd.read_csv(root/'p5_alpha_decomposition.csv');np.testing.assert_allclose(a.alpha_drop,a.sum_contributions,atol=1e-12)
        riskdf=pd.read_csv(root/'p3_full_revaluation.csv');self.assertTrue((riskdf.ES>=riskdf.VaR).all())
        self.assertEqual(len(pd.read_csv(root/'p1_iv.csv')),36)

if __name__=='__main__':unittest.main()
