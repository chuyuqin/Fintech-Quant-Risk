"""Independent numerical invariants, boundaries, and analytic cross-checks."""
import unittest
import numpy as np
from scipy import stats, integrate
from quant_risk import (historical_risk, information_criteria, near_psd, higham_psd,
                       exp_weights, ew_cov, copula_logpdf, simulate_copula, tail_dependence)


class RiskTests(unittest.TestCase):
    def test_es_boundary_mass_and_ties(self):
        # Exactly 1.5 observations enter the worst 37.5% of four observations.
        ans=historical_risk([-4,-2,1,1],.375)
        self.assertEqual(ans['VaR'],2)
        self.assertAlmostEqual(ans['ES'],10/3)
        self.assertEqual(historical_risk([-4,-2,-2,1],.75)['ES'],8/3)
        self.assertEqual(historical_risk([1,2,3,4],.25)['VaR'],-1)

    def test_es_subadditivity_noninteger_tail(self):
        rng=np.random.default_rng(11);a=rng.standard_t(4,107);b=rng.standard_t(3,107)
        self.assertLessEqual(historical_risk(a+b,.07)['ES'],historical_risk(a,.07)['ES']+historical_risk(b,.07)['ES']+1e-12)

    def test_repairs_and_higham_distance(self):
        a=np.array([[1,.9,.9],[.9,1,-.2],[.9,-.2,1]])
        repaired=[near_psd(a),higham_psd(a,epsilon=1e-11,tol=1e-13,max_iter=10000)]
        for b in repaired:
            np.testing.assert_allclose(np.diag(b),1,atol=1e-12)
            self.assertGreater(np.linalg.eigvalsh(b).min(),-1e-10)
            np.testing.assert_allclose(b,b.T,atol=1e-14)
        self.assertLessEqual(np.linalg.norm(repaired[1]-a),np.linalg.norm(repaired[0]-a)+1e-8)

    def test_ew_translation_and_uniform_limit(self):
        x=np.array([[1,5],[2,-1],[4,3]],dtype=float)
        np.testing.assert_allclose(ew_cov(x,1),np.cov(x,rowvar=False,ddof=0))
        np.testing.assert_allclose(ew_cov(x,.94),ew_cov(x+100,.94),atol=1e-12)
        w=exp_weights(500,.94)
        self.assertAlmostEqual(w.sum(),1)
        self.assertGreater(w[-1],w[0])
        self.assertAlmostEqual(1/(w@w),(1+.94)/(1-.94),places=9)

    def test_information_parameter_counts(self):
        self.assertAlmostEqual(information_criteria(10,2,100)['AICc'],-20+4+12/97)
        self.assertAlmostEqual(information_criteria(10,3,100)['AICc'],-20+6+24/96)

    def test_gaussian_independence_density(self):
        u=np.array([[.1,.2],[.7,.9],[.4,.3]])
        np.testing.assert_allclose(copula_logpdf(u,np.eye(2)),0,atol=1e-14)
        self.assertEqual(tail_dependence(.5),0)
        self.assertGreater(tail_dependence(.5,4),0)

    def test_copula_density_integrates_and_simulation_margins(self):
        # Gaussian copula must integrate to one; integrating conditionally over
        # the second coordinate gives one for fixed u1 (uniform marginal).
        r=np.array([[1,.4],[.4,1]])
        val,_=integrate.quad(lambda v:np.exp(copula_logpdf([[.4,v]],r)).item(),0,1,epsabs=1e-7)
        self.assertAlmostEqual(val,1,places=6)
        for nu in (None,4):
            u=simulate_copula(r,30000,nu,12)
            np.testing.assert_allclose(u.mean(axis=0),.5,atol=.007)
            np.testing.assert_allclose(u.var(axis=0),1/12,atol=.003)
            self.assertTrue(np.isfinite(copula_logpdf(u,r,nu)).all())

    def test_ols_covariance_decomposition(self):
        rng=np.random.default_rng(9);m=rng.normal(size=400)
        stocks=m[:,None]*[.8,1.2]+rng.normal(size=(400,2))
        design=np.c_[np.ones(400),m];b=np.linalg.lstsq(design,stocks,rcond=None)[0]
        residual=stocks-design@b
        reconstructed=np.outer(b[1],b[1])*m.var(ddof=1)+np.cov(residual,rowvar=False)
        np.testing.assert_allclose(reconstructed,np.cov(stocks,rowvar=False),atol=1e-14)


if __name__=='__main__':unittest.main()
