import unittest
import numpy as np
import pandas as pd
from scipy import stats
from risk_management import *


class Invariants(unittest.TestCase):
    def test_repairs_preserve_variances_and_make_psd(self):
        a = np.array([[4., 2.4, 0.], [2.4, 1., .8], [0., .8, 1.]])
        for repair in (near_psd, higham_psd):
            result = repair(a)
            np.testing.assert_allclose(np.diag(result), np.diag(a))
            self.assertGreater(np.linalg.eigvalsh(result).min(), -1e-8)

    def test_cholesky_reconstructs_singular_matrix(self):
        a = np.array([[1., 1.], [1., 1.]])
        root = chol_psd(a)
        np.testing.assert_allclose(root @ root.T, a)
        with self.assertRaises(np.linalg.LinAlgError):
            chol_psd(np.array([[0., 1.], [1., 0.]]))

    def test_ew_equal_weights_and_translation_invariance(self):
        x = np.arange(30.).reshape(10, 3)
        np.testing.assert_allclose(ew_cov(x, 1), np.cov(x, rowvar=False, ddof=0))
        np.testing.assert_allclose(ew_cov(x, .94), ew_cov(x + 100, .94))

    def test_returns_dates_and_values(self):
        prices = pd.DataFrame({"Date":["a","b","c"], "Stock":[100.,110.,99.]})
        out = calculate_returns(prices)
        self.assertEqual(list(out.Date), ["b","c"])
        np.testing.assert_allclose(out.Stock, [.1,-.1])
        np.testing.assert_allclose(calculate_returns(prices,"LOG").Stock, np.log1p(out.Stock))

    def test_simulation_mean_and_pca_dimension(self):
        cov = np.diag([4., .01])
        self.assertEqual(pca_root(cov, .99).shape, (2, 1))
        draws = simulate_normal(cov, n=50000, mean=[2.,3.], seed=87)
        np.testing.assert_allclose(draws.mean(axis=0), [2.,3.], atol=.03)
        np.testing.assert_allclose(np.diag(np.cov(draws,rowvar=False)), np.diag(cov), rtol=.03)

    def test_nig_moments_reproduce_sample_moments(self):
        x = stats.norminvgauss.rvs(3., -.6, loc=.02, scale=.06, size=10000,
                                  random_state=np.random.default_rng(82))
        fit = fit_nig_moments(x)
        d = stats.norminvgauss(fit["alpha"] * fit["delta"], fit["beta"] * fit["delta"],
                              loc=fit["mu"], scale=fit["delta"])
        np.testing.assert_allclose(d.stats(moments="mvsk"),
            [x.mean(),x.var(ddof=1),stats.skew(x),stats.kurtosis(x)], rtol=1e-10)
        with self.assertRaises(ValueError):
            fit_nig_moments(np.arange(10.))


if __name__ == "__main__":
    unittest.main()
