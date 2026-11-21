"""Independent checks with synthetic inputs; no instructor answers are loaded."""

import unittest
import numpy as np
import pandas as pd
from scipy.stats import norm
from risk_management.advanced import (
    sample_risk,
    distribution_risk,
    risk_parity,
    max_sharpe,
    attribution,
    kendall_correlation,
    simulate_copula,
    tail_dependence,
    fit_multivariate_t,
)
from risk_management.options import gbsm, american, american_discrete


class Invariants(unittest.TestCase):
    def test_risk_translation_and_scale(self):
        x = np.linspace(-3, 4, 1000)
        np.testing.assert_allclose(sample_risk(2 * x + 5), 2 * sample_risk(x) - 5)
        np.testing.assert_allclose(
            distribution_risk(2, 3), 3 * distribution_risk(0, 1) - 2
        )

    def test_normal_es_closed_form(self):
        self.assertAlmostEqual(
            distribution_risk(0, 1)[1], norm.pdf(norm.ppf(0.05)) / 0.05
        )

    def test_risk_budget(self):
        cov = np.diag([0.01, 0.04, 0.09])
        budget = np.array([1, 2, 3])
        w = risk_parity(cov, budget)
        contributions = w * (cov @ w)
        np.testing.assert_allclose(
            contributions / contributions.sum(), budget / budget.sum(), atol=1e-6
        )
        self.assertAlmostEqual(w.sum(), 1)

    def test_sharpe_solution(self):
        cov = np.diag([0.02, 0.04, 0.03])
        excess = np.array([0.04, 0.05, 0.02])
        exact = np.linalg.solve(cov, excess)
        exact /= exact.sum()
        np.testing.assert_allclose(
            max_sharpe(cov, excess + 0.01, 0.01), exact, atol=1e-6
        )

    def test_attribution_sums(self):
        r = pd.DataFrame(
            np.random.default_rng(10).normal(0, 0.01, (100, 3)), columns=["X", "Y", "Z"]
        )
        a = attribution([0.2, 0.3, 0.5], r, r, np.eye(3))
        for i in [1, 2]:
            self.assertAlmostEqual(
                a.loc[i, ["X", "Y", "Z", "Alpha"]].sum(), a.loc[i, "Portfolio"]
            )

    def test_put_call_parity_and_greeks(self):
        s, k, t, r, b, v = 105, 100, 0.7, 0.04, 0.02, 0.25
        call = gbsm(True, s, k, t, r, b, v)
        put = gbsm(False, s, k, t, r, b, v)
        self.assertAlmostEqual(
            call[0] - put[0], s * np.exp((b - r) * t) - k * np.exp(-r * t)
        )
        h = 0.001
        delta = (
            gbsm(True, s + h, k, t, r, b, v)[0] - gbsm(True, s - h, k, t, r, b, v)[0]
        ) / (2 * h)
        self.assertAlmostEqual(call[1], delta, places=7)

    def test_american_exercise_and_no_dividend_call(self):
        self.assertGreaterEqual(
            american(False, 80.0, 100.0, 1.0, 0.05, 0.05, 0.2, 400), 20
        )
        self.assertAlmostEqual(
            american(True, 100.0, 100.0, 1.0, 0.05, 0.05, 0.2, 1000),
            gbsm(True, 100, 100, 1, 0.05, 0.05, 0.2)[0],
            delta=0.003,
        )
        self.assertAlmostEqual(
            american_discrete(
                True,
                100.0,
                100.0,
                1.0,
                0.05,
                np.array([], dtype=float),
                np.array([], dtype=np.int64),
                0.2,
                100,
            ),
            american(True, 100.0, 100.0, 1.0, 0.05, 0.05, 0.2, 100),
        )

    def test_copula_margins_and_common_tail(self):
        r = np.array([[1, 0.4], [0.4, 1]])
        u = simulate_copula(r, n=100000, nu=5)
        np.testing.assert_allclose(u.mean(axis=0), 0.5, atol=0.004)
        np.testing.assert_allclose(u.var(axis=0), 1 / 12, atol=0.001)
        self.assertGreater(tail_dependence(0.4, 5), 0)
        self.assertGreater(tail_dependence(0.8, 5), tail_dependence(0.4, 5))

    def test_multivariate_scale_is_not_covariance(self):
        x = np.random.default_rng(1).normal(size=(100, 3))
        mu, s, nu, ll = fit_multivariate_t(x)
        np.testing.assert_allclose(np.diag(s) * nu / (nu - 2), x.var(axis=0, ddof=1))
        np.testing.assert_allclose(mu, x.mean(axis=0))
        self.assertGreater(np.linalg.eigvalsh(kendall_correlation(x)).min(), 0)


if __name__ == "__main__":
    unittest.main()
