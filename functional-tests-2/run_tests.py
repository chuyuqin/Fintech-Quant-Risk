"""Execute course cases 8–13, writing computed outputs and comparisons."""

from pathlib import Path
import argparse, json, sys
import numpy as np
import pandas as pd
import scipy
from scipy import stats, integrate
from risk_management.models import fit_normal, fit_t
from risk_management.advanced import *
from risk_management.options import gbsm, american_greeks, american_discrete

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    read = lambda name: pd.read_csv(args.data_dir / name)
    mat = lambda name: read(name).to_numpy(dtype=float)

    def check(case, actual, rtol=1e-6, atol=1e-8, mc=False, sampling_se=None):
        expected = read("testout" + case.replace(".", "_") + ".csv")
        if not isinstance(actual, pd.DataFrame):
            actual = pd.DataFrame(np.atleast_2d(actual), columns=expected.columns)
        assert set(actual.columns) == set(expected.columns), "Column mismatch"
        actual = actual[expected.columns]
        actual.to_csv(args.output_dir / f"actual_{case}.csv", index=False)
        assert actual.shape == expected.shape, "Shape mismatch"
        numeric = expected.select_dtypes(include="number").columns
        for col in expected.columns.difference(numeric):
            assert list(actual[col].astype(str)) == list(
                expected[col].astype(str)
            ), f"Label mismatch {col}"
        a = actual[numeric].to_numpy()
        e = expected[numeric].to_numpy()
        if sampling_se is not None:
            # Independent 10,000-draw samples: variance of their difference doubles.
            limits = 6 * np.sqrt(2) * np.asarray(sampling_se)
            assert np.all(
                abs(a - e) <= limits
            ), f"Monte Carlo error exceeds six combined standard errors: {abs(a-e)} > {limits}"
            return dict(
                max_absolute_error=float(np.max(abs(a - e))),
                comparison="six combined asymptotic sampling standard errors",
                sampling_se=list(np.asarray(sampling_se, dtype=float)),
                absolute_limits=list(limits),
            )
        np.testing.assert_allclose(a, e, rtol=rtol, atol=atol)
        return dict(
            max_absolute_error=float(np.max(abs(a - e))),
            rtol=rtol,
            atol=atol,
            comparison="independent Monte Carlo" if mc else "deterministic",
        )

    def run(case, fun, **kw):
        try:
            details = check(case, fun(), **kw)
            records.append(dict(test=case, status="PASS", **details))
            print("PASS", case, flush=True)
        except Exception as exc:
            records.append(dict(test=case, status="FAIL", error=str(exc)))
            print("FAIL", case, str(exc), flush=True)

    normal = fit_normal(mat("test7_1.csv").ravel())
    tfit = fit_t(mat("test7_2.csv").ravel())

    def analytical(model, es=False):
        result = distribution_risk(**model)
        zero = distribution_risk(**dict(model, mu=0))
        i = int(es)
        return [[result[i], zero[i]]]

    run("8.1", lambda: analytical(normal))
    run("8.2", lambda: analytical(tfit), rtol=1e-4)

    def simulation(es=False, seed=8):
        x = stats.t.ppf(
            np.random.default_rng(seed).random(10000),
            tfit["nu"],
            loc=tfit["mu"],
            scale=tfit["sigma"],
        )
        i = int(es)
        return [[sample_risk(x)[i], sample_risk(x - x.mean())[i]]]

    dist = stats.t(tfit["nu"], loc=tfit["mu"], scale=tfit["sigma"])
    alpha = 0.05
    q = dist.ppf(alpha)
    variance = dist.var()
    n = 10000

    # Influence functions for loss VaR and ES, with/without sample-mean centering.
    def se_stat(es, centered):
        er = distribution_risk(**tfit)[1]

        def influence(x):
            base = (
                (alpha - float(x <= q)) / dist.pdf(q)
                if not es
                else (x - q) * float(x <= q) / alpha + q + er
            )
            return base - (x - tfit["mu"]) if centered else base

        v = sum(
            integrate.quad(
                lambda x: influence(x) ** 2 * dist.pdf(x), a, b, epsabs=1e-10
            )[0]
            for a, b in [(-np.inf, q), (q, np.inf)]
        )
        return np.sqrt(v / n)

    var_se = [se_stat(False, c) for c in (False, True)]
    es_se = [se_stat(True, c) for c in (False, True)]
    run("8.3", simulation, sampling_se=var_se)
    run("8.4", lambda: analytical(normal, True))
    run("8.5", lambda: analytical(tfit, True), rtol=1e-4)
    run("8.6", lambda: simulation(True, 9), sampling_se=es_se)

    def gaussian_portfolio():
        x = read("test9_1_returns.csv")
        a = fit_normal(x.A)
        b = fit_t(x.B)
        u = np.column_stack(
            (
                stats.norm.cdf(x.A, loc=a["mu"], scale=a["sigma"]),
                stats.t.cdf(x.B, b["nu"], loc=b["mu"], scale=b["sigma"]),
            )
        )
        corr = stats.spearmanr(u).statistic
        sim = simulate_copula(np.array([[1, corr], [corr, 1]]), seed=9)
        ret = np.column_stack(
            (
                stats.norm.ppf(sim[:, 0], loc=a["mu"], scale=a["sigma"]),
                stats.t.ppf(sim[:, 1], b["nu"], loc=b["mu"], scale=b["sigma"]),
            )
        )
        return portfolio_risk(ret, [2000, 3000], ["A", "B"])

    run("9.1", gaussian_portfolio, rtol=0.04, atol=0.0001, mc=True)
    cov = mat("test5_2.csv")
    run("10.1", lambda: risk_parity(cov)[:, None], atol=2e-5)
    run("10.2", lambda: risk_parity(cov, [1, 1, 1, 1, 0.5])[:, None], atol=2e-5)
    cov = mat("test5_3.csv")
    means = mat("test10_3_means.csv").ravel()
    run("10.3", lambda: max_sharpe(cov, means, 0.04)[:, None], atol=2e-5)
    run(
        "10.4",
        lambda: max_sharpe(cov, means, 0.04, [(0.1, 0.5)] * 5)[:, None],
        atol=2e-5,
    )
    r = read("test11_1_returns.csv")
    run(
        "11.1",
        lambda: attribution(mat("test11_1_weights.csv").ravel(), r, r, np.eye(3)).drop(
            columns="Alpha"
        ),
    )
    run(
        "11.2",
        lambda: attribution(
            mat("test11_2_weights.csv").ravel(),
            read("test11_2_stock_returns.csv"),
            read("test11_2_factor_returns.csv"),
            read("test11_2_beta.csv").iloc[:, 1:].to_numpy(),
        ),
    )

    def options(american=False):
        rows = []
        for _, o in read("test12_1.csv").dropna(subset=["ID"]).iterrows():
            f = american_greeks if american else gbsm
            vals = f(
                o["Option Type"] == "Call",
                o.Underlying,
                o.Strike,
                o.DaysToMaturity / o.DayPerYear,
                o.RiskFreeRate,
                o.RiskFreeRate - o.DividendRate,
                o.ImpliedVol,
            )
            rows.append([o.ID, *vals])
        return pd.DataFrame(
            rows, columns=["ID", "Value", "Delta", "Gamma", "Vega", "Rho", "Theta"]
        )

    run("12.1", options)
    run("12.2", lambda: options(True), rtol=2e-5, atol=2e-5)

    def discrete():
        rows = []
        for _, o in read("test12_3.csv").dropna(subset=["ID"]).iterrows():
            value = american_discrete(
                o["Option Type"] == "Call",
                o.Underlying,
                o.Strike,
                o.DaysToMaturity / o.DayPerYear,
                o.RiskFreeRate,
                np.array([float(v) for v in o.DividendAmts.split(",")]),
                np.array([int(v) * 2 for v in o.DividendDates.split(",")]),
                o.ImpliedVol,
                int(o.DaysToMaturity * 2),
            )
            rows.append([o.ID, value])
        return pd.DataFrame(rows, columns=["ID", "Value"])

    run("12.3", discrete)
    x = mat("test13_returns.csv")
    r = kendall_correlation(x)
    mu, s, nu, ll = fit_multivariate_t(x)
    run("13.1", lambda: r)
    run("13.2", lambda: mu[:, None])
    run("13.3", lambda: s)
    run("13.4", lambda: [[nu, ll]])
    fits = [fit_t(col) for col in x.T]
    u = np.column_stack(
        [
            stats.t.cdf(col, f["nu"], loc=f["mu"], scale=f["sigma"])
            for col, f in zip(x.T, fits)
        ]
    )
    llg = copula_ll(u, r)
    nuc, llt = profile_nu(lambda nu: copula_ll(u, r, nu))
    run("13.5", lambda: [[llg]], rtol=1e-5)
    run("13.6", lambda: [[nuc, llt]], rtol=1e-5)
    criteria = pd.DataFrame(
        [
            ["Gaussian", llg, 0, -2 * llg, -2 * llg],
            ["T", llt, 1, -2 * llt + 2 + 4 / (len(x) - 2), np.log(len(x)) - 2 * llt],
        ],
        columns=["Copula", "LL", "K", "AICC", "BIC"],
    )
    run("13.7", lambda: criteria, rtol=1e-5)
    run(
        "13.8",
        lambda: pd.DataFrame(
            [
                [i + 1, j + 1, r[i, j], tail_dependence(r[i, j], nuc)]
                for i in range(5)
                for j in range(i + 1, 5)
            ],
            columns=["I", "J", "Rho", "Lambda"],
        ),
        rtol=1e-5,
    )

    def t_portfolio():
        sim = simulate_copula(r, nu=nuc, seed=13)
        ret = np.column_stack(
            [
                stats.t.ppf(sim[:, j], f["nu"], loc=f["mu"], scale=f["sigma"])
                for j, f in enumerate(fits)
            ]
        )
        port = read("test13_portfolio.csv")
        return portfolio_risk(ret, port.currentValue.to_numpy(), port.Stock)

    run("13.9", t_portfolio, rtol=0.04, atol=0.0001, mc=True)
    passed = sum(t["status"] == "PASS" for t in records)
    report = dict(
        passed=passed,
        total=25,
        numpy=np.__version__,
        scipy=scipy.__version__,
        pandas=pd.__version__,
        tests=records,
    )
    (args.output_dir / "test_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    summary = []
    for record in records:
        case = record["test"]
        summary.append(
            {
                "Case": case,
                "Status": record["status"],
                "Calculated output": f"actual_{case}.csv",
                "Reference": "testout" + case.replace(".", "_") + ".csv",
                "Max absolute error": record.get("max_absolute_error"),
                "Comparison": record.get("comparison", record.get("error", "")),
            }
        )
    pd.DataFrame(summary).to_csv(args.output_dir / "test_summary.csv", index=False)
    print(f"{passed}/25 passed")
    return int(passed != 25)


if __name__ == "__main__":
    sys.exit(main())
