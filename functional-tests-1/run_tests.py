from pathlib import Path
import argparse
import json
import sys
import numpy as np
import pandas as pd
from risk_management import *

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []

    def read(name):
        return pd.read_csv(args.data_dir / name)

    def matrix(name):
        return read(name).to_numpy(dtype=float)

    def check(case, actual, expected_name, rtol=1e-7, atol=1e-9, target=None):
        expected = read(expected_name)
        if not isinstance(actual, pd.DataFrame):
            actual = pd.DataFrame(np.atleast_2d(actual), columns=expected.columns)
        actual.to_csv(args.output_dir / f"actual_{case}.csv", index=False)
        assert actual.shape == expected.shape, "Output shape differs"
        assert list(actual.columns) == list(expected.columns), "Output columns differ"
        numeric = expected.select_dtypes(include="number").columns
        for col in expected.columns.difference(numeric):
            assert actual[col].astype(str).equals(expected[col].astype(str)), f"Mismatch: {col}"
        a, e = actual[numeric].to_numpy(), expected[numeric].to_numpy()
        record = {"test":case, "expected_file":expected_name}
        if target is None:
            record.update(max_absolute_error=float(np.max(np.abs(a - e))), rtol=rtol, atol=atol)
            np.testing.assert_allclose(a, e, rtol=rtol, atol=atol)
        else:
            # Independent Monte Carlo draws differ from Julia's. Compare global
            # Frobenius error AND entrywise errors in units of sampling SE.
            target = np.asarray(target)
            se = np.sqrt((target ** 2 + np.outer(np.diag(target), np.diag(target))) / 99999)
            reference_relative = np.linalg.norm(a - e) / np.linalg.norm(e)
            target_relative = np.linalg.norm(a - target) / np.linalg.norm(target)
            target_z = np.max(np.abs(a - target) / np.maximum(se, 1e-15))
            reference_z = np.max(np.abs(a - e) / np.maximum(np.sqrt(2) * se, 1e-15))
            record.update(reference_relative_frobenius=float(reference_relative),
                          target_relative_frobenius=float(target_relative),
                          max_target_standard_errors=float(target_z),
                          max_reference_standard_errors=float(reference_z),
                          relative_frobenius_limit=0.03, standard_error_limit=6)
            assert reference_relative < .03 and target_relative < .03, "Covariance Frobenius error exceeds 3%"
            assert target_z < 6 and reference_z < 6, "Covariance exceeds six sampling standard errors"
        records.append(dict(record, status="PASS"))

    def run(case, function):
        try:
            function()
            print(f"PASS {case}")
        except Exception as exc:
            records.append(dict(test=case, status="FAIL", error=str(exc)))
            print(f"FAIL {case}: {exc}")

    for case, skip, corr in [("1.1",True,False),("1.2",True,True),("1.3",False,False),("1.4",False,True)]:
        run(case, lambda case=case, skip=skip, corr=corr: check(case,
            missing_cov(matrix("test1.csv"), skip, corr), f"testout_{case}.csv"))
    x = matrix("test2.csv")
    run("2.1", lambda: check("2.1", ew_cov(x, .97), "testout_2.1.csv"))
    run("2.2", lambda: check("2.2", correlation(ew_cov(x, .94)), "testout_2.2.csv"))
    def combined():
        sd = np.sqrt(np.diag(ew_cov(x, .97)))
        check("2.3", correlation(ew_cov(x, .94)) * np.outer(sd, sd), "testout_2.3.csv")
    run("2.3", combined)
    for case, source, method in [("3.1","1.3",near_psd),("3.2","1.4",near_psd),
                                 ("3.3","1.3",higham_psd),("3.4","1.4",higham_psd)]:
        run(case, lambda case=case, source=source, method=method: check(case,
            method(matrix(f"testout_{source}.csv")), f"testout_{case}.csv"))
    run("4.1", lambda: check("4.1", chol_psd(matrix("testout_3.1.csv")), "testout_4.1.csv"))

    def simulation(case, filename, repair=None, pca=False):
        cov = matrix(filename)
        target = repair(cov) if repair else cov
        if pca:
            root = pca_root(cov, .99)
            target = root @ root.T
            draws = simulate_pca(cov, explained=.99)
        else:
            draws = simulate_normal(cov, fix_method=repair or near_psd)
        assert draws.shape == (100000, len(cov))
        assert np.all(np.abs(draws.mean(axis=0)) < 6 * np.sqrt(np.diag(target) / 100000))
        check(case, np.cov(draws, rowvar=False), f"testout_{case}.csv", target=target)
    for case, filename, repair, pca in [
        ("5.1","test5_1.csv",None,False), ("5.2","test5_2.csv",None,False),
        ("5.3","test5_3.csv",near_psd,False), ("5.4","test5_3.csv",higham_psd,False),
        ("5.5","test5_2.csv",None,True)]:
        run(case, lambda c=case,f=filename,r=repair,p=pca: simulation(c,f,r,p))
    for case, method in [("6.1","DISCRETE"),("6.2","LOG")]:
        run(case, lambda c=case,m=method: check(c, calculate_returns(read("test6.csv"),m),
                                               f"testout{c.replace('.', '_')}.csv"))
    run("7.1", lambda: check("7.1", pd.DataFrame([fit_normal(matrix("test7_1.csv").ravel())]), "testout7_1.csv"))
    run("7.2", lambda: check("7.2", pd.DataFrame([fit_t(matrix("test7_2.csv").ravel())]),
                            "testout7_2.csv", rtol=1e-4, atol=1e-7))
    def regression():
        data = read("test7_3.csv")
        fit = fit_regression_t(data.y, data.drop(columns="y"))
        beta = fit.pop("beta")
        fit.update(Alpha=beta[0], B1=beta[1], B2=beta[2], B3=beta[3])
        check("7.3", pd.DataFrame([fit]), "testout7_3.csv", rtol=1e-4, atol=1e-7)
    run("7.3", regression)
    def information_criterion():
        x = matrix("test7_2.csv").ravel()
        check("7.4", [[aicc(fit_t(x), x)]], "testout7_4.csv", rtol=1e-7)
    run("7.4", information_criterion)
    run("7.5", lambda: check("7.5", pd.DataFrame([fit_nig_moments(matrix("test7_5.csv").ravel())]),
                            "testout7_5.csv"))
    run("7.6", lambda: check("7.6", pd.DataFrame([fit_nig_mle(matrix("test7_5.csv").ravel())]),
                            "testout7_6.csv", rtol=1e-4, atol=1e-7))
    passed = sum(r["status"] == "PASS" for r in records)
    import scipy
    report = dict(passed=passed, total=25, numpy=np.__version__, scipy=scipy.__version__,
                  pandas=pd.__version__, tests=records)
    (args.output_dir / "test_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"\n{passed}/25 passed. Actual outputs and report: {args.output_dir}")
    return 0 if passed == 25 else 1


if __name__ == "__main__":
    sys.exit(main())
