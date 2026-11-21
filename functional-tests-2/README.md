# Functional Tests 2

Chuyu Qin — FinTech 545

Cases **8.1–13.9** (25 tests). Cases 1–7.6 are in `functional-tests-1`.

## Run

From the repository root:

```bash
cd functional-tests-2
python -m pip install -r requirements.txt
python run_tests.py
```

The script prints a PASS/FAIL for each case and exits with status 1 if any case
fails. All input files are included; no paths need to be changed. The first run
compiles the American-option functions with Numba.

For a separate environment, run `python -m venv .venv` and activate it before
installing packages (`source .venv/bin/activate` on macOS/Linux,
`.venv\Scripts\activate` on Windows). Use Python 3.11 or 3.12.

## Results

**25/25 course tests passed.**

- [Case summary](results/test_summary.csv): status, calculated output file,
  reference file, and maximum absolute error for each case.
- [Full comparison report](results/test_report.json): errors and tolerances.
- `results/actual_<case>.csv`: calculated results, with the same columns and
  row order as the corresponding reference CSV.

| Cases | Calculation | Implementation |
| --- | --- | --- |
| 8.1–8.6 | Normal/t VaR and ES | `risk_management/advanced.py` |
| 9.1 | Gaussian copula portfolio risk | `risk_management/advanced.py` |
| 10.1–10.4 | Risk parity and maximum Sharpe weights | `risk_management/advanced.py` |
| 11.1–11.2 | Return and volatility attribution | `risk_management/advanced.py` |
| 12.1–12.3 | European/American option values and Greeks | `risk_management/options.py` |
| 13.1–13.9 | Multivariate t and t copula | `risk_management/advanced.py` |

## Numerical conventions

Monte Carlo cases use fixed Python seeds, so their draws differ from Julia's.
Cases 8.3/8.6 use six combined sampling standard errors; cases 9.1/13.9 use a
4% relative tolerance. Deterministic cases use tighter tolerances recorded in
the report. [SOURCES.md](SOURCES.md) explains the methods and conventions,
including the American-option Theta and rho definitions in the course code.

## Additional checks

```bash
python -m unittest discover -s tests -v
```

Nine checks cover risk translation/scaling, risk budgets, maximum Sharpe
weights, attribution sums, option parity and Greeks, early exercise, copula
margins, and multivariate-t scale. All nine passed.

Tested with Python 3.12, NumPy 1.26.4, SciPy 1.17.1, pandas 3.0.1, and Numba
0.64.0. `requirements.txt` records these package versions.

To use another set of course CSVs or save results elsewhere:

```bash
python run_tests.py --data-dir /path/to/testfiles/data --output-dir results
```
