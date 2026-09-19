## Run

Use Python 3.10 or newer. From this folder:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_tests.py
python -m unittest discover -s tests -v
```

On Windows, activate with `.venv\Scripts\activate` instead. An existing Python
environment with NumPy, SciPy, and pandas also works; installation needs internet
access. The code and CSVs run locally without internet once dependencies exist.

`run_tests.py` prints a separate PASS/FAIL for every course case, exits nonzero
if any fail, and writes 25 calculated CSVs plus `results/test_report.json`.
The checked-in results were obtained with NumPy 1.26.4, SciPy 1.17.1, and pandas
3.0.1. All 25 cases passed. The six independent-input checks also passed.

Optional paths:

```bash
python run_tests.py --data-dir /path/to/testfiles/data --output-dir results
```

## Contents and coverage

| Cases | Implementation | Count |
| --- | --- | ---: |
| 1.1–1.4 | Missing-data covariance/correlation; complete rows and pairwise | 4 |
| 2.1–2.3 | EW covariance, EW correlation, mixed-lambda covariance | 3 |
| 3.1–3.4 | near-PSD and Higham repairs for covariance/correlation | 4 |
| 4.1 | PSD Cholesky | 1 |
| 5.1–5.5 | Normal simulation, matrix repairs, 99% PCA | 5 |
| 6.1–6.2 | Arithmetic and logarithmic returns, retaining dates | 2 |
| 7.1–7.4 | Normal fit, t fit, t regression, AICc | 4 |
| 7.5–7.6 | NIG moments and maximum likelihood | 2 |

- `risk_management/`: reusable functions, independent of test/reference files.
- `run_tests.py`: all assigned cases using the supplied inputs and answers.
- `data/`: original instructor reference CSVs.
- `results/`: calculated outputs and numerical comparison report.
- `tests/`: additional mathematical and edge-case checks.
- `SOURCES.md`: attribution, reference mapping, conventions and tolerances.

Example reuse:

```python
import pandas as pd
from risk_management import ew_cov, fit_t, simulate_normal

x = pd.read_csv("data/test2.csv").to_numpy()
cov = ew_cov(x, lam=0.97)
draws = simulate_normal(cov, n=100000, seed=1234)
model = fit_t(pd.read_csv("data/test7_2.csv").iloc[:, 0])
print(model)  # mu, sigma (t scale), nu
```

## Why NIG 7.5 and 7.6 differ

Both fit the same 1,000 observations, returning the instructor's
`mu, alpha, beta, delta` parameterization. Approximate calculated results:

| Method | mu | alpha | beta | delta |
| --- | ---: | ---: | ---: | ---: |
| Moments (7.5) | 0.0239 | 52.947 | -12.245 | 0.0575 |
| MLE (7.6) | 0.0197 | 46.924 | -8.233 | 0.0526 |

The method of moments solves for parameters reproducing the sample mean,
sample variance, skewness, and excess kurtosis. Maximum likelihood instead
maximizes the joint density of all observations. In a finite sample those
objectives generally yield different parameters. MLE achieves the higher
log likelihood (about 1974.54 versus 1974.11); the moments fit matches the sample
skewness and excess kurtosis exactly. Neither objective requires identical
parameters or exact recovery of the distribution that generated the sample.