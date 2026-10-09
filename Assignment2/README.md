# Assignment 2 - Covariance, VaR, and Copulas

The written submission is **Assignment2_Responses.pdf**. All five problems include
Predict, Fit, and Reconcile responses. All data and code needed to reproduce the
submission are in this folder.

## Run everything (Python 3.10 or newer)

From this repository's `Assignment2` directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python run_analysis.py
python -m unittest discover -s tests -v
python build_report.py
```

Windows activation: `.venv\Scripts\activate`. Package installation needs access to
PyPI unless dependencies are already installed. The analysis itself runs offline.
The copula fitting step can take a minute. The PDF builder uses ReportLab and does
not need LaTeX. It reads the generated CSVs and JSON; rerun the analysis first.
The narrative is specific to the supplied data, so changing data also requires
reviewing the written interpretation.

## Python package

`quant_risk/` contains the reusable numerical methods. `pyproject.toml` defines
its dependencies and installation configuration. After installation, the package
can be imported from other Python scripts:

```python
from quant_risk import historical_risk, near_psd, fit_margin
risk = historical_risk([-100, -20, 10, 30], alpha=0.25)
print(risk)  # {'VaR': 100.0, 'ES': 100.0}
```

The distribution name is `fintech-quant-risk`; the import name is `quant_risk`.
The covariance module adapts
`functional-tests-1/risk_management/covariance.py` from this repository.

## File map

- `data/problem1.csv` through `problem5.csv`: unchanged copies of course data.
- `predictions.md`: predictions recorded from data inspection before model fitting.
- `quant_risk/core.py`: moments, EW weights, historical VaR/ES, MLE, AICc/BIC.
- `quant_risk/covariance.py`: covariance estimation and PSD repair.
- `quant_risk/copulas.py`: Kendall correlation, copula likelihoods/fits/simulation.
- `run_analysis.py`: calls the library to solve the five problems.
- `results/*.csv` and `results/summary.json`: full-precision numerical results.
- `results/p4_daily_loglik.csv`: all 1,000 daily likelihood contributions for 4(i).
- `figures/`: the return, P&L, and rank plots used in the report.
- `build_report.py`: editable report narrative, layout, and generated tables.
- `tests/test_quant_risk.py`: numerical invariants and independent cross-checks.
- `environment_versions.json`: versions used for this submitted computation.
- `data/SHA256SUMS.txt`: data integrity hashes.

## Numerical conventions

- Arithmetic returns; n-1 sample variance; empirical-moment skewness and excess
  kurtosis. Normal MLE scale uses n, as required by maximum likelihood.
- Historical VaR is minus the left-inverse empirical P&L quantile. Negative VaR
  means a gain at the chosen quantile and is retained, particularly in Problem 3.
  ES integrates exactly alpha probability mass, including fractional boundary
  weight. All requested n*alpha values are integers and there are no threshold
  ties in the submitted data, matching the course order-statistic convention.
- EW covariance uses normalized finite-history weights, weighted-mean centering,
  and no Bessel correction, matching the course `ewCovar`. Newest return is last.
- The Problem 2 visual regime split is at return day 460 (final 40 days).
- Normal and Student t marginal parameter counts are 2 and 3. The t scale is not
  its standard deviation. X1's t MLE reaches the normal limit; its enormous
  estimated degrees of freedom should not be interpreted literally.
- Copula AICc/BIC condition on fitted margins and Kendall R, using k=0 and k=1 as
  in `Week05/choosing_between_copulas.jl`. Report also discusses counting R.
- Copula nu is profiled over [2,100], matching the course. Observed tail membership
  uses ranks/(n+1); simulated tails use the model's uniform percentiles.
- Simulations use 100,000 days, seed 5452026 for copulas and 5452031 for regressions.
  Gaussian and t copulas share Gaussian innovations. Counts and risks are still
  Monte Carlo estimates; last digits can vary with dependency versions.
- Problem 5 reports OLS intercepts but excludes them from simulation under the
  required zero-expected-return assumption. Both residual and stock sample
  covariance use n-1 so the OLS covariance identity holds exactly. The usual n-2
  regression residual standard errors are separately reported.
- Higham's tiny negative eigenvalue (~2e-13) is convergence tolerance, not an
  economically indefinite result. Ordinary Cholesky can fail on PSD boundary
  matrices; simulation should use a PSD eigen-root if a matrix is singular.
