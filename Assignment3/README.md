# Assignment 3 - Options, Option Risk, and Portfolio Construction

The written submission is **Assignment3_Responses.pdf** (15 pages). It answers
all five problems, explicitly labels the lettered subparts, and preserves the
Predict / Fit / Reconcile structure used in Assignment 2. The `Assignment3`
folder contains all code, unchanged input data, full-precision results, and plots.

## Reproduce the submission

Python 3.10 or newer. From this repository's `Assignment3` folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python preview.py > results/predict_diagnostics.txt
python run_analysis.py
python -m unittest discover -s tests -v
python build_report.py
```

Windows virtual-environment activation: `.venv\Scripts\activate`.
The package manager needs PyPI access unless dependencies are installed already;
all analysis and report generation run **offline** after installation. The PDF
builder uses ReportLab, so neither Julia, Quarto, nor LaTeX is needed. Run the
commands from this folder so the local `quant_risk` package is on the import path.
If a restricted environment cannot write its matplotlib cache, set
`MPLCONFIGDIR` to a writable temporary folder before running the analysis.

The first simulation/plot run may initialize a font cache. The numerical analysis
is typically under a minute after that; exact timing is machine dependent.
Tree storage grows with N, so the code evaluates increasing grids sequentially.
The report builder reads the saved results and does not refit anything. Its
written interpretation is specific to the supplied data: review that narrative
if changing inputs, rather than assuming a regenerated PDF adapts every sentence.

## File map

- `predictions.md`: pre-fit expectations, including uncertain predictions that
  the fitted results subsequently contradict.
- `preview.py`: parity, initial book Greeks, asset moments, preliminary CAPM,
  and CAPM residual correlations; no IV, distribution, or optimizer fitting.
- `quant_risk/options.py`: own generalized BSM, analytic Greeks, numerical IV
  inversion, and non-recombining cash-dividend CRR tree.
- `quant_risk/stats.py`: moments, loss-positive risk, exact fractional empirical
  ES, matched-normal risk, OLS inference and AICc.
- `run_analysis.py`: the five analyses, CSVs, data hashes, versions, and plots.
- `results/*.csv`, `results/summary.json`: every numerical result at full precision.
- `results/p3_mc_uncertainty.csv`: simulation standard errors by book/model/tail.
- `results/p4_bootstrap_draws.csv`: all 500 row-bootstrap weight draws.
- `figures/`: all figures, including a supplementary American convergence plot.
- `build_report.py`: editable narrative and PDF layout driven by the saved CSVs.
- `tests/test_methods.py`: independent binary-tree reference; pricing parity,
  analytic-vs-numerical Greeks, tree limit, invalid-quote handling, fractional
  tail integration, portfolio constraints, and OLS mean-accounting checks.
- `data/SHA256SUMS.txt`: integrity hashes of unchanged course data.
- `environment_versions.json`: actual analysis versions; report builder's
  ReportLab version is recorded in `report_environment.json`.

## Conventions that affect grading and numerical comparisons

### General

- Returns in code are decimals. Variance uses n-1; empirical-moment skewness and
  excess kurtosis use n in their central moments (normal excess kurtosis = 0).
- VaR is minus the left-inverse empirical P&L quantile. ES averages exactly alpha
  probability mass, using fractional threshold weight when n*alpha is noninteger.
  No clipping of negative risk estimates. The 1% historical ES in Problem 4
  averages 2.4 observations: two whole months plus 40% of the third.
- AICc includes every estimated likelihood parameter, including Gaussian
  regression residual variance. All regressions for a given fund use identical
  dates. OLS standard errors are conventional homoskedastic standard errors.
- Reported decimals are rounded; CSVs are authoritative for full precision.

### Problem 1

- T = 35/365 calendar years. Carry b = r-q. Quotes are per one-share option.
- Parity flags use the problem's **sum of full spreads**, not half that sum.
- No invalid quote is silently repaired. Bids below the zero-vol lower bound
  have no IV. Every mid and ask has an IV in this dataset.
- Vega is dollars per unit volatility (+1.00), not per 1 percentage point. The
  local vol-point change per cent of price is 1/vega. Vega is evaluated at each
  call/put's own mid IV, including the dirty quotes.

### Problem 2

- Baseline N = 200; convergence includes 50/55 through 400/405. Dividend dates
  must fall exactly on the CRR grid; all submitted N are multiples of five.
- CRR trees recombine **between** cash-dividend events. Each entering dividend
  node spawns its own shifted subtree, so there is no artificial global
  recombination or interpolation. Exercise is allowed both before and after the
  stock jump. Post-dividend immediate exercise matters for puts.
- Stock after a cash dividend is max(S-D,0); zero is absorbing. Extreme nodes
  where this convention could matter have negligible probability here; European
  dividend put-call parity agrees to the tested tolerance.
- No-dividend Europeans use GBSM, with tree values also supplied to calculate
  matched-grid exercise premiums. Spot-minus-PV-dividends GBSM is labeled an
  approximation, not the discrete-dividend tree model.
- Central spot bumps: $1, $0.1, $0.01, $0.001. Dividend-amount bumps: $0.001.
  Gamma on a fixed tree need not converge as the bump shrinks.
- The second-dividend exercise boundary uses the 40-step final subtree
  consistent with the N=200 tree and is checked against smooth BSM continuation.

### Problem 3

- S0 = 99.12; last CSV observation. 255 trading days/year; maturity 30/255 to
  20/255. Holdings already count one-share options; no contract multiplier.
- Delta, gamma, and theta aggregate holdings. Theta is dollars per year of
  elapsed time; use theta*10/255. The payoff plots show fixed holdings with no
  interim rehedging or additional cash-account financing.
- Fit normal and Student t by MLE with free location. For forecasting set each
  daily return mean to zero by removing fitted location, keeping fitted scale.
  Student t scale is not its sd. Delta-normal uses sample n-1 daily sd.
- 500,000 iid paths, ten daily arithmetic returns compounded multiplicatively;
  seed 5453026. Both fitted distributions use common underlying uniforms.
  Any daily simulated return <= -100% raises an error; none occurs in this run.
  Both fitted families have theoretical support below -100%, an acknowledged
  arithmetic-return approximation, rather than a hidden clipping convention.
- Full revaluation holds each IV fixed and reduces time to expiry by ten days.
  Delta-gamma-theta uses identical paths. Batch-based MC standard errors measure
  simulation uncertainty conditional on the fits, not estimation uncertainty.

### Problem 4

- Monthly RF = 0.03/12. Sharpe annualization sqrt(12), mean annualization 12.
- Normal VaR/ES use the portfolio's observed monthly mean (not zero) and sd.
- Unconstrained means fully invested risky assets with shorts allowed; the
  normalized inverse-covariance tangency solution is used. The denominator is
  positive for this data. Long-only portfolios use sum-to-one constraints.
- Frontier is long-only; CML uses its tangency portfolio and the risk-free asset.
- Downside CAPM refits both intercept and beta on EQ excess return < -5%.
- Mean SE assumes independent months. The 500-draw row bootstrap (seed 5454026)
  resamples all four returns together. Bootstrap weight ranges are sensitivity
  diagnostics, not claims about a known true optimum or future crash frequency.

### Problem 5

- Date joins are one-to-one and cover all 1,257 fund dates. Percentage factor
  files are divided by 100 once, including RF; fund returns are already decimals.
- Use excess fund returns and already-excess Mkt-RF. Annualization is 255.
- Latest decade is (2014-09-30, 2024-09-30], ending at the factor files' last date,
  not the fund sample end. It has 2,516 joint daily factor observations. These
  later data are used only for the requested expected-return exercise.
- Expected total return includes average RF. Add alpha separately and explicitly.
  Realized arithmetic and geometric annual returns are labeled separately.
- Worst momentum day is searched over the complete factor file (1939-09-05,
  -18.44%). Stress isolates beta_Mom times the shock; other factors, alpha, RF,
  and residual noise are held flat. Loss is normalized by the fund's sample
  **total-return** daily sd on the same $10,000,000 exposure.
- OLS mean accounting decomposes the CAPM-to-four-factor alpha change, including
  the change in market beta, not only the newly added factor contributions.

## Submission

This folder is prepared in your own repository. The class repository is only an
input source. No Git commit, remote push, or Canvas submission is performed by
these scripts. The assignment's separate Canvas instruction is to respond `done`
after you have submitted the repository work.
