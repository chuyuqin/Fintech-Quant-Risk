# Sources, conventions and validation

## Attribution

Inputs and expected CSVs are copied unmodified from the instructor's
`FinTech-545-Fall2026/testfiles/data`, as available on November 20, 2026.
`data_manifest.json` records their SHA-256 hashes. The reference generation
specification is `testfiles/test_setup.jl` and its `library/` dependencies:

| Python implementation | Instructor reference |
| --- | --- |
| `advanced.py`: risk measures, aggregation | `RiskStats.jl` |
| `advanced.py`: risk budgets, maximum Sharpe | `optimizers.jl` |
| `advanced.py`: attribution | `expost_factor.jl` |
| `advanced.py`: multivariate t | `multivariate_t.jl` |
| `advanced.py`: copulas | `copula.jl` |
| `options.py` | `gbsm.jl`, `bt_american.jl` |
| retained covariance/model/simulation/return modules | prior `functional-tests-1` submission |

The Python implementations follow these algorithms using NumPy and SciPy.
Numba compiles the recursive cash-dividend binomial tree. Expected-output
CSVs are read by `run_tests.py` for comparison; the functions in
`risk_management/` use only their arguments.

## Conventions

- Risk is expressed as positive loss: VaR = negative lower return quantile,
  ES = negative lower-tail mean. Both may be negative for sufficiently positive
  returns. “Diff from Mean” centers returns, using the sample mean in simulation.
- Sample VaR follows Julia's average of the floor/ceiling alpha*n order
  statistics (one-based); ES includes observations at or below that cutoff.
- Normal standard deviation uses ddof=1. The t sigma is scale, not standard
  deviation. The prior generalized-t MLE implementation is reused.
- Test 9 uses a Normal margin for A, a t margin for B and Spearman correlation
  of fitted probability transforms, as in the course generator.
- Test 10 uses the supplied covariance directly, including test5_3 for Sharpe;
  the covariance is not repaired before optimization.
- Test 11 drifts weights through buy-and-hold returns, uses Carino linking for
  return attribution and regression slopes for component standard deviations.
- European Theta is calendar-time decay; European rho changes r and b together
  (dividend yield fixed). American rho changes r with b fixed, and American
  “Theta” is +dV/d(time to maturity), matching test_setup.jl. American gamma
  uses a stock-price bump of 1.5, and other Greeks use central differences.
- Continuous-dividend American trees use 500 steps. Discrete-dividend trees
  use two steps per day and recursive repricing after each cash dividend.
  Option time uses the input DayPerYear, not a hard-coded annualization.
- Multivariate t uses sample means and sample SDs, Kendall sin(pi*tau/2)
  correlation and exactly the instructor's two 200-point reciprocal-nu grids.
  S is scale: covariance = nu/(nu-2)*S.
- Copulas use generalized-t margin probability transforms. Correlation is
  fixed by Kendall ranks. The Gaussian copula counts k=0, t k=1 in the
  course-specific AICc/BIC comparison; R and marginal parameters are excluded
  from both parameter counts, as in the reference generator.
- Copula t simulation uses one common chi-square shock per entire asset vector.

## Tolerances and independent checks

The JSON report records tolerances and errors per case. Default deterministic
comparison is rtol=1e-6, atol=1e-8. t analytical risks use rtol=1e-4 for optimizer
variation; copula likelihood/selection/tail cases use rtol=1e-5. Optimizer
weights use atol=2e-5. American Greeks use rtol=atol=2e-5 for finite differences.

For 8.3 and 8.6, each language has an independent 10,000-observation sample.
Asymptotic standard errors are obtained by integrating squared influence
functions against the fitted t density. Quantile influence is
(alpha-I[X<=q])/f(q); lower-tail-mean influence is
(X-q)*I[X<=q]/alpha+q+ES. Mean-centered statistics subtract X-mu from these
influence functions (overall sign has no effect on variance). The limit for
comparing independent outputs is six times sqrt(2) times the standard error.
The report records the separate errors and limits for absolute and centered
risk. Python seeds are fixed for reproducibility.

The 100,000-draw portfolio cases 9.1/13.9 use rtol=.04, atol=.0001 for both
dollar and percentage risk. This is a relative-error comparison between independent Monte Carlo samples. Independent
checks validate copula marginal moments, common-tail behavior, and risk
translation/scaling. Additional synthetic checks cover risk budgets, a known
maximum-Sharpe solution, attribution sums, put-call parity, option delta,
American early exercise, no-dividend call convergence, and t scale/covariance.

Reference files are preserved; the runner never regenerates instructor answers.
