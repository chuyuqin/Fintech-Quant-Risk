# Contents

- `Assignment1_Responses.pdf` -- the written responses (predict / fit / reconcile for each problem).
- `src/` -- one Julia script per problem (`problem1.jl` ... `problem5.jl`), plus `utils.jl`
  with a small shared `FittedModel`/`aicc` helper (same convention as `library/fitted_model.jl`
  elsewhere in the class repo).
- `figures/` -- created by the scripts on first run; holds every plot referenced in the PDF.

# How to run it

From this folder:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'   # first time only, installs the packages listed in Project.toml
julia --project=. src/problem1.jl
julia --project=. src/problem2.jl
julia --project=. src/problem3.jl
julia --project=. src/problem4.jl
julia --project=. src/problem5.jl
```

Each script reads its CSV directly from this folder, prints the numbers used in the write-up to the
console, and writes its figure(s) to `figures/`. They are independent of each other and can
be run in any order.

# Notes on conventions chosen

- **Problem 1**: `StatsBase.skewness` and `StatsBase.kurtosis` are the bias-corrected
  estimators from Week 1, sec. 4.3, and `kurtosis` returns *excess* kurtosis (kurtosis - 3).
  `Statistics.var`/`std` use the `n-1` divisor.
- **Problem 2**: the Normal-error MLE is not run through a separate optimizer -- Week 2,
  sec. 4.3 shows the Normal-MLE beta is algebraically identical to the OLS beta, so
  `problem2.jl` reuses the OLS fit and only recomputes sigma with the MLE (n, not n-p)
  divisor. The Student's t fit uses the same JuMP + Ipopt MLE pattern as
  `library/fitted_model.jl::fit_regression_t`.
- **Problem 5**: AR(1)-AR(3) and MA(1)-MA(3) are fit by conditional sum of squares (CSS)
  rather than through `StateSpaceModels.jl`'s `SARIMA`, which is what the Week 2 demo script
  (`Week02/week02.jl`) uses for its own AR(1)/MA(1) examples. CSS was chosen deliberately:
  it is a short, self-contained routine (see the comments in `problem5.jl`) whose parameter
  count going into AICc I can state and defend line by line, whereas matching `SARIMA`'s
  internal (Kalman-filter) parameter count to this course's own `aicc`/`FittedModel`
  convention is not something I could verify without a working Julia install on hand while
  writing this. Every model is fit on the same zero-padded, length-500 residual series, so
  the six AICc numbers reported are on equal footing with each other, which is what the
  problem's model-selection comparison actually needs. This choice is explained again,
  briefly, in the PDF itself ("A Note on Method").