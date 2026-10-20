"""Practice midterm calculations. Run: python solve_midterm.py"""
from quant_risk import (fit_multistart, fit_nig_mle, information_criteria, project_psd, valid_parameters)
from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parent
INPUT_DIR = PROJECT_DIR / "data"


import numpy as np
import pandas as pd

DATA_FILE = INPUT_DIR / "problem1.csv"

# Problem 1(a) — Pairwise-complete correlations and observation counts
if not DATA_FILE.is_file():
    raise FileNotFoundError(f"Required input file not found: {DATA_FILE}")

data = pd.read_csv(DATA_FILE)
if data.empty or data.shape[1] < 2:
    raise ValueError("problem1.csv must contain at least two numeric data columns.")

try:
    data = data.apply(pd.to_numeric, errors="raise")
except (ValueError, TypeError) as exc:
    raise ValueError(
        "problem1.csv contains nonnumeric columns. Explicitly specify the "
        "return columns before calculating; columns will not be guessed."
    ) from exc

if np.isinf(data.to_numpy(dtype=float)).any():
    raise ValueError("problem1.csv contains infinite values.")

# Input columns must contain the supplied daily arithmetic returns.
observed = data.notna().astype(np.int64)
pairwise_counts = observed.T @ observed
correlation = data.corr(method="pearson", min_periods=2)

print("Problem 1(a) — Pairwise observation counts")
print(pairwise_counts.to_string())
print("\nProblem 1(a) — Pairwise correlation matrix")
print(correlation.to_string(float_format=lambda x: f"{x:.10f}"))

# Problem 1(b) — Full available column histories, variance denominator n - 1
column_summary = pd.DataFrame(
    {
        "count": data.count(),
        "variance_ddof_1": data.var(axis=0, skipna=True, ddof=1),
    }
)

print("\nProblem 1(b) — Column counts and variances")
print(column_summary.to_string(float_format=lambda x: f"{x:.10f}"))

# Problem 1(c) — Sorted eigenvalues of the pairwise correlation matrix
if correlation.isna().any().any():
    undefined_pairs = [
        (correlation.index[i], correlation.columns[j])
        for i, j in np.argwhere(correlation.isna().to_numpy())
        if i <= j
    ]
    raise ValueError(
        "Cannot calculate correlation eigenvalues: undefined correlations "
        f"for {undefined_pairs}. Each pair requires at least two common "
        "observations and positive variance in both paired columns."
    )

eigenvalues = np.sort(
    np.linalg.eigvalsh(correlation.to_numpy(dtype=float))
)
print("\nProblem 1(c) — Correlation eigenvalues, ascending")
print(
    pd.Series(eigenvalues, name="eigenvalue").to_string(
        index=False, float_format=lambda x: f"{x:.12f}"
    )
)

# Problem 1(c) — Numerical checks (absolute tolerance only).
CHECK_TOLERANCE = 1e-10
correlation_array = correlation.to_numpy(dtype=float)
print("symmetry_max_error", np.max(np.abs(correlation_array - correlation_array.T)))
print("unit_diagonal_max_error", np.max(np.abs(np.diag(correlation_array) - 1)))
print("symmetry_check", int(np.allclose(correlation_array, correlation_array.T, atol=CHECK_TOLERANCE, rtol=0)))
print("unit_diagonal_check", int(np.allclose(np.diag(correlation_array), 1, atol=CHECK_TOLERANCE, rtol=0)))


# Problem 1(d) — Unweighted Higham projections with Dykstra correction
# Run after Problem 1(a)–(c), which define data and correlation.

if "data" not in globals() or "correlation" not in globals():
    raise RuntimeError(
        "Run Problem 1(a)–(c) first to define data and correlation."
    )

EIGENVALUE_TOL = 1e-12
OBJECTIVE_CHANGE_TOL = 1e-14
MAX_ITERATIONS = 10_000

labels = correlation.columns
if not correlation.index.equals(labels) or not data.columns.equals(labels):
    raise ValueError("Data and correlation matrix column labels must match.")

original = correlation.to_numpy(dtype=float)
if not np.isfinite(original).all():
    raise ValueError("The original correlation matrix must contain finite values.")
if not np.allclose(original, original.T, atol=EIGENVALUE_TOL, rtol=0):
    raise ValueError("The original correlation matrix must be symmetric.")


repaired = original.copy()
dykstra_correction = np.zeros_like(original)
previous_objective = 0.5 * np.sum((repaired - original) ** 2)

for iteration in range(1, MAX_ITERATIONS + 1):
    residual = repaired - dykstra_correction
    psd_projection = project_psd(residual)
    dykstra_correction = psd_projection - residual

    repaired = psd_projection.copy()
    np.fill_diagonal(repaired, 1.0)

    # Unweighted objective: one-half squared Frobenius distance.
    objective = 0.5 * np.sum((repaired - original) ** 2)
    objective_change = abs(objective - previous_objective)
    repaired_eigenvalues = np.linalg.eigvalsh(repaired)

    if (
        repaired_eigenvalues[0] >= -EIGENVALUE_TOL
        and objective_change <= OBJECTIVE_CHANGE_TOL
    ):
        break

    previous_objective = objective
else:
    raise RuntimeError(
        f"Higham projections did not converge within {MAX_ITERATIONS} iterations. "
        f"Minimum eigenvalue={repaired_eigenvalues[0]:.16e}; "
        f"objective change={objective_change:.16e}."
    )

assert np.allclose(
    np.diag(repaired), 1.0, atol=EIGENVALUE_TOL, rtol=0
), "Repaired correlation diagonal is not one."
assert np.allclose(
    repaired, repaired.T, atol=EIGENVALUE_TOL, rtol=0
), "Repaired correlation is not symmetric."
assert repaired_eigenvalues[0] >= -EIGENVALUE_TOL, (
    "Repaired correlation is not PSD within tolerance."
)

repaired_correlation = pd.DataFrame(repaired, index=labels, columns=labels)
frobenius_distance = np.linalg.norm(repaired - original, ord="fro")

row_indices, col_indices = np.triu_indices(len(labels), k=1)
changes = pd.DataFrame(
    {
        "column_1": labels.take(row_indices),
        "column_2": labels.take(col_indices),
        "original": original[row_indices, col_indices],
        "repaired": repaired[row_indices, col_indices],
        "change": (repaired - original)[row_indices, col_indices],
    }
)
changes["absolute_change"] = changes["change"].abs()
changes = changes.sort_values(
    "absolute_change", ascending=False, kind="stable"
).reset_index(drop=True)

print("Problem 1(d) — Repaired correlation matrix")
print(repaired_correlation.to_string(float_format=lambda x: f"{x:.12f}"))
print("\nProblem 1(d) — All eigenvalues, ascending")
print(
    pd.Series(repaired_eigenvalues, name="eigenvalue").to_string(
        index=False, float_format=lambda x: f"{x:.16e}"
    )
)
print(f"\nProblem 1(d) — Iterations: {iteration}")
print(f"Problem 1(d) — Frobenius distance: {frobenius_distance:.16e}")
print("\nProblem 1(d) — Off-diagonal changes, descending absolute size")
print(changes.to_string(index=False, float_format=lambda x: f"{x:.16e}"))

# Problem 1(e) — Covariance using full-history sample standard deviations
full_history_variances = data.var(axis=0, skipna=True, ddof=1)
variances = full_history_variances.to_numpy(dtype=float)

if not np.isfinite(variances).all() or np.any(variances <= 0):
    raise ValueError(
        "Each column must have a finite, positive full-history sample variance."
    )

standard_deviations = np.sqrt(variances)
D = np.diag(standard_deviations)
covariance_array = D @ repaired @ D
repaired_covariance = pd.DataFrame(
    covariance_array, index=labels, columns=labels
)

# Normalize by the largest variance for scale-aware covariance checks.
covariance_scale = float(variances.max())
scaled_covariance = covariance_array / covariance_scale

assert np.allclose(
    scaled_covariance, scaled_covariance.T,
    atol=EIGENVALUE_TOL, rtol=0
), "Repaired covariance is not symmetric."
assert np.linalg.eigvalsh(scaled_covariance)[0] >= -EIGENVALUE_TOL, (
    "Repaired covariance is not PSD within the scale-adjusted tolerance."
)
assert np.allclose(
    np.diag(covariance_array), variances, atol=0, rtol=1e-12
), "Original full-history sample variances were not preserved."

print("\nProblem 1(e) — Repaired covariance matrix")
print(repaired_covariance.to_string(float_format=lambda x: f"{x:.16e}"))


# Problem 2(a) — Arithmetic returns and OLS with an intercept

from scipy.stats import norm

DATA_FILE = INPUT_DIR / "problem2.csv"

if not DATA_FILE.is_file():
    raise FileNotFoundError(f"Required input file not found: {DATA_FILE}")

prices = pd.read_csv(DATA_FILE)
required_columns = ["SPY", "XYZ"]
missing_columns = [name for name in required_columns if name not in prices.columns]
if missing_columns:
    raise ValueError(f"problem2.csv is missing required columns: {missing_columns}")

try:
    prices = prices[required_columns].apply(pd.to_numeric, errors="raise")
except (ValueError, TypeError) as exc:
    raise ValueError("SPY and XYZ prices must be numeric.") from exc

if len(prices) < 4:
    raise ValueError("At least four price rows are required.")
if not np.isfinite(prices.to_numpy(dtype=float)).all():
    raise ValueError("SPY and XYZ prices must be finite and nonmissing.")
if (prices <= 0).any().any():
    raise ValueError("SPY and XYZ prices must be strictly positive.")

# Preserve the supplied chronological row order; do not demean returns.
returns = prices.pct_change(fill_method=None).iloc[1:]
market = returns["SPY"].to_numpy(dtype=float)
stock = returns["XYZ"].to_numpy(dtype=float)
n = len(returns)

if not np.isfinite(returns.to_numpy(dtype=float)).all():
    raise ValueError("Calculated arithmetic returns contain nonfinite values.")

design = np.column_stack((np.ones(n), market))
coefficients, _, rank, _ = np.linalg.lstsq(design, stock, rcond=None)
if rank != 2:
    raise ValueError("OLS requires a nonconstant SPY return series.")

alpha, beta = coefficients
residuals = stock - design @ coefficients

# The requested residual denominator is n - 1.
residual_variance = np.sum(residuals**2) / (n - 1)
market_variance = np.var(market, ddof=1)
market_mean = np.mean(market)

mu = alpha + beta * market_mean
sigma_squared = beta**2 * market_variance + residual_variance
sigma = np.sqrt(sigma_squared)
residual_standard_deviation = np.sqrt(residual_variance)
last_xyz_price = float(prices["XYZ"].iloc[-1])

np.testing.assert_allclose(
    mu,
    np.mean(stock),
    rtol=1e-10,
    atol=1e-14,
    err_msg="Model mu does not match the XYZ sample mean.",
)
np.testing.assert_allclose(
    sigma_squared,
    np.var(stock, ddof=1),
    rtol=1e-10,
    atol=1e-14,
    err_msg="Model variance does not match the XYZ sample variance.",
)

print("Problem 2(a) — OLS and daily return parameters")
for name, value in [
    ("alpha", alpha),
    ("beta", beta),
    ("residual standard deviation", residual_standard_deviation),
    ("mu", mu),
    ("sigma", sigma),
    ("last XYZ price", last_xyz_price),
]:
    print(f"{name}: {value:.12g}")

# Problem 2(b) — Normal percent VaR and ES
risk_rows = []
for confidence in (0.95, 0.99):
    q = 1.0 - confidence
    z = norm.ppf(q)
    percent_var = 100.0 * (-mu - sigma * z)
    percent_es = 100.0 * (-mu + sigma * norm.pdf(z) / q)
    risk_rows.append(
        {
            "confidence": confidence,
            "VaR (%)": percent_var,
            "ES (%)": percent_es,
        }
    )

print("\nProblem 2(b) — Daily percent VaR and ES")
print(
    pd.DataFrame(risk_rows).to_string(
        index=False, float_format=lambda x: f"{x:.10f}"
    )
)

problem2_stock_returns = stock.copy()


# Problem 3(a) — Stable, bounded, multiple-start NIG maximum likelihood





# Problem 3(a) — Read prices and calculate 1,000 arithmetic daily returns

from scipy import stats

DATA_FILE = INPUT_DIR / "problem3.csv"
STOCKS = [f"S{i}" for i in range(1, 11)]
PARAMETER_COUNTS = {"normal": 2, "student_t": 3, "NIG": 4}

if not DATA_FILE.is_file():
    raise FileNotFoundError(f"Required input file not found: {DATA_FILE}")

raw = pd.read_csv(DATA_FILE)
missing_columns = [column for column in STOCKS if column not in raw.columns]
if missing_columns:
    raise ValueError(f"problem3.csv is missing columns: {missing_columns}")

try:
    prices = raw[STOCKS].apply(pd.to_numeric, errors="raise")
except (ValueError, TypeError) as exc:
    raise ValueError("S1 through S10 must contain numeric prices.") from exc

if len(prices) != 1001:
    raise ValueError(
        f"Expected 1,001 price rows for 1,000 returns; found {len(prices)}."
    )
if not np.isfinite(prices.to_numpy(dtype=float)).all():
    raise ValueError("S1 through S10 contain missing or nonfinite prices.")
if (prices <= 0).any().any():
    raise ValueError("All prices must be strictly positive.")

# Preserve the supplied chronological row order without filling missing prices.
returns = prices.pct_change(fill_method=None).iloc[1:].copy()
if returns.shape != (1000, 10):
    raise ValueError(f"Unexpected return dimensions: {returns.shape}")
if not np.isfinite(returns.to_numpy(dtype=float)).all():
    raise ValueError("Calculated returns contain nonfinite values.")

# Problem 3(a) — Multiple-start maximum-likelihood fitting


def fit_stock(stock, observations):
    center = float(np.mean(observations))
    standardizer = float(np.std(observations, ddof=0))
    if not np.isfinite(standardizer) or standardizer <= 0:
        raise ValueError(f"{stock} has no positive finite return variation.")

    standardized = (observations - center) / standardizer
    median = float(np.median(standardized))

    t_starts = [
        ((df,), location, scale)
        for df in (2.5, 5.0, 10.0, 30.0)
        for location in (0.0, median)
        for scale in (0.5, 1.0)
    ]

    fitted = {
        "normal": (stats.norm, stats.norm.fit(standardized)),
        "student_t": (
            stats.t,
            fit_multistart(stats.t, "student_t", standardized, t_starts),
        ),
    }

    rows = []
    parameters_in_return_units = {}

    for model, (distribution, standardized_parameters) in fitted.items():
        shapes = tuple(standardized_parameters[:-2])
        location = center + standardizer * standardized_parameters[-2]
        scale = standardizer * standardized_parameters[-1]
        parameters = (*shapes, location, scale)

        if not valid_parameters(model, parameters):
            raise RuntimeError(f"{stock}: invalid {model} parameters.")

        log_density = distribution.logpdf(observations, *parameters)
        if not np.isfinite(log_density).all():
            raise RuntimeError(f"{stock}: nonfinite {model} log likelihood.")

        log_likelihood = float(log_density.sum())
        n = len(observations)
        k = PARAMETER_COUNTS[model]
        if n <= k + 1:
            raise ValueError(f"{stock}: insufficient observations for {model} AICc.")

        aicc = (
            2 * k
            - 2 * log_likelihood
            + 2 * k * (k + 1) / (n - k - 1)
        )
        bic = k * np.log(n) - 2 * log_likelihood

        parameters_in_return_units[model] = parameters
        rows.append(
            {
                "stock": stock,
                "model": model,
                "n": n,
                "k": k,
                "loc": location,
                "scale": scale,
                "df": shapes[0] if model == "student_t" else np.nan,
                "a": shapes[0] if model == "NIG" else np.nan,
                "b": shapes[1] if model == "NIG" else np.nan,
                "log_likelihood": log_likelihood,
                "AICc": aicc,
                "BIC": bic,
            }
        )

    nig_fit = fit_nig_mle(observations, stock)
    nig_fits[stock] = nig_fit
    fitted_nig = nig_fit["parameters"]
    parameters_in_return_units["NIG"] = tuple(fitted_nig[key] for key in ("a", "b", "loc", "scale"))
    rows.append({"stock": stock, "model": "NIG", "n": len(observations),
                 "k": 4, "df": np.nan, **fitted_nig})
    return rows, parameters_in_return_units

# Problem 3(a) — Print every candidate distribution without model selection
nig_fits = {}
all_rows = []
fitted_parameters = {}

for stock in STOCKS:
    stock_rows, stock_parameters = fit_stock(
        stock, returns[stock].to_numpy(dtype=float)
    )
    all_rows.extend(stock_rows)
    fitted_parameters[stock] = stock_parameters

fit_results = pd.DataFrame(all_rows)

# NIG uses scipy.stats.norminvgauss parameters (a, b, loc, scale).
# Shape parameters are dimensionless; loc and scale are in return units.
print("Problem 3(a) — Maximum-likelihood marginal fits")
print(
    fit_results.to_string(
        index=False,
        na_rep="—",
        float_format=lambda value: f"{value:.12g}",
    )
)




# Problem 3(a) — Information-criterion formula and parameter-count tests
from itertools import combinations


PARAMETER_COUNTS = {"normal": 2, "student_t": 3, "NIG": 4}


assert PARAMETER_COUNTS == {"normal": 2, "student_t": 3, "NIG": 4}

# Fixed test: ll=-10, k=2, n=7 gives AIC=24 and AICc=27.
np.testing.assert_allclose(
    information_criteria(ll=-10.0, k=2, n=7),
    (24.0, 27.0, 23.891820298110627),
    rtol=1e-14,
    atol=1e-14,
)

# Problem 3(a) — Validate return counts and candidate likelihood inputs
required_objects = ("prices", "returns", "fit_results", "nig_fits")
missing_objects = [name for name in required_objects if name not in globals()]
if missing_objects:
    raise RuntimeError(f"Run the preceding fitting code first: missing {missing_objects}")

stocks = [f"S{i}" for i in range(1, 11)]
models = list(PARAMETER_COUNTS)

for name, frame in (("prices", prices), ("returns", returns)):
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame.")
    missing_columns = [stock for stock in stocks if stock not in frame.columns]
    if missing_columns:
        raise ValueError(f"{name} is missing columns: {missing_columns}")
    if not np.isfinite(frame[stocks].to_numpy(dtype=float)).all():
        raise ValueError(f"{name} contains missing or nonfinite observations.")

assert len(returns) == 1000, "Expected 1,000 return observations."
assert len(prices) == len(returns) + 1, "Expected one more price row than return rows."

expected_returns = prices[stocks].pct_change(fill_method=None).iloc[1:]
np.testing.assert_allclose(
    returns[stocks].to_numpy(dtype=float),
    expected_returns.to_numpy(dtype=float),
    rtol=1e-12,
    atol=1e-14,
    err_msg="returns does not match arithmetic returns calculated from prices.",
)

required_fields = {"stock", "model", "n", "k", "log_likelihood"}
if not required_fields.issubset(fit_results.columns):
    raise ValueError(f"fit_results requires columns: {sorted(required_fields)}")

# Problem 3(a) — AIC, AICc, and BIC for every stock and candidate
rows = []
for stock in stocks:
    n = int(returns[stock].count())
    assert n == len(returns)
    assert n != int(prices[stock].count()), "n must count returns, not prices."

    for model in models:
        candidate = fit_results.loc[
            (fit_results["stock"] == stock) & (fit_results["model"] == model)
        ]
        if len(candidate) != 1:
            raise ValueError(f"Expected exactly one fit for {stock}, {model}.")

        candidate = candidate.iloc[0]
        k = PARAMETER_COUNTS[model]
        assert candidate["k"] == k, f"Incorrect parameter count: {stock}, {model}."
        assert candidate["n"] == n, f"Incorrect return count: {stock}, {model}."

        # Use the updated stable NIG helper's likelihood.
        if model == "NIG":
            if stock not in nig_fits:
                raise ValueError(f"Missing updated NIG fit for {stock}.")
            ll = float(nig_fits[stock]["parameters"]["log_likelihood"])
        else:
            ll = float(candidate["log_likelihood"])

        aic, aicc, bic = information_criteria(ll, k, n)
        np.testing.assert_allclose(
            aicc - aic,
            2.0 * k * (k + 1) / (n - k - 1),
            rtol=1e-9,
            atol=1e-10,
        )
        rows.append(
            {
                "stock": stock,
                "model": model,
                "n_returns": n,
                "k": k,
                "log_likelihood": ll,
                "AIC": aic,
                "AICc": aicc,
                "BIC": bic,
            }
        )

criteria_table = pd.DataFrame(rows)
print("Problem 3(a) — Candidate information criteria")
print(criteria_table.to_string(index=False, float_format=lambda x: f"{x:.12f}"))

# Problem 3(a) — Pairwise numerical gaps, defined as first minus second
gap_rows = []
for stock in stocks:
    stock_table = criteria_table.loc[
        criteria_table["stock"] == stock
    ].set_index("model")

    for first, second in combinations(models, 2):
        row = {"stock": stock, "first": first, "second": second}
        for criterion in ("AIC", "AICc", "BIC"):
            difference = float(
                stock_table.loc[first, criterion]
                - stock_table.loc[second, criterion]
            )
            row[f"{criterion}_first_minus_second"] = difference
            row[f"{criterion}_absolute_gap"] = abs(difference)
        gap_rows.append(row)

gap_table = pd.DataFrame(gap_rows)
print("\nProblem 3(a) — Pairwise numerical gaps")
print(gap_table.to_string(index=False, float_format=lambda x: f"{x:.12f}"))


# Problem 3(b) — Supplied margins and probability integral transforms
from scipy.linalg import solve_triangular
from scipy.optimize import minimize_scalar
from scipy.special import gammaln

required_objects = ("returns", "fitted_parameters", "nig_fits")
missing_objects = [name for name in required_objects if name not in globals()]
if missing_objects:
    raise RuntimeError(f"Missing inputs from Problem 3(a): {missing_objects}")

STOCKS = [f"S{i}" for i in range(1, 11)]
MARGIN_CHOICES = {
    "S1": "normal",
    "S2": "student_t",
    "S3": "NIG",
    "S4": "normal",
    "S5": "student_t",
    "S6": "NIG",
    "S7": "normal",
    "S8": "student_t",
    "S9": "NIG",
    "S10": "student_t",
}
DISTRIBUTIONS = {
    "normal": stats.norm,
    "student_t": stats.t,
    "NIG": stats.norminvgauss,
}

if not isinstance(returns, pd.DataFrame):
    raise TypeError("returns must be a pandas DataFrame.")
missing_columns = [stock for stock in STOCKS if stock not in returns.columns]
if missing_columns:
    raise ValueError(f"Missing return columns: {missing_columns}")

observations = returns[STOCKS].to_numpy(dtype=float)
if observations.shape != (1000, 10):
    raise ValueError(f"Expected 1,000 returns for 10 stocks; got {observations.shape}.")
if not np.isfinite(observations).all():
    raise ValueError("Returns must be finite and nonmissing.")

n, dimension = observations.shape
uniforms = np.empty_like(observations)

for column, stock in enumerate(STOCKS):
    model = MARGIN_CHOICES[stock]

    if model == "NIG":
        if stock not in nig_fits:
            raise ValueError(f"Missing updated NIG fit for {stock}.")
        parameters = nig_fits[stock]["parameters"]
        required_keys = ("a", "b", "loc", "scale")
        if any(key not in parameters for key in required_keys):
            raise ValueError(f"Incomplete NIG parameters for {stock}.")
        parameters = tuple(parameters[key] for key in required_keys)
    else:
        if stock not in fitted_parameters or model not in fitted_parameters[stock]:
            raise ValueError(f"Missing {model} fit for {stock}.")
        parameters = tuple(fitted_parameters[stock][model])

    if not np.isfinite(parameters).all() or parameters[-1] <= 0:
        raise ValueError(f"Invalid fitted parameters for {stock}.")
    if model == "student_t" and parameters[0] <= 0:
        raise ValueError(f"Invalid fitted degrees of freedom for {stock}.")
    if model == "NIG" and parameters[0] <= abs(parameters[1]):
        raise ValueError(f"Invalid fitted NIG shapes for {stock}.")

    uniforms[:, column] = DISTRIBUTIONS[model].cdf(
        observations[:, column], *parameters
    )

if not np.isfinite(uniforms).all():
    raise ValueError("Fitted marginal CDFs produced nonfinite uniforms.")
if np.any((uniforms <= 0) | (uniforms >= 1)):
    raise ValueError(
        "Fitted marginal CDFs produced values outside the open interval (0, 1). "
        "Copula likelihoods require interior uniforms; no clipping was applied."
    )

uniforms = pd.DataFrame(uniforms, index=returns.index, columns=STOCKS)

# Problem 3(b) — Kendall correlation matrix shared by both copulas
tau = returns[STOCKS].corr(method="kendall")
if not np.isfinite(tau.to_numpy()).all():
    raise ValueError("Kendall tau contains undefined values.")

R = np.sin(np.pi * tau.to_numpy(dtype=float) / 2.0)
np.fill_diagonal(R, 1.0)
R = (R + R.T) / 2.0

eigenvalues = np.linalg.eigvalsh(R)
print("Problem 3(b) — Kendall tau")
print(tau.to_string(float_format=lambda value: f"{value:.12f}"))
print("\nProblem 3(b) — Fixed copula correlation matrix R")
print(
    pd.DataFrame(R, index=STOCKS, columns=STOCKS).to_string(
        float_format=lambda value: f"{value:.12f}"
    )
)
print("\nProblem 3(b) — R eigenvalues, ascending")
print(
    pd.Series(eigenvalues).to_string(
        index=False, float_format=lambda value: f"{value:.16e}"
    )
)

if eigenvalues[0] <= 0:
    raise ValueError(
        f"R is not positive definite: minimum eigenvalue={eigenvalues[0]:.16e}."
    )
try:
    cholesky = np.linalg.cholesky(R)
except np.linalg.LinAlgError as exc:
    raise ValueError("Cholesky positive-definiteness check failed for R.") from exc

log_determinant = 2.0 * np.log(np.diag(cholesky)).sum()
U = uniforms.to_numpy(dtype=float)

def quadratic_forms(values):
    whitened = solve_triangular(
        cholesky, values.T, lower=True, check_finite=False
    )
    return np.sum(whitened**2, axis=0)

# Problem 3(b) — Gaussian copula likelihood with fixed R
normal_scores = stats.norm.ppf(U)
gaussian_log_likelihood = float(
    -0.5 * n * log_determinant
    - 0.5
    * np.sum(
        quadratic_forms(normal_scores)
        - np.sum(normal_scores**2, axis=1)
    )
)
if not np.isfinite(gaussian_log_likelihood):
    raise RuntimeError("Gaussian copula log likelihood is nonfinite.")

# Problem 3(b) — Student-t copula profile likelihood with fixed R
def t_copula_log_likelihood(nu):
    if not np.isfinite(nu) or nu <= 0:
        raise ValueError("Copula degrees of freedom must be finite and positive.")

    scores = stats.t.ppf(U, df=nu)
    if not np.isfinite(scores).all():
        raise RuntimeError(f"Nonfinite Student-t scores at nu={nu}.")

    quadratic = quadratic_forms(scores)
    log_joint = (
        gammaln((nu + dimension) / 2.0)
        - gammaln(nu / 2.0)
        - 0.5 * dimension * np.log(nu * np.pi)
        - 0.5 * log_determinant
        - 0.5 * (nu + dimension) * np.log1p(quadratic / nu)
    )
    log_marginals = stats.t.logpdf(scores, df=nu).sum(axis=1)
    log_likelihood = float(np.sum(log_joint - log_marginals))

    if not np.isfinite(log_likelihood):
        raise RuntimeError(f"Nonfinite Student-t copula likelihood at nu={nu}.")
    return log_likelihood

reciprocal_df_grid = np.linspace(0.01, 0.5, 100)
grid_log_likelihoods = np.array(
    [t_copula_log_likelihood(1.0 / value) for value in reciprocal_df_grid]
)
best_grid_index = int(np.argmax(grid_log_likelihoods))

# Refine between the neighbors of the best grid point.
left_index = max(0, best_grid_index - 1)
right_index = min(len(reciprocal_df_grid) - 1, best_grid_index + 1)
refinement = minimize_scalar(
    lambda reciprocal_df: -t_copula_log_likelihood(1.0 / reciprocal_df),
    bounds=(
        reciprocal_df_grid[left_index],
        reciprocal_df_grid[right_index],
    ),
    method="bounded",
    options={"xatol": 1e-12},
)
if not refinement.success or not np.isfinite(refinement.fun):
    raise RuntimeError(f"Student-t profile refinement failed: {refinement.message}")

# Include both global endpoints and the grid optimum in the final comparison.
profile_candidates = [
    (
        reciprocal_df_grid[best_grid_index],
        grid_log_likelihoods[best_grid_index],
    ),
    (float(refinement.x), float(-refinement.fun)),
    (reciprocal_df_grid[0], grid_log_likelihoods[0]),
    (reciprocal_df_grid[-1], grid_log_likelihoods[-1]),
]
best_reciprocal_df, t_log_likelihood = max(
    profile_candidates, key=lambda candidate: candidate[1]
)
fitted_nu = 1.0 / best_reciprocal_df

# Problem 3(b) — Conditional information criteria
def conditional_criteria(log_likelihood, k, n_returns):
    if n_returns <= k + 1:
        raise ValueError("Insufficient return observations for conditional AICc.")
    aicc = (
        -2.0 * log_likelihood
        + 2.0 * k
        + 2.0 * k * (k + 1) / (n_returns - k - 1)
    )
    bic = -2.0 * log_likelihood + k * np.log(n_returns)
    return aicc, bic

comparison_rows = []
for name, ll, nu, k in [
    ("Gaussian", gaussian_log_likelihood, np.nan, 0),
    ("Student-t", t_log_likelihood, fitted_nu, 1),
]:
    aicc, bic = conditional_criteria(ll, k, n)
    comparison_rows.append(
        {
            "copula": name,
            "n_returns": n,
            "conditional_k": k,
            "log_likelihood": ll,
            "nu": nu,
            "conditional_AICc": aicc,
            "conditional_BIC": bic,
        }
    )

copula_comparison = pd.DataFrame(comparison_rows)
print("\nProblem 3(b) — Fixed-R copula likelihoods and conditional criteria")
print(
    copula_comparison.to_string(
        index=False,
        na_rep="—",
        float_format=lambda value: f"{value:.12f}",
    )
)

# Problem 3(b) — Requested Student-t likelihood evaluations
t_profile_checks = pd.DataFrame(
    [
        {"nu": nu, "log_likelihood": t_copula_log_likelihood(float(nu))}
        for nu in (2, 4, 8, 20, 50, 100, 200, 1000, 10000)
    ]
)
print("\nProblem 3(b) — Student-t copula log likelihood at specified nu")
print(
    t_profile_checks.to_string(
        index=False, float_format=lambda value: f"{value:.12f}"
    )
)


# Problem 3(c) — Inputs and supplied Gaussian copula/margin choices

from scipy.stats.sampling import NumericalInversePolynomial

BASE_DIR = INPUT_DIR
STOCK_NAMES = [f"S{i}" for i in range(1, 11)]
MARGIN_CHOICES = {
    "S1": "normal", "S2": "student_t", "S3": "NIG",
    "S4": "normal", "S5": "student_t", "S6": "NIG",
    "S7": "normal", "S8": "student_t", "S9": "NIG",
    "S10": "student_t",
}
N_DRAWS = 100_000
N_TAIL = 5_000
SEED = 5452025

for name in ("R", "fitted_parameters", "nig_fits"):
    if name not in globals():
        raise RuntimeError(f"Missing Problem 3(a)/(b) input: {name}")

for filename in ("problem3.csv", "problem3_holdings.csv"):
    if not (BASE_DIR / filename).is_file():
        raise FileNotFoundError(f"Required input file not found: {BASE_DIR / filename}")

price_data = pd.read_csv(BASE_DIR / "problem3.csv")
missing = [stock for stock in STOCK_NAMES if stock not in price_data.columns]
if missing:
    raise ValueError(f"problem3.csv is missing stock columns: {missing}")
if price_data.empty:
    raise ValueError("problem3.csv contains no price rows.")

last_prices = pd.to_numeric(
    price_data[STOCK_NAMES].iloc[-1], errors="raise"
).astype(float)
if not np.isfinite(last_prices).all() or (last_prices <= 0).any():
    raise ValueError("The final price row must contain finite positive stock prices.")

holdings = pd.read_csv(BASE_DIR / "problem3_holdings.csv")
# Required headers: stock, shares, book; capitalization is ignored.
holdings.columns = holdings.columns.str.strip().str.lower()
if holdings.columns.duplicated().any():
    raise ValueError("Holdings column names are duplicated after normalization.")
if "holding" in holdings.columns:
    if "shares" in holdings.columns:
        raise ValueError("Ambiguous quantity columns: both Holding and shares.")
    holdings = holdings.rename(columns={"holding": "shares"})
missing = [name for name in ("stock", "shares", "book") if name not in holdings]
if missing:
    raise ValueError(f"problem3_holdings.csv is missing columns: {missing}")
if holdings.empty:
    raise ValueError("problem3_holdings.csv contains no positions.")
if holdings[["stock", "shares", "book"]].isna().any().any():
    raise ValueError("Holdings contain missing stock, shares, or book values.")

holdings = holdings[["stock", "shares", "book"]].copy()
holdings["stock"] = holdings["stock"].astype(str).str.strip()
holdings["book"] = holdings["book"].astype(str).str.strip()
holdings["shares"] = pd.to_numeric(holdings["shares"], errors="raise")

unknown_stocks = sorted(set(holdings["stock"]) - set(STOCK_NAMES))
if unknown_stocks:
    raise ValueError(f"Holdings contain unknown stocks: {unknown_stocks}")
if set(holdings["book"]) != {"A", "B"}:
    raise ValueError("Holdings must identify books using A and B.")
if not np.isfinite(holdings["shares"].to_numpy(dtype=float)).all():
    raise ValueError("Shares must be finite.")

holdings["last_price"] = holdings["stock"].map(last_prices)
holdings["current_value"] = holdings["shares"] * holdings["last_price"]

# Problem 3(c) — Validate R and generate correlated Gaussian draws
if isinstance(R, pd.DataFrame):
    if not R.index.is_unique or not R.columns.is_unique:
        raise ValueError("R must have unique stock labels.")
    if set(R.index) != set(STOCK_NAMES) or set(R.columns) != set(STOCK_NAMES):
        raise ValueError("R labels must match S1 through S10.")
    correlation_array = R.loc[STOCK_NAMES, STOCK_NAMES].to_numpy(dtype=float)
else:
    if "STOCKS" not in globals() or list(STOCKS) != STOCK_NAMES:
        raise ValueError("An unlabeled R requires STOCKS ordered S1 through S10.")
    correlation_array = np.asarray(R, dtype=float)

if correlation_array.shape != (10, 10):
    raise ValueError("R must be a 10-by-10 matrix.")
if not np.isfinite(correlation_array).all():
    raise ValueError("R contains nonfinite values.")
if not np.allclose(correlation_array, correlation_array.T, atol=1e-12, rtol=0):
    raise ValueError("R must be symmetric.")
if not np.allclose(np.diag(correlation_array), 1.0, atol=1e-12, rtol=0):
    raise ValueError("R must have a unit diagonal.")

try:
    L = np.linalg.cholesky(correlation_array)
except np.linalg.LinAlgError as exc:
    raise ValueError("R must be positive definite.") from exc

rng = np.random.default_rng(SEED)
independent_normals = rng.standard_normal((N_DRAWS, len(STOCK_NAMES)))
correlated_normals = independent_normals @ L.T
simulated_uniforms = stats.norm.cdf(correlated_normals)

if np.any((simulated_uniforms <= 0) | (simulated_uniforms >= 1)):
    raise RuntimeError("Gaussian CDF produced boundary uniforms; no clipping applied.")

# Problem 3(c) — Selected marginal inverse CDFs and NIG inversion checks
simulated_returns = np.empty_like(simulated_uniforms)
inversion_checks = []
nig_inverters = {}
validation_grid = np.linspace(1e-6, 1.0 - 1e-6, 1001)

for column, stock in enumerate(STOCK_NAMES):
    margin = MARGIN_CHOICES[stock]
    u = simulated_uniforms[:, column]

    if margin == "NIG":
        if stock not in nig_fits:
            raise ValueError(f"Missing NIG fit for {stock}.")
        parameters = nig_fits[stock]["parameters"]
        required = ("a", "b", "loc", "scale")
        if any(key not in parameters for key in required):
            raise ValueError(f"Incomplete NIG fit for {stock}.")

        a, b, loc, scale = (float(parameters[key]) for key in required)
        if (
            not np.isfinite([a, b, loc, scale]).all()
            or a <= abs(b)
            or scale <= 0
        ):
            raise ValueError(f"Invalid NIG parameters for {stock}.")

        # Invert the frozen distribution in return units.
        standardized_nig = stats.norminvgauss(a, b, loc=loc, scale=scale)
        inverter = NumericalInversePolynomial(
            standardized_nig,
            u_resolution=1e-12,
            random_state=rng,
        )
        grid_quantiles = inverter.ppf(validation_grid)
        inversion_error = np.abs(
            standardized_nig.cdf(grid_quantiles) - validation_grid
        )
        # SciPy's default NIG CDF quadrature can miss narrow density peaks.
        # Independently recompute suspect points with tighter integration tolerances.
        from scipy.integrate import quad
        for grid_index in np.flatnonzero((inversion_error >= 1e-8) | ~np.isfinite(inversion_error)):
            cdf_value, quadrature_error = quad(
                standardized_nig.pdf, -np.inf, grid_quantiles[grid_index],
                epsabs=1e-12, epsrel=1e-12, limit=300,
            )
            if not np.isfinite(quadrature_error) or quadrature_error > 1e-10:
                raise RuntimeError(f"{stock}: independent CDF integration failed.")
            inversion_error[grid_index] = abs(cdf_value - validation_grid[grid_index])
        if not np.isfinite(inversion_error).all():
            raise RuntimeError(f"{stock}: nonfinite NIG inversion diagnostics.")

        max_error = float(inversion_error.max())
        if max_error >= 1e-8:
            raise RuntimeError(
                f"{stock}: NIG CDF inversion error {max_error:.16e} "
                "is not below 1e-8."
            )

        inversion_checks.append(
            {"stock": stock, "maximum_CDF_inversion_error": max_error}
        )
        nig_inverters[stock] = inverter
        simulated_returns[:, column] = inverter.ppf(u)

    else:
        if stock not in fitted_parameters or margin not in fitted_parameters[stock]:
            raise ValueError(f"Missing {margin} fit for {stock}.")

        parameters = tuple(fitted_parameters[stock][margin])
        expected_length = 2 if margin == "normal" else 3
        if (
            len(parameters) != expected_length
            or not np.isfinite(parameters).all()
            or parameters[-1] <= 0
        ):
            raise ValueError(f"Invalid {margin} parameters for {stock}.")
        if margin == "student_t" and parameters[0] <= 0:
            raise ValueError(f"Invalid Student-t degrees of freedom for {stock}.")

        distribution = stats.norm if margin == "normal" else stats.t
        simulated_returns[:, column] = distribution.ppf(u, *parameters)

if not np.isfinite(simulated_returns).all():
    raise RuntimeError("Simulated marginal returns contain nonfinite values.")

print("Problem 3(c) — NIG inverse-CDF checks")
print(
    pd.DataFrame(inversion_checks).to_string(
        index=False, float_format=lambda value: f"{value:.12e}"
    )
)

# Problem 3(c) — Position and aggregate P&L within each simulation draw
stock_indices = {stock: index for index, stock in enumerate(STOCK_NAMES)}
position_indices = [stock_indices[stock] for stock in holdings["stock"]]
position_values = holdings["current_value"].to_numpy(dtype=float)

position_pnl = simulated_returns[:, position_indices] * position_values[None, :]
if not np.isfinite(position_pnl).all():
    raise RuntimeError("Position P&L contains nonfinite values.")

def dollar_var_es(pnl):
    pnl = np.asarray(pnl, dtype=float)
    if pnl.shape != (N_DRAWS,) or not np.isfinite(pnl).all():
        raise ValueError("P&L must contain exactly 100,000 finite draws.")
    sorted_pnl = np.sort(pnl)
    # Inverse empirical CDF at 5%; exactly 5,000 equally weighted tail draws.
    var_95 = -float(sorted_pnl[N_TAIL - 1])
    es_95 = -float(sorted_pnl[:N_TAIL].mean())
    return var_95, es_95

position_results = holdings.copy()
position_results.insert(0, "position", np.arange(1, len(holdings) + 1))
position_risk = [
    dollar_var_es(position_pnl[:, index])
    for index in range(len(holdings))
]
position_results["VaR_95_dollars"] = [risk[0] for risk in position_risk]
position_results["ES_95_dollars"] = [risk[1] for risk in position_risk]

aggregate_rows = []
aggregate_pnl = {}
for label in ("A", "B", "Total"):
    mask = (
        np.ones(len(holdings), dtype=bool)
        if label == "Total"
        else holdings["book"].eq(label).to_numpy()
    )
    pnl = position_pnl[:, mask].sum(axis=1)
    aggregate_pnl[label] = pnl
    var_95, es_95 = dollar_var_es(pnl)
    aggregate_rows.append(
        {
            "portfolio": label,
            "current_value": float(position_values[mask].sum()),
            "VaR_95_dollars": var_95,
            "ES_95_dollars": es_95,
        }
    )

aggregate_results = pd.DataFrame(aggregate_rows)

print("\nProblem 3(c) — Positions, valuation, and dollar risk")
print(
    position_results.to_string(
        index=False, float_format=lambda value: f"{value:.10f}"
    )
)
print("\nProblem 3(c) — Book and total valuation and dollar risk")
print(
    aggregate_results.to_string(
        index=False, float_format=lambda value: f"{value:.10f}"
    )
)


# Problem 1(d)–(e) — Repaired correlation and covariance validation

ATOL = 1e-15
RTOL = 1e-12

def require_globals(*names):
    missing = [name for name in names if name not in globals()]
    if missing:
        raise RuntimeError(f"Missing calculation inputs: {missing}")

require_globals("data", "repaired_correlation", "repaired_covariance")

if not all(
    isinstance(frame, pd.DataFrame)
    for frame in (data, repaired_correlation, repaired_covariance)
):
    raise TypeError("Problem 1 data and repaired matrices must be DataFrames.")

for matrix in (repaired_correlation, repaired_covariance):
    assert matrix.index.equals(data.columns), "Problem 1 row labels do not match."
    assert matrix.columns.equals(data.columns), "Problem 1 column labels do not match."

corr_array = repaired_correlation.to_numpy(dtype=float)
cov_array = repaired_covariance.to_numpy(dtype=float)
full_variances = data.var(skipna=True, ddof=1).to_numpy(dtype=float)

assert np.isfinite(corr_array).all(), "Repaired correlation is nonfinite."
assert np.isfinite(cov_array).all(), "Repaired covariance is nonfinite."
assert np.isfinite(full_variances).all(), "Full-history variances are nonfinite."

np.testing.assert_allclose(corr_array, corr_array.T, atol=ATOL, rtol=RTOL)
np.testing.assert_allclose(np.diag(corr_array), 1.0, atol=ATOL, rtol=RTOL)
assert np.linalg.eigvalsh(corr_array).min() >= -1e-10, (
    "Repaired correlation has an eigenvalue below -1e-10."
)
np.testing.assert_allclose(
    np.diag(cov_array), full_variances, atol=ATOL, rtol=RTOL
)

# Problem 2(a)–(b) — OLS mean and variance identities
# Use the retained Problem 2 arrays; generic prices/returns may now hold Problem 3.
require_globals("market", "problem2_stock_returns", "coefficients", "residuals")

market_2 = np.asarray(market, dtype=float)
stock_2 = np.asarray(problem2_stock_returns, dtype=float)
coefficients_2 = np.asarray(coefficients, dtype=float)
residuals_2 = np.asarray(residuals, dtype=float)

assert market_2.ndim == stock_2.ndim == residuals_2.ndim == 1
assert market_2.shape == stock_2.shape == residuals_2.shape
assert len(stock_2) > 2
assert coefficients_2.shape == (2,)
for values in (market_2, stock_2, coefficients_2, residuals_2):
    assert np.isfinite(values).all(), "Problem 2 inputs are nonfinite."

alpha_2, beta_2 = coefficients_2
np.testing.assert_allclose(
    residuals_2,
    stock_2 - (alpha_2 + beta_2 * market_2),
    atol=ATOL,
    rtol=RTOL,
)
mu_2 = alpha_2 + beta_2 * market_2.mean()
variance_2 = (
    beta_2**2 * market_2.var(ddof=1)
    + np.sum(residuals_2**2) / (len(stock_2) - 1)
)
np.testing.assert_allclose(mu_2, stock_2.mean(), atol=ATOL, rtol=RTOL)
np.testing.assert_allclose(
    variance_2, stock_2.var(ddof=1), atol=ATOL, rtol=RTOL
)

# Problem 3(a)–(c) — Selected marginal parameter validation
require_globals("fitted_parameters", "nig_fits")

selected_margins = {
    "S1": "normal", "S2": "student_t", "S3": "NIG",
    "S4": "normal", "S5": "student_t", "S6": "NIG",
    "S7": "normal", "S8": "student_t", "S9": "NIG",
    "S10": "student_t",
}

for ticker, model in selected_margins.items():
    if model == "NIG":
        if ticker not in nig_fits:
            raise ValueError(f"Missing selected NIG fit for {ticker}.")
        fitted = nig_fits[ticker]["parameters"]
        keys = ("a", "b", "loc", "scale")
        if any(key not in fitted for key in keys):
            raise ValueError(f"Incomplete selected NIG parameters for {ticker}.")
        parameters = np.asarray([fitted[key] for key in keys], dtype=float)
        assert np.isfinite(parameters).all(), f"{ticker}: nonfinite NIG parameters."
        a_3, b_3, loc_3, scale_3 = parameters
        assert a_3 > abs(b_3), f"{ticker}: NIG requires a > |b|."
        assert scale_3 > 0, f"{ticker}: NIG scale must be positive."
    else:
        if ticker not in fitted_parameters or model not in fitted_parameters[ticker]:
            raise ValueError(f"Missing selected {model} fit for {ticker}.")
        parameters = np.asarray(fitted_parameters[ticker][model], dtype=float)
        expected_length = 2 if model == "normal" else 3
        assert parameters.shape == (expected_length,)
        assert np.isfinite(parameters).all(), f"{ticker}: nonfinite parameters."
        assert parameters[-1] > 0, f"{ticker}: scale must be positive."
        if model == "student_t":
            assert parameters[0] > 0, f"{ticker}: degrees of freedom must be positive."

# Problem 3(c) — Ticker matching, valuations, and simulated P&L
require_globals(
    "holdings", "last_prices", "STOCK_NAMES", "simulated_returns",
    "position_pnl", "aggregate_pnl", "position_results", "aggregate_results",
)

assert len(STOCK_NAMES) == len(set(STOCK_NAMES)), "Duplicate simulation tickers."
assert set(STOCK_NAMES) == set(selected_margins)
assert last_prices.index.is_unique, "Duplicate last-price tickers."
assert holdings["stock"].notna().all()
assert set(holdings["stock"]).issubset(set(STOCK_NAMES))
assert set(holdings["stock"]).issubset(set(last_prices.index))
assert set(holdings["book"]) == {"A", "B"}

shares_check = holdings["shares"].to_numpy(dtype=float)
mapped_prices = last_prices.reindex(holdings["stock"]).to_numpy(dtype=float)
assert np.isfinite(shares_check).all()
assert np.isfinite(mapped_prices).all() and np.all(mapped_prices > 0)

expected_values = shares_check * mapped_prices
np.testing.assert_allclose(
    holdings["last_price"].to_numpy(dtype=float),
    mapped_prices,
    atol=ATOL,
    rtol=RTOL,
)
np.testing.assert_allclose(
    holdings["current_value"].to_numpy(dtype=float),
    expected_values,
    atol=ATOL,
    rtol=RTOL,
)

simulated_returns_check = np.asarray(simulated_returns, dtype=float)
position_pnl_check = np.asarray(position_pnl, dtype=float)
assert simulated_returns_check.shape == (100_000, len(STOCK_NAMES))
assert position_pnl_check.shape == (100_000, len(holdings))
assert np.isfinite(simulated_returns_check).all()
assert np.isfinite(position_pnl_check).all()

ticker_indices = {ticker: index for index, ticker in enumerate(STOCK_NAMES)}
mapped_indices = [ticker_indices[ticker] for ticker in holdings["stock"]]
expected_position_pnl = (
    simulated_returns_check[:, mapped_indices] * expected_values[None, :]
)
np.testing.assert_allclose(
    position_pnl_check, expected_position_pnl, atol=ATOL, rtol=RTOL
)

# Problem 3(c) — Book/total reconciliation with scale-aware rounding tolerances
def assert_sum_matches(actual, expected, absolute_sum_scale, message):
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)
    tolerance = ATOL + RTOL * np.asarray(absolute_sum_scale, dtype=float)
    assert np.isfinite(actual).all() and np.isfinite(expected).all(), message
    assert np.all(np.abs(actual - expected) <= tolerance), message

assert aggregate_results["portfolio"].is_unique
aggregate_table = aggregate_results.set_index("portfolio")
assert set(aggregate_table.index) == {"A", "B", "Total"}

for label in ("A", "B", "Total"):
    mask = (
        np.ones(len(holdings), dtype=bool)
        if label == "Total"
        else holdings["book"].eq(label).to_numpy()
    )
    expected_value = expected_values[mask].sum()
    expected_pnl = position_pnl_check[:, mask].sum(axis=1)

    if label not in aggregate_pnl:
        raise ValueError(f"Missing simulated P&L for {label}.")
    actual_pnl = np.asarray(aggregate_pnl[label], dtype=float)
    assert actual_pnl.shape == (100_000,)
    assert np.isfinite(actual_pnl).all()

    assert_sum_matches(
        aggregate_table.loc[label, "current_value"],
        expected_value,
        np.abs(expected_values[mask]).sum(),
        f"{label}: current value does not reconcile.",
    )
    assert_sum_matches(
        actual_pnl,
        expected_pnl,
        np.abs(position_pnl_check[:, mask]).sum(axis=1),
        f"{label}: simulated P&L does not reconcile.",
    )

assert_sum_matches(
    aggregate_table.loc["Total", "current_value"],
    aggregate_table.loc["A", "current_value"]
    + aggregate_table.loc["B", "current_value"],
    np.abs(expected_values).sum(),
    "Book values do not sum to total value.",
)
assert_sum_matches(
    aggregate_pnl["Total"],
    np.asarray(aggregate_pnl["A"]) + np.asarray(aggregate_pnl["B"]),
    np.abs(position_pnl_check).sum(axis=1),
    "Book P&Ls do not sum to total P&L for every draw.",
)

# Problem 3(c) — Validate empirical VaR/ES and ES inequalities
def checked_tail_metrics(pnl):
    sorted_pnl = np.sort(np.asarray(pnl, dtype=float))
    assert sorted_pnl.shape == (100_000,)
    assert np.isfinite(sorted_pnl).all()
    return -sorted_pnl[4999], -sorted_pnl[:5000].mean()

assert len(position_results) == len(holdings)
for field in ("stock", "book"):
    assert np.array_equal(
        position_results[field].to_numpy(), holdings[field].to_numpy()
    ), f"Position results do not match holdings by {field}."

for index in range(len(holdings)):
    expected_var, expected_es = checked_tail_metrics(position_pnl_check[:, index])
    reported_var = float(position_results.iloc[index]["VaR_95_dollars"])
    reported_es = float(position_results.iloc[index]["ES_95_dollars"])

    np.testing.assert_allclose(
        [reported_var, reported_es],
        [expected_var, expected_es],
        atol=ATOL,
        rtol=RTOL,
    )
    tolerance = ATOL + RTOL * max(abs(reported_var), abs(reported_es))
    assert reported_es >= reported_var - tolerance, (
        f"Position {index + 1}: ES is below VaR."
    )

for label in ("A", "B", "Total"):
    expected_var, expected_es = checked_tail_metrics(aggregate_pnl[label])
    np.testing.assert_allclose(
        aggregate_table.loc[label, ["VaR_95_dollars", "ES_95_dollars"]].to_numpy(
            dtype=float
        ),
        [expected_var, expected_es],
        atol=ATOL,
        rtol=RTOL,
    )

total_es = float(aggregate_table.loc["Total", "ES_95_dollars"])
book_es = aggregate_table.loc[["A", "B"], "ES_95_dollars"].to_numpy(dtype=float)
es_tolerance = ATOL + RTOL * max(abs(total_es), np.abs(book_es).sum())
assert total_es <= book_es.sum() + es_tolerance, (
    "Total ES exceeds the sum of book ES beyond floating-point tolerance."
)


# Problem 3(b) — Extended degrees-of-freedom sensitivity.
extended_fit = minimize_scalar(
    lambda log_nu: -t_copula_log_likelihood(np.exp(log_nu)),
    bounds=(np.log(100.0), np.log(10000.0)), method="bounded",
    options={"xatol": 1e-7},
)
if not extended_fit.success:
    raise RuntimeError("Extended copula sensitivity optimization failed.")
extended_nu = float(np.exp(extended_fit.x))
extended_ll = t_copula_log_likelihood(extended_nu)

# All Problems — Export setup
import json
import platform
import sys
from importlib.metadata import version


OUTPUT_DIR = PROJECT_DIR / "results"
INPUT_DIR = PROJECT_DIR / "data"

required = (
    "data", "correlation", "repaired_correlation", "repaired_covariance",
    "coefficients", "fit_results", "criteria_table", "gap_table",
    "nig_fits", "tau", "R", "copula_comparison",
    "reciprocal_df_grid", "grid_log_likelihoods", "t_profile_checks",
    "holdings", "position_results", "aggregate_results",
    "position_pnl", "aggregate_pnl", "STOCK_NAMES",
)
missing = [name for name in required if name not in globals()]
if missing:
    raise RuntimeError(f"Missing outputs required for export: {missing}")

if "__file__" not in globals():
    raise RuntimeError(
        "Save and run the combined calculation/export code as a Python script "
        "so the README can record the actual run command."
    )
analysis_script = Path(__file__).resolve()
if not analysis_script.is_file():
    raise FileNotFoundError(f"Analysis script not found: {analysis_script}")

for filename in ("problem1.csv", "problem2.csv", "problem3.csv", "problem3_holdings.csv"):
    if not (INPUT_DIR / filename).is_file():
        raise FileNotFoundError(f"Required input file not found: {INPUT_DIR / filename}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
exported_files = []

def save_csv(filename, frame, *, index=False):
    if not isinstance(frame, (pd.DataFrame, pd.Series)):
        raise TypeError(f"{filename}: expected a DataFrame or Series.")
    frame.to_csv(OUTPUT_DIR / filename, index=index, float_format="%.17g")
    exported_files.append(filename)

def eigenvalue_table(matrix):
    values = np.asarray(matrix, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Cannot export eigenvalues of a nonfinite matrix.")
    return pd.DataFrame(
        {
            "ascending_index": np.arange(1, len(values) + 1),
            "eigenvalue": np.linalg.eigvalsh(values),
        }
    )

# Problem 1(a) — Original pairwise correlation and observation counts
observed = data.notna().astype(np.int64)
save_csv("problem1a_pairwise_counts.csv", observed.T @ observed, index=True)
save_csv("problem1a_correlation.csv", correlation, index=True)

# Problem 1(b) — Full-history counts and sample variances
save_csv(
    "problem1b_column_statistics.csv",
    pd.DataFrame(
        {
            "count": data.count(),
            "variance": data.var(ddof=1),
            "standard_deviation": data.std(ddof=1),
        }
    ).rename_axis("column"),
    index=True,
)

# Problem 1(c) — Original correlation eigenvalues
save_csv("problem1c_correlation_eigenvalues.csv", eigenvalue_table(correlation))

save_csv("problem1c_numerical_checks.csv", pd.DataFrame([{
    "absolute_tolerance": CHECK_TOLERANCE, "relative_tolerance": 0,
    "smallest_eigenvalue": np.linalg.eigvalsh(correlation_array)[0],
    "symmetry_max_error": np.max(np.abs(correlation_array - correlation_array.T)),
    "unit_diagonal_max_error": np.max(np.abs(np.diag(correlation_array) - 1)),
}]))

# Problem 1(d) — Repaired correlation and changes
original_array = correlation.to_numpy(dtype=float)
repaired_array = repaired_correlation.to_numpy(dtype=float)
difference = repaired_array - original_array
upper_i, upper_j = np.triu_indices(len(correlation), k=1)

change_table = pd.DataFrame(
    {
        "column_1": correlation.columns.take(upper_i),
        "column_2": correlation.columns.take(upper_j),
        "original": original_array[upper_i, upper_j],
        "repaired": repaired_array[upper_i, upper_j],
        "change": difference[upper_i, upper_j],
        "absolute_change": np.abs(difference[upper_i, upper_j]),
    }
).sort_values("absolute_change", ascending=False, kind="stable")

save_csv("problem1d_repaired_correlation.csv", repaired_correlation, index=True)
save_csv(
    "problem1d_repaired_correlation_eigenvalues.csv",
    eigenvalue_table(repaired_array),
)
save_csv(
    "problem1d_correlation_changes.csv",
    pd.DataFrame(
        difference, index=correlation.index, columns=correlation.columns
    ),
    index=True,
)
save_csv("problem1d_off_diagonal_changes.csv", change_table)
save_csv(
    "problem1d_repair_metrics.csv",
    pd.DataFrame(
        [{"frobenius_distance": np.linalg.norm(difference, ord="fro")}]
    ),
)

# Problem 1(e) — Repaired covariance and eigenvalues
save_csv("problem1e_repaired_covariance.csv", repaired_covariance, index=True)
save_csv(
    "problem1e_repaired_covariance_eigenvalues.csv",
    eigenvalue_table(repaired_covariance),
)

# Problem 2(a) — Regression estimates from Problem 2's own input
problem2_prices = pd.read_csv(INPUT_DIR / "problem2.csv")
if not {"SPY", "XYZ"}.issubset(problem2_prices.columns):
    raise ValueError("problem2.csv must contain SPY and XYZ.")
problem2_prices = problem2_prices[["SPY", "XYZ"]].apply(
    pd.to_numeric, errors="raise"
)
if (
    len(problem2_prices) < 4
    or not np.isfinite(problem2_prices.to_numpy()).all()
    or (problem2_prices <= 0).any().any()
):
    raise ValueError("Problem 2 requires at least four finite positive price rows.")

problem2_returns = problem2_prices.pct_change(fill_method=None).iloc[1:]
market_2 = problem2_returns["SPY"].to_numpy()
stock_2 = problem2_returns["XYZ"].to_numpy()
design_2 = np.column_stack([np.ones(len(market_2)), market_2])
ols_2, _, rank_2, _ = np.linalg.lstsq(design_2, stock_2, rcond=None)
if rank_2 != 2:
    raise ValueError("Problem 2 regression design is rank deficient.")
np.testing.assert_allclose(ols_2, coefficients, atol=1e-15, rtol=1e-12)

alpha_2, beta_2 = ols_2
residuals_2 = stock_2 - design_2 @ ols_2
residual_variance_2 = np.sum(residuals_2**2) / (len(stock_2) - 1)
market_variance_2 = market_2.var(ddof=1)
mu_2 = alpha_2 + beta_2 * market_2.mean()
sigma_squared_2 = beta_2**2 * market_variance_2 + residual_variance_2
sigma_2 = np.sqrt(sigma_squared_2)

save_csv(
    "problem2a_regression_estimates.csv",
    pd.DataFrame(
        [{
            "n_returns": len(stock_2),
            "alpha": alpha_2,
            "beta": beta_2,
            "market_mean": market_2.mean(),
            "market_variance": market_variance_2,
            "residual_variance": residual_variance_2,
            "residual_standard_deviation": np.sqrt(residual_variance_2),
            "mu": mu_2,
            "sigma_squared": sigma_squared_2,
            "sigma": sigma_2,
            "last_XYZ_price": problem2_prices["XYZ"].iloc[-1],
        }]
    ),
)

# Problem 2(b) — Percentage VaR and ES
problem2_risk_rows = []
for confidence in (0.95, 0.99):
    q = 1.0 - confidence
    z = stats.norm.ppf(q)
    problem2_risk_rows.append(
        {
            "confidence": confidence,
            "VaR_percent": 100.0 * (-mu_2 - sigma_2 * z),
            "ES_percent": 100.0 * (-mu_2 + sigma_2 * stats.norm.pdf(z) / q),
        }
    )
save_csv("problem2b_percentage_risk.csv", pd.DataFrame(problem2_risk_rows))

# Problem 3(a) — All fitted margins, including updated NIG fits
parameter_table = fit_results.copy()
for ticker, fit in nig_fits.items():
    mask = parameter_table["stock"].eq(ticker) & parameter_table["model"].eq("NIG")
    if mask.sum() != 1:
        raise ValueError(f"Expected exactly one NIG parameter row for {ticker}.")
    for key, value in fit["parameters"].items():
        if key not in parameter_table:
            parameter_table[key] = pd.Series(
                [None] * len(parameter_table), dtype=object
            )
        parameter_table.loc[mask, key] = value

save_csv("problem3a_fitted_parameters.csv", parameter_table)
save_csv("problem3a_information_criteria.csv", criteria_table)
save_csv("problem3a_candidate_gaps.csv", gap_table)

nig_diagnostics = []
for ticker, fit in nig_fits.items():
    diagnostics = fit["diagnostics"].copy()
    diagnostics.insert(0, "stock", ticker)
    nig_diagnostics.append(diagnostics)
save_csv("problem3a_NIG_diagnostics.csv", pd.concat(nig_diagnostics, ignore_index=True))

# Problem 3(b) — Kendall tau, shared R, and copula profile
R_frame = (
    R.loc[STOCK_NAMES, STOCK_NAMES]
    if isinstance(R, pd.DataFrame)
    else pd.DataFrame(R, index=STOCK_NAMES, columns=STOCK_NAMES)
)
save_csv("problem3b_kendall_tau.csv", tau, index=True)
save_csv("problem3b_copula_R.csv", R_frame, index=True)
save_csv("problem3b_copula_R_eigenvalues.csv", eigenvalue_table(R_frame))
save_csv("problem3b_copula_criteria.csv", copula_comparison)
save_csv(
    "problem3b_t_copula_profile.csv",
    pd.DataFrame(
        {
            "reciprocal_df": reciprocal_df_grid,
            "nu": 1.0 / np.asarray(reciprocal_df_grid),
            "log_likelihood": grid_log_likelihoods,
        }
    ),
)
save_csv("problem3b_t_copula_requested_nu.csv", t_profile_checks)

save_csv("problem3b_extended_sensitivity.csv", pd.DataFrame([{
    "nu": extended_nu, "log_likelihood": extended_ll,
    "AICc": conditional_criteria(extended_ll, 1, 1000)[0],
}]))
save_csv("problem3c_inverse_cdf_checks.csv", pd.DataFrame(inversion_checks))

# Problem 3(c) — Position values and full-simulation dollar risk
save_csv(
    "problem3c_position_values.csv",
    holdings[["stock", "book", "shares", "last_price", "current_value"]],
)
save_csv("problem3c_position_dollar_risk.csv", position_results)
save_csv("problem3c_aggregate_dollar_risk.csv", aggregate_results)

# Problem 3(c) — Ten consecutive batches, without resimulation
pnl_positions = np.asarray(position_pnl, dtype=float)
if pnl_positions.shape != (100_000, len(holdings)):
    raise ValueError("Expected 100,000 P&L draws for each holding.")
if not np.isfinite(pnl_positions).all():
    raise ValueError("Position P&Ls contain nonfinite values.")

batch_rows = []
for batch in range(10):
    start = batch * 10_000
    stop = start + 10_000
    series = [
        (
            "position",
            str(index + 1),
            holdings.iloc[index]["stock"],
            holdings.iloc[index]["book"],
            pnl_positions[start:stop, index],
        )
        for index in range(len(holdings))
    ]
    for label in ("A", "B", "Total"):
        if label not in aggregate_pnl:
            raise ValueError(f"Missing aggregate P&L for {label}.")
        pnl = np.asarray(aggregate_pnl[label], dtype=float)
        if pnl.shape != (100_000,) or not np.isfinite(pnl).all():
            raise ValueError(f"Invalid aggregate P&L for {label}.")
        series.append(("aggregate", label, "", label, pnl[start:stop]))

    for scope, identifier, ticker, book, pnl in series:
        ordered = np.sort(pnl)
        batch_rows.append(
            {
                "batch": batch + 1,
                "first_draw_1_based": start + 1,
                "last_draw_1_based": stop,
                "draw_count": 10_000,
                "tail_count": 500,
                "scope": scope,
                "identifier": identifier,
                "stock": ticker,
                "book": book,
                "VaR_95_dollars": -float(ordered[499]),
                "ES_95_dollars": -float(ordered[:500].mean()),
            }
        )

save_csv("problem3c_batch_dollar_risk.csv", pd.DataFrame(batch_rows))

# All Problems — Reproducibility metadata and calculation conventions
metadata = {
    "analysis_script": analysis_script.name,
    "input_directory": str(INPUT_DIR.relative_to(PROJECT_DIR)),
    "output_directory": str(OUTPUT_DIR.relative_to(PROJECT_DIR)),
    "python_version": platform.python_version(),
    "python_executable": sys.executable,
    "package_versions": {
        package: version(package) for package in ("numpy", "pandas", "scipy")
    },
    "simulation": {
        "seed": 5452025,
        "generator": "numpy.random.default_rng",
        "draw_count": 100000,
        "batch_count": 10,
        "draws_per_batch": 10000,
        "batch_method": "Consecutive slices of the existing simulation",
    },
    "conventions": {
        "general": {
            "returns": "Daily arithmetic returns; no demeaning",
            "price_row_order": "Supplied chronological order",
            "data_files": "Separate input file for each problem",
            "sample_covariance_ddof": 1,
            "descriptive_variance_ddof": 1,
            "normal_MLE_scale_ddof": 0,
        },
        "problem1": {
            "variances": "Full available history of each column",
            "correlations": "Pairwise-complete observations",
            "repair": "Unweighted Higham alternating projections with Dykstra correction",
            "eigenvalue_tolerance": 1e-12,
            "objective": "0.5 * squared Frobenius distance",
            "objective_change_tolerance": 1e-14,
            "maximum_iterations": 10000,
            "covariance": "diag(full-history sample SD) @ repaired R @ diag(SD)",
        },
        "problem2": {
            "regression": "XYZ on an intercept and SPY, using OLS",
            "residual_variance_denominator": "n - 1",
            "market_variance_denominator": "n - 1",
            "mu": "alpha + beta * SPY sample mean",
            "sigma_squared": "beta**2 * market variance + residual variance",
            "confidence_levels": [0.95, 0.99],
            "VaR_percent": "100 * (-mu - sigma * norm.ppf(1-confidence))",
            "ES_percent": "100 * (-mu + sigma * norm.pdf(z)/(1-confidence))",
        },
        "problem3": {
            "margin_estimation": "Maximum likelihood on standardized returns",
            "parameter_units": "Location and scale converted back to return units",
            "parameter_counts": {"normal": 2, "student_t": 3, "NIG": 4},
            "n": "Number of returns, not prices",
            "AIC": "-2*ll + 2*k",
            "AICc": "-2*ll + 2*k + 2*k*(k+1)/(n-k-1)",
            "BIC": "-2*ll + k*log(n)",
            "supplied_margins": {
                "S1": "normal", "S2": "student_t", "S3": "NIG",
                "S4": "normal", "S5": "student_t", "S6": "NIG",
                "S7": "normal", "S8": "student_t", "S9": "NIG",
                "S10": "student_t",
            },
            "NIG_optimizer": {
                "method": "Nelder-Mead",
                "maxiter": 8000,
                "xatol": 1e-8,
                "fatol": 1e-8,
                "theta0_bounds": [-8, float(np.log(10000))],
                "b_bounds": [-10000, 10000],
                "loc_bounds": [-100, 100],
                "log_scale_bounds": [-8, 8],
                "starting_a": [0.5, 2, 10, 50],
                "starting_b_over_a": [-0.5, 0, 0.5],
                "start_moments": "Mean 0 and variance 1",
                "retention": "Highest likelihood among converged fits",
            },
            "copula_R": "sin(pi * Kendall tau / 2), diagonal 1; fixed for both copulas",
            "conditional_copula_parameter_counts": {"Gaussian": 0, "Student-t": 1},
            "t_copula_df_bounds": [2, 100],
            "reciprocal_df_grid": {"start": 0.01, "stop": 0.5, "count": 100},
            "profile_refinement": "Bounded refinement around grid maximum; compare endpoints",
            "supplied_simulation_copula": "Gaussian",
            "normal_draw_transform": "Independent standard normals @ cholesky(R).T",
            "uniform_transform": "norm.cdf",
            "NIG_inverse": "scipy.stats.sampling.NumericalInversePolynomial",
            "NIG_u_resolution": 1e-12,
            "NIG_inversion_check": {
                "grid_start": 1e-6,
                "grid_stop": 1 - 1e-6,
                "grid_count": 1001,
                "maximum_error_strictly_below": 1e-8,
            },
            "valuation": "Shares times last price, matched by stock name",
            "position_PnL": "Simulated arithmetic return times current position value",
            "aggregate_PnL": "Sum position P&Ls within each draw",
            "confidence": 0.95,
            "VaR": "Negative 5,000th sorted P&L for 100,000 draws",
            "ES": "Negative average of worst 5,000 P&Ls",
            "batch_VaR": "Negative 500th sorted P&L for 10,000 draws",
            "batch_ES": "Negative average of worst 500 P&Ls",
            "historical_VaR_convention": "Inverse empirical CDF",
            "ES_probability_mass": "Exactly worst 5%, with fractional boundary weight if needed",
        },
    },
    "csv_files": exported_files,
}
(OUTPUT_DIR / "calculation_metadata.json").write_text(
    json.dumps(metadata, indent=2, allow_nan=False) + "\n",
    encoding="utf-8",
)

# All Problems — Output index
(OUTPUT_DIR / "README.md").write_text(
    """# Numerical results

Run instructions are in ../README.md. Paths in calculation_metadata.json are
relative to the project folder, except the recorded Python executable.

- problem1a: pairwise correlation and observation counts.
- problem1b: available observations and full-history variances.
- problem1c: eigenvalues and numerical checks.
- problem1d: repaired correlation, eigenvalues, changes, and repair metrics.
- problem1e: repaired covariance and eigenvalues.
- problem2a: regression estimates.
- problem2b: percentage VaR and ES.
- problem3a: marginal fits, information criteria, and fitting diagnostics.
- problem3b: copula correlation, likelihood comparisons, and sensitivity checks.
- problem3c: position values, dollar risk, batch estimates, and inversion checks.
""", encoding="utf-8",
)
