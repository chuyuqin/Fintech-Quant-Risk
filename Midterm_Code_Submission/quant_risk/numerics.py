"""Reusable numerical helpers adapted from the supplied snippets."""
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from scipy.special import k1e

def project_psd(matrix):
    symmetric = (matrix + matrix.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    projected = (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T
    return (projected + projected.T) / 2.0


NIG_BOUNDS = [
    (-8.0, np.log(10000.0)),  # log(gamma)
    (-10000.0, 10000.0),     # b
    (-100.0, 100.0),         # location
    (-8.0, 8.0),             # log(scale)
]


def nig_logpdf_standardized(y, theta):
    theta0, b, loc, log_scale = np.asarray(theta, dtype=float)
    gamma = np.exp(theta0)
    a = np.hypot(gamma, b)
    scale = np.exp(log_scale)
    z = (np.asarray(y, dtype=float) - loc) / scale
    h = np.hypot(1.0, z)
    ah = a * h

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        exponent = -(b * b + (a * z) ** 2) / (gamma + ah) + b * z
        return (
            np.log(a)
            - np.log(np.pi)
            - log_scale
            - np.log(h)
            + np.log(k1e(ah))
            + exponent
        )


def fit_nig_mle(observations, stock_label):
    x = np.asarray(observations, dtype=float)
    if x.ndim != 1 or x.size <= 5:
        raise ValueError(f"{stock_label}: provide more than five return observations.")
    if not np.isfinite(x).all():
        raise ValueError(f"{stock_label}: returns must be finite and nonmissing.")

    center = float(x.mean())
    standardizer = float(x.std(ddof=0))
    if not np.isfinite(standardizer) or standardizer <= 0:
        raise ValueError(f"{stock_label}: return standard deviation must be positive.")

    y = (x - center) / standardizer

    def negative_log_likelihood(theta):
        log_density = nig_logpdf_standardized(y, theta)
        if not np.isfinite(log_density).all():
            return np.inf
        return -float(log_density.sum())

    diagnostics = []
    converged = []

    for starting_a in (0.5, 2.0, 10.0, 50.0):
        for starting_ratio in (-0.5, 0.0, 0.5):
            starting_b = starting_ratio * starting_a
            starting_gamma = np.sqrt(starting_a**2 - starting_b**2)

            # NIG mean = loc + scale*b/gamma;
            # NIG variance = scale**2 * a**2 / gamma**3.
            starting_scale = np.sqrt(starting_gamma**3 / starting_a**2)
            starting_loc = -starting_scale * starting_b / starting_gamma
            theta_start = np.array(
                [
                    np.log(starting_gamma),
                    starting_b,
                    starting_loc,
                    np.log(starting_scale),
                ]
            )

            result = minimize(
                negative_log_likelihood,
                theta_start,
                method="Nelder-Mead",
                bounds=NIG_BOUNDS,
                options={
                    "maxiter": 8000,
                    "xatol": 1e-8,
                    "fatol": 1e-8,
                },
            )

            fitted_log_density = nig_logpdf_standardized(y, result.x)
            valid = (
                bool(result.success)
                and np.isfinite(result.fun)
                and np.isfinite(fitted_log_density).all()
            )
            diagnostics.append(
                {
                    "starting_a": starting_a,
                    "starting_b/a": starting_ratio,
                    "converged": bool(result.success),
                    "valid": valid,
                    "iterations": result.nit,
                    "evaluations": result.nfev,
                    "log_likelihood": (
                        float(fitted_log_density.sum() - x.size * np.log(standardizer))
                        if np.isfinite(fitted_log_density).all()
                        else np.nan
                    ),
                    "message": str(result.message),
                }
            )
            if valid:
                converged.append(result)

    diagnostics = pd.DataFrame(diagnostics)
    print(f"\nProblem 3(a) — {stock_label}: NIG optimization diagnostics")
    print(diagnostics.to_string(index=False, float_format=lambda v: f"{v:.12g}"))

    if not converged:
        raise RuntimeError(f"{stock_label}: no NIG starting point converged.")

    best = min(converged, key=lambda result: result.fun)
    theta0, b, standardized_loc, log_standardized_scale = best.x
    gamma = float(np.exp(theta0))
    a = float(np.hypot(gamma, b))

    # Transform location and scale back to original daily return units.
    scale = float(standardizer * np.exp(log_standardized_scale))
    loc = float(center + standardizer * standardized_loc)
    alpha = a / scale
    beta = float(b) / scale
    delta = scale
    mu = loc

    log_density = nig_logpdf_standardized(y, best.x) - np.log(standardizer)
    if not np.isfinite(log_density).all():
        raise RuntimeError(f"{stock_label}: fitted log density is nonfinite.")

    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        density = np.exp(log_density)
    if not np.isfinite(density).all():
        raise RuntimeError(f"{stock_label}: fitted density is nonfinite.")

    gamma_at_upper_bound = bool(
        np.isclose(theta0, NIG_BOUNDS[0][1], rtol=0.0, atol=1e-8)
    )

    n = x.size
    k = 4
    log_likelihood = float(log_density.sum())
    fitted = {
        "a": a,
        "b": float(b),
        "gamma": gamma,
        "loc": loc,
        "scale": scale,
        "alpha": alpha,
        "beta": beta,
        "delta": delta,
        "mu": mu,
        "log_likelihood": log_likelihood,
        "AICc": 2 * k - 2 * log_likelihood + 2 * k * (k + 1) / (n - k - 1),
        "BIC": k * np.log(n) - 2 * log_likelihood,
        "gamma_at_upper_bound": gamma_at_upper_bound,
        "density_finite_at_observations": bool(np.isfinite(density).all()),
    }

    print(f"\nProblem 3(a) — {stock_label}: fitted NIG parameters and checks")
    for name, value in fitted.items():
        print(f"{name}: {value}" if isinstance(value, bool) else f"{name}: {value:.12g}")

    return {
        "parameters": fitted,
        "theta_standardized": best.x.copy(),
        "standardization_center": center,
        "standardization_scale": standardizer,
        "diagnostics": diagnostics,
        "optimizer_result": best,
    }


def valid_parameters(model, parameters):
    parameters = np.asarray(parameters, dtype=float)
    if not np.isfinite(parameters).all() or parameters[-1] <= 0:
        return False
    if model == "student_t":
        return parameters[0] > 0
    if model == "NIG":
        a, b = parameters[:2]
        return a > abs(b)
    return True


def fit_multistart(distribution, model, standardized, starts):
    candidates = []
    failures = []

    for shapes, location, scale in starts:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                parameters = distribution.fit(
                    standardized,
                    *shapes,
                    loc=location,
                    scale=scale,
                    method="MLE",
                )
                if not valid_parameters(model, parameters):
                    raise ValueError("Invalid fitted parameters.")

                log_density = distribution.logpdf(standardized, *parameters)
                if not np.isfinite(log_density).all():
                    raise ValueError("Nonfinite fitted log density.")

            candidates.append((float(log_density.sum()), parameters))
        except (ValueError, RuntimeError, FloatingPointError, OverflowError) as exc:
            failures.append(str(exc))

    if not candidates:
        raise RuntimeError(
            f"All {model} fitting starts failed. Details: {failures}"
        )

    # Retain the highest likelihood across starts within this distribution.
    return max(candidates, key=lambda candidate: candidate[0])[1]


def information_criteria(ll, k, n):
    if not np.isfinite(ll):
        raise ValueError("Log likelihood must be finite.")
    if not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError("k must be a positive integer.")
    if not isinstance(n, (int, np.integer)) or n <= k + 1:
        raise ValueError("n must be an integer greater than k + 1.")

    aic = -2.0 * ll + 2.0 * k
    aicc = aic + 2.0 * k * (k + 1) / (n - k - 1)
    bic = -2.0 * ll + k * np.log(n)
    return aic, aicc, bic

