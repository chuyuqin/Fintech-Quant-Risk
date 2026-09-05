# Assignment 1 -- Problem 2: A Regression Whose Errors Are Not Normal
#
# Run from the Assignment1 folder with:  julia --project=. src/problem2.jl

using CSV, DataFrames, Statistics, StatsBase, Distributions, GLM, Plots, JuMP, Ipopt

data_dir = joinpath(@__DIR__, "..")   # CSVs live directly in the Assignment1 folder
fig_dir  = joinpath(@__DIR__, "..", "figures")
mkpath(fig_dir)
include(joinpath(@__DIR__, "utils.jl"))

df = CSV.read(joinpath(data_dir, "problem2.csv"), DataFrame)
x, y = df.x, df.y
n = length(x)

# ---- Predict: scatter of y against x ---------------------------------------
p0 = scatter(x, y, ms=3, alpha=0.6, legend=false, xlabel="x", ylabel="y",
             title="problem2: y vs x")
savefig(p0, joinpath(fig_dir, "problem2_scatter.png"))

# ---- Fit 1: OLS, with standard errors --------------------------------------
ols = lm(@formula(y ~ x), df)
println(ols)
alpha_ols, beta_ols = coef(ols)
se_ols = stderror(ols)
resid_ols = residuals(ols)

# ---- Fit 2: MLE under a Normal error ---------------------------------------
# For a model linear in its parameters with a Normal error, the log-likelihood
# is maximized at exactly the OLS beta (Week 2, sec. 4.3): the only term
# involving beta is the sum of squared errors, with a negative sign in front
# of it, so maximizing ll over beta IS minimizing the sum of squares. The two
# fits differ only in sigma: MLE's sigma uses an n divisor, OLS's s.e. uses n-p.
sigma_mle = sqrt(sum(resid_ols.^2) / n)
normal_error_model = Normal(0.0, sigma_mle)
fm_normal = FittedModel([alpha_ols, beta_ols], normal_error_model)

println("\nNormal MLE: alpha=", alpha_ols, " beta=", beta_ols, " sigma=", sigma_mle)
println("AICc (Normal MLE) = ", aicc(fm_normal, resid_ols))

# ---- Fit 3: MLE under a Student's t error -----------------------------------
# We fit alpha, beta, the t's scale, and its degrees of freedom nu jointly by
# maximum likelihood, following the same JuMP/Ipopt pattern used throughout
# the class library (see library/fitted_model.jl :: fit_regression_t).
X = hcat(fill(1.0, n), x)
b_start = X \ y
e0 = y - X*b_start
start_nu = max(6.0/kurtosis(e0) + 4, 3.0)
start_s  = sqrt(var(e0) * (start_nu - 2)/start_nu)

function general_t_ll(mu, s, nu, resid)
    td = TDist(nu)*s + mu
    sum(log.(pdf.(td, resid)))
end

mle = Model(Ipopt.Optimizer)
set_silent(mle)
@variable(mle, a, start = b_start[1])
@variable(mle, b, start = b_start[2])
@variable(mle, s >= 1e-6, start = start_s)
@variable(mle, nu >= 2.0001, start = start_nu)

function tll(a, b, s, nu)
    resid = y .- (a .+ b .* x)
    general_t_ll(0.0, s, nu, resid)
end
register(mle, :tll, 4, tll; autodiff=true)
@NLobjective(mle, Max, tll(a, b, s, nu))
optimize!(mle)

alpha_t, beta_t, s_t, nu_t = value(a), value(b), value(s), value(nu)
t_error_model = TDist(nu_t)*s_t
resid_t = y .- (alpha_t .+ beta_t .* x)
fm_t = FittedModel([alpha_t, beta_t], t_error_model)

println("\nt MLE: alpha=", alpha_t, " beta=", beta_t, " scale=", s_t, " nu=", nu_t)
println("AICc (t MLE) = ", aicc(fm_t, resid_t))

println("\n--- alpha, beta comparison ---")
println("OLS:        ", (alpha_ols, beta_ols))
println("Normal MLE: ", (alpha_ols, beta_ols), "  (identical to OLS, see note above)")
println("t MLE:      ", (alpha_t, beta_t))

# ---- Reconcile: quantiles of the two fitted error distributions -----------
for p in (0.95, 0.995)
    println("p=", p, "  Normal q=", quantile(normal_error_model, p),
            "  t q=", quantile(t_error_model, p))
end
