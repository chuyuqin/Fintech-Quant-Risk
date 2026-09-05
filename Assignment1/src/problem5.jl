# Assignment 1 -- Problem 5: Identifying an AR or MA Order
#
# Run from the Assignment1 folder with:  julia --project=. src/problem5.jl
#
# See the README and the "Note on Method" at the end of the write-up for why
# AR/MA fitting here uses a small, hand-rolled conditional-sum-of-squares (CSS)
# routine instead of a canned ARIMA package: CSS is a standard, well documented
# estimator, every line of it is auditable below, and -- because every model
# is fit on the same zero-padded length-n residual series -- the six AICc
# values it produces are directly comparable to each other, which is the
# property this problem actually needs.

using CSV, DataFrames, Statistics, StatsBase, Distributions, Optim, Plots

data_dir = joinpath(@__DIR__, "..")   # CSVs live directly in the Assignment1 folder
fig_dir  = joinpath(@__DIR__, "..", "figures")
mkpath(fig_dir)

df = CSV.read(joinpath(data_dir, "problem5.csv"), DataFrame)
x = df.x
n = length(x)

# ---- Predict: series, ACF, PACF --------------------------------------------
lags = collect(1:20)
acfs  = autocor(x, lags)
pacfs = pacf(x, lags)
band  = 1.96/sqrt(n)     # significance band, Week 2 sec. 6.1 / Figures 3-4

p0 = plot(x, legend=false, title="Series")
p1 = bar(lags, acfs, legend=false, title="ACF")
hline!(p1, [band, -band], ls=:dash, color=:gray)
p2 = bar(lags, pacfs, legend=false, title="PACF")
hline!(p2, [band, -band], ls=:dash, color=:gray)
p = plot(p0, p1, p2, layout=(3,1), size=(800,900))
savefig(p, joinpath(fig_dir, "problem5_acf_pacf.png"))

println("ACF  (lags 1-10):  ", round.(acfs[1:10], digits=3))
println("PACF (lags 1-10):  ", round.(pacfs[1:10], digits=3))
println("significance band = +/- ", band)
# PACF cuts off after lag 2; ACF decays -> predict AR(2) (Week 2, Table 3).

# ---- Fit: AR(1..3) and MA(1..3) by conditional sum of squares --------------
xc = x .- mean(x)   # center once; each model then only has to estimate its own AR/MA coefficients

function ar_resid(phi::Vector{Float64}, y::Vector{Float64})
    p, n = length(phi), length(y)
    e = zeros(n)
    for t in 1:n
        pred = 0.0
        for i in 1:p
            t - i >= 1 && (pred += phi[i]*y[t-i])
        end
        e[t] = y[t] - pred
    end
    return e
end

function ma_resid(theta::Vector{Float64}, y::Vector{Float64})
    q, n = length(theta), length(y)
    e = zeros(n)
    for t in 1:n
        pred = 0.0
        for i in 1:q
            t - i >= 1 && (pred += theta[i]*e[t-i])
        end
        e[t] = y[t] - pred
    end
    return e
end

function fit_css(resid_fun, k::Int, y::Vector{Float64})
    obj(params) = sum(resid_fun(params, y).^2)
    res = optimize(obj, fill(0.1, k), NelderMead(),
                    Optim.Options(iterations=20_000, g_tol=1e-10))
    return Optim.minimizer(res)
end

"""
AICc for a CSS fit with `n_coef` AR or MA coefficients. k also counts the
process mean and the innovation variance (2 more parameters), matching the
p = q + 1 (+1 for the intercept) and d = 1 (fitted variance) convention from
Week 2, sec. 5.3.
"""
function css_aicc(e::Vector{Float64}, n_coef::Int)
    n = length(e)
    sigma2 = sum(e.^2)/n
    ll = -n/2*log(sigma2*2*pi) - sum(e.^2)/(2*sigma2)
    k = n_coef + 2
    return -2*ll + 2*k + 2*k*(k+1)/(n-k-1)
end

println()
results = Dict{String,Float64}()
for p in 1:3
    phi = fit_css(ar_resid, p, xc)
    e = ar_resid(phi, xc)
    results["AR($p)"] = css_aicc(e, p)
    println("AR($p): phi=", round.(phi, digits=4), "  AICc=", round(results["AR($p)"], digits=2))
end
for q in 1:3
    theta = fit_css(ma_resid, q, xc)
    e = ma_resid(theta, xc)
    results["MA($q)"] = css_aicc(e, q)
    println("MA($q): theta=", round.(theta, digits=4), "  AICc=", round(results["MA($q)"], digits=2))
end

best = argmin(results)
println("\nBest by AICc: ", best, " = ", round(results[best], digits=2))
