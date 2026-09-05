# Assignment 1 -- Problem 1: Reading the Shape of a Sample
#
# Run from the Assignment1 folder with:  julia --project=. src/problem1.jl

using CSV, DataFrames, Statistics, StatsBase, Distributions, Plots

data_dir = joinpath(@__DIR__, "..")   # CSVs live directly in the Assignment1 folder
fig_dir  = joinpath(@__DIR__, "..", "figures")
mkpath(fig_dir)

df = CSV.read(joinpath(data_dir, "problem1.csv"), DataFrame)
x = df.x
n = length(x)

# ---- Predict: the first four moments, before fitting anything -------------
m1 = mean(x)
m2 = var(x)                 # Statistics.var: n-1 (unbiased) divisor
sd = sqrt(m2)
sk = skewness(x)            # StatsBase.skewness: bias-corrected estimator (Week 1, sec. 4.3)
ku = kurtosis(x)            # StatsBase.kurtosis: EXCESS kurtosis, bias-corrected

println("n                = ", n)
println("mean             = ", m1)
println("variance         = ", m2)
println("std dev          = ", sd)
println("skewness         = ", sk)
println("excess kurtosis  = ", ku)
# See the write-up, part (a): skew != 0 and excess kurtosis > 0 together rule
# out the Normal, the symmetric Student's t, the symmetric NIG, and (on top
# of the sign of skew, plus x taking negative values) the lognormal.

# ---- Fit: a Normal matched to the sample mean and variance -----------------
fitted = Normal(m1, sd)

q01 = quantile(fitted, 0.01)
n_below_1pct = count(<(q01), x)
println("\nfitted Normal's 1% quantile = ", q01)
println("observations below it       = ", n_below_1pct, "   (expected ~", 0.01*n, ")")

# Right-tail check used in part (c) to show the asymmetry of the error
q99 = quantile(fitted, 0.99)
n_above_99pct = count(>(q99), x)
println("observations above 99% quantile = ", n_above_99pct, "   (expected ~", 0.01*n, ")")

# ---- Plot: histogram vs. fitted density, and a Normal QQ plot -------------
p1 = histogram(x, normalize=:pdf, bins=40, alpha=0.6, label="sample",
               legend=:topright, xlabel="x", title="Histogram vs. fitted Normal")
grid = range(minimum(x), maximum(x), length=400)
plot!(p1, grid, pdf.(fitted, grid), lw=2, color=:crimson, label="fitted Normal")

# Manual QQ plot against the fitted Normal (avoids an extra plotting dependency)
sorted_x = sort(x)
probs = [(i - 0.5)/n for i in 1:n]
theo_q = quantile.(fitted, probs)
p2 = scatter(theo_q, sorted_x, ms=2, alpha=0.5, legend=false,
             xlabel="theoretical quantile", ylabel="sample quantile",
             title="QQ plot against fitted Normal")
plot!(p2, theo_q, theo_q, color=:red, lw=1)

p = plot(p1, p2, layout=(1,2), size=(1000,420))
savefig(p, joinpath(fig_dir, "problem1_hist_vs_normal.png"))
println("\nsaved figures/problem1_hist_vs_normal.png")
