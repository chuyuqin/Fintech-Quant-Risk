# Assignment 1 -- Problem 4: Conditional Distributions
#
# Run from the Assignment1 folder with:  julia --project=. src/problem4.jl

using CSV, DataFrames, Statistics, Plots

data_dir = joinpath(@__DIR__, "..")   # CSVs live directly in the Assignment1 folder
fig_dir  = joinpath(@__DIR__, "..", "figures")
mkpath(fig_dir)

df = CSV.read(joinpath(data_dir, "problem4.csv"), DataFrame)
x1, x2 = df.x1, df.x2
n = length(x1)

# ---- Predict: sample covariance matrix, nothing else yet -------------------
Sigma = cov(hcat(x1, x2))
s11, s12, s22 = Sigma[1,1], Sigma[1,2], Sigma[2,2]
println("Sigma = ", Sigma)

# Partitioned result (Week 2, sec. 2.4), with x1 as block 1 and x2 as block 2:
#   Var(x2 | x1) = Sigma22 - Sigma21 * Sigma11^-1 * Sigma12
cond_var = s22 - s12^2/s11
reduction_factor = cond_var / s22    # = 1 - rho^2; does NOT depend on the value of x1 -- see write-up part (b)
println("conditional variance of x2 | x1 = ", cond_var, "   (unconditional variance = ", s22, ")")
println("uncertainty reduction factor    = ", reduction_factor, "  (= 1 - rho^2)")

# ---- Fit: conditional mean line, 95% band, scatter -------------------------
mu1, mu2 = mean(x1), mean(x2)
beta = s12/s11                       # Sigma12 * Sigma11^-1 -- also the OLS slope of x2 on x1
band = 1.96 * sqrt(cond_var)

xs = range(minimum(x1), maximum(x1), length=200)
cmean_line = mu2 .+ beta .* (xs .- mu1)

p = scatter(x1, x2, ms=2.5, alpha=0.35, label="data", xlabel="x1", ylabel="x2",
            title="Conditional mean of x2 given x1, with 95% band")
plot!(p, xs, cmean_line, lw=2, color=:crimson, label="E[x2|x1]")
plot!(p, xs, cmean_line .+ band, lw=1, ls=:dash, color=:crimson, label="95% band")
plot!(p, xs, cmean_line .- band, lw=1, ls=:dash, color=:crimson, label=false)
savefig(p, joinpath(fig_dir, "problem4_conditional.png"))
println("saved figures/problem4_conditional.png")

# ---- Reconcile: coverage overall and split by distance from mean(x1) -------
cmean_at_data = mu2 .+ beta .* (x1 .- mu1)
inside = abs.(x2 .- cmean_at_data) .<= band
println("\noverall coverage = ", mean(inside))

z = abs.(x1 .- mu1) ./ sqrt(s11)
buckets = [
    ("<= 1 sd", z .<= 1),
    ("1-2 sd",  (z .> 1) .& (z .<= 2)),
    ("> 2 sd",  z .> 2),
]
resid = x2 .- cmean_at_data
for (name, b) in buckets
    println(name, ":  n=", count(b), "  coverage=", mean(inside[b]),
            "  resid std=", std(resid[b]))
end
