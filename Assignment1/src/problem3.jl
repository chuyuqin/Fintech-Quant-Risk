# Assignment 1 -- Problem 3: Pearson Against Spearman
#
# Run from the Assignment1 folder with:  julia --project=. src/problem3.jl

using CSV, DataFrames, Statistics, StatsBase, Plots

data_dir = joinpath(@__DIR__, "..")   # CSVs live directly in the Assignment1 folder
fig_dir  = joinpath(@__DIR__, "..", "figures")
mkpath(fig_dir)

df = CSV.read(joinpath(data_dir, "problem3.csv"), DataFrame)
cols = names(df)

# ---- Predict: every pairwise scatter, plus marginal histograms ------------
plots_ = Any[]
for ci in cols, cj in cols
    if ci == cj
        push!(plots_, histogram(df[!, ci], legend=false, bins=25, color=:gray, title=ci))
    else
        push!(plots_, scatter(df[!, cj], df[!, ci], ms=2, alpha=0.4, legend=false))
    end
end
p = plot(plots_..., layout=(length(cols), length(cols)), size=(1000,1000))
savefig(p, joinpath(fig_dir, "problem3_pairplot.png"))
println("saved figures/problem3_pairplot.png")

# ---- Fit: Pearson and Spearman correlation matrices ------------------------
X = Matrix(df)
pearson  = cor(X)
spearman = corspearman(X)

println("\nPearson:")
show(stdout, "text/plain", round.(pearson, digits=3)); println()
println("\nSpearman:")
show(stdout, "text/plain", round.(spearman, digits=3)); println()

gap = abs.(pearson .- spearman)
for i in 1:size(gap,1)
    gap[i,i] = 0.0
end
println("\n|Pearson - Spearman|:")
show(stdout, "text/plain", round.(gap, digits=3)); println()

imax = argmax(gap)
println("\nLargest gap: ", cols[imax[1]], " vs ", cols[imax[2]], " = ", gap[imax])
