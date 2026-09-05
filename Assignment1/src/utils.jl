# Small shared helpers used by more than one problem script.
#
# The FittedModel/aicc pattern here is deliberately the same one used in
# library/fitted_model.jl elsewhere in the class repo: an error distribution,
# plus (optionally) a vector of regression coefficients that sit alongside it.
# k, the parameter count charged by AICc, is the number of parameters of the
# error distribution plus the number of regression coefficients (if any).

using Distributions

struct FittedModel
    beta::Union{Vector{Float64},Nothing}
    errorModel::UnivariateDistribution
end

"""
    aicc(m::FittedModel, data)

AICc = AIC + 2k(k+1)/(n-k-1), evaluated at the log-likelihood of `data` under
`m.errorModel`, with k counting both the distribution's own parameters and
(if present) the regression coefficients in `m.beta`.
"""
function aicc(m::FittedModel, data::AbstractVector{<:Real})
    d = m.errorModel
    ll = sum(logpdf.(d, data))
    n = length(data)
    k = length(Distributions.params(d))
    if m.beta !== nothing
        k += length(m.beta)
    end
    return -2*ll + 2*k + 2*k*(k+1)/(n-k-1)
end
