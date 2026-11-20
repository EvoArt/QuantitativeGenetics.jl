using LinearAlgebra

# ---------------------------------------------------------------------------
# Derived quantities from a fitted model.
#
# Standard errors of ratios (heritability especially) use the delta method with
# the sampling covariance of the variance components taken from the inverse AI
# matrix — AI⁻¹ is an estimate of the asymptotic covariance of θ̂, which is the
# usual justification in ASReml and BLUPF90 alike.
# ---------------------------------------------------------------------------

"""
    heritability(r::REMLResult; effect=1) -> (h², se)

Narrow-sense heritability of the given random effect: `σ²ᵢ / Σσ²`, with a
delta-method standard error from the fit's `vcov`.

`effect` may be an index or the effect's name. With a single random effect this
is the usual `σ²ₐ / (σ²ₐ + σ²ₑ)`.
"""
function heritability(r::REMLResult; effect=1)
    i = _effect_index(r, effect)
    all = vcat(r.σ², r.σ²ₑ)
    tot = sum(all)
    h2 = all[i] / tot

    # ∂h²/∂σ²ⱼ = (δᵢⱼ·tot − σ²ᵢ) / tot²
    g = [((j == i ? tot : 0.0) - all[i]) / tot^2 for j in eachindex(all)]
    v = dot(g, r.vcov * g)
    return (h2, v > 0 ? sqrt(v) : NaN)
end

"""
    variance_ratio(r::REMLResult; effect=1) -> σ²ᵢ / σ²ₑ

Ratio of an effect's variance to the residual variance.
"""
function variance_ratio(r::REMLResult; effect=1)
    i = _effect_index(r, effect)
    all = vcat(r.σ², r.σ²ₑ)
    return all[i] / r.σ²ₑ
end

"""
    blup(r::REMLResult; effect=1) -> Vector{Float64}

BLUP of the given random effect (the `ûᵢ` block of the final MME solution).
"""
function blup(r::REMLResult; effect=1)
    i = _effect_index(r, effect)
    i <= length(r.u) || throw(ArgumentError("effect $effect is the residual, which has no BLUP"))
    return r.u[i]
end

function _effect_index(r::REMLResult, effect)
    if effect isa Integer
        1 <= effect <= length(r.σ²) + 1 ||
            throw(BoundsError("effect index $effect out of range"))
        return Int(effect)
    elseif effect isa Symbol
        effect === :residual && return length(r.σ²) + 1
        i = findfirst(==(effect), r.names)
        i === nothing && throw(ArgumentError("no random effect named :$effect"))
        return i
    else
        throw(ArgumentError("effect must be an Integer or Symbol"))
    end
end

"""
    reliability(pev, σ²ᵢ) -> Vector{Float64}

Reliability `r² = 1 − PEV/σ²ᵢ` of each predicted breeding value, given the
prediction error variances.
"""
reliability(pev::AbstractVector, σ²::Real) = clamp.(1 .- pev ./ σ², 0.0, 1.0)

"""
    accuracy(pev, σ²ᵢ) -> Vector{Float64}

Accuracy of prediction, `sqrt(reliability)`.
"""
accuracy(pev::AbstractVector, σ²::Real) = sqrt.(reliability(pev, σ²))
