using LinearAlgebra
using SparseArrays
using SparseArrays: CHOLMOD
using Printf
using Random

# ---------------------------------------------------------------------------
# REML: objective, score, average information, and the AI/EM iteration.
#
# For y = Xβ + Σᵢ Zᵢuᵢ + e, V = Σᵢ σ²ᵢZᵢKᵢZᵢᵀ + σ²ₑI, the restricted
# log-likelihood is
#
#     −2logL = log|V| + log|XᵀV⁻¹X| + yᵀPy,     P = V⁻¹ − V⁻¹X(XᵀV⁻¹X)⁻¹XᵀV⁻¹.
#
# Evaluating that directly means an n×n V. Henderson's identity turns it into
# quantities the MME already produce:
#
#     log|V| + log|XᵀV⁻¹X| = log|C| + (n − p)·log σ²ₑ + Σᵢ (qᵢ·log σ²ᵢ + log|Kᵢ|)
#
# with C the MME coefficient matrix at λᵢ = σ²ₑ/σ²ᵢ. Every term is free:
# log|C| is the cached factor's log-determinant, log|Kᵢ| was precomputed at
# construction (and for a pedigree it is Σ log dᵢ — no linear algebra at all).
#
#     yᵀPy = (yᵀy − β̂ᵀXᵀy − Σᵢ ûᵢᵀZᵢᵀy) / σ²ₑ
#
# falls straight out of the solution vector. So one factorization plus one solve
# gives the entire objective.
#
# SCORE. ∂(−2logL)/∂σ²ᵢ needs tr(Kᵢ⁻¹C^{uᵢuᵢ}) — the ONLY term requiring more
# than the factor, and the reason sparseinv.jl exists.
#
# AVERAGE INFORMATION. The AI matrix replaces the Hessian by the average of the
# observed and expected information, and that average has the remarkable
# property that all trace terms cancel:
#
#     AIᵢⱼ = fᵢᵀPfⱼ / (2σ²ₑ),   fᵢ = Zᵢûᵢ,  f_e = ê
#
# Each Pfᵢ is obtained by absorbing fᵢ as a pseudo-observation through the MME —
# a solve against the ALREADY-COMPUTED factor. So the entire second-order
# information costs k+1 solves and no traces at all. This identity is what makes
# AI-REML fast, and it is precisely what an autodiff system cannot discover: AD
# would differentiate the factorization instead, which is strictly more work.
# ---------------------------------------------------------------------------

"""
    REMLResult

Outcome of an AI-REML fit.

- `σ²`        — estimated variance component of each random effect, in the order given.
- `σ²ₑ`       — estimated residual variance.
- `names`     — effect names.
- `β`         — BLUE of the fixed effects.
- `u`         — BLUP of each random effect.
- `loglik`    — restricted log-likelihood at the optimum (i.e. `−0.5·(−2logL)`).
- `iterations`, `converged`
- `vcov`      — sampling covariance of `[σ²...; σ²ₑ]`, from the inverse AI matrix.
- `se`        — standard errors of the variance components (`sqrt` of `vcov`'s diagonal).
"""
struct REMLResult
    names::Vector{Symbol}
    σ²::Vector{Float64}
    σ²ₑ::Float64
    β::Vector{Float64}
    u::Vector{Vector{Float64}}
    loglik::Float64
    iterations::Int
    converged::Bool
    vcov::Matrix{Float64}
    se::Vector{Float64}
end

function Base.show(io::IO, r::REMLResult)
    println(io, "REMLResult (", r.converged ? "converged" : "NOT converged",
            " in ", r.iterations, " iterations)")
    @printf(io, "  logLik = %.6f\n", r.loglik)
    println(io, "  Variance components:")
    tot = sum(r.σ²) + r.σ²ₑ
    for (i, nm) in enumerate(r.names)
        @printf(io, "    %-14s %12.6f  (se %9.6f)  prop %6.4f\n",
                String(nm), r.σ²[i], r.se[i], r.σ²[i] / tot)
    end
    @printf(io, "    %-14s %12.6f  (se %9.6f)  prop %6.4f\n",
            "residual", r.σ²ₑ, r.se[end], r.σ²ₑ / tot)
end

"""
    REMLWorkspace(ws::MMEWorkspace)

Scratch space for a REML fit: the MME workspace plus the reusable buffers for
the working variates and the multi-right-hand-side AI solve. Allocated once so
the iteration itself allocates nothing of consequence.
"""
mutable struct REMLWorkspace{W}
    const mme::W
    const k::Int                       # number of random effects
    const Wmat::SparseMatrixCSC{Float64,Int}   # [X Z₁ Z₂ ⋯], for pseudo-observation RHS
    const F::Matrix{Float64}           # n × (k+1) working variates
    const RHS::Matrix{Float64}         # m × (k+1) pseudo-observation RHS
    const PF::Matrix{Float64}          # n × (k+1) P·F
    resid::Vector{Float64}
    # Selected inverse of the CURRENT C, plus the θ it was computed at. The
    # score and the EM fallback both need it, and recomputing costs a full LDLᵀ
    # plus Takahashi sweep (~128 ms at q = 1500) — by far the most expensive
    # thing in an iteration, so it is computed at most once per θ.
    sinv::Union{Nothing,SelectedInverse}
    sinv_θ::Vector{Float64}
end

function REMLWorkspace(ws::MMEWorkspace, X::AbstractMatrix)
    k = length(ws.effects)
    n, m = ws.n, ws.p + ws.q
    Xs = SparseMatrixCSC{Float64,Int}(sparse(X))
    Wmat = k == 0 ? Xs : hcat(Xs, [e.Z for e in ws.effects]...)
    return REMLWorkspace(ws, k, Wmat,
                         Matrix{Float64}(undef, n, k + 1),
                         Matrix{Float64}(undef, m, k + 1),
                         Matrix{Float64}(undef, n, k + 1),
                         Vector{Float64}(undef, n),
                         nothing, Float64[])
end

# The selected inverse of the current C, computed once per θ and reused by both
# the score and the EM fallback.
function _get_sinv!(rw::REMLWorkspace, θ::AbstractVector{Float64})
    if rw.sinv === nothing || rw.sinv_θ != θ
        rw.sinv = selected_inverse(rw.mme.C)
        rw.sinv_θ = collect(θ)
    end
    return rw.sinv
end

"""
    reml_objective!(rw, y, θ) -> (minus2logL, ûs, β̂, residual)

Evaluate `−2logL` at `θ = [σ²₁ … σ²ₖ, σ²ₑ]`, leaving the MME solution in place.

Assembles `C` at the implied ratios, factorizes (reusing the cached symbolic
factor after the first call), solves, and combines `log|C|`, the precomputed
`log|Kᵢ|`, and `yᵀPy` as derived in this file's header.
"""
function reml_objective!(rw::REMLWorkspace, y::AbstractVector, θ::AbstractVector{Float64})
    ws = rw.mme
    k = rw.k
    σ²ₑ = θ[end]
    λ = [σ²ₑ / θ[i] for i in 1:k]

    update_mme!(ws, λ)
    # C's values just changed, so any cached selected inverse is stale. The θ
    # key alone is not enough: step-halving evaluates several candidate θ and
    # would otherwise leave the sweep describing a different C than the one the
    # score is about to be computed from.
    rw.sinv = nothing
    empty!(rw.sinv_θ)
    F = factorize_mme!(ws)
    ws.sol = F \ ws.rhs

    n, p = ws.n, ws.p
    # yᵀPy = (yᵀy − ŝᵀr) / σ²ₑ, where ŝᵀr = β̂ᵀXᵀy + Σ ûᵢᵀZᵢᵀy.
    yPy = (ws.yty - dot(ws.sol, ws.rhs)) / σ²ₑ

    # log|V| + log|XᵀV⁻¹X| = log|C| − m·log σ²ₑ + n·log σ²ₑ + Σᵢ(qᵢ·log σ²ᵢ + log|Kᵢ|)
    #
    # The −m·log σ²ₑ is easy to lose: C here is built in λ-scaling (the σ²ₑ
    # factor divided out of every block), so recovering log|V| means putting
    # back one σ²ₑ per row of C — all m = p + q of them — before adding the
    # n·log σ²ₑ that log|V| itself carries. Verified against a dense
    # log|V| + log|XᵀV⁻¹X| at several θ; the two agree to machine precision,
    # whereas dropping the −m term is exact only where σ²ᵤ = σ²ₑ, which makes
    # this a nasty bug to spot from a single test point.
    m = p + ws.q
    obj = logdet(F) - m * log(σ²ₑ) + n * log(σ²ₑ)
    for i in 1:k
        obj += ws.effects[i].q * log(θ[i]) + logdetK(ws.effects[i].K)
    end
    obj += yPy
    return obj
end

"""
    reml_gradient_ai!(rw, y, θ; trace_method=:exact, nprobe=50, rng)
        -> (score, AI, minus2logL)

Score vector and average-information matrix at `θ`, assuming
[`reml_objective!`](@ref) has just been called at the same `θ` (so the factor
and solution are current).

The score's trace terms come from the selected inverse (`:exact`) or Hutchinson
probing (`:hutchinson`). The AI matrix costs `k+1` solves against the existing
factor and no traces — see the header.

Returned quantities are derivatives of `−2logL`, so the Newton step is
`θ ← θ − AI⁻¹·(score/2)`; the driver handles the scaling.
"""
function reml_gradient_ai!(rw::REMLWorkspace, y::AbstractVector, θ::AbstractVector{Float64};
                           trace_method::Symbol=:exact, nprobe::Int=50,
                           rng=Random.default_rng())
    ws = rw.mme
    k = rw.k
    σ²ₑ = θ[end]
    n, p, m = ws.n, ws.p, ws.p + ws.q
    F = ws.factor

    # Residual ê = y − Xβ̂ − Σ Zᵢûᵢ = y − W·ŝ
    mul!(rw.resid, rw.Wmat, ws.sol)
    @inbounds for t in 1:n
        rw.resid[t] = y[t] - rw.resid[t]
    end

    # --- working variates -------------------------------------------------
    # AIᵢⱼ = ½·yᵀP·(∂V/∂θᵢ)·P·(∂V/∂θⱼ)·Py, so the correct variate for component
    # i is fᵢ = (∂V/∂θᵢ)·Py.
    #
    # For a random effect, ∂V/∂σ²ᵢ = ZᵢKᵢZᵢᵀ and the MME give ûᵢ = σ²ᵢ·KᵢZᵢᵀPy,
    # hence (∂V/∂σ²ᵢ)Py = Zᵢûᵢ / σ²ᵢ — the 1/σ²ᵢ is NOT optional. Likewise
    # ∂V/∂σ²ₑ = I and Py = ê/σ²ₑ, so the residual variate is ê/σ²ₑ.
    # Dropping these scalings leaves an AI matrix that is wrong by factors of
    # σ²ᵢσ²ⱼ, which still looks plausible (right order of magnitude, symmetric,
    # positive definite) but produces erratic steps that increase −2logL.
    for i in 1:k
        ui = random_effect(ws, i)
        mul!(view(rw.F, :, i), ws.effects[i].Z, ui)
        rw.F[:, i] ./= θ[i]
    end
    @inbounds for t in 1:n
        rw.F[t, k+1] = rw.resid[t] / σ²ₑ
    end

    # --- P·F via pseudo-observations, one multi-RHS solve -----------------
    # Absorbing f as an observation: solve C·a = Wᵀf, then Pf = (f − W·a)/σ²ₑ.
    mul!(rw.RHS, transpose(rw.Wmat), rw.F)
    A = F \ rw.RHS                        # m × (k+1), one CHOLMOD call
    mul!(rw.PF, rw.Wmat, A)
    @inbounds for c in 1:(k+1), t in 1:n
        rw.PF[t, c] = (rw.F[t, c] - rw.PF[t, c]) / σ²ₑ
    end

    # --- average information ---------------------------------------------
    # With fᵢ = (∂V/∂θᵢ)Py already formed above, AIᵢⱼ = ½·fᵢᵀ(Pfⱼ) directly.
    AI = Matrix{Float64}(undef, k + 1, k + 1)
    @inbounds for a in 1:(k+1), b in a:(k+1)
        v = 0.5 * dot(view(rw.F, :, a), view(rw.PF, :, b))
        AI[a, b] = v
        AI[b, a] = v
    end

    # --- score ------------------------------------------------------------
    # ∂(−2logL)/∂σ²ᵢ = [qᵢ − tr(Kᵢ⁻¹C^{uᵢuᵢ})·σ²ₑ/σ²ᵢ]/σ²ᵢ − ûᵢᵀKᵢ⁻¹ûᵢ/σ²ᵢ²
    # ∂(−2logL)/∂σ²ₑ = [(n − p) − Σᵢ (qᵢ − tr(...)σ²ₑ/σ²ᵢ)]/σ²ₑ − êᵀê/σ²ₑ²
    score = Vector{Float64}(undef, k + 1)
    sinv = nothing
    if k > 0 && trace_method === :exact
        # LDLᵀ of the *current* C plus the Takahashi sweep. This is a second
        # factorization per iteration; it is the price of exact traces and is
        # the dominant cost on large pedigrees (hence :hutchinson). Cached per
        # θ so the EM fallback does not pay for it a second time.
        sinv = _get_sinv!(rw, θ)
    end

    sum_eff = 0.0
    for i in 1:k
        e = ws.effects[i]
        Ki = Kinv(e.K)
        ui = random_effect(ws, i)
        off = ws.offsets[i]
        tr_i = if trace_method === :exact
            trace_product(sinv, Ki, off)
        elseif trace_method === :hutchinson
            hutchinson_trace(F, Ki, off, nprobe; rng=rng)
        else
            throw(ArgumentError("trace_method must be :exact or :hutchinson"))
        end
        uKu = dot(ui, Ki * ui)
        # tr(Kᵢ⁻¹ C^{uᵢuᵢ}) with C in λ-scaling: C⁻¹ block is PEV/σ²ₑ, so the
        # effective degrees of freedom absorbed is tr·σ²ₑ/σ²ᵢ.
        eff_i = e.q - tr_i * σ²ₑ / θ[i]
        sum_eff += eff_i
        score[i] = eff_i / θ[i] - uKu / (θ[i]^2)
    end
    score[k+1] = ((n - p) - sum_eff) / σ²ₑ - dot(rw.resid, rw.resid) / (σ²ₑ^2)

    return score, AI
end

"""
    fit_reml(y, X, effects; kwargs...) -> REMLResult

Estimate variance components by AI-REML and return the fit with BLUE/BLUP.

Keyword arguments:

- `start`         — initial `[σ²₁ … σ²ₖ, σ²ₑ]`. Default splits `var(y)` evenly
                    between the residual and the random effects.
- `maxiter`       — iteration cap (default 50).
- `tol`           — convergence tolerance on the relative change in `θ` (1e-8).
- `gtol`          — convergence tolerance on the scaled score (1e-6). Checked
                    before each step, so the iteration stops as soon as the
                    optimum is reached instead of taking further no-op steps
                    that each cost a full selected-inversion sweep.
- `trace_method`  — `:auto` (default), `:exact` (Takahashi) or `:hutchinson`.
                    `:auto` inspects the factor's fill and switches to
                    Hutchinson probing above 20k nonzeros in `L`, which
                    measurement puts well below the point where the exact sweep
                    starts to dominate.
- `nprobe`        — probes per trace when Hutchinson probing is used (100).
                    Accuracy improves only as `√nprobe` while cost is linear:
                    measured at q = 2000, going from 50 to 800 probes cut the
                    relative error in `σ²ₐ` from 4e-4 to 2e-4 while costing 4×
                    the time. Raise it only if you need variance components to
                    better than ~1e-3, and prefer `trace_method = :exact` for
                    that instead.
- `ordering`      — MME ordering strategy, see [`build_mme`](@ref).
- `verbose`       — print per-iteration `−2logL` and components.

The iteration is AI-Newton on the LOG scale (so variances stay positive by
construction), with step halving whenever `−2logL` fails to decrease, and an
EM-REML fallback step when the AI direction is not a descent direction. EM is
slower but monotone and cannot leave the parameter space; the hybrid is what
makes AI-REML robust on hard data, and it reuses the trace terms the score has
already paid for.
"""
function fit_reml(y::AbstractVector, X::AbstractMatrix, effects::Vector{<:RandomEffect};
                  start::Union{Nothing,AbstractVector}=nothing,
                  maxiter::Int=50, tol::Float64=1e-8, gtol::Float64=1e-6,
                  trace_method::Symbol=:auto, nprobe::Int=100,
                  ordering::Symbol=:auto, verbose::Bool=false,
                  rng=Random.default_rng())
    k = length(effects)
    ws = build_mme(y, X, effects; ordering=ordering)
    rw = REMLWorkspace(ws, X)

    # Pick the trace method from the factor's actual fill, not from q, because
    # the Takahashi sweep's cost tracks nnz(L) rather than the problem size.
    #
    # Measured end to end on pedigree animal models (50 probes), exact/hutch:
    #   nnz(L)   6k → 1.5×    24k → 1.1×    36k → 1.5×
    #           66k → 2.3×   106k → 3.9×   149k → 2.1×
    #          705k → 15×
    # Hutchinson is faster at every size measured and the margin grows, while
    # the estimates agree to 3–4 decimals. The threshold is therefore low: exact
    # is kept only for small problems, where the sweep is cheap in absolute
    # terms and its deterministic convergence is worth having (a stochastic
    # gradient makes the stopping rule fuzzy, which matters more when there are
    # few observations per variance component).
    if trace_method === :auto
        update_mme!(ws, fill(1.0, k))
        nzL = nnz(factorize_mme!(ws))
        trace_method = nzL > 20_000 ? :hutchinson : :exact
        verbose && @printf("trace_method = :%s (nnz(L) = %d)\n", trace_method, nzL)
    end

    vy = _var(y)
    θ = if start === nothing
        vcat(fill(vy / (2 * max(k, 1)), k), vy / 2)
    else
        length(start) == k + 1 ||
            throw(DimensionMismatch("start must have $(k+1) entries, got $(length(start))"))
        collect(Float64, start)
    end
    θ = max.(θ, 1e-10)

    obj = reml_objective!(rw, y, θ)
    converged = false
    iter = 0
    AI = Matrix{Float64}(undef, k + 1, k + 1)

    for it in 1:maxiter
        iter = it
        score, AI = reml_gradient_ai!(rw, y, θ; trace_method=trace_method,
                                      nprobe=nprobe, rng=rng)

        # Gradient test, checked BEFORE proposing a step. At the optimum the
        # score is zero, so this stops as soon as the answer is reached rather
        # than after the θ-step test happens to fall below tolerance. Without
        # it a converged fit still burned 7 further iterations at q = 2000 —
        # identical θ and −2logL each time — and each of those iterations pays
        # for a full Takahashi sweep, by far the most expensive thing here.
        # Scaled by θ because the score has units of 1/variance.
        if maximum(abs.(Diagonal(θ) * score)) < gtol
            converged = true
            break
        end

        # AI-Newton on the log scale. With φ = log θ the chain rule gives
        # gφ = diag(θ)·g and AIφ = diag(θ)·AI·diag(θ), and the update stays
        # positive for any step length.
        #
        # The factor of 2 is load-bearing: AI approximates HALF the Hessian of
        # −2logL (it is the information matrix of logL), so the Newton step is
        # −AI⁻¹·(g/2), i.e. −(2·AI)⁻¹g. Omitting it halves every step and turns
        # quadratic convergence into a crawl that limps along on step-halving —
        # ~40+ iterations instead of ~5, with the components zig-zagging.
        Dθ = Diagonal(θ)
        AIφ = Dθ * AI * Dθ
        gφ = Dθ * score
        δ = try
            -((2 .* AIφ) \ gφ)
        catch
            fill(NaN, k + 1)
        end
        if any(!isfinite, δ) || dot(δ, gφ) >= 0
            # Not a usable descent direction — fall back to an EM step, which is
            # monotone and always feasible.
            δ = _em_direction(rw, θ, log.(θ))
        end

        # Step halving on −2logL.
        step = 1.0
        newθ = θ
        newobj = obj
        ok = false
        for _ in 1:20
            cand = exp.(log.(θ) .+ step .* δ)
            cand = clamp.(cand, 1e-12, 1e12)
            cobj = reml_objective!(rw, y, cand)
            if isfinite(cobj) && cobj <= obj + 1e-10
                newθ, newobj, ok = cand, cobj, true
                break
            end
            step /= 2
        end
        if !ok
            # Could not improve: recompute at the incumbent so the factor and
            # solution match θ, then stop.
            reml_objective!(rw, y, θ)
            converged = true
            break
        end

        rel = maximum(abs.(newθ .- θ) ./ max.(abs.(θ), 1e-8))
        objdrop = obj - newobj
        θ, obj = newθ, newobj
        verbose && @printf("iter %3d  -2logL = %.8f   θ = %s\n", it, obj,
                           string(round.(θ, digits=6)))
        # Converged if the parameters stopped moving OR the objective did.
        # The second test matters: once −2logL is flat to ~1e-9 there is nothing
        # left to gain, and continuing only spends Takahashi sweeps.
        if rel < tol || objdrop < 1e-9
            converged = true
            break
        end
    end

    # Final state at θ, so the solution/factor are consistent with what we report.
    obj = reml_objective!(rw, y, θ)
    score, AI = reml_gradient_ai!(rw, y, θ; trace_method=trace_method,
                                  nprobe=nprobe, rng=rng)

    vcov = try
        M = inv(AI)
        0.5 .* (M .+ transpose(M))
    catch
        fill(NaN, k + 1, k + 1)
    end
    se = [vcov[i, i] > 0 ? sqrt(vcov[i, i]) : NaN for i in 1:(k+1)]

    β = collect(fixed_effects(ws))
    us = [collect(random_effect(ws, i)) for i in 1:k]
    # `reml_objective!` omits the (n−p)·log(2π) normalizing constant, which is
    # free of θ and so cancels in every comparison, derivative and step test.
    # It is added back HERE, at the reporting boundary, so that `loglik` is on
    # the same scale as other REML software: verified against MixedModels.jl,
    # whose objective exceeded ours by exactly (n−p)·log(2π) at n = 200, 2000
    # and 10000. Adding it inside the objective would just cost an operation per
    # evaluation for no numerical benefit.
    m2ll = obj + (ws.n - ws.p) * log(2π)
    return REMLResult([e.name for e in effects], θ[1:k], θ[end], β, us,
                      -0.5 * m2ll, iter, converged, vcov, se)
end

# EM-REML direction in log space. The EM update for a variance component is
#   σ²ᵢ⁺ = (ûᵢᵀKᵢ⁻¹ûᵢ + tr(Kᵢ⁻¹C^{uᵢuᵢ})·σ²ₑ) / qᵢ
# which is guaranteed positive and monotone. We express it as a log-space step
# so the driver's step-halving loop can use it unchanged.
function _em_direction(rw::REMLWorkspace, θ::Vector{Float64}, logθ::Vector{Float64})
    ws = rw.mme
    k = rw.k
    σ²ₑ = θ[end]
    n, p = ws.n, ws.p
    sinv = k > 0 ? _get_sinv!(rw, θ) : nothing
    new = similar(θ)
    sum_eff = 0.0
    for i in 1:k
        e = ws.effects[i]
        Ki = Kinv(e.K)
        ui = random_effect(ws, i)
        tr_i = trace_product(sinv, Ki, ws.offsets[i])
        uKu = dot(ui, Ki * ui)
        new[i] = (uKu + tr_i * σ²ₑ) / e.q
        sum_eff += e.q - tr_i * σ²ₑ / θ[i]
    end
    new[k+1] = dot(rw.resid, rw.resid) / max(n - p - sum_eff, 1e-8)
    new .= clamp.(new, 1e-12, 1e12)
    return log.(new) .- logθ
end

_var(y) = (n = length(y); m = sum(y) / n; sum(abs2, y .- m) / max(n - 1, 1))
