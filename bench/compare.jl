# Cross-validation against independent software and against an equivalent
# reparameterisation of our own model.
#
# Run:  julia --project=bench bench/compare.jl
#
# Two comparisons, because no single reference covers both cases:
#
#   1. vs MixedModels.jl — a mature, independently-written REML implementation.
#      It cannot take a pedigree relationship matrix, but it CAN fit the
#      i.i.d. random-intercept model, which exercises our entire optimization
#      loop (objective, score, AI matrix, step control) against outside code.
#
#   2. pedigree vs rotated parameterisation — the animal model
#      y = Xβ + Zu + e with u ~ N(0, σ²ᵤA) is equivalent, under A = LLᵀ, to
#      y = Xβ + (ZL)w + e with w ~ N(0, σ²ᵤI). Fitting both must give identical
#      variance components and log-likelihood, but they run through completely
#      different code: sparse Henderson A⁻¹ versus a dense identity structure.
#      This is the only way to validate the pedigree path against something
#      other than itself, and it also demonstrates why the sparse path exists.

using AnimalModels
using MixedModels
using DataFrames
using StableRNGs
using LinearAlgebra
using SparseArrays
using Statistics
using Printf

function vs_mixedmodels(n, ngrp; seed=1)
    rng = StableRNG(seed)
    grp = rand(rng, 1:ngrp, n)
    x = randn(rng, n)
    g = sqrt(2.0) .* randn(rng, ngrp)
    y = 10.0 .+ 1.5 .* x .+ g[grp] .+ randn(rng, n)

    df = DataFrame(y=y, x=x, grp=string.(grp))
    t_mm = @elapsed m = fit(MixedModel, @formula(y ~ 1 + x + (1 | grp)), df;
                            REML=true, progress=false)

    X = [ones(n) x]
    Zg = sparse(1:n, grp, ones(n), n, ngrp)
    t_qg = @elapsed r = fit_reml(y, X, [RandomEffect(:grp, Zg, IdentityRelatedness(ngrp))])

    sg_mm = only(m.sigmas.grp)^2
    se_mm = m.sigma^2
    ll_mm = -objective(m) / 2

    @printf("  n=%-6d ngrp=%-4d\n", n, ngrp)
    @printf("    σ²_grp    MixedModels %11.6f   QG %11.6f   rel %.1e\n",
            sg_mm, r.σ²[1], abs(sg_mm - r.σ²[1]) / sg_mm)
    @printf("    σ²_resid  MixedModels %11.6f   QG %11.6f   rel %.1e\n",
            se_mm, r.σ²ₑ, abs(se_mm - r.σ²ₑ) / se_mm)
    @printf("    logLik    MixedModels %11.4f   QG %11.4f   abs %.1e\n",
            ll_mm, r.loglik, abs(ll_mm - r.loglik))
    @printf("    β         MixedModels %-24s QG %s\n",
            string(round.(m.beta, digits=5)), string(round.(r.β, digits=5)))
    @printf("    time      MixedModels %8.4f s        QG %8.4f s   (%.1f× )\n\n",
            t_mm, t_qg, t_mm / t_qg)
end

function pedigree_vs_rotated(q; seed=3)
    rng = StableRNG(seed)
    sire = zeros(Int, q); dam = zeros(Int, q)
    for i in (q ÷ 20 + 1):q
        sire[i] = rand(rng, 1:i-1); dam[i] = rand(rng, 1:i-1)
    end
    K = PedigreeRelatedness(sire, dam)
    A = inv(Matrix(Kinv(K)))
    L = cholesky(Symmetric(A) + 1e-10I).L

    n = q; X = [ones(n) randn(rng, n)]; Z = sparse(1.0I, n, q)
    u = sqrt(3.0) .* (L * randn(rng, q))
    y = X * [5.0, -1.5] .+ u .+ randn(rng, n)

    t_a = @elapsed ra = fit_reml(y, X, [RandomEffect(:animal, Z, K)]; trace_method=:exact)
    ZL = sparse(Matrix(Z) * Matrix(L))
    t_b = @elapsed rb = fit_reml(y, X, [RandomEffect(:w, ZL, IdentityRelatedness(q))];
                                 trace_method=:exact)

    @printf("  q=%-5d\n", q)
    @printf("    pedigree (sparse A⁻¹)  σ²ᵤ=%10.6f σ²ₑ=%10.6f logL=%12.5f  %7.3f s\n",
            ra.σ²[1], ra.σ²ₑ, ra.loglik, t_a)
    @printf("    rotated  (dense, K=I)  σ²ᵤ=%10.6f σ²ₑ=%10.6f logL=%12.5f  %7.3f s\n",
            rb.σ²[1], rb.σ²ₑ, rb.loglik, t_b)
    @printf("    agreement              σ²ᵤ rel %.1e   σ²ₑ rel %.1e   logL abs %.1e\n",
            abs(ra.σ²[1] - rb.σ²[1]) / ra.σ²[1],
            abs(ra.σ²ₑ - rb.σ²ₑ) / ra.σ²ₑ, abs(ra.loglik - rb.loglik))
    @printf("    sparse speedup         %.0f×\n\n", t_b / t_a)
end

println("="^74)
println("1. vs MixedModels.jl  (i.i.d. random intercept — independent REML code)")
println("="^74)
vs_mixedmodels(200, 20)     # warm up JIT for both
vs_mixedmodels(200, 20)
vs_mixedmodels(2000, 100)
vs_mixedmodels(10000, 500)

println("="^74)
println("2. pedigree vs rotated parameterisation  (validates the sparse A⁻¹ path)")
println("="^74)
pedigree_vs_rotated(200)
pedigree_vs_rotated(600)
