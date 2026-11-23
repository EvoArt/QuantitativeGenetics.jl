# Benchmarks for the AI-REML animal model.
#
# Run:  julia --project=bench bench/bench.jl
#       BENCH_Q=10000 julia --project=bench bench/bench.jl
#
# Reports the per-iteration split (assembly / factorization / objective /
# selected inverse / gradient+AI) plus end-to-end fits, so that every claim in
# the source comments has a number behind it. Per the house rule: measure,
# don't assume.

using AnimalModels
using LinearAlgebra
using SparseArrays
using StableRNGs
using Statistics
using Printf

function make_data(q; seed=2024, σu=3.0, σe=1.0)
    rng = StableRNG(seed)
    nf = max(5, q ÷ 20)
    sire = zeros(Int, q); dam = zeros(Int, q)
    for i in (nf+1):q
        sire[i] = rand(rng, 1:i-1)
        dam[i] = rand(rng, 1:i-1)
    end
    K = PedigreeRelatedness(sire, dam)
    X = [ones(q) randn(rng, q)]
    Z = sparse(1.0I, q, q)
    L = cholesky(Symmetric(inv(Matrix(Kinv(K)))) + 1e-9I).L
    u = sqrt(σu) .* (L * randn(rng, q))
    y = X * [5.0, -1.5] .+ u .+ sqrt(σe) .* randn(rng, q)
    return y, X, [RandomEffect(:animal, Z, K)], u
end

bench(f, n=5) = minimum(@elapsed(f()) for _ in 1:n)

function iteration_profile(q)
    y, X, eff, _ = make_data(q)
    ws = build_mme(y, X, eff)
    rw = REMLWorkspace(ws, X)
    θ = [3.0, 1.0]
    reml_objective!(rw, y, θ)

    λ = [1 / 3]
    t_asm = bench(() -> update_mme!(ws, λ), 50)
    t_fac = bench(() -> factorize_mme!(ws))
    t_obj = bench(() -> reml_objective!(rw, y, θ))
    t_si  = bench(() -> selected_inverse(ws.C), 3)
    t_gr  = bench(() -> (reml_objective!(rw, y, θ);
                         reml_gradient_ai!(rw, y, θ; trace_method=:exact)), 3)
    t_gh  = bench(() -> (reml_objective!(rw, y, θ);
                         reml_gradient_ai!(rw, y, θ; trace_method=:hutchinson,
                                           nprobe=50)), 3)

    @printf("\n  q = %-7d  nnz(C) = %-9d nnz(L) = %-9d fill = %.1f×\n",
            q, nnz(ws.C), nnz(ws.factor), nnz(ws.factor) / nnz(ws.C))
    @printf("    assembly (update_mme!)   %9.3f ms\n", t_asm * 1e3)
    @printf("    numeric factorization    %9.3f ms\n", t_fac * 1e3)
    @printf("    objective (asm+fac+solve)%9.3f ms\n", t_obj * 1e3)
    @printf("    selected inverse         %9.3f ms\n", t_si * 1e3)
    @printf("    gradient+AI  (exact)     %9.3f ms\n", t_gr * 1e3)
    @printf("    gradient+AI  (hutch,50)  %9.3f ms\n", t_gh * 1e3)
end

function endtoend(q)
    y, X, eff, u = make_data(q)
    fit_reml(y, X, eff; trace_method=:hutchinson, nprobe=20)  # warm up

    te = @elapsed re = fit_reml(y, X, eff; trace_method=:exact)
    th = @elapsed rh = fit_reml(y, X, eff; trace_method=:hutchinson, nprobe=50,
                                rng=StableRNG(1))
    ta = @elapsed ra = fit_reml(y, X, eff)   # :auto

    @printf("\n  q = %d\n", q)
    @printf("    exact      %8.3f s  (%2d it)  σ²ᵤ=%7.4f σ²ₑ=%7.4f  cor=%.3f\n",
            te, re.iterations, re.σ²[1], re.σ²ₑ, cor(blup(re), u))
    @printf("    hutchinson %8.3f s  (%2d it)  σ²ᵤ=%7.4f σ²ₑ=%7.4f  cor=%.3f\n",
            th, rh.iterations, rh.σ²[1], rh.σ²ₑ, cor(blup(rh), u))
    @printf("    auto       %8.3f s  (%2d it)  σ²ᵤ=%7.4f σ²ₑ=%7.4f  cor=%.3f\n",
            ta, ra.iterations, ra.σ²[1], ra.σ²ₑ, cor(blup(ra), u))
end

function pedigree_bench(q)
    rng = StableRNG(1)
    sire = zeros(Int, q); dam = zeros(Int, q)
    for i in (q ÷ 20 + 1):q
        sire[i] = rand(rng, 1:i-1); dam[i] = rand(rng, 1:i-1)
    end
    t_ped = bench(() -> Pedigree(sire, dam), 3)
    ped = Pedigree(sire, dam)
    t_ai = bench(() -> ainverse(ped), 3)
    @printf("  q = %-8d Pedigree (Meuwissen&Luo) %8.3f ms   ainverse %8.3f ms\n",
            q, t_ped * 1e3, t_ai * 1e3)
end

qmax = parse(Int, get(ENV, "BENCH_Q", "5000"))

println("="^72)
println("Pedigree construction")
println("="^72)
for q in (1000, 5000, 20000, 50000)
    q <= max(qmax, 50000) && pedigree_bench(q)
end

println("\n" * "="^72)
println("Per-iteration profile")
println("="^72)
for q in (1000, 2500, 5000)
    q <= qmax && iteration_profile(q)
end

println("\n" * "="^72)
println("End-to-end fits")
println("="^72)
for q in (1000, 2500, 5000)
    q <= qmax && endtoend(q)
end
println()
