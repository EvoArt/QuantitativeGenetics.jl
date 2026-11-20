# Correctness cross-validation against R (nadiv + sommer).
#
# Run standalone:  julia --project=test/external test/external/runtests_external.jl
# In CI:           see .github/workflows/CI.yml (job `external`)
#
# This is SEPARATE from the main test suite because it needs R with `nadiv` and
# `sommer` installed. It skips cleanly (exit 0, with a notice) when R or the
# packages are missing, so it never breaks a plain `Pkg.test()`.
#
# What it asserts, on a pedigree animal model both sides can express:
#
#   * A⁻¹ matches nadiv::makeAinv         to 1e-10
#   * inbreeding coefficients match nadiv to 1e-12
#   * σ²ₐ and σ²ₑ match sommer::mmer      to 1e-3 relative
#   * fixed effects match sommer          to 1e-4
#   * logLik matches a brute-force REML computed in R to 1e-5
#
# The variance-component tolerance is looser than the rest on purpose: sommer
# and this package use different convergence criteria and stopping rules, so
# agreement to ~5e-6 (what we actually observe) is the realistic target, not
# bit-equality. The logLik check is the tight one, because it is compared
# against an explicit dense-V calculation rather than against sommer's own
# internal convergence metric, which is on a different scale entirely.

using Test
using QuantitativeGenetics
using LinearAlgebra, SparseArrays, StableRNGs, DelimitedFiles, Printf

const RSCRIPT = Sys.iswindows() ? "Rscript.exe" : "Rscript"

function r_available()
    try
        run(pipeline(`$RSCRIPT -e "quit(status=0)"`, devnull, devnull))
        out = read(`$RSCRIPT -e 'cat(all(sapply(c("nadiv","sommer"), requireNamespace, quietly=TRUE)))'`, String)
        return occursin("TRUE", out)
    catch
        return false
    end
end

function make_dataset(dir, q; seed=4242, σu=3.0, σe=1.0)
    rng = StableRNG(seed)
    sire = zeros(Int, q); dam = zeros(Int, q)
    for i in (q ÷ 20 + 1):q
        sire[i] = rand(rng, 1:i-1); dam[i] = rand(rng, 1:i-1)
    end
    K = PedigreeRelatedness(sire, dam)
    X = [ones(q) randn(rng, q)]
    Z = sparse(1.0I, q, q)
    L = cholesky(Symmetric(inv(Matrix(Kinv(K)))) + 1e-9I).L
    u = sqrt(σu) .* (L * randn(rng, q))
    y = X * [5.0, -1.5] .+ u .+ sqrt(σe) .* randn(rng, q)

    open(joinpath(dir, "ped.csv"), "w") do io
        println(io, "id,sire,dam")
        for i in 1:q
            println(io, "$i,", sire[i] == 0 ? "NA" : sire[i], ",",
                    dam[i] == 0 ? "NA" : dam[i])
        end
    end
    open(joinpath(dir, "data.csv"), "w") do io
        println(io, "id,y,x")
        for i in 1:q
            @printf(io, "%d,%.12f,%.12f\n", i, y[i], X[i, 2])
        end
    end
    Ai = Kinv(K); Ii, Jj, Vv = findnz(Ai); keep = Ii .>= Jj
    writedlm(joinpath(dir, "ainv_qg.csv"), hcat(Ii[keep], Jj[keep], Vv[keep]), ',')
    writedlm(joinpath(dir, "F_qg.csv"), inbreeding(K), ',')
    return y, X, [RandomEffect(:animal, Z, K)]
end

parse_r(out) = Dict(begin
    p = split(strip(l))
    p[1] => length(p) == 2 ? parse(Float64, p[2]) : parse.(Float64, p[2:end])
end for l in split(out, '\n') if startswith(strip(l), "R_"))

if !r_available()
    @info "R with nadiv+sommer not available — skipping external cross-validation"
    exit(0)
end

@testset "external cross-validation vs R" begin
    dir = mktempdir()
    q = 600
    y, X, eff = make_dataset(dir, q)

    rfile = joinpath(@__DIR__, "..", "..", "bench", "R", "reference.R")
    out = read(`$RSCRIPT $rfile $dir`, String)
    R = parse_r(out)

    @testset "A⁻¹ and inbreeding vs nadiv" begin
        @test haskey(R, "R_AINV_MAXDIFF")
        @test R["R_AINV_MAXDIFF"] < 1e-10
        @test R["R_F_MAXDIFF"] < 1e-12
        # Non-vacuous: the pedigree must actually contain inbred animals.
        @test R["R_F_MEAN"] > 0
    end

    @testset "variance components vs sommer" begin
        r = fit_reml(y, X, eff; trace_method=:exact)
        @test r.converged
        @test r.σ²[1] ≈ R["R_SOMMER_SIGMA_A"] rtol = 1e-3
        @test r.σ²ₑ ≈ R["R_SOMMER_SIGMA_E"] rtol = 1e-3
        @test r.β ≈ R["R_SOMMER_BETA"] atol = 1e-4
        # The arbiter: brute-force dense-V REML computed independently in R.
        @test r.loglik ≈ R["R_BRUTE_LOGLIK"] atol = 1e-5
    end
end
