# External cross-validation against R (nadiv + sommer).
#
# Run:  julia --project=bench bench/external.jl
#       QG_XVAL_Q=2000 julia --project=bench bench/external.jl
#
# Requires R on PATH with `nadiv` and `sommer` installed (see bench/R/README.md).
# Generates one dataset, hands it to R, fits the same data here, and compares.

using QuantitativeGenetics
using LinearAlgebra, SparseArrays, StableRNGs, DelimitedFiles, Printf

const RSCRIPT = Sys.iswindows() ? "Rscript.exe" : "Rscript"

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
            @printf(io, "%d,%.10f,%.10f\n", i, y[i], X[i, 2])
        end
    end
    Ai = Kinv(K); Ii, Jj, Vv = findnz(Ai); keep = Ii .>= Jj
    writedlm(joinpath(dir, "ainv_qg.csv"), hcat(Ii[keep], Jj[keep], Vv[keep]), ',')
    writedlm(joinpath(dir, "F_qg.csv"), inbreeding(K), ',')
    return y, X, [RandomEffect(:animal, Z, K)], K
end

parse_r(out) = Dict(begin
    p = split(strip(l))
    p[1] => length(p) == 2 ? parse(Float64, p[2]) : parse.(Float64, p[2:end])
end for l in split(out, '\n') if startswith(strip(l), "R_"))

function run(q)
    dir = mktempdir()
    y, X, eff, K = make_dataset(dir, q)

    rfile = joinpath(@__DIR__, "R", "reference.R")
    out = try
        read(`$RSCRIPT $rfile $dir`, String)
    catch e
        println("  R reference failed: ", e); return
    end
    R = parse_r(out)

    fit_reml(y, X, eff)                      # warm up
    t_ex = @elapsed re = fit_reml(y, X, eff; trace_method=:exact)
    t_au = @elapsed ra = fit_reml(y, X, eff)
    t_ai = @elapsed PedigreeRelatedness(K.ped)

    @printf("\n  q = %d\n", q)
    println("  " * "-"^68)
    @printf("  A⁻¹ vs nadiv          max abs diff  %.2e\n", R["R_AINV_MAXDIFF"])
    @printf("  inbreeding vs nadiv   max abs diff  %.2e   (mean F = %.4f)\n",
            R["R_F_MAXDIFF"], R["R_F_MEAN"])
    @printf("  A⁻¹ build time        nadiv %7.3f s   QG %7.4f s   (%.0f×)\n",
            R["R_AINV_TIME"], t_ai, R["R_AINV_TIME"] / t_ai)
    println()
    @printf("  σ²ₐ    sommer %12.8f   QG %12.8f   rel %.2e\n",
            R["R_SOMMER_SIGMA_A"], re.σ²[1],
            abs(R["R_SOMMER_SIGMA_A"] - re.σ²[1]) / R["R_SOMMER_SIGMA_A"])
    @printf("  σ²ₑ    sommer %12.8f   QG %12.8f   rel %.2e\n",
            R["R_SOMMER_SIGMA_E"], re.σ²ₑ,
            abs(R["R_SOMMER_SIGMA_E"] - re.σ²ₑ) / R["R_SOMMER_SIGMA_E"])
    @printf("  β      sommer %s   QG %s\n",
            string(round.(R["R_SOMMER_BETA"], digits=6)),
            string(round.(re.β, digits=6)))
    @printf("  logLik brute-force R %12.6f   QG %12.6f   abs %.2e\n",
            R["R_BRUTE_LOGLIK"], re.loglik,
            abs(R["R_BRUTE_LOGLIK"] - re.loglik))
    println()
    @printf("  fit time  sommer %8.3f s   QG(exact) %7.3f s (%5.0f×)   QG(auto) %7.3f s (%5.0f×)\n",
            R["R_SOMMER_TIME"], t_ex, R["R_SOMMER_TIME"] / t_ex,
            t_au, R["R_SOMMER_TIME"] / t_au)
end

println("="^72)
println("External cross-validation: nadiv (A⁻¹, F) and sommer (animal model)")
println("="^72)
for q in (400, 800, 2000)
    q <= parse(Int, get(ENV, "QG_XVAL_Q", "2000")) && run(q)
end
println()
