using QuantitativeGenetics
using Test
using LinearAlgebra
using SparseArrays
using StableRNGs
using Statistics
using ForwardDiff

# ---------------------------------------------------------------------------
# Reference implementations, deliberately written from first principles so they
# share no code with the package. Every fast path is checked against one of
# these rather than against a stored number.
# ---------------------------------------------------------------------------

"Dense tabular numerator relationship matrix A, by the recursive rule."
function dense_A(sire, dam)
    q = length(sire)
    A = zeros(q, q)
    for i in 1:q
        s, d = sire[i], dam[i]
        for j in 1:i-1
            asj = s == 0 ? 0.0 : A[s, j]
            adj = d == 0 ? 0.0 : A[d, j]
            A[i, j] = A[j, i] = 0.5 * (asj + adj)
        end
        A[i, i] = (s == 0 || d == 0) ? 1.0 : 1.0 + 0.5 * A[s, d]
    end
    return A
end

"Brute-force REML −2logL, straight from the definition with a dense n×n V."
function dense_m2ll(y, X, Zs, Ks, θ)
    n = length(y)
    V = θ[end] .* Matrix{Float64}(I, n, n)
    for (i, Z) in enumerate(Zs)
        V .+= θ[i] .* (Matrix(Z) * Ks[i] * Matrix(Z)')
    end
    Vi = inv(V)
    XtViX = X' * Vi * X
    P = Vi - Vi * X * (XtViX \ (X' * Vi))
    return logdet(V) + logdet(XtViX) + dot(y, P * y)
end

"Random topologically-sorted pedigree."
function random_pedigree(rng, q; nfounder=max(5, q ÷ 20))
    sire = zeros(Int, q); dam = zeros(Int, q)
    for i in (nfounder+1):q
        sire[i] = rand(rng, 1:i-1)
        dam[i] = rand(rng, 1:i-1)
    end
    return sire, dam
end

@testset "QuantitativeGenetics" begin

    @testset "pedigree: inbreeding and A⁻¹" begin
        # Pedigree with full-sib mating and a repeated ancestor, so several
        # animals are genuinely inbred.
        sire = [0, 0, 1, 1, 3, 3, 5, 1, 7, 7]
        dam  = [0, 0, 2, 2, 4, 4, 6, 6, 8, 6]
        ped = Pedigree(sire, dam)
        A = dense_A(sire, dam)

        @test ped.F ≈ diag(A) .- 1 atol = 1e-12
        @test any(>(0), ped.F)                       # the test is non-vacuous
        @test Matrix(ainverse(ped)) ≈ inv(A) atol = 1e-10
        # log|A| from the Mendelian variances, with no factorization.
        @test logdetA(ped) ≈ logdet(A) atol = 1e-10
        @test nnz(ainverse(ped)) < length(sire)^2    # genuinely sparse

        rng = StableRNG(20260719)
        s2, d2 = random_pedigree(rng, 120)
        ped2 = Pedigree(s2, d2)
        A2 = dense_A(s2, d2)
        @test ped2.F ≈ diag(A2) .- 1 atol = 1e-10
        @test Matrix(ainverse(ped2)) ≈ inv(A2) atol = 1e-8
        @test logdetA(ped2) ≈ logdet(A2) atol = 1e-8
    end

    @testset "pedigree: validation" begin
        @test_throws ArgumentError Pedigree([2, 0], [0, 0])   # parent after offspring
        @test_throws DimensionMismatch Pedigree([0, 0], [0])
        # Unrelated founders are never inbred.
        @test all(iszero, Pedigree(zeros(Int, 5), zeros(Int, 5)).F)
    end

    @testset "relatedness structures" begin
        rng = StableRNG(11)
        s, d = random_pedigree(rng, 60)
        K = PedigreeRelatedness(s, d)
        @test size(K, 1) == 60
        @test Kinv(K) isa SparseMatrixCSC
        @test logdetK(K) ≈ logdet(dense_A(s, d)) atol = 1e-8

        Kid = IdentityRelatedness(7)
        @test Matrix(Kinv(Kid)) ≈ Matrix(I, 7, 7)
        @test logdetK(Kid) == 0.0

        M = Float64.(rand(rng, 0:2, 40, 200))
        G = genomic_relationship(M)
        @test issymmetric(Symmetric(G))
        Kg = GenomicRelatedness(G; blend=0.05)
        Gb = 0.95 .* G .+ 0.05 .* Matrix(I, 40, 40)
        @test Kinv(Kg) ≈ inv(Gb) atol = 1e-8
        @test logdetK(Kg) ≈ logdet(Gb) atol = 1e-8
    end

    @testset "MME assembly and factor reuse" begin
        rng = StableRNG(5)
        q = 80
        s, d = random_pedigree(rng, q)
        K = PedigreeRelatedness(s, d)
        Ai = Matrix(Kinv(K))
        n = q
        X = [ones(n) randn(rng, n)]
        Z = sparse(1.0I, n, q); Zd = Matrix(Z)
        y = randn(rng, n)
        ws = build_mme(y, X, [RandomEffect(:animal, Z, K)])

        for λ in (2.5, 0.3, 17.0)
            update_mme!(ws, [λ])
            solve_mme!(ws)
            C = [X'X X'Zd; Zd'X Zd'Zd + Ai .* λ]
            r = [X'y; Zd'y]
            @test ws.sol ≈ C \ r atol = 1e-8
            @test logdet(ws.factor) ≈ logdet(C) atol = 1e-8
        end

        # The hot assembly path must not allocate: the whole design rests on
        # rewriting nzval in place rather than doing sparse arithmetic. The
        # ratio vector is hoisted out because a literal `[2.0]` would itself
        # allocate and be charged to the measurement.
        λbuf = [1.0]
        update_mme!(ws, λbuf)
        λbuf[1] = 2.0
        @test (@allocated update_mme!(ws, λbuf)) == 0

        # Blocks line up with the dense solution.
        update_mme!(ws, [1.0]); solve_mme!(ws)
        @test length(fixed_effects(ws)) == 2
        @test length(random_effect(ws, 1)) == q
    end

    @testset "MME ordering strategies agree" begin
        rng = StableRNG(6)
        q = 100
        s, d = random_pedigree(rng, q)
        K = PedigreeRelatedness(s, d)
        n = q; X = [ones(n) randn(rng, n)]; Z = sparse(1.0I, n, q)
        y = randn(rng, n)
        eff = [RandomEffect(:animal, Z, K)]
        sols = map((:auto, :fixedlast, :amd)) do ord
            ws = build_mme(y, X, eff; ordering=ord)
            update_mme!(ws, [1.7]); solve_mme!(ws)
            copy(ws.sol)
        end
        @test sols[1] ≈ sols[2] atol = 1e-8
        @test sols[1] ≈ sols[3] atol = 1e-8
    end

    @testset "Takahashi selected inverse" begin
        rng = StableRNG(7)
        for n in (6, 25, 90)
            B = sprandn(rng, n, n, 0.15)
            C = sparse(B * B') + n * I
            C = (C + C') / 2
            si = selected_inverse(C)
            Cinv = inv(Matrix(C))
            Ii, Jj, Vv = findnz(si.Z)
            @test !isempty(Vv)
            # Every computed entry must equal the true inverse there.
            @test maximum(abs(Vv[t] - Cinv[Ii[t], Jj[t]]) for t in eachindex(Vv)) < 1e-9
        end
    end

    @testset "trace_product matches dense trace" begin
        # The containment argument requires K's pattern to sit inside C's, which
        # is exactly how build_mme lays the MME out.
        rng = StableRNG(9)
        q = 50
        s, d = random_pedigree(rng, q)
        K = PedigreeRelatedness(s, d)
        n = q; X = [ones(n) randn(rng, n)]; Z = sparse(1.0I, n, q)
        y = randn(rng, n)
        ws = build_mme(y, X, [RandomEffect(:animal, Z, K)])
        update_mme!(ws, [1.3]); factorize_mme!(ws)

        si = selected_inverse(ws.C)
        Cinv = inv(Matrix(ws.C))
        p = ws.p
        ref = tr(Matrix(Kinv(K)) * Cinv[p+1:end, p+1:end])
        @test trace_product(si, Kinv(K), p) ≈ ref atol = 1e-8
    end

    @testset "REML objective matches dense brute force" begin
        rng = StableRNG(42)
        q = 45
        s, d = random_pedigree(rng, q)
        K = PedigreeRelatedness(s, d)
        A = dense_A(s, d)
        n = q; X = [ones(n) randn(rng, n)]; Z = sparse(1.0I, n, q)
        y = randn(rng, n) .+ 2
        ws = build_mme(y, X, [RandomEffect(:animal, Z, K)])
        rw = REMLWorkspace(ws, X)

        for θ in ([1.0, 1.0], [0.3, 2.7], [5.0, 0.8], [0.05, 3.3])
            @test reml_objective!(rw, y, θ) ≈ dense_m2ll(y, X, [Z], [A], θ) atol = 1e-8
        end
    end

    @testset "score and AI matrix vs ForwardDiff" begin
        # The AD oracle: hand-derived REML derivatives are exactly the kind of
        # thing that stays subtly wrong, so they are checked against automatic
        # differentiation of the dense reference objective.
        rng = StableRNG(43)
        q = 40
        s, d = random_pedigree(rng, q)
        K = PedigreeRelatedness(s, d)
        A = dense_A(s, d)
        n = q; X = [ones(n) randn(rng, n)]; Z = sparse(1.0I, n, q)
        y = randn(rng, n) .+ 1
        ws = build_mme(y, X, [RandomEffect(:animal, Z, K)])
        rw = REMLWorkspace(ws, X)
        f(θ) = dense_m2ll(y, X, [Z], [A], θ)

        for θ in ([1.0, 1.0], [0.3, 2.7], [2.0, 0.5])
            reml_objective!(rw, y, θ)
            score, AI = reml_gradient_ai!(rw, y, θ)
            @test score ≈ ForwardDiff.gradient(f, θ) rtol = 1e-6

            # AI is the average information: ½·yᵀP(∂V/∂θᵢ)P(∂V/∂θⱼ)Py.
            V = θ[1] .* (Matrix(Z) * A * Matrix(Z)') .+ θ[2] .* I(n)
            Vi = inv(V); XtViX = X' * Vi * X
            P = Vi - Vi * X * (XtViX \ (X' * Vi))
            dV = [Matrix(Z) * A * Matrix(Z)', Matrix{Float64}(I, n, n)]
            AIref = [0.5 * dot(y, P * dV[a] * P * dV[b] * P * y) for a in 1:2, b in 1:2]
            @test AI ≈ AIref rtol = 1e-6
            @test issymmetric(Symmetric(AI))
        end
    end

    @testset "AI-REML recovers known variance components" begin
        rng = StableRNG(2024)
        q = 1200
        s, d = random_pedigree(rng, q)
        K = PedigreeRelatedness(s, d)
        n = q
        X = [ones(n) randn(rng, n)]
        Z = sparse(1.0I, n, q)
        σu, σe = 3.0, 1.0
        L = cholesky(Symmetric(inv(Matrix(Kinv(K)))) + 1e-9I).L
        u = sqrt(σu) .* (L * randn(rng, q))
        β = [5.0, -1.5]
        y = X * β .+ u .+ sqrt(σe) .* randn(rng, n)
        eff = [RandomEffect(:animal, Z, K)]

        r = fit_reml(y, X, eff; trace_method=:exact)
        @test r.converged
        @test r.iterations < 25
        @test r.σ²[1] ≈ σu rtol = 0.25
        @test r.σ²ₑ ≈ σe rtol = 0.25
        @test r.β ≈ β atol = 0.4
        @test cor(blup(r), u) > 0.8

        h2, h2se = heritability(r)
        @test h2 ≈ σu / (σu + σe) rtol = 0.2
        @test isfinite(h2se) && h2se > 0
        @test all(isfinite, r.se)
    end

    @testset "trace methods agree" begin
        rng = StableRNG(77)
        q = 800
        s, d = random_pedigree(rng, q)
        K = PedigreeRelatedness(s, d)
        n = q; X = [ones(n) randn(rng, n)]; Z = sparse(1.0I, n, q)
        L = cholesky(Symmetric(inv(Matrix(Kinv(K)))) + 1e-9I).L
        u = sqrt(2.0) .* (L * randn(rng, q))
        y = X * [1.0, 0.5] .+ u .+ randn(rng, n)
        eff = [RandomEffect(:animal, Z, K)]

        re = fit_reml(y, X, eff; trace_method=:exact)
        rh = fit_reml(y, X, eff; trace_method=:hutchinson, nprobe=200,
                      rng=StableRNG(5))
        # Hutchinson is stochastic, so this is a loose but meaningful check.
        @test re.σ²[1] ≈ rh.σ²[1] rtol = 0.15
        @test re.σ²ₑ ≈ rh.σ²ₑ rtol = 0.15
    end

    @testset "multiple random effects" begin
        # Animal effect plus an independent group (permanent environment) effect.
        rng = StableRNG(31)
        q = 400
        s, d = random_pedigree(rng, q)
        K = PedigreeRelatedness(s, d)
        ngrp = 25
        grp = rand(rng, 1:ngrp, q)
        Zg = sparse(1:q, grp, ones(q), q, ngrp)
        Z = sparse(1.0I, q, q)
        n = q
        X = ones(n, 1)

        σu, σg, σe = 2.0, 1.0, 1.0
        L = cholesky(Symmetric(inv(Matrix(Kinv(K)))) + 1e-9I).L
        u = sqrt(σu) .* (L * randn(rng, q))
        g = sqrt(σg) .* randn(rng, ngrp)
        y = 3.0 .+ u .+ Zg * g .+ sqrt(σe) .* randn(rng, n)

        eff = [RandomEffect(:animal, Z, K),
               RandomEffect(:group, Zg, IdentityRelatedness(ngrp))]
        r = fit_reml(y, X, eff; trace_method=:exact)

        @test r.converged
        @test length(r.σ²) == 2
        @test r.names == [:animal, :group]
        @test all(>(0), r.σ²)
        @test r.σ²[1] ≈ σu rtol = 0.6
        @test r.σ²ₑ ≈ σe rtol = 0.6
        @test length(blup(r; effect=:group)) == ngrp
        @test_throws ArgumentError blup(r; effect=:nosucheffect)

        # Objective still matches the dense reference with two effects.
        ws = build_mme(y, X, eff); rw = REMLWorkspace(ws, X)
        A = dense_A(s, d)
        θ = [1.5, 0.8, 1.2]
        @test reml_objective!(rw, y, θ) ≈
              dense_m2ll(y, X, [Z, Zg], [A, Matrix(1.0I, ngrp, ngrp)], θ) atol = 1e-7
    end

    @testset "genomic relatedness end to end" begin
        rng = StableRNG(99)
        q = 150
        M = Float64.(rand(rng, 0:2, q, 400))
        G = genomic_relationship(M)
        K = GenomicRelatedness(G; blend=0.05)
        n = q
        X = ones(n, 1)
        Z = sparse(1.0I, n, q)
        Gb = Matrix(inv(Symmetric(Kinv(K))))
        L = cholesky(Symmetric(Gb) + 1e-8I).L
        u = sqrt(2.0) .* (L * randn(rng, q))
        y = 1.0 .+ u .+ randn(rng, n)
        eff = [RandomEffect(:genomic, Z, K)]

        ws = build_mme(y, X, eff); rw = REMLWorkspace(ws, X)
        θ = [1.4, 0.9]
        @test reml_objective!(rw, y, θ) ≈ dense_m2ll(y, X, [Z], [Gb], θ) atol = 1e-6

        r = fit_reml(y, X, eff)
        @test r.converged
        @test all(>(0), r.σ²)
        @test isfinite(r.loglik)
    end

    @testset "pedigree path agrees with rotated reparameterisation" begin
        # The animal model y = Xβ + Zu + e with u ~ N(0, σ²ᵤA) is equivalent,
        # under A = LLᵀ, to y = Xβ + (ZL)w + e with w ~ N(0, σ²ᵤI). The two run
        # through entirely different code — sparse Henderson A⁻¹ versus a dense
        # identity structure — so agreement validates the pedigree path against
        # something other than itself. (MixedModels.jl cannot express a pedigree
        # relationship matrix, so this stands in for an external reference here;
        # bench/compare.jl does check the i.i.d. case against MixedModels.)
        rng = StableRNG(3)
        q = 150
        s, d = random_pedigree(rng, q)
        K = PedigreeRelatedness(s, d)
        A = inv(Matrix(Kinv(K)))
        L = cholesky(Symmetric(A) + 1e-10I).L

        X = [ones(q) randn(rng, q)]
        Z = sparse(1.0I, q, q)
        u = sqrt(3.0) .* (L * randn(rng, q))
        y = X * [5.0, -1.5] .+ u .+ randn(rng, q)

        ra = fit_reml(y, X, [RandomEffect(:animal, Z, K)]; trace_method=:exact)
        ZL = sparse(Matrix(Z) * Matrix(L))
        rb = fit_reml(y, X, [RandomEffect(:w, ZL, IdentityRelatedness(q))];
                      trace_method=:exact)

        @test ra.σ²[1] ≈ rb.σ²[1] rtol = 1e-6
        @test ra.σ²ₑ ≈ rb.σ²ₑ rtol = 1e-6
        @test ra.loglik ≈ rb.loglik atol = 1e-6
        @test ra.β ≈ rb.β rtol = 1e-6
    end

    @testset "loglik includes the REML normalizing constant" begin
        # `reml_objective!` omits (n−p)·log(2π) because it is free of θ; fit_reml
        # adds it back so `loglik` is on the same scale as other REML software.
        # Verified against MixedModels.jl in bench/compare.jl, which agreed to
        # ~1e-12 once this term was included.
        rng = StableRNG(21)
        q = 60
        s, d = random_pedigree(rng, q)
        K = PedigreeRelatedness(s, d)
        X = ones(q, 1); Z = sparse(1.0I, q, q)
        y = randn(rng, q)
        eff = [RandomEffect(:animal, Z, K)]

        r = fit_reml(y, X, eff)
        ws = build_mme(y, X, eff); rw = REMLWorkspace(ws, X)
        raw = reml_objective!(rw, y, vcat(r.σ², r.σ²ₑ))
        @test -2 * r.loglik ≈ raw + (q - 1) * log(2π) atol = 1e-6
    end

    @testset "results helpers" begin
        rng = StableRNG(3)
        q = 200
        s, d = random_pedigree(rng, q)
        K = PedigreeRelatedness(s, d)
        X = ones(q, 1); Z = sparse(1.0I, q, q)
        L = cholesky(Symmetric(inv(Matrix(Kinv(K)))) + 1e-9I).L
        u = sqrt(2.0) .* (L * randn(rng, q))
        y = u .+ randn(rng, q)
        r = fit_reml(y, X, [RandomEffect(:animal, Z, K)])

        h2, se = heritability(r)
        @test 0 < h2 < 1
        @test heritability(r; effect=:animal)[1] ≈ h2
        @test variance_ratio(r) ≈ r.σ²[1] / r.σ²ₑ
        @test length(blup(r)) == q
        @test reliability([0.5, 0.1], 2.0) ≈ [0.75, 0.95]
        @test accuracy([0.5], 2.0) ≈ [sqrt(0.75)]
        # show() must not error
        @test occursin("REMLResult", sprint(show, r))
    end
end
