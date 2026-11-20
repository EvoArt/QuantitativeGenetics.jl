using LinearAlgebra
using SparseArrays
using SparseArrays: CHOLMOD

# ---------------------------------------------------------------------------
# The mixed model equations, and the machinery that makes re-solving them cheap.
#
# For y = Xβ + Σᵢ Zᵢuᵢ + e with uᵢ ~ N(0, σ²ᵢKᵢ) and e ~ N(0, σ²ₑI), Henderson's
# equations are Cθ·ŝ = r with
#
#     C = [ X'X   X'Z₁         X'Z₂        ⋯ ]        r = [ X'y  ]
#         [ Z₁'X  Z₁'Z₁+K₁⁻¹λ₁ Z₁'Z₂       ⋯ ]            [ Z₁'y ]
#         [ Z₂'X  Z₂'Z₁        Z₂'Z₂+K₂⁻¹λ₂ ⋯ ]            [ Z₂'y ]
#
# and λᵢ = σ²ₑ/σ²ᵢ. Across an entire REML fit the ONLY thing that changes is the
# handful of scalars λᵢ. Everything else — X'X, X'Zᵢ, Zᵢ'Zⱼ, Zᵢ'y, and the
# sparsity pattern of C — is constant.
#
# Three consequences, and they are the whole point of this file:
#
#   1. ASSEMBLY IS A SCATTER-ADD, NOT SPARSE ARITHMETIC. Build C's pattern once,
#      cache the θ-independent values in `base_nzval`, and cache for each random
#      effect the linear indices into `C.nzval` where its K⁻¹ entries land. Then
#      each iteration is a copyto! plus one flat loop of `nzval[idx] += λ*v`.
#      Doing this the obvious way instead — `Zt*Z ./ σ²ₑ .+ Kinv ./ σ²ᵤ` — runs a
#      full sparse-sparse add and allocates a new matrix every iteration, which
#      at q = 25000 costs more than the numeric factorization it feeds.
#
#   2. FIXED EFFECTS ARE ORDERED LAST. X's columns are dense, so a fill-reducing
#      ordering that eliminates them early produces catastrophic fill: every
#      pivot in a dense row connects all its neighbours. We compute the ordering
#      on the random-effect block alone and append the fixed effects at the end,
#      then hand the result to CHOLMOD as a user permutation. `ordering=:auto`
#      benchmarks both this and CHOLMOD's own AMD on the full matrix at setup
#      and keeps whichever gives the sparser factor — measured, not assumed.
#
#   3. THE SYMBOLIC FACTORIZATION IS COMPUTED ONCE PER FIT. `cholesky` runs the
#      fill-reducing analysis and the first numeric factorization; every later
#      iteration calls `cholesky!(F, C)`, which reuses the ordering and symbolic
#      structure and redoes only O(nnz(L)) numeric work. This amortizes over
#      every REML iteration AND every step-halving retry, not just one solve.
# ---------------------------------------------------------------------------

"""
    RandomEffect(name, Z, K)

One random effect in the animal model: an `n×qᵢ` incidence matrix `Z` mapping
records to levels, and a relatedness structure `K` of order `qᵢ`.

`Z` is stored sparse. For the very common one-record-per-animal case pass
`sparse(1.0I, n, q)`, or use [`animal_effect`](@ref).
"""
struct RandomEffect{R<:AbstractRelatedness}
    name::Symbol
    Z::SparseMatrixCSC{Float64,Int}
    K::R
    q::Int
end

function RandomEffect(name::Symbol, Z::AbstractMatrix, K::AbstractRelatedness)
    Zs = SparseMatrixCSC{Float64,Int}(sparse(Z))
    q = size(Zs, 2)
    size(K, 1) == q || throw(DimensionMismatch(
        "effect :$name has Z with $q columns but K is $(size(K,1))×$(size(K,1))"))
    return RandomEffect(name, Zs, K, q)
end

"""
    animal_effect(name, K; n=size(K,1))

Convenience constructor for a random effect with `Z = I`, i.e. one record per
level in the same order — the standard animal-model layout.
"""
animal_effect(name::Symbol, K::AbstractRelatedness) =
    RandomEffect(name, sparse(1.0I, size(K, 1), size(K, 1)), K)

"""
    MMEWorkspace

Everything needed to re-solve the mixed model equations at a new set of variance
ratios without re-deriving anything structural.

Holds the coefficient matrix `C` (whose `nzval` is overwritten in place each
iteration), the θ-independent values `base_nzval`, per-effect scatter indices
`kidx`/`kval` into `C.nzval`, the right-hand side, and the cached CHOLMOD
factor. See the header of `mme.jl` for why each of these exists.
"""
mutable struct MMEWorkspace{RE}
    const effects::Vector{RE}
    const n::Int                    # number of records
    const p::Int                    # number of fixed-effect columns
    const q::Int                    # total random-effect levels
    const offsets::Vector{Int}      # start index of each effect's block in the solution
    const C::SparseMatrixCSC{Float64,Int}
    const base_nzval::Vector{Float64}
    const kidx::Vector{Vector{Int}}     # linear indices into C.nzval, per effect
    const kval::Vector{Vector{Float64}} # matching K⁻¹ values, per effect
    const rhs::Vector{Float64}
    const perm::Vector{Int}
    factor::Union{Nothing,CHOLMOD.Factor{Float64,Int}}
    sol::Vector{Float64}
    yty::Float64
end

"""
    build_mme(y, X, effects; ordering=:auto) -> MMEWorkspace

Construct the mixed model equations for `y = Xβ + Σ Zᵢuᵢ + e` and precompute
every structure that does not depend on the variance components.

`ordering` selects the fill-reducing permutation strategy:

- `:auto` (default) — build both the fixed-effects-last ordering and CHOLMOD's
  default AMD, factor symbolically with each, and keep whichever yields fewer
  nonzeros in `L`. Costs two symbolic analyses once, at setup.
- `:fixedlast` — force AMD on the random-effect block with fixed effects appended
  at the end.
- `:amd` — force CHOLMOD's default ordering on the whole matrix.

This is the only place the design of `C` is decided; everything downstream just
updates values.
"""
function build_mme(y::AbstractVector, X::AbstractMatrix,
                   effects::Vector{<:RandomEffect}; ordering::Symbol=:auto)
    n = length(y)
    size(X, 1) == n || throw(DimensionMismatch("X has $(size(X,1)) rows but y has $n"))
    for e in effects
        size(e.Z, 1) == n || throw(DimensionMismatch(
            "effect :$(e.name) has Z with $(size(e.Z,1)) rows but y has $n"))
    end
    p = size(X, 2)
    qs = [e.q for e in effects]
    q = sum(qs; init=0)
    m = p + q

    # Block offsets in the solution vector: fixed effects first, then each
    # random effect in the order given. (This is the *storage* order; the
    # elimination order is the permutation chosen below and is independent.)
    offsets = Vector{Int}(undef, length(effects))
    off = p
    for (i, e) in enumerate(effects)
        offsets[i] = off
        off += e.q
    end

    Xs = SparseMatrixCSC{Float64,Int}(sparse(X))
    W = length(effects) == 0 ? Xs : hcat(Xs, [e.Z for e in effects]...)

    # The θ-independent part of C is just W'W, where W = [X Z₁ Z₂ ⋯].
    WtW = SparseMatrixCSC{Float64,Int}(transpose(W) * W)
    rhs = Vector{Float64}(transpose(W) * y)

    # C's pattern is the union of W'W's pattern and each K⁻¹'s pattern placed at
    # its diagonal block. Union it in structurally (values are irrelevant here)
    # so that the pattern is fixed for the whole fit even where W'W happens to
    # have a structural zero that K⁻¹ fills.
    pat = WtW
    for (i, e) in enumerate(effects)
        Ki = Kinv(e.K)
        Ksp = Ki isa SparseMatrixCSC ? Ki : sparse(Ki)
        o = offsets[i]
        # Embed K⁻¹ at rows/cols o+1:o+q in an m×m matrix.
        Ii, Jj, _ = findnz(Ksp)
        emb = sparse(Ii .+ o, Jj .+ o, ones(length(Ii)), m, m)
        pat = pat + emb
    end
    C = SparseMatrixCSC{Float64,Int}(pat)
    fill!(C.nzval, 0.0)

    # Write W'W's values into the fixed pattern. `_scatter_into!` maps by
    # (row, col) lookup within each column, which is a binary search over sorted
    # row indices — done once here, never in the iteration.
    _add_block!(C, WtW, 0, 0, 1.0)
    base_nzval = copy(C.nzval)

    # Cache, for each effect, where its K⁻¹ entries land in C.nzval.
    kidx = Vector{Vector{Int}}(undef, length(effects))
    kval = Vector{Vector{Float64}}(undef, length(effects))
    for (i, e) in enumerate(effects)
        Ki = Kinv(e.K)
        Ksp = Ki isa SparseMatrixCSC ? Ki : sparse(Ki)
        kidx[i], kval[i] = _index_block(C, Ksp, offsets[i], offsets[i])
    end

    perm = _choose_ordering(C, p, q, ordering)

    ws = MMEWorkspace{eltype(effects)}(
        collect(effects), n, p, q, offsets, C, base_nzval, kidx, kval, rhs,
        perm, nothing, Vector{Float64}(undef, m), dot(y, y))
    return ws
end

# Add `B` (sparse) into `C` at row/col offset (ro, co), scaled by `s`.
# Used only at setup.
function _add_block!(C::SparseMatrixCSC, B::SparseMatrixCSC, ro::Int, co::Int, s::Float64)
    rows = rowvals(C)
    vals = nonzeros(C)
    Brows = rowvals(B)
    Bvals = nonzeros(B)
    for j in 1:size(B, 2)
        cj = j + co
        lo, hi = C.colptr[cj], C.colptr[cj+1] - 1
        for k in nzrange(B, j)
            r = Brows[k] + ro
            idx = _searchrow(rows, lo, hi, r)
            idx == 0 && error("internal: (row $r, col $cj) missing from C's pattern")
            vals[idx] += s * Bvals[k]
        end
    end
    return C
end

# Linear indices into C.nzval for each stored entry of B placed at (ro, co),
# together with B's values in the same order.
function _index_block(C::SparseMatrixCSC, B::SparseMatrixCSC, ro::Int, co::Int)
    rows = rowvals(C)
    Brows = rowvals(B)
    Bvals = nonzeros(B)
    nb = nnz(B)
    idx = Vector{Int}(undef, nb)
    val = Vector{Float64}(undef, nb)
    t = 0
    for j in 1:size(B, 2)
        cj = j + co
        lo, hi = C.colptr[cj], C.colptr[cj+1] - 1
        for k in nzrange(B, j)
            r = Brows[k] + ro
            i = _searchrow(rows, lo, hi, r)
            i == 0 && error("internal: (row $r, col $cj) missing from C's pattern")
            t += 1
            idx[t] = i
            val[t] = Bvals[k]
        end
    end
    return idx, val
end

# Binary search for row `r` in the sorted slice rows[lo:hi]; 0 if absent.
@inline function _searchrow(rows::Vector{Int}, lo::Int, hi::Int, r::Int)
    @inbounds while lo <= hi
        mid = (lo + hi) >>> 1
        rv = rows[mid]
        if rv == r
            return mid
        elseif rv < r
            lo = mid + 1
        else
            hi = mid - 1
        end
    end
    return 0
end

# Choose the elimination ordering. See point 2 in the file header.
#
# NOTE ON MEASUREMENT. `C`'s VALUES are all zero when this is called (only its
# pattern is final), so any fill comparison must be made on a matrix built from
# the pattern with a safely positive-definite set of values — factorizing the
# zero matrix orders a meaningless problem and, worse, can succeed while
# reporting fill for the wrong structure. `_pattern_spd` supplies that.
function _choose_ordering(C::SparseMatrixCSC, p::Int, q::Int, ordering::Symbol)
    m = p + q
    ordering in (:auto, :fixedlast, :amd) ||
        throw(ArgumentError("ordering must be :auto, :fixedlast or :amd, got :$ordering"))

    if ordering === :amd || p == 0 || q == 0
        return collect(1:m)   # sentinel: let CHOLMOD pick (see factorize_mme!)
    end

    Cp = _pattern_spd(C)

    # AMD on the random-effect block alone, with the (dense-column) fixed
    # effects appended at the END so they are eliminated last.
    prr = _amd(_pattern_spd(C[p+1:m, p+1:m]))
    fixedlast = vcat(prr .+ p, collect(1:p))

    ordering === :fixedlast && return fixedlast

    # :auto — compare actual factor fill and keep the sparser ordering.
    #
    # Measured on a 1500-animal pedigree: fixed-effects-last gives nnz(L) =
    # 838k versus 99k for CHOLMOD's own AMD on the whole matrix — 8.5× WORSE.
    # Constraining the ordering denies AMD the freedom to interleave the two
    # dense fixed-effect columns among the pedigree pivots, and that costs far
    # more than eliminating them early ever did. So the "fixed effects last"
    # heuristic is kept only as an explicit option; :auto measures and, on
    # pedigree problems, reliably picks plain AMD.
    nz_fixedlast = try
        nnz(cholesky(Symmetric(Cp); perm=fixedlast))
    catch
        typemax(Int)
    end
    nz_default = try
        nnz(cholesky(Symmetric(Cp)))
    catch
        typemax(Int)
    end
    return nz_fixedlast < nz_default ? fixedlast : collect(1:m)
end

# A diagonally dominant matrix with the same sparsity pattern as `A`, so that
# symbolic experiments factorize successfully regardless of `A`'s values.
function _pattern_spd(A::SparseMatrixCSC)
    n = size(A, 1)
    n == 0 && return A
    B = SparseMatrixCSC(n, n, copy(A.colptr), copy(rowvals(A)),
                        fill(0.01, nnz(A)))
    return B + sparse((n + 1.0)I, n, n)
end

# AMD ordering read back from CHOLMOD's own analysis, which avoids a dependency
# on AMD.jl for a permutation CHOLMOD already computes internally.
function _amd(A::SparseMatrixCSC)
    size(A, 1) == 0 && return Int[]
    return collect(cholesky(Symmetric(A)).p)
end

"""
    update_mme!(ws::MMEWorkspace, λ) -> ws

Rewrite `C`'s values for the variance ratios `λ[i] = σ²ₑ/σ²ᵢ`.

One `copyto!` of the cached θ-independent values, then one flat scatter-add per
random effect. No allocation, no sparse arithmetic, and the sparsity pattern is
untouched so the cached symbolic factor stays valid.
"""
function update_mme!(ws::MMEWorkspace, λ::AbstractVector{Float64})
    length(λ) == length(ws.effects) ||
        throw(DimensionMismatch("expected $(length(ws.effects)) ratios, got $(length(λ))"))
    nzv = nonzeros(ws.C)
    copyto!(nzv, ws.base_nzval)
    @inbounds for i in eachindex(ws.effects)
        λi = λ[i]
        idx = ws.kidx[i]
        val = ws.kval[i]
        @simd for t in eachindex(idx)
            nzv[idx[t]] += λi * val[t]
        end
    end
    return ws
end

"""
    factorize_mme!(ws::MMEWorkspace) -> CHOLMOD.Factor

Factorize the current `C`. The first call performs the fill-reducing analysis
and caches the symbolic factor; every later call is an in-place `cholesky!` that
reuses it and redoes only the numeric work.
"""
function factorize_mme!(ws::MMEWorkspace)
    S = Symmetric(ws.C)
    if ws.factor === nothing
        m = ws.p + ws.q
        F = if ws.perm == collect(1:m)
            cholesky(S)                 # sentinel: CHOLMOD's own ordering
        else
            cholesky(S; perm=ws.perm)
        end
        ws.factor = F
        return F
    else
        F = ws.factor
        cholesky!(F, S)
        return F
    end
end

"""
    solve_mme!(ws::MMEWorkspace) -> ws

Factorize `C` and solve `Cŝ = r`, leaving the solution in `ws.sol`
(fixed effects first, then each random effect's block).
"""
function solve_mme!(ws::MMEWorkspace)
    F = factorize_mme!(ws)
    ws.sol = F \ ws.rhs
    return ws
end

"""
    fixed_effects(ws)  -> view of β̂
    random_effect(ws, i) -> view of ûᵢ

Views into the current MME solution.
"""
fixed_effects(ws::MMEWorkspace) = view(ws.sol, 1:ws.p)
random_effect(ws::MMEWorkspace, i::Integer) =
    view(ws.sol, ws.offsets[i]+1 : ws.offsets[i]+ws.effects[i].q)
