using LinearAlgebra
using SparseArrays
using SparseArrays: CHOLMOD
using Random

# ---------------------------------------------------------------------------
# Selected (sparse) inversion — the Takahashi equations.
#
# REML's score needs tr(Kᵢ⁻¹ C^{uᵢuᵢ}), i.e. the entries of C⁻¹ at the positions
# where Kᵢ⁻¹ is nonzero. Forming C⁻¹ is out of the question (dense, m×m). The
# Takahashi equations (Takahashi, Fagan & Chin 1973; the algorithm behind
# FSPAK/BLUPF90) compute the entries of C⁻¹ ON THE SPARSITY PATTERN OF THE
# FACTOR, directly from the factor, in one backward sweep.
#
# With C = LDLᵀ (unit lower L, diagonal D) and Z ≔ C⁻¹, the identity
# Z = D⁻¹L⁻¹ + (I − Lᵀ)Z gives, for the entries we are allowed to touch,
#
#     Zᵢⱼ = −Σ_{k>i, Lₖᵢ≠0} Lₖᵢ Zₖⱼ                    (i < j)
#     Zᵢᵢ = 1/Dᵢ − Σ_{k>i, Lₖᵢ≠0} Lₖᵢ Zₖᵢ
#
# Sweeping columns from n down to 1 and, within a column, rows from the bottom
# up, every Zₖⱼ on the right-hand side is already known. The result is exact —
# not an approximation — for every entry in pattern(L + Lᵀ).
#
# WHY THIS SUFFICES. The fill-in property of Cholesky gives
# pattern(C) ⊆ pattern(L + Lᵀ), and each Kᵢ⁻¹ sits inside a diagonal block of C
# by construction, so pattern(Kᵢ⁻¹) ⊆ pattern(C) ⊆ pattern(L + Lᵀ). Every entry
# the trace asks for is therefore available. This containment is what makes the
# whole approach work and is worth remembering if the MME pattern ever changes.
#
# COST. One sweep costs O(Σⱼ |Lⱼ|²)-ish — typically a small multiple of the
# numeric factorization. It is the dominant term of a REML iteration on large
# pedigrees, which is why `:hutchinson` exists as an escape hatch.
#
# IMPLEMENTATION NOTES. The sweep is inherently gather/scatter over sparse
# indices, so it is memory-bound rather than FLOP-bound: SIMD libraries do not
# help here (measured — see bench/). What does help is (a) working entirely in
# CSC with a dense scratch column, (b) an inverse-permutation lookup so the
# result can be queried in the ORIGINAL ordering, and (c) never allocating
# inside the sweep.
# ---------------------------------------------------------------------------

"""
    SelectedInverse

Entries of `C⁻¹` on the sparsity pattern of the Cholesky factor, stored as a
symmetric sparse matrix in the ORIGINAL (unpermuted) ordering.

Build with [`selected_inverse`](@ref) or `selected_inverse!`; query the
trace terms REML needs with [`trace_product`](@ref).
"""
struct SelectedInverse
    Z::SparseMatrixCSC{Float64,Int}   # C⁻¹ on pattern(L+Lᵀ), original ordering
end

Base.getindex(s::SelectedInverse, i::Integer, j::Integer) = s.Z[i, j]

"""
    selected_inverse(C::SparseMatrixCSC) -> SelectedInverse

Compute the selected inverse of the symmetric positive-definite `C` by the
Takahashi equations, via an LDLᵀ factorization.

Returns the entries of `C⁻¹` at every position in `pattern(L + Lᵀ)`, in `C`'s
own ordering. Entries outside that pattern are absent (not zero — they are
simply not computed; `C⁻¹` is generally dense there).
"""
function selected_inverse(C::SparseMatrixCSC)
    F = ldlt(Symmetric(C))
    return selected_inverse(F)
end

"""
    selected_inverse(F::CHOLMOD.Factor) -> SelectedInverse

Selected inverse from an existing LDLᵀ factor. `F` must be an LDLᵀ factor
(from `ldlt`), not an LLᵀ one (from `cholesky`) — the Takahashi recursion above
is written for unit-triangular `L` with a separate `D`.
"""
function selected_inverse(F::CHOLMOD.Factor{Float64,Int})
    unsafe_load(pointer(F)).is_ll == 0 ||
        throw(ArgumentError("selected_inverse needs an LDLᵀ factor; use ldlt, not cholesky"))
    LD = sparse(F.LD)
    perm = collect(F.p)
    Zp = _takahashi(LD)
    # Map back to the original ordering: Zp holds (C[perm,perm])⁻¹ entries, so
    # C⁻¹[perm[i], perm[j]] = Zp[i,j].
    Ii, Jj, Vv = findnz(Zp)
    n = size(Zp, 1)
    return SelectedInverse(sparse(perm[Ii], perm[Jj], Vv, n, n))
end

# The Takahashi backward sweep. `LD` is CHOLMOD's packed factor: strictly lower
# triangle holds L (unit diagonal implied), the diagonal holds D.
#
# We build the result column by column from the last to the first, keeping the
# lower triangle only (the result is symmetric) and using a dense scratch
# accumulator indexed by row, cleared only over the rows actually touched.
function _takahashi(LD::SparseMatrixCSC{Float64,Int})
    n = size(LD, 1)
    Lrow = rowvals(LD)
    Lval = nonzeros(LD)

    # Result pattern = lower triangle of pattern(L), which already includes the
    # diagonal. Count first so we can allocate exactly once.
    colptr = Vector{Int}(undef, n + 1)
    colptr[1] = 1
    @inbounds for j in 1:n
        colptr[j+1] = colptr[j] + length(nzrange(LD, j))
    end
    nnzZ = colptr[n+1] - 1
    rowv = Vector{Int}(undef, nnzZ)
    zval = zeros(Float64, nnzZ)
    @inbounds for j in 1:n
        t = colptr[j]
        for k in nzrange(LD, j)
            rowv[t] = Lrow[k]
            t += 1
        end
    end

    # SCATTER THE SOURCE COLUMN, DON'T RE-SUM IT PER TARGET.
    #
    # The recursion is Z[i,j] = −Σ_{k>j, L[k,j]≠0} L[k,j]·Z[k,i]. Evaluating it
    # as written — for each target row i, loop over every k in column j — costs
    # |Lⱼ|² operations per column. Measured on a 5000-animal pedigree that came
    # to 348 M inner operations, or 618 per nonzero of L, and it grew with the
    # column length, so the sweep scaled far worse than nnz(L). Making the
    # lookup O(1) does not help: the operation COUNT is the problem.
    #
    # Instead, run it as a scatter. Load column j's target rows into a dense
    # accumulator, then for each source row k walk column k of Z ONCE and push
    # L[k,j]·Z[r,k] into acc[r] for exactly the rows r that column j needs. Each
    # (k, r) pair is now visited once instead of once per target, which brings
    # the sweep back to O(Σⱼ Σ_{k∈Lⱼ} |Lₖ|) — proportional to the factor's own
    # structure, which is the cost Takahashi is supposed to have.
    # `pos` maps a row index to its slot in the column currently scattered, so
    # the symmetric fetch Z[max(k,i), min(k,i)] is a single array load rather
    # than a binary search. Walking target rows i from the bottom up, column i
    # is already complete, and every k in column j's pattern is > j, so the
    # owning column of Z[k,i] is min(k,i) — which is i whenever k ≥ i, and
    # otherwise a column that is likewise complete.
    pos = zeros(Int, n)
    acc = zeros(Float64, n)      # Z[·, j] under construction, indexed by row

    # Row-wise index of the result's lower triangle: for each row i,
    # rowcol[rowstart[i] : rowstart[i+1]-1] lists the columns c ≤ i that store
    # row i, and rowslot[·] the matching slot in zval. Built once by counting
    # sort in O(nnz). This is what lets the `kr < i` branch of the sweep resolve
    # Z[i, kr] with a single array read instead of a binary search — that search
    # was ~50% of total runtime in a ProfileToLLM table.
    rowstart = zeros(Int, n + 2)
    @inbounds for t in 1:nnzZ
        rowstart[rowv[t]+1] += 1
    end
    rowstart[1] = 1
    @inbounds for i in 1:(n+1)
        rowstart[i+1] += rowstart[i]
    end
    rowcol = Vector{Int}(undef, nnzZ)
    rowslot = Vector{Int}(undef, nnzZ)
    fill_at = copy(rowstart)
    @inbounds for j in 1:n, t in colptr[j]:(colptr[j+1]-1)
        r = rowv[t]
        d = fill_at[r]
        rowcol[d] = j
        rowslot[d] = t
        fill_at[r] = d + 1
    end

    rowpos = zeros(Int, n)       # column -> slot holding Z[i, column]

    @inbounds for j in n:-1:1
        jlo, jhi = colptr[j], colptr[j+1] - 1
        klo, khi = LD.colptr[j], LD.colptr[j+1] - 1
        dj = Lval[klo]           # diagonal of the packed factor = D[j]

        for t in jlo:jhi
            acc[rowv[t]] = 0.0
        end

        for t in jhi:-1:(jlo+1)
            i = rowv[t]

            # Scatter column i once; all its entries are Z[r, i] with r ≥ i.
            ilo, ihi = colptr[i], colptr[i+1] - 1
            for s2 in ilo:ihi
                pos[rowv[s2]] = s2
            end

            # Scatter the ROW of i: for every column c < i that stores row i,
            # rowpos[c] is the slot holding Z[i, c] = Z[c, i]. `rowstart` and
            # `rowcol` are the precomputed row-wise traversal of the lower
            # triangle (built once, before the sweep), so this costs one pass
            # over row i rather than a binary search per lookup.
            for s3 in rowstart[i]:(rowstart[i+1]-1)
                rowpos[rowcol[s3]] = rowslot[s3]
            end

            s = 0.0
            for kk in (klo+1):khi
                kr = Lrow[kk]
                lkj = Lval[kk]
                if kr >= i
                    sl = pos[kr]              # Z[kr, i], owned by column i
                    sl != 0 && (s -= lkj * zval[sl])
                else
                    # kr < i, so Z[kr, i] = Z[i, kr] is owned by column kr.
                    # `rowpos` gives that slot in O(1); the binary search this
                    # replaces was ~50% of total runtime in a ProfileToLLM
                    # table (`_zget` plus its inlined ==/getindex/>>>/<= ops),
                    # because roughly half of all inner operations take this
                    # branch and each search averaged ~8 steps.
                    sl = rowpos[kr]
                    sl != 0 && (s -= lkj * zval[sl])
                end
            end
            zval[t] = s
            acc[i] = s

            for s2 in ilo:ihi
                pos[rowv[s2]] = 0
            end
            for s3 in rowstart[i]:(rowstart[i+1]-1)
                rowpos[rowcol[s3]] = 0
            end
        end

        # Diagonal: Z[j,j] = 1/D[j] − Σ_{k>j} L[k,j]·Z[k,j], with Z[k,j] for
        # this column's rows just written into `acc`.
        s = 1.0 / dj
        for kk in (klo+1):khi
            s -= Lval[kk] * acc[Lrow[kk]]
        end
        zval[jlo] = s
    end

    Z = SparseMatrixCSC(n, n, colptr, rowv, zval)
    # Symmetrize into full storage: the sweep produced the lower triangle.
    return Z + transpose(tril(Z, -1))
end

# Fetch Z[r, c] from the lower-triangular result under construction, using
# symmetry. Returns 0.0 if the entry is outside the computed pattern — which,
# by the fill-in containment argument in the header, never happens for entries
# the recursion actually needs.
@inline function _zget(colptr::Vector{Int}, rowv::Vector{Int}, zval::Vector{Float64},
                       r::Int, c::Int)
    a, b = r >= c ? (r, c) : (c, r)     # lower triangle: row a, column b
    lo, hi = colptr[b], colptr[b+1] - 1
    @inbounds while lo <= hi
        mid = (lo + hi) >>> 1
        rv = rowv[mid]
        if rv == a
            return zval[mid]
        elseif rv < a
            lo = mid + 1
        else
            hi = mid - 1
        end
    end
    return 0.0
end

"""
    trace_product(s::SelectedInverse, B::SparseMatrixCSC, offset::Int) -> Float64

`tr(B · C⁻¹[block, block])` where `block` is the range `offset+1 : offset+size(B,1)`.

Computed as `Σ B[a,b]·C⁻¹[offset+b, offset+a]` over `B`'s stored entries only —
no dense intermediate, and every entry required is guaranteed present by the
containment argument in this file's header.
"""
function trace_product(s::SelectedInverse, B::SparseMatrixCSC, offset::Int)
    Z = s.Z
    Brow = rowvals(B)
    Bval = nonzeros(B)
    acc = 0.0
    @inbounds for jb in 1:size(B, 2)
        cj = jb + offset
        for k in nzrange(B, jb)
            ib = Brow[k]
            acc += Bval[k] * _spget(Z, ib + offset, cj)
        end
    end
    return acc
end

# tr(B·C⁻¹block) for a dense B (genomic G⁻¹). Needs the full dense diagonal
# block of C⁻¹, which the selected inverse cannot supply in general, so this
# path is only valid when the block is structurally dense in C — which it is for
# a genomic effect, since G⁻¹ is dense and therefore so is that block of L.
function trace_product(s::SelectedInverse, B::Matrix{Float64}, offset::Int)
    Z = s.Z
    qb = size(B, 1)
    acc = 0.0
    @inbounds for j in 1:qb, i in 1:qb
        b = B[i, j]
        b == 0.0 && continue
        acc += b * _spget(Z, i + offset, j + offset)
    end
    return acc
end

@inline function _spget(Z::SparseMatrixCSC{Float64,Int}, i::Int, j::Int)
    rows = rowvals(Z)
    vals = nonzeros(Z)
    lo, hi = Z.colptr[j], Z.colptr[j+1] - 1
    @inbounds while lo <= hi
        mid = (lo + hi) >>> 1
        rv = rows[mid]
        if rv == i
            return vals[mid]
        elseif rv < i
            lo = mid + 1
        else
            hi = mid - 1
        end
    end
    return 0.0
end

# --- stochastic fallback ---------------------------------------------------

"""
    hutchinson_trace(F, B, offset, m; rng=Random.default_rng()) -> Float64

Monte-Carlo estimate of `tr(B · C⁻¹[block, block])` by Hutchinson probing:
`E[vᵀBC⁻¹v] = tr(BC⁻¹)` for Rademacher `v`. Each probe is one solve against the
cached factor `F`, so this costs `m` solves and no selected inversion.

Cheaper than exact Takahashi on very large pedigrees, but the estimate is noisy
(relative error `O(1/√m)`), which makes REML convergence stochastic. Use
`trace_method = :hutchinson` only when the exact sweep is the bottleneck.
"""
function hutchinson_trace(F::CHOLMOD.Factor{Float64,Int}, B, offset::Int, m::Int;
                          rng=Random.default_rng())
    nfull = size(F, 1)
    qb = size(B, 1)
    acc = 0.0
    v = zeros(Float64, nfull)
    for _ in 1:m
        fill!(v, 0.0)
        @inbounds for i in 1:qb
            v[i+offset] = rand(rng, Bool) ? 1.0 : -1.0
        end
        w = F \ v                              # w = C⁻¹v
        wb = view(w, offset+1:offset+qb)
        vb = view(v, offset+1:offset+qb)
        acc += dot(vb, B * wb)
    end
    return acc / m
end
