using LinearAlgebra
using SparseArrays

# ---------------------------------------------------------------------------
# Pedigree: inbreeding, Mendelian sampling variances, and the sparse A⁻¹.
#
# The numerator relationship matrix A factorizes as
#
#     A = (I − P)⁻¹ D (I − P)⁻ᵀ
#
# where P holds ½ in the (i, parent) cells and D = diag(dᵢ) collects the
# Mendelian sampling variances
#
#     dᵢ = ½ − ¼(F_s + F_d)   both parents known
#        = ¾ − ¼F_p           one parent known (p the known one)
#        = 1                  neither parent known.
#
# TWO consequences drive everything in this file:
#
#   1. A⁻¹ = (I − P)ᵀ D⁻¹ (I − P) is SPARSE and can be written down directly
#      from the pedigree — Henderson's (1976) rules — in O(q) work and O(q)
#      nonzeros. A itself (dense, q×q) is never formed and never inverted.
#      This is the reason the animal model scales to millions of animals.
#
#   2. |A| = |D| = Π dᵢ, because |I − P| = 1 (P is strictly lower triangular
#      once the pedigree is topologically sorted). So
#
#          log|A| = Σ log dᵢ
#
#      is FREE — a single pass over a cached vector, no factorization. REML's
#      −2logL needs log|K| at every iteration, so this term costs nothing for
#      pedigree effects. We therefore keep `d` on the Pedigree rather than
#      folding 1/dᵢ into the A⁻¹ assembly and discarding it.
#
# Inbreeding uses Meuwissen & Luo (1992), an O(Σ mᵢ) algorithm (mᵢ = size of
# animal i's ancestor set) that walks each animal's ancestors once with an
# integer work stack. The obvious alternative — the recursive tabular rule
# a(i,j) = ½(a(i,s_j) + a(i,d_j)) with a Dict memo — allocates a growing hash
# table of O(q²) worst-case entries and recurses to pedigree depth, so it both
# slows down and risks stack overflow on deep real pedigrees.
# ---------------------------------------------------------------------------

"""
    Pedigree(sire, dam)

A topologically-sorted pedigree with its inbreeding coefficients and Mendelian
sampling variances precomputed.

`sire` and `dam` are length-`q` integer vectors; `sire[i]` / `dam[i]` give the
row index (in `1:q`) of animal `i`'s parent, or `0` if unknown. Animals must be
ordered so parents precede offspring, i.e. `sire[i] < i` and `dam[i] < i` (or
`0`) for every `i`. This is checked at construction.

Fields:

- `sire`, `dam` — the parent index vectors (stored as `Int32`).
- `F`           — inbreeding coefficient of each animal, by Meuwissen & Luo (1992).
- `d`           — Mendelian sampling variance of each animal.

`d` is retained because `log|A| = Σ log dᵢ`, which lets [`logdetK`](@ref) return
the pedigree log-determinant with no linear algebra at all — see the derivation
at the top of this file.
"""
struct Pedigree
    sire::Vector{Int32}
    dam::Vector{Int32}
    F::Vector{Float64}
    d::Vector{Float64}
end

function Pedigree(sire::AbstractVector{<:Integer}, dam::AbstractVector{<:Integer})
    q = length(sire)
    length(dam) == q || throw(DimensionMismatch("sire and dam must have equal length"))
    for i in 1:q
        (sire[i] == 0 || 1 <= sire[i] < i) ||
            throw(ArgumentError("pedigree not sorted: sire[$i] = $(sire[i]), must be 0 or < $i"))
        (dam[i] == 0 || 1 <= dam[i] < i) ||
            throw(ArgumentError("pedigree not sorted: dam[$i] = $(dam[i]), must be 0 or < $i"))
    end
    s = Vector{Int32}(sire)
    t = Vector{Int32}(dam)
    F = meuwissen_luo(s, t)
    d = mendelian_variances(s, t, F)
    return Pedigree(s, t, F, d)
end

Base.length(p::Pedigree) = length(p.sire)

"""
    mendelian_variances(sire, dam, F) -> Vector{Float64}

Mendelian sampling variance `dᵢ` of each animal given the inbreeding
coefficients `F`: `½ − ¼(F_s + F_d)` with both parents known, `¾ − ¼F_p` with
one, and `1` with neither.
"""
function mendelian_variances(sire, dam, F)
    q = length(sire)
    d = Vector{Float64}(undef, q)
    @inbounds for i in 1:q
        s, t = sire[i], dam[i]
        if s != 0 && t != 0
            d[i] = 0.5 - 0.25 * (F[s] + F[t])
        elseif s != 0
            d[i] = 0.75 - 0.25 * F[s]
        elseif t != 0
            d[i] = 0.75 - 0.25 * F[t]
        else
            d[i] = 1.0
        end
    end
    return d
end

"""
    meuwissen_luo(sire, dam) -> Vector{Float64}

Inbreeding coefficients by the Meuwissen & Luo (1992) algorithm.

`Fᵢ = ½·a(sireᵢ, damᵢ)`, and that single relationship is obtained by walking the
ancestors of animal `i` rather than by building any part of `A`. For each animal
the algorithm accumulates, for every ancestor `j`, the partial contribution
`L[j]` of `j` to `i`'s two gametes; the inbreeding then falls out as
`Σⱼ L[j]²·dⱼ − 1`.

Ancestors are processed in DECREASING index order, which is what makes a single
pass correct: because the pedigree is topologically sorted, an animal's parents
always have smaller indices, so popping the largest pending index guarantees
every contribution to a node has been accumulated before that node is expanded.
A boolean `inset` guard keeps each ancestor on the work list exactly once.

Work vectors are allocated once and cleared only over the ancestors actually
touched, so the per-animal cost is proportional to its ancestor-set size rather
than to `q`.
"""
function meuwissen_luo(sire, dam)
    q = length(sire)
    F = zeros(Float64, q)
    q == 0 && return F

    L = zeros(Float64, q)         # partial contribution of each ancestor
    d = Vector{Float64}(undef, q) # Mendelian variance, filled as we go
    inset = falses(q)             # is this ancestor currently on the work list?
    anc = Vector{Int}(undef, q)   # work list of ancestor indices (unsorted)
    touched = Vector{Int}(undef, q)

    @inbounds for i in 1:q
        s, t = Int(sire[i]), Int(dam[i])

        # Mendelian variance needs the PARENTS' F, which are already final
        # because parents precede offspring.
        if s != 0 && t != 0
            d[i] = 0.5 - 0.25 * (F[s] + F[t])
        elseif s != 0
            d[i] = 0.75 - 0.25 * F[s]
        elseif t != 0
            d[i] = 0.75 - 0.25 * F[t]
        else
            d[i] = 1.0
        end

        # A founder (either parent unknown) cannot be inbred: its parents share
        # no ancestor by definition of "unknown". Skip the walk entirely.
        if s == 0 || t == 0
            F[i] = 0.0
            continue
        end

        # Walk the ancestor set of i, seeded with i itself carrying weight 1.
        nanc = 1
        anc[1] = i
        inset[i] = true
        L[i] = 1.0
        ntouched = 1
        touched[1] = i
        Fi = 0.0

        while nanc > 0
            # Pop the LARGEST index in the work list. Linear scan is fine: the
            # list is short (ancestor frontier), and a heap's constant factor
            # costs more than it saves at these sizes.
            best = 1
            for k in 2:nanc
                if anc[k] > anc[best]
                    best = k
                end
            end
            j = anc[best]
            anc[best] = anc[nanc]
            nanc -= 1
            inset[j] = false

            Lj = L[j]
            Fi += Lj * Lj * d[j]

            # Push half of j's weight onto each of its parents.
            js, jd = Int(sire[j]), Int(dam[j])
            half = 0.5 * Lj
            if js != 0
                if !inset[js]
                    nanc += 1
                    anc[nanc] = js
                    inset[js] = true
                    ntouched += 1
                    touched[ntouched] = js
                end
                L[js] += half
            end
            if jd != 0
                if !inset[jd]
                    nanc += 1
                    anc[nanc] = jd
                    inset[jd] = true
                    ntouched += 1
                    touched[ntouched] = jd
                end
                L[jd] += half
            end
        end

        # Σ L[j]²dⱼ over the ancestor set equals a(i,i) = 1 + Fᵢ.
        F[i] = Fi - 1.0

        # Clear only what we touched — keeps the pass O(ancestors), not O(q).
        for k in 1:ntouched
            L[touched[k]] = 0.0
        end
    end
    return F
end

"""
    ainverse(ped::Pedigree) -> SparseMatrixCSC{Float64,Int}

Sparse inverse numerator relationship matrix `A⁻¹` by Henderson's (1976) rules,
built directly from the pedigree without ever forming `A`.

Each animal `i` with Mendelian variance `dᵢ` contributes `1/dᵢ` to `A⁻¹[i,i]`,
`−1/(2dᵢ)` to the `(i, parent)` and `(parent, i)` cells, and `1/(4dᵢ)` to the
`(parent, parent')` cells for each pair of present parents.

Contributions are accumulated as COO triplets and assembled with `sparse`, which
sums duplicate `(i,j)` entries. Exact nonzero counts are precomputed so the
triplet arrays are allocated once at the right size.
"""
function ainverse(ped::Pedigree)
    q = length(ped)
    sire, dam, d = ped.sire, ped.dam, ped.d

    # Exact triplet count: 1 diagonal per animal, plus 3 cells per known parent
    # ((i,p), (p,i), (p,p)), plus 2 more when both parents are known ((s,d),(d,s)).
    ntrip = q
    @inbounds for i in 1:q
        s, t = sire[i], dam[i]
        s != 0 && (ntrip += 3)
        t != 0 && (ntrip += 3)
        (s != 0 && t != 0) && (ntrip += 2)
    end

    I = Vector{Int}(undef, ntrip)
    J = Vector{Int}(undef, ntrip)
    V = Vector{Float64}(undef, ntrip)
    n = 0
    @inbounds for i in 1:q
        s, t = Int(sire[i]), Int(dam[i])
        di = 1.0 / d[i]
        half = -0.5 * di
        quarter = 0.25 * di

        n += 1; I[n] = i; J[n] = i; V[n] = di
        if s != 0
            n += 1; I[n] = i; J[n] = s; V[n] = half
            n += 1; I[n] = s; J[n] = i; V[n] = half
            n += 1; I[n] = s; J[n] = s; V[n] = quarter
        end
        if t != 0
            n += 1; I[n] = i; J[n] = t; V[n] = half
            n += 1; I[n] = t; J[n] = i; V[n] = half
            n += 1; I[n] = t; J[n] = t; V[n] = quarter
        end
        if s != 0 && t != 0
            n += 1; I[n] = s; J[n] = t; V[n] = quarter
            n += 1; I[n] = t; J[n] = s; V[n] = quarter
        end
    end
    return sparse(I, J, V, q, q)
end

"""
    logdetA(ped::Pedigree) -> Float64

`log|A| = Σ log dᵢ`, computed directly from the cached Mendelian sampling
variances. No factorization — see the derivation at the top of `pedigree.jl`.
"""
logdetA(ped::Pedigree) = sum(log, ped.d)
