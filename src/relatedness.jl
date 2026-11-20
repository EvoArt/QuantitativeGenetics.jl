using LinearAlgebra
using SparseArrays

# ---------------------------------------------------------------------------
# Relatedness structures.
#
# A random effect in the animal model needs exactly two things from its
# relatedness matrix K, and NEITHER of them is K:
#
#   * K⁻¹  — sparse for pedigree A, dense for genomic G. Enters the mixed model
#            equations as the λᵢK⁻¹ term added to the (uᵢ, uᵢ) diagonal block.
#   * log|K| — needed by REML's −2logL at every iteration.
#
# So a relatedness structure IS "K⁻¹ plus log|K|". For the pedigree case both
# come free from the pedigree itself (Henderson's rules and Σ log dᵢ); for the
# genomic case one dense Cholesky at construction supplies both; for an identity
# effect (permanent environment, litter, i.i.d. group) K⁻¹ = I and log|K| = 0,
# which is worth its own type because it makes the MME contribution a pure
# diagonal scatter.
#
# Everything here is computed ONCE at construction. Nothing in this file runs
# inside the REML iteration.
# ---------------------------------------------------------------------------

"""
    AbstractRelatedness

Supertype for relatedness structures. Implementations provide [`Kinv`](@ref)
(the inverse relationship matrix) and [`logdetK`](@ref) (`log|K|`), both
precomputed at construction and constant throughout a REML fit.
"""
abstract type AbstractRelatedness end

"""
    Kinv(r::AbstractRelatedness)

The inverse relationship matrix `K⁻¹`. Sparse for pedigree structures, dense for
genomic ones.
"""
function Kinv end

"""
    logdetK(r::AbstractRelatedness)

`log|K|`, the log-determinant of the *relationship* matrix (not its inverse).
Required by the REML log-likelihood at every iteration; precomputed here.
"""
function logdetK end

Base.size(r::AbstractRelatedness, args...) = size(Kinv(r), args...)

# --- identity -------------------------------------------------------------

"""
    IdentityRelatedness(q)

`K = I` of order `q`: independent random effects, as used for permanent
environment, litter, or any i.i.d. grouping factor.

`K⁻¹ = I` is stored explicitly as a sparse identity so the MME assembly path is
uniform, but because it is diagonal it contributes only `q` scatter-adds and
adds no off-diagonal fill. `log|K| = 0` exactly.
"""
struct IdentityRelatedness <: AbstractRelatedness
    q::Int
    Kinv::SparseMatrixCSC{Float64,Int}
end
IdentityRelatedness(q::Integer) = IdentityRelatedness(Int(q), sparse(1.0I, q, q))

Kinv(r::IdentityRelatedness) = r.Kinv
logdetK(::IdentityRelatedness) = 0.0

# --- pedigree -------------------------------------------------------------

"""
    PedigreeRelatedness(ped::Pedigree)
    PedigreeRelatedness(sire, dam)

Pedigree relatedness: sparse `A⁻¹` by Henderson's rules, with `log|A|` obtained
free from the Mendelian sampling variances (`Σ log dᵢ` — see `pedigree.jl`).

`A` itself is never formed and never inverted, so this is O(q) in both time and
storage and scales to very large pedigrees.
"""
struct PedigreeRelatedness <: AbstractRelatedness
    ped::Pedigree
    Kinv::SparseMatrixCSC{Float64,Int}
    logdet::Float64
end

function PedigreeRelatedness(ped::Pedigree)
    return PedigreeRelatedness(ped, ainverse(ped), logdetA(ped))
end
PedigreeRelatedness(sire, dam) = PedigreeRelatedness(Pedigree(sire, dam))

Kinv(r::PedigreeRelatedness) = r.Kinv
logdetK(r::PedigreeRelatedness) = r.logdet

"""
    inbreeding(r::PedigreeRelatedness)

Inbreeding coefficients of the underlying pedigree.
"""
inbreeding(r::PedigreeRelatedness) = r.ped.F

# --- genomic --------------------------------------------------------------

"""
    GenomicRelatedness(G; blend=0.0, base=I)

Genomic relatedness from a genomic relationship matrix `G` (e.g. VanRaden's
`G = ZZᵀ / 2Σpⱼ(1−pⱼ)`). One dense Cholesky at construction yields both `G⁻¹`
and `log|G|`, so the REML iteration never touches `G` again.

`blend > 0` uses `(1−blend)·G + blend·base` instead, the standard regularisation
that keeps `G` invertible when markers make it singular or near-singular. `base`
defaults to the identity (a ridge toward independence); pass the pedigree `A`
restricted to the genotyped animals to blend toward pedigree instead. On real
marker data a small blend (0.01–0.05) is usually necessary.

`G⁻¹` is dense, so a genomic random effect makes the corresponding MME diagonal
block dense. That is intrinsic to GBLUP, not an implementation artifact.
"""
struct GenomicRelatedness <: AbstractRelatedness
    Kinv::Matrix{Float64}
    logdet::Float64
end

function GenomicRelatedness(G::AbstractMatrix; blend::Real=0.0, base=I)
    q = size(G, 1)
    size(G, 2) == q || throw(DimensionMismatch("G must be square, got $(size(G))"))
    Gb = if blend > 0
        basem = base isa UniformScaling ? Matrix{Float64}(base, q, q) : Matrix{Float64}(base)
        (1 - blend) .* Matrix{Float64}(G) .+ blend .* basem
    else
        Matrix{Float64}(G)
    end
    F = cholesky!(Symmetric(Gb))          # Gb is our own copy; factor in place
    ld = logdet(F)                        # log|G| free from the same factor
    Ginv = Matrix(inv(F))
    # Symmetrize: inv() of a Cholesky can leave asymmetry at the 1e-16 level,
    # which would otherwise make the MME coefficient matrix non-symmetric and
    # upset CHOLMOD's symmetric path.
    Ginv .= 0.5 .* (Ginv .+ transpose(Ginv))
    return GenomicRelatedness(Ginv, ld)
end

Kinv(r::GenomicRelatedness) = r.Kinv
logdetK(r::GenomicRelatedness) = r.logdet

"""
    genomic_relationship(M; centered=false) -> Matrix{Float64}

VanRaden method-1 genomic relationship matrix from an `n×p` marker dosage matrix
`M` with entries in `{0,1,2}`. Columns are centred by `2pⱼ` (allele frequencies
estimated from `M` unless `centered=true`) and scaled by `2Σpⱼ(1−pⱼ)`.

Pass the result to [`GenomicRelatedness`](@ref), typically with a small `blend`.
"""
function genomic_relationship(M::AbstractMatrix; centered::Bool=false)
    Mf = Matrix{Float64}(M)
    if centered
        Z = Mf
        # Recover the frequencies implied by the centring for the scale factor.
        p = fill(0.5, size(Mf, 2))
    else
        p = vec(sum(Mf; dims=1)) ./ (2 * size(Mf, 1))
        Z = Mf .- 2 .* p'
    end
    scale = 2 * sum(pj -> pj * (1 - pj), p)
    return (Z * transpose(Z)) ./ scale
end
