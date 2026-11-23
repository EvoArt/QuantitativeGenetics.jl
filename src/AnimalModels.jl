"""
    AnimalModels

Fast quantitative-genetics and genomic-prediction algorithms, built around an
ASReml-class AI-REML animal model.

The design priority is speed on the mixed model equations:

- `A⁻¹` is written down directly from the pedigree by Henderson's rules; `A` is
  never formed and never inverted, and `log|A|` comes free from the Mendelian
  sampling variances.
- The MME coefficient matrix is assembled by scatter-adding into cached `nzval`
  indices — no sparse arithmetic and no allocation per iteration.
- The fill-reducing ordering is chosen by measuring the candidates, and the
  symbolic factorization is computed once for the entire fit.
- The average-information matrix reuses that factor via `k+1` right-hand sides,
  so second-order information costs no traces at all.
- The one trace the score does need is supplied exactly by a Takahashi selected
  inverse, with Hutchinson probing available for very large pedigrees.

Entry point: [`fit_reml`](@ref).
"""
module AnimalModels

using LinearAlgebra
using SparseArrays
using Printf
using Random

include("pedigree.jl")
include("relatedness.jl")
include("mme.jl")
include("sparseinv.jl")
include("reml.jl")
include("results.jl")

# pedigree
export Pedigree, ainverse, logdetA, meuwissen_luo, mendelian_variances

# relatedness
export AbstractRelatedness, IdentityRelatedness, PedigreeRelatedness,
       GenomicRelatedness, Kinv, logdetK, inbreeding, genomic_relationship

# mixed model equations
export RandomEffect, animal_effect, MMEWorkspace, build_mme, update_mme!,
       factorize_mme!, solve_mme!, fixed_effects, random_effect

# selected inverse
export SelectedInverse, selected_inverse, trace_product, hutchinson_trace

# REML
export REMLResult, REMLWorkspace, reml_objective!, reml_gradient_ai!, fit_reml

# results
export heritability, variance_ratio, blup, reliability, accuracy

end # module
