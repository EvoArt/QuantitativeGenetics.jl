```@meta
CurrentModule = AnimalModels
```

# AnimalModels.jl

AnimalModels.jl fits single-trait animal models by average-information restricted
maximum likelihood (AI-REML). A model can have any number of random effects, each
with its own incidence matrix and relatedness structure: a pedigree, a genomic
relationship matrix, or the identity for permanent-environment, litter and other
grouping factors. A fit gives variance components, fixed effects (BLUE) and
predicted breeding values (BLUP). Runtime dependencies are standard libraries only.

```julia
using Pkg
Pkg.add("AnimalModels")
```

## Fit a genomic animal model

Rows of `markers` correspond to individuals, columns to markers, and values are
allele dosages in `{0, 1, 2}`. This small simulated example supplies one phenotype
per individual:

```julia
using AnimalModels, Random, LinearAlgebra, SparseArrays

rng = MersenneTwister(42)
n = 60
markers = Float64.(rand(rng, 0:2, n, 120))
G = genomic_relationship(markers)
relatedness = GenomicRelatedness(G; blend=0.05)

X = ones(n, 1)                # fixed intercept
Z = sparse(1.0I, n, n)        # one record per individual
u = cholesky(Symmetric(G + 0.05I)).L * randn(rng, n)
y = 2 .+ u .+ 0.5 .* randn(rng, n)

result = fit_reml(y, X, [RandomEffect(:genomic, Z, relatedness)])
result.converged
result.σ²                       # random-effect variance
result.σ²ₑ                      # residual variance
blup(result)                    # predicted breeding values
heritability(result)            # estimate and standard error
```

Check `result.converged` before interpreting estimates.

## Where to go next

- [Fitting models](@ref) covers pedigree models, several random effects, and the
  quantities a fit returns.
- [Example: genomic prediction in wheat](@ref) fits real marker data and checks
  held-out prediction.
- [Algorithms](@ref) explains how the fit works and why it is fast.
- [Performance](@ref) and [Validation](@ref) report timings and agreement with
  nadiv, sommer, BLUPF90 and MixedModels.jl.
- The [API reference](@ref) lists every exported function.
