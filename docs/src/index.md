```@meta
CurrentModule = QuantitativeGenetics
```

# QuantitativeGenetics.jl

QuantitativeGenetics.jl fits single-trait animal models by average-information restricted maximum likelihood (AI-REML). Random effects can use a pedigree, a genomic relationship matrix, or an identity relationship. The result provides variance components, fixed effects, and predicted breeding values.

## Fit a genomic animal model

Rows of `markers` correspond to individuals, columns to markers, and values are allele dosages in `{0, 1, 2}`. This small simulated example supplies one phenotype per individual:

```julia
using QuantitativeGenetics, Random, LinearAlgebra, SparseArrays

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

Inspect `result.converged` before interpreting estimates. For a pedigree model, use `PedigreeRelatedness(sire, dam)` in place of `GenomicRelatedness`; parents must precede offspring and `0` means an unknown parent. See the [algorithm walkthrough](https://github.com/EvoArt/QuantitativeGenetics.jl/blob/master/docs/walkthrough.qmd) for details on the mixed model equations and computation.

## API reference

```@autodocs
Modules = [QuantitativeGenetics]
```
