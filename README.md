# AnimalModels.jl

[![CI](https://github.com/EvoArt/AnimalModels.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/EvoArt/AnimalModels.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![Coverage](https://codecov.io/gh/EvoArt/AnimalModels.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/EvoArt/AnimalModels.jl)
[![Dev docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://EvoArt.github.io/AnimalModels.jl/dev/)

AnimalModels.jl fits single-trait animal models by average-information REML
(AI-REML). A model can have any number of random effects, each with its own
incidence matrix and relatedness: a pedigree `A`, a genomic `G`, or the identity
for permanent-environment, litter and other grouping factors. Runtime
dependencies are standard libraries only.

```julia
using Pkg
Pkg.add("AnimalModels")
```

```julia
using AnimalModels, SparseArrays, LinearAlgebra

K = PedigreeRelatedness(sire, dam)   # parents precede offspring, 0 = unknown
q = size(K, 1)
X = [ones(q) covariate]              # fixed effects
Z = sparse(1.0I, q, q)               # one record per animal

result = fit_reml(y, X, [RandomEffect(:animal, Z, K)])
result.σ², result.σ²ₑ                # genetic and residual variance
blup(result)                         # predicted breeding values
heritability(result)                 # (h², standard error)
```

The pedigree inverse `A⁻¹` is built directly by Henderson's rules, the mixed
model equations are reassembled without allocation and refactorised with one
symbolic analysis per fit, and the REML score's traces come from a Takahashi
selected inverse or Hutchinson probing. On pedigree models this is 45–108× faster
than the R package sommer and roughly level with BLUPF90, with variance
components agreeing with both.

The [documentation](https://EvoArt.github.io/AnimalModels.jl/dev/) has a
guide to fitting models, a worked genomic-prediction example on wheat, an
explanation of the algorithms, benchmarks, validation against nadiv, sommer,
BLUPF90 and MixedModels.jl, and the API reference.
