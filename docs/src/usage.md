```@meta
CurrentModule = AnimalModels
```

# Fitting models

## Pedigree animal model

A pedigree is two integer vectors, `sire` and `dam`, giving each animal's parents
as row indices. Parents must precede their offspring, and `0` means unknown.
[`PedigreeRelatedness`](@ref) builds the sparse inverse relationship matrix
``\mathbf{A}^{-1}`` directly from the pedigree:

```@example pedigree
using AnimalModels, LinearAlgebra, SparseArrays, Random, Statistics

rng = MersenneTwister(1)
q = 500
sire = zeros(Int, q); dam = zeros(Int, q)
for i in 21:q                       # the first 20 animals are founders
    sire[i] = rand(rng, 1:i-1)
    dam[i] = rand(rng, 1:i-1)
end
K = PedigreeRelatedness(sire, dam)

A = inv(Matrix(Kinv(K)))            # dense A, only to simulate breeding values
u = cholesky(Symmetric(A)).L * randn(rng, q)
covariate = randn(rng, q)
y = 10 .+ 0.5 .* covariate .+ u .+ randn(rng, q)

X = [ones(q) covariate]             # fixed effects
Z = sparse(1.0I, q, q)              # one record per animal
result = fit_reml(y, X, [RandomEffect(:animal, Z, K)])
```

Use [`inbreeding`](@ref) for the animals' inbreeding coefficients:

```@example pedigree
maximum(inbreeding(K))
```

## What a fit returns

`fit_reml` returns a [`REMLResult`](@ref):

```@example pedigree
(result.converged, result.σ², result.σ²ₑ, result.β)
```

[`heritability`](@ref) gives the estimate with a delta-method standard error, and
[`blup`](@ref) the predicted random effects. [`reliability`](@ref) and
[`accuracy`](@ref) summarise how well each breeding value is predicted:

```@example pedigree
heritability(result)
```

```@example pedigree
cor(blup(result), u)
```

## Several random effects

Give each random effect a name, an incidence matrix and a relatedness. Here
records are repeated measurements on animals, so a permanent-environment effect
with an identity relatedness sits alongside the additive genetic effect:

```@example pedigree
nrec = 3                                # records per animal
animal = repeat(1:q, inner=nrec)
Zr = sparse(1:q*nrec, animal, 1.0, q*nrec, q)
pe = randn(rng, q)
yr = 10 .+ Zr * (u .+ pe) .+ randn(rng, q * nrec)

effects = [RandomEffect(:animal, Zr, K),
           RandomEffect(:pe, Zr, IdentityRelatedness(q))]
fit = fit_reml(yr, ones(q * nrec, 1), effects)
fit.σ², fit.σ²ₑ
```

Name the effect to extract its predictions: `blup(fit; effect = :pe)`.

## Genomic relatedness

[`genomic_relationship`](@ref) builds VanRaden's genomic relationship matrix from
marker dosages. [`GenomicRelatedness`](@ref) factorises it once for the whole fit.
A VanRaden matrix is always singular, so it is blended toward the identity before
factorising; the default `blend = 0.01` is enough for that, and 0.01–0.05 is
typical:

```julia
G = genomic_relationship(markers)
K = GenomicRelatedness(G; blend = 0.05)
result = fit_reml(y, X, [RandomEffect(:genomic, Z, K)])
```

Pass `base` to blend toward the pedigree relationships of the genotyped animals
instead of the identity.

## Large problems

`fit_reml` computes the traces in the REML score exactly for small problems and
by Hutchinson probing for large ones; set `trace_method = :exact` or
`:hutchinson` to choose. `verbose = true` prints each iteration. See
[`fit_reml`](@ref) for every option, and [Algorithms](@ref) for what they do.
