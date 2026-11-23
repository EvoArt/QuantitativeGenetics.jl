# Example: genomic prediction in wheat

599 CIMMYT wheat lines genotyped at 1,279 DArT markers, with grain yield in four
environments (Crossa et al. 2010, distributed with the R package BGLR). The lines are
inbred, so a marker's presence/absence is a 0/2 dosage.

```@example wheat
using AnimalModels, RData, Statistics, SparseArrays, LinearAlgebra
import CodecXz, Downloads                      # BGLR's .RData files are xz-compressed

file = joinpath(tempdir(), "wheat.RData")
isfile(file) || Downloads.download("https://github.com/gdlc/BGLR-R/raw/master/data/wheat.RData", file)
wheat = load(file)

markers = 2 .* Float64.(wheat["wheat.X"])
yield = Float64.(wheat["wheat.Y"][:, 1])      # environment 1, standardised
size(markers)
```

## Genomic heritability

The VanRaden relationship matrix says how alike two lines are across all markers;
`blend` mixes in 5 % of the identity so that it can be inverted.

```@example wheat
G = GenomicRelatedness(genomic_relationship(markers); blend=0.05)
gblup = fit_reml(yield, ones(length(yield), 1), [animal_effect(:genomic, G)])
heritability(gblup)
```

## Predicting lines that were not phenotyped

BGLR assigns each line to one of ten folds. Fit without fold 1, then predict its lines
from their markers alone: their records are left out of the incidence matrix `Z`, but
their breeding values are still estimated through `G`.

```@example wheat
test = wheat["wheat.sets"] .== 1
Z = sparse(1.0I, length(yield), length(yield))[.!test, :]
cv = fit_reml(yield[.!test], ones(count(.!test), 1), [RandomEffect(:genomic, Z, G)])
cor(blup(cv)[test], yield[test])
```

That correlation, between predicted breeding values and observed yield in unseen
lines, is the usual measure of genomic prediction accuracy. For Bayesian marker
regression on the same data, see
[PracticalBayesianGenomics](https://evoart.github.io/PracticalBayesianGenomics.jl/).
