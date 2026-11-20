# QuantitativeGenetics.jl

[![CI](https://github.com/EvoArt/QuantitativeGenetics.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/EvoArt/QuantitativeGenetics.jl/actions/workflows/CI.yml)
[![Dev docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://EvoArt.github.io/QuantitativeGenetics.jl/dev/)

Fast quantitative-genetics and genomic-prediction algorithms in Julia, built
around an ASReml-class **AI-REML animal model**.

Single trait, arbitrary number of random effects, each with its own incidence
matrix and relatedness structure — pedigree `A`, genomic `G`, or identity (for
permanent-environment, litter or other i.i.d. grouping factors).

Runtime dependencies are stdlib only: `LinearAlgebra`, `SparseArrays`, `Printf`,
`Random`.

Install from the repository until registration is complete:

```julia
using Pkg
Pkg.add(url="https://github.com/EvoArt/QuantitativeGenetics.jl")
```

The [documentation](https://EvoArt.github.io/QuantitativeGenetics.jl/dev/) has a complete, runnable example and an API reference.

📖 **[docs/walkthrough.qmd](docs/walkthrough.qmd)** — a full walkthrough of the
algorithms (Henderson's rules, Meuwissen & Luo, AI-REML, Takahashi selected
inversion), the implementation optimisations, the bugs that were subtle enough to
survive plausible-looking output, and a quantified explanation of why this is
45–108× faster than `sommer`. With references. Render with
`quarto render docs/walkthrough.qmd`.

## Usage

```julia
using QuantitativeGenetics, SparseArrays, LinearAlgebra

# Pedigree: parents must precede offspring, 0 = unknown.
K = PedigreeRelatedness(sire, dam)

q = size(K, 1)
X = [ones(q) covariate]          # fixed effects
Z = sparse(1.0I, q, q)           # one record per animal

effects = [RandomEffect(:animal, Z, K)]
result  = fit_reml(y, X, effects)

result.σ²[1]        # additive genetic variance
result.σ²ₑ          # residual variance
result.β            # BLUE of the fixed effects
blup(result)        # BLUP of breeding values
heritability(result)  # (h², delta-method standard error)
```

Several random effects, e.g. animal plus a permanent-environment term:

```julia
effects = [RandomEffect(:animal, Z, K),
           RandomEffect(:pe, Zpe, IdentityRelatedness(ngroups))]
result = fit_reml(y, X, effects)
blup(result; effect = :pe)
```

Genomic (GBLUP):

```julia
G = genomic_relationship(markers)          # VanRaden method 1, {0,1,2} dosages
K = GenomicRelatedness(G; blend = 0.05)    # blend keeps G invertible
result = fit_reml(y, X, [RandomEffect(:genomic, Z, K)])
```

## How it goes fast

- `A⁻¹` is written down directly from the pedigree by Henderson's rules — `A` is
  never formed or inverted — with inbreeding by Meuwissen & Luo (1992).
- `log|A| = Σ log dᵢ` comes free from the Mendelian sampling variances, so the
  determinant term REML needs each iteration costs nothing.
- The MME coefficient matrix is rebuilt by scatter-adding into cached `nzval`
  indices: **zero allocations**, 0.03 ms at q = 5000.
- The fill-reducing ordering is chosen by measuring both candidates, and the
  symbolic factorization is computed once for the whole fit.
- The average-information matrix reuses that factor via `k+1` right-hand sides
  in a single multi-RHS solve, so second-order information needs no traces.
- The one trace the score does need comes from a Takahashi selected inverse,
  with Hutchinson probing selected automatically on large problems.

## Performance

8-core Windows machine, OpenBLAS, randomly-mated pedigrees, one record per
animal, two fixed effects. `fit_reml` end to end, including all iterations:

| animals | exact traces | Hutchinson | `:auto` (default) |
|--------:|-------------:|-----------:|------------------:|
|   1,000 |      0.29 s  |    0.12 s  |           0.31 s  |
|   2,500 |      3.95 s  |    0.28 s  |           0.41 s  |
|   5,000 |     43.1 s   |    1.03 s  |           1.18 s  |

All three agree on the variance components to three decimal places. `:auto`
switches to probing above 150k nonzeros in the factor.

Per-iteration split at q = 5000 (`nnz(L)` = 705k, fill 13.2×):

| stage | time |
|---|---:|
| MME assembly | 0.034 ms |
| numeric factorization | 19.5 ms |
| objective (assemble + factor + solve) | 21.5 ms |
| gradient + AI (Hutchinson, 50 probes) | 86 ms |
| gradient + AI (exact Takahashi) | 5164 ms |

Pedigree setup is cheap: 50,000 animals take 72 ms for inbreeding and 14 ms for
`A⁻¹`.

## Comparison with established software

Cross-validated against **nadiv** (the standard R implementation of Henderson's
rules) and **sommer** (a mature R animal-model REML fitter), on identical
pedigree datasets. Run it yourself with `julia --project=bench bench/external.jl`;
CI asserts the correctness half on every push.

### Speed

Pedigree animal model, one record per animal, two fixed effects, fitted to
convergence. 8-core Windows machine, Julia 1.10.9, R 4.5.1.

| animals | sommer | QuantitativeGenetics | speedup |
|--------:|-------:|---------------------:|--------:|
|     400 |   1.9 s |             0.042 s |    45× |
|     800 |  18.2 s |              0.26 s |    72× |
|   2,000 | 323.1 s |              2.99 s |   108× |

The gap widens sharply with pedigree size: sommer's cost grows roughly cubically
in the number of animals, ours much closer to linearly. Absolute timings vary
±40% between runs on this machine (sommer at q = 2000 measured 227 s and 323 s
on two runs), but the ratio is stable.

The mechanism is the data structure, not the language: `sommer` takes the dense
relationship matrix `A` (~91% nonzero, and staying that way), while this package
takes the sparse `A⁻¹` (under 1%, and getting sparser as pedigrees grow). An R
package built on `A⁻¹` and sparse Cholesky would close most of this gap.

### vs BLUPF90 — the real benchmark

`sommer` is not the toughest opposition. **BLUPF90** (`airemlf90`) uses the same
approach this package does — sparse `A⁻¹`, AI-REML, FSPAK selected inversion — so
a comparison against it measures implementation quality on a level field. Run it
with `julia --project=bench bench/blupf90.jl` (see that file for how to obtain
the binaries).

| animals | BLUPF90 | QuantitativeGenetics | ratio |
|--------:|--------:|---------------------:|------:|
|     400 |  0.182 s |             0.047 s | **3.9× faster** |
|     800 |  0.202 s |             0.282 s | 1.4× slower |
|   2,000 |  0.517 s |             0.646 s | 1.25× slower |

Roughly at parity, ahead on small pedigrees and slightly behind on larger ones.
Variance components agree to 2e-5–2e-4 relative. BLUPF90 converges in 4 rounds
to our 5–8, so its step control remains better; its per-iteration cost is also
flatter, which points at FSPAK's selected inversion being a stronger
implementation than ours.

ASReml could not be tested — it is commercial and not on CRAN.

`A⁻¹` construction alone:

| animals | nadiv | QuantitativeGenetics | speedup |
|--------:|------:|---------------------:|--------:|
|     400 | 0.072 s |            0.0001 s |   632× |
|     800 | 0.138 s |            0.0002 s |   769× |
|   2,000 | 0.531 s |            0.0006 s |   857× |

### Correctness

Same datasets, so these are direct comparisons rather than reimplementations:

| quantity | reference | agreement |
|---|---|---|
| inbreeding coefficients `F` | nadiv | **exact** (0.0e+00) |
| `A⁻¹` | nadiv | 1.8e-15 |
| σ²ₐ, σ²ₑ | sommer | ~5e-6 relative |
| fixed effects β | sommer | 6 decimal places |
| logLik | brute-force dense-`V` REML in R | 1.0e-08 |

The log-likelihood is checked against an explicit dense-`V` calculation rather
than sommer's reported value, because sommer prints an internal convergence
metric on a different scale. That independent calculation confirms our number to
eight decimals.

## Validation within Julia

`julia --project=bench bench/compare.jl` runs two further cross-checks.

**vs MixedModels.jl** — a mature, independently written REML implementation. It
cannot take a pedigree relationship matrix, but it fits the i.i.d. random-intercept
model, which exercises the entire optimization loop against outside code:

| n | σ²_group rel. diff | σ²_resid rel. diff | logLik abs. diff | speed |
|---:|---:|---:|---:|---:|
|    200 | 1.1e-08 | 1.1e-09 | 6.0e-12 |  0.9× |
|  2,000 | 2.1e-06 | 1.0e-07 | 1.2e-10 | 86.7× |
| 10,000 | 2.4e-08 | 1.2e-09 | 2.1e-09 | 10.7× |

Fixed effects match to five decimal places throughout.

**Pedigree vs rotated parameterisation** — `u ~ N(0, σ²ᵤA)` with `A = LLᵀ` is
equivalent to `w ~ N(0, σ²ᵤI)` on the design `ZL`. The two run through entirely
different code (sparse Henderson `A⁻¹` versus a dense identity structure), so
agreement validates the pedigree path against something other than itself:

| animals | σ²ᵤ rel. diff | σ²ₑ rel. diff | logLik abs. diff | sparse speedup |
|---:|---:|---:|---:|---:|
| 200 | 7.0e-15 | 2.5e-10 | 4.0e-12 | 13× |
| 600 | 2.4e-15 | 3.6e-10 | 1.3e-11 | 58× |

Not compared: BLUPF90/ASReml (not installed) and R `sommer`/`nadiv` (not
installed, though R 4.5.1 is present). Those would be the natural next
references for the pedigree path specifically.

## Correctness

86 tests, each checking a fast path against an independent reference rather than
a stored number:

- `A⁻¹` and inbreeding against a dense tabular `A` built from the recursive rule.
- `log|A|` against `logdet(A)`.
- The MME solution and `log|C|` against the dense coefficient matrix.
- The Takahashi selected inverse against `inv(Matrix(C))`.
- `−2logL` against a brute-force REML that forms the explicit n×n `V`.
- The analytic score against **ForwardDiff**, and the AI matrix against
  `½·yᵀP(∂V/∂θᵢ)P(∂V/∂θⱼ)Py`.
- End-to-end recovery of known variance components from simulated data.

```
julia --project=. -e 'using Pkg; Pkg.test()'
julia --project=bench bench/bench.jl
```
