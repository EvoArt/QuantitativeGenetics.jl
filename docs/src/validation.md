# Validation

Every fast path is checked against an independent reference, never against a
stored number. This page collects those checks: against established software,
against other code paths within Julia, and in the package test suite.

## Agreement with reference implementations


| quantity | reference | agreement |
|---|---|---|
| inbreeding ``F`` | `nadiv::makeAinv` | **exact** (0.0e+00) |
| ``\mathbf{A}^{-1}`` | `nadiv::makeAinv` | 1.8e-15 |
| ``\sigma^2_a, \sigma^2_e`` | `sommer::mmer` | ~5e-6 relative |
| ``\boldsymbol{\beta}`` | `sommer::mmer` | 6 decimal places |
| ``\log L`` | brute-force dense-``\mathbf{V}`` REML in R | 1.0e-08 |
| ``-2\log L`` | dense-``\mathbf{V}`` REML in Julia | machine precision |
| score, AI matrix | **ForwardDiff** on the dense objective | ~1e-15 |
| selected inverse | `inv(Matrix(C))` | 1e-17 |

The autodiff check deserves a note. Sparse AD cannot accelerate AI-REML — the
speed comes from an algebraic identity AD cannot find, and CHOLMOD has no AD
rule. But ForwardDiff applied to a small *dense* reference objective immediately
caught two errors in the hand-derived score and AI matrix that both produced
plausible output. **Autodiff as a test oracle, not a speed path.**

The comparisons with nadiv and sommer use identical pedigree datasets. The
log-likelihood is checked against an explicit dense-``\mathbf{V}`` calculation
rather than sommer's reported value, because sommer prints an internal convergence
metric on a different scale. The `external` CI job reruns the R comparison and
asserts this agreement on every push.

## Cross-checks within Julia

`julia --project=bench bench/compare.jl` runs two further checks.

**Against MixedModels.jl.** MixedModels.jl is a mature, independently written REML
implementation. It cannot take a pedigree relationship matrix, but it fits the
i.i.d. random-intercept model, which exercises the whole optimisation loop against
outside code:

| n | ``\sigma^2_\text{group}`` rel. diff | ``\sigma^2_\text{resid}`` rel. diff | logLik abs. diff | speed |
|---:|---:|---:|---:|---:|
|    200 | 1.1e-08 | 1.1e-09 | 6.0e-12 |  0.9× |
|  2,000 | 2.1e-06 | 1.0e-07 | 1.2e-10 | 86.7× |
| 10,000 | 2.4e-08 | 1.2e-09 | 2.1e-09 | 10.7× |

Fixed effects match to five decimal places throughout.

**Pedigree versus rotated parameterisation.** ``\mathbf{u} \sim N(\mathbf{0}, \sigma^2_u\mathbf{A})``
with ``\mathbf{A} = \mathbf{L}\mathbf{L}'`` is equivalent to
``\mathbf{w} \sim N(\mathbf{0}, \sigma^2_u\mathbf{I})`` on the design ``\mathbf{Z}\mathbf{L}``.
The two run through entirely different code (sparse Henderson ``\mathbf{A}^{-1}``
versus a dense identity structure), so agreement checks the pedigree path against
something other than itself:

| animals | ``\sigma^2_u`` rel. diff | ``\sigma^2_e`` rel. diff | logLik abs. diff | sparse speedup |
|---:|---:|---:|---:|---:|
| 200 | 7.0e-15 | 2.5e-10 | 4.0e-12 | 13× |
| 600 | 2.4e-15 | 3.6e-10 | 1.3e-11 | 58× |

## Test suite

`julia --project=. -e 'using Pkg; Pkg.test()'` checks each fast path against an
independent reference:

- ``\mathbf{A}^{-1}`` and inbreeding against a dense tabular ``\mathbf{A}`` built from the recursive rule.
- ``\log|\mathbf{A}|`` against `logdet(A)`.
- The MME solution and ``\log|\mathbf{C}|`` against the dense coefficient matrix.
- The Takahashi selected inverse against `inv(Matrix(C))`.
- ``-2\log L`` against a brute-force REML that forms the explicit ``n \times n`` ``\mathbf{V}``.
- The analytic score against ForwardDiff, and the AI matrix against
  ``\tfrac12\mathbf{y}'\mathbf{P}\dot{\mathbf{V}}_i\mathbf{P}\dot{\mathbf{V}}_j\mathbf{P}\mathbf{y}``.
- End-to-end recovery of known variance components from simulated data.
