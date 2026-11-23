# Performance

Timings below are from an 8-core Windows machine with OpenBLAS, on randomly mated
pedigrees with one record per animal and two fixed effects. Absolute times vary
by up to ±40% between runs on this machine; ratios between methods are stable.

## Fit times

`fit_reml` end to end, including all iterations, for each way of computing the
traces the REML score needs (see [The one hard term: traces via selected inversion](@ref)):

| animals | exact traces | Hutchinson (50 probes) |
|--------:|-------------:|-----------:|
|   1,000 |      0.29 s  |    0.12 s  |
|   2,500 |      3.95 s  |    0.28 s  |
|   5,000 |     43.1 s   |    1.03 s  |

Both agree on the variance components to three decimal places. The default,
`trace_method = :auto`, uses the exact sweep for small problems and switches to
Hutchinson probing above 20,000 nonzeros in the Cholesky factor.

Per-iteration split at q = 5000 (`nnz(L)` = 705k, fill 13.2×):

| stage | time |
|---|---:|
| MME assembly | 0.034 ms |
| numeric factorization | 19.5 ms |
| objective (assemble + factor + solve) | 21.5 ms |
| gradient + AI (Hutchinson, 50 probes) | 86 ms |
| gradient + AI (exact Takahashi) | 5164 ms |

Pedigree setup is cheap: 50,000 animals take 72 ms for inbreeding and 14 ms for
``\mathbf{A}^{-1}``.

## ``\mathbf{A}^{-1}`` construction versus nadiv

Julia 1.10.9 against R 4.5.1 and `nadiv::makeAinv` [nadiv](@cite), on identical pedigrees:

| animals | nadiv | AnimalModels | speedup |
|--------:|------:|-------------:|--------:|
|     400 | 0.072 s |     0.0001 s |   632× |
|     800 | 0.138 s |     0.0002 s |   769× |
|   2,000 | 0.531 s |     0.0006 s |   857× |

## Why this is faster than the R packages

Cross-validated against `nadiv` and `sommer` on identical pedigree datasets:

| animals | sommer | AnimalModels | speedup |
|--------:|-------:|-------------:|--------:|
|     400 |   1.9 s |     0.042 s |   45× |
|     800 |  18.2 s |      0.26 s |   72× |
|   2,000 | 323.1 s |      2.99 s |  108× |

The speedup *grows* with size, which tells you it is complexity, not constants.
Three reasons, in order of importance.

**1. Sparse ``\mathbf{A}^{-1}`` versus dense ``\mathbf{A}``.** This is the big one.
`sommer`'s interface takes `Gu = A` — the relationship matrix itself, dense,
``O(q^2)`` to store and ``O(q^3)`` to factor. We take ``\mathbf{A}^{-1}``, which is
under 1% dense and getting sparser. The cost of one factorisation:

| animals | dense ``\mathbf{V}``-style | sparse MME | ratio |
|--------:|-------:|------:|------:|
|   400 | 0.0069 s | 0.00029 s |  24× |
|   800 | 0.0874 s | 0.00093 s |  94× |
| 1,600 | 0.3236 s | 0.00255 s | 127× |
| 2,000 | 0.4283 s | 0.00346 s | 124× |

That 24×→124× progression tracks the observed 45×→108× end-to-end speedup
closely. The rest of the difference is everything below.

**2. Factorisation reuse.** We pay for fill-reducing analysis once per fit. A
dense implementation has no symbolic phase to amortise, and re-does the full
``O(q^3)`` decomposition every iteration.

**3. Allocation discipline.** Zero-allocation assembly, a cached selected
inverse reused by both the score and the EM fallback, and preallocated
multi-RHS buffers. R's copy-on-modify semantics make this pattern hard to
achieve.

It is worth being clear about what is *not* the reason. The algorithms are the
same ones ASReml and BLUPF90 use, and `sommer` implements AI-REML competently.
Julia versus R matters far less than the data structure: an R package built on
``\mathbf{A}^{-1}`` and sparse Cholesky (as `pedigreemm` [pedigreemm](@cite) partly is,
via `lme4`'s sparse machinery) would close most of this gap. **The lesson is
about representation, not language.**

## Comparison with BLUPF90

`sommer` is not the toughest opposition, and it would be misleading to stop
there. **BLUPF90** (`airemlf90`) uses the same data structure and the same
algorithms as this package — sparse ``\mathbf{A}^{-1}``, AI-REML, FSPAK selected
inversion [perezenciso1994](@cite) — so a comparison against it measures implementation
quality on a level field:

| animals | BLUPF90 | AnimalModels | ratio |
|--------:|--------:|-------------:|------:|
|     400 |  0.182 s |     0.047 s | **3.9× faster** |
|     800 |  0.202 s |     0.282 s | 1.4× slower |
|   2,000 |  0.517 s |     0.646 s | 1.25× slower |

Roughly at parity: ahead on small pedigrees, slightly behind on larger ones.
Variance components agree to 2e-5–2e-4 relative.

Getting here took two fixes worth recording, both found by measurement rather
than reasoning.

**The stopping rule was wasting half the iterations.** It tested only the
relative change in ``\theta``, so a fit that had already converged kept going: at
``q = 2000`` it ran 12 rounds with *identical* ``\theta`` and ``-2\log L`` from round 5
onward. Each of those no-op rounds paid for a full Takahashi sweep. Testing the
scaled score before proposing a step, plus an objective-drop test after, cut it
to 6 rounds for the same estimates.

**Half the sweep's runtime was binary search.** A `ProfileToLLM` table put
`_zget` and its inlined comparison/shift operations at ~50% of total runtime. The
cause: roughly half of all inner operations take the ``k < i`` branch, each
resolving a lookup by binary search averaging ~8 steps — 80 M search steps at
``q = 2000`` for a factor with only 148 k nonzeros. Building a row-wise index of
the result's lower triangle once, by counting sort in ``O(\mathrm{nnz})``, makes
that branch a single array read.


**Where BLUPF90 is still ahead.** It converges in 4 rounds to our 5–8, so its
step control is better. Its per-iteration cost is also flatter with pedigree
size, which points at FSPAK's selected inversion being a stronger implementation
than the sweep here. Both are worth revisiting.

ASReml could not be tested: it is commercial software from VSN International, not
on CRAN (verified), and requires a node-locked licence. If you hold one,
`bench/R/reference.R` is generic over which R fitter it calls.

## Reproducing the benchmarks

```bash
julia --project=bench bench/bench.jl      # timings and per-iteration profile
julia --project=bench bench/external.jl   # versus nadiv and sommer (needs R)
julia --project=bench bench/blupf90.jl    # versus BLUPF90 (see the file for binaries)
```

The R comparison needs `install.packages(c("nadiv", "sommer"))`. The
`workflow_dispatch` CI job reruns the R speed comparison and publishes the table as
a job summary.
