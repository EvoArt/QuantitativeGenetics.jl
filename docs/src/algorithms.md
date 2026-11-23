# Algorithms

This page explains how AnimalModels.jl fits the animal model: which algorithms it
uses, the implementation choices that make them fast, and the bugs that produced
plausible-looking output before they were caught. Measured speed and agreement
with other software are on the [Performance](@ref) and [Validation](@ref) pages.

## The problem

The animal model is the workhorse of quantitative genetics [henderson1984](@cite):

```math
\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \sum_i \mathbf{Z}_i\mathbf{u}_i + \mathbf{e},
\qquad
\mathbf{u}_i \sim N(\mathbf{0}, \sigma^2_i \mathbf{K}_i),
\qquad
\mathbf{e} \sim N(\mathbf{0}, \sigma^2_e \mathbf{I})
```

where ``\mathbf{K}_i`` is a relatedness matrix — the pedigree numerator
relationship matrix ``\mathbf{A}``, a genomic relationship matrix ``\mathbf{G}``, or
the identity for i.i.d. effects such as permanent environment. The task is to
estimate the variance components ``\sigma^2_i, \sigma^2_e`` by restricted maximum
likelihood [pattersonthompson1971](@cite), and to predict ``\hat{\mathbf{u}}`` (BLUP) and
``\hat{\boldsymbol{\beta}}`` (BLUE).

Writing ``\mathbf{V} = \sum_i \sigma^2_i \mathbf{Z}_i\mathbf{K}_i\mathbf{Z}_i' + \sigma^2_e\mathbf{I}``,
the restricted log-likelihood is

```math
-2\log L = \log|\mathbf{V}| + \log|\mathbf{X}'\mathbf{V}^{-1}\mathbf{X}| + \mathbf{y}'\mathbf{P}\mathbf{y},
\qquad
\mathbf{P} = \mathbf{V}^{-1} - \mathbf{V}^{-1}\mathbf{X}(\mathbf{X}'\mathbf{V}^{-1}\mathbf{X})^{-1}\mathbf{X}'\mathbf{V}^{-1}.
```

Taken literally this is hopeless: ``\mathbf{V}`` is ``n \times n`` and dense, and it
changes at every iteration. Every fast implementation avoids forming it. What
follows is how, and where the remaining time actually goes.

## Never form ``\mathbf{A}``; never invert it

### Henderson's rules

The single most important structural fact is that while ``\mathbf{A}`` is dense,
``\mathbf{A}^{-1}`` is *sparse*, and can be written down directly from the pedigree
without ever forming ``\mathbf{A}`` [henderson1976, quaas1976](@cite).

Decompose ``\mathbf{A} = (\mathbf{I}-\mathbf{P})^{-1}\mathbf{D}(\mathbf{I}-\mathbf{P})^{-\top}``,
where ``\mathbf{P}`` holds ``\tfrac12`` in the (animal, parent) cells and
``\mathbf{D} = \mathrm{diag}(d_i)`` collects the Mendelian sampling variances

```math
d_i = \begin{cases}
\tfrac12 - \tfrac14(F_s + F_d) & \text{both parents known}\\
\tfrac34 - \tfrac14 F_p & \text{one parent known}\\
1 & \text{neither known.}
\end{cases}
```

Then ``\mathbf{A}^{-1} = (\mathbf{I}-\mathbf{P})^\top\mathbf{D}^{-1}(\mathbf{I}-\mathbf{P})``,
which is a handful of contributions per animal:

```julia
for i in 1:q
    s, t = sire[i], dam[i]
    di = 1.0 / d[i]
    C[i,i] += di
    if s != 0
        C[i,s] -= di/2;  C[s,i] -= di/2;  C[s,s] += di/4
    end
    if t != 0
        C[i,t] -= di/2;  C[t,i] -= di/2;  C[t,t] += di/4
    end
    if s != 0 && t != 0
        C[s,t] += di/4;  C[t,s] += di/4
    end
end
```

This is ``O(q)`` work producing ``O(q)`` nonzeros. How much this matters is worth
seeing numerically — measured on real pedigrees:

| animals | nonzeros in ``\mathbf{A}`` | nonzeros in ``\mathbf{A}^{-1}`` |
|--------:|--------:|--------:|
|   300 | 91.5% | 1.61% |
|   600 | 90.9% | 0.81% |
| 1,200 | 90.6% | 0.40% |

``\mathbf{A}`` is ~91% dense and stays that way. ``\mathbf{A}^{-1}`` is under 1% and
gets *sparser* as the pedigree grows. **This single asymmetry is the main reason
this package is two orders of magnitude faster than the R alternatives**, and we
return to it in [Why this is faster than the R packages](@ref).

### ``\log|\mathbf{A}|`` comes free

Because ``\mathbf{P}`` is strictly lower triangular in a topologically sorted
pedigree, ``|\mathbf{I}-\mathbf{P}| = 1``, so ``|\mathbf{A}| = |\mathbf{D}|`` and

```math
\log|\mathbf{A}| = \sum_i \log d_i .
```

REML needs this every iteration; it costs one pass over a cached vector and no
linear algebra at all. This is why `Pedigree` retains `d` rather than folding
``1/d_i`` into the assembly and discarding it.

### Inbreeding: Meuwissen & Luo

The ``d_i`` need inbreeding coefficients, and ``F_i = \tfrac12 a(s_i, d_i)`` — a
single off-diagonal entry of ``\mathbf{A}``. The obvious recursive tabular rule
with memoisation works but allocates a growing hash table and recurses to
pedigree depth. [meuwissenluo1992](@citet) instead walk each animal's ancestor set once,
accumulating each ancestor's contribution ``L_j`` to the animal's two gametes, and
recovering ``F_i = \sum_j L_j^2 d_j - 1``.

The crucial detail is that ancestors are processed in **decreasing index order**.
Because the pedigree is topologically sorted, popping the largest pending index
guarantees every contribution to a node has arrived before that node is expanded:

```julia
while nanc > 0
    best = argmax over the pending list          # largest index
    j = anc[best];  nanc -= 1
    Fi += L[j]^2 * d[j]
    half = 0.5 * L[j]
    for parent in (sire[j], dam[j])
        parent == 0 && continue
        push parent onto the list if absent
        L[parent] += half
    end
end
F[i] = Fi - 1.0
```

Work vectors are allocated once and cleared only over the ancestors actually
touched, so per-animal cost is proportional to ancestor-set size, not ``q``.

Measured against the recursive-`Dict` approach (both give *identical* answers):

| animals | recursive Dict | Meuwissen & Luo | speedup |
|--------:|---------------:|----------------:|--------:|
|  2,000 | 0.015 s | 0.0020 s |  7.7× |
|  5,000 | 0.071 s | 0.0082 s |  8.7× |
| 10,000 | 0.214 s | 0.0186 s | 11.5× |

**Also used by:** every serious animal-breeding package. `nadiv` [nadiv](@cite) in R,
BLUPF90 [blupf90](@cite), ASReml [asreml](@cite), and MTDFREML all build ``\mathbf{A}^{-1}``
this way. This part of the algorithm is not where implementations differ.

## Henderson's mixed model equations

Rather than working with ``\mathbf{V}``, solve the MME [henderson1950](@cite) with
``\lambda_i = \sigma^2_e/\sigma^2_i``:

```math
\mathbf{C} =
\begin{bmatrix}
\mathbf{X}'\mathbf{X} & \mathbf{X}'\mathbf{Z}_1 & \cdots\\
\mathbf{Z}_1'\mathbf{X} & \mathbf{Z}_1'\mathbf{Z}_1 + \mathbf{K}_1^{-1}\lambda_1 & \cdots\\
\vdots & & \ddots
\end{bmatrix},
\qquad
\mathbf{r} = \begin{bmatrix}\mathbf{X}'\mathbf{y}\\ \mathbf{Z}_1'\mathbf{y}\\ \vdots\end{bmatrix}.
```

``\mathbf{C}`` is sparse whenever the ``\mathbf{K}_i^{-1}`` are, and the identity

```math
\log|\mathbf{V}| + \log|\mathbf{X}'\mathbf{V}^{-1}\mathbf{X}|
= \log|\mathbf{C}| - m\log\sigma^2_e + n\log\sigma^2_e + \sum_i\!\big(q_i\log\sigma^2_i + \log|\mathbf{K}_i|\big)
```

(``m = p + \sum q_i``) turns the whole objective into quantities one sparse
Cholesky already provides. The residual quadratic form likewise falls out of the
solution vector:

```math
\mathbf{y}'\mathbf{P}\mathbf{y} = \big(\mathbf{y}'\mathbf{y} - \hat{\mathbf{s}}'\mathbf{r}\big)/\sigma^2_e .
```

!!! warning "The ``-m\log\sigma^2_e`` term is easy to lose"
    ``\mathbf{C}`` is built in ``\lambda``-scaling, so recovering ``\log|\mathbf{V}|`` means
    putting back one ``\sigma^2_e`` per row of ``\mathbf{C}`` — all ``m`` of them. Omitting
    it is exact **only where ``\sigma^2_u = \sigma^2_e``**, so a single test at
    ``\theta = (1,1)`` passes while every other point is wrong. This cost me an
    afternoon; it is now checked at four separate ``\theta`` against a brute-force
    dense-``\mathbf{V}`` reference.

## AI-REML

### The average information trick

Newton–Raphson on the REML likelihood needs the Hessian, whose terms involve
``\mathrm{tr}(\mathbf{P}\,\partial\mathbf{V}/\partial\theta_i\,\mathbf{P}\,\partial\mathbf{V}/\partial\theta_j)``
— extremely expensive. Fisher scoring needs the expected information, also
expensive. [gilmour1995](@citet) observed that the *average* of the observed and expected
information has all the trace terms cancel, leaving

```math
\mathrm{AI}_{ij} = \tfrac12\, \mathbf{f}_i'\,\mathbf{P}\,\mathbf{f}_j,
\qquad
\mathbf{f}_i = \frac{\partial\mathbf{V}}{\partial\theta_i}\mathbf{P}\mathbf{y}.
```

For a random effect ``\partial\mathbf{V}/\partial\sigma^2_i = \mathbf{Z}_i\mathbf{K}_i\mathbf{Z}_i'``
and the MME give ``\hat{\mathbf{u}}_i = \sigma^2_i\mathbf{K}_i\mathbf{Z}_i'\mathbf{P}\mathbf{y}``,
so ``\mathbf{f}_i = \mathbf{Z}_i\hat{\mathbf{u}}_i/\sigma^2_i`` — already available.
Each ``\mathbf{P}\mathbf{f}_i`` comes from absorbing ``\mathbf{f}_i`` as a
pseudo-observation, i.e. **a solve against the factor we already have**:

```julia
mul!(rw.RHS, transpose(rw.Wmat), rw.F)   # W'f for all k+1 variates at once
A = F \ rw.RHS                           # ONE multi-RHS CHOLMOD solve
mul!(rw.PF, rw.Wmat, A)
@inbounds for c in 1:(k+1), t in 1:n
    rw.PF[t,c] = (rw.F[t,c] - rw.PF[t,c]) / σ²ₑ
end
```

So the entire second-order information costs ``k{+}1`` triangular solves and **no
traces**. This is the identity that makes AI-REML fast, and it is exactly what an
autodiff system cannot discover — AD would differentiate the factorisation
instead, which is strictly more work.

!!! warning "The ``1/\sigma^2_i`` scaling is not optional"
    Using ``\mathbf{f}_i = \mathbf{Z}_i\hat{\mathbf{u}}_i`` without dividing by
    ``\sigma^2_i`` yields an AI matrix that is symmetric, positive definite, and the
    right order of magnitude — but wrong by factors of ``\sigma^2_i\sigma^2_j``. The
    resulting steps *increase* ``-2\log L``. Verified against
    ``\tfrac12\mathbf{y}'\mathbf{P}\frac{\partial\mathbf{V}}{\partial\theta_i}\mathbf{P}\frac{\partial\mathbf{V}}{\partial\theta_j}\mathbf{P}\mathbf{y}``.

Two further practical points. AI approximates *half* the Hessian of ``-2\log L``,
so the step is ``-(2\,\mathrm{AI})^{-1}\mathbf{g}``; dropping the 2 turns quadratic
convergence into a 50-iteration crawl. And the iteration runs on the **log
scale**, so variances stay positive for any step length, with step halving and an
EM fallback [dempster1977](@cite) when AI proposes a non-improving direction — EM is
slower but monotone and cannot leave the parameter space. This hybrid is what
ASReml does, and it is why real data does not break it.

**Also used by:** ASReml [asreml](@cite), BLUPF90's `airemlf90` [blupf90](@cite), `sommer`
[sommer](@cite), and `breedR`. The algorithm is standard; the implementation is where
the time goes.

### The one hard term: traces via selected inversion

The *score* is not so lucky. It needs
``\mathrm{tr}(\mathbf{K}_i^{-1}\mathbf{C}^{u_iu_i})`` — elements of
``\mathbf{C}^{-1}`` at the positions where ``\mathbf{K}_i^{-1}`` is nonzero. Forming
``\mathbf{C}^{-1}`` is out of the question.

The Takahashi equations [takahashi1973](@cite) compute exactly the entries of
``\mathbf{C}^{-1}`` *on the sparsity pattern of the factor*, from the factor, in one
backward sweep. With ``\mathbf{C} = \mathbf{L}\mathbf{D}\mathbf{L}'`` and
``\mathbf{Z} = \mathbf{C}^{-1}``:

```math
Z_{ij} = -\!\!\sum_{k>j,\,L_{kj}\neq 0}\!\! L_{kj}Z_{ki} \quad (i>j),
\qquad
Z_{jj} = \frac{1}{D_j} - \!\!\sum_{k>j}\!\! L_{kj}Z_{kj}.
```

Sweeping columns from ``n`` down to 1, every right-hand-side entry is already
known. Crucially, fill-in gives
``\mathrm{pattern}(\mathbf{K}_i^{-1}) \subseteq \mathrm{pattern}(\mathbf{C}) \subseteq \mathrm{pattern}(\mathbf{L}+\mathbf{L}')``,
so every entry the trace asks for is available. This is what FSPAK
[perezenciso1994](@cite) computes inside BLUPF90.

!!! danger "A fast rewrite that was subtly wrong"
    Restricting ``k`` to column ``j``'s stored pattern is correct. A scatter-based
    reformulation that walks only the *source* column ran **8× faster and was wrong
    by 4e-5** — it silently drops the transposed ``r<k`` contributions. An error that
    size still looks like a converged fit. It was caught only by checking every
    computed entry against `inv(Matrix(C))`, which is now a permanent test.

For large pedigrees the exact sweep becomes the bottleneck, so Hutchinson
probing [hutchinson1990](@cite) is available: ``E[\mathbf{v}'\mathbf{B}\mathbf{C}^{-1}\mathbf{v}] = \mathrm{tr}(\mathbf{B}\mathbf{C}^{-1})``
for Rademacher ``\mathbf{v}``, each probe one solve against the cached factor.
Measured at ``q = 5000``: **43.1 s exact versus 1.03 s with 50 probes**, agreeing
to three decimals. `trace_method = :auto` switches to probing when ``\mathrm{nnz}(\mathbf{L}) > 20{,}000``.
This mirrors BLUPF90's Monte-Carlo AI-REML option [matilainen2013](@cite).

## Implementation optimisations

### Assemble by scatter-add, not sparse arithmetic

Across an entire fit, only the scalars ``\lambda_i`` change. So build
``\mathbf{C}``'s pattern once, cache the ``\theta``-independent values, and cache the
linear indices into `C.nzval` where each ``\mathbf{K}_i^{-1}`` entry lands:

```julia
function update_mme!(ws::MMEWorkspace, λ)
    nzv = nonzeros(ws.C)
    copyto!(nzv, ws.base_nzval)          # X'X, X'Zᵢ, Zᵢ'Zⱼ — never change
    @inbounds for i in eachindex(ws.effects)
        λi, idx, val = λ[i], ws.kidx[i], ws.kval[i]
        @simd for t in eachindex(idx)
            nzv[idx[t]] += λi * val[t]   # scatter-add
        end
    end
end
```

**Zero allocations, 0.034 ms at ``q=5000``.** The obvious alternative,
`ZtZ ./ σ²ₑ .+ Kinv ./ σ²ᵤ`, runs a full sparse-sparse add and allocates a new
matrix every iteration.

### One symbolic factorisation per fit

The sparsity pattern of ``\mathbf{C}`` is *constant* across the whole optimisation.
So `cholesky` runs the fill-reducing analysis once and `cholesky!(F, C)` reuses
it thereafter, redoing only the ``O(\mathrm{nnz}(\mathbf{L}))`` numeric work — for
every iteration *and* every step-halving retry.

### Measure the ordering; don't trust the folklore

Standard advice says to eliminate dense fixed-effect columns last, to avoid
catastrophic fill. Measured here, it is worth almost nothing: 98,850 versus
99,016 nonzeros against CHOLMOD's own AMD [amd](@cite). `:auto` builds both and keeps
the sparser factor.

!!! note "Measure the measurement"
    My first implementation reported fixed-effects-last as **8.5× worse** — because
    it was computing fill on a matrix whose values were all still zero. An all-zero
    sparse matrix factorises happily and reports meaningless structure. `_pattern_spd`
    now supplies diagonally dominant values for any symbolic experiment.

### What was *not* worth doing

`LoopVectorization` and `Octavian` were considered and rejected for the sparse
core. The Takahashi sweep and MME assembly are irregular gather/scatter —
memory-bandwidth-bound, not FLOP-bound — and `@turbo` cannot vectorise indirect
indexing well. `LoopVectorization` is also effectively unmaintained on Julia
≥1.11, so adopting it would pin the package to 1.10. Careful `@inbounds`,
column-major CSC access and zero allocation are what actually matter.

## Software using the same algorithms

| package | language | ``\mathbf{A}^{-1}`` | REML | selected inverse |
|---|---|---|---|---|
| **ASReml** [asreml](@cite) | Fortran (commercial) | Henderson | AI-REML | sparse |
| **BLUPF90** [blupf90](@cite) | Fortran | Henderson | AI-REML / EM | FSPAK [perezenciso1994](@cite) |
| **DMU** | Fortran | Henderson | AI-REML | sparse |
| **sommer** [sommer](@cite) | R/C++ | via dense ``\mathbf{A}`` | AI/NR | — |
| **pedigreemm** [pedigreemm](@cite) | R | Henderson | ML/REML via `lme4` | — |
| **nadiv** [nadiv](@cite) | R | Henderson | (matrices only) | — |
| **MixedModels.jl** | Julia | n/a | REML, sparse | — |
| **AnimalModels.jl** | Julia | Henderson + Meuwissen–Luo | AI-REML + EM | Takahashi + Hutchinson |
