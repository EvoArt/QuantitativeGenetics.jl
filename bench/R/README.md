# External cross-validation against R

Requires R with `nadiv`, `sommer` and `Matrix`:

```r
install.packages(c("nadiv", "sommer"))
```

Run from the repository root:

```
julia --project=bench bench/external.jl
```

That script generates a shared dataset, runs the R references via `Rscript`,
fits the same data with AnimalModels, and prints the comparison. The
`.R` files here are invoked by it; they are not meant to be run standalone.
