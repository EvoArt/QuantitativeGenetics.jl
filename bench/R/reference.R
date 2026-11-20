# R reference implementations, invoked by bench/external.jl.
#
# Three independent checks of the Julia package:
#
#   1. nadiv::makeAinv  — the standard R implementation of Henderson's rules and
#      pedigree inbreeding. Validates our A⁻¹ and F directly.
#   2. sommer::mmer     — a real animal-model REML fitter (Newton-Raphson / AI).
#      Validates our variance components and fixed effects on a pedigree model.
#   3. A brute-force REML −2logL formed from an explicit dense V. This is the
#      arbiter for the log-likelihood: sommer's reported value is an internal
#      convergence metric on a different scale, so it cannot be compared
#      directly, whereas this can.
#
# Usage: Rscript reference.R <data-dir>

suppressMessages({
  library(nadiv)
  library(sommer)
  library(Matrix)
})

d <- commandArgs(trailingOnly = TRUE)[1]

ped <- read.csv(file.path(d, "ped.csv"))
dat <- read.csv(file.path(d, "data.csv"))
ped$id   <- as.character(ped$id)
ped$sire <- ifelse(is.na(ped$sire), NA, as.character(ped$sire))
ped$dam  <- ifelse(is.na(ped$dam),  NA, as.character(ped$dam))

# --- 1. A⁻¹ and inbreeding vs nadiv ---------------------------------------
t0 <- Sys.time()
ai <- makeAinv(ped)
t_ainv <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
Ainv_r <- ai$Ainv
F_r <- ai$f

qg <- read.csv(file.path(d, "ainv_qg.csv"), header = FALSE)
names(qg) <- c("i", "j", "v")
Aq <- sparseMatrix(i = qg$i, j = qg$j, x = qg$v, symmetric = TRUE,
                   dims = dim(Ainv_r))
cat(sprintf("R_AINV_MAXDIFF %.4e\n",
            max(abs(as.matrix(Aq) - as.matrix(as(Ainv_r, "symmetricMatrix"))))))
cat(sprintf("R_AINV_TIME %.4f\n", t_ainv))

Fq <- scan(file.path(d, "F_qg.csv"), quiet = TRUE)
cat(sprintf("R_F_MAXDIFF %.4e\n", max(abs(Fq - F_r))))
cat(sprintf("R_F_MEAN %.6f\n", mean(F_r)))

# --- 2. animal model vs sommer --------------------------------------------
A <- as.matrix(makeA(ped))
rownames(A) <- colnames(A) <- ped$id
dat$idf <- factor(as.character(dat$id), levels = ped$id)

t0 <- Sys.time()
fm <- mmer(y ~ x, random = ~ vsr(idf, Gu = A), rcov = ~ units,
           data = dat, verbose = FALSE)
t_sommer <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

vc <- unlist(fm$sigma)
be <- unlist(fm$Beta$Estimate)
cat(sprintf("R_SOMMER_SIGMA_A %.8f\n", vc[1]))
cat(sprintf("R_SOMMER_SIGMA_E %.8f\n", vc[2]))
cat(sprintf("R_SOMMER_BETA %.6f %.6f\n", be[1], be[2]))
cat(sprintf("R_SOMMER_TIME %.4f\n", t_sommer))

# --- 3. brute-force REML logLik at sommer's estimates ----------------------
# The arbiter for the likelihood scale. Includes the (n−p)·log(2π) constant,
# matching what fit_reml reports.
n <- nrow(dat)
X <- cbind(1, dat$x)
p <- ncol(X)
V  <- vc[1] * A + vc[2] * diag(n)
Vi <- solve(V)
XtViX <- t(X) %*% Vi %*% X
P  <- Vi - Vi %*% X %*% solve(XtViX) %*% t(X) %*% Vi
m2ll <- determinant(V, log = TRUE)$modulus +
        determinant(XtViX, log = TRUE)$modulus +
        t(dat$y) %*% P %*% dat$y + (n - p) * log(2 * pi)
cat(sprintf("R_BRUTE_LOGLIK %.6f\n", -0.5 * as.numeric(m2ll)))
