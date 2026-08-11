library(testthat)
library(gkwdist)

# =============================================================================
# Componentwise validation of the analytical derivatives.
#
# The existing derivative tests evaluate generic interior parameter values.
# The bugs fixed in 1.1.5 lived exactly where those tests do not look:
#
#   * the degenerate points gamma = 1, beta = 1, lambda = 1 and delta = 0,
#     where a mixed second derivative loses the factor that the pure second
#     derivatives keep, so guards of the form `if (abs(p - 1) > eps)` silently
#     zeroed true entries;
#   * data with observations near 0 combined with moderate alpha, where
#     1 - x^alpha and 1 - v^beta round to 1 and 0 in double precision and the
#     old clamping at 1e-10 replaced log(w) = -53 by log(1e-10) = -23.
#
# Every gradient component and every Hessian entry is therefore compared
# individually against two independent references:
#   (1) the general GKw routines restricted to the constrained parameter point
#       (exact by construction: the sub-families are GKw with fixed parameters);
#   (2) numDeriv Richardson extrapolation on the sub-family's own likelihood.
# =============================================================================

# family -> full (alpha, beta, gamma, delta, lambda) embedding
to_full <- function(family, p) {
  switch(family,
    bkw  = c(p[1], p[2], p[3], p[4], 1),
    kkw  = c(p[1], p[2], 1, p[3], p[4]),
    ekw  = c(p[1], p[2], 1, 0, p[3]),
    mc   = c(1, 1, p[1], p[2], p[3]),
    kw   = c(p[1], p[2], 1, 0, 1),
    beta = c(1, 1, p[1], p[2], 1)
  )
}

# indices of the free parameters inside the full GKw vector
free_idx <- list(bkw = 1:4, kkw = c(1, 2, 4, 5), ekw = c(1, 2, 5),
                 mc = 3:5, kw = 1:2, beta = 3:4)

par_names <- list(bkw = c("alpha", "beta", "gamma", "delta"),
                  kkw = c("alpha", "beta", "delta", "lambda"),
                  ekw = c("alpha", "beta", "lambda"),
                  mc = c("gamma", "delta", "lambda"),
                  kw = c("alpha", "beta"),
                  beta = c("gamma", "delta"))

# Parameter grids centred on the degenerate values, with alpha large enough
# that the near-boundary observations matter.
grids <- list(
  bkw  = expand.grid(alpha = c(0.5, 2, 3.5), beta = c(0.5, 1, 3),
                     gamma = c(0.5, 1, 2), delta = c(0, 0.5, 2)),
  kkw  = expand.grid(alpha = c(0.5, 2, 3.5), beta = c(0.5, 1, 3),
                     delta = c(0, 0.5, 2), lambda = c(0.5, 1, 2.5)),
  ekw  = expand.grid(alpha = c(0.5, 2, 3.5), beta = c(0.5, 1, 3),
                     lambda = c(0.5, 1, 2.5)),
  mc   = expand.grid(gamma = c(0.5, 1, 2), delta = c(0, 0.5, 2),
                     lambda = c(0.5, 1, 2.5)),
  kw   = expand.grid(alpha = c(0.5, 2, 3.5), beta = c(0.5, 1, 3)),
  beta = expand.grid(gamma = c(0.5, 1, 2), delta = c(0, 0.5, 2))
)

# "extreme" carries observations down to ~1e-7, which is what broke the old
# clamped implementations; "moderate" is an ordinary well-behaved sample.
make_datasets <- function() {
  set.seed(7)
  list(moderate = rgkw(300, 2, 3, 1.5, 0.5, 2),
       extreme  = rgkw(250, 0.7, 1.4, 0.9, 1.5, 0.8))
}

rel_err <- function(actual, expected) {
  max(abs(actual - expected) / pmax(abs(expected), 1))
}

test_that("sub-family log-likelihoods match the general GKw likelihood", {
  datasets <- make_datasets()
  for (family in names(grids)) {
    ll_fn <- get(paste0("ll", family))
    grid <- grids[[family]]
    for (x in datasets) {
      for (i in seq_len(nrow(grid))) {
        p <- as.numeric(grid[i, ])
        target <- llgkw(to_full(family, p), x)
        if (!is.finite(target)) next
        expect_equal(ll_fn(p, x), target, tolerance = 1e-8,
                     info = sprintf("%s at (%s)", family,
                                    paste(p, collapse = ", ")))
      }
    }
  }
})

test_that("every gradient component matches the general GKw score", {
  datasets <- make_datasets()
  for (family in names(grids)) {
    ll_fn <- get(paste0("ll", family))
    gr_fn <- get(paste0("gr", family))
    grid <- grids[[family]]
    idx <- free_idx[[family]]
    for (x in datasets) {
      for (i in seq_len(nrow(grid))) {
        p <- as.numeric(grid[i, ])
        full <- to_full(family, p)
        if (!is.finite(llgkw(full, x))) next
        reference <- grgkw(full, x)[idx]
        analytic <- gr_fn(p, x)
        if (!all(is.finite(reference)) || !all(is.finite(analytic))) next
        # componentwise, so a single wrong entry cannot hide behind the others
        for (k in seq_along(analytic)) {
          expect_equal(analytic[k], reference[k], tolerance = 1e-7,
                       info = sprintf("%s d/d%s at (%s)", family,
                                      par_names[[family]][k],
                                      paste(p, collapse = ", ")))
        }
      }
    }
  }
})

test_that("every Hessian entry matches the general GKw Hessian", {
  datasets <- make_datasets()
  for (family in names(grids)) {
    hs_fn <- get(paste0("hs", family))
    grid <- grids[[family]]
    idx <- free_idx[[family]]
    for (x in datasets) {
      for (i in seq_len(nrow(grid))) {
        p <- as.numeric(grid[i, ])
        full <- to_full(family, p)
        if (!is.finite(llgkw(full, x))) next
        reference <- hsgkw(full, x)[idx, idx]
        analytic <- hs_fn(p, x)
        if (!all(is.finite(reference)) || !all(is.finite(analytic))) next
        for (r in seq_len(nrow(analytic))) {
          for (cc in seq_len(ncol(analytic))) {
            expect_equal(analytic[r, cc], reference[r, cc], tolerance = 1e-7,
                         info = sprintf("%s d2/d%s d%s at (%s)", family,
                                        par_names[[family]][r],
                                        par_names[[family]][cc],
                                        paste(p, collapse = ", ")))
          }
        }
      }
    }
  }
})

test_that("derivatives agree with numDeriv at the degenerate parameter values", {
  skip_if_not_installed("numDeriv")
  datasets <- make_datasets()

  # one representative degenerate configuration per family
  cases <- list(
    list(family = "bkw", par = c(2, 1, 1, 0.5)),    # gamma = 1 and beta = 1
    list(family = "kkw", par = c(2, 1, 0.5, 1)),    # beta = 1 and lambda = 1
    list(family = "ekw", par = c(2, 1, 1)),         # beta = 1 and lambda = 1
    list(family = "mc",  par = c(1, 0.5, 1)),       # gamma = 1 and lambda = 1
    list(family = "kw",  par = c(1, 1)),
    list(family = "beta", par = c(1, 0.5))
  )

  for (case in cases) {
    ll_fn <- get(paste0("ll", case$family))
    gr_fn <- get(paste0("gr", case$family))
    hs_fn <- get(paste0("hs", case$family))
    for (x in datasets) {
      f <- function(z) ll_fn(z, x)
      numeric_grad <- numDeriv::grad(f, case$par)
      numeric_hess <- numDeriv::hessian(f, case$par)
      expect_equal(gr_fn(case$par, x), numeric_grad, tolerance = 1e-5,
                   info = sprintf("%s gradient", case$family))
      expect_equal(hs_fn(case$par, x), numeric_hess, tolerance = 1e-5,
                   info = sprintf("%s Hessian", case$family))
    }
  }
})

test_that("likelihoods stay accurate for observations near zero", {
  # x^alpha underflows so that 1 - x^alpha is exactly 1 in double precision;
  # the log-space formulation is the only way to recover log(w) here.
  log1mexp_r <- function(u) ifelse(u > -log(2), log(-expm1(u)), log1p(-exp(u)))
  reference_nll <- function(x, a, b, g, d, l) {
    lx <- log(x)
    lv <- log1mexp_r(a * lx)
    lw <- log1mexp_r(b * lv)
    lz <- log1mexp_r(l * lw)
    -sum(log(l) + log(a) + log(b) - lbeta(g, d + 1) +
           (a - 1) * lx + (b - 1) * lv + (g * l - 1) * lw + d * lz)
  }

  x <- c(1e-7, 1e-5, 1e-3, 0.05, 0.3, 0.7, 0.95)

  expect_equal(llekw(c(3.5, 0.5, 0.5), x),
               reference_nll(x, 3.5, 0.5, 1, 0, 0.5), tolerance = 1e-10)
  expect_equal(llkkw(c(3.5, 0.5, 0.5, 0.5), x),
               reference_nll(x, 3.5, 0.5, 1, 0.5, 0.5), tolerance = 1e-10)
  expect_equal(llbkw(c(3.5, 0.5, 1.2, 0.5), x),
               reference_nll(x, 3.5, 0.5, 1.2, 0.5, 1), tolerance = 1e-10)
  expect_equal(llgkw(c(3.5, 0.5, 1.2, 0.5, 0.5), x),
               reference_nll(x, 3.5, 0.5, 1.2, 0.5, 0.5), tolerance = 1e-10)
})
