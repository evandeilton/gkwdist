# tests/testthat/test-mcdonald-no-clamping.R
# Regression tests for llmc(), grmc() and hsmc() after the data clamps were
# removed from bpmc.cpp.
#
# Three defects lived in these loops:
#
#   1. Every observation was clamped to [1e-10, 1-1e-10] before use. That moved
#      the log-likelihood by (gamma*lambda - 1) * (log(1e-10) - log(x)) nats --
#      23 nats for a single observation at 1e-20 -- and broke the identity
#      llmc(gamma, delta, 1) == llbeta(gamma, delta), where the two are the same
#      model. The discrepancy reached 1140 nats.
#
#   2. For delta > 1000 the term delta * log(1 - x^lambda) was floored at
#      -700 * n, so the objective stopped growing with delta while the constant
#      term kept growing. The negative log-likelihood became unbounded below:
#      llmc(c(1e300, 1e300, 1e-6), x) returned -2.77e+302, a global minimum at
#      absurd parameters, with a visible step at delta = 1000.
#
#   3. grmc() and hsmc() floored v = 1 - x^lambda at 1e-10 and capped their
#      lambda terms at +/-1e6, so the gradient plateaued where the objective
#      kept moving.
#
# The clamps mattered most because grmc() and hsmc() shared them with llmc():
# checking the analytic gradient against numDeriv::grad(llmc) passed anyway.
# These tests use a closed-form reference written in R instead.

ref_nll <- function(p, x) {
  g <- p[1]; d <- p[2]; l <- p[3]
  -sum(log(l) - lbeta(g, d + 1) + (g * l - 1) * log(x) +
         d * log(-expm1(l * log(x))))
}

test_that("llmc uses observations as given, without clamping", {
  # x below 1e-10 used to be pulled up to 1e-10.
  for (xv in c(1e-10, 1e-11, 1e-15, 1e-20)) {
    data <- c(xv, 0.4, 0.6)
    expect_equal(llmc(c(2, 3, 1.5), data), ref_nll(c(2, 3, 1.5), data),
                 tolerance = 1e-12)
  }
  # x above 1 - 1e-10 used to be pulled down.
  for (xv in c(1 - 1e-10, 1 - 1e-12, 1 - 1e-15)) {
    data <- c(0.2, 0.5, xv)
    expect_equal(llmc(c(2, 3, 1.5), data), ref_nll(c(2, 3, 1.5), data),
                 tolerance = 1e-12)
  }
})

test_that("llmc equals llbeta at lambda = 1, the model they share", {
  datasets <- list(
    c(1e-20, 0.4, 0.6),
    c(1e-15, 1e-6, 0.3, 0.7, 1 - 1e-6, 1 - 1e-15),
    c(0.2, 0.5, 0.8, 1 - 1e-4, 1 - 1e-8, 1 - 1e-12)
  )
  for (data in datasets) {
    for (gd in list(c(2, 3), c(0.5, 0.5), c(50, 50))) {
      target <- -sum(stats::dbeta(data, gd[1], gd[2] + 1, log = TRUE))
      expect_equal(llmc(c(gd, 1), data), target, tolerance = 1e-10)
      expect_equal(llmc(c(gd, 1), data), llbeta(gd, data), tolerance = 1e-10)
    }
  }
})

test_that("the negative log-likelihood is bounded below", {
  set.seed(2024)
  x <- rmc(200, 2.0, 1.5, 1.3)
  # This used to return -2.77e+302: a global minimum any optimiser would take.
  expect_gt(llmc(c(1e300, 1e300, 1e-6), x), 0)
  expect_gt(llmc(c(1e6, 1e6, 1e-3), x), 0)
})

test_that("there is no step in llmc at delta = 1000", {
  set.seed(2024)
  x <- rmc(200, 2.0, 1.5, 1.3)
  for (d in c(999, 1000, 1000.001, 1e4, 1e6)) {
    expect_equal(llmc(c(2, d, 1.3), x), -sum(dmc(x, 2, d, 1.3, log = TRUE)),
                 tolerance = 1e-8)
  }
  # continuity across the former threshold
  lo <- llmc(c(2, 999.999, 1.3), x)
  hi <- llmc(c(2, 1000.001, 1.3), x)
  expect_lt(abs(hi - lo), 1)
})

test_that("grmc and hsmc agree with the closed-form reference", {
  skip_if_not_installed("numDeriv")
  datasets <- list(
    { set.seed(11); runif(60, 0.08, 0.92) },
    c(1e-12, 1e-8, 1e-4, 0.2, 0.5, 0.8),
    c(0.2, 0.5, 0.8, 1 - 1e-4, 1 - 1e-8, 1 - 1e-12)
  )
  # gamma + delta kept under 100: beyond that grmc switches to an asymptotic
  # digamma expansion, a separate defect this change does not touch.
  pars <- list(c(2, 3, 1.5), c(0.5, 0.5, 0.5), c(0.3, 2, 0.2), c(2, 3, 12))
  for (data in datasets) {
    for (p in pars) {
      g_ref <- numDeriv::grad(function(q) ref_nll(q, data), p)
      expect_equal(as.numeric(grmc(p, data)), g_ref, tolerance = 1e-6)

      H_ref <- numDeriv::jacobian(function(q) as.numeric(grmc(q, data)), p)
      expect_equal(max(abs(hsmc(p, data) - H_ref) / pmax(abs(H_ref), 1e-8)), 0,
                   tolerance = 1e-5)
    }
  }
})

test_that("the lambda gradient keeps its limit as x approaches 1", {
  skip_if_not_installed("numDeriv")
  # x^lambda log(x) / (1 - x^lambda) converges to -1/lambda as x -> 1, so the
  # gradient settles rather than diverging. The point is which value it settles
  # on: the former v = max(v, 1e-10) drove the contribution to about -1e-5
  # instead of -1/1.3 = -0.769, five orders of magnitude adrift.
  base <- c(0.3, 0.5, 0.7)
  par <- c(2, 3, 1.3)
  for (xv in c(1 - 1e-9, 1 - 1e-11, 1 - 1e-13, 1 - 1e-15)) {
    data <- c(base, xv)
    g <- as.numeric(grmc(par, data))
    expect_true(all(is.finite(g)))
    expect_equal(g, numDeriv::grad(function(q) ref_nll(q, data), par),
                 tolerance = 1e-6)
  }

  # The delta component carries sum(log(1 - x^lambda)), which is where the
  # floor actually bit: for x = 1 - 1e-15 and lambda = 1.3 the true log(v) is
  # -34.28, and clamping v at 1e-10 reported -23.03 instead.
  for (xv in c(1 - 1e-12, 1 - 1e-15)) {
    data <- c(base, xv)
    g <- as.numeric(grmc(par, data))
    expect_equal(g[2], numDeriv::grad(function(q) ref_nll(q, data), par)[2],
                 tolerance = 1e-7)
  }
})

test_that("well-behaved data is untouched by the change", {
  set.seed(99)
  x <- rmc(2000, 2.0, 1.5, 1.3)
  fit <- optim(c(1, 1, 1), function(q) llmc(q, x),
               gr = function(q) as.numeric(grmc(q, x)), method = "BFGS")
  expect_equal(fit$convergence, 0L)
  expect_equal(fit$par, c(2.0, 1.5, 1.3), tolerance = 0.15)
  expect_equal(fit$value, ref_nll(fit$par, x), tolerance = 1e-10)
})
