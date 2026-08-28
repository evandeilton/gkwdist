# tests/testthat/test-hessian-degenerate.R
# Regression tests for hsgkw() when the log-space chain underflows.
#
# hsgkw() used to `continue` past any observation whose log(1-x^alpha),
# log(1-v^beta) or log(1-w^lambda) came out non-finite. That silently computed
# the Hessian of a smaller sample and returned it as a finite, symmetric matrix
# with no NaN and no warning -- the worst possible failure mode for a quantity
# whose entire purpose is to produce standard errors.
#
# With beta = 500 and four observations, every observation was dropped and only
# the parameter-only terms survived: H(alpha, alpha) came back as
# n / alpha^2 = 4 against a true value of 1996.3, an error of a factor of 499.
# With five observations one survived, and the function returned the Hessian of
# a single point as if it were the Hessian of five.
#
# Computing those terms correctly needs the log-space rework of gkw.cpp. Until
# then the honest answer is a visible failure, which is what the function's own
# intermediate-value check already does further down.

test_that("hsgkw returns NaN and warns when the chain underflows", {
  degenerate <- list(
    list(x = c(0.80, 0.85, 0.90, 0.95), par = c(1, 500, 1, 0, 1)),
    list(x = c(0.80, 0.85, 0.90, 0.95, 0.10), par = c(1, 500, 1, 0, 1)),
    list(x = c(0.5, 0.6, 0.7), par = c(1, 2000, 1, 0, 1))
  )
  for (cs in degenerate) {
    expect_warning(H <- hsgkw(cs$par, cs$x), "underflow")
    expect_true(is.matrix(H))
    expect_equal(dim(H), c(5L, 5L))
    expect_true(all(is.nan(H)))
  }
})

test_that("hsgkw no longer reports the Hessian of a smaller sample", {
  # The specific number the old code returned: n / alpha^2 with every
  # observation dropped. Nothing about it looked wrong from the outside.
  x <- c(0.80, 0.85, 0.90, 0.95)
  H <- suppressWarnings(hsgkw(c(1, 500, 1, 0, 1), x))
  expect_false(isTRUE(all.equal(H[1, 1], length(x))))
  expect_true(is.nan(H[1, 1]))
})

test_that("well-behaved parameters are untouched", {
  set.seed(3)
  x <- runif(200, 0.05, 0.95)
  healthy <- list(
    c(2, 3, 1.5, 2, 1.2),
    c(40, 25, 15, 10, 12),
    c(1, 60, 1, 0, 1),
    c(0.5, 0.5, 0.5, 0.5, 0.5)
  )
  for (par in healthy) {
    H <- hsgkw(par, x)
    expect_true(all(is.finite(H)))
    expect_equal(H, t(H))              # symmetric
    expect_silent(hsgkw(par, x))       # no spurious warning
  }
})

test_that("the Hessian still matches the gradient jacobian where both are finite", {
  skip_if_not_installed("numDeriv")
  set.seed(3)
  x <- runif(200, 0.05, 0.95)
  for (par in list(c(2, 3, 1.5, 2, 1.2), c(2, 3, 1.5, 2, 12))) {
    H <- hsgkw(par, x)
    J <- numDeriv::jacobian(function(q) as.numeric(grgkw(q, x)), par)
    expect_equal(max(abs(H - J) / pmax(abs(J), 1e-30)), 0, tolerance = 1e-6)
  }
})
