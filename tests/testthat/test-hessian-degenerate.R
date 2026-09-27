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
# The interim fix returned NaN with a warning, the honest answer until the
# chain was reworked in log space. That rework is done: hsgkw() is now built
# from the same log-space blocks as hskkw(), so these cases have their true
# Hessians. The reference is the nesting identity -- at gamma = 1, delta = 0,
# lambda = 1 the (alpha, beta) block is hskw() -- and numDeriv where delta is
# far enough from 0 for a central difference.

test_that("hsgkw is finite where the chain underflows, and equals the nested Kw", {
  degenerate <- list(
    list(x = c(0.80, 0.85, 0.90, 0.95), par = c(1, 500, 1, 0, 1)),
    list(x = c(0.80, 0.85, 0.90, 0.95, 0.10), par = c(1, 500, 1, 0, 1)),
    list(x = c(0.5, 0.6, 0.7), par = c(1, 2000, 1, 0, 1))
  )
  for (cs in degenerate) {
    expect_silent(H <- hsgkw(cs$par, cs$x))
    expect_true(is.matrix(H))
    expect_equal(dim(H), c(5L, 5L))
    expect_true(all(is.finite(H)))
    expect_equal(H, t(H))
    expect_equal(H[1:2, 1:2], hskw(cs$par[1:2], cs$x), tolerance = 1e-12)
  }
})

test_that("hsgkw no longer reports the Hessian of a smaller sample", {
  # The specific number the old code returned: n / alpha^2 with every
  # observation dropped. The true value is 1996.267.
  x <- c(0.80, 0.85, 0.90, 0.95)
  H <- hsgkw(c(1, 500, 1, 0, 1), x)
  expect_false(isTRUE(all.equal(H[1, 1], length(x))))
  expect_equal(H[1, 1], 1996.267, tolerance = 1e-6)
})

test_that("hsgkw matches numDeriv in the regime that used to return NaN", {
  skip_if_not_installed("numDeriv")
  x5 <- c(0.10, 0.25, 0.40, 0.72, 0.99)
  cases <- list(
    list(x = c(0.80, 0.85, 0.90, 0.95), par = c(1, 500, 1, 0.3, 1)),
    list(x = c(0.5, 0.6, 0.7), par = c(1, 2000, 1, 0.3, 1)),
    list(x = x5, par = c(1, 200, 1.5, 2, 1)),
    list(x = x5, par = c(2, 300, 1, 0.5, 1)),
    list(x = c(x5, 1e-9), par = c(40, 2, 0.5, 0.5, 0.3))
  )
  for (cs in cases) {
    H <- hsgkw(cs$par, cs$x)
    J <- numDeriv::jacobian(function(q) as.numeric(grgkw(q, cs$x)), cs$par)
    Hn <- numDeriv::hessian(function(q) llgkw(q, cs$x), cs$par)
    info <- paste(cs$par, collapse = ",")
    expect_true(all(is.finite(H)), info = info)
    expect_lt(max(abs(H - J)) / max(abs(J)), 1e-7)
    expect_lt(max(abs(H - Hn)) / max(abs(Hn)), 1e-7)
  }
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
