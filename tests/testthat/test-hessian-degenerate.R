# tests/testthat/test-hessian-degenerate.R
# hsgkw() where the chain underflows in linear space.
#
# Every observation must contribute to the Hessian, including those whose
# log(1-x^alpha), log(1-v^beta) or log(1-w^lambda) is non-finite in linear
# space. Skipping them would yield the Hessian of a smaller sample as a finite,
# symmetric matrix with no NaN and no warning, and wrong standard errors.
#
# With beta = 500 and four such observations, only the parameter-only terms
# would remain: H(alpha, alpha) = n / alpha^2 = 4 against a true 1996.3.
#
# The reference is the nesting identity -- at gamma = 1, delta = 0,
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

test_that("hsgkw does not report the Hessian of a smaller sample", {
  # n / alpha^2 is the value with every observation skipped. The true value
  # is 1996.267.
  x <- c(0.80, 0.85, 0.90, 0.95)
  H <- hsgkw(c(1, 500, 1, 0, 1), x)
  expect_false(isTRUE(all.equal(H[1, 1], length(x))))
  expect_equal(H[1, 1], 1996.267, tolerance = 1e-6)
})

test_that("hsgkw matches numDeriv where the chain underflows", {
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

test_that("the Hessian matches the gradient jacobian where both are finite", {
  skip_if_not_installed("numDeriv")
  set.seed(3)
  x <- runif(200, 0.05, 0.95)
  for (par in list(c(2, 3, 1.5, 2, 1.2), c(2, 3, 1.5, 2, 12))) {
    H <- hsgkw(par, x)
    J <- numDeriv::jacobian(function(q) as.numeric(grgkw(q, x)), par)
    expect_equal(max(abs(H - J) / pmax(abs(J), 1e-30)), 0, tolerance = 1e-6)
  }
})
