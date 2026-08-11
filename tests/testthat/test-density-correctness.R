library(testthat)
library(gkwdist)

# =============================================================================
# Value-level checks on the density and distribution functions.
#
# The pre-existing PDF tests assert only type, length, non-negativity and
# finiteness. A vector of zeros satisfies all four, which is how dgkw() could
# return 0 everywhere (a name collision with R's own Rf_log1mexp, fixed in
# 1.1.5) while the suite stayed green. These tests pin down actual values.
# =============================================================================

test_that("every density integrates to one over (0, 1)", {
  configs <- list(
    c(2, 3, 1.5, 0.5, 2), c(0.8, 1.2, 2, 1, 0.9), c(3.5, 0.7, 0.8, 2, 1.5),
    c(1, 1, 1, 0, 1), c(2, 3, 1, 0, 1), c(1.5, 2.5, 1.2, 0.3, 1.8)
  )
  for (p in configs) {
    area <- integrate(function(z) dgkw(z, p[1], p[2], p[3], p[4], p[5]),
                      lower = 0, upper = 1, subdivisions = 2000)$value
    expect_equal(area, 1, tolerance = 1e-5,
                 info = sprintf("dgkw(%s)", paste(p, collapse = ", ")))
  }

  expect_equal(integrate(function(z) dbkw(z, 2, 3, 1.5, 0.5), 0, 1)$value, 1,
               tolerance = 1e-5)
  expect_equal(integrate(function(z) dkkw(z, 2, 3, 0.5, 2), 0, 1)$value, 1,
               tolerance = 1e-5)
  expect_equal(integrate(function(z) dekw(z, 2, 3, 2), 0, 1)$value, 1,
               tolerance = 1e-5)
  expect_equal(integrate(function(z) dmc(z, 1.5, 0.5, 2), 0, 1)$value, 1,
               tolerance = 1e-5)
  expect_equal(integrate(function(z) dkw(z, 2, 3), 0, 1)$value, 1,
               tolerance = 1e-5)
  expect_equal(integrate(function(z) dbeta_(z, 2, 3), 0, 1)$value, 1,
               tolerance = 1e-5)
})

test_that("the uniform special case returns exactly one", {
  # GKw(1, 1, 1, 0, 1) is Uniform(0, 1)
  x <- c(0.05, 0.2, 0.5, 0.8, 0.95)
  expect_equal(dgkw(x), rep(1, length(x)))
  expect_equal(pgkw(x), x)
  expect_equal(dkw(x, 1, 1), rep(1, length(x)))
})

test_that("sub-family densities equal the GKw density at the constrained point", {
  x <- c(0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95)
  expect_equal(dgkw(x, 2, 3, 1.5, 0.5, 1), dbkw(x, 2, 3, 1.5, 0.5))
  expect_equal(dgkw(x, 2, 3, 1, 0.5, 2), dkkw(x, 2, 3, 0.5, 2))
  expect_equal(dgkw(x, 2, 3, 1, 0, 2), dekw(x, 2, 3, 2))
  expect_equal(dgkw(x, 1, 1, 1.5, 0.5, 2), dmc(x, 1.5, 0.5, 2))
  expect_equal(dgkw(x, 2, 3, 1, 0, 1), dkw(x, 2, 3))
  expect_equal(dgkw(x, 1, 1, 2, 3, 1), dbeta_(x, 2, 3))

  # and the same for the CDFs
  expect_equal(pgkw(x, 2, 3, 1.5, 0.5, 1), pbkw(x, 2, 3, 1.5, 0.5))
  expect_equal(pgkw(x, 2, 3, 1, 0, 2), pekw(x, 2, 3, 2))
  expect_equal(pgkw(x, 2, 3, 1, 0, 1), pkw(x, 2, 3))
})

test_that("the Beta special case matches base R", {
  x <- c(0.05, 0.2, 0.5, 0.8, 0.95)
  # the package parameterises the Beta sub-family as Beta(gamma, delta + 1)
  expect_equal(dgkw(x, 1, 1, 2, 3, 1), dbeta(x, 2, 4))
  expect_equal(dbeta_(x, 2, 3), dbeta(x, 2, 4))
  expect_equal(pbeta_(x, 2, 3), pbeta(x, 2, 4))
  expect_equal(qbeta_(c(0.1, 0.5, 0.9), 2, 3), qbeta(c(0.1, 0.5, 0.9), 2, 4))

  # delta = 0 is the legitimate Beta(gamma, 1) boundary, not an invalid value
  expect_equal(dbeta_(x, 2, 0), dbeta(x, 2, 1))
  expect_equal(pbeta_(x, 2, 0), pbeta(x, 2, 1))
  expect_true(all(is.finite(rbeta_(5, 2, 0))))
})

test_that("the Kumaraswamy special case matches its closed form", {
  x <- c(0.05, 0.2, 0.5, 0.8, 0.95)
  a <- 2.3; b <- 1.7
  expect_equal(dkw(x, a, b), a * b * x^(a - 1) * (1 - x^a)^(b - 1))
  expect_equal(pkw(x, a, b), 1 - (1 - x^a)^b)
  p <- c(0.1, 0.25, 0.5, 0.75, 0.9)
  expect_equal(qkw(p, a, b), (1 - (1 - p)^(1 / b))^(1 / a))
})

test_that("densities are the derivative of their distribution functions", {
  skip_if_not_installed("numDeriv")
  cases <- list(
    list(d = function(z) dgkw(z, 2, 3, 1.5, 0.5, 2),
         p = function(z) pgkw(z, 2, 3, 1.5, 0.5, 2)),
    list(d = function(z) dbkw(z, 2, 3, 1.5, 0.5),
         p = function(z) pbkw(z, 2, 3, 1.5, 0.5)),
    list(d = function(z) dkkw(z, 2, 3, 0.5, 2),
         p = function(z) pkkw(z, 2, 3, 0.5, 2)),
    list(d = function(z) dekw(z, 2, 3, 2), p = function(z) pekw(z, 2, 3, 2)),
    list(d = function(z) dmc(z, 1.5, 0.5, 2), p = function(z) pmc(z, 1.5, 0.5, 2)),
    list(d = function(z) dkw(z, 2, 3), p = function(z) pkw(z, 2, 3)),
    list(d = function(z) dbeta_(z, 2, 3), p = function(z) pbeta_(z, 2, 3))
  )
  for (u in c(0.2, 0.5, 0.8)) {
    for (case in cases) {
      expect_equal(case$d(u), numDeriv::grad(case$p, u), tolerance = 1e-6)
    }
  }
})

test_that("quantile functions invert the distribution functions", {
  p <- c(0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)
  expect_equal(pgkw(qgkw(p, 2, 3, 1.5, 0.5, 2), 2, 3, 1.5, 0.5, 2), p,
               tolerance = 1e-8)
  expect_equal(pbkw(qbkw(p, 2, 3, 1.5, 0.5), 2, 3, 1.5, 0.5), p, tolerance = 1e-8)
  expect_equal(pkkw(qkkw(p, 2, 3, 0.5, 2), 2, 3, 0.5, 2), p, tolerance = 1e-8)
  expect_equal(pekw(qekw(p, 2, 3, 2), 2, 3, 2), p, tolerance = 1e-8)
  expect_equal(pmc(qmc(p, 1.5, 0.5, 2), 1.5, 0.5, 2), p, tolerance = 1e-8)
  expect_equal(pkw(qkw(p, 2, 3), 2, 3), p, tolerance = 1e-8)
  expect_equal(pbeta_(qbeta_(p, 2, 3), 2, 3), p, tolerance = 1e-8)
})

test_that("the package helpers do not collide with R's Rmath API", {
  # Rmath.h defines log1mexp/log1pexp with a different convention
  # (log(1 - exp(-x)) for x >= 0). Binding to those silently produced NaN and
  # zeroed dgkw(). This guards the fix at the level users can observe.
  expect_gt(dgkw(0.5, 2, 3, 1.5, 0.5, 2), 0)
  expect_true(all(dgkw(seq(0.05, 0.95, by = 0.05), 2, 3, 1.5, 0.5, 2) > 0))
  expect_true(all(is.finite(dgkw(seq(0.05, 0.95, by = 0.05), 2, 3, 1.5, 0.5, 2,
                                 log = TRUE))))
})
