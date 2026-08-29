# tests/testthat/test-cdf-log-space.R
# Regression tests for the seven CDFs after the log-space rewrite.
#
# Each p*() formed 1 - x^alpha and 1 - (1 - x^alpha)^beta in linear space.
# Once x^alpha fell below 1.1e-16 the first rounded to exactly 1 and the second
# to exactly 0, so the CDF collapsed to 0 or 1. This was not a last-digit loss:
#
#   pekw(5.62e-09, 2, 5, 0.02)  returned 0     against a true 0.483
#   pekw(0.14, 20, 20, 0.1)     returned 0     against a true 0.0264
#   pkw(1e-09, 2, 5, log.p=T)   returned -Inf  against a true -39.84
#
# The second of those sits at x = 0.14, nowhere near a tail: a small lambda
# compresses the result toward 1 and brings the collapse into the body of the
# distribution.
#
# lower.tail and log.p were also applied after the fact, as 1 - p and log(p),
# rather than passed to R::pbeta, which loses the whole opposite tail.

l1m <- function(u) ifelse(u > -log(2), log(-expm1(u)), log1p(-exp(u)))

test_that("the CDFs no longer collapse to 0 or 1", {
  expect_equal(pekw(5.6234132519034910e-09, 2, 5, 0.02), 0.48303588962930716,
               tolerance = 1e-12)
  expect_equal(pekw(0.14, 20, 20, 0.1), 0.026445943814401841, tolerance = 1e-12)
  expect_equal(pkw(1e-09, 2, 5, log.p = TRUE), l1m(5 * l1m(2 * log(1e-09))),
               tolerance = 1e-12)
  expect_equal(pkw(1e-06, 3, 100), -expm1(100 * log1p(-(1e-06)^3)), tolerance = 1e-12)
  expect_equal(pgkw(1e-09, 2, 3, 1.5, 2, 0.8),
               stats::pbeta(exp(0.8 * l1m(3 * l1m(2 * log(1e-09)))), 1.5, 3),
               tolerance = 1e-10)
  expect_equal(pkkw(0.5, 1000, 1000, 100, 0.001), 1, tolerance = 1e-12)
})

test_that("the upper tail survives", {
  expect_equal(pkw(1 - 1e-06, 2, 3, lower.tail = FALSE),
               exp(3 * l1m(2 * log(1 - 1e-06))), tolerance = 1e-12)
  expect_equal(pmc(1 - 1e-06, 2, 3, 2.5, lower.tail = FALSE),
               stats::pbeta((1 - 1e-06)^2.5, 2, 4, lower.tail = FALSE), tolerance = 1e-8)
  expect_gt(pkw(1 - 1e-06, 2, 3, lower.tail = FALSE), 0)
  expect_gt(pekw(1 - 1e-08, 2, 3, 1.5, lower.tail = FALSE), 0)
})

test_that("log.p reaches where the linear scale underflows", {
  expect_equal(pbeta_(1e-200, 2, 3, log.p = TRUE),
               stats::pbeta(1e-200, 2, 4, log.p = TRUE), tolerance = 1e-12)
  expect_true(is.finite(pkw(1e-100, 2, 3, log.p = TRUE)))
  expect_true(is.finite(pekw(1e-50, 2, 3, 1.5, log.p = TRUE)))
  expect_true(is.finite(pmc(1e-100, 2, 3, 1.5, log.p = TRUE)))
})

test_that("every CDF stays a CDF", {
  x <- sort(c(10^-(1:60), 0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1 - 10^-(1:12)))
  x <- x[x > 0 & x < 1]
  cases <- list(
    list(f = pkw,    p = list(2, 3)),
    list(f = pekw,   p = list(2, 3, 1.5)),
    list(f = pkkw,   p = list(2, 3, 0.5, 1.5)),
    list(f = pbkw,   p = list(2, 3, 1.5, 2)),
    list(f = pgkw,   p = list(2, 3, 1.5, 2, 0.8)),
    list(f = pmc,    p = list(2, 3, 2.5)),
    list(f = pbeta_, p = list(2, 3))
  )
  for (cs in cases) {
    F <- do.call(cs$f, c(list(x), cs$p))
    S <- do.call(cs$f, c(list(x), cs$p, list(lower.tail = FALSE)))
    lF <- do.call(cs$f, c(list(x), cs$p, list(log.p = TRUE)))
    expect_true(all(F >= 0 & F <= 1))
    expect_true(all(diff(F) >= 0))            # monotone
    expect_equal(F + S, rep(1, length(F)), tolerance = 1e-14)
    expect_equal(exp(lF), F, tolerance = 1e-14)
  }
})

test_that("the CDF nesting identities hold", {
  x <- c(1e-12, 1e-06, 0.01, 0.3, 0.7, 0.99, 1 - 1e-09)
  expect_equal(pkw(x, 2, 3),          pgkw(x, 2, 3, 1, 0, 1),     tolerance = 1e-13)
  expect_equal(pekw(x, 2, 3, 1.5),    pgkw(x, 2, 3, 1, 0, 1.5),   tolerance = 1e-13)
  expect_equal(pbkw(x, 2, 3, 1.5, 2), pgkw(x, 2, 3, 1.5, 2, 1),   tolerance = 1e-13)
  expect_equal(pkkw(x, 2, 3, 2, 1.5), pgkw(x, 2, 3, 1, 2, 1.5),   tolerance = 1e-13)
  expect_equal(pmc(x, 2, 3, 2.5),     pgkw(x, 1, 1, 2, 3, 2.5),   tolerance = 1e-13)
  expect_equal(pbeta_(x, 2, 3),       stats::pbeta(x, 2, 4),      tolerance = 1e-13)
})

test_that("the CDF is the integral of the density", {
  for (q in c(0.1, 0.3, 0.5, 0.8)) {
    expect_equal(pkw(q, 2, 3),
                 integrate(function(t) dkw(t, 2, 3), 0, q, rel.tol = 1e-10)$value,
                 tolerance = 1e-8)
    expect_equal(pgkw(q, 2, 3, 1.5, 2, 0.8),
                 integrate(function(t) dgkw(t, 2, 3, 1.5, 2, 0.8), 0, q,
                           rel.tol = 1e-10)$value,
                 tolerance = 1e-8)
  }
})
