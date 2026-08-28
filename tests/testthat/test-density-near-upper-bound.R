# tests/testthat/test-density-near-upper-bound.R
# Regression tests for dgkw() as x approaches 1.
#
# dgkw() used to form x^alpha in linear space and bail out with a density of
# zero whenever x^alpha >= 1 - sqrt(.Machine$double.eps). The guard existed
# because log(x^alpha) loses its significant digits there -- doubles are spaced
# 2.2e-16 apart near 1, so the relative error reaches 4e-6 by 1 - x = 1e-12 --
# but returning zero is a far worse answer than an imprecise one. For
# GKw(0.1, 0.1, 10, 0.1, 0.1) the discarded band held 13% of the probability
# mass, and it broke the nesting identity: dgkw(x, 1, 0.1, 1, 0, 1) returned 0
# where dkw(x, 1, 0.1), the same density, returned 1.26e7.
#
# Taking log(x^alpha) as alpha * log(x) removes both the imprecision and the
# need for the guard; gkw_log1mexp() already handles the resulting regime.

log1mexp_ref <- function(u) ifelse(u > -log(2), log(-expm1(u)), log1p(-exp(u)))

ref_ldgkw <- function(x, a, b, g, d, l) {
  lx <- log(x)
  lv <- log1mexp_ref(a * lx)
  lw <- log1mexp_ref(b * lv)
  lz <- log1mexp_ref(l * lw)
  log(l) + log(a) + log(b) - lbeta(g, d + 1) +
    (a - 1) * lx + (b - 1) * lv + (g * l - 1) * lw + d * lz
}

# The band the old guard rejected: x^alpha within sqrt(eps) of 1.
near_one <- 1 - c(1e-8, 1.3e-8, 1e-9, 1e-10, 1e-11, 1e-13, 1e-15)

test_that("dgkw is non-zero throughout the band the guard used to reject", {
  configs <- list(
    c(1, 0.1, 1, 0, 1),
    c(1, 0.2, 1, 0, 1),
    c(0.5, 0.5, 1, 0, 1),
    c(2, 3, 1.5, 2, 1.2),
    c(0.1, 0.1, 10, 0.1, 0.1)
  )
  for (p in configs) {
    got <- dgkw(near_one, p[1], p[2], p[3], p[4], p[5])
    ref <- exp(ref_ldgkw(near_one, p[1], p[2], p[3], p[4], p[5]))
    expect_true(all(got > 0))
    expect_equal(got, ref, tolerance = 1e-10)
  }
})

test_that("the nesting identity holds as x approaches 1", {
  # dgkw(x, a, b, 1, 0, 1) and dkw(x, a, b) are the same density. Before the
  # fix the first returned 0 in this band while the second did not.
  for (ab in list(c(2, 3), c(0.5, 0.3), c(1, 0.1), c(40, 25))) {
    # dgkw() still evaluates log(1 - w^lambda) and bails out when it is not
    # finite, even for gamma*lambda = 1 and delta = 0 where that term drops out
    # of the density entirely. Once beta * log(1 - x^alpha) falls below
    # log(DBL_MIN) the inner exp() underflows, w rounds to 1, and the guard
    # fires. That is a pre-existing defect of the same family as the
    # observation-dropping guards in hsgkw(), unchanged here, and it belongs
    # with the log-space rework. Assert the identity where the chain survives.
    lv <- log1mexp_ref(ab[1] * log(near_one))
    ok <- ab[2] * lv > -745
    expect_true(any(ok))

    expect_equal(dgkw(near_one, ab[1], ab[2], 1, 0, 1), dkw(near_one, ab[1], ab[2]),
                 tolerance = 1e-12)
    expect_equal(dgkw(near_one[ok], ab[1], ab[2], 1, 0, 1, log = TRUE),
                 dkw(near_one[ok], ab[1], ab[2], log = TRUE), tolerance = 1e-13)
  }

  # Beta and McDonald reach the same band through alpha = beta = 1.
  expect_equal(dgkw(near_one, 1, 1, 2, 2, 1), dbeta_(near_one, 2, 2), tolerance = 1e-12)
  expect_equal(dgkw(near_one, 1, 1, 2, 2, 1.5), dmc(near_one, 2, 2, 1.5), tolerance = 1e-12)
})

test_that("recovered mass shows up in the integral", {
  # This parameterisation lost 13% of its mass to the guard.
  mass <- integrate(function(t) dgkw(t, 0.1, 0.1, 10, 0.1, 0.1), 0, 1,
                    subdivisions = 5000)$value
  expect_equal(mass, 1, tolerance = 1e-3)

  for (p in list(c(1, 0.1, 1, 0, 1), c(0.5, 0.5, 1, 0, 1), c(2, 3, 1.5, 2, 1.2))) {
    m <- integrate(function(t) dgkw(t, p[1], p[2], p[3], p[4], p[5]), 0, 1,
                   subdivisions = 5000)$value
    expect_equal(m, 1, tolerance = 1e-6)
  }
})

test_that("the density stays monotone into the upper tail", {
  # For beta < 1 the density diverges as x -> 1; the guard produced a cliff to
  # zero instead of a rising tail.
  x <- 1 - 10^-(6:14)
  d <- dgkw(x, 1, 0.1, 1, 0, 1)
  expect_true(all(is.finite(d)))
  expect_true(all(diff(d) > 0))
})

test_that("the bulk of the distribution is unchanged", {
  x <- c(0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
  expect_equal(dgkw(x, 2, 3, 1.5, 2, 1.2),
               exp(ref_ldgkw(x, 2, 3, 1.5, 2, 1.2)), tolerance = 1e-12)
  expect_equal(dgkw(x, 2, 3, 1, 0, 1), dkw(x, 2, 3), tolerance = 1e-13)
  expect_true(all(is.finite(dgkw(x, 0.3, 0.2, 0.4, 0, 0.25))))
})
