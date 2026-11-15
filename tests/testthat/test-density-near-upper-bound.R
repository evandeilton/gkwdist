# tests/testthat/test-density-near-upper-bound.R
# dgkw() as x approaches 1.
#
# log(x^alpha) formed from a linear-space x^alpha loses its significant digits
# once x^alpha >= 1 - sqrt(.Machine$double.eps): doubles are spaced 2.2e-16
# apart near 1, so the relative error reaches 4e-6 by 1 - x = 1e-12. The
# density must stay accurate through that band. For GKw(0.1, 0.1, 10, 0.1, 0.1)
# the band holds 13% of the probability mass, and the nesting identity needs
# it: dgkw(x, 1, 0.1, 1, 0, 1) and dkw(x, 1, 0.1) are the same density, 1.26e7
# at x = 1 - 1e-9.
#
# Taking log(x^alpha) as alpha * log(x) keeps full precision, and
# gkw_log1mexp() handles the resulting regime.

log1mexp_ref <- function(u) ifelse(u > -log(2), log(-expm1(u)), log1p(-exp(u)))

ref_ldgkw <- function(x, a, b, g, d, l) {
  lx <- log(x)
  lv <- log1mexp_ref(a * lx)
  lw <- log1mexp_ref(b * lv)
  lz <- log1mexp_ref(l * lw)
  log(l) + log(a) + log(b) - lbeta(g, d + 1) +
    (a - 1) * lx + (b - 1) * lv + (g * l - 1) * lw + d * lz
}

# The band where x^alpha is within sqrt(eps) of 1.
near_one <- 1 - c(1e-8, 1.3e-8, 1e-9, 1e-10, 1e-11, 1e-13, 1e-15)

test_that("dgkw is non-zero where x^alpha is within sqrt(eps) of 1", {
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
  # dgkw(x, a, b, 1, 0, 1) and dkw(x, a, b) are the same density.
  for (ab in list(c(2, 3), c(0.5, 0.3), c(1, 0.1), c(40, 25))) {
    # Restricted to beta * log(1 - x^alpha) > -745, where (1 - x^alpha)^beta
    # is representable.
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

test_that("the density integrates to one including the band near 1", {
  # This parameterization holds 13% of its mass in the band.
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
  # For beta < 1 the density diverges as x -> 1: a rising tail, with no cliff
  # to zero.
  x <- 1 - 10^-(6:14)
  d <- dgkw(x, 1, 0.1, 1, 0, 1)
  expect_true(all(is.finite(d)))
  expect_true(all(diff(d) > 0))
})

test_that("the bulk of the distribution matches the reference", {
  x <- c(0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
  expect_equal(dgkw(x, 2, 3, 1.5, 2, 1.2),
               exp(ref_ldgkw(x, 2, 3, 1.5, 2, 1.2)), tolerance = 1e-12)
  expect_equal(dgkw(x, 2, 3, 1, 0, 1), dkw(x, 2, 3), tolerance = 1e-13)
  expect_true(all(is.finite(dgkw(x, 0.3, 0.2, 0.4, 0, 0.25))))
})
