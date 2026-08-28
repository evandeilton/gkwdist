# tests/testthat/test-deep-tail-precision.R
# Regression tests for the logarithmic bound constants in src/utils.h.
#
# LOG_DBL_MAX held log10(DBL_MAX) = 308.25 while being used as a natural-log
# threshold, so safe_exp() and safe_pow() returned +Inf for everything above
# exp(308.25), discarding 174 orders of magnitude of representable range.
#
# safe_log() scaled by LOG_DBL_MIN = log(DBL_MIN) while dividing by
# DBL_MIN_SAFE = 10 * DBL_MIN, so every result below 2.225e-307 was off by
# exactly log(10) = 2.302585 -- a finite, plausible, wrong number.
#
# Both regimes are far out in the tail, but they are reachable from the public
# API and the second one is silent, which is why they are pinned here.

# Reference computed entirely in log space, never through the package helpers.
log1mexp_ref <- function(u) ifelse(u > -log(2), log(-expm1(u)), log1p(-exp(u)))

ref_ldkw <- function(x, a, b) {
  log(a) + log(b) + (a - 1) * log(x) + (b - 1) * log1mexp_ref(a * log(x))
}

test_that("log-density is exact for subnormal data (safe_log scaling)", {
  # Values below DBL_MIN_SAFE = 2.2250738585072014e-307 take the scaled branch.
  for (x in c(2.2e-307, 1e-307, 1e-308, 1e-310, 1e-315, 5e-324)) {
    expect_equal(dkw(x, 2, 3, log = TRUE), ref_ldkw(x, 2, 3), tolerance = 1e-13)
    expect_equal(dkw(x, 0.5, 1.5, log = TRUE), ref_ldkw(x, 0.5, 1.5), tolerance = 1e-13)
    expect_equal(dbeta_(x, 2, 2, log = TRUE), stats::dbeta(x, 2, 3, log = TRUE),
                 tolerance = 1e-13)
  }
})

test_that("the log(10) offset is gone from the log-likelihood", {
  for (x in c(1e-307, 1e-308, 1e-315)) {
    data <- c(x, 0.3, 0.7)
    expect_equal(llkw(c(2, 3), data), -sum(ref_ldkw(data, 2, 3)), tolerance = 1e-13)
    expect_equal(llbeta(c(2, 2), data),
                 -sum(stats::dbeta(data, 2, 3, log = TRUE)), tolerance = 1e-13)
  }
})

test_that("large densities are finite instead of saturating at exp(308)", {
  # alpha < 1 with tiny x drives the density far above the old 7.5e133 ceiling
  # while staying well inside double range.
  cases <- list(
    list(x = 1e-268, a = 0.5,  b = 2),
    list(x = 1e-300, a = 0.5,  b = 2),
    list(x = 1e-280, a = 0.3,  b = 2),
    list(x = 1e-300, a = 0.1,  b = 2),
    list(x = 1e-300, a = 0.05, b = 3)
  )
  for (cs in cases) {
    got <- dkw(cs$x, cs$a, cs$b)
    ref <- exp(ref_ldkw(cs$x, cs$a, cs$b))
    expect_true(is.finite(got))
    expect_equal(got, ref, tolerance = 1e-12)
  }

  expect_true(is.finite(dbeta_(1e-300, 0.5, 0)))
  expect_equal(dbeta_(1e-300, 0.5, 0), stats::dbeta(1e-300, 0.5, 1), tolerance = 1e-12)
  expect_true(is.finite(dgkw(1e-300, 0.5, 2, 1, 0, 1)))
})

test_that("the nesting identity survives into the deep tail", {
  # dgkw and dkw are the same density here; before the fix one returned Inf
  # while the other did not.
  for (x in c(1e-200, 1e-280, 1e-300)) {
    expect_equal(dgkw(x, 0.5, 2, 1, 0, 1), dkw(x, 0.5, 2), tolerance = 1e-12)
  }

  # On the log scale the identity is asserted only while x^alpha stays a normal
  # double. dgkw() forms x^alpha in linear space before taking its logarithm,
  # and safe_pow() flushes anything below DBL_MIN to zero, so dgkw() returns
  # -Inf once alpha * log(x) < -708.4 while dkw(), which works from
  # alpha * log(x) directly, still returns the correct finite value. That is a
  # separate, pre-existing defect -- unchanged by the constants fixed here --
  # and belongs with the log-space rework of gkw.cpp.
  for (x in c(1e-100, 1e-140, 1e-150)) {
    expect_true(x^2 > .Machine$double.xmin)  # guard: x^alpha is a normal double
    expect_equal(dgkw(x, 2, 3, 1, 0, 1, log = TRUE), dkw(x, 2, 3, log = TRUE),
                 tolerance = 1e-13)
  }
})

test_that("ordinary values in the bulk are untouched", {
  x <- c(0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
  expect_equal(dkw(x, 2, 3), ref_ldkw(x, 2, 3) |> exp(), tolerance = 1e-12)
  expect_equal(dbeta_(x, 2, 2), stats::dbeta(x, 2, 3), tolerance = 1e-12)
  expect_true(all(is.finite(dgkw(x, 2, 3, 1.5, 2, 1.2))))
  expect_equal(integrate(function(t) dkw(t, 2, 3), 0, 1)$value, 1, tolerance = 1e-8)
})
