# tests/testthat/test-invalid-parameter-warning.R
# Regression tests for the invalid-parameter warning in the vectorised routines.
#
# dgkw, pgkw, qgkw, rgkw and the six nested r* called
#
#     Rcpp::warning("...: invalid parameters at index %d", i + 1)
#
# from inside the per-element loop. On 50,000 values with half the parameters
# invalid that cost 51x, and under options(warn = 2) each call longjmps out of
# the loop through C++ frames holding live Armadillo objects. R's own convention
# is one warning per call, not one per element.
#
# The message also has to match what the function actually returns, which is the
# defect this release fixed for q*(): the wrappers promised NaN while the C++
# saturated, so defensive code testing is.nan() saw nothing. base R pairs them --
# dbeta(0.5, -1, 1) is NaN and warns "NaNs produced", rbeta(2, -1, 1) is NA and
# warns "NAs produced".
#
# The invalid-parameter path is reachable from the exported API only through an
# Inf parameter: every R wrapper stop()s on <= 0, NA and NaN. These tests reach
# it that way, and through .Call for the per-element case.

test_that("one warning per call, not one per element", {
  x <- runif(2000, 0.01, 0.99)
  n <- 0
  withCallingHandlers(
    .Call("_gkwdist_dgkw", x, c(2, -1), 3, 1, 0, 1, FALSE, PACKAGE = "gkwdist"),
    warning = function(c) { n <<- n + 1; invokeRestart("muffleWarning") }
  )
  expect_equal(n, 1)

  n <- 0
  withCallingHandlers(
    .Call("_gkwdist_rkw", 2000L, c(2, -1), 3, PACKAGE = "gkwdist"),
    warning = function(c) { n <<- n + 1; invokeRestart("muffleWarning") }
  )
  expect_equal(n, 1)
})

test_that("valid parameters warn not at all", {
  x <- runif(500, 0.01, 0.99)
  expect_silent(dgkw(x, 2, 3, 1.5, 0.5, 1.2))
  expect_silent(pgkw(x, 2, 3, 1.5, 0.5, 1.2))
  expect_silent(dkw(x, 2, 3))
  set.seed(1)
  expect_silent(rkw(500, 2, 3))
})

test_that("the message names the value the routine actually returns", {
  # base R sets the convention: value NaN -> "NaNs produced",
  # value NA -> "NAs produced".
  grab <- function(expr) {
    m <- ""
    v <- withCallingHandlers(expr,
      warning = function(c) { m <<- conditionMessage(c); invokeRestart("muffleWarning") })
    list(v = v, m = m)
  }

  # p*, q* and r* fill NA_REAL, so they must say NAs
  r <- grab(.Call("_gkwdist_pgkw", c(.3, .4), c(2, Inf), 3, 1, 0, 1, TRUE, FALSE,
                  PACKAGE = "gkwdist"))
  expect_true(is.na(r$v[2]) && !is.nan(r$v[2]))
  expect_match(r$m, "NAs produced")

  r <- grab(.Call("_gkwdist_rkw", 2L, c(2, Inf), 3, PACKAGE = "gkwdist"))
  expect_true(is.na(r$v[2]) && !is.nan(r$v[2]))
  expect_match(r$m, "NAs produced")

  # dgkw leaves the fill value, 0, so it must NOT claim to have produced
  # a missing value of either kind
  r <- grab(.Call("_gkwdist_dgkw", c(.3, .4), c(2, Inf), 3, 1, 0, 1, FALSE,
                  PACKAGE = "gkwdist"))
  expect_identical(r$v[2], 0)
  expect_false(grepl("produced", r$m))
  expect_match(r$m, "invalid parameters")
})

test_that("the flag does not fire on paths that are not invalid parameters", {
  # NA data, out-of-support data and the closed boundary all have their own
  # handling and none of them is an invalid parameter
  expect_silent(dkw(c(NA_real_, NaN), 2, 3))
  expect_silent(dkw(c(-1, 2), 2, 3))
  expect_silent(dkw(c(0, 1), 2, 3))
  expect_silent(pkw(c(-Inf, Inf), 2, 3))
})

test_that("values on the invalid-parameter path are unchanged", {
  v <- suppressWarnings(
    .Call("_gkwdist_dgkw", c(.3, .4, .5, .6), c(2, -1), 3, 1, 0, 1, FALSE,
          PACKAGE = "gkwdist"))
  expect_identical(v[c(2, 4)], c(0, 0))
  expect_true(all(v[c(1, 3)] > 0))

  v <- suppressWarnings(
    .Call("_gkwdist_pgkw", c(.3, .4, .5, .6), c(2, -1), 3, 1, 0, 1, TRUE, FALSE,
          PACKAGE = "gkwdist"))
  expect_identical(v[c(2, 4)], c(NA_real_, NA_real_))
})
