# tests/testthat/test-zero-length-input.R
# Regression tests for zero-length arguments.
#
# Before these guards existed, a zero-length argument reached the
# `i % vec.n_elem` recycling in C++ with a zero divisor. Integer division by
# zero is undefined behaviour and raised SIGFPE, killing the R process: not a
# catchable error, no message, no traceback. A filtered vector that happens to
# be empty (`x[x > 1]`) was enough to trigger it in all 28 d/p/q/r functions.
#
# The expected behaviour follows R's own recycling convention:
#   dbeta(numeric(0), 1, 1)   -> numeric(0)
#   dbeta(0.5, numeric(0), 1) -> numeric(0)
#   rbeta(3, numeric(0), 1)   -> NA NA NA, with a warning

# Default arguments are valid for every family, so a single positional
# argument exercises the "x is empty, parameters are not" path.
dpq_funs <- c(
  "dgkw", "pgkw", "qgkw",
  "dbkw", "pbkw", "qbkw",
  "dkkw", "pkkw", "qkkw",
  "dekw", "pekw", "qekw",
  "dmc", "pmc", "qmc",
  "dkw", "pkw", "qkw",
  "dbeta_", "pbeta_", "qbeta_"
)

r_funs <- c("rgkw", "rbkw", "rkkw", "rekw", "rmc", "rkw", "rbeta_")

test_that("d/p/q return numeric(0) when the first argument is empty", {
  for (fn in dpq_funs) {
    result <- do.call(fn, list(numeric(0)))
    expect_type(result, "double")
    expect_length(result, 0L)
  }
})

test_that("d/p/q return numeric(0) when a parameter is empty", {
  expect_length(dkw(0.5, numeric(0), 2), 0L)
  expect_length(dkw(0.5, 2, numeric(0)), 0L)
  expect_length(pkw(0.5, numeric(0), 2), 0L)
  expect_length(qkw(0.5, 2, numeric(0)), 0L)
  expect_length(dgkw(0.5, 2, 3, numeric(0), 0, 1), 0L)
  expect_length(dgkw(0.5, 2, 3, 1, 0, numeric(0)), 0L)
  expect_length(pbkw(0.5, 2, 3, 1, numeric(0)), 0L)
  expect_length(dkkw(0.5, 2, 3, numeric(0), 1), 0L)
  expect_length(dekw(0.5, 2, numeric(0), 1), 0L)
  expect_length(dmc(0.5, numeric(0), 0, 1), 0L)
  expect_length(dbeta_(0.5, numeric(0), 0), 0L)
})

test_that("d/p/q match the zero-length convention of stats::dbeta", {
  expect_identical(dbeta_(numeric(0), 2, 2), stats::dbeta(numeric(0), 2, 3))
  expect_identical(pbeta_(numeric(0), 2, 2), stats::pbeta(numeric(0), 2, 3))
  expect_identical(qbeta_(numeric(0), 2, 2), stats::qbeta(numeric(0), 2, 3))
  expect_identical(dbeta_(0.5, numeric(0), 2), stats::dbeta(0.5, numeric(0), 3))
})

test_that("an empty subset does not crash the session", {
  x <- c(0.2, 0.5)
  empty <- x[x > 1]
  expect_length(empty, 0L)
  expect_length(dkw(empty, 2, 3), 0L)
  expect_length(pkw(empty, 2, 3), 0L)
  expect_length(qkw(empty, 2, 3), 0L)
  expect_length(dgkw(empty, 2, 3, 1, 0, 1), 0L)
})

test_that("r* return NAs with a warning when a parameter is empty", {
  for (fn in r_funs) {
    expect_warning(result <- do.call(fn, list(3L, numeric(0))), "NAs produced")
    expect_type(result, "double")
    expect_length(result, 3L)
    expect_true(all(is.na(result)))
  }
})

test_that("zero-length guards leave ordinary results untouched", {
  x <- c(0.1, 0.25, 0.5, 0.75, 0.9)

  # A representative value per family, and the recycling path that shares the
  # `i % vec.n_elem` arithmetic the guards protect.
  expect_length(dkw(x, 2, 3), length(x))
  expect_length(dkw(x, c(1, 2), 3), length(x))
  expect_length(dkw(0.5, c(1, 2, 3), 3), 3L)
  expect_length(dgkw(x, 2, 3, 1.5, 2, 1.2), length(x))
  expect_length(dgkw(0.5, c(1, 2), c(1, 2, 3), 1, 0, 1), 3L)

  expect_true(all(is.finite(dkw(x, 2, 3))))
  expect_true(all(is.finite(pkw(x, 2, 3))))

  # Nesting identity still holds: Kw is GKw with gamma=1, delta=0, lambda=1.
  expect_equal(dkw(x, 2, 3), dgkw(x, 2, 3, 1, 0, 1))
  expect_equal(pkw(x, 2, 3), pgkw(x, 2, 3, 1, 0, 1))
})
