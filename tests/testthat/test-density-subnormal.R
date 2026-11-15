# tests/testthat/test-density-subnormal.R
# Densities in the subnormal range [4.94e-324, 2.2e-308) and the size guard
# of rgkw().
#
# Every family below reduces to Beta(100, 1) at the chosen parameters, whose
# density 100 * x^99 equals 10^e at x = 10^((e - 2) / 99). The reference is
# stats::dbeta().

# One density per family, all equal to dbeta(x, 100, 1)
beta100 <- list(
  kw    = function(x, log = FALSE) dkw(x, 100, 1, log = log),
  beta_ = function(x, log = FALSE) dbeta_(x, 100, 0, log = log),
  mc    = function(x, log = FALSE) dmc(x, 100, 0, 1, log = log),
  gkw   = function(x, log = FALSE) dgkw(x, 100, 1, 1, 0, 1, log = log),
  ekw   = function(x, log = FALSE) dekw(x, 100, 1, 1, log = log),
  bkw   = function(x, log = FALSE) dbkw(x, 100, 1, 1, 0, log = log),
  kkw   = function(x, log = FALSE) dkkw(x, 100, 1, 0, 1, log = log)
)

test_that("densities below DBL_MIN are returned as subnormals", {
  # One ulp is 4.94e-324: about 5e-9 relative at 1e-315 and 5e-4 at 1e-320
  cases <- list(c(e = -312, tol = 1e-8), c(e = -315, tol = 1e-7),
                c(e = -320, tol = 1e-2))
  for (cs in cases) {
    x <- 10^((cs[["e"]] - 2) / 99)
    ref <- stats::dbeta(x, 100, 1)
    expect_gt(ref, 0)
    for (fam in names(beta100)) {
      val <- beta100[[fam]](x)
      expect_gt(val, 0, label = paste(fam, "at 1e", cs[["e"]]))
      expect_equal(val, ref, tolerance = cs[["tol"]],
                   label = paste(fam, "at 1e", cs[["e"]]))
    }
  }
})

test_that("log-densities agree with dbeta() across the subnormal range", {
  for (e in c(-312, -315, -320)) {
    x <- 10^((e - 2) / 99)
    ref <- stats::dbeta(x, 100, 1, log = TRUE)
    for (fam in names(beta100)) {
      expect_equal(beta100[[fam]](x, log = TRUE), ref, tolerance = 1e-12,
                   label = paste(fam, "log at 1e", e))
    }
  }
})

test_that("a density below the smallest subnormal is exactly 0", {
  # True density 1e-325 rounds to 0 in double precision
  x <- 10^(-327 / 99)
  expect_identical(stats::dbeta(x, 100, 1), 0)
  for (fam in names(beta100)) {
    expect_identical(beta100[[fam]](x), 0, label = fam)
  }
})

test_that("rgkw() rejects a size that does not fit an integer", {
  # as.integer() maps both to NA_integer_, which must stop before allocating
  expect_error(suppressWarnings(rgkw(Inf)), "non-negative")
  expect_error(suppressWarnings(rgkw(2^31)), "non-negative")
  expect_length(rgkw(0), 0L)
})
