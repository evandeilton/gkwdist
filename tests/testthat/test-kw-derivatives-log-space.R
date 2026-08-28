# tests/testthat/test-kw-derivatives-log-space.R
# Regression tests for grkw() and hskw() after the log-space rewrite.
#
# kw.cpp was the last family file whose gradient and Hessian were not in log
# space. It formed v = 1 - x^alpha in linear arithmetic and then applied
#
#   v = arma::clamp(v, eps, 1 - eps)     eps = 2.22e-14
#
# which froze log(v) at -31.4384832 for every observation near 1, independent of
# the data. The frozen value +30.938483203129 in the beta component is the
# signature.
#
#   par = (0.5, 2), x = 1 - 1e-14
#     grkw            -2.44999999999999   30.938483203129
#     numDeriv::grad  -3.99999999995549   32.430138079912
#     grekw(lambda=1) -3.99999999999997   32.430138079907   <- same distribution
#
# The damage did not need pathological data: c(1-1e-9, 1-1e-11, 0.5) already
# diverged by 3.4e-05, and hskw() was off by 47% in H[alpha,alpha] and 78% in
# H[alpha,beta].
#
# grkw() is now grekw() with lambda fixed at 1, so the two agree bit-for-bit.
# Against 1.1.5 the nesting block below fails 26 assertions.

PARS <- list(c(0.5, 2), c(2, 3), c(0.2, 2), c(5, 0.5), c(20, 20),
             c(0.1, 0.2), c(3, 100), c(1, 1))
DATA <- list(
  easy      = c(0.1, 0.3, 0.5, 0.7, 0.9),
  near_one  = c(1 - 1e-9, 1 - 1e-11, 0.5),
  at_one    = 1 - 1e-14,
  extreme   = c(1 - 1e-14, 1 - 1e-15, 1 - 1e-16),
  near_zero = c(1e-9, 1e-12, 0.5),
  mixed     = c(1e-14, 0.5, 1 - 1e-14)
)

test_that("grkw and hskw equal the EKw path with lambda = 1", {
  # ekw.cpp has been in log space since before this change, so it is an
  # independent implementation of the same quantity.
  for (nm in names(DATA)) {
    for (p in PARS) {
      d <- DATA[[nm]]
      expect_equal(grkw(p, d), grekw(c(p, 1), d)[1:2],
                   tolerance = 1e-12,
                   info = paste(nm, paste(p, collapse = ",")))
      expect_equal(hskw(p, d), hsekw(c(p, 1), d)[1:2, 1:2],
                   tolerance = 1e-10,
                   info = paste(nm, paste(p, collapse = ",")))
    }
  }
})

test_that("the frozen clamp value is gone", {
  g <- grkw(c(0.5, 2), 1 - 1e-14)
  expect_false(isTRUE(all.equal(g[2], 30.938483203129, tolerance = 1e-9)))
  expect_equal(g[1], -4, tolerance = 1e-9)
  expect_equal(g[2], 32.430138079912, tolerance = 1e-9)
})

test_that("grkw matches a numerical gradient where numDeriv is usable", {
  skip_if_not_installed("numDeriv")
  for (nm in c("easy", "near_zero", "near_one")) {
    for (p in list(c(0.5, 2), c(2, 3), c(20, 20))) {
      d <- DATA[[nm]]
      expect_equal(grkw(p, d),
                   numDeriv::grad(function(q) llkw(q, d), p),
                   tolerance = 1e-6,
                   info = paste(nm, paste(p, collapse = ",")))
    }
  }
})

test_that("ordinary data is unaffected and the Hessian stays symmetric", {
  set.seed(4)
  x <- rkw(200, 2, 3)
  for (p in PARS) {
    g <- grkw(p, x)
    h <- hskw(p, x)
    expect_true(all(is.finite(g)))
    expect_true(all(is.finite(h)))
    expect_identical(h[1, 2], h[2, 1])
  }
  # H[beta,beta] is -n/beta^2 negated, with no data dependence at all
  expect_equal(hskw(c(2, 3), x)[2, 2], 200 / 9, tolerance = 1e-12)
})
