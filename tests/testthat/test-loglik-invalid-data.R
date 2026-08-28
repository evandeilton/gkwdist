# tests/testthat/test-loglik-invalid-data.R
# Regression tests for the sign of ll*() on data outside the open support.
#
# ll*() returns the NEGATIVE log-likelihood, so an invalid point has to be +Inf:
# the value optim() moves away from. llgkw() returned -Inf, making bad data the
# global minimum of the objective, and it was the only one of the seven families
# with that sign.
#
# The damage was not in optim(), which refuses to start at either infinity, but
# in comparing likelihoods. On data holding a single zero -- ordinary in
# untransformed proportions -- the GKw family won every selection:
#
#   gkw  nll = -Inf        argmin = gkw
#   bkw  nll =  Inf        AIC    = -Inf
#   ...  nll =  Inf
#
# Against 1.1.5 the first block below fails 8 assertions (four invalid vectors,
# each checked for +Inf and for agreement with the other six families).

ll_all <- list(
  llgkw  = c(2, 3, 1.5, 2, 0.8),
  llbkw  = c(2, 3, 1.5, 2),
  llkkw  = c(2, 3, 1.5, 2),
  llekw  = c(2, 3, 1.5),
  llmc   = c(2, 3, 1.5),
  llkw   = c(2, 3),
  llbeta = c(2, 3)
)

test_that("every family returns +Inf for data outside the open support", {
  bad <- list(c(0.5, 0), c(0.5, 1), c(0.5, -0.1), c(0.5, 1.2), 0, 1)
  for (x in bad) {
    got <- vapply(names(ll_all),
                  function(f) suppressWarnings(do.call(f, list(ll_all[[f]], x))),
                  numeric(1))
    expect_true(all(got == Inf),
                info = paste("x =", paste(x, collapse = ", "),
                             "->", paste(names(got), got, collapse = " ")))
  }
})

test_that("a single invalid observation cannot win a family comparison", {
  set.seed(7)
  x <- c(rgkw(500, 2, 3, 1.5, 2, 0.8), 0)
  got <- vapply(names(ll_all),
                function(f) suppressWarnings(do.call(f, list(ll_all[[f]], x))),
                numeric(1))
  # No family may look better than any other on data none of them can fit.
  expect_true(all(got == Inf))
  expect_false(any(is.finite(got)))
  expect_false(any(got == -Inf))
})

test_that("invalid parameters still give +Inf, and valid data is untouched", {
  set.seed(11)
  g <- rgkw(2000, 2, 3, 1.5, 2, 0.8)

  # the parameter path was already correct and must stay that way
  expect_identical(llgkw(c(-1, 3, 1.5, 2, 0.8), g), Inf)
  expect_identical(llgkw(c(2, 0, 1.5, 2, 0.8), g), Inf)

  # valid data: finite, and still equal to the nested families
  expect_true(is.finite(llgkw(c(2, 3, 1.5, 2, 0.8), g)))
  expect_equal(llgkw(c(2, 3, 1, 0, 1), g), llkw(c(2, 3), g), tolerance = 1e-12)
  expect_equal(llgkw(c(2, 3, 1.5, 2, 1), g), llbkw(c(2, 3, 1.5, 2), g),
               tolerance = 1e-12)
  expect_equal(llgkw(c(2, 3, 1, 0, 1.5), g), llekw(c(2, 3, 1.5), g),
               tolerance = 1e-12)
})
