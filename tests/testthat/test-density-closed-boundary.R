# tests/testthat/test-density-closed-boundary.R
# Regression tests for d*() at x = 0 and x = 1.
#
# The GKw support is the open interval, but base R's density functions return
# the LIMIT at the closed boundary rather than 0 -- dbeta(0, 0.5, 1) is Inf and
# dbeta(1, 2, 1) is 2 -- and any code that plots a density across [0,1] depends
# on it. Every family here returned 0 at both ends, so curves fell to zero
# exactly where they should have diverged.
#
# Substituting the first-order forms the log chain already uses,
#
#   x -> 0:  log_v -> 0,  log_w -> log(beta) + alpha*log(x),  log_z -> 0
#   x -> 1:  log_x -> 0,  log_w -> 0,  log_z -> log(lambda) + beta*log_v
#
# the log-density collapses to a constant plus one power of the vanishing
# quantity, and the exponent alone decides:
#
#   at 0:  alpha*gamma*lambda - 1      at 1:  beta*(delta + 1) - 1
#
# positive gives 0, negative gives Inf, zero leaves the constant. The Beta
# parameterisation is the one case where an external arbiter exists, so the
# whole rule is checked against stats::dbeta there, and against the nesting
# identities everywhere else.
#
# Against 1.1.5 this file fails 34 assertions.

test_that("the Beta family reproduces stats::dbeta at both boundaries", {
  # gamma = shape1, delta = shape2 - 1
  cases <- list(c(0, 0.5, 1), c(0, 1, 1), c(0, 1, 3), c(0, 2, 1), c(0, 3, 2),
                c(1, 2, 1), c(1, 1, 1), c(1, 2, 3), c(1, 0.5, 1), c(1, 5, 1))
  for (z in cases) {
    x <- z[1]; s1 <- z[2]; s2 <- z[3]
    expect_equal(dbeta_(x, s1, s2 - 1), stats::dbeta(x, s1, s2),
                 info = sprintf("dbeta(%g, %g, %g)", x, s1, s2))
  }
})

test_that("the audit's two cases now give the limit", {
  expect_identical(dkw(0, 0.5, 1), Inf)      # density diverges
  expect_equal(dbeta_(1, 2, 0), 2)           # finite, equals stats::dbeta(1, 2, 1)
  expect_identical(dkw(1, 2, 0.5), Inf)
  expect_equal(dkw(0, 1, 2), 2)
  expect_equal(dkw(1, 2, 1), 2)
})

test_that("every family agrees with GKw at both boundaries", {
  for (x in c(0, 1)) {
    expect_equal(dkw(x, 0.5, 2),         dgkw(x, 0.5, 2, 1, 0, 1),      info = x)
    expect_equal(dbkw(x, 0.5, 2, 1.5, 2), dgkw(x, 0.5, 2, 1.5, 2, 1),   info = x)
    expect_equal(dkkw(x, 0.5, 2, 2, 1.5), dgkw(x, 0.5, 2, 1, 2, 1.5),   info = x)
    expect_equal(dekw(x, 0.5, 2, 1.5),    dgkw(x, 0.5, 2, 1, 0, 1.5),   info = x)
    expect_equal(dmc(x, 1.5, 2, 0.8),     dgkw(x, 1, 1, 1.5, 2, 0.8),   info = x)
    expect_equal(dbeta_(x, 1.5, 2),       dgkw(x, 1, 1, 1.5, 2, 1),     info = x)
  }
})

test_that("the boundary value is the limit of the interior", {
  # approach each boundary and check the density converges to what x = 0 and
  # x = 1 report, in all three regimes
  for (p in list(c(0.5, 2), c(1, 2), c(3, 2))) {      # alpha < 1, = 1, > 1
    v0 <- dkw(0, p[1], p[2])
    near <- dkw(10^-(8:12), p[1], p[2])
    if (is.finite(v0) && v0 > 0) {
      expect_equal(near[length(near)], v0, tolerance = 1e-6, info = paste(p, collapse = ","))
    } else if (is.infinite(v0)) {
      expect_true(all(diff(near) > 0), info = paste(p, collapse = ","))  # growing
    } else {
      expect_true(all(near < 1e-6), info = paste(p, collapse = ","))     # decaying to 0
    }
  }
})

test_that("outside the closed interval the density is still zero", {
  for (x in c(-1, -1e-300, 1 + 1e-12, 2, 1e6)) {
    expect_equal(dkw(x, 2, 3), 0, info = x)
    expect_equal(dgkw(x, 2, 3, 1.5, 2, 0.8), 0, info = x)
  }
  expect_equal(stats::dbeta(-1, 2, 3), 0)      # the reference agrees
  expect_equal(stats::dbeta(2, 2, 3), 0)
})

test_that("log = TRUE gives the log of the same limit", {
  expect_identical(dkw(0, 0.5, 1, log = TRUE), Inf)
  expect_identical(dkw(0, 2, 3, log = TRUE), -Inf)
  expect_equal(dkw(1, 2, 1, log = TRUE), log(2))
  expect_equal(dbeta_(1, 2, 0, log = TRUE), log(2))
  expect_equal(dbeta_(0, 1, 2, log = TRUE), stats::dbeta(0, 1, 3, log = TRUE))
})

test_that("the interior is untouched", {
  x <- c(1e-300, 1e-10, 0.01, 0.25, 0.5, 0.75, 0.99, 1 - 1e-12)
  expect_true(all(is.finite(dkw(x, 2, 3))))
  expect_equal(dkw(0.5, 2, 3), 1.6875)
  expect_equal(dgkw(0.5, 2, 3, 1.5, 0.5, 2), 2.343797, tolerance = 1e-6)
})
