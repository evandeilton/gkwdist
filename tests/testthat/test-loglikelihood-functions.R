# tests/testthat/test-loglikelihood-functions.R
# Test log-likelihood functions for all families
#
# Every ll*() in this package returns the NEGATIVE log-likelihood, which is what
# optim() and nlminb() minimise. Its defining relationship is therefore
#
#     ll*(par, data) == -sum(d*(data, <par>, log = TRUE))
#
# and that is what these tests assert, together with type, length and
# finiteness. They deliberately do NOT assert `result < 0`: the sign of the NLL
# carries no information about correctness, because the log-density of a
# distribution on (0, 1) may be positive or negative. For a uniform sample,
#
#     llkw(c(1, 1), data) =  0        (the uniform: log-density is exactly 0)
#     llkw(c(2, 2), data) = +2.0976   (a positive NLL)
#
# so `result < 0` held only by accident of the parameters chosen here. The last
# test in this file pins those two counterexamples.

test_that("llgkw works correctly", {
  set.seed(123)
  data <- rgkw(50, alpha = 2, beta = 3, gamma = 1.5, delta = 2, lambda = 1.2)
  par <- c(2, 3, 1.5, 2, 1.2)

  result <- llgkw(par, data)

  expect_type(result, "double")
  expect_length(result, 1)
  expect_true(is.finite(result))
  expect_equal(
    result,
    -sum(dgkw(data,
      alpha = 2, beta = 3, gamma = 1.5, delta = 2, lambda = 1.2,
      log = TRUE
    ))
  )
})

test_that("llbkw works correctly", {
  set.seed(456)
  data <- rbkw(50, alpha = 2, beta = 3, gamma = 1.5, delta = 2)
  par <- c(2, 3, 1.5, 2)

  result <- llbkw(par, data)

  expect_type(result, "double")
  expect_length(result, 1)
  expect_true(is.finite(result))
  expect_equal(
    result,
    -sum(dbkw(data, alpha = 2, beta = 3, gamma = 1.5, delta = 2, log = TRUE))
  )
})

test_that("llkkw works correctly", {
  set.seed(789)
  data <- rkkw(50, alpha = 2, beta = 3, delta = 2, lambda = 1.2)
  par <- c(2, 3, 2, 1.2)

  result <- llkkw(par, data)

  expect_type(result, "double")
  expect_length(result, 1)
  expect_true(is.finite(result))
  expect_equal(
    result,
    -sum(dkkw(data, alpha = 2, beta = 3, delta = 2, lambda = 1.2, log = TRUE))
  )
})

test_that("llekw works correctly", {
  set.seed(101)
  data <- rekw(50, alpha = 2, beta = 3, lambda = 1.5)
  par <- c(2, 3, 1.5)

  result <- llekw(par, data)

  expect_type(result, "double")
  expect_length(result, 1)
  expect_true(is.finite(result))
  expect_equal(
    result,
    -sum(dekw(data, alpha = 2, beta = 3, lambda = 1.5, log = TRUE))
  )
})

test_that("llmc works correctly", {
  set.seed(202)
  data <- rmc(50, gamma = 2, delta = 3, lambda = 1.2)
  par <- c(2, 3, 1.2)

  result <- llmc(par, data)

  expect_type(result, "double")
  expect_length(result, 1)
  expect_true(is.finite(result))
  expect_equal(
    result,
    -sum(dmc(data, gamma = 2, delta = 3, lambda = 1.2, log = TRUE))
  )
})

test_that("llkw works correctly", {
  set.seed(303)
  data <- rkw(50, alpha = 2, beta = 3)
  par <- c(2, 3)

  result <- llkw(par, data)

  expect_type(result, "double")
  expect_length(result, 1)
  expect_true(is.finite(result))
  expect_equal(
    result,
    -sum(dkw(data, alpha = 2, beta = 3, log = TRUE))
  )
})

test_that("llbeta works correctly", {
  set.seed(404)
  data <- rbeta_(50, gamma = 2, delta = 3)
  par <- c(2, 3)

  result <- llbeta(par, data)

  expect_type(result, "double")
  expect_length(result, 1)
  expect_true(is.finite(result))
  expect_equal(
    result,
    -sum(dbeta_(data, gamma = 2, delta = 3, log = TRUE))
  )
})

test_that("the NLL sign is not a property of ll*()", {
  # Guards the assertion that was replaced here. `result < 0` is false for a
  # uniform sample scored at the uniform parameters, and false again at a
  # perfectly ordinary interior point, so it never was a property of ll*().
  set.seed(303)
  data <- rkw(50, alpha = 1, beta = 1)

  expect_equal(llkw(c(1, 1), data), 0)
  expect_gt(llkw(c(2, 2), data), 0)
})
