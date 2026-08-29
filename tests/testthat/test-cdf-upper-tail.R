# tests/testthat/test-cdf-upper-tail.R
# Regression tests for the upper tail of pgkw() and pbkw().
#
# The CDF batch moved lower.tail and log.p onto R::pbeta rather than applying
# them afterwards, which fixed the lower tail. The upper tail kept a second
# defect, in the ARGUMENT rather than the tail flag.
#
# pgkw builds y = [1 - (1 - x^alpha)^beta]^lambda and calls I_y(gamma, delta+1).
# As x -> 1 the exponent lambda*log_w passes below 1.1e-16, exp() returns
# exactly 1, and R::pbeta(1, ., ., lower = FALSE) returns exactly 0. pbkw does
# the same through -expm1 of an exponent running off to -Inf.
#
# Against a 300-digit incomplete beta, pgkw(x, 2, 3, 1.5, 2, 0.8, lower = FALSE):
#
#   1-x     returned        exact           rel err
#   1e-4    5.731692e-34    5.731820e-34    2.2e-05
#   1e-5    5.840671e-43    5.734142e-43    1.9e-02
#   1e-6    0               5.734374e-52    1.00      <- collapse
#   1e-16   0               1.469535e-141   1.00
#
# The true tail stays representable sixteen decades past the point where the
# routine gave up. I_y(a,b) = 1 - I_{1-y}(b,a) is exact and 1 - y comes from
# -expm1 of the same exponent, so reflecting above y = 1/2 sends the small
# quantity into pbeta. Below the crossover nothing changes.
#
# Against 1.1.5 this file fails 30 assertions.

test_that("the upper tail does not collapse to zero", {
  x <- 1 - 10^-(2:16)
  for (z in list(list(pgkw(x, 2, 3, 1.5, 2, 0.8, lower.tail = FALSE), "pgkw"),
                 list(pgkw(x, 0.5, 0.5, 0.5, 0.5, 0.5, lower.tail = FALSE), "pgkw2"),
                 list(pbkw(x, 2, 3, 1.5, 2, lower.tail = FALSE), "pbkw"),
                 list(pbkw(x, 0.5, 0.5, 0.5, 0, lower.tail = FALSE), "pbkw2"))) {
    expect_false(any(z[[1]] == 0), info = z[[2]])
    expect_true(all(z[[1]] > 0), info = z[[2]])
    expect_true(all(diff(z[[1]]) <= 0), info = paste(z[[2]], "monotone"))
  }
})

test_that("the upper tail matches the value a 300-digit reference gives", {
  x <- 1 - 10^-c(4, 5, 6, 9, 12, 16)
  ref <- c(5.731820e-34, 5.734142e-43, 5.734374e-52,
           5.734399e-79, 5.733258e-106, 1.469535e-141)
  got <- pgkw(x, 2, 3, 1.5, 2, 0.8, lower.tail = FALSE)
  expect_equal(got, ref, tolerance = 1e-6)
})

test_that("the reduced families agree in the upper tail", {
  # pmc and pbeta_ live in other translation units, and pmc was corrected
  # independently of pgkw. Agreement across the nesting identities is therefore
  # a reference, not a restatement.
  x <- sort(unique(c(1 - 10^-(1:16), seq(0.5, 0.999, by = 0.01), 10^-(1:12))))
  s <- function(v) v[v > 0]
  g1 <- pgkw(x, 1, 1, 1.5, 2, 0.8, lower.tail = FALSE)
  expect_equal(g1, pmc(x, 1.5, 2, 0.8, lower.tail = FALSE), tolerance = 1e-12)
  g2 <- pgkw(x, 1, 1, 1.5, 2, 1, lower.tail = FALSE)
  expect_equal(g2, pbeta_(x, 1.5, 2, lower.tail = FALSE), tolerance = 1e-12)
  g3 <- pgkw(x, 2, 3, 1.5, 2, 1, lower.tail = FALSE)
  expect_equal(g3, pbkw(x, 2, 3, 1.5, 2, lower.tail = FALSE), tolerance = 1e-12)
})

test_that("the lower tail and the closed boundaries are untouched", {
  x <- sort(unique(c(10^-(1:20), seq(0.001, 0.999, by = 0.001), 1 - 10^-(1:16))))
  # the sum of the two tails is 1 wherever both are representable
  for (z in list(list(pgkw(x, 2, 3, 1.5, 2, 0.8), pgkw(x, 2, 3, 1.5, 2, 0.8, lower.tail = FALSE)),
                 list(pbkw(x, 2, 3, 1.5, 2), pbkw(x, 2, 3, 1.5, 2, lower.tail = FALSE)))) {
    lo <- z[[1]]; up <- z[[2]]
    f <- lo > 1e-12 & lo < 1 - 1e-12
    expect_equal(lo[f] + up[f], rep(1, sum(f)), tolerance = 1e-12)
  }
  expect_equal(pgkw(0, 2, 3, 1.5, 2, 0.8, lower.tail = FALSE), 1)
  expect_equal(pgkw(1, 2, 3, 1.5, 2, 0.8, lower.tail = FALSE), 0)
  expect_equal(pbkw(0, 2, 3, 1.5, 2, lower.tail = FALSE), 1)
  expect_equal(pbkw(1, 2, 3, 1.5, 2, lower.tail = FALSE), 0)
})

test_that("log.p in the upper tail stays consistent with the linear scale", {
  x <- sort(unique(c(1 - 10^-(1:16), seq(0.5, 0.99, by = 0.01))))
  for (z in list(list(pgkw(x, 2, 3, 1.5, 2, 0.8, lower.tail = FALSE, log.p = TRUE),
                      pgkw(x, 2, 3, 1.5, 2, 0.8, lower.tail = FALSE)),
                 list(pbkw(x, 2, 3, 1.5, 2, lower.tail = FALSE, log.p = TRUE),
                      pbkw(x, 2, 3, 1.5, 2, lower.tail = FALSE)))) {
    f <- z[[2]] > 0
    expect_equal(z[[1]][f], log(z[[2]][f]), tolerance = 1e-12)
    expect_true(all(is.finite(z[[1]])))
  }
})
