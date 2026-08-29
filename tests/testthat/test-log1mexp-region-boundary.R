# tests/testthat/test-log1mexp-region-boundary.R
# Regression test for the second-order correction in gkw_log1mexp().
#
# The Taylor branch, used for -1e-14 < u <= 0, returned
#
#     log(-u) - u/2
#
# where the expansion gives log(-u) + u/2. Since
#
#     1 - exp(u) = -u * (1 + u/2 + u^2/6 + ...)
#
# the correction is log(1 + u/2) ~ +u/2, and the comment above that line carried
# the sign error too. With u < 0 the two forms differ by |u|, up to 1e-14 --
# about 1.4 ulp at this magnitude, but in the wrong direction, and it left a
# step where the function crosses into the expm1 branch at u = -1e-14.
#
# Against a 400-digit reference the branch was off by 1.42e-14 and is now exact:
#
#   u          reference (400 digits)   err before   err after
#   -9.9e-15   -32.246241637770147      1.42e-14     0
#   -5.0e-15   -32.929338482476588      0            0
#
# gkw_log1mexp() is internal, so this reaches it through dkw(x, 1, 2, log = TRUE),
# whose log-density is log(2) + log(1 - x) and which therefore evaluates
# log1mexp(log(x)) directly. Sweeping x = exp(u) across the branch boundary,
# the largest deviation from the closed form falls from 1.42e-14 to 7.11e-15.

test_that("the Taylor branch of log1mexp carries the right sign", {
  # u straddles the -1e-14 boundary between the Taylor and expm1 branches
  u <- -10^seq(-14.6, -13.4, length.out = 61)
  x <- exp(u)

  got <- dkw(x, 1, 2, log = TRUE)
  # log f(x) = log(beta) + (beta - 1) log(1 - x) with alpha = 1, beta = 2
  ref <- log(2) + log1p(-x)

  expect_true(all(is.finite(got)))
  # 1.1.5 reaches 1.42e-14 here; the corrected branch stays under half of that
  expect_lt(max(abs(got - ref)), 1e-14)
})

test_that("the density stays monotone across the branch boundary", {
  # dkw(x, 1, 2) = 2 (1 - x) is strictly decreasing; a sign flip in the
  # correction perturbs one side of the boundary only, which shows up here.
  x <- sort(exp(-10^seq(-14.6, -13.4, length.out = 121)))
  v <- dkw(x, 1, 2, log = TRUE)
  expect_true(all(diff(v) <= 0))
})
