# tests/testthat/test-rng-log-space.R
# Regression tests for the random generators after the log-space rewrite.
#
# rbkw() drew V ~ Beta(gamma, delta+1) and then formed 1.0 - V. For V below
# 1.1e-16 that rounds to exactly 1 and the generator returned 0 -- a value
# outside the open support (0,1), which llbkw() then rejects. R::rbeta itself
# never returned a zero: every one was fabricated by the subtraction.
#
#   rbkw(1e5, 2, 3, 0.02, 0)   48,602 exact zeros   48.6% of the sample
#   rbkw(1e5, 2, 3, 0.05, 0)   16,440 exact zeros
#   rkkw(1e5, 0.2, 3, 0, 0.3)       6 exact zeros
#
# Six zeros in a hundred thousand were enough to break the package's own
# simulate-then-fit workflow: llkkw() at the true parameters returned Inf and
# optim() stopped with "L-BFGS-B needs finite values of 'fn'".
#
# The five affected generators (rgkw, rbkw, rkkw, rekw, rkw) now invert in log
# space. rmc() and rbeta_() were already free of the defect -- the first is
# U^(1/lambda) over an rbeta draw, the second delegates straight to R::rbeta --
# and are untouched.
#
# The draws themselves are unchanged, so set.seed() reproduces the same stream
# as before; only the inversion that follows differs.

log1mexp_ref <- function(u) ifelse(u > -log(2), log(-expm1(u)), log1p(-exp(u)))

test_that("generators stay inside the open support", {
  cases <- list(
    list(f = rbkw, par = list(2, 3, 0.02, 0)),
    list(f = rbkw, par = list(2, 3, 0.05, 0)),
    list(f = rbkw, par = list(2, 3, 0.10, 0)),
    list(f = rkkw, par = list(0.2, 3, 0, 0.3)),
    list(f = rkkw, par = list(2, 3, 0.5, 1.5)),
    list(f = rgkw, par = list(2, 3, 1.5, 2, 0.8)),
    list(f = rekw, par = list(2, 3, 1.5)),
    list(f = rkw,  par = list(2, 3)),
    list(f = rmc,  par = list(2, 3, 2.5)),
    list(f = rbeta_, par = list(2, 3))
  )
  for (cs in cases) {
    set.seed(5)
    x <- do.call(cs$f, c(list(20000L), cs$par))
    expect_length(x, 20000L)
    expect_true(all(x > 0), info = paste("zeros:", sum(x <= 0)))
    expect_true(all(x < 1))
    expect_false(anyNA(x))
  }
})

test_that("rbkw reproduces the closed-form inversion of its own draw", {
  # The RNG stream is unchanged, so the underlying Beta draws can be replayed.
  for (par in list(c(2, 3, 0.02, 0), c(2, 3, 1.5, 2), c(0.5, 0.5, 0.5, 0))) {
    set.seed(5)
    x <- rbkw(20000L, par[1], par[2], par[3], par[4])
    set.seed(5)
    V <- stats::rbeta(20000L, par[3], par[4] + 1)
    ref <- exp(log1mexp_ref(log1p(-V) / par[2]) / par[1])
    expect_equal(x, ref, tolerance = 1e-12)
  }
})

test_that("the draws are unchanged, so set.seed still reproduces", {
  for (cs in list(list(f = rkw, par = list(2, 3)),
                  list(f = rbkw, par = list(2, 3, 1.5, 2)),
                  list(f = rgkw, par = list(2, 3, 1.5, 2, 0.8)))) {
    set.seed(11); a <- do.call(cs$f, c(list(500L), cs$par))
    set.seed(11); b <- do.call(cs$f, c(list(500L), cs$par))
    expect_identical(a, b)
  }

  # the generator consumes the stream exactly once per variate
  set.seed(3); first <- rkw(5, 2, 3)
  set.seed(3); both <- rkw(10, 2, 3)
  expect_equal(first, both[1:5], tolerance = 1e-14)
})

test_that("the sample follows its own distribution", {
  cases <- list(
    list(r = rbkw, p = pbkw, par = list(2, 3, 0.02, 0)),
    list(r = rbkw, p = pbkw, par = list(2, 3, 1.5, 2)),
    list(r = rkkw, p = pkkw, par = list(0.2, 3, 0, 0.3)),
    list(r = rgkw, p = pgkw, par = list(2, 3, 1.5, 2, 0.8)),
    list(r = rekw, p = pekw, par = list(2, 3, 1.5)),
    list(r = rkw,  p = pkw,  par = list(2, 3))
  )
  for (cs in cases) {
    set.seed(77)
    x <- do.call(cs$r, c(list(20000L), cs$par))
    k <- suppressWarnings(
      stats::ks.test(x, function(q) do.call(cs$p, c(list(q), cs$par))))
    expect_gt(k$p.value, 0.001)
  }
})

test_that("simulate then fit works end to end", {
  # This is the flow that six zeros in 100,000 used to break.
  set.seed(5)
  x <- rkkw(20000L, 0.2, 3, 0, 0.3)
  expect_true(all(x > 0 & x < 1))
  expect_true(is.finite(llkkw(c(0.2, 3, 0, 0.3), x)))

  fit <- optim(c(0.2, 3, 0, 0.3), llkkw, gr = grkkw, data = x,
               method = "L-BFGS-B", lower = c(1e-6, 1e-6, 0, 1e-6))
  expect_equal(fit$convergence, 0L)
  expect_true(is.finite(fit$value))

  set.seed(5)
  xb <- rbkw(20000L, 2, 3, 0.02, 0)
  expect_true(is.finite(llbkw(c(2, 3, 0.02, 0), xb)))
})

test_that("rmc and rbeta_ are byte-for-byte what they always were", {
  # Neither forms 1 - u, so neither was touched. rbeta_ must still be exactly
  # R's own Beta draw.
  set.seed(21); a <- rbeta_(5000L, 2, 3)
  set.seed(21); b <- stats::rbeta(5000L, 2, 4)
  expect_identical(a, b)

  set.seed(21); m <- rmc(5000L, 2, 3, 2.5)
  set.seed(21); u <- stats::rbeta(5000L, 2, 4)
  expect_equal(m, u^(1 / 2.5), tolerance = 1e-14)
})
