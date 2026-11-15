# tests/testthat/test-rng-log-space.R
# The random generators never return a value outside the open support (0,1).
#
# A draw V ~ Beta(gamma, delta+1) followed by 1.0 - V in linear space rounds to
# exactly 1 for V below 1.1e-16, which would make the variate exactly 0 -- a
# value llbkw() rejects. In linear space the regime is common:
#
#   rbkw(1e5, 2, 3, 0.02, 0)   about 48% of the draws have V < 1.1e-16
#   rbkw(1e5, 2, 3, 0.05, 0)   about 16%
#   rkkw(1e5, 0.2, 3, 0, 0.3)  a few draws in 1e5 reach the same regime
#
# A single zero breaks a simulate-then-fit workflow: the likelihood at the true
# parameters is Inf and optim() stops with "L-BFGS-B needs finite values of
# 'fn'".
#
# rgkw, rbkw, rkkw, rekw and rkw invert in log space. rmc() is U^(1/lambda)
# over an rbeta draw and rbeta_() delegates straight to R::rbeta.

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
  # The underlying Beta draws can be replayed with the same seed.
  for (par in list(c(2, 3, 0.02, 0), c(2, 3, 1.5, 2), c(0.5, 0.5, 0.5, 0))) {
    set.seed(5)
    x <- rbkw(20000L, par[1], par[2], par[3], par[4])
    set.seed(5)
    V <- stats::rbeta(20000L, par[3], par[4] + 1)
    ref <- exp(log1mexp_ref(log1p(-V) / par[2]) / par[1])
    expect_equal(x, ref, tolerance = 1e-12)
  }
})

test_that("set.seed reproduces the draws", {
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
  # A single exact zero in the sample would make the likelihood Inf.
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

test_that("rmc and rbeta_ are plain transformations of R's Beta draw", {
  # Neither forms 1 - u, so neither is affected. rbeta_ must be exactly
  # R's own Beta draw.
  set.seed(21); a <- rbeta_(5000L, 2, 3)
  set.seed(21); b <- stats::rbeta(5000L, 2, 4)
  expect_identical(a, b)

  set.seed(21); m <- rmc(5000L, 2, 3, 2.5)
  set.seed(21); u <- stats::rbeta(5000L, 2, 4)
  expect_equal(m, u^(1 / 2.5), tolerance = 1e-14)
})
