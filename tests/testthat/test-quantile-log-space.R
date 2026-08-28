# tests/testthat/test-quantile-log-space.R
# Regression tests for the seven quantile functions after the log-space rewrite.
#
# Each q*() converted log.p back with exp(), folded the upper tail with 1 - p,
# and then inverted using 1 - u in linear space at every step. The result was
# not merely imprecise: it left the open support (0,1) entirely.
#
#   qekw(0.02, 20, 0.1, 0.1)      returned 0   true value 0.1587
#   qekw(1e-08, 5, 2, 0.5)        returned 0   true value 5.49e-04
#   qkw(1e-16, 0.2, 2)            returned 0   true value 3.125e-82
#   qbeta_(-1000, 2, 3, log.p=T)  returned 0   true value 2.25e-218
#
# A quantile of exactly 0 or 1 then feeds d*() and ll*() a value outside the
# support they accept, so the damage propagates.
#
# The inversion now runs on log(u) and log(1-u), both carried from whatever
# scale and tail the caller used, and the incomplete-beta families hand
# lower_tail/log_p to R::qbeta rather than undoing them first.

l1m <- function(u) ifelse(u > -log(2), log(-expm1(u)), log1p(-exp(u)))
logs <- function(p, lower, logp) {
  if (logp) { if (lower) list(u = p, m = l1m(p)) else list(u = l1m(p), m = p) }
  else      { if (lower) list(u = log(p), m = log1p(-p)) else list(u = log1p(-p), m = log(p)) }
}

test_that("quantiles stay inside the open support", {
  expect_equal(qekw(0.02, 20, 0.1, 0.1), 0.15867737153067335, tolerance = 1e-12)
  expect_equal(qekw(1e-08, 5, 2, 0.5), 5.4928027165305809e-04, tolerance = 1e-12)
  expect_equal(qkw(1e-16, 0.2, 2), 3.125e-82, tolerance = 1e-12)
  expect_equal(qekw(0.01, 0.1, 0.1, 0.1), 1.0000000000000528e-190, tolerance = 1e-10)
  expect_equal(qbeta_(-1000, 2, 3, log.p = TRUE),
               stats::qbeta(-1000, 2, 4, log.p = TRUE), tolerance = 1e-12)

  # every one of these used to be exactly 0
  for (v in c(qekw(0.02, 20, 0.1, 0.1), qekw(1e-08, 5, 2, 0.5), qkw(1e-16, 0.2, 2),
              qgkw(1e-12, 1, 1, 0.1, 0.1, 1), qmc(0.01, 0.1, 0.1, 0.1))) {
    expect_gt(v, 0)
    expect_lt(v, 1)
  }
})

test_that("the closed-form inversion is reproduced", {
  us <- c(1e-300, 1e-100, 1e-12, 1e-06, 0.01, 0.3, 0.5, 0.9, 1 - 1e-06, 1 - 1e-12)
  for (p in list(c(2, 3), c(0.2, 2), c(20, 20), c(0.5, 0.5))) {
    for (lower in c(TRUE, FALSE)) {
      L <- logs(us, lower, FALSE)
      expect_equal(qkw(us, p[1], p[2], lower.tail = lower),
                   exp(l1m(L$m / p[2]) / p[1]), tolerance = 1e-11)
    }
  }
  for (p in list(c(2, 3, 1.5), c(20, 0.1, 0.1), c(5, 2, 0.5))) {
    L <- logs(us, TRUE, FALSE)
    expect_equal(qekw(us, p[1], p[2], p[3]),
                 exp(l1m(l1m(L$u / p[3]) / p[2]) / p[1]), tolerance = 1e-11)
  }
})

test_that("p(q(u)) recovers u where the quantile is representable", {
  # Restricted to quantiles strictly inside the support and away from the
  # double's edges: once q lands within a few ulps of 0 or 1 the round trip
  # cannot close, whichever formula is used.
  us <- c(1e-100, 1e-40, 1e-12, 1e-06, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9)
  cases <- list(
    list(q = qkw,    p = pkw,    par = list(2, 3)),
    list(q = qekw,   p = pekw,   par = list(2, 3, 1.5)),
    list(q = qkkw,   p = pkkw,   par = list(2, 3, 0.5, 1.5)),
    list(q = qbkw,   p = pbkw,   par = list(2, 3, 1.5, 2)),
    list(q = qgkw,   p = pgkw,   par = list(2, 3, 1.5, 2, 0.8)),
    list(q = qmc,    p = pmc,    par = list(2, 3, 2.5)),
    list(q = qbeta_, p = pbeta_, par = list(2, 3))
  )
  for (cs in cases) {
    qq <- do.call(cs$q, c(list(us), cs$par))
    expect_true(all(qq > 0 & qq < 1))
    ok <- qq > 1e-290 & qq < 1 - 1e-14
    back <- do.call(cs$p, c(list(qq[ok]), cs$par, list(log.p = TRUE)))
    expect_equal(back, log(us[ok]), tolerance = 1e-8)
  }
})

test_that("log.p reaches quantiles the linear scale cannot", {
  # Below about log(p) = -745 the chain cannot help: exp(log u) underflows, so
  # 1 - u rounds to exactly 1 and the inversion has nothing left to invert. The
  # first-order recovery there, log(u) - log(beta), needs each step to carry
  # both log(q) and log(1-q) and belongs with its own change. 1.1.5 already
  # returned 0 from log(p) = -40 downward.
  for (lp in c(-700, -300, -100, -30)) {
    expect_gt(qbeta_(lp, 2, 3, log.p = TRUE), 0)
    expect_gt(qkw(lp, 2, 3, log.p = TRUE), 0)
    expect_gt(qekw(lp, 2, 3, 1.5, log.p = TRUE), 0)
  }
  expect_equal(qkw(-700, 2, 3, log.p = TRUE),
               exp(l1m(l1m(-700) / 3) / 2), tolerance = 1e-11)
  expect_equal(qkw(-1000, 2, 3, log.p = TRUE), 0)   # documented limit
})

test_that("the boundary conventions are unchanged", {
  # These are the values 1.1.5 returned; whether out-of-range p should be NaN
  # is a separate question and is deliberately not settled here.
  expect_equal(qkw(0, 2, 3), 0)
  expect_equal(qkw(1, 2, 3), 1)
  expect_equal(qkw(0, 2, 3, lower.tail = FALSE), 1)
  expect_equal(qkw(1, 2, 3, lower.tail = FALSE), 0)
  expect_equal(suppressWarnings(qkw(-0.5, 2, 3)), 0)
  expect_equal(suppressWarnings(qkw(1.5, 2, 3)), 1)
  expect_equal(suppressWarnings(qkw(-0.5, 2, 3, lower.tail = FALSE)), 1)
  expect_equal(suppressWarnings(qkw(1.5, 2, 3, lower.tail = FALSE)), 0)
  expect_equal(qkw(-Inf, 2, 3, log.p = TRUE), 0)
  expect_equal(qkw(0, 2, 3, log.p = TRUE), 1)
  expect_true(is.na(suppressWarnings(qkw(1, 2, 3, log.p = TRUE))))
})

test_that("the quantile nesting identities hold", {
  u <- c(1e-12, 1e-06, 0.01, 0.3, 0.7, 0.99)
  expect_equal(qkw(u, 2, 3),          qgkw(u, 2, 3, 1, 0, 1),   tolerance = 1e-11)
  expect_equal(qekw(u, 2, 3, 1.5),    qgkw(u, 2, 3, 1, 0, 1.5), tolerance = 1e-11)
  expect_equal(qbkw(u, 2, 3, 1.5, 2), qgkw(u, 2, 3, 1.5, 2, 1), tolerance = 1e-11)
  expect_equal(qmc(u, 2, 3, 2.5),     qgkw(u, 1, 1, 2, 3, 2.5), tolerance = 1e-11)
  expect_equal(qbeta_(u, 2, 3),       stats::qbeta(u, 2, 4),    tolerance = 1e-12)
})

test_that("quantiles are monotone in p", {
  u <- sort(c(10^-(1:80), seq(0.01, 0.99, by = 0.01)))
  for (cs in list(list(f = qkw, p = list(2, 3)), list(f = qekw, p = list(2, 3, 1.5)),
                  list(f = qgkw, p = list(2, 3, 1.5, 2, 0.8)))) {
    q <- do.call(cs$f, c(list(u), cs$p))
    expect_true(all(diff(q) >= 0))
  }
})
