# tests/testthat/test-mcdonald-precision.R
# Regression tests for the numerical defects in bpmc.cpp.
#
# ---------------------------------------------------------------------------
# 1. dmc() lost the density as x approached 1.
#
# dmc() formed x^lambda in linear arithmetic and then took log(1 - x^lambda).
# Doubles are spaced 2.2e-16 apart just below 1, so 1 - x^lambda carries an
# absolute error of one ulp of 1 however small it truly is, and x^lambda rounds
# to exactly 1 once 1 - x drops under about 1e-16. A guard then returned a
# density of zero (-Inf in log) where the true density is finite.
#
# Mc(gamma, delta, lambda) is GKw(1, 1, gamma, delta, lambda), so dgkw() with
# alpha = beta = 1 is the same density computed a different way. It uses
# gkw_log1mexp(lambda * log(x)) and was right all along:
#
#   gamma = 1.5, delta = 2, lambda = 0.8   exact          dgkw(x,1,1,..)   dmc
#     x = 1 - 1e-13                  -58.65464965    -58.65464965   -58.65409479
#     x = 1 - 1e-15                  -67.86721101    -67.86721101   -67.92355276
#     x = 1 - 1e-16                  -72.26166017    -72.26166017   -71.81537306
#
# The exact column is a 400-digit `decimal` reference fed the exact doubles R
# holds. The disagreement reaches 0.446 nats, and at x = 1 - 1.1e-16 dmc()
# returned -Inf outright for six of the parameter settings tested.
#
# llmc() and grmc() carried the mirror image of the same defect at the other
# end of the support: both used log(-expm1(u)), which has to represent a number
# just below 1 and so cannot resolve log(1 - x^lambda) once x^lambda falls under
# 1.1e-16. gkw_log1mexp() switches to log1p(-exp(u)) past -log(2) and keeps the
# full relative accuracy. All four now go through the same helper.

log1mexp_ref <- function(u) ifelse(u > -log(2), log(-expm1(u)), log1p(-exp(u)))

# The McDonald log-density, written the way the identity dictates.
ref_ldmc <- function(x, g, d, l) {
  log(l) - lbeta(g, d + 1) + (g * l - 1) * log(x) + d * log1mexp_ref(l * log(x))
}

near_one <- 1 - c(1e-8, 1e-10, 1e-12, 1e-13, 1e-14, 1e-15, 1e-16, .Machine$double.eps / 2)

pars <- list(
  c(1.5, 2.0, 0.8),
  c(2.0, 3.0, 2.5),
  c(0.3, 0.2, 0.4),
  c(4.0, 2.0, 0.25),
  c(1.0, 1.0, 0.5),
  c(50.0, 40.0, 1.2)
)

test_that("dmc is finite throughout the band where x^lambda rounds to 1", {
  for (p in pars) {
    v <- dmc(near_one, p[1], p[2], p[3], log = TRUE)
    expect_true(all(is.finite(v)),
                info = sprintf("gamma=%g delta=%g lambda=%g", p[1], p[2], p[3]))
    # On the natural scale the density is positive wherever it is representable
    # at all; Mc(50, 40, 1.2) genuinely underflows there, at a log-density of
    # -1398.7.
    representable <- v > log(.Machine$double.xmin)
    expect_true(all(dmc(near_one, p[1], p[2], p[3])[representable] > 0),
                info = sprintf("gamma=%g delta=%g lambda=%g", p[1], p[2], p[3]))
  }
})

test_that("dmc equals dgkw(x, 1, 1, gamma, delta, lambda), the same density", {
  # This is the nesting identity Mc(g,d,l) == GKw(1,1,g,d,l). It used to fail by
  # up to 7.29 nats, and by an infinite margin wherever dmc returned -Inf.
  for (p in pars) {
    got <- dmc(near_one, p[1], p[2], p[3], log = TRUE)
    want <- dgkw(near_one, 1, 1, p[1], p[2], p[3], log = TRUE)
    expect_equal(got, want, tolerance = 1e-13,
                 info = sprintf("gamma=%g delta=%g lambda=%g", p[1], p[2], p[3]))
  }
})

test_that("dmc matches the closed-form reference at the reported points", {
  x <- c(1 - 1e-13, 1 - 1e-15, 1 - 1e-16)
  got <- dmc(x, 1.5, 2, 0.8, log = TRUE)
  # 400-digit values, rounded to the double they belong to. dmc() used to
  # return -58.65409479, -67.92355276 and -71.81537306 here.
  expect_equal(got, c(-58.654649650162412, -67.867211010706654,
                      -72.261660165379084),
               tolerance = 1e-14)
  expect_equal(got, ref_ldmc(x, 1.5, 2, 0.8), tolerance = 1e-13)
})

test_that("dmc(x, gamma, delta, 1) is the Beta(gamma, delta+1) density", {
  # lambda = 1 collapses the McDonald family onto the Beta, so stats::dbeta is
  # an independent implementation of the same numbers.
  x <- c(1e-12, 1e-6, 0.01, 0.3, 0.7, 0.99, 1 - 1e-8, 1 - 1e-12, 1 - 1e-15)
  for (gd in list(c(2, 3), c(0.5, 0.5), c(1.5, 2), c(50, 40))) {
    expect_equal(dmc(x, gd[1], gd[2], 1, log = TRUE),
                 stats::dbeta(x, gd[1], gd[2] + 1, log = TRUE),
                 tolerance = 1e-13,
                 info = sprintf("gamma=%g delta=%g", gd[1], gd[2]))
  }
  # Deep lower tail with a large delta: safe_log(1 - x^lambda) returned exactly
  # 0 for every x under one ulp of 1, and delta multiplied the missing term.
  # The gap reached 5.0e-05 at delta = 1e12 against stats::dbeta.
  deep <- c(1e-20, 1e-18, 1e-17, 5e-17, 1e-16, 1e-15)
  for (d in c(1e8, 1e10, 1e12))
    expect_equal(dmc(deep, 2, d, 1, log = TRUE),
                 stats::dbeta(deep, 2, d + 1, log = TRUE),
                 tolerance = 1e-13, info = sprintf("delta=%g", d))
})

test_that("dmc integrates to the probability pmc assigns the same band", {
  # The rejected band carried real probability mass. Integrating the density
  # over the upper tail must reproduce pmc's upper tail, an independent route
  # through R::pbeta.
  for (p in pars) {
    tot <- stats::integrate(function(z) dmc(z, p[1], p[2], p[3]),
                            0, 1, rel.tol = 1e-10, subdivisions = 2000)$value
    expect_equal(tot, 1, tolerance = 1e-6,
                 info = sprintf("gamma=%g delta=%g lambda=%g", p[1], p[2], p[3]))
    for (lo in c(1 - 1e-6, 1 - 1e-8)) {
      band <- stats::integrate(function(z) dmc(z, p[1], p[2], p[3]), lo, 1,
                               rel.tol = 1e-10)$value
      tail <- pmc(lo, p[1], p[2], p[3], lower.tail = FALSE)
      # A relative comparison: the tail probability is as small as 1.1e-24 here
      # and an absolute tolerance would accept the zero the old dmc() returned,
      # which is what this is for. The tolerance is loose because the quadrature
      # is the weaker of the two sides -- for Mc(0.3, 0.2, 0.4) the density
      # diverges as (1-x)^-0.8 at the upper end and integrate() comes back
      # 3.3e-05 relative high against a 200-digit incomplete beta.
      expect_equal(band / tail, 1, tolerance = 1e-3,
                   info = sprintf("gamma=%g delta=%g lambda=%g lo=%g",
                                  p[1], p[2], p[3], lo))
    }
  }
})

test_that("llmc and grmc resolve log(1 - x^lambda) below one ulp of 1", {
  # log(-expm1(u)) has to represent a number just below 1, which doubles cannot
  # resolve more finely than 1.11e-16, so log(1 - x^lambda) came back as a
  # multiple of that -- usually as exactly 0 -- for every x^lambda under one
  # ulp. At delta = 1e12 the missing term is worth 2e-05 nats an observation.
  #
  # Both data sets have the same length, so the constant term
  # n*(log(lambda) - lbeta(gamma, delta+1)) and the digamma terms of the
  # gradient cancel in the difference. Those carry defects of their own above
  # delta = 100 and would otherwise mask this one.
  A <- c(2e-17, 5e-17, 8e-17, 1.2e-16, 3e-16)
  B <- c(1e-17, 4e-17, 7e-17, 1.0e-16, 2e-16)
  for (d in c(1e6, 1e9, 1e12)) {
    par <- c(2, d, 1)
    lab <- sprintf("delta=%g", d)
    expect_equal(llmc(par, A) - llmc(par, B),
                 -sum(ref_ldmc(A, par[1], par[2], par[3])) +
                   sum(ref_ldmc(B, par[1], par[2], par[3])),
                 tolerance = 1e-12, info = lab)
    # The delta component of the gradient is sum(log(1 - x^lambda)) itself.
    expect_equal(as.numeric(grmc(par, A))[2] - as.numeric(grmc(par, B))[2],
                 -sum(log1mexp_ref(par[3] * log(A))) +
                   sum(log1mexp_ref(par[3] * log(B))),
                 tolerance = 1e-12, info = lab)
  }
})

test_that("llmc equals -sum(dmc(log = TRUE)), the objective it optimises", {
  # The two used different expressions for delta * log(1 - x^lambda) and so
  # disagreed by up to 0.45 nats an observation near x = 1.
  datasets <- list(
    c(0.1, 0.5, 0.9),
    c(1e-15, 1e-8, 0.3, 0.7, 1 - 1e-13),
    1 - c(1e-9, 1e-12, 1e-14, 1e-15, 1e-16),
    c(1e-200, 1e-100, 0.2, 0.8)
  )
  for (data in datasets)
    for (p in pars)
      expect_equal(llmc(p, data), -sum(dmc(data, p[1], p[2], p[3], log = TRUE)),
                   tolerance = 1e-12)
})

test_that("llmc equals llgkw with alpha = beta = 1", {
  datasets <- list(
    c(0.2, 0.5, 0.8),
    1 - c(1e-10, 1e-13, 1e-15, 1e-16),
    c(1e-100, 1e-20, 0.4, 1 - 1e-14)
  )
  for (data in datasets)
    for (p in pars)
      expect_equal(llmc(p, data), llgkw(c(1, 1, p), data), tolerance = 1e-12)
})


# ---------------------------------------------------------------------------
# 2. llmc() replaced R::lbeta with a difference of lgamma above gamma or
#    delta = 100.
#
# That difference is the cancellation R::lbeta exists to avoid. At gamma = 1e12,
# delta = 2 the two outer lgamma values are 2.66e13, where one ulp is 3.9e-03,
# so their difference cannot resolve an answer of -82.2 any better than that:
#
#   R::lbeta(1e12, 3)                          -82.1999161672287   (exact)
#   lgamma(1e12) + lgamma(3) - lgamma(1e12+3)  -82.203125          (off 3.2e-03)
#
# dmc() and llbeta() always called R::lbeta, so llmc() disagreed with both.

test_that("llmc uses R::lbeta at every gamma and delta", {
  # x is chosen so (gamma*lambda - 1) * log(x) stays of order 1; the answer is
  # then made almost entirely of the constant term, which is where the defect
  # lived. The exact column is a 200-digit reference.
  cases <- list(
    list(par = c(1e12, 2, 1), x = 1 - 1e-12, exact = -25.937851813162602),
    list(par = c(1e10, 2, 1), x = 1 - 1e-10, exact = -21.332703832470877),
    list(par = c(1e8,  2, 1), x = 1 - 1e-8,  exact = -16.727533603417179),
    list(par = c(2, 1e12, 1), x = 1e-12,     exact = -26.631021115931048),
    list(par = c(2, 1e10, 1), x = 1e-10,     exact = -22.025850930190458),
    list(par = c(2, 1e8,  1), x = 1e-8,      exact = -17.420680768952366)
  )
  # llmc used to return -25.9371543959339 for the first of these.
  for (cc in cases)
    expect_equal(llmc(cc$par, cc$x), cc$exact, tolerance = 1e-13,
                 info = sprintf("gamma=%g delta=%g", cc$par[1], cc$par[2]))
})

test_that("llmc agrees with -sum(dmc(log = TRUE)) above the former cut-off", {
  # dmc() always called R::lbeta, so the two disagreed for gamma or delta > 100.
  data <- c(0.05, 0.2, 0.5, 0.8, 0.95)
  for (p in list(c(150, 20, 1), c(20, 150, 1), c(1e6, 2, 1), c(2, 1e6, 1),
                 c(1e12, 2, 1), c(2, 1e12, 1), c(1e5, 1e5, 1.5)))
    expect_equal(llmc(p, data), -sum(dmc(data, p[1], p[2], p[3], log = TRUE)),
                 tolerance = 1e-14,
                 info = sprintf("gamma=%g delta=%g", p[1], p[2]))
})

test_that("llmc equals llbeta at lambda = 1 above the former cut-off", {
  # Mc(gamma, delta, 1) is Beta(gamma, delta+1), and llbeta() always called
  # R::lbeta. The data sits just below 1 so that (gamma-1)*sum(log x) stays of
  # order 1: with bulk data that term reaches -6.2e10 and one ulp of the answer
  # is wider than the whole defect.
  for (g in c(1e8, 1e10, 1e12)) {
    data <- 1 - c(1, 2, 5) / g
    expect_equal(llmc(c(g, 2, 1), data), llbeta(c(g, 2), data),
                 tolerance = 1e-13, info = sprintf("gamma=%g", g))
  }
  # delta beyond the cut-off, with bulk data where llbeta is itself reliable.
  data <- c(0.05, 0.2, 0.5, 0.8, 0.95)
  for (gd in list(c(150, 20), c(20, 150), c(1e5, 1e5)))
    expect_equal(llmc(c(gd, 1), data), llbeta(gd, data), tolerance = 1e-13,
                 info = sprintf("gamma=%g delta=%g", gd[1], gd[2]))
})
