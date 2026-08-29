# tests/testthat/test-gkw-log-chain.R
# Regression tests for the shared log-space chain in gkw.cpp.
#
# dgkw(), llgkw() and grgkw() all walk
#
#   v = 1 - x^alpha     w = 1 - v^beta     z = 1 - w^lambda
#
# and each had its own way of falling apart on the way down.
#
# 1. llgkw() went through vec_safe_log(vec_safe_pow(x, alpha)) instead of
#    alpha*log(x), a round trip that also made it disagree with dgkw().
#
# 2. Two of the three transformations underflow to a boundary that log1mexp()
#    cannot recover from, because its argument arrives as exactly 0 and
#    log(1 - exp(0)) is -Inf. With a zero coefficient in front, 0 * -Inf is NaN:
#
#      llgkw(c(1, 300, 1, 0, 1), c(.8,.85,.9,.95))  ->  NaN   (exact: 2609.84)
#
# 3. grgkw() built 1/v, 1/w and 1/z as separate reciprocals, each overflowing
#    on its own long before the product it belonged to was large:
#
#      par = (1, 70, 1.5, 2, 1), x = c(.10,.25,.40,.72,.99)
#        llgkw           1386.78983044   (finite, correct)
#        grgkw           NaN NaN NaN NaN NaN
#        numDeriv::grad  -662.27 20.27 -6.76 472.41 -15.00
#
# Against 1.1.5 these blocks fail 47 assertions.

PARS <- list(c(2,3,1.5,2,0.8), c(1,70,1.5,2,1), c(1,200,1.5,2,1), c(1,1000,1.5,2,1),
             c(1,300,1,0,1), c(1,1,0.1,0.1,1), c(0.5,0.5,0.5,0.5,0.5),
             c(2.5,2.5,1,0,20), c(3,100,2,1,0.3), c(1,5000,1,0,1))
DATA <- list(facil   = c(0.10, 0.25, 0.40, 0.72, 0.99),
             alto    = c(.8, .85, .9, .95),
             perto1  = c(1 - 1e-9, 1 - 1e-12, 0.5),
             perto0  = c(1e-9, 1e-14, 0.5),
             extremo = c(1e-300, 0.5, 1 - 1e-15))

test_that("llgkw is finite wherever the density is", {
  for (nm in names(DATA)) {
    for (p in PARS) {
      v <- suppressWarnings(llgkw(p, DATA[[nm]]))
      expect_true(is.finite(v), info = paste(nm, paste(p, collapse = ",")))
    }
  }
  # the case the audit reported, to the digit
  expect_equal(llgkw(c(1, 300, 1, 0, 1), c(.8, .85, .9, .95)),
               2609.842574, tolerance = 1e-6)
})

test_that("grgkw is finite wherever llgkw is", {
  for (nm in names(DATA)) {
    for (p in PARS) {
      g <- suppressWarnings(grgkw(p, DATA[[nm]]))
      expect_true(all(is.finite(g)), info = paste(nm, paste(p, collapse = ",")))
    }
  }
})

test_that("grgkw agrees with the analytic BKw gradient at lambda = 1", {
  # bkw.cpp is an independent implementation of the same quantity, so this is a
  # stronger reference than numDeriv, whose step is quantised away by the
  # subnormal log_w that arises for large beta.
  P4 <- list(c(1, 70, 1.5, 2), c(1, 200, 1.5, 2), c(1, 1000, 1.5, 2),
             c(2, 3, 1.5, 2), c(3, 100, 2, 1))
  set.seed(3)
  sets <- list(c(0.10, 0.25, 0.40, 0.72, 0.99), c(1 - 1e-9, 1 - 1e-12, 0.5),
               rgkw(300, 2, 3, 1.5, 2, 0.8))
  for (d in sets) {
    for (p in P4) {
      g <- suppressWarnings(grgkw(c(p, 1), d))[1:4]
      b <- suppressWarnings(grbkw(p, d))
      if (all(is.finite(b))) {
        expect_equal(g, b, tolerance = 1e-6,
                     info = paste(paste(p, collapse = ","), length(d)))
      }
    }
  }
})

test_that("llgkw still equals every nested family", {
  set.seed(8)
  s <- rgkw(400, 2, 3, 1.5, 2, 0.8)
  expect_equal(llgkw(c(2, 3, 1, 0, 1), s),     llkw(c(2, 3), s),        tolerance = 1e-12)
  expect_equal(llgkw(c(2, 3, 1.5, 2, 1), s),   llbkw(c(2, 3, 1.5, 2), s), tolerance = 1e-12)
  expect_equal(llgkw(c(2, 3, 1, 0, 1.5), s),   llekw(c(2, 3, 1.5), s),  tolerance = 1e-12)
  expect_equal(llgkw(c(2, 3, 1, 2, 1.5), s),   llkkw(c(2, 3, 2, 1.5), s), tolerance = 1e-12)
  expect_equal(llgkw(c(1, 1, 1.5, 2, 0.8), s), llmc(c(1.5, 2, 0.8), s), tolerance = 1e-12)
})

test_that("dgkw no longer zeroes densities the nested families report", {
  X <- sort(unique(c(10^-(1:300), 1 - 10^-(1:16), seq(0.01, 0.99, by = 0.01))))
  X <- X[X > 0 & X < 1]
  cases <- list(
    list(dgkw(X, 2, 3, 1, 0, 1, log = TRUE),      dkw(X, 2, 3, log = TRUE)),
    list(dgkw(X, 1, 70, 1, 0, 1, log = TRUE),     dkw(X, 1, 70, log = TRUE)),
    list(dgkw(X, 1, 200, 1.5, 2, 1, log = TRUE),  dbkw(X, 1, 200, 1.5, 2, log = TRUE)),
    list(dgkw(X, 2, 3, 1, 0, 1.5, log = TRUE),    dekw(X, 2, 3, 1.5, log = TRUE)),
    list(dgkw(X, 2, 3, 1, 2, 1.5, log = TRUE),    dkkw(X, 2, 3, 2, 1.5, log = TRUE)))
  for (z in cases) {
    a <- z[[1]]; b <- z[[2]]
    # no point where the reduced family is finite may be dropped by dgkw
    expect_equal(sum(is.finite(b) & !is.finite(a)), 0)
    fin <- is.finite(a) & is.finite(b)
    expect_lt(max(abs(a - b)[fin]), 1e-10)
  }
})
