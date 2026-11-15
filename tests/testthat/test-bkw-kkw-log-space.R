# tests/testthat/test-bkw-kkw-log-space.R
# The log-space chain in bkw.cpp and kkw.cpp.
#
# Both files walk the same transformations as gkw.cpp,
#
#   v = 1 - x^alpha     w = 1 - v^beta     z = 1 - w^lambda
#
# with lambda = 1 for BKw (so z = v^beta) and gamma = 1 for KKw. The tests
# cover the point where the linear quantity underflows to a boundary where the
# argument of log1mexp() is exactly 0 and log(1 - exp(0)) is -Inf.
#
# 1. The likelihood is finite once alpha*log(min(x)) crosses -745, on entirely
#    ordinary data. With x = c(0.01, 0.3, 0.6, 0.9):
#
#      alpha = 161  llbkw(c(a,2,1.5,1), x) = 1515.5201319383
#      alpha = 162  llbkw(c(a,2,1.5,1), x) = 1525.1333577380
#      alpha = 161  llkkw(c(a,2,1,1.5), x) = 1516.4127060577
#      alpha = 162  llkkw(c(a,2,1,1.5), x) = 1526.0259318659
#
#    An Inf from there on would be an artificial plateau on the likelihood
#    surface. At alpha = 161, x^alpha is a subnormal (alpha*log(0.01) =
#    -741.4) and its few remaining bits are not exact. The values above are
#    from a __float128 evaluation of the same likelihood; see
#    test-subnormal-band.R.
#
# 2. dbkw() and dkkw() keep such observations:
#    dbkw(1e-300, 2, 1.5, 1.2, 0.5, log = TRUE) equals
#    dgkw(1e-300, 2, 1.5, 1.2, 0.5, 1, log = TRUE) = -965.2651.
#
# 3. In the gradients and Hessians the ratios v^beta/w and w^lambda/z each
#    overflow to +Inf on their own while the quantity they multiply underflows
#    to 0, and 0 * Inf is NaN. The derivatives are finite where the likelihood
#    is:
#
#      par = (200, 2, 1.5, 1), x = c(.01,.30,.60,.90)
#        grgkw(lambda=1)  9.617994  -3.000000  1278.026571  -2.721489
#      par = (200, 2, 1, 1.5), x = c(.01,.30,.60,.90)
#        grgkw(gamma=1)   9.617994  -3.000000  -2.000000  1279.626571
#
# The nesting identities are the reference used throughout:
# BKw(a,b,g,d) == GKw(a,b,g,d,1), KKw(a,b,d,l) == GKw(a,b,1,d,l),
# BKw(a,b,1,0) == Kw(a,b) == KKw(a,b,0,1).

X_PLATEAU <- c(0.01, 0.3, 0.6, 0.9)

BKW_PARS <- list(c(160, 2, 1.5, 1), c(162, 2, 1.5, 1), c(200, 2, 1.5, 1),
                 c(500, 2, 1.5, 1), c(1000, 0.5, 2, 3), c(3, 500, 1, 2),
                 c(1, 300, 1, 0), c(5, 70, 1.5, 2))
KKW_PARS <- list(c(160, 2, 1, 1.5), c(162, 2, 1, 1.5), c(200, 2, 1, 1.5),
                 c(500, 2, 1, 1.5), c(1000, 0.5, 3, 2), c(3, 500, 2, 1),
                 c(1, 300, 0, 1), c(5, 70, 2, 1.5))

DATA <- list(plateau = X_PLATEAU,
             extremo = c(1e-300, 0.5, 1 - 1e-15),
             perto0  = c(1e-10, 1e-5, 0.5, 0.999999),
             perto1  = c(0.001, 0.002, 0.5, 0.998, 0.999))


test_that("llbkw and llkkw stay finite past alpha*log(min(x)) = -745", {
  # The collapse is a step: the last good alpha and the first bad one differ by
  # 1, and the likelihood is perfectly smooth across it.
  for (a in c(160, 161, 162, 163, 200, 500, 1000, 5000)) {
    lb <- llbkw(c(a, 2, 1.5, 1), X_PLATEAU)
    lk <- llkkw(c(a, 2, 1, 1.5), X_PLATEAU)
    expect_true(is.finite(lb), info = paste("llbkw alpha =", a))
    expect_true(is.finite(lk), info = paste("llkkw alpha =", a))
    # and they agree with the GKw parent
    expect_equal(lb, llgkw(c(a, 2, 1.5, 1, 1), X_PLATEAU), tolerance = 1e-12,
                 info = paste("llbkw vs llgkw, alpha =", a))
    expect_equal(lk, llgkw(c(a, 2, 1, 1, 1.5), X_PLATEAU), tolerance = 1e-12,
                 info = paste("llkkw vs llgkw, alpha =", a))
  }
  # the exact values either side of the underflow of x^alpha
  expect_equal(llbkw(c(161, 2, 1.5, 1), X_PLATEAU), 1515.5201319383, tolerance = 1e-10)
  expect_equal(llbkw(c(162, 2, 1.5, 1), X_PLATEAU), 1525.1333577380, tolerance = 1e-10)
  expect_equal(llkkw(c(161, 2, 1, 1.5), X_PLATEAU), 1516.4127060577, tolerance = 1e-10)
  expect_equal(llkkw(c(162, 2, 1, 1.5), X_PLATEAU), 1526.0259318659, tolerance = 1e-10)
  # the likelihood is monotone in alpha here, so no plateau survives
  ll <- vapply(160:200, function(a) llbkw(c(a, 2, 1.5, 1), X_PLATEAU), numeric(1))
  expect_true(all(diff(ll) > 0), info = "llbkw plateau in alpha")
  lk <- vapply(160:200, function(a) llkkw(c(a, 2, 1, 1.5), X_PLATEAU), numeric(1))
  expect_true(all(diff(lk) > 0), info = "llkkw plateau in alpha")
})


test_that("llbkw and llkkw match the GKw parent on the whole grid", {
  for (nm in names(DATA)) {
    x <- DATA[[nm]]
    for (p in BKW_PARS)
      expect_equal(llbkw(p, x), llgkw(c(p, 1), x), tolerance = 1e-10,
                   info = paste("llbkw", nm, paste(p, collapse = ",")))
    for (p in KKW_PARS)
      expect_equal(llkkw(p, x), llgkw(c(p[1], p[2], 1, p[3], p[4]), x),
                   tolerance = 1e-10,
                   info = paste("llkkw", nm, paste(p, collapse = ",")))
  }
})


test_that("dbkw and dkkw keep the log-density in the deep tail", {
  # x^alpha underflows to zero here.
  expect_equal(dbkw(1e-300, 2, 1.5, 1.2, 0.5, log = TRUE),
               dgkw(1e-300, 2, 1.5, 1.2, 0.5, 1, log = TRUE), tolerance = 1e-12)
  expect_equal(dkkw(1e-300, 2, 1.5, 0.5, 1.2, log = TRUE),
               dgkw(1e-300, 2, 1.5, 1, 0.5, 1.2, log = TRUE), tolerance = 1e-12)
  expect_true(is.finite(dbkw(1e-300, 2, 1.5, 1.2, 0.5, log = TRUE)))
  expect_true(is.finite(dkkw(1e-300, 2, 1.5, 0.5, 1.2, log = TRUE)))

  xs <- c(1e-300, 1e-200, 1e-100, 1e-50, 1e-10, 0.01, 0.5, 0.9, 1 - 1e-10, 1 - 1e-16)
  for (p in BKW_PARS)
    expect_equal(dbkw(xs, p[1], p[2], p[3], p[4], log = TRUE),
                 dgkw(xs, p[1], p[2], p[3], p[4], 1, log = TRUE), tolerance = 1e-12,
                 info = paste("dbkw", paste(p, collapse = ",")))
  for (p in KKW_PARS)
    expect_equal(dkkw(xs, p[1], p[2], p[3], p[4], log = TRUE),
                 dgkw(xs, p[1], p[2], 1, p[3], p[4], log = TRUE), tolerance = 1e-12,
                 info = paste("dkkw", paste(p, collapse = ",")))
})


test_that("dbkw and dkkw reduce to dkw exactly at the Kumaraswamy corner", {
  xs <- c(1e-300, 1e-100, 1e-20, 0.01, 0.5, 0.9, 1 - 1e-12)
  for (ab in list(c(2, 3), c(0.5, 0.5), c(200, 2), c(3, 500), c(5000, 1))) {
    ref <- dkw(xs, ab[1], ab[2], log = TRUE)
    expect_identical(dbkw(xs, ab[1], ab[2], 1, 0, log = TRUE), ref,
                     info = paste("dbkw == dkw", paste(ab, collapse = ",")))
    expect_identical(dkkw(xs, ab[1], ab[2], 0, 1, log = TRUE), ref,
                     info = paste("dkkw == dkw", paste(ab, collapse = ",")))
  }
})


test_that("grbkw and grkkw are finite and match the GKw parent", {
  # par = (200, 2, 1.5, 1): 0 * Inf in linear space, finite in log space.
  g <- grbkw(c(200, 2, 1.5, 1), X_PLATEAU)
  expect_true(all(is.finite(g)))
  expect_equal(g, grgkw(c(200, 2, 1.5, 1, 1), X_PLATEAU)[1:4], tolerance = 1e-10)
  expect_equal(g, c(9.617994, -3.000000, 1278.026571, -2.721489), tolerance = 1e-6)

  k <- grkkw(c(200, 2, 1, 1.5), X_PLATEAU)
  expect_true(all(is.finite(k)))
  expect_equal(k, grgkw(c(200, 2, 1, 1, 1.5), X_PLATEAU)[c(1, 2, 4, 5)], tolerance = 1e-10)
  expect_equal(k, c(9.617994, -3.000000, -2.000000, 1279.626571), tolerance = 1e-6)

  for (nm in names(DATA)) {
    x <- DATA[[nm]]
    for (p in BKW_PARS) {
      gg <- grbkw(p, x)
      expect_true(all(is.finite(gg)),
                  info = paste("grbkw finite", nm, paste(p, collapse = ",")))
      expect_equal(gg, grgkw(c(p, 1), x)[1:4], tolerance = 1e-8,
                   info = paste("grbkw vs grgkw", nm, paste(p, collapse = ",")))
    }
    for (p in KKW_PARS) {
      kk <- grkkw(p, x)
      expect_true(all(is.finite(kk)),
                  info = paste("grkkw finite", nm, paste(p, collapse = ",")))
      expect_equal(kk, grgkw(c(p[1], p[2], 1, p[3], p[4]), x)[c(1, 2, 4, 5)],
                   tolerance = 1e-8,
                   info = paste("grkkw vs grgkw", nm, paste(p, collapse = ",")))
    }
  }
})


test_that("grbkw and grkkw agree with numDeriv away from the boundary", {
  # numDeriv is only a witness where a central difference is defined. delta = 0
  # is the edge of the parameter space, so the step to delta = -h returns +Inf
  # and the difference is NaN; and in the deep tail a subnormal intermediate
  # quantizes the step away entirely. Both are numDeriv's limits, not the
  # gradient's, so this block stays in the interior.
  skip_if_not_installed("numDeriv")
  x <- c(0.10, 0.25, 0.40, 0.72, 0.99)
  for (p in list(c(2, 3, 1.5, 0.5), c(5, 70, 1.5, 2), c(1, 300, 1.5, 2))) {
    expect_equal(grbkw(p, x),
                 numDeriv::grad(function(q) llbkw(q, x), p), tolerance = 1e-5,
                 info = paste("grbkw vs numDeriv", paste(p, collapse = ",")))
  }
  for (p in list(c(2, 3, 0.5, 1.5), c(5, 70, 2, 1.5), c(1, 300, 2, 1.5))) {
    expect_equal(grkkw(p, x),
                 numDeriv::grad(function(q) llkkw(q, x), p), tolerance = 1e-5,
                 info = paste("grkkw vs numDeriv", paste(p, collapse = ",")))
  }
})


test_that("hsbkw and hskkw stay finite and symmetric where the likelihood is", {
  # The same 0 * Inf arises in the Hessian.
  for (nm in names(DATA)) {
    x <- DATA[[nm]]
    for (p in BKW_PARS) {
      H <- hsbkw(p, x)
      expect_true(all(is.finite(H)),
                  info = paste("hsbkw finite", nm, paste(p, collapse = ",")))
      expect_identical(H, t(H),
                       info = paste("hsbkw symmetric", nm, paste(p, collapse = ",")))
    }
    for (p in KKW_PARS) {
      H <- hskkw(p, x)
      expect_true(all(is.finite(H)),
                  info = paste("hskkw finite", nm, paste(p, collapse = ",")))
      expect_identical(H, t(H),
                       info = paste("hskkw symmetric", nm, paste(p, collapse = ",")))
    }
  }
})


test_that("grkkw and hskkw refuse bad input the way grbkw and hsbkw do", {
  # A refused input warns, so the caller can tell a rejected call from a
  # genuine boundary.
  x <- c(0.2, 0.5, 0.8)
  expect_warning(g <- grkkw(c(-1, 3, 0.5, 1.5), x), "Invalid parameters in grkkw")
  expect_true(all(is.nan(g)))
  expect_warning(H <- hskkw(c(-1, 3, 0.5, 1.5), x), "Invalid parameters in hskkw")
  expect_true(all(is.nan(H)))

  # A NaN compares false against both bounds, so the support test must check
  # for it explicitly. The result is NaN throughout, never partly finite.
  xn <- c(0.2, NaN, 0.7)
  expect_warning(g <- grkkw(c(2, 3, 0.5, 1.5), xn), "strictly in \\(0,1\\)")
  expect_true(all(is.nan(g)))
  expect_warning(H <- hskkw(c(2, 3, 0.5, 1.5), xn), "strictly in \\(0,1\\)")
  expect_true(all(is.nan(H)))

  # the BKw pair, which is the behavior being matched
  expect_warning(gb <- grbkw(c(2, 3, 1.5, 0.5), xn), "strictly in \\(0,1\\)")
  expect_true(all(is.nan(gb)))
  expect_warning(Hb <- hsbkw(c(2, 3, 1.5, 0.5), xn), "strictly in \\(0,1\\)")
  expect_true(all(is.nan(Hb)))
})


test_that("grkkw and hskkw fail uniformly at a genuine boundary", {
  # alpha = beta = 1e-8 with lambda = 1e300 is past what the chain can carry.
  # A gradient with finite components next to an -Inf, or a Hessian with a few
  # NaN entries among finite ones, could be inverted for a standard error.
  x <- c(0.01, 0.3, 0.6, 0.9)
  p <- c(1e-8, 1e-8, 0, 1e300)          # KKw: (alpha, beta, delta, lambda)
  expect_warning(g <- grkkw(p, x), "non-finite values in grkkw")
  expect_true(all(is.nan(g)))
  expect_warning(H <- hskkw(p, x), "non-finite values in hskkw")
  expect_true(all(is.nan(H)))

  # the same distribution through BKw, which fails uniformly too
  pb <- c(1e-8, 1e-8, 1e300, 0)         # BKw: (alpha, beta, gamma, delta)
  expect_warning(gb <- grbkw(pb, x), "non-finite values in grbkw")
  expect_true(all(is.nan(gb)))
})


test_that("the likelihood surface has no plateau for an optimizer to sit on", {
  # alpha = 300 is past the underflow of x^alpha. L-BFGS-B cannot start from
  # an infinite objective ("L-BFGS-B needs finite values of 'fn'"), and a
  # method that tolerates the start would sit on a plateau.
  x <- X_PLATEAU
  expect_true(is.finite(llbkw(c(300, 2, 1.5, 1), x)))
  expect_true(is.finite(llkkw(c(300, 2, 1, 1.5), x)))

  fit <- optim(c(300, 2, 1.5, 1), llbkw, gr = grbkw, data = x,
               method = "L-BFGS-B",
               lower = c(1e-4, 1e-4, 1e-4, 0), upper = c(1e4, 1e4, 1e4, 1e4))
  expect_true(is.finite(fit$value))
  expect_true(fit$value < llbkw(c(300, 2, 1.5, 1), x))

  fitk <- optim(c(300, 2, 1, 1.5), llkkw, gr = grkkw, data = x,
                method = "L-BFGS-B",
                lower = c(1e-4, 1e-4, 0, 1e-4), upper = c(1e4, 1e4, 1e4, 1e4))
  expect_true(is.finite(fitk$value))
  expect_true(fitk$value < llkkw(c(300, 2, 1, 1.5), x))
})
