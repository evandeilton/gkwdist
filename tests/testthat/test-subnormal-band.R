# tests/testthat/test-subnormal-band.R
# Regression tests for the second round of log-space chain repairs.
#
# Every family walks some part of the chain
#
#   v = 1 - x^alpha     w = 1 - v^beta     z = 1 - w^lambda
#
# and the first repairs bridged a link only where its linear quantity had
# underflowed to exactly 0. Five defects survived that:
#
# 1. The band just before 0. For x^alpha in [5e-324, 2.2e-308) log(v) is a
#    subnormal with a few significant bits, and log1mexp() read it as exact.
#    llbkw(c(161.8, 2, 1.5, 1), c(.01,.3,.6,.9)) was 0.34 nats off and the
#    alpha component of the gradient dipped by 16% across the band.
# 2. EKw never received the bridge at all: dekw() returned -Inf, llekw() +Inf
#    and grekw() NaN/Inf where the nested GKw is finite.
# 3. The p/q/r functions of GKw, BKw, KKw, EKw, Kw and Mc flushed the lower
#    tail to exactly 0 once x^alpha (or w^lambda, or x^lambda) underflowed --
#    rgkw() drew thousands of exact zeros, which llgkw() then rejects -- and
#    the deep upper tail on the log scale to 1, or onto a plateau.
# 4. qgkw() and qmc() took log(y) of a y rounded to the double grid near 1, so
#    the upper tail was lost; qbkw() had always reflected.
# 5. hsgkw() still worked in linear space and returned NaN where every other
#    derivative is finite.
#
# References are independent of the code under test: a __float128 evaluation
# of the likelihood, integrate() of the density, round trips through the
# matching p function, the nesting identities, and numDeriv.
#
# Against commit 5330814 the blocks below fail 71 assertions and error once;
# with these fixes all 116 pass.

X_BAND <- c(0.01, 0.3, 0.6, 0.9)


test_that("BKw, KKw and GKw likelihoods are exact through the subnormal band", {
  # __float128 references
  expect_equal(llbkw(c(160, 2, 1.5, 1), X_BAND), 1505.907060458247, tolerance = 1e-12)
  expect_equal(llbkw(c(161, 2, 1.5, 1), X_BAND), 1515.520131938338, tolerance = 1e-12)
  expect_equal(llbkw(c(161.8, 2, 1.5, 1), X_BAND), 1523.210700324179, tolerance = 1e-12)
  expect_equal(llkkw(c(161, 2, 1, 1.5), X_BAND), 1516.412706057704, tolerance = 1e-12)
  expect_equal(llgkw(c(161.8, 2, 1.5, 1, 1), X_BAND), 1523.210700324179, tolerance = 1e-12)

  # One level down: log_w is the subnormal here, and a step of 1e-6 in lambda
  # moved the old objective by 0.35 nats around a smooth 1108.12903.
  x1 <- 0.93181528325658292
  expect_equal(llkkw(c(0.1, 150, 0.5, 0.5), x1), 1108.12903, tolerance = 1e-8)
  expect_equal(llkkw(c(0.1, 150, 0.5, 0.5 - 1e-6), x1), 1108.12903, tolerance = 1e-8)

  # Smooth across the band: second differences in alpha are tiny
  a <- seq(155, 168, by = 0.1)
  ll <- vapply(a, function(al) llbkw(c(al, 2, 1.5, 1), X_BAND), numeric(1))
  expect_lt(max(abs(diff(ll, differences = 2))), 1e-3)

  # and so is the gradient, which dipped by 16% before
  g <- vapply(c(160, 161, 161.7, 161.8, 161.85),
              function(al) grbkw(c(al, 2, 1.5, 1), 0.01)[1], numeric(1))
  expect_lt(diff(range(g)), 1e-3)
})


test_that("gradients and Hessians match numDeriv through the band", {
  skip_if_not_installed("numDeriv")
  fams <- list(
    list(ll = llgkw, gr = grgkw, hs = hsgkw, p = c(161.8, 2, 1.5, 1, 1)),
    list(ll = llbkw, gr = grbkw, hs = hsbkw, p = c(161.8, 2, 1.5, 1)),
    list(ll = llkkw, gr = grkkw, hs = hskkw, p = c(161.8, 2, 1, 1.5)),
    list(ll = llekw, gr = grekw, hs = hsekw, p = c(161.8, 2, 1.5))
  )
  for (f in fams) {
    g <- f$gr(f$p, X_BAND)
    gn <- numDeriv::grad(function(q) f$ll(q, X_BAND), f$p)
    expect_lt(max(abs(g - gn)) / max(1, abs(gn)), 1e-8)
    H <- f$hs(f$p, X_BAND)
    J <- numDeriv::jacobian(function(q) as.numeric(f$gr(q, X_BAND)), f$p)
    expect_lt(max(abs(H - J)) / max(1, abs(J)), 1e-7)
    expect_equal(H, t(H))
  }
})


test_that("EKw is exactly the nested GKw where x^alpha underflows", {
  expect_equal(dekw(1e-200, 4, 3, 0.2, log = TRUE),
               dgkw(1e-200, 4, 3, 1, 0, 0.2, log = TRUE), tolerance = 1e-13)
  expect_equal(dekw(1e-9, 40, 2, 0.02), dgkw(1e-9, 40, 2, 1, 0, 0.02), tolerance = 1e-13)
  x <- c(1e-9, 1e-12, 0.3, 0.5, 0.8, 0.95, 1 - 1e-12)
  expect_equal(llekw(c(40, 2, 0.3), x), llgkw(c(40, 2, 1, 0, 0.3), x), tolerance = 1e-13)
  for (a in c(162, 170, 200)) {
    expect_equal(llekw(c(a, 2, 1.5), X_BAND), llkkw(c(a, 2, 0, 1.5), X_BAND),
                 tolerance = 1e-13, info = paste("alpha =", a))
    expect_true(all(is.finite(grekw(c(a, 2, 1.5), X_BAND))))
    expect_true(all(is.finite(hsekw(c(a, 2, 1.5), X_BAND))))
  }
  # the optimiser that used to stop with "non-finite value supplied"
  fit <- optim(c(161, 2, 1.5), llekw, grekw, data = X_BAND,
               method = "L-BFGS-B", lower = 1e-3)
  expect_equal(fit$convergence, 0L)
})


test_that("the lower tail of p* is no longer flushed to 0", {
  tol <- 1e-6
  expect_equal(pgkw(1e-9, 40, 2, 0.05, 0.5, 0.1),
               integrate(function(t) dgkw(t, 40, 2, 0.05, 0.5, 0.1), 0, 1e-9,
                         rel.tol = 1e-12)$value, tolerance = tol)
  expect_equal(pbkw(1e-9, 40, 2, 0.01, 0.5),
               integrate(function(t) dbkw(t, 40, 2, 0.01, 0.5), 0, 1e-9,
                         rel.tol = 1e-12)$value, tolerance = tol)
  expect_equal(pekw(1e-9, 40, 2, 0.02),
               integrate(function(t) dekw(t, 40, 2, 0.02), 0, 1e-9,
                         rel.tol = 1e-12)$value, tolerance = tol)
  expect_equal(pkkw(1e-9, 40, 2, 0.5, 0.02),
               integrate(function(t) dkkw(t, 40, 2, 0.5, 0.02), 0, 1e-9,
                         rel.tol = 1e-12)$value, tolerance = tol)
  expect_equal(pmc(1e-10, 0.05, 0.5, 40),
               integrate(function(t) dmc(t, 0.05, 0.5, 40), 0, 1e-10,
                         rel.tol = 1e-12)$value, tolerance = tol)
  # F = 1 - (1 - x^a)^b ~ b x^a, exactly in log space
  expect_equal(pkw(1e-100, 4, 3, log.p = TRUE), log(3) + 4 * log(1e-100),
               tolerance = 1e-13)
})


test_that("q* inverts p* in the deep lower tail instead of returning 0", {
  rt <- list(
    function(p) pgkw(qgkw(p, 40, 2, 0.05, 0.5, 0.1), 40, 2, 0.05, 0.5, 0.1),
    function(p) pbkw(qbkw(p, 40, 2, 0.01, 0.5), 40, 2, 0.01, 0.5),
    function(p) pkkw(qkkw(p, 40, 2, 0.5, 0.02), 40, 2, 0.5, 0.02),
    function(p) pekw(qekw(p, 40, 2, 0.02), 40, 2, 0.02),
    function(p) pmc(qmc(p, 0.05, 0.5, 40), 0.05, 0.5, 40)
  )
  for (k in seq_along(rt)) for (p in c(1e-300, 1e-100, 1e-8, 0.01)) {
    expect_equal(rt[[k]](p), p, tolerance = 1e-8, info = paste("family", k, "p =", p))
  }
  expect_equal(pkw(qkw(1e-320, 100, 2), 100, 2), 1e-320, tolerance = 1e-2)
  # a quantile below the smallest double is 0, not a saturated 1e-308
  expect_identical(qbkw(log(1e-300), 1, 0.3, 0.5, 0, log.p = TRUE), 0)
})


test_that("r* no longer draws exact zeros from the lower tail", {
  set.seed(1)
  r <- rgkw(1e5, 40, 2, 0.05, 0.5, 0.1)
  expect_equal(sum(r == 0), 0)
  expect_true(is.finite(llgkw(c(40, 2, 0.05, 0.5, 0.1), r)))
  set.seed(1); expect_equal(sum(rekw(1e5, 40, 2, 0.01) == 0), 0)
  set.seed(1); expect_equal(sum(rkkw(1e5, 40, 2, 0.5, 0.01) == 0), 0)
  set.seed(1); expect_equal(sum(rbkw(1e5, 40, 2, 0.002, 0.5) == 0), 0)
})


test_that("qgkw and qmc keep the upper tail, as qbkw always did", {
  # 1 - x from the reflected Beta quantile, written out by hand
  a <- 2; b <- 3; g <- 1.5; d <- 0.5; l <- 1.2
  for (p in c(1e-18, 1e-22, 1e-26)) {
    omy <- qbeta(p, d + 1, g)
    omw <- -expm1(log1p(-omy) / l)
    ref <- -expm1(log1p(-exp(log(omw) / b)) / a)
    expect_equal(1 - qgkw(p, a, b, g, d, l, lower.tail = FALSE), ref,
                 tolerance = 1e-6, info = paste("p =", p))
  }
  expect_equal(qgkw(1e-50, 2, 3, 1.5, 1.2, 1, lower.tail = FALSE),
               qbkw(1e-50, 2, 3, 1.5, 1.2, lower.tail = FALSE), tolerance = 1e-14)
  expect_lt(qgkw(1e-50, 2, 3, 1.5, 1.2, 1, lower.tail = FALSE), 1)
  q <- qmc(1e-15, 2, 0.5, 1.2, lower.tail = FALSE)
  expect_equal(pmc(q, 2, 0.5, 1.2, lower.tail = FALSE), 1e-15, tolerance = 1e-6)
})


test_that("the deep upper tail on the log scale is neither flushed nor saturated", {
  # Kw(2, 100) nested in every family; qkw()/pkw() are closed form and exact.
  # Below log p ~ -718 the reflected quantity 1 - y is under DBL_MIN: qekw and
  # qkkw returned 1, qbkw and qgkw sat on a plateau at 1 - x = 4.16e-04, and
  # pbkw/pgkw returned -Inf.
  lp <- -c(600, 718, 750, 800, 1000, 3000)
  ref <- qkw(lp, 2, 100, lower.tail = FALSE, log.p = TRUE)
  q <- list(qgkw(lp, 2, 100, 1, 0, 1, lower.tail = FALSE, log.p = TRUE),
            qbkw(lp, 2, 100, 1, 0, lower.tail = FALSE, log.p = TRUE),
            qkkw(lp, 2, 100, 0, 1, lower.tail = FALSE, log.p = TRUE),
            qekw(lp, 2, 100, 1, lower.tail = FALSE, log.p = TRUE))
  for (k in seq_along(q)) expect_equal(1 - q[[k]], 1 - ref, tolerance = 1e-6,
                                       info = paste("family", k))
  p <- list(pgkw(ref, 2, 100, 1, 0, 1, lower.tail = FALSE, log.p = TRUE),
            pbkw(ref, 2, 100, 1, 0, lower.tail = FALSE, log.p = TRUE),
            pkkw(ref, 2, 100, 0, 1, lower.tail = FALSE, log.p = TRUE),
            pekw(ref, 2, 100, 1, lower.tail = FALSE, log.p = TRUE))
  for (k in seq_along(p)) expect_equal(p[[k]], pkw(ref, 2, 100, lower.tail = FALSE, log.p = TRUE),
                                       tolerance = 1e-10, info = paste("family", k))
})


test_that("the lower tail on the log scale is not rounded to 0 near 1", {
  # log F = log(1 - U) with U the (accurate) upper tail
  up <- pgkw(1 - 1e-6, 2, 3, 1.5, 2, 0.8, lower.tail = FALSE)
  expect_equal(pgkw(1 - 1e-6, 2, 3, 1.5, 2, 0.8, log.p = TRUE), log1p(-up),
               tolerance = 1e-10)
  expect_lt(pgkw(1 - 1e-6, 2, 3, 1.5, 2, 0.8, log.p = TRUE), 0)
  up <- pbkw(1 - 1e-6, 2, 3, 1.5, 2, lower.tail = FALSE)
  expect_equal(pbkw(1 - 1e-6, 2, 3, 1.5, 2, log.p = TRUE), log1p(-up),
               tolerance = 1e-10)
})


test_that("hsgkw at gamma = 1 is hskkw", {
  x <- c(0.10, 0.25, 0.40, 0.72, 0.99)
  H <- hsgkw(c(3, 2, 1, 0.7, 1.4), x)
  expect_equal(H[-3, -3], hskkw(c(3, 2, 0.7, 1.4), x), tolerance = 1e-12)
  H <- hsgkw(c(1, 200, 1, 2, 1), x)
  expect_true(all(is.finite(H)))
  expect_equal(H[-3, -3], hskkw(c(1, 200, 2, 1), x), tolerance = 1e-10)
})


test_that("missing data give the documented value in every family", {
  y <- c(0.2, NA, 0.6)
  ll <- list(llgkw(c(2, 3, 1.5, 2, 1.2), y), llbkw(c(2, 3, 1.5, 2), y),
             llkkw(c(2, 3, 2, 1.2), y), llekw(c(2, 3, 1.2), y),
             llmc(c(1.5, 2, 1.2), y), llkw(c(2, 3), y), llbeta(c(2, 3), y))
  for (v in ll) expect_identical(v, Inf)
  gr <- list(grgkw(c(2, 3, 1.5, 2, 1.2), y), suppressWarnings(grbkw(c(2, 3, 1.5, 2), y)),
             suppressWarnings(grkkw(c(2, 3, 2, 1.2), y)), grekw(c(2, 3, 1.2), y),
             grmc(c(1.5, 2, 1.2), y), grkw(c(2, 3), y), grbeta(c(2, 3), y))
  for (v in gr) expect_true(all(is.nan(v)))
  hs <- list(hsgkw(c(2, 3, 1.5, 2, 1.2), y), suppressWarnings(hsbkw(c(2, 3, 1.5, 2), y)),
             suppressWarnings(hskkw(c(2, 3, 2, 1.2), y)), hsekw(c(2, 3, 1.2), y),
             hsmc(c(1.5, 2, 1.2), y), hskw(c(2, 3), y), hsbeta(c(2, 3), y))
  for (v in hs) expect_true(all(is.nan(v)))
})


test_that("a density above DBL_MAX / 10 is returned, not Inf", {
  # dkw(x, a, 1) = a x^(a-1)
  expect_equal(dkw(5e-324, 0.045, 1), 2.5742349e307, tolerance = 1e-7)
  expect_true(is.finite(dkw(5e-324, 0.045, 1)))
})


test_that("a warning caught by a handler does not leak the C++ state", {
  skip_on_cran()
  skip_if_not(file.exists("/proc/self/status"))
  rss <- function() {
    s <- grep("VmRSS", readLines("/proc/self/status"), value = TRUE)
    as.numeric(sub("\\D+(\\d+).*", "\\1", s)) / 1024
  }
  x <- c(stats::runif(1e6, 0.1, 0.9), 2)
  for (i in 1:3) invisible(tryCatch(grbkw(c(1, 1, 1, 1), x), warning = function(w) NULL))
  invisible(gc()); r0 <- rss()
  # each leaked call held an 8 MB copy of the data
  for (i in 1:20) invisible(tryCatch(grbkw(c(1, 1, 1, 1), x), warning = function(w) NULL))
  invisible(gc())
  expect_lt(rss() - r0, 40)
  # the condition itself is unchanged
  w <- tryCatch(grbkw(c(1, 1, 1, 1), c(0.5, 2)), warning = function(w) w)
  expect_match(conditionMessage(w), "Data must be strictly in \\(0,1\\)")
  expect_identical(deparse(conditionCall(w)), "grbkw(c(1, 1, 1, 1), c(0.5, 2))")
  # A handler that leaves by an error takes the same longjmp as options(warn = 2),
  # which testthat's own warning handler would intercept here. The error must
  # arrive intact -- gkwgetstartvalues() has a catch (...) that used to be able
  # to swallow it.
  boom <- function(w) stop("boom: ", conditionMessage(w))
  expect_error(withCallingHandlers(grbkw(c(1, 1, 1, 1), c(0.5, 2)), warning = boom),
               "boom: Data must be strictly")
  expect_error(withCallingHandlers(gkwgetstartvalues(c(stats::runif(20), 5), "kw"),
                                   warning = boom),
               "boom: gkwgetstartvalues: 1 of 21")
})


test_that("the Beta starting point uses the package's Beta(gamma, delta + 1)", {
  set.seed(1)
  x <- rbeta(500, 4, 7)        # gamma = 4, delta = 6
  sv <- gkwgetstartvalues(x, "beta")
  m <- mean(x); v <- mean(x^2) - m^2; k <- m * (1 - m) / v - 1
  expect_equal(unname(sv), c(m * k, (1 - m) * k - 1), tolerance = 0.05)
})
