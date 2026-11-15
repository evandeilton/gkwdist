# tests/testthat/test-pbeta-log-deep-tail.R
# Log-scale distribution functions of the Beta-based families (Beta, Mc, BKw,
# GKw) in the deep tail, where log F is below about -450.
#
# References are the log of the regularized incomplete Beta function at the
# Beta argument of each family, evaluated with mpmath at 80 digits.

# x, gamma, delta, log I_x(gamma, delta + 1)
deep <- data.frame(
  x     = c(0.92, 0.90, 0.93, 0.80, 0.50, 0.50, 0.9280510927),
  gamma = c(1e4, 1e4, 1e4, 1e4, 2000, 1500, 1e4),
  delta = c(30, 30, 30, 30, 30, 38, 20),
  ref   = c(-707.854458077390, -920.956939025592, -603.745690230974,
            -2078.00815646744, -1253.47351384489, -890.610736375256,
            -657.402614815071)
)

test_that("pbeta_(log.p = TRUE) matches the reference in the deep lower tail", {
  for (i in seq_len(nrow(deep))) {
    val <- expect_no_warning(
      pbeta_(deep$x[i], deep$gamma[i], deep$delta[i], log.p = TRUE)
    )
    expect_true(is.finite(val))
    expect_equal(val, deep$ref[i], tolerance = 1e-10,
                 label = sprintf("pbeta_(%g, %g, %g)", deep$x[i],
                                 deep$gamma[i], deep$delta[i]))
  }
})

test_that("the deep upper tail mirrors the lower one", {
  # 1 - I_x(a, b) = I_{1-x}(b, a)
  val <- expect_no_warning(
    pbeta_(0.08, 31, 9999, lower.tail = FALSE, log.p = TRUE)
  )
  expect_equal(val, -707.854458077390, tolerance = 1e-10)
})

test_that("Mc, BKw and GKw match the reference in the deep tail", {
  # Lower tail: log I_y(gamma, delta + 1) at y = x^lambda (Mc) and
  # y = 1 - (1 - x^alpha)^beta (BKw)
  expect_equal(pmc(0.95, 1e4, 30, 1.5, log.p = TRUE),
               -645.751414438903, tolerance = 1e-10)
  expect_equal(pbkw(0.7544, 2, 3, 1e4, 30, log.p = TRUE),
               -707.817642810424, tolerance = 1e-10)
  # Upper tail: log I_{1-y}(delta + 1, gamma)
  expect_equal(pmc(0.3, 20, 5000, 1.5, lower.tail = FALSE, log.p = TRUE),
               -809.472943946186, tolerance = 1e-10)
  expect_equal(pgkw(0.3, 2, 3, 20, 5000, 1.4, lower.tail = FALSE, log.p = TRUE),
               -673.184043045263, tolerance = 1e-10)
})

test_that("Mc, BKw and GKw reduce to Beta in the deep tail", {
  for (i in seq_len(nrow(deep))) {
    x <- deep$x[i]; g <- deep$gamma[i]; d <- deep$delta[i]
    ref <- pbeta_(x, g, d, log.p = TRUE)
    expect_equal(pmc(x, g, d, 1, log.p = TRUE), ref, tolerance = 1e-12)
    expect_equal(pbkw(x, 1, 1, g, d, log.p = TRUE), ref, tolerance = 1e-12)
    expect_equal(pgkw(x, 1, 1, g, d, 1, log.p = TRUE), ref, tolerance = 1e-12)
  }
})

test_that("log F is monotone in x through the deep tail", {
  # Crosses the -450 level, where the evaluation method changes
  x <- seq(0.80, 0.97, length.out = 200)
  lp <- pbeta_(x, 1e4, 30, log.p = TRUE)
  expect_true(all(is.finite(lp)))
  expect_true(all(diff(lp) > 0))
})

test_that("ordinary log probabilities are stats::pbeta() unchanged", {
  grid <- expand.grid(
    q     = c(0.001, 0.05, 0.3, 0.5, 0.8, 0.99),
    gamma = c(0.3, 1, 5, 50, 400),
    delta = c(0, 0.5, 4, 30)
  )
  for (lt in c(TRUE, FALSE)) {
    ref <- stats::pbeta(grid$q, grid$gamma, grid$delta + 1,
                        lower.tail = lt, log.p = TRUE)
    val <- pbeta_(grid$q, grid$gamma, grid$delta, lower.tail = lt, log.p = TRUE)
    keep <- ref > -400
    expect_gt(sum(keep), 100)
    expect_identical(val[keep], ref[keep])
  }
})
