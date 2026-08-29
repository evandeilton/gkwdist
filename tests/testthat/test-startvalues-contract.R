# tests/testthat/test-startvalues-contract.R
# Regression tests for the three contracts gkwgetstartvalues() advertises:
# that n_starts buys a wider search, that the answer is deterministic, and that
# data outside the support is not altered in silence.
#
# 1. n_starts was inert. The selection loop decided whether to run Nelder-Mead
#    from a candidate by comparing the RAW objective at that starting point
#    against best_obj -- which after the first iteration already held an
#    OPTIMISED value. A merely poor start, exactly what multi-start exists to
#    rescue, was thrown away before the optimiser saw it, so only the first
#    candidate was ever optimised:
#
#      set.seed(202); x <- rgkw(500, 5, 1.2, 3, 0.5, 2)
#        n_starts =   1     objective 6.262579e-04
#        n_starts =  10     objective 6.262579e-04   bit-identical
#        n_starts = 200     objective 6.262579e-04   bit-identical
#
#    The cost was not just a suboptimal start. The single optimised path could
#    walk into a corner where the numerical integral of the density underflows,
#    moment_theoretical() falls back to its closed-form Kumaraswamy moment, and
#    the optimiser is rewarded for parameters whose real moments are nothing
#    like the sample's -- a method-of-moments estimate that misses the first
#    moment by five orders of magnitude:
#
#      set.seed(3050); x <- rgkw(50, 2, 3, 1.5, 2, 0.8)   sample mean 0.296299
#        alpha 0.100000 (pinned at the lower bound), beta 9.874503,
#        gamma 0.574504, delta 0.131984, lambda 1.276909
#        implied mean 1.394060e-06
#
#    Starting optim() from that corner, 8 of 10 GKw samples of size 500
#    converged to a POSITIVE negative log-likelihood, around +4100 to +4900
#    where the correct region is near -300.
#
# 2. The starting-point grid is drawn from a generator with a fixed internal
#    seed. That is deliberate -- the output seeds optimisers elsewhere, so two
#    calls on the same data have to agree -- but nothing said so, and the
#    determinism only became a meaningful claim once the draws were actually
#    used for something.
#
# 3. Observations outside (0,1) were clamped into [1e-10, 1-1e-10] in silence.
#    Truncation moves every sample moment and therefore every estimate:
#
#      set.seed(1); y <- rkw(300, 2, 3)          alpha       beta
#        gkwgetstartvalues(y, "kw")            2.137549   3.506511
#        gkwgetstartvalues(c(y, 5, -3), "kw")  2.047567   3.251306   silent
#        gkwgetstartvalues(y * 100, "kw")     50.000000  50.000000   silent
#
#    The last row is the defect entire: a sample handed over on a 0-100 scale
#    came back with both parameters pinned at the upper edge of the parameter
#    box, indistinguishable from a fit.
#
# Against 1.1.5 every test_that() block below fails.

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Mirror of moment_theoretical() / objective_function() in src/gkwinit.cpp:
# Simpson's rule on 51 points over [0,1], weighted squared relative errors
# against the sample moments of orders 1..5.
.sv_to_gkw <- function(theta, family) {
  switch(family,
    gkw  = theta,
    bkw  = c(theta[1], theta[2], theta[3], theta[4], 1),
    kkw  = c(theta[1], theta[2], 1, theta[3], theta[4]),
    ekw  = c(theta[1], theta[2], 1, 0, theta[3]),
    mc   = c(1, 1, theta[1], theta[2], theta[3]),
    kw   = c(theta[1], theta[2], 1, 0, 1),
    beta = c(1, 1, theta[1], theta[2], 1)
  )
}

.sv_moment <- function(theta, family, r, np = 51L) {
  p <- .sv_to_gkw(theta, family)
  xs <- seq(0, 1, length.out = np)
  w <- rep(2, np)
  w[seq(2, np - 1, by = 2)] <- 4
  w[c(1, np)] <- 1
  fx <- suppressWarnings(dgkw(xs, p[1], p[2], p[3], p[4], p[5]))
  fx[!is.finite(fx) | xs <= 0 | xs >= 1] <- 0
  (1 / (np - 1) / 3) * sum(w * xs^r * fx)
}

.sv_objective <- function(theta, family, sample_moments) {
  wt <- c(1.0, 0.8, 0.6, 0.4, 0.2)
  th <- vapply(1:5, function(r) .sv_moment(theta, family, r), numeric(1))
  sum(wt * ((th - sample_moments) / sample_moments)^2)
}

.sv_sample_moments <- function(x) vapply(1:5, function(r) mean(x^r), numeric(1))

# First moment implied by a parameter vector, via E[X] = integral of Q(p) dp.
# Deliberately independent of the density quadrature the estimator itself uses.
.sv_implied_mean <- function(theta, family, np = 2000L) {
  pr <- (seq_len(np) - 0.5) / np
  q <- suppressWarnings(switch(family,
    gkw  = qgkw(pr, theta[1], theta[2], theta[3], theta[4], theta[5]),
    bkw  = qbkw(pr, theta[1], theta[2], theta[3], theta[4]),
    kkw  = qkkw(pr, theta[1], theta[2], theta[3], theta[4]),
    ekw  = qekw(pr, theta[1], theta[2], theta[3]),
    mc   = qmc(pr, theta[1], theta[2], theta[3]),
    kw   = qkw(pr, theta[1], theta[2]),
    beta = qbeta_(pr, theta[1], theta[2])
  ))
  mean(q[is.finite(q)])
}

# One multi-modal sample, fitted once at three values of n_starts and reused by
# the two tests below; each GKw fit at n_starts = 40 costs a couple of seconds.
.sv_ns_grid <- c(1L, 10L, 40L)
.sv_mm_data <- local({ set.seed(202); rgkw(500, 5, 1.2, 3, 0.5, 2) })
.sv_mm_fits <- lapply(.sv_ns_grid,
                      function(k) gkwgetstartvalues(.sv_mm_data, "gkw", n_starts = k))

# ---------------------------------------------------------------------------
# 1. n_starts drives a real multi-start
# ---------------------------------------------------------------------------

test_that("n_starts is not inert: more starting points change the answer", {
  # Against 1.1.5 all three of these are bit-identical, because Nelder-Mead ran
  # exactly once no matter what n_starts was set to.
  expect_false(identical(.sv_mm_fits[[1]], .sv_mm_fits[[3]]))
  expect_false(identical(.sv_mm_fits[[1]], .sv_mm_fits[[2]]))

  # Whatever changes, the result is still a valid GKw parameter vector.
  for (p in .sv_mm_fits) {
    expect_length(p, 5)
    expect_equal(names(p), c("alpha", "beta", "gamma", "delta", "lambda"))
    expect_true(all(is.finite(p)))
    expect_true(all(p > 0))
  }
})

test_that("more starting points never return a worse moment fit", {
  sm <- .sv_sample_moments(.sv_mm_data)
  obj <- vapply(.sv_mm_fits, .sv_objective, numeric(1),
                family = "gkw", sample_moments = sm)

  # Non-increasing: the candidate set grows with n_starts and the best is kept.
  # A small tolerance absorbs the difference between this quadrature and the
  # fallbacks the C++ objective can take.
  for (k in seq_len(length(obj) - 1L)) {
    expect_lte(obj[k + 1L], obj[k] * (1 + 1e-6))
  }

  # And on this sample the extra starts genuinely buy something. This is the
  # assertion 1.1.5 cannot satisfy: there, obj is one constant repeated.
  expect_lt(obj[length(obj)], obj[1] * 0.5)
})

test_that("the returned parameters reproduce the first sample moment", {
  # A method-of-moments estimate that misses the mean is not an estimate. On
  # 1.1.5 each of these returns a degenerate corner whose implied mean is
  # ~1e-6 against sample means near 0.3, i.e. a ratio of about 5e-06.
  cases <- list(c(3050, 50), c(6050, 50), c(1200, 200),
                c(5200, 200), c(6200, 200), c(8200, 200))
  for (cs in cases) {
    set.seed(cs[1])
    x <- rgkw(cs[2], 2, 3, 1.5, 2, 0.8)
    p <- gkwgetstartvalues(x, family = "gkw")
    ratio <- .sv_implied_mean(as.numeric(p), "gkw") / mean(x)
    expect_equal(ratio, 1, tolerance = 0.05,
                 info = paste("seed", cs[1], "n", cs[2],
                              "params", paste(signif(p, 7), collapse = " ")))
  }
})

# ---------------------------------------------------------------------------
# 2. Determinism
# ---------------------------------------------------------------------------

test_that("starting values ignore the seed in force and do not consume it", {
  x <- .sv_mm_data

  # n_starts = 25 puts 21 drawn points into the grid, so this is a regime where
  # the draws demonstrably affect the answer -- which is what makes the
  # seed-independence below a claim about anything. On 1.1.5 the drawn points
  # are never used and this expectation fails.
  wide <- gkwgetstartvalues(x, "gkw", n_starts = 25L)
  expect_false(identical(wide, .sv_mm_fits[[1]]))

  set.seed(1)
  a <- gkwgetstartvalues(x, "gkw", n_starts = 25L)
  set.seed(9999)
  b <- gkwgetstartvalues(x, "gkw", n_starts = 25L)
  expect_identical(a, b)
  expect_identical(a, wide)

  # R's stream is neither read nor advanced: the caller's next draw is the one
  # they would have got without the call.
  set.seed(4321)
  before <- .Random.seed
  u_direct <- runif(1)

  set.seed(4321)
  invisible(gkwgetstartvalues(x, "gkw", n_starts = 25L))
  expect_identical(.Random.seed, before)
  expect_identical(runif(1), u_direct)
})

test_that("the determinism contract is stated on the help page", {
  db <- tools::Rd_db("gkwdist", lib.loc = dirname(find.package("gkwdist")))
  expect_true("gkwgetstartvalues.Rd" %in% names(db))
  txt <- paste(
    capture.output(tools::Rd2txt(db[["gkwgetstartvalues.Rd"]])),
    collapse = " ")

  # 1.1.5 documents "multiple random starting points" and stops there, leaving
  # a reader to assume a seed control that does not exist.
  expect_match(txt, "[Dd]eterminis")
  expect_match(txt, "set\\.seed")
  expect_match(txt, "\\.Random\\.seed")
  # And the truncation of out-of-support data is now advertised as a warning.
  expect_match(txt, "warning")
})

# ---------------------------------------------------------------------------
# 3. Out-of-support data is not truncated in silence
# ---------------------------------------------------------------------------

test_that("observations outside (0,1) raise a warning naming how many", {
  set.seed(1)
  y <- rkw(300, 2, 3)

  # Wrong scale: the signature case. 1.1.5 returns (50, 50) -- both parameters
  # pinned at the upper edge of the box -- and says nothing.
  expect_warning(gkwgetstartvalues(y * 100, family = "kw"),
                 "300 of 300 observations lie outside the open interval")

  # A couple of stray values, e.g. a 0-100 column merged into a proportion one.
  expect_warning(gkwgetstartvalues(c(y, 5, -3), family = "kw"),
                 "2 of 302 observations lie outside the open interval")

  # The closed boundary is outside the open support too, matching what ll*()
  # enforces.
  expect_warning(gkwgetstartvalues(c(y, 0), family = "kw"),
                 "1 of 301 observations lie outside the open interval")
  expect_warning(gkwgetstartvalues(c(y, 1), family = "kw"),
                 "1 of 301 observations lie outside the open interval")

  # The observed range is reported so the scale is visible in the message.
  expect_warning(gkwgetstartvalues(c(y, 5, -3), family = "kw"),
                 "observed range \\[-3, 5\\]")

  # Every family reports it, not just the two-parameter ones.
  for (fam in c("gkw", "bkw", "kkw", "ekw", "mc", "kw", "beta")) {
    expect_warning(gkwgetstartvalues(c(y, 5, -3), family = fam),
                   "outside the open interval",
                   info = fam)
  }
})

test_that("clean, NA and non-finite data do not raise the truncation warning", {
  set.seed(1)
  y <- rkw(300, 2, 3)

  # No false positives: nothing here leaves the open interval. expect_silent()
  # rather than expect_no_warning() so the file stays within the testthat
  # version DESCRIPTION declares.
  expect_silent(gkwgetstartvalues(y, family = "kw"))
  expect_silent(gkwgetstartvalues(c(y, NA), family = "kw"))
  expect_silent(gkwgetstartvalues(c(y, Inf, -Inf), family = "kw"))
  expect_silent(gkwgetstartvalues(y, family = "gkw"))

  # Warning or not, the return contract is unchanged: a named, finite,
  # strictly positive vector of the right length. On 1.1.5 the contaminated
  # call is silent, so this block fails there on the expect_warning() alone.
  expect_warning(gkwgetstartvalues(c(y, 5, -3), family = "kw"),
                 "outside the open interval")
  clamped <- suppressWarnings(gkwgetstartvalues(c(y, 5, -3), family = "kw"))
  expect_length(clamped, 2)
  expect_equal(names(clamped), c("alpha", "beta"))
  expect_true(all(is.finite(clamped)) && all(clamped > 0))

  # And the clamp still does what it always did: dropping the two offenders by
  # hand gives the same answer as letting the function clamp them.
  expect_equal(clamped,
               suppressWarnings(gkwgetstartvalues(c(y, 1 - 1e-10, 1e-10),
                                                  family = "kw")))
})
