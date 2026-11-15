# =============================================================================
# Test Suite: Argument Contract
# =============================================================================
# The documented stop() conditions of the R wrappers and the handling of
# non-finite input (NA_real_, NaN, +Inf, -Inf), with stats::dbeta(), pbeta()
# and qbeta() as the reference for the d/p/q convention.
#
#   1. ERROR CONTRACT. Every documented stop() in every d/p/q/r/ll/gr/hs of all
#      seven families.
#
#   2. STRUCTURAL INVARIANTS under non-finite input: length, type, and the
#      absence of an error.
#
#   3. NON-FINITE CASES AT THE ENDS OF THE SUPPORT: d(+-Inf) = 0, p(-Inf) = 0,
#      q(NA) = NA and q(NaN) = NaN.
#
#   4. PROPAGATION OF NON-FINITE INPUT, and the guards on logical flags and NA
#      parameters:
#
#        call             result
#        d*(NA_real_)     NA
#        d*(NaN)          NaN
#        p*(NA_real_)     NA
#        p*(NaN)          NaN
#        p*(+Inf)         1
#        q*(+-Inf)        NaN, with a warning
#        q*(p < 0)        NaN, with a warning
#        q*(p > 1)        NaN, with a warning
#
#   5. THE SHAPE OF THE RESULT: dim, dimnames and names of the first argument.
#
#   6. SAMPLE SIZE RANGE, LOG-PROBABILITIES ABOVE 0, WARNING CALLS.
# =============================================================================

# -----------------------------------------------------------------------------
# Family table. `dpq` is the suffix of d*/p*/q*/r*; `llgrhs` is the suffix of
# ll*/gr*/hs*, which differs for Beta (dbeta_ but llbeta).
# -----------------------------------------------------------------------------
FAMILIES <- list(
  list(
    dpq = "gkw", llgrhs = "gkw",
    pars = list(alpha = 2, beta = 3, gamma = 1.5, delta = 0.5, lambda = 2),
    bounds = c(alpha = "pos", beta = "pos", gamma = "pos", delta = "nonneg", lambda = "pos")
  ),
  list(
    dpq = "bkw", llgrhs = "bkw",
    pars = list(alpha = 2, beta = 3, gamma = 1.5, delta = 0.5),
    bounds = c(alpha = "pos", beta = "pos", gamma = "pos", delta = "nonneg")
  ),
  list(
    dpq = "kkw", llgrhs = "kkw",
    pars = list(alpha = 2, beta = 3, delta = 0.5, lambda = 2),
    bounds = c(alpha = "pos", beta = "pos", delta = "nonneg", lambda = "pos")
  ),
  list(
    dpq = "ekw", llgrhs = "ekw",
    pars = list(alpha = 2, beta = 3, lambda = 2),
    bounds = c(alpha = "pos", beta = "pos", lambda = "pos")
  ),
  list(
    dpq = "mc", llgrhs = "mc",
    pars = list(gamma = 1.5, delta = 0.5, lambda = 2),
    bounds = c(gamma = "pos", delta = "nonneg", lambda = "pos")
  ),
  list(
    dpq = "kw", llgrhs = "kw",
    pars = list(alpha = 2, beta = 3),
    bounds = c(alpha = "pos", beta = "pos")
  ),
  list(
    dpq = "beta_", llgrhs = "beta",
    pars = list(gamma = 1.5, delta = 0.5),
    bounds = c(gamma = "pos", delta = "nonneg")
  )
)

bound_msg <- function(par, kind) {
  sprintf("'%s' must be %s", par, if (kind == "pos") "positive" else "non-negative")
}

# The value that violates each bound.
bad_value <- function(kind) if (kind == "pos") 0 else -1

call_fn <- function(prefix, fam, args) do.call(match.fun(paste0(prefix, fam)), args)

NONFINITE <- c(NA_real_, NaN, Inf, -Inf)

# =============================================================================
# SECTION 1 -- THE DOCUMENTED stop() CONDITIONS
# =============================================================================

test_that("d*/p*/q* reject a non-numeric first argument", {
  for (f in FAMILIES) {
    expect_error(call_fn("d", f$dpq, c(list("a"), f$pars)), "'x' must be numeric")
    expect_error(call_fn("p", f$dpq, c(list("a"), f$pars)), "'q' must be numeric")
    expect_error(call_fn("q", f$dpq, c(list("a"), f$pars)), "'p' must be numeric")
    # is.numeric(TRUE) is FALSE, so a logical is rejected too.
    expect_error(call_fn("d", f$dpq, c(list(TRUE), f$pars)), "'x' must be numeric")
  }
})

test_that("d*/p*/q*/r* reject every out-of-bound shape parameter", {
  for (f in FAMILIES) {
    for (par in names(f$bounds)) {
      kind <- f$bounds[[par]]
      bad <- f$pars
      bad[[par]] <- bad_value(kind)
      msg <- bound_msg(par, kind)
      info <- paste(f$dpq, par, kind)

      expect_error(call_fn("d", f$dpq, c(list(0.5), bad)), msg, info = info)
      expect_error(call_fn("p", f$dpq, c(list(0.5), bad)), msg, info = info)
      expect_error(call_fn("q", f$dpq, c(list(0.5), bad)), msg, info = info)
      expect_error(call_fn("r", f$dpq, c(list(5), bad)), msg, info = info)
    }
  }
})

test_that("a valid boundary parameter is accepted where the bound allows it", {
  # delta >= 0 means delta = 0 must NOT error: it is the Beta(gamma, 1) edge.
  for (f in FAMILIES) {
    if (!"delta" %in% names(f$bounds)) next
    ok <- f$pars
    ok[["delta"]] <- 0
    expect_no_error(call_fn("d", f$dpq, c(list(0.5), ok)))
    expect_no_error(call_fn("p", f$dpq, c(list(0.5), ok)))
  }
})

test_that("the logical flags reject anything that is not a single logical", {
  for (f in FAMILIES) {
    for (bad in list(c(TRUE, TRUE), logical(0), "yes", 1)) {
      expect_error(
        call_fn("d", f$dpq, c(list(0.5), f$pars, list(log = bad))),
        "'log' must be a single logical value",
        info = paste(f$dpq, "log")
      )
      expect_error(
        call_fn("p", f$dpq, c(list(0.5), f$pars, list(lower.tail = bad))),
        "'lower.tail' must be a single logical value",
        info = paste(f$dpq, "lower.tail")
      )
      expect_error(
        call_fn("p", f$dpq, c(list(0.5), f$pars, list(log.p = bad))),
        "'log.p' must be a single logical value",
        info = paste(f$dpq, "log.p")
      )
      expect_error(
        call_fn("q", f$dpq, c(list(0.5), f$pars, list(lower.tail = bad))),
        "'lower.tail' must be a single logical value",
        info = paste(f$dpq, "q lower.tail")
      )
      expect_error(
        call_fn("q", f$dpq, c(list(0.5), f$pars, list(log.p = bad))),
        "'log.p' must be a single logical value",
        info = paste(f$dpq, "q log.p")
      )
    }
  }
})

test_that("r* rejects a sample size that is not a non-negative integer", {
  # n = 0 is legal and tested just below: stats::rbeta(0, 2, 3) is numeric(0)
  for (f in FAMILIES) {
    for (bad in list(-1, NA, "5")) {
      expect_error(
        call_fn("r", f$dpq, c(list(bad), f$pars)),
        "'n' must be a single non-negative integer",
        info = paste(f$dpq, "n =", format(bad))
      )
    }
  }
})

test_that("r*(0) returns numeric(0), as base R does", {
  expect_length(stats::rbeta(0, 2, 3), 0)
  for (f in FAMILIES) {
    v <- call_fn("r", f$dpq, c(list(0), f$pars))
    expect_length(v, 0)
    expect_type(v, "double")
  }
})

test_that("r* follows the base-R convention that length(n) > 1 means n <- length(n)", {
  set.seed(11)
  for (f in FAMILIES) {
    expect_length(call_fn("r", f$dpq, c(list(c(1, 2, 3)), f$pars)), 3)
  }
})

test_that("ll*/gr*/hs* reject a par vector of the wrong length", {
  for (f in FAMILIES) {
    n <- length(f$pars)
    msg <- sprintf("'par' must be a numeric vector of length %d", n)
    for (pre in c("ll", "gr", "hs")) {
      expect_error(
        call_fn(pre, f$llgrhs, list(rep(1, n - 1), c(0.3, 0.5))), msg,
        info = paste(pre, f$llgrhs, "short")
      )
      expect_error(
        call_fn(pre, f$llgrhs, list(rep(1, n + 1), c(0.3, 0.5))), msg,
        info = paste(pre, f$llgrhs, "long")
      )
    }
  }
})

test_that("ll*/gr*/hs* reject non-numeric and empty data", {
  for (f in FAMILIES) {
    ok <- rep(1, length(f$pars))
    for (pre in c("ll", "gr", "hs")) {
      expect_error(
        call_fn(pre, f$llgrhs, list(ok, "a")), "'data' must be numeric",
        info = paste(pre, f$llgrhs)
      )
      expect_error(
        call_fn(pre, f$llgrhs, list(ok, numeric(0))),
        "'data' must have at least one observation",
        info = paste(pre, f$llgrhs)
      )
    }
  }
})

# =============================================================================
# SECTION 2 -- STRUCTURAL INVARIANTS UNDER NON-FINITE INPUT
#
# These hold whatever the values are. They rule out a crash, an error, or a
# result of the wrong length silently misaligning a vectorized call.
# =============================================================================

test_that("d*/p*/q* accept non-finite input without erroring", {
  for (f in FAMILIES) {
    expect_no_error(call_fn("d", f$dpq, c(list(NONFINITE), f$pars)))
    expect_no_error(call_fn("p", f$dpq, c(list(NONFINITE), f$pars)))
    expect_no_error(suppressWarnings(call_fn("q", f$dpq, c(list(NONFINITE), f$pars))))
  }
})

test_that("d*/p*/q* return one double per input element for non-finite input", {
  for (f in FAMILIES) {
    for (pre in c("d", "p")) {
      v <- call_fn(pre, f$dpq, c(list(NONFINITE), f$pars))
      expect_length(v, length(NONFINITE))
      expect_type(v, "double")
    }
    v <- suppressWarnings(call_fn("q", f$dpq, c(list(NONFINITE), f$pars)))
    expect_length(v, length(NONFINITE))
    expect_type(v, "double")
  }
})

test_that("a non-finite element does not disturb its finite neighbors", {
  # The vectorized call must give each finite element the same answer it gets
  # on its own, whatever the non-finite elements evaluate to.
  x <- c(0.25, NA_real_, 0.5, NaN, 0.75, Inf, -Inf)
  finite <- c(1, 3, 5)
  for (f in FAMILIES) {
    for (pre in c("d", "p")) {
      mixed <- call_fn(pre, f$dpq, c(list(x), f$pars))
      alone <- call_fn(pre, f$dpq, c(list(x[finite]), f$pars))
      expect_equal(mixed[finite], alone, info = paste(pre, f$dpq))
    }
    mixed <- suppressWarnings(call_fn("q", f$dpq, c(list(x), f$pars)))
    alone <- call_fn("q", f$dpq, c(list(x[finite]), f$pars))
    expect_equal(mixed[finite], alone, info = paste("q", f$dpq))
  }
})

# =============================================================================
# SECTION 3 -- NON-FINITE CASES AT THE ENDS OF THE SUPPORT
# =============================================================================

test_that("d*(+-Inf) is 0, as in base R", {
  # dbeta(Inf, 1.5, 0.5) and dbeta(-Inf, 1.5, 0.5) are both 0.
  expect_equal(dbeta(c(Inf, -Inf), 1.5, 0.5), c(0, 0))
  for (f in FAMILIES) {
    expect_equal(
      call_fn("d", f$dpq, c(list(c(Inf, -Inf)), f$pars)), c(0, 0),
      info = f$dpq
    )
  }
})

test_that("p*(-Inf) is 0, as in base R", {
  expect_equal(pbeta(-Inf, 1.5, 0.5), 0)
  for (f in FAMILIES) {
    expect_equal(call_fn("p", f$dpq, c(list(-Inf), f$pars)), 0, info = f$dpq)
  }
})

test_that("q* propagates NA and NaN, as in base R", {
  expect_identical(qbeta(NA_real_, 1.5, 0.5), NA_real_)
  expect_identical(qbeta(NaN, 1.5, 0.5), NaN)
  for (f in FAMILIES) {
    v <- call_fn("q", f$dpq, c(list(c(NA_real_, NaN)), f$pars))
    expect_true(is.na(v[1]), info = paste(f$dpq, "q(NA) is NA"))
    expect_true(is.nan(v[2]), info = paste(f$dpq, "q(NaN) is NaN"))
  }
})

test_that("q* warns for probabilities outside [0, 1]", {
  # The warning itself is correct and is pinned here. What it promises -- NaN --
  # is section 4's business.
  for (f in FAMILIES) {
    expect_warning(
      call_fn("q", f$dpq, c(list(-0.5), f$pars)),
      "'p' values outside [0, 1] will produce NaN",
      fixed = TRUE, info = f$dpq
    )
    expect_warning(
      call_fn("q", f$dpq, c(list(1.5), f$pars)),
      "'p' values outside [0, 1] will produce NaN",
      fixed = TRUE, info = f$dpq
    )
  }
})

test_that("q* does not warn at the closed boundary p = 0 and p = 1", {
  for (f in FAMILIES) {
    expect_silent(call_fn("q", f$dpq, c(list(c(0, 1)), f$pars)))
  }
})

# =============================================================================
# SECTION 4 -- NON-FINITE INPUT PROPAGATES, AS IN BASE R
#
# d*(NA) is NA and d*(NaN) is NaN, p*(+Inf) is 1, and a probability outside
# [0, 1] gives NaN rather than saturating at a bound of the support.
# =============================================================================

test_that("d* propagates NA and NaN like base R", {
  # base R: dbeta(NA) is NA, dbeta(NaN) is NaN.
  for (f in FAMILIES) {
    v <- call_fn("d", f$dpq, c(list(c(NA_real_, NaN)), f$pars))
    expect_identical(v[1], NA_real_, info = paste(f$dpq, "d(NA)"))
    expect_identical(v[2], NaN, info = paste(f$dpq, "d(NaN)"))
  }
})

test_that("p* propagates NA and NaN like base R", {
  # base R: pbeta(NA) is NA, pbeta(NaN) is NaN.
  for (f in FAMILIES) {
    v <- call_fn("p", f$dpq, c(list(c(NA_real_, NaN)), f$pars))
    expect_identical(v[1], NA_real_, info = paste(f$dpq, "p(NA)"))
    expect_identical(v[2], NaN, info = paste(f$dpq, "p(NaN)"))
  }
})

test_that("p*(+Inf) is 1, like base R", {
  # The support is (0, 1), so all the mass lies below +Inf and the CDF there
  # is 1: pbeta(Inf, 1.5, 0.5) is 1.
  expect_equal(pbeta(Inf, 1.5, 0.5), 1)
  for (f in FAMILIES) {
    expect_equal(call_fn("p", f$dpq, c(list(Inf), f$pars)), 1, info = f$dpq)
  }
})

test_that("q* returns NaN for probabilities outside [0, 1], as it warns it will", {
  # base R: qbeta() of -0.5, 1.5, Inf and -Inf is NaN, each with a warning.
  expect_true(all(is.nan(suppressWarnings(qbeta(c(-0.5, 1.5, Inf, -Inf), 1.5, 0.5)))))
  for (f in FAMILIES) {
    v <- suppressWarnings(
      call_fn("q", f$dpq, c(list(c(-0.5, 1.5, Inf, -Inf)), f$pars))
    )
    expect_true(all(is.nan(v)), info = f$dpq)
  }
})

test_that("the logical flags reject NA", {
  # is.logical(NA) is TRUE and length(NA) is 1, so the guard must test is.na()
  # too: an NA flag would reach C++ as NA_LOGICAL and be read as TRUE.
  for (f in FAMILIES) {
    expect_error(
      call_fn("d", f$dpq, c(list(0.5), f$pars, list(log = NA))),
      "'log' must be a single logical value", info = f$dpq
    )
    expect_error(
      call_fn("p", f$dpq, c(list(0.5), f$pars, list(lower.tail = NA))),
      "'lower.tail' must be a single logical value", info = f$dpq
    )
    expect_error(
      call_fn("p", f$dpq, c(list(0.5), f$pars, list(log.p = NA))),
      "'log.p' must be a single logical value", info = f$dpq
    )
  }
})

test_that("an NA shape parameter raises the package's own error in every family", {
  # The documented message, not R's "missing value where TRUE/FALSE needed"
  # from `if (any(par <= 0))`, and not a silent 0 from `na.rm = TRUE`.
  for (f in FAMILIES) {
    for (par in names(f$bounds)) {
      bad <- f$pars
      bad[[par]] <- NA_real_
      expect_error(
        call_fn("d", f$dpq, c(list(0.5), bad)),
        bound_msg(par, f$bounds[[par]]),
        info = paste(f$dpq, par)
      )
    }
  }
})

# =============================================================================
# SECTION 5 -- THE SHAPE OF THE RESULT
#
# base R's d/p/q carry the first argument's attributes through to the output,
# so code that indexes or plots by shape keeps working. The copy is conditional
# on the lengths agreeing, which is what base R does once a recycled parameter
# makes the output longer than the input.
# =============================================================================

test_that("d, p and q carry dim, dimnames and names, as base R does", {
  m <- matrix(c(.1, .3, .5, .7), 2, 2,
              dimnames = list(c("r1", "r2"), c("c1", "c2")))
  v <- c(a = .2, b = .5)

  expect_identical(dim(stats::dbeta(m, 2, 3)), c(2L, 2L))   # the reference
  expect_identical(names(stats::dbeta(v, 2, 3)), c("a", "b"))

  for (f in FAMILIES) {
    for (pref in c("d", "p", "q")) {
      r <- call_fn(pref, f$dpq, c(list(m), f$pars))
      expect_identical(dim(r), c(2L, 2L), info = paste(pref, f$dpq))
      expect_identical(dimnames(r), dimnames(m), info = paste(pref, f$dpq))

      r <- call_fn(pref, f$dpq, c(list(v), f$pars))
      expect_identical(names(r), c("a", "b"), info = paste(pref, f$dpq))
    }
  }
})

test_that("the shape is dropped when a parameter recycles longer, as base R does", {
  m <- matrix(c(.1, .3, .5, .7), 2, 2)
  expect_null(dim(stats::dbeta(m, c(2, 3, 4, 5, 6), 3)))     # the reference
  for (f in FAMILIES) {
    long <- f$pars
    long[[1]] <- rep(long[[1]], length.out = 5)
    expect_null(dim(call_fn("d", f$dpq, c(list(m), long))), info = f$dpq)
  }
})

# =============================================================================
# SECTION 6 -- SAMPLE SIZE RANGE, LOG-PROBABILITIES ABOVE 0, WARNING CALLS
# =============================================================================

test_that("r* rejects a size that does not fit an integer", {
  for (f in FAMILIES) {
    for (bad in list(Inf, 2^31, 3e9)) {
      # The first condition signaled is the error: no coercion warning before it
      cond <- tryCatch(call_fn("r", f$dpq, c(list(bad), f$pars)),
                       condition = identity)
      expect_s3_class(cond, "error")
      expect_match(conditionMessage(cond),
                   "'n' must be a single non-negative integer",
                   info = paste(f$dpq, "n =", format(bad)))
    }
  }
})

test_that("q*(log.p = TRUE) warns for a log-probability above 0", {
  for (f in FAMILIES) {
    expect_warning(
      r <- call_fn("q", f$dpq, c(list(c(-1, 0, 1)), f$pars, list(log.p = TRUE))),
      "'p' values above 0 will produce NaN",
      info = f$dpq
    )
    expect_true(is.nan(r[3]), info = f$dpq)
    expect_true(all(is.finite(r[1:2])), info = f$dpq)
    # Valid log-probabilities stay silent
    expect_no_warning(
      call_fn("q", f$dpq, c(list(c(-Inf, -1, 0, NA)), f$pars, list(log.p = TRUE)))
    )
  }
})

test_that("a warning raised in C++ reports the user-facing call", {
  for (fn in c("dgkw", "pgkw", "qgkw")) {
    w <- tryCatch(do.call(fn, list(0.5, Inf)), warning = identity)
    expect_s3_class(w, "warning")
    expect_identical(as.character(conditionCall(w)[[1]]), fn)
  }
})
