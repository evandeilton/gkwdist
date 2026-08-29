# tests/testthat/test-return-contract.R
# Regression tests for audit finding A-10 (the @return contract) and for the
# out-of-support warning in ll*().
#
# A-10. The @return blocks of the 28 d/p/q/r functions promised "NaN if
# parameters are invalid" while the R wrappers stop(). The audit asked for one
# contract chosen and propagated. stop() is the one kept: it is what the package
# has shipped for its whole CRAN life, a named error is a better diagnostic than
# a NaN for the case that actually occurs -- a mistyped parameter -- and
# switching to NaN would break every tryCatch(..., error = ) guard written
# against it, which is a larger break than the one being fixed.
#
# gr*() and hs*() genuinely DO return NaN for an invalid parameter, so their 14
# @return blocks are deliberately unchanged, and that asymmetry is pinned here.
#
# Known and deliberately NOT changed here: an infinite parameter is not
# intercepted by the wrapper. `any(alpha <= 0)` is FALSE for Inf and anyNA() does
# not catch it, so it reaches C++, where check_*_pars() rejects it and the
# routine leaves its fill value. Rejecting it at the R layer is correct in
# principle but is a behaviour change that breaks gkwreg::predict(), which clamps
# only the lower bound of its linear predictor and so can hand an Inf through on
# extrapolated newdata. That call currently returns 0, which is the correct limit
# -- Kw(alpha, beta) concentrates at 1 as alpha grows -- and one such row would
# abort the whole prediction vector. The gkwreg clamp is the prerequisite. The
# current behaviour is pinned below so the eventual change is visible.

FAMS <- list(
  list(d = dgkw, p = pgkw, q = qgkw, r = rgkw, ll = llgkw, ok = list(2, 3, 1.5, 0.5, 2),
       bounds = list(alpha = 1, beta = 2, gamma = 3, delta = 4, lambda = 5), nonneg = "delta"),
  list(d = dbkw, p = pbkw, q = qbkw, r = rbkw, ll = llbkw, ok = list(2, 3, 1.5, 0.5),
       bounds = list(alpha = 1, beta = 2, gamma = 3, delta = 4), nonneg = "delta"),
  list(d = dkkw, p = pkkw, q = qkkw, r = rkkw, ll = llkkw, ok = list(2, 3, 0.5, 2),
       bounds = list(alpha = 1, beta = 2, delta = 3, lambda = 4), nonneg = "delta"),
  list(d = dekw, p = pekw, q = qekw, r = rekw, ll = llekw, ok = list(2, 3, 2),
       bounds = list(alpha = 1, beta = 2, lambda = 3), nonneg = ""),
  list(d = dmc, p = pmc, q = qmc, r = rmc, ll = llmc, ok = list(1.5, 0.5, 2),
       bounds = list(gamma = 1, delta = 2, lambda = 3), nonneg = "delta"),
  list(d = dkw, p = pkw, q = qkw, r = rkw, ll = llkw, ok = list(2, 3),
       bounds = list(alpha = 1, beta = 2), nonneg = ""),
  list(d = dbeta_, p = pbeta_, q = qbeta_, r = rbeta_, ll = llbeta, ok = list(1.5, 0.5),
       bounds = list(gamma = 1, delta = 2), nonneg = "delta")
)

msg_for <- function(par, nonneg) {
  sprintf("'%s' must be %s", par,
          if (identical(par, nonneg)) "non-negative" else "positive")
}

test_that("an out-of-bound parameter is an error in every position", {
  for (f in FAMS) {
    for (par in names(f$bounds)) {
      bad <- if (identical(par, f$nonneg)) -1 else 0   # first illegal value
      p <- f$ok; p[[f$bounds[[par]]]] <- bad
      m <- msg_for(par, f$nonneg)
      expect_error(do.call(f$d, c(list(0.5), p)), m, info = par)
      expect_error(do.call(f$p, c(list(0.5), p)), m, info = par)
      expect_error(do.call(f$q, c(list(0.5), p)), m, info = par)
      expect_error(do.call(f$r, c(list(2), p)), m, info = par)
    }
  }
})

test_that("out-of-bound and missing take the same route, in every position", {
  for (f in FAMS) {
    for (par in names(f$bounds)) {
      lo <- if (identical(par, f$nonneg)) -1 else 0
      msgs <- vapply(list(lo, -1, NA_real_, NaN), function(bad) {
        p <- f$ok; p[[f$bounds[[par]]]] <- bad
        tryCatch({ do.call(f$d, c(list(0.5), p)); "no error" },
                 error = function(e) conditionMessage(e))
      }, character(1))
      expect_true(all(msgs == msgs[1]),
                  info = paste(par, paste(unique(msgs), collapse = " | ")))
      # ekw and beta_ append the bound -- "(alpha > 0)" -- where the other five
      # stop at "positive". Match the common prefix, which is what the R-level
      # contract actually guarantees.
      expect_match(msgs[1], msg_for(par, f$nonneg), fixed = TRUE, info = par)
    }
  }
})

test_that("delta = 0 is still accepted, in every family that has it", {
  # delta's bound is >= 0, not > 0 -- the one place an off-by-one could hide.
  for (f in FAMS) {
    if (!nzchar(f$nonneg)) next
    p <- f$ok; p[[f$bounds[[f$nonneg]]]] <- 0
    expect_silent(do.call(f$d, c(list(0.5), p)))
    expect_true(is.finite(do.call(f$d, c(list(0.5), p))))
    expect_silent(do.call(f$p, c(list(0.5), p)))
    expect_silent(do.call(f$q, c(list(0.5), p)))
  }
})

test_that("an infinite parameter is NOT intercepted by the wrapper", {
  # Pinning the current state, not endorsing it. See the header: the wrapper
  # lets Inf through, C++ rejects it, and the routine leaves its fill value.
  # When the gkwreg clamp lands and this is tightened, these expectations are
  # the ones to flip.
  expect_identical(suppressWarnings(dkw(0.5, Inf, 3)), 0)
  expect_identical(suppressWarnings(dbeta_(0.5, Inf, 3)), 0)
  expect_true(is.na(suppressWarnings(pkw(0.5, Inf, 3))))
  expect_true(is.na(suppressWarnings(qkw(0.5, Inf, 3))))
  # and 0 is the correct limit here, which is why this is not urgent
  expect_equal(dkw(0.5, 1e5, 3), 0)
})

test_that("gr and hs still return NaN, as their documentation says", {
  x <- c(0.2, 0.5, 0.8)
  expect_true(all(is.nan(grkw(c(-1, 3), x))))
  expect_true(all(is.nan(hskw(c(-1, 3), x))))
  expect_true(all(is.nan(grbeta(c(-1, 3), x))))
  expect_true(all(is.nan(grgkw(c(-1, 3, 1, 0, 1), x))))
})

test_that("every family warns about data outside the open support", {
  # llekw and llbeta already did; the other five were silent, so the same
  # corrupted sample was announced in two families out of seven.
  for (f in FAMS) {
    expect_warning(do.call(f$ll, list(unlist(f$ok), c(0.5, 1.5))), "outside \\(0, 1\\)")
    expect_warning(do.call(f$ll, list(unlist(f$ok), c(0.5, 0))), "outside \\(0, 1\\)")
  }
})

test_that("valid input is silent and unchanged", {
  for (f in FAMS) {
    expect_silent(do.call(f$d, c(list(0.5), f$ok)))
    expect_silent(do.call(f$ll, list(unlist(f$ok), c(0.2, 0.5, 0.8))))
  }
  expect_equal(dkw(0.5, 2, 3), 1.6875)
  expect_equal(llkw(c(2, 3), c(0.2, 0.5, 0.8)), -0.1492391, tolerance = 1e-6)
})
