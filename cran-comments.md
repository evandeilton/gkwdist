## Summary of Changes

This is a patch release (1.1.4 -> 1.1.5) correcting six numerical defects. The
user-facing API is unchanged; no functions were added, removed or renamed.

### Corrections

* `dgkw()` returned 0 for every input. The internal helpers `log1mexp()` and
  `log1pexp()` collide with functions of the same name in R's public `Rmath.h`
  API, which use the convention `log(1 - exp(-x))` for `x >= 0`. Where the
  `Rmath.h` macro was active the calls resolved to R's version, which returns
  `NaN` for the negative arguments used here, so every density evaluation failed
  its finiteness guard and fell through to the initialised value. The helpers are
  renamed `gkw_log1mexp()` and `gkw_log1pexp()`.

* The log-likelihoods of the EKw, KKw and BKw sub-families clamped the
  intermediate quantities `1 - x^alpha` and `1 - (1 - x^alpha)^beta` at `1e-10`
  rather than working in log space. For small `x` and moderate `alpha` these
  round to 1 and 0 respectively in double precision, so the clamp substituted
  `log(1e-10)` for values near `-53`. Deviations reached 6,100 log-units, which
  affects AIC, BIC and likelihood ratio tests. The three families now use the
  same log-space kernel as the parent GKw routines.

* Mixed second derivatives were zeroed at degenerate parameter values in
  `hsbkw()`, `hsekw()` and `hskkw()`; gradient terms were clamped at 1000 in
  `grkkw()` and `llkkw()`; `grkkw()` and `hskkw()` skipped a required block at
  `delta = 0`; and `check_beta_pars()` rejected `delta = 0`, which is the valid
  `Beta(gamma, 1)` boundary in this parameterisation.

### Validation added

Two test files were added. Every gradient component and every Hessian entry of
all seven sub-families is now compared individually against two independent
references: the general GKw routines restricted to the constrained parameter
point, and `numDeriv` Richardson extrapolation. Grids cover the degenerate values
`gamma = 1`, `beta = 1`, `lambda = 1` and `delta = 0` and samples with
observations near zero. Densities are checked to integrate to one and to match
base R for the Beta and closed-form Kumaraswamy cases.

One timing-based test in `test-mle-performance.R` was also made robust. It
compared wall-clock times of fits that take a few milliseconds, where the ratio
is dominated by call overhead rather than computation, and failed intermittently
when run with `NOT_CRAN` set. It remains guarded by `skip_on_cran()`.

## Test environments

* Local: Ubuntu 26.04 LTS, R 4.6.1, GCC/g++ 15.2.0, x86_64-linux

## R CMD check results

```
0 errors | 0 warnings | 1 note
```

The note reports the installed size (libs ~9.6Mb), which is inherent to the
compiled C++ routines for seven distribution families.

## Downstream dependencies

`gkwreg` (>= 2.0.0) imports `gkwdist`. Because this release changes the values
returned by `dgkw()` and by the EKw, KKw and BKw likelihood routines, a reverse
dependency check was run against `gkwreg` 2.1.14 with this version installed.
Its examples and its `testthat` suite both pass. The check reports one warning,
for a vignette that requires the suggested package `betareg`, which was not
installed in the checking environment; it is unrelated to this release.
