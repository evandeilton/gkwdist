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

The only note is local to the checking machine: HTML validation was skipped
because the `tidy` command is not installed, and math rendering was skipped
because the `V8` package is unavailable. Neither applies to the package itself.

## Installed size

The check reports the installed size as INFO on this machine and may report it
as a NOTE on yours:

```
* checking installed package size ... INFO
  installed size is 10.5Mb
  sub-directories of 1Mb or more:
    libs   9.1Mb
```

Almost none of that is code. Of the 9,525,168 bytes of `libs/gkwdist.so`,
`.text` accounts for 391,164 -- about 4% -- and the eight `.debug_*` sections
account for 8,951,368, or 94.5%. The size is debug information the toolchain
emits because R compiles packages with `-g -O2`, not the compiled routines for
the seven distribution families.

The package does not try to remove it, for three reasons.

`-g0` in `src/Makevars` cannot work. R appends its own `$(CXXFLAGS)` after
`$(PKG_CXXFLAGS)`, so the later flag wins:

```
g++ ... -fopenmp -g0 -fpic  -g -O2  -c gkw.cpp -o gkw.o
              PKG_CXXFLAGS ^^^     ^^ R's CXXFLAGS
```

Measured: identical output, 9,525,200 bytes against 9,525,168.

A post-link `strip --strip-debug` rule in `src/Makevars` does work -- it brings
the shared object to 572,816 bytes and the installed size to 1.5Mb, and
`R CMD check --as-cran` then reports `checking installed package size ... OK`
with no complaint from any of the Makevars checks. It is not used because
`--strip-debug` is GNU binutils syntax that the `strip` on macOS does not
accept, and a failing recipe aborts `make` and takes the whole installation
with it. Trading a size NOTE for a broken macOS build is not a good trade.

Stripping also destroys the sanitizer diagnostics the package now relies on.
`.github/workflows/sanitizers.yaml` runs the test suite under UBSan and ASan,
and those reports are only actionable with debug information present:

```
gkw.cpp:134:21: runtime error: division by zero
    #0 ... in dgkw(...)     src/gkw.cpp:134
```

Anyone who wants the smaller installation can ask for it at install time,
which is where the choice belongs: `R CMD INSTALL --strip` gives 516,712 bytes
and a 1.5Mb installed size, with the full test suite passing (0 failures,
9,644 assertions).

## Downstream dependencies

`gkwreg` (>= 2.0.0) imports `gkwdist`. Because this release changes the values
returned by `dgkw()` and by the EKw, KKw and BKw likelihood routines, a reverse
dependency check was run against `gkwreg` 2.1.14 with this version installed.
Its examples and its `testthat` suite both pass. The check reports one warning,
for a vignette that requires the suggested package `betareg`, which was not
installed in the checking environment; it is unrelated to this release.
