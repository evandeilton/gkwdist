## Submission

Patch release, 1.1.5 -> 1.1.7. The API is unchanged. The changes since the
CRAN version are listed in NEWS.md (sections 1.1.6 and 1.1.7).

This release fixes numerical defects in the C++ code:

* The log-space chain now handles the subnormal range, where the
  log-likelihoods, gradients and Hessians were silently wrong.
* The tails of the p, q and r functions are no longer flushed to 0 or 1, and
  `qgkw()`/`qmc()` keep their upper tail.
* `hsgkw()` is computed in log space and no longer returns NaN where the
  likelihood is finite.
* A memory leak when a warning raised from C++ was caught is fixed.

Analytic gradients and Hessians were checked against `numDeriv` for all seven
families.

## Test environments

* Local: Ubuntu 26.04 LTS, R 4.6.1, GCC 15.2.0
* GitHub Actions: ubuntu-latest (devel, release, oldrel-1), windows-latest
  (release), macos-latest (release)

## R CMD check results

0 errors | 0 warnings | 1 note

* The note is local to the checking machine: HTML validation and math
  rendering were skipped because `tidy` and the `V8` package are not installed.

## Reverse dependencies

One reverse dependency, gkwreg (same maintainer). R CMD check of gkwreg 2.1.18
against this version: no new problems.
