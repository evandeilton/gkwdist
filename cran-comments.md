## Summary of Changes

This is a patch release (1.1.3 → 1.1.4) fixing the CRAN check ERRORs reported on
2026-05-25. No functional changes; all user-facing code is unchanged.

### Fix

* `tests/testthat/test-mle-performance.R`: Added `skip_on_cran()` to all
  timing-based benchmark tests. These tests compare wall-clock run times of
  analytical vs. numerical gradient optimisation and are inherently sensitive to
  machine load. On CRAN's Linux check hosts the baseline ("no gradient") run
  happened to be anomalously fast, causing the 2× ratio tolerance to be breached
  intermittently. The tests remain intact for local development.

## Test environments

* Local: Zorin OS 18.1 (Ubuntu 24.04 base), R 4.5.2, GCC 13.3.0, x86_64-linux

## R CMD check results

```
0 errors | 0 warnings | 0 notes
```

## Downstream dependencies

None. `gkwdist` has no reverse dependencies on CRAN.
