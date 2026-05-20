# CRAN Submission Comments — gkwdist 1.1.3

## Summary of Changes

This patch release fixes critical bugs that affected MLE reliability, standard error
computation, and distribution-family initial value generation. It also corrects a
systematic numerical error in `safe_exp()` and removes arbitrary clamping that
distorted gradients/Hessians for large parameter values.

Full details are in `NEWS.md`.

## Test Environments

* Local: Ubuntu 24.04.2 LTS, R 4.5.2, GCC 13.3.0, x86_64
* R CMD check run locally with `--as-cran`

## R CMD CHECK Results

```
Status: OK
0 errors | 0 warnings | 0 notes
```

## Downstream Dependencies

None. `gkwdist` has no reverse dependencies on CRAN.

## Key Bug Fixes (for CRAN reviewer context)

1. `llgkw()` returned `R_NegInf` for invalid parameters — optimizers treated −∞ as
   the global minimum. Fixed to `R_PosInf`.

2. `gkwinit.cpp` rejected `delta = 0` and mapped EKw/Kw to `delta = 1` (wrong);
   these sub-families require `delta = 0`.

3. `hsbkw()` computed `v^(β−1) = 1` for β < 1 due to a branch error, producing
   a wrong Hessian and therefore wrong standard errors for β ∈ (0, 1).

4. `safe_exp()` returned `10 × exp(x)` in its moderate-underflow branch — a 10×
   systematic scaling error.
