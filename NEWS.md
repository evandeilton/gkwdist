# gkwdist 1.1.6

## Critical Bug Fixes

* **Zero-length arguments crashed the R process** (all seven families): the
  vectorised `d*()`, `p*()`, `q*()` and `r*()` routines size their output as the
  maximum length of their inputs and then recycle with `i % vec.n_elem`. When one
  argument had length zero while another did not, the output length stayed at one
  or more and the recycling evaluated `i % 0`. Integer division by zero is
  undefined behaviour; on x86-64 it raises `SIGFPE`, terminating the R process
  with no error, no message and nothing for `tryCatch()` to catch. The R-level
  validation did not intercept it either, because `any(numeric(0) <= 0)` is
  `FALSE`. All 28 exported routines now short-circuit before the loop:
  `d*()`, `p*()` and `q*()` return `numeric(0)`, matching
  `stats::dbeta(numeric(0), 1, 1)`, and `r*()` return `n` missing values with a
  warning, matching `stats::rbeta(3, numeric(0), 1)`. A filtered vector that
  happened to be empty, such as `dkw(x[x > 1], 2, 3)`, was enough to trigger the
  crash. Numerical output is unchanged for every non-empty input.

## Documentation Fixes

* **`grmc()` gradient formula had inverted digamma signs** (`R/bpmc.R`): the
  `@details` block documented `psi(gamma + delta + 1) - psi(gamma)` for the gamma
  component and `psi(gamma + delta + 1) - psi(delta + 1)` for delta. Since
  `d log B(gamma, delta+1) / d gamma = psi(gamma) - psi(gamma + delta + 1)`, both
  signs were reversed, and the documented formula disagreed with the returned
  value by two orders of magnitude. `R/beta.R` documented the opposite sign for
  the same quantity. The implementation was correct throughout; only the
  documentation is changed. The same block's Hessian entry for
  `d2l/dgamma ddelta` in `hsmc()` had the same sign reversal.

* **README example 6 inverted the sign of the observed information matrix**
  (`README.Rmd`): `hsekw()` already returns the Hessian of the negative
  log-likelihood, so negating it again produced a negative definite matrix and
  printed `NaN` for every asymptotic standard error. Example 3 of the same README
  and the vignettes were already correct.

* **GKw attribution corrected** (`R/gkw.R`): the main help page credited Cordeiro
  & de Castro (2011), which introduces the Kw-G family, rather than Carrasco,
  Ferrari & Cordeiro (2010), which introduces the five-parameter generalized
  Kumaraswamy distribution implemented here. `DESCRIPTION`, the README and the
  vignettes already cited the latter.

## Testing

* New `tests/testthat/test-zero-length-input.R` covers all 28 routines with
  zero-length data and zero-length parameters, the empty-subset idiom, and the
  correspondence with the `stats` package's convention.

# gkwdist 1.1.5

## Critical Bug Fixes

* **`dgkw()` returned zero for every input** (`gkw.cpp`, `utils.h`): the package's
  numerical helpers `log1mexp()` and `log1pexp()` collide with functions of the same
  name in R's public `Rmath.h` API, which use the opposite convention
  (`log(1 - exp(-x))` for `x >= 0`). In translation units where `Rmath.h`'s macro was
  active, calls bound to R's version, which returns `NaN` for the negative arguments
  used here; every density evaluation then failed its finiteness guard and returned 0.
  The helpers are now named `gkw_log1mexp()` and `gkw_log1pexp()`. The sub-family
  densities were unaffected, as was `llgkw()`, which routes through `vec_log1mexp()`.

* **Log-likelihoods of EKw, KKw and BKw were wrong for data near zero**
  (`ekw.cpp`, `kkw.cpp`, `bkw.cpp`): these routines clamped `v = 1 - x^alpha` and
  `w = 1 - v^beta` at `1e-10` instead of working in log space. For small `x` and
  moderate `alpha`, `x^alpha` rounds to 1 in double precision and `w` collapses to
  zero, so the clamp replaced `log(w) = -53` by `log(1e-10) = -23`. Deviations
  reached 6,100 log-units, which silently corrupts AIC, BIC and likelihood ratio
  tests. All three families now use the same `gkw_log1mexp()` formulation as
  `gkw.cpp`, and their scores and Hessians are expressed as ratios of logarithms.

* **Mixed second derivatives were zeroed at degenerate parameter values**
  (`bkw.cpp`, `ekw.cpp`, `kkw.cpp`): guards of the form `if (abs(p - 1) > eps)`
  gated mixed partial derivatives that do not carry the vanishing factor. Because
  `d2l/dalpha dgamma` is obtained by differentiating `(gamma-1)*log(w)` once in
  `gamma`, the `(gamma-1)` factor is consumed and the term survives at `gamma = 1`.
  `hsbkw()` returned 0 where the correct value was 271.12; `hsekw()` and `hskkw()`
  had the same defect at `beta = 1`.

* **`grkkw()` clamped gradient terms at 1000** (`kkw.cpp`): arbitrary
  `std::min(..., 1000.0)` caps distorted the score in `beta` by up to 5%. The same
  clamp appeared as `effective_delta` inside `llkkw()`, capping the likelihood for
  `delta > 1000`. This is the defect removed from `ekw.cpp` in 1.1.3, which had
  survived here.

* **`grkkw()` and `hskkw()` skipped the `z` block at `delta = 0`** (`kkw.cpp`):
  the shortcut omitted `sum(log(z))` from `dl/ddelta` and zeroed
  `d2l/dalpha ddelta`, `d2l/dbeta ddelta` and `d2l/ddelta dlambda`, none of which
  carry a `delta` factor. `delta = 0` is a valid interior value of the likelihood.

* **The Beta sub-family rejected `delta = 0`** (`utils.h`): `check_beta_pars()`
  required `delta > 0`, unlike the other five validators. Since the sub-family is
  parameterised as `Beta(gamma, delta + 1)`, `delta = 0` is the legitimate
  `Beta(gamma, 1)` boundary; `dbeta_()`, `pbeta_()`, `qbeta_()`, `rbeta_()`,
  `llbeta()`, `grbeta()` and `hsbeta()` all returned `NA`/`Inf` there.

## Validation

* **`test-boundary-derivatives.R`** (new): every gradient component and every
  Hessian entry of all seven sub-families is now compared individually against two
  independent references, the general GKw routines restricted to the constrained
  parameter point and `numDeriv` Richardson extrapolation, over grids that include
  the degenerate values `gamma = 1`, `beta = 1`, `lambda = 1` and `delta = 0` and
  samples containing observations near zero.

* **`test-density-correctness.R`** (new): densities are checked to integrate to one,
  to agree with the general GKw density at the constrained parameter point, to match
  base R for the Beta and closed-form Kumaraswamy cases, and to be the derivative of
  the corresponding distribution function. The previous PDF tests asserted only type,
  length, non-negativity and finiteness, all of which a vector of zeros satisfies.

## Accuracy after the fixes

Componentwise maximum relative error over 720 parameter configurations per family,
against the general GKw routines and against `numDeriv`:

| Family | log-likelihood | gradient | Hessian |
|:-------|---------------:|---------:|--------:|
| GKw    | 1.5e-11 | 1.5e-08 | 1.5e-07 |
| BKw    | 1.6e-12 | 3.7e-08 | 3.6e-08 |
| KKw    | 3.1e-13 | 3.5e-08 | 4.9e-08 |
| EKw    | 5.4e-14 | 7.8e-09 | 2.7e-08 |
| Mc     | 3.5e-13 | 2.5e-09 | 9.5e-10 |
| Kw     | 4.1e-15 | 7.2e-10 | 1.2e-09 |
| Beta   | 7.2e-15 | 4.6e-10 | 8.9e-11 |

The residual gradient and Hessian errors are at the accuracy limit of Richardson
extrapolation itself; against the GKw reference all seven families agree to 1e-13.

## Documentation and Project Infrastructure

* **`inst/paper/`**: the JOSS manuscript was rewritten. It now positions the
  package explicitly as the distribution layer of the GKw ecosystem, records that
  the split from `gkwreg` was made at the request of JOSS reviewers during that
  package's review, reports the measured validation and timing results in place of
  the previous unverified figures, and follows the current JOSS AI disclosure
  policy. The bibliography was expanded to 19 entries with DOIs verified against
  Crossref.

* **`CONTRIBUTING.md`** and **`CODE_OF_CONDUCT.md`** (new): contribution workflow,
  support expectations, governance, and the testing standard numerical
  contributions are held to. The contributing guide documents the cross-check that
  makes bug reports actionable: every sub-family routine must agree with the
  general GKw routine evaluated at the corresponding constrained parameter point.

* **`README`**: the claim that the C++ routines are "10-50x faster than equivalent
  R implementations" was replaced with measured figures. The original benchmark
  compared `-sum(log(dkw(x, 2, 3)))` against `llkw()`, which is C++ against C++
  plus R loop overhead, and gives roughly 3x. The genuine gain is in the
  derivatives: the analytical score is about 9x faster than Richardson
  extrapolation and the analytical Hessian about 38x faster at n = 20,000.

* **`inst/CITATION`**: updated to version 1.1.5 and pointed at the CRAN canonical
  URL; it had been stale at version 1.0.8.

* Test coverage rose from 70.8% to 74.2% of combined R and C++ lines. The largest
  single gain is in `src/gkw.cpp` (42.9% to 71.5%), which reflects that `dgkw()`
  now executes its density computation instead of falling through to its
  finiteness guard.

# gkwdist 1.1.4

## CRAN Fix

* **`test-mle-performance.R`**: Added `skip_on_cran()` to all timing-based benchmark
  tests. These tests compare wall-clock times of analytical vs. numerical gradients and
  are inherently unreliable on shared/loaded CRAN check machines, causing spurious
  `ERROR` results. The tests remain available for local development.

# gkwdist 1.1.3

## Bug Fixes

* **`llgkw()` invalid parameter return** (`gkw.cpp`): Fixed critical error where the
  negative log-likelihood returned `R_NegInf` (−∞) for invalid parameters instead of
  `R_PosInf` (+∞). Gradient-based MLE optimizers interpret −∞ as a global minimum,
  causing them to converge to the invalid boundary rather than the true MLE.

* **`gkwinit.cpp` — delta validation** (`gkwinit.cpp`): Fixed internal `gkw_pdf()`
  rejecting `delta = 0` (a valid GKw parameter value) due to a strict `delta <= 0`
  check that should have been `delta < 0`.

* **`gkwinit.cpp` — EKw/Kw sub-family PDF mapping** (`gkwinit.cpp`): Fixed
  `ekw_pdf()` and `kw_pdf()` passing `delta = 1` instead of the correct `delta = 0`
  when delegating to `gkw_pdf()`. EKw and Kw are GKw sub-families with `delta = 0`,
  not `delta = 1`. This produced wrong starting values for MLE of these families.

* **`hsbkw()` — v^(β−1) computation** (`bkw.cpp`): Fixed the Hessian of the BKw
  negative log-likelihood returning a wrong value for β < 1. The ternary expression
  `(beta > 1.0) ? v_beta/v : 1.0` coincidentally produces the correct result for
  β = 1 but is wrong for all 0 < β < 1. Replaced with the exact formula
  `safe_exp((beta - 1.0) * ln_v)`.

## Numerical Stability

* **`safe_exp()` underflow scaling** (`utils.h`): Fixed a systematic 10× error in
  the moderate-underflow branch. The previous implementation used
  `DBL_MIN_SAFE * exp(x − log(DBL_MIN))` where `DBL_MIN_SAFE = 10 * DBL_MIN`,
  yielding `10 * exp(x)` instead of `exp(x)`. The fix uses
  `DBL_MIN * exp(x − log(DBL_MIN)) = exp(x)` exactly.

* **`dgkw()` silent boundary truncation removed** (`gkw.cpp`): Removed a block that
  silently skipped data points within `SQRT_EPSILON^(1/α)` of 0 or 1, returning
  density 0 for those points without warning. The log-space computation handles
  near-boundary values correctly without this truncation.

* **`llekw()` / `grekw()` — lambda clamping removed** (`ekw.cpp`): Removed the
  arbitrary cap `lambda_factor = min(lambda_factor, 1000)` applied to gradient and
  Hessian terms when λ > 1000. This distorted optimization for large-λ scenarios and
  produced incorrect standard errors.

## Code Quality

* **`gkwinit.cpp`**: Removed `using namespace Rcpp;` at file scope; replaced with
  explicit `Rcpp::` qualifications. Added NA/NaN filtering before moment computation
  to prevent silent corruption when input data contains missing values.

* **`bkw.cpp`**: Removed spurious `try/catch` blocks wrapping `Rcpp::as<arma::vec>()`
  conversions in `grbkw()` and `hsbkw()`. These conversions cannot throw in this
  context and the silent fallback masked type errors.

* **`gkw.cpp` / `ekw.cpp`**: Refactored Hessian accumulation to build only the upper
  triangle inside the observation loop and symmetrize once afterwards with
  `arma::symmatu()`, eliminating O(n × p²) redundant assignments.

* **`utils.h` — `vec_safe_pow()` UB guard**: Added guard preventing undefined
  behaviour when casting large `y_rounded` values (> `INT_MAX`) to `int` for
  odd-exponent sign detection.

* **`utils.h` — `vec_safe_pow()` SIMD fast path**: Added an early-return path
  `arma::exp(y * arma::log(x))` for the common case (y > 0, all x > 0) that
  is fully auto-vectorizable, improving throughput in gradient/Hessian evaluation.

# gkwdist 1.1.2

## Code Cleanup and Testing Enhancement

### C++ Code Cleanup

* **Removed legacy commented code**: Cleaned up all C++ source files (`gkw.cpp`, `bkw.cpp`, `kkw.cpp`, `ekw.cpp`, `kw.cpp`, `bpmc.cpp`, `beta_.cpp`) by removing old commented-out implementations that were kept for reference.
* **Code formatting**: Improved R wrapper formatting with consistent indentation and alignment in `.Call()` invocations and roxygen examples.

### New Test Suites

* **Analytical derivatives validation** (`test-derivatives-validation.R`):
  - 70 comprehensive tests validating gradient (`gr*`) and Hessian (`hs*`) functions
  - Compares analytical derivatives against numerical differentiation via `numDeriv`
  - Covers all 7 subfamilies: GKw, BKw, KKw, EKw, Mc, Kw, Beta
  - Multiple parameter configurations per subfamily for robustness

* **MLE performance benchmarks** (`test-mle-performance.R`):
  - Compares optimization efficiency across three scenarios: numerical-only, analytical gradient, and analytical gradient + Hessian
  - Validates that analytical derivatives provide equivalent or better accuracy
  - Tests convergence rates and computational time across all distribution families

### JOSS Paper

* **Added paper for JOSS submission** (`inst/paper/`):
  - Complete manuscript describing the package's statistical framework
  - Comprehensive bibliography with foundational references
  - Compiled PDF ready for submission

---

# gkwdist 1.1.1

## Major Refactoring Release

This release represents a comprehensive refactoring of the entire package codebase, focusing on numerical stability, code consistency, and maintainability.

### C++ Backend Overhaul

* **Unified utility functions**: Introduced `utils.h` header providing numerically stable implementations of critical functions:

  - `log1mexp()`: Stable computation of log(1 - exp(x)) using Mächler (2012) methodology
  - `log1pexp()`: Overflow-protected computation of log(1 + exp(x))
  - `safe_log()`, `safe_exp()`, `safe_pow()`: Protected arithmetic operations with graceful handling of edge cases
  - Vectorized versions (`vec_safe_log`, `vec_log1mexp`, etc.) for efficient array operations

* **Consistent parameter validation**: All distribution families now use dedicated parameter checkers (`check_pars()`, `check_kw_pars()`, `check_ekw_pars()`, etc.) that properly handle NaN, Inf, and boundary conditions.

* **Complete documentation**: All C++ source files now include comprehensive Doxygen-style documentation headers describing:
  - Mathematical formulas for PDF, CDF, quantile, and random generation
  - Parameter constraints and special cases
  - Numerical stability considerations
  - Relationship to parent GKw distribution

### Bug Fixes

* **Fixed critical bug in `qgkw()`**: Corrected logic error where `lower_tail` transformation was incorrectly applied when `log_p = TRUE`. The probability is now properly converted to linear scale before tail adjustment.

* **Fixed gradient calculation in `grkkw()`**: Resolved issue where `log_z` was not recomputed after clamping `z` to minimum threshold, causing corrupted gradient values near boundaries.

* **Fixed Hessian calculation in `hsmc()`**: Corrected sign errors and formula for the lambda component of the Hessian matrix for the Beta-Power/McDonald distribution.

* **Fixed gradient signs in `grmc()`**: Ensured consistent computation of log-likelihood gradient before negation for optimization.

### Code Quality Improvements

* **Eliminated unused variables**: Removed declared but unused constants (`exp_threshold`) and intermediate variables across all distribution files.

* **Removed redundant calculations**: Streamlined computations, notably in `pgkw()` where logarithm was computed twice for the same quantity.

* **Simplified parameter recycling**: Replaced double-modulo indexing pattern (`idx = i % k; vec[idx % vec.n_elem]`) with direct single-modulo access (`vec[i % vec.n_elem]`) in random generation functions.

* **Standardized function signatures**: All distribution functions now follow consistent patterns for parameter order, validation, and return value handling.

### R Wrapper Layer

* **Complete separation of R and C++ interfaces**: All exported R functions now serve as wrappers around internal C++ implementations (`.dgkw_cpp`, `.pgkw_cpp`, etc.), providing:
  - Enhanced input validation with informative error messages
  - Consistent argument checking across all distribution families
  - Proper NA/NaN propagation
  - Documentation accessible via standard R help system

### Distribution Families

All seven distribution families have been refactored with identical improvements:

| Distribution | Parameters | File |
|--------------|------------|------|
| Generalized Kumaraswamy (GKw) | α, β, γ, δ, λ | `gkw.cpp` |
| Kumaraswamy-Kumaraswamy (KKw) | α, β, δ, λ | `kkw.cpp` |
| Beta-Kumaraswamy (BKw) | α, β, γ, δ | `bkw.cpp` |
| Exponentiated Kumaraswamy (EKw) | α, β, λ | `ekw.cpp` |
| Beta-Power/McDonald (BP/Mc) | γ, δ, λ | `bpmc.cpp` |
| Kumaraswamy (Kw) | α, β | `kw.cpp` |
| Beta (GKw-style) | γ, δ | `beta.cpp` |

Each family includes: density (`d*`), distribution (`p*`), quantile (`q*`), random generation (`r*`), negative log-likelihood (`ll*`), gradient (`gr*`), and Hessian (`hs*`) functions.

### Technical Notes

* Minimum supported R version remains 3.5.0
* C++11 standard required (enabled via Rcpp plugin)
* Depends on RcppArmadillo for efficient linear algebra operations

### Acknowledgments

Special thanks to the thorough code review process that identified subtle numerical issues in edge cases, particularly for extreme parameter values and observations near distribution boundaries.

# gkwdist 1.0.7

# gkwdist 1.0.5

## Documentation Improvements

* **Enhanced Examples for Likelihood Functions**: All `ll*`, `gr*`, and `hs*` functions now include comprehensive examples demonstrating:
  - Maximum likelihood estimation with analytical gradients
  - Univariate profile likelihoods with confidence thresholds
  - 2D likelihood surfaces with confidence regions (90%, 95%, 99%)
  - Confidence ellipses with marginal intervals for parameter pairs
  - Numerical vs analytical derivative verification
  - Likelihood ratio tests and score tests

* **Professional Visualization Standards**: 
  - Consistent color scheme across all examples
  - Grid-adaptive algorithms for computational efficiency
  - Base R only - no external dependencies required

* **Complete Coverage**: Enhanced documentation for all distribution families (Kw, EKw, KKw, GKw) covering 2 to 5 parameters

* **Theoretical References**: Documentation cites foundational work by Carrasco et al. (2010), Jones (2009), Kumaraswamy (1980), and standard inference theory from Casella & Berger (2002)


# gkwdist 1.0.3
* **README.md**: Fix typos and faill link
  - Fix zzz.R file by removing useless texts

# gkwdist 1.0.2

# gkwdist 1.0.1

## Major Improvements

### Enhanced `gkwgetstartvalues()` Function
* **NEW**: Added `family` parameter to support all distribution families
  - Automatically returns correct number of parameters for each family
  - Family-specific initial value strategies for better convergence
  - Supported families: `"gkw"`, `"bkw"`, `"kkw"`, `"ekw"`, `"mc"`, `"kw"`, `"beta"`
  - Case-insensitive family names for user convenience

### Documentation Enhancements
* **README.md**: Complete rewrite with mathematical rigor
  - All LaTeX formulas corrected and verified for proper rendering
  - Eight comprehensive examples using `optim()` with analytical gradients
  - Corrected function signatures: all `ll*()`, `gr*()`, and `hs*()` functions use `(par, data)` signature
  - Added performance benchmarks demonstrating 10-50× speedup with C++ implementation
  - Hierarchical structure diagram for all distribution families
  - Model selection workflow and practical guidelines
  - Removed all references to deprecated `gkwfit()` function

### CRAN Submission Readiness
* **DESCRIPTION**: Fixed to meet CRAN requirements
  - Proper `Authors@R` field formatting
  - Removed unused dependencies (`numDeriv`)
  - Corrected package dependencies (`RcppArmadillo` only in `LinkingTo`)
  - Enhanced description with DOI references
  - Fixed maintainer email formatting

## Bug Fixes

* Fixed function call signatures in all README examples to match actual implementation
* Corrected parameter passing in optimization examples (now consistently use `(par, data)`)
* Fixed LaTeX rendering issues with `\left`/`\right` delimiters in GitHub Markdown

## Testing

* **NEW**: Comprehensive test suite using `testthat`
  - 100+ tests covering all exported functions
  - Tests for all 7 distribution families (GKw, BKw, KKw, EKw, MC, Kw, Beta)
  - PDF, CDF, quantile, and random generation tests
  - Log-likelihood, gradient, and Hessian validation
  - Parameter recovery tests with MLE
  - Edge cases and boundary condition handling
  - Integration tests for PDF-CDF consistency

## Performance

* All functions implemented in C++ for maximum computational efficiency
* Analytical derivatives (gradient and Hessian) provide exact computations
* Optimized numerical stability for extreme parameter values

## Notes

* This is the initial CRAN submission
* Package focuses exclusively on distribution functions (no high-level fitting interface)
* Companion package `gkwreg` provides regression modeling capabilities
* All user-facing functions maintain backward compatibility
* C++ implementation uses RcppArmadillo for linear algebra operations
* Analytical functions use robust log-scale computations to prevent overflow/underflow
* Random generation uses inverse CDF method where closed-form solutions exist

# gkwdist 0.1.0

## New Features

* Initial CRAN release
* Generalized Kumaraswamy distribution (5 parameters)
* Six nested sub-families: Beta, Kumaraswamy, Exponentiated-Kumaraswamy, 
  Kumaraswamy-Kumaraswamy, Beta-Kumaraswamy, and McDonald distributions
* Complete set of distribution functions (d/p/q/r)
* Log-likelihood, gradient, and Hessian functions for all families

## Performance

* Optimized C++ implementation via Rcpp
* Vectorized operations for speed
