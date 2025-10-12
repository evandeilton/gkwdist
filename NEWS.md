# gkwdist 1.0.3

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
