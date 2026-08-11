# Contributing to gkwdist

Thanks for your interest in gkwdist. Bug reports, questions and pull
requests are all welcome, and this document explains how they are
handled.

## Getting support and asking questions

- **Questions about how to use the package**: open a [GitHub
  issue](https://github.com/evandeilton/gkwdist/issues) with the label
  `question`. Please include a small reproducible example.
- **Statistical background**: the theory vignette,
  [`vignette("theory-gkwdist")`](https://evandeilton.github.io/gkwdist/articles/theory-gkwdist.md),
  derives the densities, score vectors, Hessians and the asymptotic
  covariance matrix for all seven sub-families. Most questions about
  what a parameter means or why a model behaves a certain way are
  answered there.

The maintainer aims to acknowledge new issues within two weeks. This is
an academic project maintained by one person, so please be patient; an
unanswered issue has not been ignored.

## Reporting a bug

Open an issue that includes:

1.  A minimal reproducible example, ideally with
    [`set.seed()`](https://rdrr.io/r/base/Random.html) and simulated
    data rather than a private dataset.
2.  The output of
    [`sessionInfo()`](https://rdrr.io/r/utils/sessionInfo.html).
3.  What you expected and what happened instead.

Numerical problems are the most valuable reports this package can
receive. If you believe a density, log-likelihood, score or Hessian is
wrong, the most useful comparison is against the general GKw routines
evaluated at the corresponding constrained parameter point, since every
sub-family is GKw with parameters fixed:

``` r

# EKw(alpha, beta, lambda) is GKw(alpha, beta, 1, 0, lambda)
llekw(c(2, 3, 2), x)
llgkw(c(2, 3, 1, 0, 2), x)   # must agree

grekw(c(2, 3, 2), x)
grgkw(c(2, 3, 1, 0, 2), x)[c(1, 2, 5)]   # must agree componentwise
```

A disagreement between these two is always a bug in one of them.

## Proposing a change

1.  Fork the repository and create a branch from `main`.
2.  Make your change, with tests.
3.  Run `devtools::test()` and `devtools::check()`; both must be clean.
4.  Open a pull request describing what changed and why.

### What the tests must cover

Derivative and density code has to be tested at *values*, not only at
types and shapes. A test that asserts a density is numeric, has the
right length, is non-negative and is finite passes even when the
function returns zero everywhere; this is exactly how a defect survived
until version 1.1.5. New numerical code is expected to be tested against
an independent reference:

- densities: that they integrate to one, and that they equal the general
  GKw density at the constrained point;
- score and Hessian: componentwise, against both `numDeriv` and the
  general GKw routines;
- parameter grids: including the degenerate values `gamma = 1`,
  `beta = 1`, `lambda = 1` and `delta = 0`, and data containing
  observations near zero and one, which is where numerical formulations
  break down.

See `tests/testthat/test-boundary-derivatives.R` and
`tests/testthat/test-density-correctness.R` for the pattern.

### Code style

- R code follows the tidyverse style guide; documentation is written
  with roxygen2.
- C++ code targets C++17 and uses the numerically stable helpers in
  `src/utils.h` (`gkw_log1mexp()`, `safe_log()`, `safe_exp()`,
  `safe_pow()`). Do not introduce new helper names that collide with R’s
  `Rmath.h` API.
- Keep all likelihood computations in log space. Clamping an
  intermediate quantity such as `w = 1 - v^beta` to a small constant
  silently changes the function being optimised; use `gkw_log1mexp()`
  instead.
- Guards of the form `if (abs(p - 1) > eps)` must never gate a mixed
  second derivative: differentiating once in `p` consumes the `(p - 1)`
  factor, so those terms survive at `p = 1`.

## Scope

gkwdist deliberately covers the distribution layer only: densities,
distribution and quantile functions, random generation, and analytical
likelihood derivatives. Regression modelling with covariates belongs in
[gkwreg](https://github.com/evandeilton/gkwreg), which imports this
package.

## Governance

The package is maintained by José Evandeilton Lopes. Decisions about
scope and releases rest with the maintainer. Contributors are credited
in `DESCRIPTION` under `ctb` when a contribution is merged.

## Code of conduct

By participating in this project you agree to abide by its [Code of
Conduct](https://evandeilton.github.io/gkwdist/CODE_OF_CONDUCT.md).
