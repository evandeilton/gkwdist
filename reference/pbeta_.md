# Cumulative Distribution Function (CDF) of the Beta Distribution (gamma, delta+1 Parameterization)

Computes the cumulative distribution function (CDF), \\F(q) = P(X \le
q)\\, for the standard Beta distribution, using a parameterization
common in generalized distribution families. The distribution is
parameterized by `gamma` (\\\gamma\\) and `delta` (\\\delta\\),
corresponding to the standard Beta distribution with shape parameters
`shape1 = gamma` and `shape2 = delta + 1`.

## Usage

``` r
pbeta_(q, gamma = 1, delta = 0, lower.tail = TRUE, log.p = FALSE)
```

## Arguments

- q:

  Vector of quantiles (values generally between 0 and 1).

- gamma:

  First shape parameter (`shape1`), \\\gamma \> 0\\. Can be a scalar or
  a vector. Default: 1.0.

- delta:

  Second shape parameter is `delta + 1` (`shape2`), requires \\\delta
  \ge 0\\ so that `shape2 >= 1`. Can be a scalar or a vector. Default:
  0.0 (leading to `shape2 = 1`).

- lower.tail:

  Logical; if `TRUE` (default), probabilities are \\P(X \le q)\\,
  otherwise, \\P(X \> q)\\.

- log.p:

  Logical; if `TRUE`, probabilities \\p\\ are given as \\\log(p)\\.
  Default: `FALSE`.

## Value

A vector of probabilities, \\F(q)\\, or their logarithms/complements
depending on `lower.tail` and `log.p`. The length of the result is
determined by the recycling rule applied to the arguments (`q`, `gamma`,
`delta`). When `lower.tail = TRUE`, returns `0` (or `-Inf` if
`log.p = TRUE`) for `q <= 0` and `1` (or `0` if `log.p = TRUE`) for
`q >= 1`. An out-of-bound or missing parameter is an error, not a return
value: the wrapper stops with a message naming the parameter. An
infinite parameter is not currently intercepted there and reaches the
C++ layer, which treats it as invalid. Boundary return values are
adjusted accordingly for `lower.tail = FALSE`.

## Details

This function computes the CDF of a Beta distribution with parameters
`shape1 = gamma` and `shape2 = delta + 1`. It is equivalent to calling
`stats::pbeta(q, shape1 = gamma, shape2 = delta + 1, lower.tail = lower.tail, log.p = log.p)`.

This distribution arises as a special case of the five-parameter
Generalized Kumaraswamy (GKw) distribution
([`pgkw`](https://evandeilton.github.io/gkwdist/reference/pgkw.md))
obtained by setting \\\alpha = 1\\, \\\beta = 1\\, and \\\lambda = 1\\.
It is therefore also equivalent to the McDonald (Mc)/Beta Power
distribution
([`pmc`](https://evandeilton.github.io/gkwdist/reference/pmc.md)) with
\\\lambda = 1\\.

The function likely calls R's underlying `pbeta` function but ensures
consistent parameter recycling and handling within the C++ environment,
matching the style of other functions in the related families.

## References

Johnson, N. L., Kotz, S., & Balakrishnan, N. (1995). *Continuous
Univariate Distributions, Volume 2* (2nd ed.). Wiley.

Cordeiro, G. M., & de Castro, M. (2011). A new family of generalized
distributions. *Journal of Statistical Computation and Simulation*,
*81*(7), 883-898.
[doi:10.1080/00949650903530745](https://doi.org/10.1080/00949650903530745)

## See also

[`pbeta`](https://rdrr.io/r/stats/Beta.html) (standard R
implementation),
[`pgkw`](https://evandeilton.github.io/gkwdist/reference/pgkw.md)
(parent distribution CDF),
[`pmc`](https://evandeilton.github.io/gkwdist/reference/pmc.md)
(McDonald/Beta Power CDF),
[`dbeta_`](https://evandeilton.github.io/gkwdist/reference/dbeta_.md),
[`qbeta_`](https://evandeilton.github.io/gkwdist/reference/qbeta_.md),
[`rbeta_`](https://evandeilton.github.io/gkwdist/reference/rbeta_.md).

Other cumulative distribution functions:
[`pbkw()`](https://evandeilton.github.io/gkwdist/reference/pbkw.md),
[`pekw()`](https://evandeilton.github.io/gkwdist/reference/pekw.md),
[`pgkw()`](https://evandeilton.github.io/gkwdist/reference/pgkw.md),
[`pkkw()`](https://evandeilton.github.io/gkwdist/reference/pkkw.md),
[`pkw()`](https://evandeilton.github.io/gkwdist/reference/pkw.md),
[`pmc()`](https://evandeilton.github.io/gkwdist/reference/pmc.md)

## Author

Lopes, J. E.

## Examples

``` r
q <- c(0.2, 0.5, 0.8)
pbeta_(q, gamma = 2, delta = 3)
#> [1] 0.26272 0.81250 0.99328
pbeta_(q, gamma = 2, delta = 3, lower.tail = FALSE)  # P(X > q)
#> [1] 0.73728 0.18750 0.00672
pbeta_(q, gamma = 2, delta = 3, log.p = TRUE)
#> [1] -1.336666453 -0.207639365 -0.006742681

## pbeta_() is the integral of dbeta_()
Fq <- pbeta_(0.5, gamma = 2, delta = 3)
all.equal(Fq, integrate(dbeta_, 0, 0.5, gamma = 2, delta = 3, rel.tol = 1e-10)$value)
#> [1] TRUE
```
