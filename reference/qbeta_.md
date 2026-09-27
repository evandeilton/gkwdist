# Quantile Function of the Beta Distribution (gamma, delta+1 Parameterization)

Computes the quantile function (inverse CDF) for the standard Beta
distribution, using a parameterization common in generalized
distribution families. It finds the value `q` such that \\P(X \le q) =
p\\. The distribution is parameterized by `gamma` (\\\gamma\\) and
`delta` (\\\delta\\), corresponding to the standard Beta distribution
with shape parameters `shape1 = gamma` and `shape2 = delta + 1`.

## Usage

``` r
qbeta_(p, gamma = 1, delta = 0, lower.tail = TRUE, log.p = FALSE)
```

## Arguments

- p:

  Vector of probabilities (values between 0 and 1).

- gamma:

  First shape parameter (`shape1`), \\\gamma \> 0\\. Can be a scalar or
  a vector. Default: 1.0.

- delta:

  Second shape parameter is `delta + 1` (`shape2`), requires \\\delta
  \ge 0\\ so that `shape2 >= 1`. Can be a scalar or a vector. Default:
  0.0 (leading to `shape2 = 1`).

- lower.tail:

  Logical; if `TRUE` (default), probabilities are \\p = P(X \le q)\\,
  otherwise, probabilities are \\p = P(X \> q)\\.

- log.p:

  Logical; if `TRUE`, probabilities `p` are given as \\\log(p)\\.
  Default: `FALSE`.

## Value

A vector of quantiles corresponding to the given probabilities `p`. The
length of the result is determined by the recycling rule applied to the
arguments (`p`, `gamma`, `delta`). Returns:

- `0` for `p = 0` (or `p = -Inf` if `log.p = TRUE`, when
  `lower.tail = TRUE`).

- `1` for `p = 1` (or `p = 0` if `log.p = TRUE`, when
  `lower.tail = TRUE`).

- `NaN` for `p < 0` or `p > 1` (or corresponding log scale).

- An out-of-bound or missing parameter is an error, not a return value:
  the wrapper stops with a message naming the parameter. An infinite
  parameter is not currently intercepted there and reaches the C++
  layer, which treats it as invalid.

Boundary return values are adjusted accordingly for
`lower.tail = FALSE`.

## Details

This function computes the quantiles of a Beta distribution with
parameters `shape1 = gamma` and `shape2 = delta + 1`. It is equivalent
to calling
`stats::qbeta(p, shape1 = gamma, shape2 = delta + 1, lower.tail = lower.tail, log.p = log.p)`.

This distribution arises as a special case of the five-parameter
Generalized Kumaraswamy (GKw) distribution
([`qgkw`](https://evandeilton.github.io/gkwdist/reference/qgkw.md))
obtained by setting \\\alpha = 1\\, \\\beta = 1\\, and \\\lambda = 1\\.
It is therefore also equivalent to the McDonald (Mc)/Beta Power
distribution
([`qmc`](https://evandeilton.github.io/gkwdist/reference/qmc.md)) with
\\\lambda = 1\\.

The function likely calls R's underlying `qbeta` function but ensures
consistent parameter recycling and handling within the C++ environment,
matching the style of other functions in the related families. Boundary
conditions (p=0, p=1) are handled explicitly.

## References

Johnson, N. L., Kotz, S., & Balakrishnan, N. (1995). *Continuous
Univariate Distributions, Volume 2* (2nd ed.). Wiley.

Cordeiro, G. M., & de Castro, M. (2011). A new family of generalized
distributions. *Journal of Statistical Computation and Simulation*,
*81*(7), 883-898.
[doi:10.1080/00949650903530745](https://doi.org/10.1080/00949650903530745)

## See also

[`qbeta`](https://rdrr.io/r/stats/Beta.html) (standard R
implementation),
[`qgkw`](https://evandeilton.github.io/gkwdist/reference/qgkw.md)
(parent distribution quantile function),
[`qmc`](https://evandeilton.github.io/gkwdist/reference/qmc.md)
(McDonald/Beta Power quantile function),
[`dbeta_`](https://evandeilton.github.io/gkwdist/reference/dbeta_.md),
[`pbeta_`](https://evandeilton.github.io/gkwdist/reference/pbeta_.md),
[`rbeta_`](https://evandeilton.github.io/gkwdist/reference/rbeta_.md).

Other quantile functions:
[`qbkw()`](https://evandeilton.github.io/gkwdist/reference/qbkw.md),
[`qekw()`](https://evandeilton.github.io/gkwdist/reference/qekw.md),
[`qgkw()`](https://evandeilton.github.io/gkwdist/reference/qgkw.md),
[`qkkw()`](https://evandeilton.github.io/gkwdist/reference/qkkw.md),
[`qkw()`](https://evandeilton.github.io/gkwdist/reference/qkw.md),
[`qmc()`](https://evandeilton.github.io/gkwdist/reference/qmc.md)

## Author

Lopes, J. E.

## Examples

``` r
p <- c(0.1, 0.5, 0.9)
qbeta_(p, gamma = 2, delta = 3)
#> [1] 0.1122350 0.3138102 0.5838904
qbeta_(p, gamma = 2, delta = 3, lower.tail = FALSE)  # upper-tail quantiles
#> [1] 0.5838904 0.3138102 0.1122350

## qbeta_() inverts pbeta_()
all.equal(pbeta_(qbeta_(p, gamma = 2, delta = 3), gamma = 2, delta = 3), p)
#> [1] TRUE
```
