# Density of the Beta Distribution (gamma, delta+1 Parameterization)

Computes the probability density function (PDF) for the standard Beta
distribution, using a parameterization common in generalized
distribution families. The distribution is parameterized by `gamma`
(\\\gamma\\) and `delta` (\\\delta\\), corresponding to the standard
Beta distribution with shape parameters `shape1 = gamma` and
`shape2 = delta + 1`. The distribution is defined on the interval (0,
1).

## Usage

``` r
dbeta_(x, gamma = 1, delta = 0, log = FALSE)
```

## Arguments

- x:

  Vector of quantiles (values between 0 and 1).

- gamma:

  First shape parameter (`shape1`), \\\gamma \> 0\\. Can be a scalar or
  a vector. Default: 1.0.

- delta:

  Second shape parameter is `delta + 1` (`shape2`), requires \\\delta
  \ge 0\\ so that `shape2 >= 1`. Can be a scalar or a vector. Default:
  0.0 (leading to `shape2 = 1`).

- log:

  Logical; if `TRUE`, the logarithm of the density is returned
  (\\\log(f(x))\\). Default: `FALSE`.

## Value

A vector of density values (\\f(x)\\) or log-density values
(\\\log(f(x))\\). The length of the result is determined by the
recycling rule applied to the arguments (`x`, `gamma`, `delta`). Returns
`0` (or `-Inf` if `log = TRUE`) for `x` strictly outside the interval
\[0, 1\]. At the closed boundaries `x = 0` and `x = 1` the limiting
density is returned rather than `0`, matching
[`dbeta`](https://rdrr.io/r/stats/Beta.html) with `shape1 = gamma` and
`shape2 = delta + 1`; depending on the parameters that limit is `0`, a
finite positive value, or `Inf`. An out-of-bound or missing parameter is
an error, not a return value: the wrapper stops with a message naming
the parameter. An infinite parameter is not currently intercepted there
and reaches the C++ layer, which treats it as invalid.

## Details

The probability density function (PDF) calculated by this function
corresponds to a standard Beta distribution \\Beta(\gamma, \delta+1)\\:
\$\$ f(x; \gamma, \delta) = \frac{x^{\gamma-1}
(1-x)^{(\delta+1)-1}}{B(\gamma, \delta+1)} = \frac{x^{\gamma-1}
(1-x)^{\delta}}{B(\gamma, \delta+1)} \$\$ for \\0 \< x \< 1\\, where
\\B(a,b)\\ is the Beta function
([`beta`](https://rdrr.io/r/base/Special.html)).

This specific parameterization arises as a special case of the
five-parameter Generalized Kumaraswamy (GKw) distribution
([`dgkw`](https://evandeilton.github.io/gkwdist/reference/dgkw.md))
obtained by setting the parameters \\\alpha = 1\\, \\\beta = 1\\, and
\\\lambda = 1\\. It is therefore equivalent to the McDonald (Mc)/Beta
Power distribution
([`dmc`](https://evandeilton.github.io/gkwdist/reference/dmc.md)) with
\\\lambda = 1\\.

Note the difference in the second parameter compared to
[`dbeta`](https://rdrr.io/r/stats/Beta.html), where
`dbeta(x, shape1, shape2)` uses `shape2` directly. Here,
`shape1 = gamma` and `shape2 = delta + 1`.

## References

Johnson, N. L., Kotz, S., & Balakrishnan, N. (1995). *Continuous
Univariate Distributions, Volume 2* (2nd ed.). Wiley.

Cordeiro, G. M., & de Castro, M. (2011). A new family of generalized
distributions. *Journal of Statistical Computation and Simulation*,
*81*(7), 883-898.
[doi:10.1080/00949650903530745](https://doi.org/10.1080/00949650903530745)

## See also

[`dbeta`](https://rdrr.io/r/stats/Beta.html) (standard R
implementation),
[`dgkw`](https://evandeilton.github.io/gkwdist/reference/dgkw.md)
(parent distribution density),
[`dmc`](https://evandeilton.github.io/gkwdist/reference/dmc.md)
(McDonald/Beta Power density),
[`pbeta_`](https://evandeilton.github.io/gkwdist/reference/pbeta_.md),
[`qbeta_`](https://evandeilton.github.io/gkwdist/reference/qbeta_.md),
[`rbeta_`](https://evandeilton.github.io/gkwdist/reference/rbeta_.md).

Other density functions:
[`dbkw()`](https://evandeilton.github.io/gkwdist/reference/dbkw.md),
[`dekw()`](https://evandeilton.github.io/gkwdist/reference/dekw.md),
[`dgkw()`](https://evandeilton.github.io/gkwdist/reference/dgkw.md),
[`dkkw()`](https://evandeilton.github.io/gkwdist/reference/dkkw.md),
[`dkw()`](https://evandeilton.github.io/gkwdist/reference/dkw.md),
[`dmc()`](https://evandeilton.github.io/gkwdist/reference/dmc.md)

## Author

Lopes, J. E.

## Examples

``` r
x <- c(0.1, 0.3, 0.5, 0.7, 0.9)
dbeta_(x, gamma = 2, delta = 3)
#> [1] 1.458 2.058 1.250 0.378 0.018
dbeta_(x, gamma = 2, delta = 3, log = TRUE)
#> [1]  0.3770656  0.7217346  0.2231436 -0.9728611 -4.0173835

## The package's Beta(gamma, delta) is stats::dbeta with shapes gamma, delta + 1
all.equal(dbeta_(x, 2, 3), stats::dbeta(x, 2, 4))
#> [1] TRUE

## The density integrates to one
integrate(dbeta_, 0, 1, gamma = 2, delta = 3, rel.tol = 1e-10)
#> 1 with absolute error < 1.1e-14

curve(dbeta_(x, gamma = 2, delta = 3), from = 0, to = 1, ylab = "density")

```
