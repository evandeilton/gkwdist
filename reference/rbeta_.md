# Random Number Generation for the Beta Distribution (gamma, delta+1 Parameterization)

Generates random deviates from the standard Beta distribution, using a
parameterization common in generalized distribution families. The
distribution is parameterized by `gamma` (\\\gamma\\) and `delta`
(\\\delta\\), corresponding to the standard Beta distribution with shape
parameters `shape1 = gamma` and `shape2 = delta + 1`. This is a special
case of the Generalized Kumaraswamy (GKw) distribution where \\\alpha =
1\\, \\\beta = 1\\, and \\\lambda = 1\\.

## Usage

``` r
rbeta_(n, gamma = 1, delta = 0)
```

## Arguments

- n:

  Number of observations. If `length(n) > 1`, the length is taken to be
  the number required. Must be a non-negative integer.

- gamma:

  First shape parameter (`shape1`), \\\gamma \> 0\\. Can be a scalar or
  a vector. Default: 1.0.

- delta:

  Second shape parameter is `delta + 1` (`shape2`), requires \\\delta
  \ge 0\\ so that `shape2 >= 1`. Can be a scalar or a vector. Default:
  0.0 (leading to `shape2 = 1`, i.e., Uniform).

## Value

A numeric vector of length `n` containing random deviates from the
Beta(\\\gamma, \delta+1\\) distribution, with values in (0, 1). The
length of the result is determined by `n` and the recycling rule applied
to the parameters (`gamma`, `delta`). An out-of-bound or missing
parameter is an error, not a return value: the wrapper stops with a
message naming the parameter. An infinite parameter is not currently
intercepted there and reaches the C++ layer, which treats it as invalid.

## Details

This function generates samples from a Beta distribution with parameters
`shape1 = gamma` and `shape2 = delta + 1`. It is equivalent to calling
`stats::rbeta(n, shape1 = gamma, shape2 = delta + 1)`.

This distribution arises as a special case of the five-parameter
Generalized Kumaraswamy (GKw) distribution
([`rgkw`](https://evandeilton.github.io/gkwdist/reference/rgkw.md))
obtained by setting \\\alpha = 1\\, \\\beta = 1\\, and \\\lambda = 1\\.
It is therefore also equivalent to the McDonald (Mc)/Beta Power
distribution
([`rmc`](https://evandeilton.github.io/gkwdist/reference/rmc.md)) with
\\\lambda = 1\\.

The function likely calls R's underlying `rbeta` function but ensures
consistent parameter recycling and handling within the C++ environment,
matching the style of other functions in the related families.

## References

Johnson, N. L., Kotz, S., & Balakrishnan, N. (1995). *Continuous
Univariate Distributions, Volume 2* (2nd ed.). Wiley.

Cordeiro, G. M., & de Castro, M. (2011). A new family of generalized
distributions. *Journal of Statistical Computation and Simulation*,
*81*(7), 883-898.
[doi:10.1080/00949650903530745](https://doi.org/10.1080/00949650903530745)

Devroye, L. (1986). *Non-Uniform Random Variate Generation*.
Springer-Verlag.

## See also

[`rbeta`](https://rdrr.io/r/stats/Beta.html) (standard R
implementation),
[`rgkw`](https://evandeilton.github.io/gkwdist/reference/rgkw.md)
(parent distribution random generation),
[`rmc`](https://evandeilton.github.io/gkwdist/reference/rmc.md)
(McDonald/Beta Power random generation),
[`dbeta_`](https://evandeilton.github.io/gkwdist/reference/dbeta_.md),
[`pbeta_`](https://evandeilton.github.io/gkwdist/reference/pbeta_.md),
[`qbeta_`](https://evandeilton.github.io/gkwdist/reference/qbeta_.md).

Other random generation functions:
[`rbkw()`](https://evandeilton.github.io/gkwdist/reference/rbkw.md),
[`rekw()`](https://evandeilton.github.io/gkwdist/reference/rekw.md),
[`rgkw()`](https://evandeilton.github.io/gkwdist/reference/rgkw.md),
[`rkkw()`](https://evandeilton.github.io/gkwdist/reference/rkkw.md),
[`rkw()`](https://evandeilton.github.io/gkwdist/reference/rkw.md),
[`rmc()`](https://evandeilton.github.io/gkwdist/reference/rmc.md)

## Author

Lopes, J. E.

## Examples

``` r
set.seed(123)
x <- rbeta_(1000, gamma = 2, delta = 3)
summary(x)
#>    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
#> 0.01114 0.20168 0.31755 0.33709 0.45444 0.90972 

## The sample follows the distribution
hist(x, breaks = 30, freq = FALSE, main = "")
curve(dbeta_(x, gamma = 2, delta = 3), add = TRUE)

ks.test(x, pbeta_, gamma = 2, delta = 3)
#> 
#>  Asymptotic one-sample Kolmogorov-Smirnov test
#> 
#> data:  x
#> D = 0.026144, p-value = 0.5013
#> alternative hypothesis: two-sided
#> 
```
