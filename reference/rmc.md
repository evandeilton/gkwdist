# Random Number Generation for the McDonald (Mc)/Beta Power Distribution

Generates random deviates from the McDonald (Mc) distribution (also
known as Beta Power) with parameters `gamma` (\\\gamma\\), `delta`
(\\\delta\\), and `lambda` (\\\lambda\\). This distribution is a special
case of the Generalized Kumaraswamy (GKw) distribution where \\\alpha =
1\\ and \\\beta = 1\\.

## Usage

``` r
rmc(n, gamma = 1, delta = 0, lambda = 1)
```

## Arguments

- n:

  Number of observations. If `length(n) > 1`, the length is taken to be
  the number required. Must be a non-negative integer.

- gamma:

  Shape parameter `gamma` \> 0. Can be a scalar or a vector. Default:
  1.0.

- delta:

  Shape parameter `delta` \>= 0. Can be a scalar or a vector. Default:
  0.0.

- lambda:

  Shape parameter `lambda` \> 0. Can be a scalar or a vector. Default:
  1.0.

## Value

A vector of length `n` containing random deviates from the Mc
distribution, with values in (0, 1). The length of the result is
determined by `n` and the recycling rule applied to the parameters
(`gamma`, `delta`, `lambda`). An out-of-bound or missing parameter is an
error, not a return value: the wrapper stops with a message naming the
parameter. An infinite parameter is not currently intercepted there and
reaches the C++ layer, which treats it as invalid.

## Details

The generation method uses the relationship between the GKw distribution
and the Beta distribution. The general procedure for GKw
([`rgkw`](https://evandeilton.github.io/gkwdist/reference/rgkw.md)) is:
If \\W \sim \mathrm{Beta}(\gamma, \delta+1)\\, then \\X = \\1 - \[1 -
W^{1/\lambda}\]^{1/\beta}\\^{1/\alpha}\\ follows the GKw(\\\alpha,
\beta, \gamma, \delta, \lambda\\) distribution.

For the Mc distribution, \\\alpha=1\\ and \\\beta=1\\. Therefore, the
algorithm simplifies significantly:

1.  Generate \\U \sim \mathrm{Beta}(\gamma, \delta+1)\\ using
    [`rbeta`](https://rdrr.io/r/stats/Beta.html).

2.  Compute the Mc variate \\X = U^{1/\lambda}\\.

This procedure is implemented efficiently, handling parameter recycling
as needed.

## References

McDonald, J. B. (1984). Some generalized functions for the size
distribution of income. *Econometrica*, *52*(3), 647-663.
[doi:10.2307/1913469](https://doi.org/10.2307/1913469)

Cordeiro, G. M., & de Castro, M. (2011). A new family of generalized
distributions. *Journal of Statistical Computation and Simulation*,
*81*(7), 883-898.
[doi:10.1080/00949650903530745](https://doi.org/10.1080/00949650903530745)

Kumaraswamy, P. (1980). A generalized probability density function for
double-bounded random processes. *Journal of Hydrology*, *46*(1-2),
79-88.
[doi:10.1016/0022-1694(80)90036-0](https://doi.org/10.1016/0022-1694%2880%2990036-0)

Devroye, L. (1986). *Non-Uniform Random Variate Generation*.
Springer-Verlag. (General methods for random variate generation).

## See also

[`rgkw`](https://evandeilton.github.io/gkwdist/reference/rgkw.md)
(parent distribution random generation),
[`dmc`](https://evandeilton.github.io/gkwdist/reference/dmc.md),
[`pmc`](https://evandeilton.github.io/gkwdist/reference/pmc.md),
[`qmc`](https://evandeilton.github.io/gkwdist/reference/qmc.md) (other
Mc functions), [`rbeta`](https://rdrr.io/r/stats/Beta.html)

Other random generation functions:
[`rbeta_()`](https://evandeilton.github.io/gkwdist/reference/rbeta_.md),
[`rbkw()`](https://evandeilton.github.io/gkwdist/reference/rbkw.md),
[`rekw()`](https://evandeilton.github.io/gkwdist/reference/rekw.md),
[`rgkw()`](https://evandeilton.github.io/gkwdist/reference/rgkw.md),
[`rkkw()`](https://evandeilton.github.io/gkwdist/reference/rkkw.md),
[`rkw()`](https://evandeilton.github.io/gkwdist/reference/rkw.md)

## Author

Lopes, J. E.

## Examples

``` r
set.seed(123)
x <- rmc(1000, gamma = 0.5, delta = 5, lambda = 3)
summary(x)
#>     Min.  1st Qu.   Median     Mean  3rd Qu.     Max. 
#> 0.002166 0.189371 0.327717 0.334765 0.464918 0.873018 

## The sample follows the distribution
hist(x, breaks = 30, freq = FALSE, main = "")
curve(dmc(x, gamma = 0.5, delta = 5, lambda = 3), add = TRUE)

ks.test(x, pmc, gamma = 0.5, delta = 5, lambda = 3)
#> 
#>  Asymptotic one-sample Kolmogorov-Smirnov test
#> 
#> data:  x
#> D = 0.035509, p-value = 0.1606
#> alternative hypothesis: two-sided
#> 
```
