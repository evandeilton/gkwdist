# Random Number Generation for the Beta-Kumaraswamy (BKw) Distribution

Generates random deviates from the Beta-Kumaraswamy (BKw) distribution
with parameters `alpha` (\\\alpha\\), `beta` (\\\beta\\), `gamma`
(\\\gamma\\), and `delta` (\\\delta\\). This distribution is a special
case of the Generalized Kumaraswamy (GKw) distribution where the
parameter \\\lambda = 1\\.

## Usage

``` r
rbkw(n, alpha = 1, beta = 1, gamma = 1, delta = 0)
```

## Arguments

- n:

  Number of observations. If `length(n) > 1`, the length is taken to be
  the number required. Must be a non-negative integer.

- alpha:

  Shape parameter `alpha` \> 0. Can be a scalar or a vector. Default:
  1.0.

- beta:

  Shape parameter `beta` \> 0. Can be a scalar or a vector. Default:
  1.0.

- gamma:

  Shape parameter `gamma` \> 0. Can be a scalar or a vector. Default:
  1.0.

- delta:

  Shape parameter `delta` \>= 0. Can be a scalar or a vector. Default:
  0.0.

## Value

A vector of length `n` containing random deviates from the BKw
distribution. The length of the result is determined by `n` and the
recycling rule applied to the parameters (`alpha`, `beta`, `gamma`,
`delta`). An out-of-bound or missing parameter is an error, not a return
value: the wrapper stops with a message naming the parameter. An
infinite parameter is not currently intercepted there and reaches the
C++ layer, which treats it as invalid.

## Details

The generation method uses the relationship between the GKw distribution
and the Beta distribution. The general procedure for GKw
([`rgkw`](https://evandeilton.github.io/gkwdist/reference/rgkw.md)) is:
If \\W \sim \mathrm{Beta}(\gamma, \delta+1)\\, then \\X = \\1 - \[1 -
W^{1/\lambda}\]^{1/\beta}\\^{1/\alpha}\\ follows the GKw(\\\alpha,
\beta, \gamma, \delta, \lambda\\) distribution.

For the BKw distribution, \\\lambda=1\\. Therefore, the algorithm
simplifies to:

1.  Generate \\V \sim \mathrm{Beta}(\gamma, \delta+1)\\ using
    [`rbeta`](https://rdrr.io/r/stats/Beta.html).

2.  Compute the BKw variate \\X = \\1 - (1 -
    V)^{1/\beta}\\^{1/\alpha}\\.

This procedure is implemented efficiently, handling parameter recycling
as needed.

## References

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
[`dbkw`](https://evandeilton.github.io/gkwdist/reference/dbkw.md),
[`pbkw`](https://evandeilton.github.io/gkwdist/reference/pbkw.md),
[`qbkw`](https://evandeilton.github.io/gkwdist/reference/qbkw.md) (other
BKw functions), [`rbeta`](https://rdrr.io/r/stats/Beta.html)

Other random generation functions:
[`rbeta_()`](https://evandeilton.github.io/gkwdist/reference/rbeta_.md),
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
x <- rbkw(1000, alpha = 2, beta = 3, gamma = 1.5, delta = 0.5)
summary(x)
#>    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
#> 0.04977 0.34173 0.45853 0.46130 0.57929 0.93144 

## The sample follows the distribution
hist(x, breaks = 30, freq = FALSE, main = "")
curve(dbkw(x, alpha = 2, beta = 3, gamma = 1.5, delta = 0.5), add = TRUE)

ks.test(x, pbkw, alpha = 2, beta = 3, gamma = 1.5, delta = 0.5)
#> 
#>  Asymptotic one-sample Kolmogorov-Smirnov test
#> 
#> data:  x
#> D = 0.024673, p-value = 0.5766
#> alternative hypothesis: two-sided
#> 
```
