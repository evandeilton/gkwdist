# Random Number Generation for the Kumaraswamy (Kw) Distribution

Generates random deviates from the two-parameter Kumaraswamy (Kw)
distribution with shape parameters `alpha` (\\\alpha\\) and `beta`
(\\\beta\\).

## Usage

``` r
rkw(n, alpha = 1, beta = 1)
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

## Value

A vector of length `n` containing random deviates from the Kw
distribution, with values in (0, 1). The length of the result is
determined by `n` and the recycling rule applied to the parameters
(`alpha`, `beta`). An out-of-bound or missing parameter is an error, not
a return value: the wrapper stops with a message naming the parameter.
An infinite parameter is not currently intercepted there and reaches the
C++ layer, which treats it as invalid.

## Details

The generation method uses the inverse transform (quantile) method. That
is, if \\U\\ is a random variable following a standard Uniform
distribution on (0, 1), then \\X = Q(U)\\ follows the Kw distribution,
where \\Q(p)\\ is the Kw quantile function
([`qkw`](https://evandeilton.github.io/gkwdist/reference/qkw.md)): \$\$
Q(p) = \left\\ 1 - (1 - p)^{1/\beta} \right\\^{1/\alpha} \$\$ The
implementation generates \\U\\ using
[`runif`](https://rdrr.io/r/stats/Uniform.html) and applies this
transformation. This is equivalent to the general GKw generation method
([`rgkw`](https://evandeilton.github.io/gkwdist/reference/rgkw.md))
evaluated at \\\gamma=1, \delta=0, \lambda=1\\.

## References

Kumaraswamy, P. (1980). A generalized probability density function for
double-bounded random processes. *Journal of Hydrology*, *46*(1-2),
79-88.
[doi:10.1016/0022-1694(80)90036-0](https://doi.org/10.1016/0022-1694%2880%2990036-0)

Jones, M. C. (2009). Kumaraswamy's distribution: A beta-type
distribution with some tractability advantages. *Statistical
Methodology*, *6*(1), 70-81.
[doi:10.1016/j.stamet.2008.04.001](https://doi.org/10.1016/j.stamet.2008.04.001)

Devroye, L. (1986). *Non-Uniform Random Variate Generation*.
Springer-Verlag. (General methods for random variate generation).

## See also

[`rgkw`](https://evandeilton.github.io/gkwdist/reference/rgkw.md)
(parent distribution random generation),
[`dkw`](https://evandeilton.github.io/gkwdist/reference/dkw.md),
[`pkw`](https://evandeilton.github.io/gkwdist/reference/pkw.md),
[`qkw`](https://evandeilton.github.io/gkwdist/reference/qkw.md) (other
Kw functions), [`runif`](https://rdrr.io/r/stats/Uniform.html)

Other random generation functions:
[`rbeta_()`](https://evandeilton.github.io/gkwdist/reference/rbeta_.md),
[`rbkw()`](https://evandeilton.github.io/gkwdist/reference/rbkw.md),
[`rekw()`](https://evandeilton.github.io/gkwdist/reference/rekw.md),
[`rgkw()`](https://evandeilton.github.io/gkwdist/reference/rgkw.md),
[`rkkw()`](https://evandeilton.github.io/gkwdist/reference/rkkw.md),
[`rmc()`](https://evandeilton.github.io/gkwdist/reference/rmc.md)

## Author

Lopes, J. E.

## Examples

``` r
set.seed(123)
x <- rkw(1000, alpha = 2, beta = 3)
summary(x)
#>    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
#> 0.01246 0.30481 0.44835 0.45516 0.60611 0.95701 

## The sample follows the distribution
hist(x, breaks = 30, freq = FALSE, main = "")
curve(dkw(x, alpha = 2, beta = 3), add = TRUE)

ks.test(x, pkw, alpha = 2, beta = 3)
#> 
#>  Asymptotic one-sample Kolmogorov-Smirnov test
#> 
#> data:  x
#> D = 0.014051, p-value = 0.9891
#> alternative hypothesis: two-sided
#> 
```
