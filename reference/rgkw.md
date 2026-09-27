# Random Number Generation for the Generalized Kumaraswamy Distribution

Generates random deviates from the five-parameter Generalized
Kumaraswamy (GKw) distribution defined on the interval (0, 1).

## Usage

``` r
rgkw(n, alpha = 1, beta = 1, gamma = 1, delta = 0, lambda = 1)
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

- lambda:

  Shape parameter `lambda` \> 0. Can be a scalar or a vector. Default:
  1.0.

## Value

A vector of length `n` containing random deviates from the GKw
distribution. The length of the result is determined by `n` and the
recycling rule applied to the parameters (`alpha`, `beta`, `gamma`,
`delta`, `lambda`). An out-of-bound or missing parameter is an error,
not a return value: the wrapper stops with a message naming the
parameter. An infinite parameter is not currently intercepted there and
reaches the C++ layer, which treats it as invalid.

## Details

The generation method relies on the transformation property: if \\V \sim
\mathrm{Beta}(\gamma, \delta+1)\\, then the random variable `X` defined
as \$\$ X = \left\\ 1 - \left\[ 1 - V^{1/\lambda} \right\]^{1/\beta}
\right\\^{1/\alpha} \$\$ follows the GKw(\\\alpha, \beta, \gamma,
\delta, \lambda\\) distribution.

The algorithm proceeds as follows:

1.  Generate `V` from
    `stats::rbeta(n, shape1 = gamma, shape2 = delta + 1)`.

2.  Calculate \\v = V^{1/\lambda}\\.

3.  Calculate \\w = (1 - v)^{1/\beta}\\.

4.  Calculate \\x = (1 - w)^{1/\alpha}\\.

Parameters (`alpha`, `beta`, `gamma`, `delta`, `lambda`) are recycled to
match the length required by `n`. Numerical stability is maintained by
handling potential edge cases during the transformations.

## References

Carrasco, J. M. F., Ferrari, S. L. P., & Cordeiro, G. M. (2010). A new
generalized Kumaraswamy distribution. *arXiv preprint arXiv:1004.0911*.
[doi:10.48550/arXiv.1004.0911](https://doi.org/10.48550/arXiv.1004.0911)

Cordeiro, G. M., & de Castro, M. (2011). A new family of generalized
distributions. *Journal of Statistical Computation and Simulation*,
*81*(7), 883-898.
[doi:10.1080/00949650903530745](https://doi.org/10.1080/00949650903530745)

Kumaraswamy, P. (1980). A generalized probability density function for
double-bounded random processes. *Journal of Hydrology*, *46*(1-2),
79-88.
[doi:10.1016/0022-1694(80)90036-0](https://doi.org/10.1016/0022-1694%2880%2990036-0)

## See also

[`dgkw`](https://evandeilton.github.io/gkwdist/reference/dgkw.md),
[`pgkw`](https://evandeilton.github.io/gkwdist/reference/pgkw.md),
[`qgkw`](https://evandeilton.github.io/gkwdist/reference/qgkw.md),
[`rbeta`](https://rdrr.io/r/stats/Beta.html),
[`set.seed`](https://rdrr.io/r/base/Random.html)

Other random generation functions:
[`rbeta_()`](https://evandeilton.github.io/gkwdist/reference/rbeta_.md),
[`rbkw()`](https://evandeilton.github.io/gkwdist/reference/rbkw.md),
[`rekw()`](https://evandeilton.github.io/gkwdist/reference/rekw.md),
[`rkkw()`](https://evandeilton.github.io/gkwdist/reference/rkkw.md),
[`rkw()`](https://evandeilton.github.io/gkwdist/reference/rkw.md),
[`rmc()`](https://evandeilton.github.io/gkwdist/reference/rmc.md)

## Author

Lopes, J. E.

## Examples

``` r
set.seed(123)
x <- rgkw(1000, alpha = 2, beta = 3, gamma = 1.5, delta = 0.5, lambda = 1.2)
summary(x)
#>    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
#> 0.07502 0.38251 0.49412 0.49439 0.60747 0.93562 

## The sample follows the distribution
hist(x, breaks = 30, freq = FALSE, main = "")
curve(dgkw(x, alpha = 2, beta = 3, gamma = 1.5, delta = 0.5, lambda = 1.2),
    add = TRUE)

ks.test(x, pgkw, alpha = 2, beta = 3, gamma = 1.5, delta = 0.5, lambda = 1.2)
#> 
#>  Asymptotic one-sample Kolmogorov-Smirnov test
#> 
#> data:  x
#> D = 0.024673, p-value = 0.5766
#> alternative hypothesis: two-sided
#> 
```
