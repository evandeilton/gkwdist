# Random Number Generation for the Exponentiated Kumaraswamy (EKw) Distribution

Generates random deviates from the Exponentiated Kumaraswamy (EKw)
distribution with parameters `alpha` (\\\alpha\\), `beta` (\\\beta\\),
and `lambda` (\\\lambda\\). This distribution is a special case of the
Generalized Kumaraswamy (GKw) distribution where \\\gamma = 1\\ and
\\\delta = 0\\.

## Usage

``` r
rekw(n, alpha = 1, beta = 1, lambda = 1)
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

- lambda:

  Shape parameter `lambda` \> 0 (exponent parameter). Can be a scalar or
  a vector. Default: 1.0.

## Value

A vector of length `n` containing random deviates from the EKw
distribution. The length of the result is determined by `n` and the
recycling rule applied to the parameters (`alpha`, `beta`, `lambda`). An
out-of-bound or missing parameter is an error, not a return value: the
wrapper stops with a message naming the parameter. An infinite parameter
is not currently intercepted there and reaches the C++ layer, which
treats it as invalid.

## Details

The generation method uses the inverse transform (quantile) method. That
is, if \\U\\ is a random variable following a standard Uniform
distribution on (0, 1), then \\X = Q(U)\\ follows the EKw distribution,
where \\Q(u)\\ is the EKw quantile function
([`qekw`](https://evandeilton.github.io/gkwdist/reference/qekw.md)):
\$\$ Q(u) = \left\\ 1 - \left\[ 1 - u^{1/\lambda} \right\]^{1/\beta}
\right\\^{1/\alpha} \$\$ This is computationally equivalent to the
general GKw generation method
([`rgkw`](https://evandeilton.github.io/gkwdist/reference/rgkw.md)) when
specialized for \\\gamma=1, \delta=0\\, as the required Beta(1, 1)
random variate is equivalent to a standard Uniform(0, 1) variate. The
implementation generates \\U\\ using
[`runif`](https://rdrr.io/r/stats/Uniform.html) and applies the
transformation above.

## References

Nadarajah, S., Cordeiro, G. M., & Ortega, E. M. (2012). The
exponentiated Kumaraswamy distribution. *Journal of the Franklin
Institute*, *349*(3),

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
[`dekw`](https://evandeilton.github.io/gkwdist/reference/dekw.md),
[`pekw`](https://evandeilton.github.io/gkwdist/reference/pekw.md),
[`qekw`](https://evandeilton.github.io/gkwdist/reference/qekw.md) (other
EKw functions), [`runif`](https://rdrr.io/r/stats/Uniform.html)

Other random generation functions:
[`rbeta_()`](https://evandeilton.github.io/gkwdist/reference/rbeta_.md),
[`rbkw()`](https://evandeilton.github.io/gkwdist/reference/rbkw.md),
[`rgkw()`](https://evandeilton.github.io/gkwdist/reference/rgkw.md),
[`rkkw()`](https://evandeilton.github.io/gkwdist/reference/rkkw.md),
[`rkw()`](https://evandeilton.github.io/gkwdist/reference/rkw.md),
[`rmc()`](https://evandeilton.github.io/gkwdist/reference/rmc.md)

## Author

Lopes, J. E.

## Examples

``` r
set.seed(123)
x <- rekw(1000, alpha = 2, beta = 3, lambda = 1.2)
summary(x)
#>    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
#> 0.02361 0.34658 0.48448 0.48738 0.63249 0.95960 

## The sample follows the distribution
hist(x, breaks = 30, freq = FALSE, main = "")
curve(dekw(x, alpha = 2, beta = 3, lambda = 1.2), add = TRUE)

ks.test(x, pekw, alpha = 2, beta = 3, lambda = 1.2)
#> 
#>  Asymptotic one-sample Kolmogorov-Smirnov test
#> 
#> data:  x
#> D = 0.014051, p-value = 0.9891
#> alternative hypothesis: two-sided
#> 
```
