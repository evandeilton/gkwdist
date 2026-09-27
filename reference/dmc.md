# Density of the McDonald (Mc)/Beta Power Distribution

Computes the probability density function (PDF) for the McDonald (Mc)
distribution (also previously referred to as Beta Power) with parameters
`gamma` (\\\gamma\\), `delta` (\\\delta\\), and `lambda` (\\\lambda\\).
This distribution is defined on the interval (0, 1).

## Usage

``` r
dmc(x, gamma = 1, delta = 0, lambda = 1, log = FALSE)
```

## Arguments

- x:

  Vector of quantiles (values between 0 and 1).

- gamma:

  Shape parameter `gamma` \> 0. Can be a scalar or a vector. Default:
  1.0.

- delta:

  Shape parameter `delta` \>= 0. Can be a scalar or a vector. Default:
  0.0.

- lambda:

  Shape parameter `lambda` \> 0. Can be a scalar or a vector. Default:
  1.0.

- log:

  Logical; if `TRUE`, the logarithm of the density is returned
  (\\\log(f(x))\\). Default: `FALSE`.

## Value

A vector of density values (\\f(x)\\) or log-density values
(\\\log(f(x))\\). The length of the result is determined by the
recycling rule applied to the arguments (`x`, `gamma`, `delta`,
`lambda`). Returns `0` (or `-Inf` if `log = TRUE`) for `x` strictly
outside the interval \[0, 1\]. At the closed boundaries `x = 0` and
`x = 1` the limiting density is returned rather than `0`, following the
convention of base R's density functions (compare
[`dbeta`](https://rdrr.io/r/stats/Beta.html)); depending on the
parameters that limit is `0`, a finite positive value, or `Inf`. An
out-of-bound or missing parameter is an error, not a return value: the
wrapper stops with a message naming the parameter. An infinite parameter
is not currently intercepted there and reaches the C++ layer, which
treats it as invalid.

## Details

The probability density function (PDF) of the McDonald (Mc) distribution
is given by: \$\$ f(x; \gamma, \delta, \lambda) =
\frac{\lambda}{B(\gamma,\delta+1)} x^{\gamma \lambda - 1} (1 -
x^\lambda)^\delta \$\$ for \\0 \< x \< 1\\, where \\B(a,b)\\ is the Beta
function ([`beta`](https://rdrr.io/r/base/Special.html)).

The Mc distribution is a special case of the five-parameter Generalized
Kumaraswamy (GKw) distribution
([`dgkw`](https://evandeilton.github.io/gkwdist/reference/dgkw.md))
obtained by setting the parameters \\\alpha = 1\\ and \\\beta = 1\\. It
was introduced by McDonald (1984) and is related to the Generalized Beta
distribution of the first kind (GB1). When \\\lambda=1\\, it simplifies
to the standard Beta distribution with parameters \\\gamma\\ and
\\\delta+1\\.

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

## See also

[`dgkw`](https://evandeilton.github.io/gkwdist/reference/dgkw.md)
(parent distribution density),
[`pmc`](https://evandeilton.github.io/gkwdist/reference/pmc.md),
[`qmc`](https://evandeilton.github.io/gkwdist/reference/qmc.md),
[`rmc`](https://evandeilton.github.io/gkwdist/reference/rmc.md) (other
Mc functions), [`dbeta`](https://rdrr.io/r/stats/Beta.html)

Other density functions:
[`dbeta_()`](https://evandeilton.github.io/gkwdist/reference/dbeta_.md),
[`dbkw()`](https://evandeilton.github.io/gkwdist/reference/dbkw.md),
[`dekw()`](https://evandeilton.github.io/gkwdist/reference/dekw.md),
[`dgkw()`](https://evandeilton.github.io/gkwdist/reference/dgkw.md),
[`dkkw()`](https://evandeilton.github.io/gkwdist/reference/dkkw.md),
[`dkw()`](https://evandeilton.github.io/gkwdist/reference/dkw.md)

## Author

Lopes, J. E.

## Examples

``` r
x <- c(0.1, 0.3, 0.5, 0.7, 0.9)
dmc(x, gamma = 0.5, delta = 5, lambda = 3)
#> [1] 1.277650206 1.939587413 1.472684770 0.415872685 0.005630568
dmc(x, gamma = 0.5, delta = 5, lambda = 3, log = TRUE)
#> [1]  0.2450226  0.6624753  0.3870871 -0.8773761 -5.1795449

## Mc is GKw with alpha = beta = 1
all.equal(dmc(x, 0.5, 5, 3), dgkw(x, 1, 1, 0.5, 5, 3))
#> [1] TRUE

## The density integrates to one
integrate(dmc, 0, 1, gamma = 0.5, delta = 5, lambda = 3, rel.tol = 1e-10)
#> 1 with absolute error < 1.4e-12

curve(dmc(x, gamma = 0.5, delta = 5, lambda = 3), from = 0, to = 1, ylab = "density")

```
