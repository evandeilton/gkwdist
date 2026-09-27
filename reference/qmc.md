# Quantile Function of the McDonald (Mc)/Beta Power Distribution

Computes the quantile function (inverse CDF) for the McDonald (Mc)
distribution (also known as Beta Power) with parameters `gamma`
(\\\gamma\\), `delta` (\\\delta\\), and `lambda` (\\\lambda\\). It finds
the value `q` such that \\P(X \le q) = p\\. This distribution is a
special case of the Generalized Kumaraswamy (GKw) distribution where
\\\alpha = 1\\ and \\\beta = 1\\.

## Usage

``` r
qmc(p, gamma = 1, delta = 0, lambda = 1, lower.tail = TRUE, log.p = FALSE)
```

## Arguments

- p:

  Vector of probabilities (values between 0 and 1).

- gamma:

  Shape parameter `gamma` \> 0. Can be a scalar or a vector. Default:
  1.0.

- delta:

  Shape parameter `delta` \>= 0. Can be a scalar or a vector. Default:
  0.0.

- lambda:

  Shape parameter `lambda` \> 0. Can be a scalar or a vector. Default:
  1.0.

- lower.tail:

  Logical; if `TRUE` (default), probabilities are \\p = P(X \le q)\\,
  otherwise, probabilities are \\p = P(X \> q)\\.

- log.p:

  Logical; if `TRUE`, probabilities `p` are given as \\\log(p)\\.
  Default: `FALSE`.

## Value

A vector of quantiles corresponding to the given probabilities `p`. The
length of the result is determined by the recycling rule applied to the
arguments (`p`, `gamma`, `delta`, `lambda`). Returns:

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

The quantile function \\Q(p)\\ is the inverse of the CDF \\F(q)\\. The
CDF for the Mc (\\\alpha=1, \beta=1\\) distribution is \\F(q) =
I\_{q^\lambda}(\gamma, \delta+1)\\, where \\I_z(a,b)\\ is the
regularized incomplete beta function (see
[`pmc`](https://evandeilton.github.io/gkwdist/reference/pmc.md)).

To find the quantile \\q\\, we first invert the Beta function part: let
\\y = I^{-1}\_{p}(\gamma, \delta+1)\\, where \\I^{-1}\_p(a,b)\\ is the
inverse computed via [`qbeta`](https://rdrr.io/r/stats/Beta.html). We
then solve \\q^\lambda = y\\ for \\q\\, yielding the quantile function:
\$\$ Q(p) = \left\[ I^{-1}\_{p}(\gamma, \delta+1) \right\]^{1/\lambda}
\$\$ The function uses this formula, calculating \\I^{-1}\_{p}(\gamma,
\delta+1)\\ via `qbeta(p, gamma, delta + 1, ...)` while respecting the
`lower.tail` and `log.p` arguments. This is equivalent to the general
GKw quantile function
([`qgkw`](https://evandeilton.github.io/gkwdist/reference/qgkw.md))
evaluated with \\\alpha=1, \beta=1\\.

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

[`qgkw`](https://evandeilton.github.io/gkwdist/reference/qgkw.md)
(parent distribution quantile function),
[`dmc`](https://evandeilton.github.io/gkwdist/reference/dmc.md),
[`pmc`](https://evandeilton.github.io/gkwdist/reference/pmc.md),
[`rmc`](https://evandeilton.github.io/gkwdist/reference/rmc.md) (other
Mc functions), [`qbeta`](https://rdrr.io/r/stats/Beta.html)

Other quantile functions:
[`qbeta_()`](https://evandeilton.github.io/gkwdist/reference/qbeta_.md),
[`qbkw()`](https://evandeilton.github.io/gkwdist/reference/qbkw.md),
[`qekw()`](https://evandeilton.github.io/gkwdist/reference/qekw.md),
[`qgkw()`](https://evandeilton.github.io/gkwdist/reference/qgkw.md),
[`qkkw()`](https://evandeilton.github.io/gkwdist/reference/qkkw.md),
[`qkw()`](https://evandeilton.github.io/gkwdist/reference/qkw.md)

## Author

Lopes, J. E.

## Examples

``` r
p <- c(0.1, 0.5, 0.9)
qmc(p, gamma = 0.5, delta = 5, lambda = 3)
#> [1] 0.1110876 0.3383841 0.5937371
# upper-tail quantiles
qmc(p, gamma = 0.5, delta = 5, lambda = 3, lower.tail = FALSE)
#> [1] 0.5937371 0.3383841 0.1110876

## qmc() inverts pmc()
all.equal(pmc(qmc(p, gamma = 0.5, delta = 5, lambda = 3), gamma = 0.5,
    delta = 5, lambda = 3), p)
#> [1] TRUE
```
