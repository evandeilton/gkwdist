# Quantile Function of the Kumaraswamy-Kumaraswamy (KKw) Distribution

Computes the quantile function (inverse CDF) for the
Kumaraswamy-Kumaraswamy (KKw) distribution with parameters `alpha`
(\\\alpha\\), `beta` (\\\beta\\), `delta` (\\\delta\\), and `lambda`
(\\\lambda\\). It finds the value `q` such that \\P(X \le q) = p\\. This
distribution is a special case of the Generalized Kumaraswamy (GKw)
distribution where the parameter \\\gamma = 1\\.

## Usage

``` r
qkkw(
  p,
  alpha = 1,
  beta = 1,
  delta = 0,
  lambda = 1,
  lower.tail = TRUE,
  log.p = FALSE
)
```

## Arguments

- p:

  Vector of probabilities (values between 0 and 1).

- alpha:

  Shape parameter `alpha` \> 0. Can be a scalar or a vector. Default:
  1.0.

- beta:

  Shape parameter `beta` \> 0. Can be a scalar or a vector. Default:
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
arguments (`p`, `alpha`, `beta`, `delta`, `lambda`). Returns:

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
CDF for the KKw (\\\gamma=1\\) distribution is (see
[`pkkw`](https://evandeilton.github.io/gkwdist/reference/pkkw.md)): \$\$
F(q) = 1 - \bigl\\1 - \bigl\[1 - (1 -
q^\alpha)^\beta\bigr\]^\lambda\bigr\\^{\delta + 1} \$\$ Inverting this
equation for \\q\\ yields the quantile function: \$\$ Q(p) = \left\[ 1 -
\left\\ 1 - \left\[ 1 - (1 - p)^{1/(\delta+1)} \right\]^{1/\lambda}
\right\\^{1/\beta} \right\]^{1/\alpha} \$\$ The function uses this
closed-form expression and correctly handles the `lower.tail` and
`log.p` arguments by transforming `p` appropriately before applying the
formula.

## References

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
[`dkkw`](https://evandeilton.github.io/gkwdist/reference/dkkw.md),
[`pkkw`](https://evandeilton.github.io/gkwdist/reference/pkkw.md),
[`rkkw`](https://evandeilton.github.io/gkwdist/reference/rkkw.md),
[`qbeta`](https://rdrr.io/r/stats/Beta.html)

Other quantile functions:
[`qbeta_()`](https://evandeilton.github.io/gkwdist/reference/qbeta_.md),
[`qbkw()`](https://evandeilton.github.io/gkwdist/reference/qbkw.md),
[`qekw()`](https://evandeilton.github.io/gkwdist/reference/qekw.md),
[`qgkw()`](https://evandeilton.github.io/gkwdist/reference/qgkw.md),
[`qkw()`](https://evandeilton.github.io/gkwdist/reference/qkw.md),
[`qmc()`](https://evandeilton.github.io/gkwdist/reference/qmc.md)

## Author

Lopes, J. E.

## Examples

``` r
p <- c(0.1, 0.5, 0.9)
qkkw(p, alpha = 2, beta = 3, delta = 0.5, lambda = 1.2)
#> [1] 0.1916721 0.4172992 0.6574119
# upper-tail quantiles
qkkw(p, alpha = 2, beta = 3, delta = 0.5, lambda = 1.2, lower.tail = FALSE)
#> [1] 0.6574119 0.4172992 0.1916721

## qkkw() inverts pkkw()
all.equal(pkkw(qkkw(p, alpha = 2, beta = 3, delta = 0.5, lambda = 1.2),
    alpha = 2, beta = 3, delta = 0.5, lambda = 1.2), p)
#> [1] TRUE
```
