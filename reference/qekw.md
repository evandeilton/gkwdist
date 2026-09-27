# Quantile Function of the Exponentiated Kumaraswamy (EKw) Distribution

Computes the quantile function (inverse CDF) for the Exponentiated
Kumaraswamy (EKw) distribution with parameters `alpha` (\\\alpha\\),
`beta` (\\\beta\\), and `lambda` (\\\lambda\\). It finds the value `q`
such that \\P(X \le q) = p\\. This distribution is a special case of the
Generalized Kumaraswamy (GKw) distribution where \\\gamma = 1\\ and
\\\delta = 0\\.

## Usage

``` r
qekw(p, alpha = 1, beta = 1, lambda = 1, lower.tail = TRUE, log.p = FALSE)
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

- lambda:

  Shape parameter `lambda` \> 0 (exponent parameter). Can be a scalar or
  a vector. Default: 1.0.

- lower.tail:

  Logical; if `TRUE` (default), probabilities are \\p = P(X \le q)\\,
  otherwise, probabilities are \\p = P(X \> q)\\.

- log.p:

  Logical; if `TRUE`, probabilities `p` are given as \\\log(p)\\.
  Default: `FALSE`.

## Value

A vector of quantiles corresponding to the given probabilities `p`. The
length of the result is determined by the recycling rule applied to the
arguments (`p`, `alpha`, `beta`, `lambda`). Returns:

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
CDF for the EKw (\\\gamma=1, \delta=0\\) distribution is \\F(q) = \[1 -
(1 - q^\alpha)^\beta \]^\lambda\\ (see
[`pekw`](https://evandeilton.github.io/gkwdist/reference/pekw.md)).
Inverting this equation for \\q\\ yields the quantile function: \$\$
Q(p) = \left\\ 1 - \left\[ 1 - p^{1/\lambda} \right\]^{1/\beta}
\right\\^{1/\alpha} \$\$ The function uses this closed-form expression
and correctly handles the `lower.tail` and `log.p` arguments by
transforming `p` appropriately before applying the formula. This is
equivalent to the general GKw quantile function
([`qgkw`](https://evandeilton.github.io/gkwdist/reference/qgkw.md))
evaluated with \\\gamma=1, \delta=0\\.

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

## See also

[`qgkw`](https://evandeilton.github.io/gkwdist/reference/qgkw.md)
(parent distribution quantile function),
[`dekw`](https://evandeilton.github.io/gkwdist/reference/dekw.md),
[`pekw`](https://evandeilton.github.io/gkwdist/reference/pekw.md),
[`rekw`](https://evandeilton.github.io/gkwdist/reference/rekw.md) (other
EKw functions), [`qunif`](https://rdrr.io/r/stats/Uniform.html)

Other quantile functions:
[`qbeta_()`](https://evandeilton.github.io/gkwdist/reference/qbeta_.md),
[`qbkw()`](https://evandeilton.github.io/gkwdist/reference/qbkw.md),
[`qgkw()`](https://evandeilton.github.io/gkwdist/reference/qgkw.md),
[`qkkw()`](https://evandeilton.github.io/gkwdist/reference/qkkw.md),
[`qkw()`](https://evandeilton.github.io/gkwdist/reference/qkw.md),
[`qmc()`](https://evandeilton.github.io/gkwdist/reference/qmc.md)

## Author

Lopes, J. E.

## Examples

``` r
p <- c(0.1, 0.5, 0.9)
qekw(p, alpha = 2, beta = 3, lambda = 1.2)
#> [1] 0.2270178 0.4900199 0.7496334
# upper-tail quantiles
qekw(p, alpha = 2, beta = 3, lambda = 1.2, lower.tail = FALSE)
#> [1] 0.7496334 0.4900199 0.2270178

## qekw() inverts pekw()
all.equal(pekw(qekw(p, alpha = 2, beta = 3, lambda = 1.2), alpha = 2,
    beta = 3, lambda = 1.2), p)
#> [1] TRUE
```
