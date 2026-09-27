# Quantile Function of the Generalized Kumaraswamy Distribution

Computes the quantile function (inverse CDF) for the five-parameter
Generalized Kumaraswamy (GKw) distribution. Finds the value `x` such
that \\P(X \le x) = p\\, where `X` follows the GKw distribution.

## Usage

``` r
qgkw(
  p,
  alpha = 1,
  beta = 1,
  gamma = 1,
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

  Logical; if `TRUE` (default), probabilities are \\P(X \le x)\\,
  otherwise, \\P(X \> x)\\.

- log.p:

  Logical; if `TRUE`, probabilities `p` are given as \\\log(p)\\.
  Default: `FALSE`.

## Value

A vector of quantiles corresponding to the given probabilities `p`. The
length of the result is determined by the recycling rule applied to the
arguments (`p`, `alpha`, `beta`, `gamma`, `delta`, `lambda`). Returns:

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

The quantile function \\Q(p)\\ is the inverse of the CDF \\F(x)\\. Given
\\F(x) = I\_{y(x)}(\gamma, \delta+1)\\ where \\y(x) =
\[1-(1-x^{\alpha})^{\beta}\]^{\lambda}\\, the quantile function is: \$\$
Q(p) = x = \left\\ 1 - \left\[ 1 - \left( I^{-1}\_{p}(\gamma, \delta+1)
\right)^{1/\lambda} \right\]^{1/\beta} \right\\^{1/\alpha} \$\$ where
\\I^{-1}\_{p}(a, b)\\ is the inverse of the regularized incomplete beta
function, which corresponds to the quantile function of the Beta
distribution, [`qbeta`](https://rdrr.io/r/stats/Beta.html).

The computation proceeds as follows:

1.  Calculate
    `y = stats::qbeta(p, shape1 = gamma, shape2 = delta + 1, lower.tail = lower.tail, log.p = log.p)`.

2.  Calculate \\v = y^{1/\lambda}\\.

3.  Calculate \\w = (1 - v)^{1/\beta}\\. Note: Requires \\v \le 1\\.

4.  Calculate \\q = (1 - w)^{1/\alpha}\\. Note: Requires \\w \le 1\\.

Numerical stability is maintained by handling boundary cases (`p = 0`,
`p = 1`) directly and checking intermediate results (e.g., ensuring
arguments to powers are non-negative).

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
[`rgkw`](https://evandeilton.github.io/gkwdist/reference/rgkw.md),
[`qbeta`](https://rdrr.io/r/stats/Beta.html)

Other quantile functions:
[`qbeta_()`](https://evandeilton.github.io/gkwdist/reference/qbeta_.md),
[`qbkw()`](https://evandeilton.github.io/gkwdist/reference/qbkw.md),
[`qekw()`](https://evandeilton.github.io/gkwdist/reference/qekw.md),
[`qkkw()`](https://evandeilton.github.io/gkwdist/reference/qkkw.md),
[`qkw()`](https://evandeilton.github.io/gkwdist/reference/qkw.md),
[`qmc()`](https://evandeilton.github.io/gkwdist/reference/qmc.md)

## Author

Lopes, J. E.

## Examples

``` r
p <- c(0.1, 0.5, 0.9)
qgkw(p, alpha = 2, beta = 3, gamma = 1.5, delta = 0.5, lambda = 1.2)
#> [1] 0.2771293 0.4900199 0.7004042
qgkw(p, alpha = 2, beta = 3, gamma = 1.5, delta = 0.5, lambda = 1.2,
    lower.tail = FALSE)  # upper-tail quantiles
#> [1] 0.7004042 0.4900199 0.2771293

## qgkw() inverts pgkw()
all.equal(pgkw(qgkw(p, alpha = 2, beta = 3, gamma = 1.5, delta = 0.5,
    lambda = 1.2), alpha = 2, beta = 3, gamma = 1.5, delta = 0.5, lambda = 1.2),
    p)
#> [1] TRUE
```
