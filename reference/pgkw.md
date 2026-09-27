# Cumulative Distribution Function (CDF) of the Generalized Kumaraswamy Distribution

Computes the cumulative distribution function (CDF) for the
five-parameter Generalized Kumaraswamy (GKw) distribution, defined on
the interval (0, 1). Calculates \\P(X \le q)\\.

## Usage

``` r
pgkw(
  q,
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

- q:

  Vector of quantiles (values generally between 0 and 1).

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

  Logical; if `TRUE` (default), probabilities are \\P(X \le q)\\,
  otherwise, \\P(X \> q)\\.

- log.p:

  Logical; if `TRUE`, probabilities \\p\\ are given as \\\log(p)\\.
  Default: `FALSE`.

## Value

A vector of probabilities, \\F(q)\\, or their logarithms/complements
depending on `lower.tail` and `log.p`. The length of the result is
determined by the recycling rule applied to the arguments (`q`, `alpha`,
`beta`, `gamma`, `delta`, `lambda`). When `lower.tail = TRUE`, returns
`0` (or `-Inf` if `log.p = TRUE`) for `q <= 0` and `1` (or `0` if
`log.p = TRUE`) for `q >= 1`. An out-of-bound or missing parameter is an
error, not a return value: the wrapper stops with a message naming the
parameter. An infinite parameter is not currently intercepted there and
reaches the C++ layer, which treats it as invalid. Boundary return
values are adjusted accordingly for `lower.tail = FALSE`.

## Details

The cumulative distribution function (CDF) of the Generalized
Kumaraswamy (GKw) distribution with parameters `alpha` (\\\alpha\\),
`beta` (\\\beta\\), `gamma` (\\\gamma\\), `delta` (\\\delta\\), and
`lambda` (\\\lambda\\) is given by: \$\$ F(q; \alpha, \beta, \gamma,
\delta, \lambda) = I\_{x(q)}(\gamma, \delta+1) \$\$ where \\x(q) =
\[1-(1-q^{\alpha})^{\beta}\]^{\lambda}\\ and \\I_x(a, b)\\ is the
regularized incomplete beta function, defined as: \$\$ I_x(a, b) =
\frac{B_x(a, b)}{B(a, b)} = \frac{\int_0^x t^{a-1}(1-t)^{b-1}
dt}{\int_0^1 t^{a-1}(1-t)^{b-1} dt} \$\$ This corresponds to the
[`pbeta`](https://rdrr.io/r/stats/Beta.html) function in R, such that
\\F(q; \alpha, \beta, \gamma, \delta, \lambda) = \code{pbeta}(x(q),
\code{shape1} = \gamma, \code{shape2} = \delta+1)\\.

The GKw distribution includes several special cases, such as the
Kumaraswamy, Beta, and Exponentiated Kumaraswamy distributions (see
[`dgkw`](https://evandeilton.github.io/gkwdist/reference/dgkw.md) for
details). The function utilizes numerical algorithms for computing the
regularized incomplete beta function accurately, especially near the
boundaries.

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
[`qgkw`](https://evandeilton.github.io/gkwdist/reference/qgkw.md),
[`rgkw`](https://evandeilton.github.io/gkwdist/reference/rgkw.md),
[`pbeta`](https://rdrr.io/r/stats/Beta.html)

Other cumulative distribution functions:
[`pbeta_()`](https://evandeilton.github.io/gkwdist/reference/pbeta_.md),
[`pbkw()`](https://evandeilton.github.io/gkwdist/reference/pbkw.md),
[`pekw()`](https://evandeilton.github.io/gkwdist/reference/pekw.md),
[`pkkw()`](https://evandeilton.github.io/gkwdist/reference/pkkw.md),
[`pkw()`](https://evandeilton.github.io/gkwdist/reference/pkw.md),
[`pmc()`](https://evandeilton.github.io/gkwdist/reference/pmc.md)

## Author

Lopes, J. E.

## Examples

``` r
q <- c(0.2, 0.5, 0.8)
pgkw(q, alpha = 2, beta = 3, gamma = 1.5, delta = 0.5, lambda = 1.2)
#> [1] 0.03395476 0.52305959 0.97804668
pgkw(q, alpha = 2, beta = 3, gamma = 1.5, delta = 0.5, lambda = 1.2,
    lower.tail = FALSE)  # P(X > q)
#> [1] 0.96604524 0.47694041 0.02195332
pgkw(q, alpha = 2, beta = 3, gamma = 1.5, delta = 0.5, lambda = 1.2,
    log.p = TRUE)
#> [1] -3.38272635 -0.64805989 -0.02219788

## pgkw() is the integral of dgkw()
Fq <- pgkw(0.5, alpha = 2, beta = 3, gamma = 1.5, delta = 0.5, lambda = 1.2)
all.equal(Fq, integrate(dgkw, 0, 0.5, alpha = 2, beta = 3, gamma = 1.5,
    delta = 0.5, lambda = 1.2, rel.tol = 1e-10)$value)
#> [1] TRUE
```
