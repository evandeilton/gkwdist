# Cumulative Distribution Function (CDF) of the EKw Distribution

Computes the cumulative distribution function (CDF), \\P(X \le q)\\, for
the Exponentiated Kumaraswamy (EKw) distribution with parameters `alpha`
(\\\alpha\\), `beta` (\\\beta\\), and `lambda` (\\\lambda\\). This
distribution is defined on the interval (0, 1) and is a special case of
the Generalized Kumaraswamy (GKw) distribution where \\\gamma = 1\\ and
\\\delta = 0\\.

## Usage

``` r
pekw(q, alpha = 1, beta = 1, lambda = 1, lower.tail = TRUE, log.p = FALSE)
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

- lambda:

  Shape parameter `lambda` \> 0 (exponent parameter). Can be a scalar or
  a vector. Default: 1.0.

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
`beta`, `lambda`). When `lower.tail = TRUE`, returns `0` (or `-Inf` if
`log.p = TRUE`) for `q <= 0` and `1` (or `0` if `log.p = TRUE`) for
`q >= 1`. An out-of-bound or missing parameter is an error, not a return
value: the wrapper stops with a message naming the parameter. An
infinite parameter is not currently intercepted there and reaches the
C++ layer, which treats it as invalid. Boundary return values are
adjusted accordingly for `lower.tail = FALSE`.

## Details

The Exponentiated Kumaraswamy (EKw) distribution is a special case of
the five-parameter Generalized Kumaraswamy distribution
([`pgkw`](https://evandeilton.github.io/gkwdist/reference/pgkw.md))
obtained by setting parameters \\\gamma = 1\\ and \\\delta = 0\\.

The CDF of the GKw distribution is \\F\_{GKw}(q) = I\_{y(q)}(\gamma,
\delta+1)\\, where \\y(q) = \[1-(1-q^{\alpha})^{\beta}\]^{\lambda}\\ and
\\I_x(a,b)\\ is the regularized incomplete beta function
([`pbeta`](https://rdrr.io/r/stats/Beta.html)). Setting \\\gamma=1\\ and
\\\delta=0\\ gives \\I\_{y(q)}(1, 1)\\. Since \\I_x(1, 1) = x\\, the CDF
simplifies to \\y(q)\\: \$\$ F(q; \alpha, \beta, \lambda) = \bigl\[1 -
(1 - q^\alpha)^\beta \bigr\]^\lambda \$\$ for \\0 \< q \< 1\\. The
implementation uses this closed-form expression for efficiency and
handles `lower.tail` and `log.p` arguments appropriately.

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

[`pgkw`](https://evandeilton.github.io/gkwdist/reference/pgkw.md)
(parent distribution CDF),
[`dekw`](https://evandeilton.github.io/gkwdist/reference/dekw.md),
[`qekw`](https://evandeilton.github.io/gkwdist/reference/qekw.md),
[`rekw`](https://evandeilton.github.io/gkwdist/reference/rekw.md) (other
EKw functions),

Other cumulative distribution functions:
[`pbeta_()`](https://evandeilton.github.io/gkwdist/reference/pbeta_.md),
[`pbkw()`](https://evandeilton.github.io/gkwdist/reference/pbkw.md),
[`pgkw()`](https://evandeilton.github.io/gkwdist/reference/pgkw.md),
[`pkkw()`](https://evandeilton.github.io/gkwdist/reference/pkkw.md),
[`pkw()`](https://evandeilton.github.io/gkwdist/reference/pkw.md),
[`pmc()`](https://evandeilton.github.io/gkwdist/reference/pmc.md)

## Author

Lopes, J. E.

## Examples

``` r
q <- c(0.2, 0.5, 0.8)
pekw(q, alpha = 2, beta = 3, lambda = 1.2)
#> [1] 0.07482254 0.51811492 0.94427733
pekw(q, alpha = 2, beta = 3, lambda = 1.2, lower.tail = FALSE)  # P(X > q)
#> [1] 0.92517746 0.48188508 0.05572267
pekw(q, alpha = 2, beta = 3, lambda = 1.2, log.p = TRUE)
#> [1] -2.59263616 -0.65755820 -0.05733537

## pekw() is the integral of dekw()
Fq <- pekw(0.5, alpha = 2, beta = 3, lambda = 1.2)
all.equal(Fq, integrate(dekw, 0, 0.5, alpha = 2, beta = 3, lambda = 1.2,
    rel.tol = 1e-10)$value)
#> [1] TRUE
```
