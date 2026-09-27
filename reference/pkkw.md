# Cumulative Distribution Function (CDF) of the KKw Distribution

Computes the cumulative distribution function (CDF), \\P(X \le q)\\, for
the Kumaraswamy-Kumaraswamy (KKw) distribution with parameters `alpha`
(\\\alpha\\), `beta` (\\\beta\\), `delta` (\\\delta\\), and `lambda`
(\\\lambda\\). This distribution is defined on the interval (0, 1).

## Usage

``` r
pkkw(
  q,
  alpha = 1,
  beta = 1,
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
`beta`, `delta`, `lambda`). When `lower.tail = TRUE`, returns `0` (or
`-Inf` if `log.p = TRUE`) for `q <= 0` and `1` (or `0` if
`log.p = TRUE`) for `q >= 1`. An out-of-bound or missing parameter is an
error, not a return value: the wrapper stops with a message naming the
parameter. An infinite parameter is not currently intercepted there and
reaches the C++ layer, which treats it as invalid. Boundary return
values are adjusted accordingly for `lower.tail = FALSE`.

## Details

The Kumaraswamy-Kumaraswamy (KKw) distribution is a special case of the
five-parameter Generalized Kumaraswamy distribution
([`pgkw`](https://evandeilton.github.io/gkwdist/reference/pgkw.md))
obtained by setting the shape parameter \\\gamma = 1\\.

The CDF of the GKw distribution is \\F\_{GKw}(q) = I\_{y(q)}(\gamma,
\delta+1)\\, where \\y(q) = \[1-(1-q^{\alpha})^{\beta}\]^{\lambda}\\ and
\\I_x(a,b)\\ is the regularized incomplete beta function
([`pbeta`](https://rdrr.io/r/stats/Beta.html)). Setting \\\gamma=1\\
utilizes the property \\I_x(1, b) = 1 - (1-x)^b\\, yielding the KKw CDF:
\$\$ F(q; \alpha, \beta, \delta, \lambda) = 1 - \bigl\\1 - \bigl\[1 -
(1 - q^\alpha)^\beta\bigr\]^\lambda\bigr\\^{\delta + 1} \$\$ for \\0 \<
q \< 1\\.

The implementation uses this closed-form expression for efficiency and
handles `lower.tail` and `log.p` arguments appropriately.

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

[`pgkw`](https://evandeilton.github.io/gkwdist/reference/pgkw.md)
(parent distribution CDF),
[`dkkw`](https://evandeilton.github.io/gkwdist/reference/dkkw.md),
[`qkkw`](https://evandeilton.github.io/gkwdist/reference/qkkw.md),
[`rkkw`](https://evandeilton.github.io/gkwdist/reference/rkkw.md),
[`pbeta`](https://rdrr.io/r/stats/Beta.html)

Other cumulative distribution functions:
[`pbeta_()`](https://evandeilton.github.io/gkwdist/reference/pbeta_.md),
[`pbkw()`](https://evandeilton.github.io/gkwdist/reference/pbkw.md),
[`pekw()`](https://evandeilton.github.io/gkwdist/reference/pekw.md),
[`pgkw()`](https://evandeilton.github.io/gkwdist/reference/pgkw.md),
[`pkw()`](https://evandeilton.github.io/gkwdist/reference/pkw.md),
[`pmc()`](https://evandeilton.github.io/gkwdist/reference/pmc.md)

## Author

Lopes, J. E.

## Examples

``` r
q <- c(0.2, 0.5, 0.8)
pkkw(q, alpha = 2, beta = 3, delta = 0.5, lambda = 1.2)
#> [1] 0.1101075 0.6654853 0.9868463
# P(X > q)
pkkw(q, alpha = 2, beta = 3, delta = 0.5, lambda = 1.2, lower.tail = FALSE)
#> [1] 0.8898925 0.3345147 0.0131537
pkkw(q, alpha = 2, beta = 3, delta = 0.5, lambda = 1.2, log.p = TRUE)
#> [1] -2.20629852 -0.40723874 -0.01324097

## pkkw() is the integral of dkkw()
Fq <- pkkw(0.5, alpha = 2, beta = 3, delta = 0.5, lambda = 1.2)
all.equal(Fq, integrate(dkkw, 0, 0.5, alpha = 2, beta = 3, delta = 0.5,
    lambda = 1.2, rel.tol = 1e-10)$value)
#> [1] TRUE
```
