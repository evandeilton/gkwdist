# Cumulative Distribution Function (CDF) of the Kumaraswamy (Kw) Distribution

Computes the cumulative distribution function (CDF), \\P(X \le q)\\, for
the two-parameter Kumaraswamy (Kw) distribution with shape parameters
`alpha` (\\\alpha\\) and `beta` (\\\beta\\). This distribution is
defined on the interval (0, 1).

## Usage

``` r
pkw(q, alpha = 1, beta = 1, lower.tail = TRUE, log.p = FALSE)
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
`beta`). When `lower.tail = TRUE`, returns `0` (or `-Inf` if
`log.p = TRUE`) for `q <= 0` and `1` (or `0` if `log.p = TRUE`) for
`q >= 1`. An out-of-bound or missing parameter is an error, not a return
value: the wrapper stops with a message naming the parameter. An
infinite parameter is not currently intercepted there and reaches the
C++ layer, which treats it as invalid. Boundary return values are
adjusted accordingly for `lower.tail = FALSE`.

## Details

The cumulative distribution function (CDF) of the Kumaraswamy (Kw)
distribution is given by: \$\$ F(x; \alpha, \beta) = 1 - (1 -
x^\alpha)^\beta \$\$ for \\0 \< x \< 1\\, \\\alpha \> 0\\, and \\\beta
\> 0\\.

The Kw distribution is a special case of several generalized
distributions:

- Generalized Kumaraswamy
  ([`pgkw`](https://evandeilton.github.io/gkwdist/reference/pgkw.md))
  with \\\gamma=1, \delta=0, \lambda=1\\.

- Exponentiated Kumaraswamy
  ([`pekw`](https://evandeilton.github.io/gkwdist/reference/pekw.md))
  with \\\lambda=1\\.

- Kumaraswamy-Kumaraswamy
  ([`pkkw`](https://evandeilton.github.io/gkwdist/reference/pkkw.md))
  with \\\delta=0, \lambda=1\\.

The implementation uses the closed-form expression for efficiency.

## References

Kumaraswamy, P. (1980). A generalized probability density function for
double-bounded random processes. *Journal of Hydrology*, *46*(1-2),
79-88.
[doi:10.1016/0022-1694(80)90036-0](https://doi.org/10.1016/0022-1694%2880%2990036-0)

Jones, M. C. (2009). Kumaraswamy's distribution: A beta-type
distribution with some tractability advantages. *Statistical
Methodology*, *6*(1), 70-81.
[doi:10.1016/j.stamet.2008.04.001](https://doi.org/10.1016/j.stamet.2008.04.001)

## See also

[`pgkw`](https://evandeilton.github.io/gkwdist/reference/pgkw.md),
[`pekw`](https://evandeilton.github.io/gkwdist/reference/pekw.md),
[`pkkw`](https://evandeilton.github.io/gkwdist/reference/pkkw.md)
(related generalized CDFs),
[`dkw`](https://evandeilton.github.io/gkwdist/reference/dkw.md),
[`qkw`](https://evandeilton.github.io/gkwdist/reference/qkw.md),
[`rkw`](https://evandeilton.github.io/gkwdist/reference/rkw.md) (other
Kw functions), [`pbeta`](https://rdrr.io/r/stats/Beta.html)

Other cumulative distribution functions:
[`pbeta_()`](https://evandeilton.github.io/gkwdist/reference/pbeta_.md),
[`pbkw()`](https://evandeilton.github.io/gkwdist/reference/pbkw.md),
[`pekw()`](https://evandeilton.github.io/gkwdist/reference/pekw.md),
[`pgkw()`](https://evandeilton.github.io/gkwdist/reference/pgkw.md),
[`pkkw()`](https://evandeilton.github.io/gkwdist/reference/pkkw.md),
[`pmc()`](https://evandeilton.github.io/gkwdist/reference/pmc.md)

## Author

Lopes, J. E.

## Examples

``` r
q <- c(0.2, 0.5, 0.8)
pkw(q, alpha = 2, beta = 3)
#> [1] 0.115264 0.578125 0.953344
pkw(q, alpha = 2, beta = 3, lower.tail = FALSE)  # P(X > q)
#> [1] 0.884736 0.421875 0.046656
pkw(q, alpha = 2, beta = 3, log.p = TRUE)
#> [1] -2.16053013 -0.54796517 -0.04777948

## pkw() is the integral of dkw()
Fq <- pkw(0.5, alpha = 2, beta = 3)
all.equal(Fq, integrate(dkw, 0, 0.5, alpha = 2, beta = 3, rel.tol = 1e-10)$value)
#> [1] TRUE
```
