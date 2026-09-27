# Cumulative Distribution Function (CDF) of the McDonald (Mc)/Beta Power Distribution

Computes the cumulative distribution function (CDF), \\F(q) = P(X \le
q)\\, for the McDonald (Mc) distribution (also known as Beta Power) with
parameters `gamma` (\\\gamma\\), `delta` (\\\delta\\), and `lambda`
(\\\lambda\\). This distribution is defined on the interval (0, 1) and
is a special case of the Generalized Kumaraswamy (GKw) distribution
where \\\alpha = 1\\ and \\\beta = 1\\.

## Usage

``` r
pmc(q, gamma = 1, delta = 0, lambda = 1, lower.tail = TRUE, log.p = FALSE)
```

## Arguments

- q:

  Vector of quantiles (values generally between 0 and 1).

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
determined by the recycling rule applied to the arguments (`q`, `gamma`,
`delta`, `lambda`). When `lower.tail = TRUE`, returns `0` (or `-Inf` if
`log.p = TRUE`) for `q <= 0` and `1` (or `0` if `log.p = TRUE`) for
`q >= 1`. An out-of-bound or missing parameter is an error, not a return
value: the wrapper stops with a message naming the parameter. An
infinite parameter is not currently intercepted there and reaches the
C++ layer, which treats it as invalid. Boundary return values are
adjusted accordingly for `lower.tail = FALSE`.

## Details

The McDonald (Mc) distribution is a special case of the five-parameter
Generalized Kumaraswamy (GKw) distribution
([`pgkw`](https://evandeilton.github.io/gkwdist/reference/pgkw.md))
obtained by setting parameters \\\alpha = 1\\ and \\\beta = 1\\.

The CDF of the GKw distribution is \\F\_{GKw}(q) = I\_{y(q)}(\gamma,
\delta+1)\\, where \\y(q) = \[1-(1-q^{\alpha})^{\beta}\]^{\lambda}\\ and
\\I_x(a,b)\\ is the regularized incomplete beta function
([`pbeta`](https://rdrr.io/r/stats/Beta.html)). Setting \\\alpha=1\\ and
\\\beta=1\\ simplifies \\y(q)\\ to \\q^\lambda\\, yielding the Mc CDF:
\$\$ F(q; \gamma, \delta, \lambda) = I\_{q^\lambda}(\gamma, \delta+1)
\$\$ This is evaluated using the
[`pbeta`](https://rdrr.io/r/stats/Beta.html) function as
`pbeta(q^lambda, shape1 = gamma, shape2 = delta + 1)`.

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

[`pgkw`](https://evandeilton.github.io/gkwdist/reference/pgkw.md)
(parent distribution CDF),
[`dmc`](https://evandeilton.github.io/gkwdist/reference/dmc.md),
[`qmc`](https://evandeilton.github.io/gkwdist/reference/qmc.md),
[`rmc`](https://evandeilton.github.io/gkwdist/reference/rmc.md) (other
Mc functions), [`pbeta`](https://rdrr.io/r/stats/Beta.html)

Other cumulative distribution functions:
[`pbeta_()`](https://evandeilton.github.io/gkwdist/reference/pbeta_.md),
[`pbkw()`](https://evandeilton.github.io/gkwdist/reference/pbkw.md),
[`pekw()`](https://evandeilton.github.io/gkwdist/reference/pekw.md),
[`pgkw()`](https://evandeilton.github.io/gkwdist/reference/pgkw.md),
[`pkkw()`](https://evandeilton.github.io/gkwdist/reference/pkkw.md),
[`pkw()`](https://evandeilton.github.io/gkwdist/reference/pkw.md)

## Author

Lopes, J. E.

## Examples

``` r
q <- c(0.2, 0.5, 0.8)
pmc(q, gamma = 0.5, delta = 5, lambda = 3)
#> [1] 0.2389267 0.7850539 0.9959906
pmc(q, gamma = 0.5, delta = 5, lambda = 3, lower.tail = FALSE)  # P(X > q)
#> [1] 0.761073272 0.214946121 0.004009438
pmc(q, gamma = 0.5, delta = 5, lambda = 3, log.p = TRUE)
#> [1] -1.431598352 -0.242002927 -0.004017497

## pmc() is the integral of dmc()
Fq <- pmc(0.5, gamma = 0.5, delta = 5, lambda = 3)
all.equal(Fq, integrate(dmc, 0, 0.5, gamma = 0.5, delta = 5, lambda = 3,
    rel.tol = 1e-10)$value)
#> [1] TRUE
```
