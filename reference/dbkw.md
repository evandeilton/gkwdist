# Density of the Beta-Kumaraswamy (BKw) Distribution

Computes the probability density function (PDF) for the Beta-Kumaraswamy
(BKw) distribution with parameters `alpha` (\\\alpha\\), `beta`
(\\\beta\\), `gamma` (\\\gamma\\), and `delta` (\\\delta\\). This
distribution is defined on the interval (0, 1).

## Usage

``` r
dbkw(x, alpha = 1, beta = 1, gamma = 1, delta = 0, log = FALSE)
```

## Arguments

- x:

  Vector of quantiles (values between 0 and 1).

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

- log:

  Logical; if `TRUE`, the logarithm of the density is returned
  (\\\log(f(x))\\). Default: `FALSE`.

## Value

A vector of density values (\\f(x)\\) or log-density values
(\\\log(f(x))\\). The length of the result is determined by the
recycling rule applied to the arguments (`x`, `alpha`, `beta`, `gamma`,
`delta`). Returns `0` (or `-Inf` if `log = TRUE`) for `x` strictly
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

The probability density function (PDF) of the Beta-Kumaraswamy (BKw)
distribution is given by: \$\$ f(x; \alpha, \beta, \gamma, \delta) =
\frac{\alpha \beta}{B(\gamma, \delta+1)} x^{\alpha - 1} \bigl(1 -
x^\alpha\bigr)^{\beta(\delta+1) - 1} \bigl\[1 - \bigl(1 -
x^\alpha\bigr)^\beta\bigr\]^{\gamma - 1} \$\$ for \\0 \< x \< 1\\, where
\\B(a,b)\\ is the Beta function
([`beta`](https://rdrr.io/r/base/Special.html)).

The BKw distribution is a special case of the five-parameter Generalized
Kumaraswamy (GKw) distribution
([`dgkw`](https://evandeilton.github.io/gkwdist/reference/dgkw.md))
obtained by setting the parameter \\\lambda = 1\\. Numerical evaluation
is performed using algorithms similar to those for `dgkw`, ensuring
stability.

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

[`dgkw`](https://evandeilton.github.io/gkwdist/reference/dgkw.md)
(parent distribution density),
[`pbkw`](https://evandeilton.github.io/gkwdist/reference/pbkw.md),
[`qbkw`](https://evandeilton.github.io/gkwdist/reference/qbkw.md),
[`rbkw`](https://evandeilton.github.io/gkwdist/reference/rbkw.md) (other
BKw functions),

Other density functions:
[`dbeta_()`](https://evandeilton.github.io/gkwdist/reference/dbeta_.md),
[`dekw()`](https://evandeilton.github.io/gkwdist/reference/dekw.md),
[`dgkw()`](https://evandeilton.github.io/gkwdist/reference/dgkw.md),
[`dkkw()`](https://evandeilton.github.io/gkwdist/reference/dkkw.md),
[`dkw()`](https://evandeilton.github.io/gkwdist/reference/dkw.md),
[`dmc()`](https://evandeilton.github.io/gkwdist/reference/dmc.md)

## Author

Lopes, J. E.

## Examples

``` r
x <- c(0.1, 0.3, 0.5, 0.7, 0.9)
dbkw(x, alpha = 2, beta = 3, gamma = 1.5, delta = 0.5)
#> [1] 0.25421436 1.63569904 2.12220174 0.94358703 0.04097103
dbkw(x, alpha = 2, beta = 3, gamma = 1.5, delta = 0.5, log = TRUE)
#> [1] -1.36957743  0.49207026  0.75245411 -0.05806668 -3.19488993

## BKw is GKw with lambda = 1
all.equal(dbkw(x, 2, 3, 1.5, 0.5), dgkw(x, 2, 3, 1.5, 0.5, lambda = 1))
#> [1] TRUE

## The density integrates to one
integrate(dbkw, 0, 1, alpha = 2, beta = 3, gamma = 1.5, delta = 0.5,
    rel.tol = 1e-10)
#> 1 with absolute error < 2.5e-11

curve(dbkw(x, alpha = 2, beta = 3, gamma = 1.5, delta = 0.5), from = 0,
    to = 1, ylab = "density")

```
