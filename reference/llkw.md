# Negative Log-Likelihood for the Kumaraswamy (Kw) Distribution

Computes the negative log-likelihood function for the two-parameter
Kumaraswamy (Kw) distribution with parameters `alpha` (\\\alpha\\) and
`beta` (\\\beta\\), given a vector of observations. This function is
suitable for maximum likelihood estimation.

## Usage

``` r
llkw(par, data)
```

## Arguments

- par:

  A numeric vector of length 2 containing the distribution parameters in
  the order: `alpha` (\\\alpha \> 0\\), `beta` (\\\beta \> 0\\).

- data:

  A numeric vector of observations. All values must be strictly between
  0 and 1 (exclusive).

## Value

Returns a single `double` value representing the negative log-likelihood
(\\-\ell(\theta\|\mathbf{x})\\). Returns `Inf` if any parameter values
in `par` are invalid according to their constraints, or if any value in
`data` is not in the interval (0, 1); in the latter case a warning
naming `data` is also signaled, because an infinite objective offers an
optimizer no gradient direction to follow and more often means a sample
on the wrong scale than a genuine fit failure.

## Details

The Kumaraswamy (Kw) distribution's probability density function (PDF)
is (see
[`dkw`](https://evandeilton.github.io/gkwdist/reference/dkw.md)): \$\$
f(x \| \theta) = \alpha \beta x^{\alpha-1} (1 - x^\alpha)^{\beta-1} \$\$
for \\0 \< x \< 1\\ and \\\theta = (\alpha, \beta)\\. The log-likelihood
function \\\ell(\theta \| \mathbf{x})\\ for a sample \\\mathbf{x} =
(x_1, \dots, x_n)\\ is \\\sum\_{i=1}^n \ln f(x_i \| \theta)\\: \$\$
\ell(\theta \| \mathbf{x}) = n\[\ln(\alpha) + \ln(\beta)\] +
\sum\_{i=1}^{n} \[(\alpha-1)\ln(x_i) + (\beta-1)\ln(v_i)\] \$\$ where
\\v_i = 1 - x_i^{\alpha}\\. This function computes and returns the
*negative* log-likelihood, \\-\ell(\theta\|\mathbf{x})\\, suitable for
minimization using optimization routines like
[`optim`](https://rdrr.io/r/stats/optim.html). It is equivalent to the
negative log-likelihood of the GKw distribution
([`llgkw`](https://evandeilton.github.io/gkwdist/reference/llgkw.md))
evaluated at \\\gamma=1, \delta=0, \lambda=1\\.

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

[`llgkw`](https://evandeilton.github.io/gkwdist/reference/llgkw.md)
(parent distribution negative log-likelihood),
[`dkw`](https://evandeilton.github.io/gkwdist/reference/dkw.md),
[`pkw`](https://evandeilton.github.io/gkwdist/reference/pkw.md),
[`qkw`](https://evandeilton.github.io/gkwdist/reference/qkw.md),
[`rkw`](https://evandeilton.github.io/gkwdist/reference/rkw.md),
[`grkw`](https://evandeilton.github.io/gkwdist/reference/grkw.md)
(gradient),
[`hskw`](https://evandeilton.github.io/gkwdist/reference/hskw.md)
(Hessian), [`optim`](https://rdrr.io/r/stats/optim.html)

Other log-likelihood functions:
[`llbeta()`](https://evandeilton.github.io/gkwdist/reference/llbeta.md),
[`llbkw()`](https://evandeilton.github.io/gkwdist/reference/llbkw.md),
[`llekw()`](https://evandeilton.github.io/gkwdist/reference/llekw.md),
[`llgkw()`](https://evandeilton.github.io/gkwdist/reference/llgkw.md),
[`llkkw()`](https://evandeilton.github.io/gkwdist/reference/llkkw.md),
[`llmc()`](https://evandeilton.github.io/gkwdist/reference/llmc.md)

## Author

Lopes, J. E.

## Examples

``` r
set.seed(123)
x <- rkw(1000, alpha = 2, beta = 3)
par <- c(alpha = 2, beta = 3)

## llkw() is the negative log-likelihood, -sum(log f(x))
llkw(par, x)
#> [1] -213.2127
-sum(dkw(x, alpha = 2, beta = 3, log = TRUE))
#> [1] -213.2127

## Maximum likelihood: minimize llkw(), with grkw() as its gradient
start <- gkwgetstartvalues(x, family = "kw")
fit <- optim(start, llkw, grkw, data = x, method = "L-BFGS-B", lower = 1e-4)
fit$convergence  # 0: converged
#> [1] 0
fit$par
#>    alpha     beta 
#> 2.008977 3.059364 
fit$value <= llkw(par, x)  # at least as good as the true values
#> [1] TRUE
```
