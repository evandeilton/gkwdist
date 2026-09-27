# Negative Log-Likelihood for the Generalized Kumaraswamy Distribution

Computes the negative log-likelihood function for the five-parameter
Generalized Kumaraswamy (GKw) distribution given a vector of
observations. This function is designed for use in optimization routines
(e.g., maximum likelihood estimation).

## Usage

``` r
llgkw(par, data)
```

## Arguments

- par:

  A numeric vector of length 5 containing the distribution parameters in
  the order: `alpha` (\\\alpha \> 0\\), `beta` (\\\beta \> 0\\), `gamma`
  (\\\gamma \> 0\\), `delta` (\\\delta \ge 0\\), `lambda` (\\\lambda \>
  0\\).

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

The probability density function (PDF) of the GKw distribution is given
in [`dgkw`](https://evandeilton.github.io/gkwdist/reference/dgkw.md).
The log-likelihood function \\\ell(\theta)\\ for a sample \\\mathbf{x} =
(x_1, \dots, x_n)\\ is: \$\$ \ell(\theta \| \mathbf{x}) =
n\ln(\lambda\alpha\beta) - n\ln B(\gamma,\delta+1) + \sum\_{i=1}^{n}
\[(\alpha-1)\ln(x_i) + (\beta-1)\ln(v_i) + (\gamma\lambda-1)\ln(w_i) +
\delta\ln(z_i)\] \$\$ where \\\theta = (\alpha, \beta, \gamma, \delta,
\lambda)\\, \\B(a,b)\\ is the Beta function
([`beta`](https://rdrr.io/r/base/Special.html)), and:

- \\v_i = 1 - x_i^{\alpha}\\

- \\w_i = 1 - v_i^{\beta} = 1 - (1-x_i^{\alpha})^{\beta}\\

- \\z_i = 1 - w_i^{\lambda} = 1 -
  \[1-(1-x_i^{\alpha})^{\beta}\]^{\lambda}\\

This function computes \\-\ell(\theta\|\mathbf{x})\\.

Numerical stability is prioritized using:

- [`lbeta`](https://rdrr.io/r/base/Special.html) function for the
  log-Beta term.

- Log-transformations of intermediate terms (\\v_i, w_i, z_i\\) and use
  of [`log1p`](https://rdrr.io/r/base/Log.html) where appropriate to
  handle values close to 0 or 1 accurately.

- Checks for invalid parameters and data.

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
[`qgkw`](https://evandeilton.github.io/gkwdist/reference/qgkw.md),
[`rgkw`](https://evandeilton.github.io/gkwdist/reference/rgkw.md),
[`grgkw`](https://evandeilton.github.io/gkwdist/reference/grgkw.md),
[`hsgkw`](https://evandeilton.github.io/gkwdist/reference/hsgkw.md)
(gradient and Hessian), [`optim`](https://rdrr.io/r/stats/optim.html),
[`lbeta`](https://rdrr.io/r/base/Special.html),
[`log1p`](https://rdrr.io/r/base/Log.html)

Other log-likelihood functions:
[`llbeta()`](https://evandeilton.github.io/gkwdist/reference/llbeta.md),
[`llbkw()`](https://evandeilton.github.io/gkwdist/reference/llbkw.md),
[`llekw()`](https://evandeilton.github.io/gkwdist/reference/llekw.md),
[`llkkw()`](https://evandeilton.github.io/gkwdist/reference/llkkw.md),
[`llkw()`](https://evandeilton.github.io/gkwdist/reference/llkw.md),
[`llmc()`](https://evandeilton.github.io/gkwdist/reference/llmc.md)

## Author

Lopes, J. E.

## Examples

``` r
set.seed(123)
x <- rgkw(1000, alpha = 2, beta = 3, gamma = 1.5, delta = 0.5, lambda = 1.2)
par <- c(alpha = 2, beta = 3, gamma = 1.5, delta = 0.5, lambda = 1.2)

## llgkw() is the negative log-likelihood, -sum(log f(x))
llgkw(par, x)
#> [1] -392.775
-sum(dgkw(x, alpha = 2, beta = 3, gamma = 1.5, delta = 0.5, lambda = 1.2,
    log = TRUE))
#> [1] -392.775

## Maximum likelihood: minimize llgkw(), with grgkw() as its gradient
start <- gkwgetstartvalues(x, family = "gkw")
fit <- optim(start, llgkw, grgkw, data = x, method = "L-BFGS-B", lower = 1e-4)
fit$convergence  # 0: converged
#> [1] 0
## The parameters of this family are weakly identified: compare likelihoods
fit$value <= llgkw(par, x)  # at least as good as the true values
#> [1] TRUE
```
