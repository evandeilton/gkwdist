# Negative Log-Likelihood for the Beta-Kumaraswamy (BKw) Distribution

Computes the negative log-likelihood function for the Beta-Kumaraswamy
(BKw) distribution with parameters `alpha` (\\\alpha\\), `beta`
(\\\beta\\), `gamma` (\\\gamma\\), and `delta` (\\\delta\\), given a
vector of observations. This distribution is the special case of the
Generalized Kumaraswamy (GKw) distribution where \\\lambda = 1\\. This
function is typically used for maximum likelihood estimation via
numerical optimization.

## Usage

``` r
llbkw(par, data)
```

## Arguments

- par:

  A numeric vector of length 4 containing the distribution parameters in
  the order: `alpha` (\\\alpha \> 0\\), `beta` (\\\beta \> 0\\), `gamma`
  (\\\gamma \> 0\\), `delta` (\\\delta \ge 0\\).

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

The Beta-Kumaraswamy (BKw) distribution is the GKw distribution
([`dgkw`](https://evandeilton.github.io/gkwdist/reference/dgkw.md)) with
\\\lambda=1\\. Its probability density function (PDF) is: \$\$ f(x \|
\theta) = \frac{\alpha \beta}{B(\gamma, \delta+1)} x^{\alpha - 1}
\bigl(1 - x^\alpha\bigr)^{\beta(\delta+1) - 1} \bigl\[1 - \bigl(1 -
x^\alpha\bigr)^\beta\bigr\]^{\gamma - 1} \$\$ for \\0 \< x \< 1\\,
\\\theta = (\alpha, \beta, \gamma, \delta)\\, and \\B(a,b)\\ is the Beta
function ([`beta`](https://rdrr.io/r/base/Special.html)). The
log-likelihood function \\\ell(\theta \| \mathbf{x})\\ for a sample
\\\mathbf{x} = (x_1, \dots, x_n)\\ is \\\sum\_{i=1}^n \ln f(x_i \|
\theta)\\: \$\$ \ell(\theta \| \mathbf{x}) = n\[\ln(\alpha) +
\ln(\beta) - \ln B(\gamma, \delta+1)\] + \sum\_{i=1}^{n}
\[(\alpha-1)\ln(x_i) + (\beta(\delta+1)-1)\ln(v_i) +
(\gamma-1)\ln(w_i)\] \$\$ where:

- \\v_i = 1 - x_i^{\alpha}\\

- \\w_i = 1 - v_i^{\beta} = 1 - (1-x_i^{\alpha})^{\beta}\\

This function computes and returns the *negative* log-likelihood,
\\-\ell(\theta\|\mathbf{x})\\, suitable for minimization using
optimization routines like
[`optim`](https://rdrr.io/r/stats/optim.html). Numerical stability is
maintained similarly to
[`llgkw`](https://evandeilton.github.io/gkwdist/reference/llgkw.md).

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

[`llgkw`](https://evandeilton.github.io/gkwdist/reference/llgkw.md)
(parent distribution negative log-likelihood),
[`dbkw`](https://evandeilton.github.io/gkwdist/reference/dbkw.md),
[`pbkw`](https://evandeilton.github.io/gkwdist/reference/pbkw.md),
[`qbkw`](https://evandeilton.github.io/gkwdist/reference/qbkw.md),
[`rbkw`](https://evandeilton.github.io/gkwdist/reference/rbkw.md),
[`grbkw`](https://evandeilton.github.io/gkwdist/reference/grbkw.md)
(gradient),
[`hsbkw`](https://evandeilton.github.io/gkwdist/reference/hsbkw.md)
(Hessian), [`optim`](https://rdrr.io/r/stats/optim.html),
[`lbeta`](https://rdrr.io/r/base/Special.html)

Other log-likelihood functions:
[`llbeta()`](https://evandeilton.github.io/gkwdist/reference/llbeta.md),
[`llekw()`](https://evandeilton.github.io/gkwdist/reference/llekw.md),
[`llgkw()`](https://evandeilton.github.io/gkwdist/reference/llgkw.md),
[`llkkw()`](https://evandeilton.github.io/gkwdist/reference/llkkw.md),
[`llkw()`](https://evandeilton.github.io/gkwdist/reference/llkw.md),
[`llmc()`](https://evandeilton.github.io/gkwdist/reference/llmc.md)

## Author

Lopes, J. E.

## Examples

``` r
set.seed(123)
x <- rbkw(1000, alpha = 2, beta = 3, gamma = 1.5, delta = 0.5)
par <- c(alpha = 2, beta = 3, gamma = 1.5, delta = 0.5)

## llbkw() is the negative log-likelihood, -sum(log f(x))
llbkw(par, x)
#> [1] -351.0014
-sum(dbkw(x, alpha = 2, beta = 3, gamma = 1.5, delta = 0.5, log = TRUE))
#> [1] -351.0014

## Maximum likelihood: minimize llbkw(), with grbkw() as its gradient
start <- gkwgetstartvalues(x, family = "bkw")
fit <- optim(start, llbkw, grbkw, data = x, method = "L-BFGS-B", lower = 1e-4)
fit$convergence  # 0: converged
#> [1] 0
## The parameters of this family are weakly identified: compare likelihoods
fit$value <= llbkw(par, x)  # at least as good as the true values
#> [1] TRUE
```
