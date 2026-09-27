# Negative Log-Likelihood for the KKw Distribution

Computes the negative log-likelihood function for the
Kumaraswamy-Kumaraswamy (KKw) distribution with parameters `alpha`
(\\\alpha\\), `beta` (\\\beta\\), `delta` (\\\delta\\), and `lambda`
(\\\lambda\\), given a vector of observations. This distribution is a
special case of the Generalized Kumaraswamy (GKw) distribution where
\\\gamma = 1\\.

## Usage

``` r
llkkw(par, data)
```

## Arguments

- par:

  A numeric vector of length 4 containing the distribution parameters in
  the order: `alpha` (\\\alpha \> 0\\), `beta` (\\\beta \> 0\\), `delta`
  (\\\delta \ge 0\\), `lambda` (\\\lambda \> 0\\).

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

The KKw distribution is the GKw distribution
([`dgkw`](https://evandeilton.github.io/gkwdist/reference/dgkw.md)) with
\\\gamma=1\\. Its probability density function (PDF) is: \$\$ f(x \|
\theta) = (\delta + 1) \lambda \alpha \beta x^{\alpha - 1} (1 -
x^\alpha)^{\beta - 1} \bigl\[1 - (1 - x^\alpha)^\beta\bigr\]^{\lambda -
1} \bigl\\1 - \bigl\[1 - (1 -
x^\alpha)^\beta\bigr\]^\lambda\bigr\\^{\delta} \$\$ for \\0 \< x \< 1\\
and \\\theta = (\alpha, \beta, \delta, \lambda)\\. The log-likelihood
function \\\ell(\theta \| \mathbf{x})\\ for a sample \\\mathbf{x} =
(x_1, \dots, x_n)\\ is \\\sum\_{i=1}^n \ln f(x_i \| \theta)\\: \$\$
\ell(\theta \| \mathbf{x}) = n\[\ln(\delta+1) + \ln(\lambda) +
\ln(\alpha) + \ln(\beta)\] + \sum\_{i=1}^{n} \[(\alpha-1)\ln(x_i) +
(\beta-1)\ln(v_i) + (\lambda-1)\ln(w_i) + \delta\ln(z_i)\] \$\$ where:

- \\v_i = 1 - x_i^{\alpha}\\

- \\w_i = 1 - v_i^{\beta} = 1 - (1-x_i^{\alpha})^{\beta}\\

- \\z_i = 1 - w_i^{\lambda} = 1 -
  \[1-(1-x_i^{\alpha})^{\beta}\]^{\lambda}\\

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
[`dkkw`](https://evandeilton.github.io/gkwdist/reference/dkkw.md),
[`pkkw`](https://evandeilton.github.io/gkwdist/reference/pkkw.md),
[`qkkw`](https://evandeilton.github.io/gkwdist/reference/qkkw.md),
[`rkkw`](https://evandeilton.github.io/gkwdist/reference/rkkw.md),
[`grkkw`](https://evandeilton.github.io/gkwdist/reference/grkkw.md)
(gradient),
[`hskkw`](https://evandeilton.github.io/gkwdist/reference/hskkw.md)
(Hessian), [`optim`](https://rdrr.io/r/stats/optim.html)

Other log-likelihood functions:
[`llbeta()`](https://evandeilton.github.io/gkwdist/reference/llbeta.md),
[`llbkw()`](https://evandeilton.github.io/gkwdist/reference/llbkw.md),
[`llekw()`](https://evandeilton.github.io/gkwdist/reference/llekw.md),
[`llgkw()`](https://evandeilton.github.io/gkwdist/reference/llgkw.md),
[`llkw()`](https://evandeilton.github.io/gkwdist/reference/llkw.md),
[`llmc()`](https://evandeilton.github.io/gkwdist/reference/llmc.md)

## Author

Lopes, J. E.

## Examples

``` r
set.seed(123)
x <- rkkw(1000, alpha = 2, beta = 3, delta = 0.5, lambda = 1.2)
par <- c(alpha = 2, beta = 3, delta = 0.5, lambda = 1.2)

## llkkw() is the negative log-likelihood, -sum(log f(x))
llkkw(par, x)
#> [1] -349.5218
-sum(dkkw(x, alpha = 2, beta = 3, delta = 0.5, lambda = 1.2, log = TRUE))
#> [1] -349.5218

## Maximum likelihood: minimize llkkw(), with grkkw() as its gradient
start <- gkwgetstartvalues(x, family = "kkw")
fit <- optim(start, llkkw, grkkw, data = x, method = "L-BFGS-B", lower = 1e-4)
fit$convergence  # 0: converged
#> [1] 0
## The parameters of this family are weakly identified: compare likelihoods
fit$value <= llkkw(par, x)  # at least as good as the true values
#> [1] TRUE
```
