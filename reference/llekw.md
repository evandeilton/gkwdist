# Negative Log-Likelihood for the Exponentiated Kumaraswamy (EKw) Distribution

Computes the negative log-likelihood function for the Exponentiated
Kumaraswamy (EKw) distribution with parameters `alpha` (\\\alpha\\),
`beta` (\\\beta\\), and `lambda` (\\\lambda\\), given a vector of
observations. This distribution is the special case of the Generalized
Kumaraswamy (GKw) distribution where \\\gamma = 1\\ and \\\delta = 0\\.
This function is suitable for maximum likelihood estimation.

## Usage

``` r
llekw(par, data)
```

## Arguments

- par:

  A numeric vector of length 3 containing the distribution parameters in
  the order: `alpha` (\\\alpha \> 0\\), `beta` (\\\beta \> 0\\),
  `lambda` (\\\lambda \> 0\\).

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

The Exponentiated Kumaraswamy (EKw) distribution is the GKw distribution
([`dekw`](https://evandeilton.github.io/gkwdist/reference/dekw.md)) with
\\\gamma=1\\ and \\\delta=0\\. Its probability density function (PDF)
is: \$\$ f(x \| \theta) = \lambda \alpha \beta x^{\alpha-1} (1 -
x^\alpha)^{\beta-1} \bigl\[1 - (1 - x^\alpha)^\beta \bigr\]^{\lambda -
1} \$\$ for \\0 \< x \< 1\\ and \\\theta = (\alpha, \beta, \lambda)\\.
The log-likelihood function \\\ell(\theta \| \mathbf{x})\\ for a sample
\\\mathbf{x} = (x_1, \dots, x_n)\\ is \\\sum\_{i=1}^n \ln f(x_i \|
\theta)\\: \$\$ \ell(\theta \| \mathbf{x}) = n\[\ln(\lambda) +
\ln(\alpha) + \ln(\beta)\] + \sum\_{i=1}^{n} \[(\alpha-1)\ln(x_i) +
(\beta-1)\ln(v_i) + (\lambda-1)\ln(w_i)\] \$\$ where:

- \\v_i = 1 - x_i^{\alpha}\\

- \\w_i = 1 - v_i^{\beta} = 1 - (1-x_i^{\alpha})^{\beta}\\

This function computes and returns the *negative* log-likelihood,
\\-\ell(\theta\|\mathbf{x})\\, suitable for minimization using
optimization routines like
[`optim`](https://rdrr.io/r/stats/optim.html). Numerical stability is
maintained similarly to
[`llgkw`](https://evandeilton.github.io/gkwdist/reference/llgkw.md).

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

[`llgkw`](https://evandeilton.github.io/gkwdist/reference/llgkw.md)
(parent distribution negative log-likelihood),
[`dekw`](https://evandeilton.github.io/gkwdist/reference/dekw.md),
[`pekw`](https://evandeilton.github.io/gkwdist/reference/pekw.md),
[`qekw`](https://evandeilton.github.io/gkwdist/reference/qekw.md),
[`rekw`](https://evandeilton.github.io/gkwdist/reference/rekw.md),
[`grekw`](https://evandeilton.github.io/gkwdist/reference/grekw.md)
(gradient),
[`hsekw`](https://evandeilton.github.io/gkwdist/reference/hsekw.md)
(Hessian), [`optim`](https://rdrr.io/r/stats/optim.html)

Other log-likelihood functions:
[`llbeta()`](https://evandeilton.github.io/gkwdist/reference/llbeta.md),
[`llbkw()`](https://evandeilton.github.io/gkwdist/reference/llbkw.md),
[`llgkw()`](https://evandeilton.github.io/gkwdist/reference/llgkw.md),
[`llkkw()`](https://evandeilton.github.io/gkwdist/reference/llkkw.md),
[`llkw()`](https://evandeilton.github.io/gkwdist/reference/llkw.md),
[`llmc()`](https://evandeilton.github.io/gkwdist/reference/llmc.md)

## Author

Lopes, J. E.

## Examples

``` r
set.seed(123)
x <- rekw(1000, alpha = 2, beta = 3, lambda = 1.2)
par <- c(alpha = 2, beta = 3, lambda = 1.2)

## llekw() is the negative log-likelihood, -sum(log f(x))
llekw(par, x)
#> [1] -245.6122
-sum(dekw(x, alpha = 2, beta = 3, lambda = 1.2, log = TRUE))
#> [1] -245.6122

## Maximum likelihood: minimize llekw(), with grekw() as its gradient
start <- gkwgetstartvalues(x, family = "ekw")
fit <- optim(start, llekw, grekw, data = x, method = "L-BFGS-B", lower = 1e-4)
fit$convergence  # 0: converged
#> [1] 0
fit$par
#>    alpha     beta   lambda 
#> 2.110857 3.121146 1.125558 
fit$value <= llekw(par, x)  # at least as good as the true values
#> [1] TRUE
```
