# Negative Log-Likelihood for the McDonald (Mc)/Beta Power Distribution

Computes the negative log-likelihood function for the McDonald (Mc)
distribution (also known as Beta Power) with parameters `gamma`
(\\\gamma\\), `delta` (\\\delta\\), and `lambda` (\\\lambda\\), given a
vector of observations. This distribution is the special case of the
Generalized Kumaraswamy (GKw) distribution where \\\alpha = 1\\ and
\\\beta = 1\\. This function is suitable for maximum likelihood
estimation.

## Usage

``` r
llmc(par, data)
```

## Arguments

- par:

  A numeric vector of length 3 containing the distribution parameters in
  the order: `gamma` (\\\gamma \> 0\\), `delta` (\\\delta \ge 0\\),
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

The McDonald (Mc) distribution is the GKw distribution
([`dmc`](https://evandeilton.github.io/gkwdist/reference/dmc.md)) with
\\\alpha=1\\ and \\\beta=1\\. Its probability density function (PDF) is:
\$\$ f(x \| \theta) = \frac{\lambda}{B(\gamma,\delta+1)} x^{\gamma
\lambda - 1} (1 - x^\lambda)^\delta \$\$ for \\0 \< x \< 1\\, \\\theta =
(\gamma, \delta, \lambda)\\, and \\B(a,b)\\ is the Beta function
([`beta`](https://rdrr.io/r/base/Special.html)). The log-likelihood
function \\\ell(\theta \| \mathbf{x})\\ for a sample \\\mathbf{x} =
(x_1, \dots, x_n)\\ is \\\sum\_{i=1}^n \ln f(x_i \| \theta)\\: \$\$
\ell(\theta \| \mathbf{x}) = n\[\ln(\lambda) - \ln B(\gamma,
\delta+1)\] + \sum\_{i=1}^{n} \[(\gamma\lambda - 1)\ln(x_i) +
\delta\ln(1 - x_i^\lambda)\] \$\$ This function computes and returns the
*negative* log-likelihood, \\-\ell(\theta\|\mathbf{x})\\, suitable for
minimization using optimization routines like
[`optim`](https://rdrr.io/r/stats/optim.html). Numerical stability is
maintained, including using the log-gamma function
([`lgamma`](https://rdrr.io/r/base/Special.html)) for the Beta function
term.

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

[`llgkw`](https://evandeilton.github.io/gkwdist/reference/llgkw.md)
(parent distribution negative log-likelihood),
[`dmc`](https://evandeilton.github.io/gkwdist/reference/dmc.md),
[`pmc`](https://evandeilton.github.io/gkwdist/reference/pmc.md),
[`qmc`](https://evandeilton.github.io/gkwdist/reference/qmc.md),
[`rmc`](https://evandeilton.github.io/gkwdist/reference/rmc.md),
[`grmc`](https://evandeilton.github.io/gkwdist/reference/grmc.md)
(gradient),
[`hsmc`](https://evandeilton.github.io/gkwdist/reference/hsmc.md)
(Hessian), [`optim`](https://rdrr.io/r/stats/optim.html),
[`lbeta`](https://rdrr.io/r/base/Special.html)

Other log-likelihood functions:
[`llbeta()`](https://evandeilton.github.io/gkwdist/reference/llbeta.md),
[`llbkw()`](https://evandeilton.github.io/gkwdist/reference/llbkw.md),
[`llekw()`](https://evandeilton.github.io/gkwdist/reference/llekw.md),
[`llgkw()`](https://evandeilton.github.io/gkwdist/reference/llgkw.md),
[`llkkw()`](https://evandeilton.github.io/gkwdist/reference/llkkw.md),
[`llkw()`](https://evandeilton.github.io/gkwdist/reference/llkw.md)

## Author

Lopes, J. E.

## Examples

``` r
set.seed(123)
x <- rmc(1000, gamma = 0.5, delta = 5, lambda = 3)
par <- c(gamma = 0.5, delta = 5, lambda = 3)

## llmc() is the negative log-likelihood, -sum(log f(x))
llmc(par, x)
#> [1] -349.7525
-sum(dmc(x, gamma = 0.5, delta = 5, lambda = 3, log = TRUE))
#> [1] -349.7525

## Maximum likelihood: minimize llmc(), with grmc() as its gradient
start <- gkwgetstartvalues(x, family = "mc")
fit <- optim(start, llmc, grmc, data = x, method = "L-BFGS-B", lower = 1e-4)
fit$convergence  # 0: converged
#> [1] 0
fit$par
#>     gamma     delta    lambda 
#> 0.4712336 5.2348706 3.0191417 
fit$value <= llmc(par, x)  # at least as good as the true values
#> [1] TRUE
```
