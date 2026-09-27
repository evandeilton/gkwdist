# Negative Log-Likelihood for the Beta Distribution (gamma, delta+1 Parameterization)

Computes the negative log-likelihood function for the standard Beta
distribution, using a parameterization common in generalized
distribution families. The distribution is parameterized by `gamma`
(\\\gamma\\) and `delta` (\\\delta\\), corresponding to the standard
Beta distribution with shape parameters `shape1 = gamma` and
`shape2 = delta + 1`. This function is suitable for maximum likelihood
estimation.

## Usage

``` r
llbeta(par, data)
```

## Arguments

- par:

  A numeric vector of length 2 containing the distribution parameters in
  the order: `gamma` (\\\gamma \> 0\\), `delta` (\\\delta \ge 0\\).

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

This function calculates the negative log-likelihood for a Beta
distribution with parameters `shape1 = gamma` (\\\gamma\\) and
`shape2 = delta + 1` (\\\delta+1\\). The probability density function
(PDF) is: \$\$ f(x \| \gamma, \delta) = \frac{x^{\gamma-1}
(1-x)^{\delta}}{B(\gamma, \delta+1)} \$\$ for \\0 \< x \< 1\\, where
\\B(a,b)\\ is the Beta function
([`beta`](https://rdrr.io/r/base/Special.html)). The log-likelihood
function \\\ell(\theta \| \mathbf{x})\\ for a sample \\\mathbf{x} =
(x_1, \dots, x_n)\\ is \\\sum\_{i=1}^n \ln f(x_i \| \theta)\\: \$\$
\ell(\theta \| \mathbf{x}) = \sum\_{i=1}^{n} \[(\gamma-1)\ln(x_i) +
\delta\ln(1-x_i)\] - n \ln B(\gamma, \delta+1) \$\$ where \\\theta =
(\gamma, \delta)\\. This function computes and returns the *negative*
log-likelihood, \\-\ell(\theta\|\mathbf{x})\\, suitable for minimization
using optimization routines like
[`optim`](https://rdrr.io/r/stats/optim.html). It is equivalent to the
negative log-likelihood of the GKw distribution
([`llgkw`](https://evandeilton.github.io/gkwdist/reference/llgkw.md))
evaluated at \\\alpha=1, \beta=1, \lambda=1\\, and also to the negative
log-likelihood of the McDonald distribution
([`llmc`](https://evandeilton.github.io/gkwdist/reference/llmc.md))
evaluated at \\\lambda=1\\. The term \\\ln B(\gamma, \delta+1)\\ is
typically computed using log-gamma functions
([`lgamma`](https://rdrr.io/r/base/Special.html)) for numerical
stability.

## References

Johnson, N. L., Kotz, S., & Balakrishnan, N. (1995). *Continuous
Univariate Distributions, Volume 2* (2nd ed.). Wiley.

Cordeiro, G. M., & de Castro, M. (2011). A new family of generalized
distributions. *Journal of Statistical Computation and Simulation*,
*81*(7), 883-898.
[doi:10.1080/00949650903530745](https://doi.org/10.1080/00949650903530745)

## See also

[`llgkw`](https://evandeilton.github.io/gkwdist/reference/llgkw.md),
[`llmc`](https://evandeilton.github.io/gkwdist/reference/llmc.md)
(related negative log-likelihoods),
[`dbeta_`](https://evandeilton.github.io/gkwdist/reference/dbeta_.md),
[`pbeta_`](https://evandeilton.github.io/gkwdist/reference/pbeta_.md),
[`qbeta_`](https://evandeilton.github.io/gkwdist/reference/qbeta_.md),
[`rbeta_`](https://evandeilton.github.io/gkwdist/reference/rbeta_.md),
[`grbeta`](https://evandeilton.github.io/gkwdist/reference/grbeta.md)
(gradient),
[`hsbeta`](https://evandeilton.github.io/gkwdist/reference/hsbeta.md)
(Hessian), [`optim`](https://rdrr.io/r/stats/optim.html),
[`lbeta`](https://rdrr.io/r/base/Special.html).

Other log-likelihood functions:
[`llbkw()`](https://evandeilton.github.io/gkwdist/reference/llbkw.md),
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
x <- rbeta_(1000, gamma = 2, delta = 3)
par <- c(gamma = 2, delta = 3)

## llbeta() is the negative log-likelihood, -sum(log f(x))
llbeta(par, x)
#> [1] -359.6415
-sum(dbeta_(x, gamma = 2, delta = 3, log = TRUE))
#> [1] -359.6415

## Maximum likelihood: minimize llbeta(), with grbeta() as its gradient
start <- gkwgetstartvalues(x, family = "beta")
fit <- optim(start, llbeta, grbeta, data = x, method = "L-BFGS-B", lower = 1e-4)
fit$convergence  # 0: converged
#> [1] 0
fit$par
#>    gamma    delta 
#> 2.028741 2.997411 
fit$value <= llbeta(par, x)  # at least as good as the true values
#> [1] TRUE
```
