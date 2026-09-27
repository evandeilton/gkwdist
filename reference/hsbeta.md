# Hessian Matrix of the Negative Log-Likelihood for the Beta Distribution (gamma, delta+1 Parameterization)

Computes the analytic 2x2 Hessian matrix (matrix of second partial
derivatives) of the negative log-likelihood function for the standard
Beta distribution, using a parameterization common in generalized
distribution families. The distribution is parameterized by `gamma`
(\\\gamma\\) and `delta` (\\\delta\\), corresponding to the standard
Beta distribution with shape parameters `shape1 = gamma` and
`shape2 = delta + 1`. The Hessian is useful for estimating standard
errors and in optimization algorithms.

## Usage

``` r
hsbeta(par, data)
```

## Arguments

- par:

  A numeric vector of length 2 containing the distribution parameters in
  the order: `gamma` (\\\gamma \> 0\\), `delta` (\\\delta \ge 0\\).

- data:

  A numeric vector of observations. All values must be strictly between
  0 and 1 (exclusive).

## Value

Returns a 2x2 numeric matrix representing the Hessian matrix of the
negative log-likelihood function, \\-\partial^2 \ell / (\partial
\theta_i \partial \theta_j)\\, where \\\theta = (\gamma, \delta)\\.
Returns a 2x2 matrix populated with `NaN` if any parameter values are
invalid according to their constraints, or if any value in `data` is not
in the interval (0, 1).

## Details

This function calculates the analytic second partial derivatives of the
negative log-likelihood function (\\-\ell(\theta\|\mathbf{x})\\) for a
Beta distribution with parameters `shape1 = gamma` (\\\gamma\\) and
`shape2 = delta + 1` (\\\delta+1\\). The components of the Hessian
matrix (\\-\mathbf{H}(\theta)\\) are:

\$\$ -\frac{\partial^2 \ell}{\partial \gamma^2} = n\[\psi'(\gamma) -
\psi'(\gamma+\delta+1)\] \$\$ \$\$ -\frac{\partial^2 \ell}{\partial
\gamma \partial \delta} = -n\psi'(\gamma+\delta+1) \$\$ \$\$
-\frac{\partial^2 \ell}{\partial \delta^2} = n\[\psi'(\delta+1) -
\psi'(\gamma+\delta+1)\] \$\$

where \\\psi'(\cdot)\\ is the trigamma function
([`trigamma`](https://rdrr.io/r/base/Special.html)). These formulas
represent the second derivatives of \\-\ell(\theta)\\, consistent with
minimizing the negative log-likelihood. They correspond to the relevant
2x2 submatrix of the general GKw Hessian
([`hsgkw`](https://evandeilton.github.io/gkwdist/reference/hsgkw.md))
evaluated at \\\alpha=1, \beta=1, \lambda=1\\. Note the parameterization
difference from the standard Beta distribution (`shape2 = delta + 1`).

The returned matrix is symmetric.

## References

Johnson, N. L., Kotz, S., & Balakrishnan, N. (1995). *Continuous
Univariate Distributions, Volume 2* (2nd ed.). Wiley.

Cordeiro, G. M., & de Castro, M. (2011). A new family of generalized
distributions. *Journal of Statistical Computation and Simulation*,
*81*(7), 883-898.
[doi:10.1080/00949650903530745](https://doi.org/10.1080/00949650903530745)

(Note: Specific Hessian formulas might be derived or sourced from
additional references).

## See also

[`hsgkw`](https://evandeilton.github.io/gkwdist/reference/hsgkw.md),
[`hsmc`](https://evandeilton.github.io/gkwdist/reference/hsmc.md)
(related Hessians),
[`llbeta`](https://evandeilton.github.io/gkwdist/reference/llbeta.md)
(negative log-likelihood function),
[`grbeta`](https://evandeilton.github.io/gkwdist/reference/grbeta.md)
(gradient),
[`dbeta_`](https://evandeilton.github.io/gkwdist/reference/dbeta_.md),
[`pbeta_`](https://evandeilton.github.io/gkwdist/reference/pbeta_.md),
[`qbeta_`](https://evandeilton.github.io/gkwdist/reference/qbeta_.md),
[`rbeta_`](https://evandeilton.github.io/gkwdist/reference/rbeta_.md),
[`optim`](https://rdrr.io/r/stats/optim.html),
[`hessian`](https://rdrr.io/pkg/numDeriv/man/hessian.html) (for
numerical Hessian comparison),
[`trigamma`](https://rdrr.io/r/base/Special.html).

Other Hessian functions:
[`hsbkw()`](https://evandeilton.github.io/gkwdist/reference/hsbkw.md),
[`hsekw()`](https://evandeilton.github.io/gkwdist/reference/hsekw.md),
[`hsgkw()`](https://evandeilton.github.io/gkwdist/reference/hsgkw.md),
[`hskkw()`](https://evandeilton.github.io/gkwdist/reference/hskkw.md),
[`hskw()`](https://evandeilton.github.io/gkwdist/reference/hskw.md),
[`hsmc()`](https://evandeilton.github.io/gkwdist/reference/hsmc.md)

## Author

Lopes, J. E.

## Examples

``` r
set.seed(123)
x <- rbeta_(1000, gamma = 2, delta = 3)
par <- c(gamma = 2, delta = 3)

## Hessian of the negative log-likelihood llbeta()
H <- hsbeta(par, x)
isSymmetric(H)
#> [1] TRUE

## Agrees with a numerical Hessian of llbeta()
if (requireNamespace("numDeriv", quietly = TRUE))
  all.equal(H, numDeriv::hessian(llbeta, par, data = x), tolerance = 1e-8)
#> [1] TRUE

## At the MLE it is the observed information; its inverse estimates the
## covariance of the estimates
fit <- optim(par, llbeta, grbeta, data = x, method = "L-BFGS-B", lower = 1e-4)
sqrt(diag(solve(hsbeta(fit$par, x))))  # standard errors
#> [1] 0.08495446 0.17768971
```
