# Gradient of the Negative Log-Likelihood for the Beta Distribution (gamma, delta+1 Parameterization)

Computes the gradient vector (vector of first partial derivatives) of
the negative log-likelihood function for the standard Beta distribution,
using a parameterization common in generalized distribution families.
The distribution is parameterized by `gamma` (\\\gamma\\) and `delta`
(\\\delta\\), corresponding to the standard Beta distribution with shape
parameters `shape1 = gamma` and `shape2 = delta + 1`. The gradient is
useful for optimization algorithms.

## Usage

``` r
grbeta(par, data)
```

## Arguments

- par:

  A numeric vector of length 2 containing the distribution parameters in
  the order: `gamma` (\\\gamma \> 0\\), `delta` (\\\delta \ge 0\\).

- data:

  A numeric vector of observations. All values must be strictly between
  0 and 1 (exclusive).

## Value

Returns a numeric vector of length 2 containing the partial derivatives
of the negative log-likelihood function \\-\ell(\theta \| \mathbf{x})\\
with respect to each parameter: \\(-\partial \ell/\partial \gamma,
-\partial \ell/\partial \delta)\\. Returns a vector of `NaN` if any
parameter values are invalid according to their constraints, or if any
value in `data` is not in the interval (0, 1).

## Details

This function calculates the gradient of the negative log-likelihood for
a Beta distribution with parameters `shape1 = gamma` (\\\gamma\\) and
`shape2 = delta + 1` (\\\delta+1\\). The components of the gradient
vector (\\-\nabla \ell(\theta \| \mathbf{x})\\) are:

\$\$ -\frac{\partial \ell}{\partial \gamma} = n\[\psi(\gamma) -
\psi(\gamma+\delta+1)\] - \sum\_{i=1}^{n}\ln(x_i) \$\$ \$\$
-\frac{\partial \ell}{\partial \delta} = n\[\psi(\delta+1) -
\psi(\gamma+\delta+1)\] - \sum\_{i=1}^{n}\ln(1-x_i) \$\$

where \\\psi(\cdot)\\ is the digamma function
([`digamma`](https://rdrr.io/r/base/Special.html)). These formulas
represent the derivatives of \\-\ell(\theta)\\, consistent with
minimizing the negative log-likelihood. They correspond to the relevant
components of the general GKw gradient
([`grgkw`](https://evandeilton.github.io/gkwdist/reference/grgkw.md))
evaluated at \\\alpha=1, \beta=1, \lambda=1\\. Note the
parameterization: the standard Beta shape parameters are \\\gamma\\ and
\\\delta+1\\.

## References

Johnson, N. L., Kotz, S., & Balakrishnan, N. (1995). *Continuous
Univariate Distributions, Volume 2* (2nd ed.). Wiley.

Cordeiro, G. M., & de Castro, M. (2011). A new family of generalized
distributions. *Journal of Statistical Computation and Simulation*,
*81*(7), 883-898.
[doi:10.1080/00949650903530745](https://doi.org/10.1080/00949650903530745)

(Note: Specific gradient formulas might be derived or sourced from
additional references).

## See also

[`grgkw`](https://evandeilton.github.io/gkwdist/reference/grgkw.md),
[`grmc`](https://evandeilton.github.io/gkwdist/reference/grmc.md)
(related gradients),
[`llbeta`](https://evandeilton.github.io/gkwdist/reference/llbeta.md)
(negative log-likelihood function),
[`hsbeta`](https://evandeilton.github.io/gkwdist/reference/hsbeta.md)
(Hessian),
[`dbeta_`](https://evandeilton.github.io/gkwdist/reference/dbeta_.md),
[`pbeta_`](https://evandeilton.github.io/gkwdist/reference/pbeta_.md),
[`qbeta_`](https://evandeilton.github.io/gkwdist/reference/qbeta_.md),
[`rbeta_`](https://evandeilton.github.io/gkwdist/reference/rbeta_.md),
[`optim`](https://rdrr.io/r/stats/optim.html),
[`grad`](https://rdrr.io/pkg/numDeriv/man/grad.html) (for numerical
gradient comparison), [`digamma`](https://rdrr.io/r/base/Special.html).

Other gradient functions:
[`grbkw()`](https://evandeilton.github.io/gkwdist/reference/grbkw.md),
[`grekw()`](https://evandeilton.github.io/gkwdist/reference/grekw.md),
[`grgkw()`](https://evandeilton.github.io/gkwdist/reference/grgkw.md),
[`grkkw()`](https://evandeilton.github.io/gkwdist/reference/grkkw.md),
[`grkw()`](https://evandeilton.github.io/gkwdist/reference/grkw.md),
[`grmc()`](https://evandeilton.github.io/gkwdist/reference/grmc.md)

## Author

Lopes, J. E.

## Examples

``` r
set.seed(123)
x <- rbeta_(200, gamma = 2, delta = 3)
par <- c(gamma = 2, delta = 3)

## Gradient of the negative log-likelihood llbeta(), not of the log-likelihood
g <- grbeta(par, x)
g
#> [1] -2.704143 -2.096572

## A small step against the gradient lowers llbeta()
llbeta(par - 1e-4 * g, x) < llbeta(par, x)
#> [1] TRUE

## Agrees with a numerical derivative of llbeta()
if (requireNamespace("numDeriv", quietly = TRUE))
  all.equal(g, numDeriv::grad(llbeta, par, data = x), tolerance = 1e-6)
#> [1] TRUE
```
