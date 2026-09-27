# Hessian Matrix of the Negative Log-Likelihood for the KKw Distribution

Computes the analytic 4x4 Hessian matrix (matrix of second partial
derivatives) of the negative log-likelihood function for the
Kumaraswamy-Kumaraswamy (KKw) distribution with parameters `alpha`
(\\\alpha\\), `beta` (\\\beta\\), `delta` (\\\delta\\), and `lambda`
(\\\lambda\\). This distribution is the special case of the Generalized
Kumaraswamy (GKw) distribution where \\\gamma = 1\\. The Hessian is
useful for estimating standard errors and in optimization algorithms.

## Usage

``` r
hskkw(par, data)
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

Returns a 4x4 numeric matrix representing the Hessian matrix of the
negative log-likelihood function, \\-\partial^2 \ell / (\partial
\theta_i \partial \theta_j)\\, where \\\theta = (\alpha, \beta, \delta,
\lambda)\\. Returns a 4x4 matrix populated with `NaN` if any parameter
values are invalid according to their constraints, or if any value in
`data` is not in the interval (0, 1).

## Details

This function calculates the analytic second partial derivatives of the
negative log-likelihood function based on the KKw log-likelihood
(\\\gamma=1\\ case of GKw, see
[`llkkw`](https://evandeilton.github.io/gkwdist/reference/llkkw.md)):
\$\$ \ell(\theta \| \mathbf{x}) = n\[\ln(\delta+1) + \ln(\lambda) +
\ln(\alpha) + \ln(\beta)\] + \sum\_{i=1}^{n} \[(\alpha-1)\ln(x_i) +
(\beta-1)\ln(v_i) + (\lambda-1)\ln(w_i) + \delta\ln(z_i)\] \$\$ where
\\\theta = (\alpha, \beta, \delta, \lambda)\\ and intermediate terms
are:

- \\v_i = 1 - x_i^{\alpha}\\

- \\w_i = 1 - v_i^{\beta} = 1 - (1-x_i^{\alpha})^{\beta}\\

- \\z_i = 1 - w_i^{\lambda} = 1 -
  \[1-(1-x_i^{\alpha})^{\beta}\]^{\lambda}\\

The Hessian matrix returned contains the elements \\- \frac{\partial^2
\ell(\theta \| \mathbf{x})}{\partial \theta_i \partial \theta_j}\\ for
\\\theta_i, \theta_j \in \\\alpha, \beta, \delta, \lambda\\\\.

Key properties of the returned matrix:

- Dimensions: 4x4.

- Symmetry: The matrix is symmetric.

- Ordering: Rows and columns correspond to the parameters in the order
  \\\alpha, \beta, \delta, \lambda\\.

- Content: Analytic second derivatives of the *negative* log-likelihood.

This corresponds to the relevant submatrix of the 5x5 GKw Hessian
([`hsgkw`](https://evandeilton.github.io/gkwdist/reference/hsgkw.md))
evaluated at \\\gamma=1\\. The exact analytical formulas are implemented
directly.

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

[`hsgkw`](https://evandeilton.github.io/gkwdist/reference/hsgkw.md)
(parent distribution Hessian),
[`llkkw`](https://evandeilton.github.io/gkwdist/reference/llkkw.md)
(negative log-likelihood for KKw),
[`grkkw`](https://evandeilton.github.io/gkwdist/reference/grkkw.md)
(gradient for KKw),
[`dkkw`](https://evandeilton.github.io/gkwdist/reference/dkkw.md)
(density for KKw), [`optim`](https://rdrr.io/r/stats/optim.html),
[`hessian`](https://rdrr.io/pkg/numDeriv/man/hessian.html) (for
numerical Hessian comparison).

Other Hessian functions:
[`hsbeta()`](https://evandeilton.github.io/gkwdist/reference/hsbeta.md),
[`hsbkw()`](https://evandeilton.github.io/gkwdist/reference/hsbkw.md),
[`hsekw()`](https://evandeilton.github.io/gkwdist/reference/hsekw.md),
[`hsgkw()`](https://evandeilton.github.io/gkwdist/reference/hsgkw.md),
[`hskw()`](https://evandeilton.github.io/gkwdist/reference/hskw.md),
[`hsmc()`](https://evandeilton.github.io/gkwdist/reference/hsmc.md)

## Author

Lopes, J. E.

## Examples

``` r
set.seed(123)
x <- rkkw(1000, alpha = 2, beta = 3, delta = 0.5, lambda = 1.2)
par <- c(alpha = 2, beta = 3, delta = 0.5, lambda = 1.2)

## Hessian of the negative log-likelihood llkkw()
H <- hskkw(par, x)
isSymmetric(H)
#> [1] TRUE

## Agrees with a numerical Hessian of llkkw()
if (requireNamespace("numDeriv", quietly = TRUE))
  all.equal(H, numDeriv::hessian(llkkw, par, data = x), tolerance = 1e-8)
#> [1] TRUE
```
