# Hessian Matrix of the Negative Log-Likelihood for the BKw Distribution

Computes the analytic 4x4 Hessian matrix (matrix of second partial
derivatives) of the negative log-likelihood function for the
Beta-Kumaraswamy (BKw) distribution with parameters `alpha`
(\\\alpha\\), `beta` (\\\beta\\), `gamma` (\\\gamma\\), and `delta`
(\\\delta\\). This distribution is the special case of the Generalized
Kumaraswamy (GKw) distribution where \\\lambda = 1\\. The Hessian is
useful for estimating standard errors and in optimization algorithms.

## Usage

``` r
hsbkw(par, data)
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

Returns a 4x4 numeric matrix representing the Hessian matrix of the
negative log-likelihood function, \\-\partial^2 \ell / (\partial
\theta_i \partial \theta_j)\\, where \\\theta = (\alpha, \beta, \gamma,
\delta)\\. Returns a 4x4 matrix populated with `NaN` if any parameter
values are invalid according to their constraints, or if any value in
`data` is not in the interval (0, 1).

## Details

This function calculates the analytic second partial derivatives of the
negative log-likelihood function based on the BKw log-likelihood
(\\\lambda=1\\ case of GKw, see
[`llbkw`](https://evandeilton.github.io/gkwdist/reference/llbkw.md)):
\$\$ \ell(\theta \| \mathbf{x}) = n\[\ln(\alpha) + \ln(\beta) - \ln
B(\gamma, \delta+1)\] + \sum\_{i=1}^{n} \[(\alpha-1)\ln(x_i) +
(\beta(\delta+1)-1)\ln(v_i) + (\gamma-1)\ln(w_i)\] \$\$ where \\\theta =
(\alpha, \beta, \gamma, \delta)\\, \\B(a,b)\\ is the Beta function
([`beta`](https://rdrr.io/r/base/Special.html)), and intermediate terms
are:

- \\v_i = 1 - x_i^{\alpha}\\

- \\w_i = 1 - v_i^{\beta} = 1 - (1-x_i^{\alpha})^{\beta}\\

The Hessian matrix returned contains the elements \\- \frac{\partial^2
\ell(\theta \| \mathbf{x})}{\partial \theta_i \partial \theta_j}\\ for
\\\theta_i, \theta_j \in \\\alpha, \beta, \gamma, \delta\\\\.

Key properties of the returned matrix:

- Dimensions: 4x4.

- Symmetry: The matrix is symmetric.

- Ordering: Rows and columns correspond to the parameters in the order
  \\\alpha, \beta, \gamma, \delta\\.

- Content: Analytic second derivatives of the *negative* log-likelihood.

This corresponds to the relevant 4x4 submatrix of the 5x5 GKw Hessian
([`hsgkw`](https://evandeilton.github.io/gkwdist/reference/hsgkw.md))
evaluated at \\\lambda=1\\. The exact analytical formulas are
implemented directly.

## References

Cordeiro, G. M., & de Castro, M. (2011). A new family of generalized
distributions. *Journal of Statistical Computation and Simulation*,
*81*(7), 883-898.
[doi:10.1080/00949650903530745](https://doi.org/10.1080/00949650903530745)

Kumaraswamy, P. (1980). A generalized probability density function for
double-bounded random processes. *Journal of Hydrology*, *46*(1-2),
79-88.
[doi:10.1016/0022-1694(80)90036-0](https://doi.org/10.1016/0022-1694%2880%2990036-0)

(Note: Specific Hessian formulas might be derived or sourced from
additional references).

## See also

[`hsgkw`](https://evandeilton.github.io/gkwdist/reference/hsgkw.md)
(parent distribution Hessian),
[`llbkw`](https://evandeilton.github.io/gkwdist/reference/llbkw.md)
(negative log-likelihood for BKw),
[`grbkw`](https://evandeilton.github.io/gkwdist/reference/grbkw.md)
(gradient for BKw),
[`dbkw`](https://evandeilton.github.io/gkwdist/reference/dbkw.md)
(density for BKw), [`optim`](https://rdrr.io/r/stats/optim.html),
[`hessian`](https://rdrr.io/pkg/numDeriv/man/hessian.html) (for
numerical Hessian comparison).

Other Hessian functions:
[`hsbeta()`](https://evandeilton.github.io/gkwdist/reference/hsbeta.md),
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
x <- rbkw(1000, alpha = 2, beta = 3, gamma = 1.5, delta = 0.5)
par <- c(alpha = 2, beta = 3, gamma = 1.5, delta = 0.5)

## Hessian of the negative log-likelihood llbkw()
H <- hsbkw(par, x)
isSymmetric(H)
#> [1] TRUE

## Agrees with a numerical Hessian of llbkw()
if (requireNamespace("numDeriv", quietly = TRUE))
  all.equal(H, numDeriv::hessian(llbkw, par, data = x), tolerance = 1e-8)
#> [1] TRUE
```
