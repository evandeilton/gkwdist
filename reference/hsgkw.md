# Hessian Matrix of the Negative Log-Likelihood for the GKw Distribution

Computes the analytic Hessian matrix (matrix of second partial
derivatives) of the negative log-likelihood function for the
five-parameter Generalized Kumaraswamy (GKw) distribution. This is
typically used to estimate standard errors of maximum likelihood
estimates or in optimization algorithms.

## Usage

``` r
hsgkw(par, data)
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

Returns a 5x5 numeric matrix representing the Hessian matrix of the
negative log-likelihood function, i.e., the matrix of second partial
derivatives \\-\partial^2 \ell / (\partial \theta_i \partial
\theta_j)\\. Returns a 5x5 matrix populated with `NaN` if any parameter
values are invalid according to their constraints, or if any value in
`data` is not in the interval (0, 1).

## Details

This function calculates the analytic second partial derivatives of the
negative log-likelihood function based on the GKw PDF (see
[`dgkw`](https://evandeilton.github.io/gkwdist/reference/dgkw.md)). The
log-likelihood function \\\ell(\theta \| \mathbf{x})\\ is given by: \$\$
\ell(\theta) = n \ln(\lambda\alpha\beta) - n \ln B(\gamma, \delta+1) +
\sum\_{i=1}^{n} \[(\alpha-1) \ln(x_i) + (\beta-1) \ln(v_i) +
(\gamma\lambda - 1) \ln(w_i) + \delta \ln(z_i)\] \$\$ where \\\theta =
(\alpha, \beta, \gamma, \delta, \lambda)\\, \\B(a,b)\\ is the Beta
function ([`beta`](https://rdrr.io/r/base/Special.html)), and
intermediate terms are:

- \\v_i = 1 - x_i^{\alpha}\\

- \\w_i = 1 - v_i^{\beta} = 1 - (1-x_i^{\alpha})^{\beta}\\

- \\z_i = 1 - w_i^{\lambda} = 1 -
  \[1-(1-x_i^{\alpha})^{\beta}\]^{\lambda}\\

The Hessian matrix returned contains the elements \\- \frac{\partial^2
\ell(\theta \| \mathbf{x})}{\partial \theta_i \partial \theta_j}\\.

Key properties of the returned matrix:

- Dimensions: 5x5.

- Symmetry: The matrix is symmetric.

- Ordering: Rows and columns correspond to the parameters in the order
  \\\alpha, \beta, \gamma, \delta, \lambda\\.

- Content: Analytic second derivatives of the *negative* log-likelihood.

The exact analytical formulas for the second derivatives are implemented
directly (often derived using symbolic differentiation) for accuracy and
efficiency, typically using C++.

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

[`llgkw`](https://evandeilton.github.io/gkwdist/reference/llgkw.md)
(negative log-likelihood function),
[`grgkw`](https://evandeilton.github.io/gkwdist/reference/grgkw.md)
(gradient vector),
[`dgkw`](https://evandeilton.github.io/gkwdist/reference/dgkw.md)
(density function), [`optim`](https://rdrr.io/r/stats/optim.html),
[`hessian`](https://rdrr.io/pkg/numDeriv/man/hessian.html) (for
numerical Hessian comparison).

Other Hessian functions:
[`hsbeta()`](https://evandeilton.github.io/gkwdist/reference/hsbeta.md),
[`hsbkw()`](https://evandeilton.github.io/gkwdist/reference/hsbkw.md),
[`hsekw()`](https://evandeilton.github.io/gkwdist/reference/hsekw.md),
[`hskkw()`](https://evandeilton.github.io/gkwdist/reference/hskkw.md),
[`hskw()`](https://evandeilton.github.io/gkwdist/reference/hskw.md),
[`hsmc()`](https://evandeilton.github.io/gkwdist/reference/hsmc.md)

## Author

Lopes, J. E.

## Examples

``` r
set.seed(123)
x <- rgkw(1000, alpha = 2, beta = 3, gamma = 1.5, delta = 0.5, lambda = 1.2)
par <- c(alpha = 2, beta = 3, gamma = 1.5, delta = 0.5, lambda = 1.2)

## Hessian of the negative log-likelihood llgkw()
H <- hsgkw(par, x)
isSymmetric(H)
#> [1] TRUE

## Agrees with a numerical Hessian of llgkw()
if (requireNamespace("numDeriv", quietly = TRUE))
  all.equal(H, numDeriv::hessian(llgkw, par, data = x), tolerance = 1e-8)
#> [1] TRUE
```
