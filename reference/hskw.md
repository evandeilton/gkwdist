# Hessian Matrix of the Negative Log-Likelihood for the Kw Distribution

Computes the analytic 2x2 Hessian matrix (matrix of second partial
derivatives) of the negative log-likelihood function for the
two-parameter Kumaraswamy (Kw) distribution with parameters `alpha`
(\\\alpha\\) and `beta` (\\\beta\\). The Hessian is useful for
estimating standard errors and in optimization algorithms.

## Usage

``` r
hskw(par, data)
```

## Arguments

- par:

  A numeric vector of length 2 containing the distribution parameters in
  the order: `alpha` (\\\alpha \> 0\\), `beta` (\\\beta \> 0\\).

- data:

  A numeric vector of observations. All values must be strictly between
  0 and 1 (exclusive).

## Value

Returns a 2x2 numeric matrix representing the Hessian matrix of the
negative log-likelihood function, \\-\partial^2 \ell / (\partial
\theta_i \partial \theta_j)\\, where \\\theta = (\alpha, \beta)\\.
Returns a 2x2 matrix populated with `NaN` if any parameter values are
invalid according to their constraints, or if any value in `data` is not
in the interval (0, 1).

## Details

This function calculates the analytic second partial derivatives of the
negative log-likelihood function (\\-\ell(\theta\|\mathbf{x})\\). The
components are the negative of the second derivatives of the
log-likelihood \\\ell\\ (derived from the PDF in
[`dkw`](https://evandeilton.github.io/gkwdist/reference/dkw.md)).

Let \\v_i = 1 - x_i^{\alpha}\\. The second derivatives of the positive
log-likelihood (\\\ell\\) are: \$\$ \frac{\partial^2 \ell}{\partial
\alpha^2} = -\frac{n}{\alpha^2} -
(\beta-1)\sum\_{i=1}^{n}\frac{x_i^{\alpha}(\ln(x_i))^2}{v_i^2} \$\$ \$\$
\frac{\partial^2 \ell}{\partial \alpha \partial \beta} = -
\sum\_{i=1}^{n}\frac{x_i^{\alpha}\ln(x_i)}{v_i} \$\$ \$\$
\frac{\partial^2 \ell}{\partial \beta^2} = -\frac{n}{\beta^2} \$\$ The
function returns the Hessian matrix containing the negative of these
values.

Key properties of the returned matrix:

- Dimensions: 2x2.

- Symmetry: The matrix is symmetric.

- Ordering: Rows and columns correspond to the parameters in the order
  \\\alpha, \beta\\.

- Content: Analytic second derivatives of the *negative* log-likelihood.

This corresponds to the relevant 2x2 submatrix of the 5x5 GKw Hessian
([`hsgkw`](https://evandeilton.github.io/gkwdist/reference/hsgkw.md))
evaluated at \\\gamma=1, \delta=0, \lambda=1\\.

## References

Kumaraswamy, P. (1980). A generalized probability density function for
double-bounded random processes. *Journal of Hydrology*, *46*(1-2),
79-88.
[doi:10.1016/0022-1694(80)90036-0](https://doi.org/10.1016/0022-1694%2880%2990036-0)

Jones, M. C. (2009). Kumaraswamy's distribution: A beta-type
distribution with some tractability advantages. *Statistical
Methodology*, *6*(1), 70-81.
[doi:10.1016/j.stamet.2008.04.001](https://doi.org/10.1016/j.stamet.2008.04.001)

(Note: Specific Hessian formulas might be derived or sourced from
additional references).

## See also

[`hsgkw`](https://evandeilton.github.io/gkwdist/reference/hsgkw.md)
(parent distribution Hessian),
[`llkw`](https://evandeilton.github.io/gkwdist/reference/llkw.md)
(negative log-likelihood for Kw),
[`grkw`](https://evandeilton.github.io/gkwdist/reference/grkw.md)
(gradient for Kw),
[`dkw`](https://evandeilton.github.io/gkwdist/reference/dkw.md) (density
for Kw), [`optim`](https://rdrr.io/r/stats/optim.html),
[`hessian`](https://rdrr.io/pkg/numDeriv/man/hessian.html) (for
numerical Hessian comparison).

Other Hessian functions:
[`hsbeta()`](https://evandeilton.github.io/gkwdist/reference/hsbeta.md),
[`hsbkw()`](https://evandeilton.github.io/gkwdist/reference/hsbkw.md),
[`hsekw()`](https://evandeilton.github.io/gkwdist/reference/hsekw.md),
[`hsgkw()`](https://evandeilton.github.io/gkwdist/reference/hsgkw.md),
[`hskkw()`](https://evandeilton.github.io/gkwdist/reference/hskkw.md),
[`hsmc()`](https://evandeilton.github.io/gkwdist/reference/hsmc.md)

## Author

Lopes, J. E.

## Examples

``` r
set.seed(123)
x <- rkw(1000, alpha = 2, beta = 3)
par <- c(alpha = 2, beta = 3)

## Hessian of the negative log-likelihood llkw()
H <- hskw(par, x)
isSymmetric(H)
#> [1] TRUE

## Agrees with a numerical Hessian of llkw()
if (requireNamespace("numDeriv", quietly = TRUE))
  all.equal(H, numDeriv::hessian(llkw, par, data = x), tolerance = 1e-8)
#> [1] TRUE

## At the MLE it is the observed information; its inverse estimates the
## covariance of the estimates
fit <- optim(par, llkw, grkw, data = x, method = "L-BFGS-B", lower = 1e-4)
sqrt(diag(solve(hskw(fit$par, x))))  # standard errors
#> [1] 0.06543864 0.15878554
```
