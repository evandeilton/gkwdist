# Hessian Matrix of the Negative Log-Likelihood for the McDonald (Mc)/Beta Power Distribution

Computes the analytic 3x3 Hessian matrix (matrix of second partial
derivatives) of the negative log-likelihood function for the McDonald
(Mc) distribution (also known as Beta Power) with parameters `gamma`
(\\\gamma\\), `delta` (\\\delta\\), and `lambda` (\\\lambda\\). This
distribution is the special case of the Generalized Kumaraswamy (GKw)
distribution where \\\alpha = 1\\ and \\\beta = 1\\. The Hessian is
useful for estimating standard errors and in optimization algorithms.

## Usage

``` r
hsmc(par, data)
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

Returns a 3x3 numeric matrix representing the Hessian matrix of the
negative log-likelihood function, \\-\partial^2 \ell / (\partial
\theta_i \partial \theta_j)\\, where \\\theta = (\gamma, \delta,
\lambda)\\. Returns a 3x3 matrix populated with `NaN` if any parameter
values are invalid according to their constraints, or if any value in
`data` is not in the interval (0, 1).

## Details

This function calculates the analytic second partial derivatives of the
negative log-likelihood function (\\-\ell(\theta\|\mathbf{x})\\). The
components are based on the second derivatives of the log-likelihood
\\\ell\\ (derived from the PDF in
[`dmc`](https://evandeilton.github.io/gkwdist/reference/dmc.md)).

**Note:** The formulas below represent the second derivatives of the
positive log-likelihood (\\\ell\\). The function returns the
**negative** of these values.

\$\$ \frac{\partial^2 \ell}{\partial \gamma^2} = -n\[\psi'(\gamma) -
\psi'(\gamma+\delta+1)\] \$\$ \$\$ \frac{\partial^2 \ell}{\partial
\gamma \partial \delta} = n\psi'(\gamma+\delta+1) \$\$ \$\$
\frac{\partial^2 \ell}{\partial \gamma \partial \lambda} =
\sum\_{i=1}^{n}\ln(x_i) \$\$ \$\$ \frac{\partial^2 \ell}{\partial
\delta^2} = -n\[\psi'(\delta+1) - \psi'(\gamma+\delta+1)\] \$\$ \$\$
\frac{\partial^2 \ell}{\partial \delta \partial \lambda} =
-\sum\_{i=1}^{n}\frac{x_i^{\lambda}\ln(x_i)}{1-x_i^{\lambda}} \$\$ \$\$
\frac{\partial^2 \ell}{\partial \lambda^2} = -\frac{n}{\lambda^2} -
\delta\sum\_{i=1}^{n}\frac{x_i^{\lambda}\[\ln(x_i)\]^2}{(1-x_i^{\lambda})^2}
\$\$

where \\\psi'(\cdot)\\ is the trigamma function
([`trigamma`](https://rdrr.io/r/base/Special.html)). The \\\partial^2
\ell / \partial \lambda^2\\ term matches the C++ implementation
(`src/bpmc.cpp`) and is covered by the numerical Hessian checks in
`tests/testthat/test-derivatives-validation.R`.

The returned matrix is symmetric, with rows/columns corresponding to
\\\gamma, \delta, \lambda\\.

## References

McDonald, J. B. (1984). Some generalized functions for the size
distribution of income. *Econometrica*, *52*(3), 647-663.
[doi:10.2307/1913469](https://doi.org/10.2307/1913469)

Cordeiro, G. M., & de Castro, M. (2011). A new family of generalized
distributions. *Journal of Statistical Computation and Simulation*,
*81*(7), 883-898.
[doi:10.1080/00949650903530745](https://doi.org/10.1080/00949650903530745)

(Note: Specific Hessian formulas might be derived or sourced from
additional references).

## See also

[`hsgkw`](https://evandeilton.github.io/gkwdist/reference/hsgkw.md)
(parent distribution Hessian),
[`llmc`](https://evandeilton.github.io/gkwdist/reference/llmc.md)
(negative log-likelihood for Mc),
[`grmc`](https://evandeilton.github.io/gkwdist/reference/grmc.md)
(gradient for Mc),
[`dmc`](https://evandeilton.github.io/gkwdist/reference/dmc.md) (density
for Mc), [`optim`](https://rdrr.io/r/stats/optim.html),
[`hessian`](https://rdrr.io/pkg/numDeriv/man/hessian.html) (for
numerical Hessian comparison),
[`trigamma`](https://rdrr.io/r/base/Special.html).

Other Hessian functions:
[`hsbeta()`](https://evandeilton.github.io/gkwdist/reference/hsbeta.md),
[`hsbkw()`](https://evandeilton.github.io/gkwdist/reference/hsbkw.md),
[`hsekw()`](https://evandeilton.github.io/gkwdist/reference/hsekw.md),
[`hsgkw()`](https://evandeilton.github.io/gkwdist/reference/hsgkw.md),
[`hskkw()`](https://evandeilton.github.io/gkwdist/reference/hskkw.md),
[`hskw()`](https://evandeilton.github.io/gkwdist/reference/hskw.md)

## Author

Lopes, J. E.

## Examples

``` r
set.seed(123)
x <- rmc(1000, gamma = 0.5, delta = 5, lambda = 3)
par <- c(gamma = 0.5, delta = 5, lambda = 3)

## Hessian of the negative log-likelihood llmc()
H <- hsmc(par, x)
isSymmetric(H)
#> [1] TRUE

## Agrees with a numerical Hessian of llmc()
if (requireNamespace("numDeriv", quietly = TRUE))
  all.equal(H, numDeriv::hessian(llmc, par, data = x), tolerance = 1e-8)
#> [1] TRUE

## At the MLE it is the observed information; its inverse estimates the
## covariance of the estimates
fit <- optim(par, llmc, grmc, data = x, method = "L-BFGS-B", lower = 1e-4)
sqrt(diag(solve(hsmc(fit$par, x))))  # standard errors
#> [1] 0.09391694 0.91559406 0.47943159
```
