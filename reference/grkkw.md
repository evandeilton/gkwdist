# Gradient of the Negative Log-Likelihood for the KKw Distribution

Computes the gradient vector (vector of first partial derivatives) of
the negative log-likelihood function for the Kumaraswamy-Kumaraswamy
(KKw) distribution with parameters `alpha` (\\\alpha\\), `beta`
(\\\beta\\), `delta` (\\\delta\\), and `lambda` (\\\lambda\\). This
distribution is the special case of the Generalized Kumaraswamy (GKw)
distribution where \\\gamma = 1\\. The gradient is typically used in
optimization algorithms for maximum likelihood estimation.

## Usage

``` r
grkkw(par, data)
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

Returns a numeric vector of length 4 containing the partial derivatives
of the negative log-likelihood function \\-\ell(\theta \| \mathbf{x})\\
with respect to each parameter: \\(-\partial \ell/\partial \alpha,
-\partial \ell/\partial \beta, -\partial \ell/\partial \delta, -\partial
\ell/\partial \lambda)\\. Returns a vector of `NaN` if any parameter
values are invalid according to their constraints, or if any value in
`data` is not in the interval (0, 1).

## Details

The components of the gradient vector of the negative log-likelihood
(\\-\nabla \ell(\theta \| \mathbf{x})\\) for the KKw (\\\gamma=1\\)
model are:

\$\$ -\frac{\partial \ell}{\partial \alpha} = -\frac{n}{\alpha} -
\sum\_{i=1}^{n}\ln(x_i) +
(\beta-1)\sum\_{i=1}^{n}\frac{x_i^{\alpha}\ln(x_i)}{v_i} -
(\lambda-1)\sum\_{i=1}^{n}\frac{\beta v_i^{\beta-1}
x_i^{\alpha}\ln(x_i)}{w_i} + \delta\sum\_{i=1}^{n}\frac{\lambda
w_i^{\lambda-1} \beta v_i^{\beta-1} x_i^{\alpha}\ln(x_i)}{z_i} \$\$ \$\$
-\frac{\partial \ell}{\partial \beta} = -\frac{n}{\beta} -
\sum\_{i=1}^{n}\ln(v_i) +
(\lambda-1)\sum\_{i=1}^{n}\frac{v_i^{\beta}\ln(v_i)}{w_i} -
\delta\sum\_{i=1}^{n}\frac{\lambda w_i^{\lambda-1}
v_i^{\beta}\ln(v_i)}{z_i} \$\$ \$\$ -\frac{\partial \ell}{\partial
\delta} = -\frac{n}{\delta+1} - \sum\_{i=1}^{n}\ln(z_i) \$\$ \$\$
-\frac{\partial \ell}{\partial \lambda} = -\frac{n}{\lambda} -
\sum\_{i=1}^{n}\ln(w_i) +
\delta\sum\_{i=1}^{n}\frac{w_i^{\lambda}\ln(w_i)}{z_i} \$\$

where:

- \\v_i = 1 - x_i^{\alpha}\\

- \\w_i = 1 - v_i^{\beta} = 1 - (1-x_i^{\alpha})^{\beta}\\

- \\z_i = 1 - w_i^{\lambda} = 1 -
  \[1-(1-x_i^{\alpha})^{\beta}\]^{\lambda}\\

These formulas represent the derivatives of \\-\ell(\theta)\\,
consistent with minimizing the negative log-likelihood. They correspond
to the general GKw gradient
([`grgkw`](https://evandeilton.github.io/gkwdist/reference/grgkw.md))
components for \\\alpha, \beta, \delta, \lambda\\ evaluated at
\\\gamma=1\\. Note that the component for \\\gamma\\ is omitted.
Numerical stability is maintained through careful implementation.

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

[`grgkw`](https://evandeilton.github.io/gkwdist/reference/grgkw.md)
(parent distribution gradient),
[`llkkw`](https://evandeilton.github.io/gkwdist/reference/llkkw.md)
(negative log-likelihood for KKw),
[`hskkw`](https://evandeilton.github.io/gkwdist/reference/hskkw.md)
(Hessian for KKw),
[`dkkw`](https://evandeilton.github.io/gkwdist/reference/dkkw.md)
(density for KKw), [`optim`](https://rdrr.io/r/stats/optim.html),
[`grad`](https://rdrr.io/pkg/numDeriv/man/grad.html) (for numerical
gradient comparison).

Other gradient functions:
[`grbeta()`](https://evandeilton.github.io/gkwdist/reference/grbeta.md),
[`grbkw()`](https://evandeilton.github.io/gkwdist/reference/grbkw.md),
[`grekw()`](https://evandeilton.github.io/gkwdist/reference/grekw.md),
[`grgkw()`](https://evandeilton.github.io/gkwdist/reference/grgkw.md),
[`grkw()`](https://evandeilton.github.io/gkwdist/reference/grkw.md),
[`grmc()`](https://evandeilton.github.io/gkwdist/reference/grmc.md)

## Author

Lopes, J. E.

## Examples

``` r
set.seed(123)
x <- rkkw(200, alpha = 2, beta = 3, delta = 0.5, lambda = 1.2)
par <- c(alpha = 2, beta = 3, delta = 0.5, lambda = 1.2)

## Gradient of the negative log-likelihood llkkw(), not of the log-likelihood
g <- grkkw(par, x)
g
#> [1]  -9.2993777  -0.8090053  -1.9275849 -13.6860170

## A small step against the gradient lowers llkkw()
llkkw(par - 1e-4 * g, x) < llkkw(par, x)
#> [1] TRUE

## Agrees with a numerical derivative of llkkw()
if (requireNamespace("numDeriv", quietly = TRUE))
  all.equal(g, numDeriv::grad(llkkw, par, data = x), tolerance = 1e-6)
#> [1] TRUE
```
