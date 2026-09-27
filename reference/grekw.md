# Gradient of the Negative Log-Likelihood for the EKw Distribution

Computes the gradient vector (vector of first partial derivatives) of
the negative log-likelihood function for the Exponentiated Kumaraswamy
(EKw) distribution with parameters `alpha` (\\\alpha\\), `beta`
(\\\beta\\), and `lambda` (\\\lambda\\). This distribution is the
special case of the Generalized Kumaraswamy (GKw) distribution where
\\\gamma = 1\\ and \\\delta = 0\\. The gradient is useful for
optimization.

## Usage

``` r
grekw(par, data)
```

## Arguments

- par:

  A numeric vector of length 3 containing the distribution parameters in
  the order: `alpha` (\\\alpha \> 0\\), `beta` (\\\beta \> 0\\),
  `lambda` (\\\lambda \> 0\\).

- data:

  A numeric vector of observations. All values must be strictly between
  0 and 1 (exclusive).

## Value

Returns a numeric vector of length 3 containing the partial derivatives
of the negative log-likelihood function \\-\ell(\theta \| \mathbf{x})\\
with respect to each parameter: \\(-\partial \ell/\partial \alpha,
-\partial \ell/\partial \beta, -\partial \ell/\partial \lambda)\\.
Returns a vector of `NaN` if any parameter values are invalid according
to their constraints, or if any value in `data` is not in the interval
(0, 1).

## Details

The components of the gradient vector of the negative log-likelihood
(\\-\nabla \ell(\theta \| \mathbf{x})\\) for the EKw (\\\gamma=1,
\delta=0\\) model are:

\$\$ -\frac{\partial \ell}{\partial \alpha} = -\frac{n}{\alpha} -
\sum\_{i=1}^{n}\ln(x_i) + \sum\_{i=1}^{n}\left\[x_i^{\alpha} \ln(x_i)
\left(\frac{\beta-1}{v_i} - \frac{(\lambda-1) \beta
v_i^{\beta-1}}{w_i}\right)\right\] \$\$ \$\$ -\frac{\partial
\ell}{\partial \beta} = -\frac{n}{\beta} - \sum\_{i=1}^{n}\ln(v_i) +
\sum\_{i=1}^{n}\left\[\frac{(\lambda-1) v_i^{\beta}
\ln(v_i)}{w_i}\right\] \$\$ \$\$ -\frac{\partial \ell}{\partial \lambda}
= -\frac{n}{\lambda} - \sum\_{i=1}^{n}\ln(w_i) \$\$

where:

- \\v_i = 1 - x_i^{\alpha}\\

- \\w_i = 1 - v_i^{\beta} = 1 - (1-x_i^{\alpha})^{\beta}\\

These formulas represent the derivatives of \\-\ell(\theta)\\,
consistent with minimizing the negative log-likelihood. They correspond
to the relevant components of the general GKw gradient
([`grgkw`](https://evandeilton.github.io/gkwdist/reference/grgkw.md))
evaluated at \\\gamma=1, \delta=0\\.

## References

Nadarajah, S., Cordeiro, G. M., & Ortega, E. M. (2012). The
exponentiated Kumaraswamy distribution. *Journal of the Franklin
Institute*, *349*(3),

Cordeiro, G. M., & de Castro, M. (2011). A new family of generalized
distributions. *Journal of Statistical Computation and Simulation*,
*81*(7), 883-898.
[doi:10.1080/00949650903530745](https://doi.org/10.1080/00949650903530745)

Kumaraswamy, P. (1980). A generalized probability density function for
double-bounded random processes. *Journal of Hydrology*, *46*(1-2),
79-88.
[doi:10.1016/0022-1694(80)90036-0](https://doi.org/10.1016/0022-1694%2880%2990036-0)

(Note: Specific gradient formulas might be derived or sourced from
additional references).

## See also

[`grgkw`](https://evandeilton.github.io/gkwdist/reference/grgkw.md)
(parent distribution gradient),
[`llekw`](https://evandeilton.github.io/gkwdist/reference/llekw.md)
(negative log-likelihood for EKw),
[`hsekw`](https://evandeilton.github.io/gkwdist/reference/hsekw.md)
(Hessian for EKw),
[`dekw`](https://evandeilton.github.io/gkwdist/reference/dekw.md)
(density for EKw), [`optim`](https://rdrr.io/r/stats/optim.html),
[`grad`](https://rdrr.io/pkg/numDeriv/man/grad.html) (for numerical
gradient comparison).

Other gradient functions:
[`grbeta()`](https://evandeilton.github.io/gkwdist/reference/grbeta.md),
[`grbkw()`](https://evandeilton.github.io/gkwdist/reference/grbkw.md),
[`grgkw()`](https://evandeilton.github.io/gkwdist/reference/grgkw.md),
[`grkkw()`](https://evandeilton.github.io/gkwdist/reference/grkkw.md),
[`grkw()`](https://evandeilton.github.io/gkwdist/reference/grkw.md),
[`grmc()`](https://evandeilton.github.io/gkwdist/reference/grmc.md)

## Author

Lopes, J. E.

## Examples

``` r
set.seed(123)
x <- rekw(200, alpha = 2, beta = 3, lambda = 1.2)
par <- c(alpha = 2, beta = 3, lambda = 1.2)

## Gradient of the negative log-likelihood llekw(), not of the log-likelihood
g <- grekw(par, x)
g
#> [1]  -9.1941157  -0.7666647 -12.8434717

## A small step against the gradient lowers llekw()
llekw(par - 1e-4 * g, x) < llekw(par, x)
#> [1] TRUE

## Agrees with a numerical derivative of llekw()
if (requireNamespace("numDeriv", quietly = TRUE))
  all.equal(g, numDeriv::grad(llekw, par, data = x), tolerance = 1e-6)
#> [1] TRUE
```
