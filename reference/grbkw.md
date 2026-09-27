# Gradient of the Negative Log-Likelihood for the BKw Distribution

Computes the gradient vector (vector of first partial derivatives) of
the negative log-likelihood function for the Beta-Kumaraswamy (BKw)
distribution with parameters `alpha` (\\\alpha\\), `beta` (\\\beta\\),
`gamma` (\\\gamma\\), and `delta` (\\\delta\\). This distribution is the
special case of the Generalized Kumaraswamy (GKw) distribution where
\\\lambda = 1\\. The gradient is typically used in optimization
algorithms for maximum likelihood estimation.

## Usage

``` r
grbkw(par, data)
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

Returns a numeric vector of length 4 containing the partial derivatives
of the negative log-likelihood function \\-\ell(\theta \| \mathbf{x})\\
with respect to each parameter: \\(-\partial \ell/\partial \alpha,
-\partial \ell/\partial \beta, -\partial \ell/\partial \gamma, -\partial
\ell/\partial \delta)\\. Returns a vector of `NaN` if any parameter
values are invalid according to their constraints, or if any value in
`data` is not in the interval (0, 1).

## Details

The components of the gradient vector of the negative log-likelihood
(\\-\nabla \ell(\theta \| \mathbf{x})\\) for the BKw (\\\lambda=1\\)
model are:

\$\$ -\frac{\partial \ell}{\partial \alpha} = -\frac{n}{\alpha} -
\sum\_{i=1}^{n}\ln(x_i) + \sum\_{i=1}^{n}\left\[x_i^{\alpha} \ln(x_i)
\left(\frac{\beta(\delta+1)-1}{v_i} - \frac{(\gamma-1) \beta
v_i^{\beta-1}}{w_i}\right)\right\] \$\$ \$\$ -\frac{\partial
\ell}{\partial \beta} = -\frac{n}{\beta} -
(\delta+1)\sum\_{i=1}^{n}\ln(v_i) +
\sum\_{i=1}^{n}\left\[\frac{(\gamma-1) v_i^{\beta}
\ln(v_i)}{w_i}\right\] \$\$ \$\$ -\frac{\partial \ell}{\partial \gamma}
= n\[\psi(\gamma) - \psi(\gamma+\delta+1)\] - \sum\_{i=1}^{n}\ln(w_i)
\$\$ \$\$ -\frac{\partial \ell}{\partial \delta} = n\[\psi(\delta+1) -
\psi(\gamma+\delta+1)\] - \beta\sum\_{i=1}^{n}\ln(v_i) \$\$

where:

- \\v_i = 1 - x_i^{\alpha}\\

- \\w_i = 1 - v_i^{\beta} = 1 - (1-x_i^{\alpha})^{\beta}\\

- \\\psi(\cdot)\\ is the digamma function
  ([`digamma`](https://rdrr.io/r/base/Special.html)).

These formulas represent the derivatives of \\-\ell(\theta)\\,
consistent with minimizing the negative log-likelihood. They correspond
to the general GKw gradient
([`grgkw`](https://evandeilton.github.io/gkwdist/reference/grgkw.md))
components for \\\alpha, \beta, \gamma, \delta\\ evaluated at
\\\lambda=1\\. Note that the component for \\\lambda\\ is omitted.
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

(Note: Specific gradient formulas might be derived or sourced from
additional references).

## See also

[`grgkw`](https://evandeilton.github.io/gkwdist/reference/grgkw.md)
(parent distribution gradient),
[`llbkw`](https://evandeilton.github.io/gkwdist/reference/llbkw.md)
(negative log-likelihood for BKw),
[`hsbkw`](https://evandeilton.github.io/gkwdist/reference/hsbkw.md)
(Hessian for BKw),
[`dbkw`](https://evandeilton.github.io/gkwdist/reference/dbkw.md)
(density for BKw), [`optim`](https://rdrr.io/r/stats/optim.html),
[`grad`](https://rdrr.io/pkg/numDeriv/man/grad.html) (for numerical
gradient comparison), [`digamma`](https://rdrr.io/r/base/Special.html).

Other gradient functions:
[`grbeta()`](https://evandeilton.github.io/gkwdist/reference/grbeta.md),
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
x <- rbkw(200, alpha = 2, beta = 3, gamma = 1.5, delta = 0.5)
par <- c(alpha = 2, beta = 3, gamma = 1.5, delta = 0.5)

## Gradient of the negative log-likelihood llbkw(), not of the log-likelihood
g <- grbkw(par, x)
g
#> [1] -1.6023655  0.4745672 -3.0711866  1.1604209

## A small step against the gradient lowers llbkw()
llbkw(par - 1e-4 * g, x) < llbkw(par, x)
#> [1] TRUE

## Agrees with a numerical derivative of llbkw()
if (requireNamespace("numDeriv", quietly = TRUE))
  all.equal(g, numDeriv::grad(llbkw, par, data = x), tolerance = 1e-6)
#> [1] TRUE
```
