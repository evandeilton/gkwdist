# Gradient of the Negative Log-Likelihood for the GKw Distribution

Computes the gradient vector (vector of partial derivatives) of the
negative log-likelihood function for the five-parameter Generalized
Kumaraswamy (GKw) distribution. This provides the analytical gradient,
often used for efficient optimization via maximum likelihood estimation.

## Usage

``` r
grgkw(par, data)
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

Returns a numeric vector of length 5 containing the partial derivatives
of the negative log-likelihood function \\-\ell(\theta \| \mathbf{x})\\
with respect to each parameter: \\(-\partial \ell/\partial \alpha,
-\partial \ell/\partial \beta, -\partial \ell/\partial \gamma, -\partial
\ell/\partial \delta, -\partial \ell/\partial \lambda)\\. Returns a
vector of `NaN` if any parameter values are invalid according to their
constraints, or if any value in `data` is not in the interval (0, 1).

## Details

The components of the gradient vector of the negative log-likelihood
(\\-\nabla \ell(\theta \| \mathbf{x})\\) are:

\$\$ -\frac{\partial \ell}{\partial \alpha} = -\frac{n}{\alpha} -
\sum\_{i=1}^{n}\ln(x_i) + \sum\_{i=1}^{n}\left\[x_i^{\alpha} \ln(x_i)
\left(\frac{\beta-1}{v_i} - \frac{(\gamma\lambda-1) \beta
v_i^{\beta-1}}{w_i} + \frac{\delta \lambda \beta v_i^{\beta-1}
w_i^{\lambda-1}}{z_i}\right)\right\] \$\$ \$\$ -\frac{\partial
\ell}{\partial \beta} = -\frac{n}{\beta} - \sum\_{i=1}^{n}\ln(v_i) +
\sum\_{i=1}^{n}\left\[v_i^{\beta} \ln(v_i)
\left(\frac{\gamma\lambda-1}{w_i} - \frac{\delta \lambda
w_i^{\lambda-1}}{z_i}\right)\right\] \$\$ \$\$ -\frac{\partial
\ell}{\partial \gamma} = n\[\psi(\gamma) - \psi(\gamma+\delta+1)\] -
\lambda\sum\_{i=1}^{n}\ln(w_i) \$\$ \$\$ -\frac{\partial \ell}{\partial
\delta} = n\[\psi(\delta+1) - \psi(\gamma+\delta+1)\] -
\sum\_{i=1}^{n}\ln(z_i) \$\$ \$\$ -\frac{\partial \ell}{\partial
\lambda} = -\frac{n}{\lambda} - \gamma\sum\_{i=1}^{n}\ln(w_i) +
\delta\sum\_{i=1}^{n}\frac{w_i^{\lambda}\ln(w_i)}{z_i} \$\$

where:

- \\v_i = 1 - x_i^{\alpha}\\

- \\w_i = 1 - v_i^{\beta} = 1 - (1-x_i^{\alpha})^{\beta}\\

- \\z_i = 1 - w_i^{\lambda} = 1 -
  \[1-(1-x_i^{\alpha})^{\beta}\]^{\lambda}\\

- \\\psi(\cdot)\\ is the digamma function
  ([`digamma`](https://rdrr.io/r/base/Special.html)).

Numerical stability is ensured through careful implementation, including
checks for valid inputs and handling of intermediate calculations
involving potentially small or large numbers, often leveraging the
Armadillo C++ library for efficiency.

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
(negative log-likelihood),
[`hsgkw`](https://evandeilton.github.io/gkwdist/reference/hsgkw.md)
(Hessian matrix),
[`dgkw`](https://evandeilton.github.io/gkwdist/reference/dgkw.md)
(density), [`optim`](https://rdrr.io/r/stats/optim.html),
[`grad`](https://rdrr.io/pkg/numDeriv/man/grad.html) (for numerical
gradient comparison), [`digamma`](https://rdrr.io/r/base/Special.html)

Other gradient functions:
[`grbeta()`](https://evandeilton.github.io/gkwdist/reference/grbeta.md),
[`grbkw()`](https://evandeilton.github.io/gkwdist/reference/grbkw.md),
[`grekw()`](https://evandeilton.github.io/gkwdist/reference/grekw.md),
[`grkkw()`](https://evandeilton.github.io/gkwdist/reference/grkkw.md),
[`grkw()`](https://evandeilton.github.io/gkwdist/reference/grkw.md),
[`grmc()`](https://evandeilton.github.io/gkwdist/reference/grmc.md)

## Author

Lopes, J. E.

## Examples

``` r
set.seed(123)
x <- rgkw(200, alpha = 2, beta = 3, gamma = 1.5, delta = 0.5, lambda = 1.2)
par <- c(alpha = 2, beta = 3, gamma = 1.5, delta = 0.5, lambda = 1.2)

## Gradient of the negative log-likelihood llgkw(), not of the log-likelihood
g <- grgkw(par, x)
g
#> [1] -1.6713047  0.4076473 -3.0711866  1.1604209 -3.8057074

## A small step against the gradient lowers llgkw()
llgkw(par - 1e-4 * g, x) < llgkw(par, x)
#> [1] TRUE

## Agrees with a numerical derivative of llgkw()
if (requireNamespace("numDeriv", quietly = TRUE))
  all.equal(g, numDeriv::grad(llgkw, par, data = x), tolerance = 1e-6)
#> [1] TRUE
```
