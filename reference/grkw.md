# Gradient of the Negative Log-Likelihood for the Kumaraswamy (Kw) Distribution

Computes the gradient vector (vector of first partial derivatives) of
the negative log-likelihood function for the two-parameter Kumaraswamy
(Kw) distribution with parameters `alpha` (\\\alpha\\) and `beta`
(\\\beta\\). This provides the analytical gradient often used for
efficient optimization via maximum likelihood estimation.

## Usage

``` r
grkw(par, data)
```

## Arguments

- par:

  A numeric vector of length 2 containing the distribution parameters in
  the order: `alpha` (\\\alpha \> 0\\), `beta` (\\\beta \> 0\\).

- data:

  A numeric vector of observations. All values must be strictly between
  0 and 1 (exclusive).

## Value

Returns a numeric vector of length 2 containing the partial derivatives
of the negative log-likelihood function \\-\ell(\theta \| \mathbf{x})\\
with respect to each parameter: \\(-\partial \ell/\partial \alpha,
-\partial \ell/\partial \beta)\\. Returns a vector of `NaN` if any
parameter values are invalid according to their constraints, or if any
value in `data` is not in the interval (0, 1).

## Details

The components of the gradient vector of the negative log-likelihood
(\\-\nabla \ell(\theta \| \mathbf{x})\\) for the Kw model are:

\$\$ -\frac{\partial \ell}{\partial \alpha} = -\frac{n}{\alpha} -
\sum\_{i=1}^{n}\ln(x_i) +
(\beta-1)\sum\_{i=1}^{n}\frac{x_i^{\alpha}\ln(x_i)}{v_i} \$\$ \$\$
-\frac{\partial \ell}{\partial \beta} = -\frac{n}{\beta} -
\sum\_{i=1}^{n}\ln(v_i) \$\$

where \\v_i = 1 - x_i^{\alpha}\\. These formulas represent the
derivatives of \\-\ell(\theta)\\, consistent with minimizing the
negative log-likelihood. They correspond to the relevant components of
the general GKw gradient
([`grgkw`](https://evandeilton.github.io/gkwdist/reference/grgkw.md))
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

(Note: Specific gradient formulas might be derived or sourced from
additional references).

## See also

[`grgkw`](https://evandeilton.github.io/gkwdist/reference/grgkw.md)
(parent distribution gradient),
[`llkw`](https://evandeilton.github.io/gkwdist/reference/llkw.md)
(negative log-likelihood for Kw),
[`hskw`](https://evandeilton.github.io/gkwdist/reference/hskw.md)
(Hessian for Kw),
[`dkw`](https://evandeilton.github.io/gkwdist/reference/dkw.md) (density
for Kw), [`optim`](https://rdrr.io/r/stats/optim.html),
[`grad`](https://rdrr.io/pkg/numDeriv/man/grad.html) (for numerical
gradient comparison).

Other gradient functions:
[`grbeta()`](https://evandeilton.github.io/gkwdist/reference/grbeta.md),
[`grbkw()`](https://evandeilton.github.io/gkwdist/reference/grbkw.md),
[`grekw()`](https://evandeilton.github.io/gkwdist/reference/grekw.md),
[`grgkw()`](https://evandeilton.github.io/gkwdist/reference/grgkw.md),
[`grkkw()`](https://evandeilton.github.io/gkwdist/reference/grkkw.md),
[`grmc()`](https://evandeilton.github.io/gkwdist/reference/grmc.md)

## Author

Lopes, J. E.

## Examples

``` r
set.seed(123)
x <- rkw(200, alpha = 2, beta = 3)
par <- c(alpha = 2, beta = 3)

## Gradient of the negative log-likelihood llkw(), not of the log-likelihood
g <- grkw(par, x)
g
#> [1] -8.7375184 -0.9637925

## A small step against the gradient lowers llkw()
llkw(par - 1e-4 * g, x) < llkw(par, x)
#> [1] TRUE

## Agrees with a numerical derivative of llkw()
if (requireNamespace("numDeriv", quietly = TRUE))
  all.equal(g, numDeriv::grad(llkw, par, data = x), tolerance = 1e-6)
#> [1] TRUE
```
