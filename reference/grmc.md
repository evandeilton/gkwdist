# Gradient of the Negative Log-Likelihood for the McDonald (Mc)/Beta Power Distribution

Computes the gradient vector (vector of first partial derivatives) of
the negative log-likelihood function for the McDonald (Mc) distribution
(also known as Beta Power) with parameters `gamma` (\\\gamma\\), `delta`
(\\\delta\\), and `lambda` (\\\lambda\\). This distribution is the
special case of the Generalized Kumaraswamy (GKw) distribution where
\\\alpha = 1\\ and \\\beta = 1\\. The gradient is useful for
optimization.

## Usage

``` r
grmc(par, data)
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

Returns a numeric vector of length 3 containing the partial derivatives
of the negative log-likelihood function \\-\ell(\theta \| \mathbf{x})\\
with respect to each parameter: \\(-\partial \ell/\partial \gamma,
-\partial \ell/\partial \delta, -\partial \ell/\partial \lambda)\\.
Returns a vector of `NaN` if any parameter values are invalid according
to their constraints, or if any value in `data` is not in the interval
(0, 1).

## Details

The components of the gradient vector of the negative log-likelihood
(\\-\nabla \ell(\theta \| \mathbf{x})\\) for the Mc (\\\alpha=1,
\beta=1\\) model are:

\$\$ -\frac{\partial \ell}{\partial \gamma} = n\[\psi(\gamma) -
\psi(\gamma+\delta+1)\] - \lambda\sum\_{i=1}^{n}\ln(x_i) \$\$ \$\$
-\frac{\partial \ell}{\partial \delta} = n\[\psi(\delta+1) -
\psi(\gamma+\delta+1)\] - \sum\_{i=1}^{n}\ln(1-x_i^{\lambda}) \$\$ \$\$
-\frac{\partial \ell}{\partial \lambda} = -\frac{n}{\lambda} -
\gamma\sum\_{i=1}^{n}\ln(x_i) +
\delta\sum\_{i=1}^{n}\frac{x_i^{\lambda}\ln(x_i)}{1-x_i^{\lambda}} \$\$

where \\\psi(\cdot)\\ is the digamma function
([`digamma`](https://rdrr.io/r/base/Special.html)). These formulas
represent the derivatives of \\-\ell(\theta)\\, consistent with
minimizing the negative log-likelihood. They correspond to the relevant
components of the general GKw gradient
([`grgkw`](https://evandeilton.github.io/gkwdist/reference/grgkw.md))
evaluated at \\\alpha=1, \beta=1\\.

## References

McDonald, J. B. (1984). Some generalized functions for the size
distribution of income. *Econometrica*, *52*(3), 647-663.
[doi:10.2307/1913469](https://doi.org/10.2307/1913469)

Cordeiro, G. M., & de Castro, M. (2011). A new family of generalized
distributions. *Journal of Statistical Computation and Simulation*,
*81*(7), 883-898.
[doi:10.1080/00949650903530745](https://doi.org/10.1080/00949650903530745)

(Note: Specific gradient formulas might be derived or sourced from
additional references).

## See also

[`grgkw`](https://evandeilton.github.io/gkwdist/reference/grgkw.md)
(parent distribution gradient),
[`llmc`](https://evandeilton.github.io/gkwdist/reference/llmc.md)
(negative log-likelihood for Mc),
[`hsmc`](https://evandeilton.github.io/gkwdist/reference/hsmc.md)
(Hessian for Mc),
[`dmc`](https://evandeilton.github.io/gkwdist/reference/dmc.md) (density
for Mc), [`optim`](https://rdrr.io/r/stats/optim.html),
[`grad`](https://rdrr.io/pkg/numDeriv/man/grad.html) (for numerical
gradient comparison), [`digamma`](https://rdrr.io/r/base/Special.html).

Other gradient functions:
[`grbeta()`](https://evandeilton.github.io/gkwdist/reference/grbeta.md),
[`grbkw()`](https://evandeilton.github.io/gkwdist/reference/grbkw.md),
[`grekw()`](https://evandeilton.github.io/gkwdist/reference/grekw.md),
[`grgkw()`](https://evandeilton.github.io/gkwdist/reference/grgkw.md),
[`grkkw()`](https://evandeilton.github.io/gkwdist/reference/grkkw.md),
[`grkw()`](https://evandeilton.github.io/gkwdist/reference/grkw.md)

## Author

Lopes, J. E.

## Examples

``` r
set.seed(123)
x <- rmc(200, gamma = 0.5, delta = 5, lambda = 3)
par <- c(gamma = 0.5, delta = 5, lambda = 3)

## Gradient of the negative log-likelihood llmc(), not of the log-likelihood
g <- grmc(par, x)
g
#> [1]  4.494737  2.750941 -4.223447

## A small step against the gradient lowers llmc()
llmc(par - 1e-4 * g, x) < llmc(par, x)
#> [1] TRUE

## Agrees with a numerical derivative of llmc()
if (requireNamespace("numDeriv", quietly = TRUE))
  all.equal(g, numDeriv::grad(llmc, par, data = x), tolerance = 1e-6)
#> [1] TRUE
```
