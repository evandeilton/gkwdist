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
# \donttest{
## Example 1: Basic Gradient Evaluation

# Generate sample data with more stable parameters
set.seed(123)
n <- 1000
true_params <- c(gamma = 2.0, delta = 2.5, lambda = 1.5)
data <- rmc(n,
  gamma = true_params[1], delta = true_params[2],
  lambda = true_params[3]
)

# Evaluate the gradient at the true parameters
grad_true <- grmc(par = true_params, data = data)
names(grad_true) <- c("gamma", "delta", "lambda")
cat("Gradient at true parameters:\n")
#> Gradient at true parameters:
print(grad_true)
#>      gamma      delta     lambda 
#> -13.504792   6.525193 -24.769059 
cat("Gradient norm:", sqrt(sum(grad_true^2)), "\n")
#> Gradient norm: 28.95624 


## Example 2: Numerical Verification with numDeriv::grad()

# grmc() returns the gradient of the *negative* log-likelihood minimized
# by llmc(). numDeriv::grad() differentiates llmc() itself by finite
# differences, so the two should agree closely at any parameter vector,
# not just at the MLE.
if (requireNamespace("numDeriv", quietly = TRUE)) {
  test_points <- rbind(
    c(1.5, 2.0, 1.0),
    as.numeric(true_params),
    c(2.5, 3.0, 2.0)
  )

  cat("\nAnalytical (grmc) vs numerical (numDeriv::grad) gradient:\n")
  for (i in seq_len(nrow(test_points))) {
    par_i <- test_points[i, ]
    grad_analytic <- grmc(par = par_i, data = data)
    grad_numeric <- numDeriv::grad(func = llmc, x = par_i, data = data)

    comparison <- data.frame(
      Parameter = c("gamma", "delta", "lambda"),
      Analytical = grad_analytic,
      Numerical = grad_numeric,
      Abs_Diff = abs(grad_analytic - grad_numeric)
    )
    cat("\nPoint", i, ": (", paste(round(par_i, 2), collapse = ", "), ")\n")
    print(comparison, digits = 8, row.names = FALSE)
  }
}
#> 
#> Analytical (grmc) vs numerical (numDeriv::grad) gradient:
#> 
#> Point 1 : ( 1.5, 2, 1 )
#>  Parameter  Analytical   Numerical      Abs_Diff
#>      gamma  -569.17827  -569.17827 1.7215825e-08
#>      delta   298.19112   298.19112 3.8869075e-08
#>     lambda -1172.76622 -1172.76622 1.1249836e-07
#> 
#> Point 2 : ( 2, 2.5, 1.5 )
#>  Parameter  Analytical   Numerical      Abs_Diff
#>      gamma -13.5047924 -13.5047924 5.1955441e-08
#>      delta   6.5251933   6.5251933 2.8501302e-08
#>     lambda -24.7690588 -24.7690589 9.6862262e-08
#> 
#> Point 3 : ( 2.5, 3, 2 )
#>  Parameter Analytical  Numerical      Abs_Diff
#>      gamma  476.65067  476.65067 4.7074764e-08
#>      delta -168.29540 -168.29540 9.6466408e-09
#>     lambda  769.57550  769.57550 6.4076062e-08


## Example 3: Gradient-Based Optimization Convergence

# Supplying the analytical gradient lets BFGS skip its internal
# finite-difference approximation
fit_with_grad <- optim(
  par = c(1.5, 2.0, 1.0),
  fn = llmc,
  gr = grmc,
  data = data,
  method = "BFGS",
  control = list(trace = 0)
)

# Same starting point and objective, relying on optim()'s own
# finite-difference gradient instead
fit_no_grad <- optim(
  par = c(1.5, 2.0, 1.0),
  fn = llmc,
  data = data,
  method = "BFGS",
  control = list(trace = 0)
)

mle <- fit_with_grad$par
names(mle) <- c("gamma", "delta", "lambda")

comparison <- data.frame(
  Method = c("Analytical gradient", "Finite-difference"),
  Gamma = c(fit_with_grad$par[1], fit_no_grad$par[1]),
  Delta = c(fit_with_grad$par[2], fit_no_grad$par[2]),
  Lambda = c(fit_with_grad$par[3], fit_no_grad$par[3]),
  NegLogLik = c(fit_with_grad$value, fit_no_grad$value),
  Fn_Evals = c(fit_with_grad$counts[1], fit_no_grad$counts[1])
)
cat("\nOptimization comparison:\n")
#> 
#> Optimization comparison:
print(comparison, digits = 6, row.names = FALSE)
#>               Method   Gamma   Delta  Lambda NegLogLik Fn_Evals
#>  Analytical gradient 1.45821 2.64432 1.95583  -310.101       82
#>    Finite-difference 1.46074 2.64340 1.95329  -310.101       80

# At the MLE, the gradient of the negative log-likelihood should vanish
grad_at_mle <- grmc(par = mle, data = data)
cat("\nGradient at MLE:", grad_at_mle, "\n")
#> 
#> Gradient at MLE: -0.06057962 0.01657138 -0.05932882 
cat("Max absolute component:", max(abs(grad_at_mle)), "\n")
#> Max absolute component: 0.06057962 
# }
```
