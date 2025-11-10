# Quantile Function of the McDonald (Mc)/Beta Power Distribution

Computes the quantile function (inverse CDF) for the McDonald (Mc)
distribution (also known as Beta Power) with parameters `gamma`
(\\\gamma\\), `delta` (\\\delta\\), and `lambda` (\\\lambda\\). It finds
the value `q` such that \\P(X \le q) = p\\. This distribution is a
special case of the Generalized Kumaraswamy (GKw) distribution where
\\\alpha = 1\\ and \\\beta = 1\\.

## Usage

``` r
qmc(p, gamma, delta, lambda, lower_tail = TRUE, log_p = FALSE)
```

## Arguments

- p:

  Vector of probabilities (values between 0 and 1).

- gamma:

  Shape parameter `gamma` \> 0. Can be a scalar or a vector. Default:
  1.0.

- delta:

  Shape parameter `delta` \>= 0. Can be a scalar or a vector. Default:
  0.0.

- lambda:

  Shape parameter `lambda` \> 0. Can be a scalar or a vector. Default:
  1.0.

- lower_tail:

  Logical; if `TRUE` (default), probabilities are \\p = P(X \le q)\\,
  otherwise, probabilities are \\p = P(X \> q)\\.

- log_p:

  Logical; if `TRUE`, probabilities `p` are given as \\\log(p)\\.
  Default: `FALSE`.

## Value

A vector of quantiles corresponding to the given probabilities `p`. The
length of the result is determined by the recycling rule applied to the
arguments (`p`, `gamma`, `delta`, `lambda`). Returns:

- `0` for `p = 0` (or `p = -Inf` if `log_p = TRUE`, when
  `lower_tail = TRUE`).

- `1` for `p = 1` (or `p = 0` if `log_p = TRUE`, when
  `lower_tail = TRUE`).

- `NaN` for `p < 0` or `p > 1` (or corresponding log scale).

- `NaN` for invalid parameters (e.g., `gamma <= 0`, `delta < 0`,
  `lambda <= 0`).

Boundary return values are adjusted accordingly for
`lower_tail = FALSE`.

## Details

The quantile function \\Q(p)\\ is the inverse of the CDF \\F(q)\\. The
CDF for the Mc (\\\alpha=1, \beta=1\\) distribution is \\F(q) =
I\_{q^\lambda}(\gamma, \delta+1)\\, where \\I_z(a,b)\\ is the
regularized incomplete beta function (see
[`pmc`](https://evandeilton.github.io/gkwdist/reference/pmc.md)).

To find the quantile \\q\\, we first invert the Beta function part: let
\\y = I^{-1}\_{p}(\gamma, \delta+1)\\, where \\I^{-1}\_p(a,b)\\ is the
inverse computed via [`qbeta`](https://rdrr.io/r/stats/Beta.html). We
then solve \\q^\lambda = y\\ for \\q\\, yielding the quantile function:
\$\$ Q(p) = \left\[ I^{-1}\_{p}(\gamma, \delta+1) \right\]^{1/\lambda}
\$\$ The function uses this formula, calculating \\I^{-1}\_{p}(\gamma,
\delta+1)\\ via `qbeta(p, gamma, delta + 1, ...)` while respecting the
`lower_tail` and `log_p` arguments. This is equivalent to the general
GKw quantile function
([`qgkw`](https://evandeilton.github.io/gkwdist/reference/qgkw.md))
evaluated with \\\alpha=1, \beta=1\\.

## References

McDonald, J. B. (1984). Some generalized functions for the size
distribution of income. *Econometrica*, 52(3), 647-663.

Cordeiro, G. M., & de Castro, M. (2011). A new family of generalized
distributions. *Journal of Statistical Computation and Simulation*,

Kumaraswamy, P. (1980). A generalized probability density function for
double-bounded random processes. *Journal of Hydrology*, *46*(1-2),
79-88.

## See also

[`qgkw`](https://evandeilton.github.io/gkwdist/reference/qgkw.md)
(parent distribution quantile function),
[`dmc`](https://evandeilton.github.io/gkwdist/reference/dmc.md),
[`pmc`](https://evandeilton.github.io/gkwdist/reference/pmc.md),
[`rmc`](https://evandeilton.github.io/gkwdist/reference/rmc.md) (other
Mc functions), [`qbeta`](https://rdrr.io/r/stats/Beta.html)

## Author

Lopes, J. E.

## Examples

``` r
# \donttest{
# Example values
p_vals <- c(0.1, 0.5, 0.9)
gamma_par <- 2.0
delta_par <- 1.5
lambda_par <- 1.0 # Equivalent to Beta(gamma, delta+1)

# Calculate quantiles using qmc
quantiles <- qmc(p_vals, gamma_par, delta_par, lambda_par)
print(quantiles)
#> [1] 0.1649288 0.4355544 0.7379563
# Compare with Beta quantiles
print(stats::qbeta(p_vals, shape1 = gamma_par, shape2 = delta_par + 1))
#> [1] 0.1649288 0.4355544 0.7379563

# Calculate quantiles for upper tail probabilities P(X > q) = p
quantiles_upper <- qmc(p_vals, gamma_par, delta_par, lambda_par,
                       lower_tail = FALSE)
print(quantiles_upper)
#> [1] 0.7379563 0.4355544 0.1649288
# Check: qmc(p, ..., lt=F) == qmc(1-p, ..., lt=T)
print(qmc(1 - p_vals, gamma_par, delta_par, lambda_par))
#> [1] 0.7379563 0.4355544 0.1649288

# Calculate quantiles from log probabilities
log_p_vals <- log(p_vals)
quantiles_logp <- qmc(log_p_vals, gamma_par, delta_par, lambda_par, log_p = TRUE)
print(quantiles_logp)
#> [1] 0.1649288 0.4355544 0.7379563
# Check: should match original quantiles
print(quantiles)
#> [1] 0.1649288 0.4355544 0.7379563

# Compare with qgkw setting alpha = 1, beta = 1
quantiles_gkw <- qgkw(p_vals, alpha = 1.0, beta = 1.0, gamma = gamma_par,
                      delta = delta_par, lambda = lambda_par)
print(paste("Max difference:", max(abs(quantiles - quantiles_gkw)))) # Should be near zero
#> [1] "Max difference: 5.55111512312578e-17"

# Verify inverse relationship with pmc
p_check <- 0.75
q_calc <- qmc(p_check, gamma_par, delta_par, lambda_par) # Use lambda != 1
p_recalc <- pmc(q_calc, gamma_par, delta_par, lambda_par)
print(paste("Original p:", p_check, " Recalculated p:", p_recalc))
#> [1] "Original p: 0.75  Recalculated p: 0.75"
# abs(p_check - p_recalc) < 1e-9 # Should be TRUE

# Boundary conditions
print(qmc(c(0, 1), gamma_par, delta_par, lambda_par)) # Should be 0, 1
#> [1] 0 1
print(qmc(c(-Inf, 0), gamma_par, delta_par, lambda_par, log_p = TRUE)) # Should be 0, 1
#> [1] 0 1

# }
```
