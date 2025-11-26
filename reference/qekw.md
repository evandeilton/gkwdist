# Quantile Function of the Exponentiated Kumaraswamy (EKw) Distribution

Computes the quantile function (inverse CDF) for the Exponentiated
Kumaraswamy (EKw) distribution with parameters `alpha` (\\\alpha\\),
`beta` (\\\beta\\), and `lambda` (\\\lambda\\). It finds the value `q`
such that \\P(X \le q) = p\\. This distribution is a special case of the
Generalized Kumaraswamy (GKw) distribution where \\\gamma = 1\\ and
\\\delta = 0\\.

## Usage

``` r
qekw(p, alpha = 1, beta = 1, lambda = 1, lower.tail = TRUE, log.p = FALSE)
```

## Arguments

- p:

  Vector of probabilities (values between 0 and 1).

- alpha:

  Shape parameter `alpha` \> 0. Can be a scalar or a vector. Default:
  1.0.

- beta:

  Shape parameter `beta` \> 0. Can be a scalar or a vector. Default:
  1.0.

- lambda:

  Shape parameter `lambda` \> 0 (exponent parameter). Can be a scalar or
  a vector. Default: 1.0.

- lower.tail:

  Logical; if `TRUE` (default), probabilities are \\p = P(X \le q)\\,
  otherwise, probabilities are \\p = P(X \> q)\\.

- log.p:

  Logical; if `TRUE`, probabilities `p` are given as \\\log(p)\\.
  Default: `FALSE`.

## Value

A vector of quantiles corresponding to the given probabilities `p`. The
length of the result is determined by the recycling rule applied to the
arguments (`p`, `alpha`, `beta`, `lambda`). Returns:

- `0` for `p = 0` (or `p = -Inf` if `log.p = TRUE`, when
  `lower.tail = TRUE`).

- `1` for `p = 1` (or `p = 0` if `log.p = TRUE`, when
  `lower.tail = TRUE`).

- `NaN` for `p < 0` or `p > 1` (or corresponding log scale).

- `NaN` for invalid parameters (e.g., `alpha <= 0`, `beta <= 0`,
  `lambda <= 0`).

Boundary return values are adjusted accordingly for
`lower.tail = FALSE`.

## Details

The quantile function \\Q(p)\\ is the inverse of the CDF \\F(q)\\. The
CDF for the EKw (\\\gamma=1, \delta=0\\) distribution is \\F(q) = \[1 -
(1 - q^\alpha)^\beta \]^\lambda\\ (see
[`pekw`](https://evandeilton.github.io/gkwdist/reference/pekw.md)).
Inverting this equation for \\q\\ yields the quantile function: \$\$
Q(p) = \left\\ 1 - \left\[ 1 - p^{1/\lambda} \right\]^{1/\beta}
\right\\^{1/\alpha} \$\$ The function uses this closed-form expression
and correctly handles the `lower.tail` and `log.p` arguments by
transforming `p` appropriately before applying the formula. This is
equivalent to the general GKw quantile function
([`qgkw`](https://evandeilton.github.io/gkwdist/reference/qgkw.md))
evaluated with \\\gamma=1, \delta=0\\.

## References

Nadarajah, S., Cordeiro, G. M., & Ortega, E. M. (2012). The
exponentiated Kumaraswamy distribution. *Journal of the Franklin
Institute*, *349*(3),

Cordeiro, G. M., & de Castro, M. (2011). A new family of generalized
distributions. *Journal of Statistical Computation and Simulation*,

Kumaraswamy, P. (1980). A generalized probability density function for
double-bounded random processes. *Journal of Hydrology*, *46*(1-2),
79-88.

## See also

[`qgkw`](https://evandeilton.github.io/gkwdist/reference/qgkw.md)
(parent distribution quantile function),
[`dekw`](https://evandeilton.github.io/gkwdist/reference/dekw.md),
[`pekw`](https://evandeilton.github.io/gkwdist/reference/pekw.md),
[`rekw`](https://evandeilton.github.io/gkwdist/reference/rekw.md) (other
EKw functions), [`qunif`](https://rdrr.io/r/stats/Uniform.html)

## Author

Lopes, J. E.

## Examples

``` r
# \donttest{
# Example values
p_vals <- c(0.1, 0.5, 0.9)
alpha_par <- 2.0
beta_par <- 3.0
lambda_par <- 1.5

# Calculate quantiles
quantiles <- qekw(p_vals, alpha_par, beta_par, lambda_par)
print(quantiles)
#> [1] 0.2787375 0.5311017 0.7695287

# Calculate quantiles for upper tail probabilities P(X > q) = p
quantiles_upper <- qekw(p_vals, alpha_par, beta_par, lambda_par,
                        lower.tail = FALSE)
print(quantiles_upper)
#> [1] 0.7695287 0.5311017 0.2787375
# Check: qekw(p, ..., lt=F) == qekw(1-p, ..., lt=T)
print(qekw(1 - p_vals, alpha_par, beta_par, lambda_par))
#> [1] 0.7695287 0.5311017 0.2787375

# Calculate quantiles from log probabilities
log.p_vals <- log(p_vals)
quantiles_logp <- qekw(log.p_vals, alpha_par, beta_par, lambda_par,
                       log.p = TRUE)
print(quantiles_logp)
#> [1] 0.2787375 0.5311017 0.7695287
# Check: should match original quantiles
print(quantiles)
#> [1] 0.2787375 0.5311017 0.7695287

# Compare with qgkw setting gamma = 1, delta = 0
quantiles_gkw <- qgkw(p_vals, alpha = alpha_par, beta = beta_par,
                     gamma = 1.0, delta = 0.0, lambda = lambda_par)
print(paste("Max difference:", max(abs(quantiles - quantiles_gkw)))) # Should be near zero
#> [1] "Max difference: 0"

# Verify inverse relationship with pekw
p_check <- 0.75
q_calc <- qekw(p_check, alpha_par, beta_par, lambda_par)
p_recalc <- pekw(q_calc, alpha_par, beta_par, lambda_par)
print(paste("Original p:", p_check, " Recalculated p:", p_recalc))
#> [1] "Original p: 0.75  Recalculated p: 0.75"
# abs(p_check - p_recalc) < 1e-9 # Should be TRUE

# Boundary conditions
print(qekw(c(0, 1), alpha_par, beta_par, lambda_par)) # Should be 0, 1
#> [1] 0 1
print(qekw(c(-Inf, 0), alpha_par, beta_par, lambda_par, log.p = TRUE)) # Should be 0, 1
#> [1] 0 1
# }
```
