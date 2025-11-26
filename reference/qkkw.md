# Quantile Function of the Kumaraswamy-Kumaraswamy (kkw) Distribution

Computes the quantile function (inverse CDF) for the
Kumaraswamy-Kumaraswamy (kkw) distribution with parameters `alpha`
(\\\alpha\\), `beta` (\\\beta\\), `delta` (\\\delta\\), and `lambda`
(\\\lambda\\). It finds the value `q` such that \\P(X \le q) = p\\. This
distribution is a special case of the Generalized Kumaraswamy (GKw)
distribution where the parameter \\\gamma = 1\\.

## Usage

``` r
qkkw(
  p,
  alpha = 1,
  beta = 1,
  delta = 0,
  lambda = 1,
  lower.tail = TRUE,
  log.p = FALSE
)
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

- delta:

  Shape parameter `delta` \>= 0. Can be a scalar or a vector. Default:
  0.0.

- lambda:

  Shape parameter `lambda` \> 0. Can be a scalar or a vector. Default:
  1.0.

- lower.tail:

  Logical; if `TRUE` (default), probabilities are \\p = P(X \le q)\\,
  otherwise, probabilities are \\p = P(X \> q)\\.

- log.p:

  Logical; if `TRUE`, probabilities `p` are given as \\\log(p)\\.
  Default: `FALSE`.

## Value

A vector of quantiles corresponding to the given probabilities `p`. The
length of the result is determined by the recycling rule applied to the
arguments (`p`, `alpha`, `beta`, `delta`, `lambda`). Returns:

- `0` for `p = 0` (or `p = -Inf` if `log.p = TRUE`, when
  `lower.tail = TRUE`).

- `1` for `p = 1` (or `p = 0` if `log.p = TRUE`, when
  `lower.tail = TRUE`).

- `NaN` for `p < 0` or `p > 1` (or corresponding log scale).

- `NaN` for invalid parameters (e.g., `alpha <= 0`, `beta <= 0`,
  `delta < 0`, `lambda <= 0`).

Boundary return values are adjusted accordingly for
`lower.tail = FALSE`.

## Details

The quantile function \\Q(p)\\ is the inverse of the CDF \\F(q)\\. The
CDF for the kkw (\\\gamma=1\\) distribution is (see
[`pkkw`](https://evandeilton.github.io/gkwdist/reference/pkkw.md)): \$\$
F(q) = 1 - \bigl\\1 - \bigl\[1 - (1 -
q^\alpha)^\beta\bigr\]^\lambda\bigr\\^{\delta + 1} \$\$ Inverting this
equation for \\q\\ yields the quantile function: \$\$ Q(p) = \left\[ 1 -
\left\\ 1 - \left\[ 1 - (1 - p)^{1/(\delta+1)} \right\]^{1/\lambda}
\right\\^{1/\beta} \right\]^{1/\alpha} \$\$ The function uses this
closed-form expression and correctly handles the `lower.tail` and
`log.p` arguments by transforming `p` appropriately before applying the
formula.

## References

Cordeiro, G. M., & de Castro, M. (2011). A new family of generalized
distributions. *Journal of Statistical Computation and Simulation*,

Kumaraswamy, P. (1980). A generalized probability density function for
double-bounded random processes. *Journal of Hydrology*, *46*(1-2),
79-88.

## See also

[`qgkw`](https://evandeilton.github.io/gkwdist/reference/qgkw.md)
(parent distribution quantile function),
[`dkkw`](https://evandeilton.github.io/gkwdist/reference/dkkw.md),
[`pkkw`](https://evandeilton.github.io/gkwdist/reference/pkkw.md),
[`rkkw`](https://evandeilton.github.io/gkwdist/reference/rkkw.md),
[`qbeta`](https://rdrr.io/r/stats/Beta.html)

## Author

Lopes, J. E.

## Examples

``` r
# \donttest{
# Example values
p_vals <- c(0.1, 0.5, 0.9)
alpha_par <- 2.0
beta_par <- 3.0
delta_par <- 0.5
lambda_par <- 1.5

# Calculate quantiles
quantiles <- qkkw(p_vals, alpha_par, beta_par, delta_par, lambda_par)
print(quantiles)
#> [1] 0.2425575 0.4631919 0.6851540

# Calculate quantiles for upper tail probabilities P(X > q) = p
# e.g., for p=0.1, find q such that P(X > q) = 0.1 (90th percentile)
quantiles_upper <- qkkw(p_vals, alpha_par, beta_par, delta_par, lambda_par,
                         lower.tail = FALSE)
print(quantiles_upper)
#> [1] 0.6851540 0.4631919 0.2425575
# Check: qkkw(p, ..., lt=F) == qkkw(1-p, ..., lt=T)
print(qkkw(1 - p_vals, alpha_par, beta_par, delta_par, lambda_par))
#> [1] 0.6851540 0.4631919 0.2425575

# Calculate quantiles from log probabilities
log.p_vals <- log(p_vals)
quantiles_logp <- qkkw(log.p_vals, alpha_par, beta_par, delta_par, lambda_par,
                        log.p = TRUE)
print(quantiles_logp)
#> [1] 0.2425575 0.4631919 0.6851540
# Check: should match original quantiles
print(quantiles)
#> [1] 0.2425575 0.4631919 0.6851540

# Compare with qgkw setting gamma = 1
quantiles_gkw <- qgkw(p_vals, alpha_par, beta_par, gamma = 1.0,
                      delta_par, lambda_par)
print(paste("Max difference:", max(abs(quantiles - quantiles_gkw)))) # Should be near zero
#> [1] "Max difference: 0"

# Verify inverse relationship with pkkw
p_check <- 0.75
q_calc <- qkkw(p_check, alpha_par, beta_par, delta_par, lambda_par)
p_recalc <- pkkw(q_calc, alpha_par, beta_par, delta_par, lambda_par)
print(paste("Original p:", p_check, " Recalculated p:", p_recalc))
#> [1] "Original p: 0.75  Recalculated p: 0.75"
# abs(p_check - p_recalc) < 1e-9 # Should be TRUE

# Boundary conditions
print(qkkw(c(0, 1), alpha_par, beta_par, delta_par, lambda_par)) # Should be 0, 1
#> [1] 0 1
print(qkkw(c(-Inf, 0), alpha_par, beta_par, delta_par, lambda_par, log.p = TRUE)) # Should be 0, 1
#> [1] 0 1

# }
```
