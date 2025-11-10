# Quantile Function of the Kumaraswamy (Kw) Distribution

Computes the quantile function (inverse CDF) for the two-parameter
Kumaraswamy (Kw) distribution with shape parameters `alpha` (\\\alpha\\)
and `beta` (\\\beta\\). It finds the value `q` such that \\P(X \le q) =
p\\.

## Usage

``` r
qkw(p, alpha, beta, lower_tail = TRUE, log_p = FALSE)
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

- lower_tail:

  Logical; if `TRUE` (default), probabilities are \\p = P(X \le q)\\,
  otherwise, probabilities are \\p = P(X \> q)\\.

- log_p:

  Logical; if `TRUE`, probabilities `p` are given as \\\log(p)\\.
  Default: `FALSE`.

## Value

A vector of quantiles corresponding to the given probabilities `p`. The
length of the result is determined by the recycling rule applied to the
arguments (`p`, `alpha`, `beta`). Returns:

- `0` for `p = 0` (or `p = -Inf` if `log_p = TRUE`, when
  `lower_tail = TRUE`).

- `1` for `p = 1` (or `p = 0` if `log_p = TRUE`, when
  `lower_tail = TRUE`).

- `NaN` for `p < 0` or `p > 1` (or corresponding log scale).

- `NaN` for invalid parameters (e.g., `alpha <= 0`, `beta <= 0`).

Boundary return values are adjusted accordingly for
`lower_tail = FALSE`.

## Details

The quantile function \\Q(p)\\ is the inverse of the CDF \\F(q)\\. The
CDF for the Kumaraswamy distribution is \\F(q) = 1 - (1 -
q^\alpha)^\beta\\ (see
[`pkw`](https://evandeilton.github.io/gkwdist/reference/pkw.md)).
Inverting this equation for \\q\\ yields the quantile function: \$\$
Q(p) = \left\\ 1 - (1 - p)^{1/\beta} \right\\^{1/\alpha} \$\$ The
function uses this closed-form expression and correctly handles the
`lower_tail` and `log_p` arguments by transforming `p` appropriately
before applying the formula. This is equivalent to the general GKw
quantile function
([`qgkw`](https://evandeilton.github.io/gkwdist/reference/qgkw.md))
evaluated with \\\gamma=1, \delta=0, \lambda=1\\.

## References

Kumaraswamy, P. (1980). A generalized probability density function for
double-bounded random processes. *Journal of Hydrology*, *46*(1-2),
79-88.

Jones, M. C. (2009). Kumaraswamy's distribution: A beta-type
distribution with some tractability advantages. *Statistical
Methodology*, *6*(1), 70-81.

## See also

[`qgkw`](https://evandeilton.github.io/gkwdist/reference/qgkw.md)
(parent distribution quantile function),
[`dkw`](https://evandeilton.github.io/gkwdist/reference/dkw.md),
[`pkw`](https://evandeilton.github.io/gkwdist/reference/pkw.md),
[`rkw`](https://evandeilton.github.io/gkwdist/reference/rkw.md) (other
Kw functions), [`qbeta`](https://rdrr.io/r/stats/Beta.html),
[`qunif`](https://rdrr.io/r/stats/Uniform.html)

## Author

Lopes, J. E.

## Examples

``` r
# \donttest{
# Example values
p_vals <- c(0.1, 0.5, 0.9)
alpha_par <- 2.0
beta_par <- 3.0

# Calculate quantiles using qkw
quantiles <- qkw(p_vals, alpha_par, beta_par)
print(quantiles)
#> [1] 0.1857703 0.4542020 0.7320117

# Calculate quantiles for upper tail probabilities P(X > q) = p
quantiles_upper <- qkw(p_vals, alpha_par, beta_par, lower_tail = FALSE)
print(quantiles_upper)
#> [1] 0.7320117 0.4542020 0.1857703

# Calculate quantiles from log probabilities
log_p_vals <- log(p_vals)
quantiles_logp <- qkw(log_p_vals, alpha_par, beta_par, log_p = TRUE)
print(quantiles_logp)
#> [1] 0.1857703 0.4542020 0.7320117
# Check: should match original quantiles
print(quantiles)
#> [1] 0.1857703 0.4542020 0.7320117

# Compare with qgkw setting gamma = 1, delta = 0, lambda = 1
quantiles_gkw <- qgkw(p_vals, alpha = alpha_par, beta = beta_par,
                     gamma = 1.0, delta = 0.0, lambda = 1.0)
print(paste("Max difference:", max(abs(quantiles - quantiles_gkw)))) # Should be near zero
#> [1] "Max difference: 2.77555756156289e-17"

# Verify inverse relationship with pkw
p_check <- 0.75
q_calc <- qkw(p_check, alpha_par, beta_par)
p_recalc <- pkw(q_calc, alpha_par, beta_par)
print(paste("Original p:", p_check, " Recalculated p:", p_recalc))
#> [1] "Original p: 0.75  Recalculated p: 0.75"
# abs(p_check - p_recalc) < 1e-9 # Should be TRUE

# Boundary conditions
print(qkw(c(0, 1), alpha_par, beta_par)) # Should be 0, 1
#> [1] 0 1
print(qkw(c(-Inf, 0), alpha_par, beta_par, log_p = TRUE)) # Should be 0, 1
#> [1] 0 1

# }
```
