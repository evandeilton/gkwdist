# Quantile Function of the Beta Distribution (gamma, delta+1 Parameterization)

Computes the quantile function (inverse CDF) for the standard Beta
distribution, using a parameterization common in generalized
distribution families. It finds the value `q` such that \\P(X \le q) =
p\\. The distribution is parameterized by `gamma` (\\\gamma\\) and
`delta` (\\\delta\\), corresponding to the standard Beta distribution
with shape parameters `shape1 = gamma` and `shape2 = delta + 1`.

## Usage

``` r
qbeta_(p, gamma, delta, lower_tail = TRUE, log_p = FALSE)
```

## Arguments

- p:

  Vector of probabilities (values between 0 and 1).

- gamma:

  First shape parameter (`shape1`), \\\gamma \> 0\\. Can be a scalar or
  a vector. Default: 1.0.

- delta:

  Second shape parameter is `delta + 1` (`shape2`), requires \\\delta
  \ge 0\\ so that `shape2 >= 1`. Can be a scalar or a vector. Default:
  0.0 (leading to `shape2 = 1`).

- lower_tail:

  Logical; if `TRUE` (default), probabilities are \\p = P(X \le q)\\,
  otherwise, probabilities are \\p = P(X \> q)\\.

- log_p:

  Logical; if `TRUE`, probabilities `p` are given as \\\log(p)\\.
  Default: `FALSE`.

## Value

A vector of quantiles corresponding to the given probabilities `p`. The
length of the result is determined by the recycling rule applied to the
arguments (`p`, `gamma`, `delta`). Returns:

- `0` for `p = 0` (or `p = -Inf` if `log_p = TRUE`, when
  `lower_tail = TRUE`).

- `1` for `p = 1` (or `p = 0` if `log_p = TRUE`, when
  `lower_tail = TRUE`).

- `NaN` for `p < 0` or `p > 1` (or corresponding log scale).

- `NaN` for invalid parameters (e.g., `gamma <= 0`, `delta < 0`).

Boundary return values are adjusted accordingly for
`lower_tail = FALSE`.

## Details

This function computes the quantiles of a Beta distribution with
parameters `shape1 = gamma` and `shape2 = delta + 1`. It is equivalent
to calling
`stats::qbeta(p, shape1 = gamma, shape2 = delta + 1, lower.tail = lower_tail, log.p = log_p)`.

This distribution arises as a special case of the five-parameter
Generalized Kumaraswamy (GKw) distribution
([`qgkw`](https://evandeilton.github.io/gkwdist/reference/qgkw.md))
obtained by setting \\\alpha = 1\\, \\\beta = 1\\, and \\\lambda = 1\\.
It is therefore also equivalent to the McDonald (Mc)/Beta Power
distribution
([`qmc`](https://evandeilton.github.io/gkwdist/reference/qmc.md)) with
\\\lambda = 1\\.

The function likely calls R's underlying `qbeta` function but ensures
consistent parameter recycling and handling within the C++ environment,
matching the style of other functions in the related families. Boundary
conditions (p=0, p=1) are handled explicitly.

## References

Johnson, N. L., Kotz, S., & Balakrishnan, N. (1995). *Continuous
Univariate Distributions, Volume 2* (2nd ed.). Wiley.

Cordeiro, G. M., & de Castro, M. (2011). A new family of generalized
distributions. *Journal of Statistical Computation and Simulation*,

## See also

[`qbeta`](https://rdrr.io/r/stats/Beta.html) (standard R
implementation),
[`qgkw`](https://evandeilton.github.io/gkwdist/reference/qgkw.md)
(parent distribution quantile function),
[`qmc`](https://evandeilton.github.io/gkwdist/reference/qmc.md)
(McDonald/Beta Power quantile function), `dbeta_`, `pbeta_`, `rbeta_`
(other functions for this parameterization, if they exist).

## Author

Lopes, J. E.

## Examples

``` r
# \donttest{
# Example values
p_vals <- c(0.1, 0.5, 0.9)
gamma_par <- 2.0 # Corresponds to shape1
delta_par <- 3.0 # Corresponds to shape2 - 1
shape1 <- gamma_par
shape2 <- delta_par + 1

# Calculate quantiles using qbeta_
quantiles <- qbeta_(p_vals, gamma_par, delta_par)
print(quantiles)
#> [1] 0.1122350 0.3138102 0.5838904

# Compare with stats::qbeta
quantiles_stats <- stats::qbeta(p_vals, shape1 = shape1, shape2 = shape2)
print(paste("Max difference vs stats::qbeta:", max(abs(quantiles - quantiles_stats))))
#> [1] "Max difference vs stats::qbeta: 0"

# Compare with qgkw setting alpha=1, beta=1, lambda=1
quantiles_gkw <- qgkw(p_vals, alpha = 1.0, beta = 1.0, gamma = gamma_par,
                      delta = delta_par, lambda = 1.0)
print(paste("Max difference vs qgkw:", max(abs(quantiles - quantiles_gkw))))
#> [1] "Max difference vs qgkw: 5.55111512312578e-17"

# Compare with qmc setting lambda=1
quantiles_mc <- qmc(p_vals, gamma = gamma_par, delta = delta_par, lambda = 1.0)
print(paste("Max difference vs qmc:", max(abs(quantiles - quantiles_mc))))
#> [1] "Max difference vs qmc: 0"

# Calculate quantiles for upper tail
quantiles_upper <- qbeta_(p_vals, gamma_par, delta_par, lower_tail = FALSE)
print(quantiles_upper)
#> [1] 0.5838904 0.3138102 0.1122350
print(stats::qbeta(p_vals, shape1, shape2, lower.tail = FALSE))
#> [1] 0.5838904 0.3138102 0.1122350

# Calculate quantiles from log probabilities
log_p_vals <- log(p_vals)
quantiles_logp <- qbeta_(log_p_vals, gamma_par, delta_par, log_p = TRUE)
print(quantiles_logp)
#> [1] 0.1122350 0.3138102 0.5838904
print(stats::qbeta(log_p_vals, shape1, shape2, log.p = TRUE))
#> [1] 0.1122350 0.3138102 0.5838904

# Verify inverse relationship with pbeta_
p_check <- 0.75
q_calc <- qbeta_(p_check, gamma_par, delta_par)
p_recalc <- pbeta_(q_calc, gamma_par, delta_par)
print(paste("Original p:", p_check, " Recalculated p:", p_recalc))
#> [1] "Original p: 0.75  Recalculated p: 0.75"
# abs(p_check - p_recalc) < 1e-9 # Should be TRUE

# Boundary conditions
print(qbeta_(c(0, 1), gamma_par, delta_par)) # Should be 0, 1
#> [1] 0 1
print(qbeta_(c(-Inf, 0), gamma_par, delta_par, log_p = TRUE)) # Should be 0, 1
#> [1] 0 1

# }
```
