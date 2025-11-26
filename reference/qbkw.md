# Quantile Function of the Beta-Kumaraswamy (BKw) Distribution

Computes the quantile function (inverse CDF) for the Beta-Kumaraswamy
(BKw) distribution with parameters `alpha` (\\\alpha\\), `beta`
(\\\beta\\), `gamma` (\\\gamma\\), and `delta` (\\\delta\\). It finds
the value `q` such that \\P(X \le q) = p\\. This distribution is a
special case of the Generalized Kumaraswamy (GKw) distribution where the
parameter \\\lambda = 1\\.

## Usage

``` r
qbkw(
  p,
  alpha = 1,
  beta = 1,
  gamma = 1,
  delta = 0,
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

- gamma:

  Shape parameter `gamma` \> 0. Can be a scalar or a vector. Default:
  1.0.

- delta:

  Shape parameter `delta` \>= 0. Can be a scalar or a vector. Default:
  0.0.

- lower.tail:

  Logical; if `TRUE` (default), probabilities are \\p = P(X \le q)\\,
  otherwise, probabilities are \\p = P(X \> q)\\.

- log.p:

  Logical; if `TRUE`, probabilities `p` are given as \\\log(p)\\.
  Default: `FALSE`.

## Value

A vector of quantiles corresponding to the given probabilities `p`. The
length of the result is determined by the recycling rule applied to the
arguments (`p`, `alpha`, `beta`, `gamma`, `delta`). Returns:

- `0` for `p = 0` (or `p = -Inf` if `log.p = TRUE`, when
  `lower.tail = TRUE`).

- `1` for `p = 1` (or `p = 0` if `log.p = TRUE`, when
  `lower.tail = TRUE`).

- `NaN` for `p < 0` or `p > 1` (or corresponding log scale).

- `NaN` for invalid parameters (e.g., `alpha <= 0`, `beta <= 0`,
  `gamma <= 0`, `delta < 0`).

Boundary return values are adjusted accordingly for
`lower.tail = FALSE`.

## Details

The quantile function \\Q(p)\\ is the inverse of the CDF \\F(q)\\. The
CDF for the BKw (\\\lambda=1\\) distribution is \\F(q) =
I\_{y(q)}(\gamma, \delta+1)\\, where \\y(q) = 1 - (1 - q^\alpha)^\beta\\
and \\I_z(a,b)\\ is the regularized incomplete beta function (see
[`pbkw`](https://evandeilton.github.io/gkwdist/reference/pbkw.md)).

To find the quantile \\q\\, we first invert the outer Beta part: let \\y
= I^{-1}\_{p}(\gamma, \delta+1)\\, where \\I^{-1}\_p(a,b)\\ is the
inverse of the regularized incomplete beta function, computed via
[`qbeta`](https://rdrr.io/r/stats/Beta.html). Then, we invert the inner
Kumaraswamy part: \\y = 1 - (1 - q^\alpha)^\beta\\, which leads to \\q =
\\1 - (1-y)^{1/\beta}\\^{1/\alpha}\\. Substituting \\y\\ gives the
quantile function: \$\$ Q(p) = \left\\ 1 - \left\[ 1 -
I^{-1}\_{p}(\gamma, \delta+1) \right\]^{1/\beta} \right\\^{1/\alpha}
\$\$ The function uses this formula, calculating \\I^{-1}\_{p}(\gamma,
\delta+1)\\ via `qbeta(p, gamma, delta + 1, ...)` while respecting the
`lower.tail` and `log.p` arguments.

## References

Cordeiro, G. M., & de Castro, M. (2011). A new family of generalized
distributions. *Journal of Statistical Computation and Simulation*

Kumaraswamy, P. (1980). A generalized probability density function for
double-bounded random processes. *Journal of Hydrology*, *46*(1-2),
79-88.

## See also

[`qgkw`](https://evandeilton.github.io/gkwdist/reference/qgkw.md)
(parent distribution quantile function),
[`dbkw`](https://evandeilton.github.io/gkwdist/reference/dbkw.md),
[`pbkw`](https://evandeilton.github.io/gkwdist/reference/pbkw.md),
[`rbkw`](https://evandeilton.github.io/gkwdist/reference/rbkw.md) (other
BKw functions), [`qbeta`](https://rdrr.io/r/stats/Beta.html)

## Author

Lopes, J. E.

## Examples

``` r
# \donttest{
# Example values
p_vals <- c(0.1, 0.5, 0.9)
alpha_par <- 2.0
beta_par <- 1.5
gamma_par <- 1.0
delta_par <- 0.5

# Calculate quantiles
quantiles <- qbkw(p_vals, alpha_par, beta_par, gamma_par, delta_par)
print(quantiles)
#> [1] 0.2138865 0.5149104 0.8003866

# Calculate quantiles for upper tail probabilities P(X > q) = p
quantiles_upper <- qbkw(p_vals, alpha_par, beta_par, gamma_par, delta_par,
                        lower.tail = FALSE)
print(quantiles_upper)
#> [1] 0.8003866 0.5149104 0.2138865
# Check: qbkw(p, ..., lt=F) == qbkw(1-p, ..., lt=T)
print(qbkw(1 - p_vals, alpha_par, beta_par, gamma_par, delta_par))
#> [1] 0.8003866 0.5149104 0.2138865

# Calculate quantiles from log probabilities
log.p_vals <- log(p_vals)
quantiles_logp <- qbkw(log.p_vals, alpha_par, beta_par, gamma_par, delta_par,
                       log.p = TRUE)
print(quantiles_logp)
#> [1] 0.2138865 0.5149104 0.8003866
# Check: should match original quantiles
print(quantiles)
#> [1] 0.2138865 0.5149104 0.8003866

# Compare with qgkw setting lambda = 1
quantiles_gkw <- qgkw(p_vals, alpha_par, beta_par, gamma = gamma_par,
                     delta = delta_par, lambda = 1.0)
print(paste("Max difference:", max(abs(quantiles - quantiles_gkw)))) # Should be near zero
#> [1] "Max difference: 0"

# Verify inverse relationship with pbkw
p_check <- 0.75
q_calc <- qbkw(p_check, alpha_par, beta_par, gamma_par, delta_par)
p_recalc <- pbkw(q_calc, alpha_par, beta_par, gamma_par, delta_par)
print(paste("Original p:", p_check, " Recalculated p:", p_recalc))
#> [1] "Original p: 0.75  Recalculated p: 0.75"
# abs(p_check - p_recalc) < 1e-9 # Should be TRUE

# Boundary conditions
print(qbkw(c(0, 1), alpha_par, beta_par, gamma_par, delta_par)) # Should be 0, 1
#> [1] 0 1
print(qbkw(c(-Inf, 0), alpha_par, beta_par, gamma_par, delta_par, log.p = TRUE)) # Should be 0, 1
#> [1] 0 1

# }
```
