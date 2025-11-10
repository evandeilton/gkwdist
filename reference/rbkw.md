# Random Number Generation for the Beta-Kumaraswamy (BKw) Distribution

Generates random deviates from the Beta-Kumaraswamy (BKw) distribution
with parameters `alpha` (\\\alpha\\), `beta` (\\\beta\\), `gamma`
(\\\gamma\\), and `delta` (\\\delta\\). This distribution is a special
case of the Generalized Kumaraswamy (GKw) distribution where the
parameter \\\lambda = 1\\.

## Usage

``` r
rbkw(n, alpha, beta, gamma, delta)
```

## Arguments

- n:

  Number of observations. If `length(n) > 1`, the length is taken to be
  the number required. Must be a non-negative integer.

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

## Value

A vector of length `n` containing random deviates from the BKw
distribution. The length of the result is determined by `n` and the
recycling rule applied to the parameters (`alpha`, `beta`, `gamma`,
`delta`). Returns `NaN` if parameters are invalid (e.g., `alpha <= 0`,
`beta <= 0`, `gamma <= 0`, `delta < 0`).

## Details

The generation method uses the relationship between the GKw distribution
and the Beta distribution. The general procedure for GKw
([`rgkw`](https://evandeilton.github.io/gkwdist/reference/rgkw.md)) is:
If \\W \sim \mathrm{Beta}(\gamma, \delta+1)\\, then \\X = \\1 - \[1 -
W^{1/\lambda}\]^{1/\beta}\\^{1/\alpha}\\ follows the GKw(\\\alpha,
\beta, \gamma, \delta, \lambda\\) distribution.

For the BKw distribution, \\\lambda=1\\. Therefore, the algorithm
simplifies to:

1.  Generate \\V \sim \mathrm{Beta}(\gamma, \delta+1)\\ using
    [`rbeta`](https://rdrr.io/r/stats/Beta.html).

2.  Compute the BKw variate \\X = \\1 - (1 -
    V)^{1/\beta}\\^{1/\alpha}\\.

This procedure is implemented efficiently, handling parameter recycling
as needed.

## References

Cordeiro, G. M., & de Castro, M. (2011). A new family of generalized
distributions. *Journal of Statistical Computation and Simulation*,

Kumaraswamy, P. (1980). A generalized probability density function for
double-bounded random processes. *Journal of Hydrology*, *46*(1-2),
79-88.

Devroye, L. (1986). *Non-Uniform Random Variate Generation*.
Springer-Verlag. (General methods for random variate generation).

## See also

[`rgkw`](https://evandeilton.github.io/gkwdist/reference/rgkw.md)
(parent distribution random generation),
[`dbkw`](https://evandeilton.github.io/gkwdist/reference/dbkw.md),
[`pbkw`](https://evandeilton.github.io/gkwdist/reference/pbkw.md),
[`qbkw`](https://evandeilton.github.io/gkwdist/reference/qbkw.md) (other
BKw functions), [`rbeta`](https://rdrr.io/r/stats/Beta.html)

## Author

Lopes, J. E.

## Examples

``` r
# \donttest{
set.seed(2026) # for reproducibility

# Generate 1000 random values from a specific BKw distribution
alpha_par <- 2.0
beta_par <- 1.5
gamma_par <- 1.0
delta_par <- 0.5

x_sample_bkw <- rbkw(1000, alpha = alpha_par, beta = beta_par,
                     gamma = gamma_par, delta = delta_par)
summary(x_sample_bkw)
#>    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
#> 0.01187 0.36143 0.51194 0.51019 0.67906 0.99278 

# Histogram of generated values compared to theoretical density
hist(x_sample_bkw, breaks = 30, freq = FALSE, # freq=FALSE for density
     main = "Histogram of BKw Sample", xlab = "x", ylim = c(0, 2.5))
curve(dbkw(x, alpha = alpha_par, beta = beta_par, gamma = gamma_par,
           delta = delta_par),
      add = TRUE, col = "red", lwd = 2, n = 201)
legend("topright", legend = "Theoretical PDF", col = "red", lwd = 2, bty = "n")


# Comparing empirical and theoretical quantiles (Q-Q plot)
prob_points <- seq(0.01, 0.99, by = 0.01)
theo_quantiles <- qbkw(prob_points, alpha = alpha_par, beta = beta_par,
                       gamma = gamma_par, delta = delta_par)
emp_quantiles <- quantile(x_sample_bkw, prob_points, type = 7)

plot(theo_quantiles, emp_quantiles, pch = 16, cex = 0.8,
     main = "Q-Q Plot for BKw Distribution",
     xlab = "Theoretical Quantiles", ylab = "Empirical Quantiles (n=1000)")
abline(a = 0, b = 1, col = "blue", lty = 2)


# Compare summary stats with rgkw(..., lambda=1, ...)
# Note: individual values will differ due to randomness
x_sample_gkw <- rgkw(1000, alpha = alpha_par, beta = beta_par, gamma = gamma_par,
                     delta = delta_par, lambda = 1.0)
print("Summary stats for rbkw sample:")
#> [1] "Summary stats for rbkw sample:"
print(summary(x_sample_bkw))
#>    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
#> 0.01187 0.36143 0.51194 0.51019 0.67906 0.99278 
print("Summary stats for rgkw(lambda=1) sample:")
#> [1] "Summary stats for rgkw(lambda=1) sample:"
print(summary(x_sample_gkw)) # Should be similar
#>    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
#> 0.04279 0.36298 0.52400 0.51632 0.67744 0.98445 

# }
```
