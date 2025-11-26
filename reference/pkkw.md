# Cumulative Distribution Function (CDF) of the kkw Distribution

Computes the cumulative distribution function (CDF), \\P(X \le q)\\, for
the Kumaraswamy-Kumaraswamy (kkw) distribution with parameters `alpha`
(\\\alpha\\), `beta` (\\\beta\\), `delta` (\\\delta\\), and `lambda`
(\\\lambda\\). This distribution is defined on the interval (0, 1).

## Usage

``` r
pkkw(
  q,
  alpha = 1,
  beta = 1,
  delta = 0,
  lambda = 1,
  lower.tail = TRUE,
  log.p = FALSE
)
```

## Arguments

- q:

  Vector of quantiles (values generally between 0 and 1).

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

  Logical; if `TRUE` (default), probabilities are \\P(X \le q)\\,
  otherwise, \\P(X \> q)\\.

- log.p:

  Logical; if `TRUE`, probabilities \\p\\ are given as \\\log(p)\\.
  Default: `FALSE`.

## Value

A vector of probabilities, \\F(q)\\, or their logarithms/complements
depending on `lower.tail` and `log.p`. The length of the result is
determined by the recycling rule applied to the arguments (`q`, `alpha`,
`beta`, `delta`, `lambda`). Returns `0` (or `-Inf` if `log.p = TRUE`)
for `q <= 0` and `1` (or `0` if `log.p = TRUE`) for `q >= 1`. Returns
`NaN` for invalid parameters.

## Details

The Kumaraswamy-Kumaraswamy (kkw) distribution is a special case of the
five-parameter Generalized Kumaraswamy distribution
([`pgkw`](https://evandeilton.github.io/gkwdist/reference/pgkw.md))
obtained by setting the shape parameter \\\gamma = 1\\.

The CDF of the GKw distribution is \\F\_{GKw}(q) = I\_{y(q)}(\gamma,
\delta+1)\\, where \\y(q) = \[1-(1-q^{\alpha})^{\beta}\]^{\lambda}\\ and
\\I_x(a,b)\\ is the regularized incomplete beta function
([`pbeta`](https://rdrr.io/r/stats/Beta.html)). Setting \\\gamma=1\\
utilizes the property \\I_x(1, b) = 1 - (1-x)^b\\, yielding the kkw CDF:
\$\$ F(q; \alpha, \beta, \delta, \lambda) = 1 - \bigl\\1 - \bigl\[1 -
(1 - q^\alpha)^\beta\bigr\]^\lambda\bigr\\^{\delta + 1} \$\$ for \\0 \<
q \< 1\\.

The implementation uses this closed-form expression for efficiency and
handles `lower.tail` and `log.p` arguments appropriately.

## References

Cordeiro, G. M., & de Castro, M. (2011). A new family of generalized
distributions. *Journal of Statistical Computation and Simulation*

Kumaraswamy, P. (1980). A generalized probability density function for
double-bounded random processes. *Journal of Hydrology*, *46*(1-2),
79-88.

## See also

[`pgkw`](https://evandeilton.github.io/gkwdist/reference/pgkw.md)
(parent distribution CDF),
[`dkkw`](https://evandeilton.github.io/gkwdist/reference/dkkw.md),
[`qkkw`](https://evandeilton.github.io/gkwdist/reference/qkkw.md),
[`rkkw`](https://evandeilton.github.io/gkwdist/reference/rkkw.md),
[`pbeta`](https://rdrr.io/r/stats/Beta.html)

## Author

Lopes, J. E.

## Examples

``` r
# \donttest{
# Example values
q_vals <- c(0.2, 0.5, 0.8)
alpha_par <- 2.0
beta_par <- 3.0
delta_par <- 0.5
lambda_par <- 1.5

# Calculate CDF P(X <= q)
probs <- pkkw(q_vals, alpha_par, beta_par, delta_par, lambda_par)
print(probs)
#> [1] 0.05812108 0.58045681 0.98181161

# Calculate upper tail P(X > q)
probs_upper <- pkkw(q_vals, alpha_par, beta_par, delta_par, lambda_par,
                     lower.tail = FALSE)
print(probs_upper)
#> [1] 0.94187892 0.41954319 0.01818839
# Check: probs + probs_upper should be 1
print(probs + probs_upper)
#> [1] 1 1 1

# Calculate log CDF
logs <- pkkw(q_vals, alpha_par, beta_par, delta_par, lambda_par,
                   log.p = TRUE)
print(logs)
#> [1] -2.84522685 -0.54393988 -0.01835583
# Check: should match log(probs)
print(log(probs))
#> [1] -2.84522685 -0.54393988 -0.01835583

# Compare with pgkw setting gamma = 1
probs_gkw <- pgkw(q_vals, alpha_par, beta_par, gamma = 1.0,
                  delta_par, lambda_par)
print(paste("Max difference:", max(abs(probs - probs_gkw)))) # Should be near zero
#> [1] "Max difference: 1.04083408558608e-16"

# Plot the CDF
curve_q <- seq(0.01, 0.99, length.out = 200)
curve_p <- pkkw(curve_q, alpha_par, beta_par, delta_par, lambda_par)
plot(curve_q, curve_p, type = "l", main = "kkw CDF Example",
     xlab = "q", ylab = "F(q)", col = "blue", ylim = c(0, 1))


# }
```
