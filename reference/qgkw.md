# Generalized Kumaraswamy Distribution Quantile Function

Computes the quantile function (inverse CDF) for the five-parameter
Generalized Kumaraswamy (GKw) distribution. Finds the value `x` such
that \\P(X \le x) = p\\, where `X` follows the GKw distribution.

## Usage

``` r
qgkw(
  p,
  alpha = 1,
  beta = 1,
  gamma = 1,
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

- gamma:

  Shape parameter `gamma` \> 0. Can be a scalar or a vector. Default:
  1.0.

- delta:

  Shape parameter `delta` \>= 0. Can be a scalar or a vector. Default:
  0.0.

- lambda:

  Shape parameter `lambda` \> 0. Can be a scalar or a vector. Default:
  1.0.

- lower.tail:

  Logical; if `TRUE` (default), probabilities are \\P(X \le x)\\,
  otherwise, \\P(X \> x)\\.

- log.p:

  Logical; if `TRUE`, probabilities `p` are given as \\\log(p)\\.
  Default: `FALSE`.

## Value

A vector of quantiles corresponding to the given probabilities `p`. The
length of the result is determined by the recycling rule applied to the
arguments (`p`, `alpha`, `beta`, `gamma`, `delta`, `lambda`). Returns:

- `0` for `p = 0` (or `p = -Inf` if `log.p = TRUE`).

- `1` for `p = 1` (or `p = 0` if `log.p = TRUE`).

- `NaN` for `p < 0` or `p > 1` (or corresponding log scale).

- `NaN` for invalid parameters (e.g., `alpha <= 0`, `beta <= 0`,
  `gamma <= 0`, `delta < 0`, `lambda <= 0`).

## Details

The quantile function \\Q(p)\\ is the inverse of the CDF \\F(x)\\. Given
\\F(x) = I\_{y(x)}(\gamma, \delta+1)\\ where \\y(x) =
\[1-(1-x^{\alpha})^{\beta}\]^{\lambda}\\, the quantile function is: \$\$
Q(p) = x = \left\\ 1 - \left\[ 1 - \left( I^{-1}\_{p}(\gamma, \delta+1)
\right)^{1/\lambda} \right\]^{1/\beta} \right\\^{1/\alpha} \$\$ where
\\I^{-1}\_{p}(a, b)\\ is the inverse of the regularized incomplete beta
function, which corresponds to the quantile function of the Beta
distribution, [`qbeta`](https://rdrr.io/r/stats/Beta.html).

The computation proceeds as follows:

1.  Calculate
    `y = stats::qbeta(p, shape1 = gamma, shape2 = delta + 1, lower.tail = lower.tail, log.p = log.p)`.

2.  Calculate \\v = y^{1/\lambda}\\.

3.  Calculate \\w = (1 - v)^{1/\beta}\\. Note: Requires \\v \le 1\\.

4.  Calculate \\q = (1 - w)^{1/\alpha}\\. Note: Requires \\w \le 1\\.

Numerical stability is maintained by handling boundary cases (`p = 0`,
`p = 1`) directly and checking intermediate results (e.g., ensuring
arguments to powers are non-negative).

## References

Cordeiro, G. M., & de Castro, M. (2011). A new family of generalized
distributions. *Journal of Statistical Computation and Simulation*

Kumaraswamy, P. (1980). A generalized probability density function for
double-bounded random processes. *Journal of Hydrology*, *46*(1-2),
79-88.

## See also

[`dgkw`](https://evandeilton.github.io/gkwdist/reference/dgkw.md),
[`pgkw`](https://evandeilton.github.io/gkwdist/reference/pgkw.md),
[`rgkw`](https://evandeilton.github.io/gkwdist/reference/rgkw.md),
[`qbeta`](https://rdrr.io/r/stats/Beta.html)

## Author

Lopes, J. E.

## Examples

``` r
# \donttest{
# Basic quantile calculation (median)
median_val <- qgkw(0.5, alpha = 2, beta = 3, gamma = 1, delta = 0, lambda = 1)
print(median_val)
#> [1] 0.454202

# Computing multiple quantiles
probs <- c(0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)
quantiles <- qgkw(probs, alpha = 2, beta = 3, gamma = 1, delta = 0, lambda = 1)
print(quantiles)
#> [1] 0.05783171 0.18577033 0.30238999 0.45420202 0.60830870 0.73201169 0.88575196

# Upper tail quantile (e.g., find x such that P(X > x) = 0.1, which is 90th percentile)
q90 <- qgkw(0.1, alpha = 2, beta = 3, gamma = 1, delta = 0, lambda = 1,
            lower.tail = FALSE)
print(q90)
#> [1] 0.7320117
# Check: should match quantile for p = 0.9 with lower.tail = TRUE
print(qgkw(0.9, alpha = 2, beta = 3, gamma = 1, delta = 0, lambda = 1))
#> [1] 0.7320117

# Log probabilities
median_logp <- qgkw(log(0.5), alpha = 2, beta = 3, gamma = 1, delta = 0, lambda = 1,
                    log.p = TRUE)
print(median_logp) # Should match median_val
#> [1] 0.454202

# Vectorized parameters
alphas_vec <- c(0.5, 1.0, 2.0)
betas_vec <- c(1.0, 2.0, 3.0)
# Get median for 3 different GKw distributions
medians_vec <- qgkw(0.5, alpha = alphas_vec, beta = betas_vec, gamma = 1, delta = 0, lambda = 1)
print(medians_vec)
#> [1] 0.2500000 0.2928932 0.4542020

# Verify inverse relationship with pgkw
p_val <- 0.75
x_val <- qgkw(p_val, alpha = 2, beta = 3, gamma = 1, delta = 0, lambda = 1)
p_check <- pgkw(x_val, alpha = 2, beta = 3, gamma = 1, delta = 0, lambda = 1)
print(paste("Calculated p:", p_check, " (Expected:", p_val, ")"))
#> [1] "Calculated p: 0.75  (Expected: 0.75 )"
# }
```
