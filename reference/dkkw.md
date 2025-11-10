# Density of the Kumaraswamy-Kumaraswamy (kkw) Distribution

Computes the probability density function (PDF) for the
Kumaraswamy-Kumaraswamy (kkw) distribution with parameters `alpha`
(\\\alpha\\), `beta` (\\\beta\\), `delta` (\\\delta\\), and `lambda`
(\\\lambda\\). This distribution is defined on the interval (0, 1).

## Usage

``` r
dkkw(x, alpha, beta, delta, lambda, log_prob = FALSE)
```

## Arguments

- x:

  Vector of quantiles (values between 0 and 1).

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

- log_prob:

  Logical; if `TRUE`, the logarithm of the density is returned
  (\\\log(f(x))\\). Default: `FALSE`.

## Value

A vector of density values (\\f(x)\\) or log-density values
(\\\log(f(x))\\). The length of the result is determined by the
recycling rule applied to the arguments (`x`, `alpha`, `beta`, `delta`,
`lambda`). Returns `0` (or `-Inf` if `log_prob = TRUE`) for `x` outside
the interval (0, 1), or `NaN` if parameters are invalid (e.g.,
`alpha <= 0`, `beta <= 0`, `delta < 0`, `lambda <= 0`).

## Details

The Kumaraswamy-Kumaraswamy (kkw) distribution is a special case of the
five-parameter Generalized Kumaraswamy distribution
([`dgkw`](https://evandeilton.github.io/gkwdist/reference/dgkw.md))
obtained by setting the parameter \\\gamma = 1\\.

The probability density function is given by: \$\$ f(x; \alpha, \beta,
\delta, \lambda) = (\delta + 1) \lambda \alpha \beta x^{\alpha - 1} (1 -
x^\alpha)^{\beta - 1} \bigl\[1 - (1 - x^\alpha)^\beta\bigr\]^{\lambda -
1} \bigl\\1 - \bigl\[1 - (1 -
x^\alpha)^\beta\bigr\]^\lambda\bigr\\^{\delta} \$\$ for \\0 \< x \< 1\\.
Note that \\1/(\delta+1)\\ corresponds to the Beta function term \\B(1,
\delta+1)\\ when \\\gamma=1\\.

Numerical evaluation follows similar stability considerations as
[`dgkw`](https://evandeilton.github.io/gkwdist/reference/dgkw.md).

## References

Cordeiro, G. M., & de Castro, M. (2011). A new family of generalized
distributions. *Journal of Statistical Computation and Simulation*

Kumaraswamy, P. (1980). A generalized probability density function for
double-bounded random processes. *Journal of Hydrology*, *46*(1-2),
79-88.

## See also

[`dgkw`](https://evandeilton.github.io/gkwdist/reference/dgkw.md)
(parent distribution density),
[`pkkw`](https://evandeilton.github.io/gkwdist/reference/pkkw.md),
[`qkkw`](https://evandeilton.github.io/gkwdist/reference/qkkw.md),
[`rkkw`](https://evandeilton.github.io/gkwdist/reference/rkkw.md) (if
they exist), [`dbeta`](https://rdrr.io/r/stats/Beta.html)

## Author

Lopes, J. E.

## Examples

``` r
# \donttest{
# Example values
x_vals <- c(0.2, 0.5, 0.8)
alpha_par <- 2.0
beta_par <- 3.0
delta_par <- 0.5
lambda_par <- 1.5

# Calculate density
densities <- dkkw(x_vals, alpha_par, beta_par, delta_par, lambda_par)
print(densities)
#> [1] 0.8281038 2.1612055 0.3594057

# Calculate log-density
log_densities <- dkkw(x_vals, alpha_par, beta_par, delta_par, lambda_par,
                       log_prob = TRUE)
print(log_densities)
#> [1] -0.1886168  0.7706662 -1.0233034
# Check: should match log(densities)
print(log(densities))
#> [1] -0.1886168  0.7706662 -1.0233034

# Compare with dgkw setting gamma = 1
densities_gkw <- dgkw(x_vals, alpha_par, beta_par, gamma = 1.0,
                      delta_par, lambda_par)
print(paste("Max difference:", max(abs(densities - densities_gkw)))) # Should be near zero
#> [1] "Max difference: 8.88178419700125e-16"

# Plot the density
curve_x <- seq(0.01, 0.99, length.out = 200)
curve_y <- dkkw(curve_x, alpha_par, beta_par, delta_par, lambda_par)
plot(curve_x, curve_y, type = "l", main = "kkw Density Example",
     xlab = "x", ylab = "f(x)", col = "blue")


# }
```
