# Negative Log-Likelihood for the Exponentiated Kumaraswamy (EKw) Distribution

Computes the negative log-likelihood function for the Exponentiated
Kumaraswamy (EKw) distribution with parameters `alpha` (\\\alpha\\),
`beta` (\\\beta\\), and `lambda` (\\\lambda\\), given a vector of
observations. This distribution is the special case of the Generalized
Kumaraswamy (GKw) distribution where \\\gamma = 1\\ and \\\delta = 0\\.
This function is suitable for maximum likelihood estimation.

## Usage

``` r
llekw(par, data)
```

## Arguments

- par:

  A numeric vector of length 3 containing the distribution parameters in
  the order: `alpha` (\\\alpha \> 0\\), `beta` (\\\beta \> 0\\),
  `lambda` (\\\lambda \> 0\\).

- data:

  A numeric vector of observations. All values must be strictly between
  0 and 1 (exclusive).

## Value

Returns a single `double` value representing the negative log-likelihood
(\\-\ell(\theta\|\mathbf{x})\\). Returns `Inf` if any parameter values
in `par` are invalid according to their constraints, or if any value in
`data` is not in the interval (0, 1).

## Details

The Exponentiated Kumaraswamy (EKw) distribution is the GKw distribution
([`dekw`](https://evandeilton.github.io/gkwdist/reference/dekw.md)) with
\\\gamma=1\\ and \\\delta=0\\. Its probability density function (PDF)
is: \$\$ f(x \| \theta) = \lambda \alpha \beta x^{\alpha-1} (1 -
x^\alpha)^{\beta-1} \bigl\[1 - (1 - x^\alpha)^\beta \bigr\]^{\lambda -
1} \$\$ for \\0 \< x \< 1\\ and \\\theta = (\alpha, \beta, \lambda)\\.
The log-likelihood function \\\ell(\theta \| \mathbf{x})\\ for a sample
\\\mathbf{x} = (x_1, \dots, x_n)\\ is \\\sum\_{i=1}^n \ln f(x_i \|
\theta)\\: \$\$ \ell(\theta \| \mathbf{x}) = n\[\ln(\lambda) +
\ln(\alpha) + \ln(\beta)\] + \sum\_{i=1}^{n} \[(\alpha-1)\ln(x_i) +
(\beta-1)\ln(v_i) + (\lambda-1)\ln(w_i)\] \$\$ where:

- \\v_i = 1 - x_i^{\alpha}\\

- \\w_i = 1 - v_i^{\beta} = 1 - (1-x_i^{\alpha})^{\beta}\\

This function computes and returns the *negative* log-likelihood,
\\-\ell(\theta\|\mathbf{x})\\, suitable for minimization using
optimization routines like
[`optim`](https://rdrr.io/r/stats/optim.html). Numerical stability is
maintained similarly to
[`llgkw`](https://evandeilton.github.io/gkwdist/reference/llgkw.md).

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

[`llgkw`](https://evandeilton.github.io/gkwdist/reference/llgkw.md)
(parent distribution negative log-likelihood),
[`dekw`](https://evandeilton.github.io/gkwdist/reference/dekw.md),
[`pekw`](https://evandeilton.github.io/gkwdist/reference/pekw.md),
[`qekw`](https://evandeilton.github.io/gkwdist/reference/qekw.md),
[`rekw`](https://evandeilton.github.io/gkwdist/reference/rekw.md),
`grekw` (gradient, if available), `hsekw` (Hessian, if available),
[`optim`](https://rdrr.io/r/stats/optim.html)

## Author

Lopes, J. E.

## Examples

``` r
# \donttest{
## Example 1: Basic Log-Likelihood Evaluation

# Generate sample data
set.seed(123)
n <- 1000
true_params <- c(alpha = 2.5, beta = 3.5, lambda = 2.0)
data <- rekw(n, alpha = true_params[1], beta = true_params[2],
             lambda = true_params[3])

# Evaluate negative log-likelihood at true parameters
nll_true <- llekw(par = true_params, data = data)
cat("Negative log-likelihood at true parameters:", nll_true, "\n")
#> Negative log-likelihood at true parameters: -491.1647 

# Evaluate at different parameter values
test_params <- rbind(
  c(2.0, 3.0, 1.5),
  c(2.5, 3.5, 2.0),
  c(3.0, 4.0, 2.5)
)

nll_values <- apply(test_params, 1, function(p) llekw(p, data))
results <- data.frame(
  Alpha = test_params[, 1],
  Beta = test_params[, 2],
  Lambda = test_params[, 3],
  NegLogLik = nll_values
)
print(results, digits = 4)
#>   Alpha Beta Lambda NegLogLik
#> 1   2.0  3.0    1.5    -371.3
#> 2   2.5  3.5    2.0    -491.2
#> 3   3.0  4.0    2.5    -375.3


## Example 2: Maximum Likelihood Estimation

# Optimization using BFGS with analytical gradient
fit <- optim(
  par = c(2, 3, 1.5),
  fn = llekw,
  gr = grekw,
  data = data,
  method = "BFGS",
  hessian = TRUE
)

mle <- fit$par
names(mle) <- c("alpha", "beta", "lambda")
se <- sqrt(diag(solve(fit$hessian)))

results <- data.frame(
  Parameter = c("alpha", "beta", "lambda"),
  True = true_params,
  MLE = mle,
  SE = se,
  CI_Lower = mle - 1.96 * se,
  CI_Upper = mle + 1.96 * se
)
print(results, digits = 4)
#>        Parameter True   MLE     SE CI_Lower CI_Upper
#> alpha      alpha  2.5 2.663 0.5852   1.5156    3.810
#> beta        beta  3.5 3.653 0.4023   2.8643    4.441
#> lambda    lambda  2.0 1.843 0.5776   0.7107    2.975

cat("\nNegative log-likelihood at MLE:", fit$value, "\n")
#> 
#> Negative log-likelihood at MLE: -491.2984 
cat("AIC:", 2 * fit$value + 2 * length(mle), "\n")
#> AIC: -976.5968 
cat("BIC:", 2 * fit$value + length(mle) * log(n), "\n")
#> BIC: -961.8735 


## Example 3: Comparing Optimization Methods

methods <- c("BFGS", "L-BFGS-B", "Nelder-Mead", "CG")
start_params <- c(2, 3, 1.5)

comparison <- data.frame(
  Method = character(),
  Alpha = numeric(),
  Beta = numeric(),
  Lambda = numeric(),
  NegLogLik = numeric(),
  Convergence = integer(),
  stringsAsFactors = FALSE
)

for (method in methods) {
  if (method %in% c("BFGS", "CG")) {
    fit_temp <- optim(
      par = start_params,
      fn = llekw,
      gr = grekw,
      data = data,
      method = method
    )
  } else if (method == "L-BFGS-B") {
    fit_temp <- optim(
      par = start_params,
      fn = llekw,
      gr = grekw,
      data = data,
      method = method,
      lower = c(0.01, 0.01, 0.01),
      upper = c(100, 100, 100)
    )
  } else {
    fit_temp <- optim(
      par = start_params,
      fn = llekw,
      data = data,
      method = method
    )
  }

  comparison <- rbind(comparison, data.frame(
    Method = method,
    Alpha = fit_temp$par[1],
    Beta = fit_temp$par[2],
    Lambda = fit_temp$par[3],
    NegLogLik = fit_temp$value,
    Convergence = fit_temp$convergence,
    stringsAsFactors = FALSE
  ))
}

print(comparison, digits = 4, row.names = FALSE)
#>       Method Alpha  Beta Lambda NegLogLik Convergence
#>         BFGS 2.663 3.653  1.843    -491.3           0
#>     L-BFGS-B 2.663 3.653  1.843    -491.3           0
#>  Nelder-Mead 2.662 3.652  1.843    -491.3           0
#>           CG 2.505 3.552  2.008    -491.3           1


## Example 4: Likelihood Ratio Test

# Test H0: lambda = 2 vs H1: lambda free
loglik_full <- -fit$value

restricted_ll <- function(params_restricted, data, lambda_fixed) {
  llekw(par = c(params_restricted[1], params_restricted[2],
                lambda_fixed), data = data)
}

fit_restricted <- optim(
  par = c(mle[1], mle[2]),
  fn = restricted_ll,
  data = data,
  lambda_fixed = 2,
  method = "BFGS"
)

loglik_restricted <- -fit_restricted$value
lr_stat <- 2 * (loglik_full - loglik_restricted)
p_value <- pchisq(lr_stat, df = 1, lower.tail = FALSE)

cat("LR Statistic:", round(lr_stat, 4), "\n")
#> LR Statistic: 0.0657 
cat("P-value:", format.pval(p_value, digits = 4), "\n")
#> P-value: 0.7977 


## Example 5: Univariate Profile Likelihoods

# Profile for alpha
alpha_grid <- seq(mle[1] - 1, mle[1] + 1, length.out = 50)
alpha_grid <- alpha_grid[alpha_grid > 0]
profile_ll_alpha <- numeric(length(alpha_grid))

for (i in seq_along(alpha_grid)) {
  profile_fit <- optim(
    par = mle[-1],
    fn = function(p) llekw(c(alpha_grid[i], p), data),
    method = "BFGS"
  )
  profile_ll_alpha[i] <- -profile_fit$value
}

# Profile for beta
beta_grid <- seq(mle[2] - 1, mle[2] + 1, length.out = 50)
beta_grid <- beta_grid[beta_grid > 0]
profile_ll_beta <- numeric(length(beta_grid))

for (i in seq_along(beta_grid)) {
  profile_fit <- optim(
    par = mle[-2],
    fn = function(p) llekw(c(p[1], beta_grid[i], p[2]), data),
    method = "BFGS"
  )
  profile_ll_beta[i] <- -profile_fit$value
}

# Profile for lambda
lambda_grid <- seq(mle[3] - 1, mle[3] + 1, length.out = 50)
lambda_grid <- lambda_grid[lambda_grid > 0]
profile_ll_lambda <- numeric(length(lambda_grid))

for (i in seq_along(lambda_grid)) {
  profile_fit <- optim(
    par = mle[-3],
    fn = function(p) llekw(c(p[1], p[2], lambda_grid[i]), data),
    method = "BFGS"
  )
  profile_ll_lambda[i] <- -profile_fit$value
}

# 95% confidence threshold
chi_crit <- qchisq(0.95, df = 1)
threshold <- max(profile_ll_alpha) - chi_crit / 2

# Plot all profiles

plot(alpha_grid, profile_ll_alpha, type = "l", lwd = 2, col = "#2E4057",
     xlab = expression(alpha), ylab = "Profile Log-Likelihood",
     main = expression(paste("Profile: ", alpha)), las = 1)
abline(v = mle[1], col = "#8B0000", lty = 2, lwd = 2)
abline(v = true_params[1], col = "#006400", lty = 2, lwd = 2)
abline(h = threshold, col = "#808080", lty = 3, lwd = 1.5)
legend("topright", legend = c("MLE", "True", "95% CI"),
       col = c("#8B0000", "#006400", "#808080"),
       lty = c(2, 2, 3), lwd = 2, bty = "n", cex = 0.8)
grid(col = "gray90")


plot(beta_grid, profile_ll_beta, type = "l", lwd = 2, col = "#2E4057",
     xlab = expression(beta), ylab = "Profile Log-Likelihood",
     main = expression(paste("Profile: ", beta)), las = 1)
abline(v = mle[2], col = "#8B0000", lty = 2, lwd = 2)
abline(v = true_params[2], col = "#006400", lty = 2, lwd = 2)
abline(h = threshold, col = "#808080", lty = 3, lwd = 1.5)
legend("topright", legend = c("MLE", "True", "95% CI"),
       col = c("#8B0000", "#006400", "#808080"),
       lty = c(2, 2, 3), lwd = 2, bty = "n", cex = 0.8)
grid(col = "gray90")


plot(lambda_grid, profile_ll_lambda, type = "l", lwd = 2, col = "#2E4057",
     xlab = expression(lambda), ylab = "Profile Log-Likelihood",
     main = expression(paste("Profile: ", lambda)), las = 1)
abline(v = mle[3], col = "#8B0000", lty = 2, lwd = 2)
abline(v = true_params[3], col = "#006400", lty = 2, lwd = 2)
abline(h = threshold, col = "#808080", lty = 3, lwd = 1.5)
legend("topright", legend = c("MLE", "True", "95% CI"),
       col = c("#8B0000", "#006400", "#808080"),
       lty = c(2, 2, 3), lwd = 2, bty = "n", cex = 0.8)
grid(col = "gray90")



## Example 6: 2D Log-Likelihood Surface (Alpha vs Beta)

# Create 2D grid
alpha_2d <- seq(mle[1] - 0.8, mle[1] + 0.8, length.out = round(n/25))
beta_2d <- seq(mle[2] - 0.8, mle[2] + 0.8, length.out = round(n/25))
alpha_2d <- alpha_2d[alpha_2d > 0]
beta_2d <- beta_2d[beta_2d > 0]

# Compute log-likelihood surface
ll_surface_ab <- matrix(NA, nrow = length(alpha_2d), ncol = length(beta_2d))

for (i in seq_along(alpha_2d)) {
  for (j in seq_along(beta_2d)) {
    ll_surface_ab[i, j] <- -llekw(c(alpha_2d[i], beta_2d[j], mle[3]), data)
  }
}

# Confidence region levels
max_ll_ab <- max(ll_surface_ab, na.rm = TRUE)
levels_90_ab <- max_ll_ab - qchisq(0.90, df = 2) / 2
levels_95_ab <- max_ll_ab - qchisq(0.95, df = 2) / 2
levels_99_ab <- max_ll_ab - qchisq(0.99, df = 2) / 2

# Plot contour
contour(alpha_2d, beta_2d, ll_surface_ab,
        xlab = expression(alpha), ylab = expression(beta),
        main = "2D Log-Likelihood: Alpha vs Beta",
        levels = seq(min(ll_surface_ab, na.rm = TRUE), max_ll_ab, length.out = 20),
        col = "#2E4057", las = 1, lwd = 1)

contour(alpha_2d, beta_2d, ll_surface_ab,
        levels = c(levels_90_ab, levels_95_ab, levels_99_ab),
        col = c("#FFA07A", "#FF6347", "#8B0000"),
        lwd = c(2, 2.5, 3), lty = c(3, 2, 1),
        add = TRUE, labcex = 0.8)

points(mle[1], mle[2], pch = 19, col = "#8B0000", cex = 1.5)
points(true_params[1], true_params[2], pch = 17, col = "#006400", cex = 1.5)

legend("topright",
       legend = c("MLE", "True", "90% CR", "95% CR", "99% CR"),
       col = c("#8B0000", "#006400", "#FFA07A", "#FF6347", "#8B0000"),
       pch = c(19, 17, NA, NA, NA),
       lty = c(NA, NA, 3, 2, 1),
       lwd = c(NA, NA, 2, 2.5, 3),
       bty = "n", cex = 0.8)
grid(col = "gray90")



## Example 7: 2D Log-Likelihood Surface (Alpha vs Lambda)

# Create 2D grid
alpha_2d_2 <- seq(mle[1] - 0.8, mle[1] + 0.8, length.out = round(n/25))
lambda_2d <- seq(mle[3] - 0.8, mle[3] + 0.8, length.out = round(n/25))
alpha_2d_2 <- alpha_2d_2[alpha_2d_2 > 0]
lambda_2d <- lambda_2d[lambda_2d > 0]

# Compute log-likelihood surface
ll_surface_al <- matrix(NA, nrow = length(alpha_2d_2), ncol = length(lambda_2d))

for (i in seq_along(alpha_2d_2)) {
  for (j in seq_along(lambda_2d)) {
    ll_surface_al[i, j] <- -llekw(c(alpha_2d_2[i], mle[2], lambda_2d[j]), data)
  }
}

# Confidence region levels
max_ll_al <- max(ll_surface_al, na.rm = TRUE)
levels_90_al <- max_ll_al - qchisq(0.90, df = 2) / 2
levels_95_al <- max_ll_al - qchisq(0.95, df = 2) / 2
levels_99_al <- max_ll_al - qchisq(0.99, df = 2) / 2

# Plot contour
contour(alpha_2d_2, lambda_2d, ll_surface_al,
        xlab = expression(alpha), ylab = expression(lambda),
        main = "2D Log-Likelihood: Alpha vs Lambda",
        levels = seq(min(ll_surface_al, na.rm = TRUE), max_ll_al, length.out = 20),
        col = "#2E4057", las = 1, lwd = 1)

contour(alpha_2d_2, lambda_2d, ll_surface_al,
        levels = c(levels_90_al, levels_95_al, levels_99_al),
        col = c("#FFA07A", "#FF6347", "#8B0000"),
        lwd = c(2, 2.5, 3), lty = c(3, 2, 1),
        add = TRUE, labcex = 0.8)

points(mle[1], mle[3], pch = 19, col = "#8B0000", cex = 1.5)
points(true_params[1], true_params[3], pch = 17, col = "#006400", cex = 1.5)

legend("topright",
       legend = c("MLE", "True", "90% CR", "95% CR", "99% CR"),
       col = c("#8B0000", "#006400", "#FFA07A", "#FF6347", "#8B0000"),
       pch = c(19, 17, NA, NA, NA),
       lty = c(NA, NA, 3, 2, 1),
       lwd = c(NA, NA, 2, 2.5, 3),
       bty = "n", cex = 0.8)
grid(col = "gray90")



## Example 8: 2D Log-Likelihood Surface (Beta vs Lambda)

# Create 2D grid
beta_2d_2 <- seq(mle[2] - 0.8, mle[2] + 0.8, length.out = round(n/25))
lambda_2d_2 <- seq(mle[3] - 0.8, mle[3] + 0.8, length.out = round(n/25))
beta_2d_2 <- beta_2d_2[beta_2d_2 > 0]
lambda_2d_2 <- lambda_2d_2[lambda_2d_2 > 0]

# Compute log-likelihood surface
ll_surface_bl <- matrix(NA, nrow = length(beta_2d_2), ncol = length(lambda_2d_2))

for (i in seq_along(beta_2d_2)) {
  for (j in seq_along(lambda_2d_2)) {
    ll_surface_bl[i, j] <- -llekw(c(mle[1], beta_2d_2[i], lambda_2d_2[j]), data)
  }
}

# Confidence region levels
max_ll_bl <- max(ll_surface_bl, na.rm = TRUE)
levels_90_bl <- max_ll_bl - qchisq(0.90, df = 2) / 2
levels_95_bl <- max_ll_bl - qchisq(0.95, df = 2) / 2
levels_99_bl <- max_ll_bl - qchisq(0.99, df = 2) / 2

# Plot contour
contour(beta_2d_2, lambda_2d_2, ll_surface_bl,
        xlab = expression(beta), ylab = expression(lambda),
        main = "2D Log-Likelihood: Beta vs Lambda",
        levels = seq(min(ll_surface_bl, na.rm = TRUE), max_ll_bl, length.out = 20),
        col = "#2E4057", las = 1, lwd = 1)

contour(beta_2d_2, lambda_2d_2, ll_surface_bl,
        levels = c(levels_90_bl, levels_95_bl, levels_99_bl),
        col = c("#FFA07A", "#FF6347", "#8B0000"),
        lwd = c(2, 2.5, 3), lty = c(3, 2, 1),
        add = TRUE, labcex = 0.8)

points(mle[2], mle[3], pch = 19, col = "#8B0000", cex = 1.5)
points(true_params[2], true_params[3], pch = 17, col = "#006400", cex = 1.5)

legend("topright",
       legend = c("MLE", "True", "90% CR", "95% CR", "99% CR"),
       col = c("#8B0000", "#006400", "#FFA07A", "#FF6347", "#8B0000"),
       pch = c(19, 17, NA, NA, NA),
       lty = c(NA, NA, 3, 2, 1),
       lwd = c(NA, NA, 2, 2.5, 3),
       bty = "n", cex = 0.8)
grid(col = "gray90")


# }
```
