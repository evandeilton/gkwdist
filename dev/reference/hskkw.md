# Hessian Matrix of the Negative Log-Likelihood for the kkw Distribution

Computes the analytic 4x4 Hessian matrix (matrix of second partial
derivatives) of the negative log-likelihood function for the
Kumaraswamy-Kumaraswamy (kkw) distribution with parameters `alpha`
(\\\alpha\\), `beta` (\\\beta\\), `delta` (\\\delta\\), and `lambda`
(\\\lambda\\). This distribution is the special case of the Generalized
Kumaraswamy (GKw) distribution where \\\gamma = 1\\. The Hessian is
useful for estimating standard errors and in optimization algorithms.

## Usage

``` r
hskkw(par, data)
```

## Arguments

- par:

  A numeric vector of length 4 containing the distribution parameters in
  the order: `alpha` (\\\alpha \> 0\\), `beta` (\\\beta \> 0\\), `delta`
  (\\\delta \ge 0\\), `lambda` (\\\lambda \> 0\\).

- data:

  A numeric vector of observations. All values must be strictly between
  0 and 1 (exclusive).

## Value

Returns a 4x4 numeric matrix representing the Hessian matrix of the
negative log-likelihood function, \\-\partial^2 \ell / (\partial
\theta_i \partial \theta_j)\\, where \\\theta = (\alpha, \beta, \delta,
\lambda)\\. Returns a 4x4 matrix populated with `NaN` if any parameter
values are invalid according to their constraints, or if any value in
`data` is not in the interval (0, 1).

## Details

This function calculates the analytic second partial derivatives of the
negative log-likelihood function based on the kkw log-likelihood
(\\\gamma=1\\ case of GKw, see
[`llkkw`](https://evandeilton.github.io/gkwdist/dev/reference/llkkw.md)):
\$\$ \ell(\theta \| \mathbf{x}) = n\[\ln(\delta+1) + \ln(\lambda) +
\ln(\alpha) + \ln(\beta)\] + \sum\_{i=1}^{n} \[(\alpha-1)\ln(x_i) +
(\beta-1)\ln(v_i) + (\lambda-1)\ln(w_i) + \delta\ln(z_i)\] \$\$ where
\\\theta = (\alpha, \beta, \delta, \lambda)\\ and intermediate terms
are:

- \\v_i = 1 - x_i^{\alpha}\\

- \\w_i = 1 - v_i^{\beta} = 1 - (1-x_i^{\alpha})^{\beta}\\

- \\z_i = 1 - w_i^{\lambda} = 1 -
  \[1-(1-x_i^{\alpha})^{\beta}\]^{\lambda}\\

The Hessian matrix returned contains the elements \\- \frac{\partial^2
\ell(\theta \| \mathbf{x})}{\partial \theta_i \partial \theta_j}\\ for
\\\theta_i, \theta_j \in \\\alpha, \beta, \delta, \lambda\\\\.

Key properties of the returned matrix:

- Dimensions: 4x4.

- Symmetry: The matrix is symmetric.

- Ordering: Rows and columns correspond to the parameters in the order
  \\\alpha, \beta, \delta, \lambda\\.

- Content: Analytic second derivatives of the *negative* log-likelihood.

This corresponds to the relevant submatrix of the 5x5 GKw Hessian
([`hsgkw`](https://evandeilton.github.io/gkwdist/dev/reference/hsgkw.md))
evaluated at \\\gamma=1\\. The exact analytical formulas are implemented
directly.

## References

Cordeiro, G. M., & de Castro, M. (2011). A new family of generalized
distributions. *Journal of Statistical Computation and Simulation*

Kumaraswamy, P. (1980). A generalized probability density function for
double-bounded random processes. *Journal of Hydrology*, *46*(1-2),
79-88.

## See also

[`hsgkw`](https://evandeilton.github.io/gkwdist/dev/reference/hsgkw.md)
(parent distribution Hessian),
[`llkkw`](https://evandeilton.github.io/gkwdist/dev/reference/llkkw.md)
(negative log-likelihood for kkw),
[`grkkw`](https://evandeilton.github.io/gkwdist/dev/reference/grkkw.md)
(gradient for kkw),
[`dkkw`](https://evandeilton.github.io/gkwdist/dev/reference/dkkw.md)
(density for kkw), [`optim`](https://rdrr.io/r/stats/optim.html),
[`hessian`](https://rdrr.io/pkg/numDeriv/man/hessian.html) (for
numerical Hessian comparison).

## Author

Lopes, J. E.

## Examples

``` r
# \donttest{
## Example 1: Basic Hessian Evaluation

# Generate sample data
set.seed(123)
n <- 1000
true_params <- c(alpha = 2.0, beta = 3.0, delta = 1.5, lambda = 2.0)
data <- rkkw(n, alpha = true_params[1], beta = true_params[2],
             delta = true_params[3], lambda = true_params[4])

# Evaluate Hessian at true parameters
hess_true <- hskkw(par = true_params, data = data)
cat("Hessian matrix at true parameters:\n")
#> Hessian matrix at true parameters:
print(hess_true, digits = 4)
#>        [,1]   [,2]   [,3]   [,4]
#> [1,] 1557.6 -564.6 -425.5  910.9
#> [2,] -564.6  247.6  195.1 -307.5
#> [3,] -425.5  195.1  160.0 -225.7
#> [4,]  910.9 -307.5 -225.7  541.1

# Check symmetry
cat("\nSymmetry check (max |H - H^T|):",
    max(abs(hess_true - t(hess_true))), "\n")
#> 
#> Symmetry check (max |H - H^T|): 0 


## Example 2: Hessian Properties at MLE

# Fit model
fit <- optim(
  par = c(1.5, 2.5, 1.0, 1.5),
  fn = llkkw,
  gr = grkkw,
  data = data,
  method = "BFGS",
  hessian = TRUE
)

mle <- fit$par
names(mle) <- c("alpha", "beta", "delta", "lambda")

# Hessian at MLE
hessian_at_mle <- hskkw(par = mle, data = data)
cat("\nHessian at MLE:\n")
#> 
#> Hessian at MLE:
print(hessian_at_mle, digits = 4)
#>        [,1]   [,2]   [,3]   [,4]
#> [1,] 1188.9 -371.1 -423.3  881.2
#> [2,] -371.1  140.8  168.0 -251.3
#> [3,] -423.3  168.0  202.5 -280.4
#> [4,]  881.2 -251.3 -280.4  678.3

# Compare with optim's numerical Hessian
cat("\nComparison with optim Hessian:\n")
#> 
#> Comparison with optim Hessian:
cat("Max absolute difference:",
    max(abs(hessian_at_mle - fit$hessian)), "\n")
#> Max absolute difference: 0.0003078667 

# Eigenvalue analysis
eigenvals <- eigen(hessian_at_mle, only.values = TRUE)$values
cat("\nEigenvalues:\n")
#> 
#> Eigenvalues:
print(eigenvals)
#> [1] 2.109731e+03 9.911840e+01 1.552331e+00 9.866428e-03

cat("\nPositive definite:", all(eigenvals > 0), "\n")
#> 
#> Positive definite: TRUE 
cat("Condition number:", max(eigenvals) / min(eigenvals), "\n")
#> Condition number: 213829.2 


## Example 3: Standard Errors and Confidence Intervals

# Observed information matrix
obs_info <- hessian_at_mle

# Variance-covariance matrix
vcov_matrix <- solve(obs_info)
cat("\nVariance-Covariance Matrix:\n")
#> 
#> Variance-Covariance Matrix:
print(vcov_matrix, digits = 6)
#>          [,1]     [,2]      [,3]      [,4]
#> [1,]  4.72527  17.8381  -9.97973  -3.65563
#> [2,] 17.83805  71.2209 -40.54649 -13.54957
#> [3,] -9.97973 -40.5465  23.21460   7.54000
#> [4,] -3.65563 -13.5496   7.54000   2.84775

# Standard errors
se <- sqrt(diag(vcov_matrix))
names(se) <- c("alpha", "beta", "delta", "lambda")

# Correlation matrix
corr_matrix <- cov2cor(vcov_matrix)
cat("\nCorrelation Matrix:\n")
#> 
#> Correlation Matrix:
print(corr_matrix, digits = 4)
#>         [,1]    [,2]    [,3]    [,4]
#> [1,]  1.0000  0.9724 -0.9529 -0.9965
#> [2,]  0.9724  1.0000 -0.9972 -0.9514
#> [3,] -0.9529 -0.9972  1.0000  0.9273
#> [4,] -0.9965 -0.9514  0.9273  1.0000

# Confidence intervals
z_crit <- qnorm(0.975)
results <- data.frame(
  Parameter = c("alpha", "beta", "delta", "lambda"),
  True = true_params,
  MLE = mle,
  SE = se,
  CI_Lower = mle - z_crit * se,
  CI_Upper = mle + z_crit * se
)
print(results, digits = 4)
#>        Parameter True   MLE    SE CI_Lower CI_Upper
#> alpha      alpha  2.0 2.304 2.174   -1.956    6.565
#> beta        beta  3.0 3.610 8.439  -12.931   20.150
#> delta      delta  1.5 1.222 4.818   -8.221   10.666
#> lambda    lambda  2.0 1.705 1.688   -1.603    5.012


## Example 4: Determinant and Trace Analysis

# Compute at different points
test_params <- rbind(
  c(1.5, 2.5, 1.0, 1.5),
  c(2.0, 3.0, 1.5, 2.0),
  mle,
  c(2.5, 3.5, 2.0, 2.5)
)

hess_properties <- data.frame(
  Alpha = numeric(),
  Beta = numeric(),
  Delta = numeric(),
  Lambda = numeric(),
  Determinant = numeric(),
  Trace = numeric(),
  Min_Eigenval = numeric(),
  Max_Eigenval = numeric(),
  Cond_Number = numeric(),
  stringsAsFactors = FALSE
)

for (i in 1:nrow(test_params)) {
  H <- hskkw(par = test_params[i, ], data = data)
  eigs <- eigen(H, only.values = TRUE)$values

  hess_properties <- rbind(hess_properties, data.frame(
    Alpha = test_params[i, 1],
    Beta = test_params[i, 2],
    Delta = test_params[i, 3],
    Lambda = test_params[i, 4],
    Determinant = det(H),
    Trace = sum(diag(H)),
    Min_Eigenval = min(eigs),
    Max_Eigenval = max(eigs),
    Cond_Number = max(eigs) / min(eigs)
  ))
}

cat("\nHessian Properties at Different Points:\n")
#> 
#> Hessian Properties at Different Points:
print(hess_properties, digits = 4, row.names = FALSE)
#>  Alpha Beta Delta Lambda Determinant Trace Min_Eigenval Max_Eigenval
#>  1.500 2.50 1.000  1.500  -5.769e+09  3399   -1.056e+02         3061
#>  2.000 3.00 1.500  2.000  -1.317e+06  2506   -2.387e+00         2413
#>  2.304 3.61 1.222  1.705   3.203e+03  2210    9.866e-03         2110
#>  2.500 3.50 2.000  2.500  -2.204e+09  1873   -2.309e+02         1954
#>  Cond_Number
#>      -28.985
#>    -1011.104
#>   213829.226
#>       -8.462


## Example 5: Curvature Visualization (Alpha vs Beta)

# Create grid around MLE
alpha_grid <- seq(mle[1] - 1, mle[1] + 1, length.out = round(n/4))
beta_grid <- seq(mle[2] - 1, mle[2] + 1, length.out = round(n/4))
alpha_grid <- alpha_grid[alpha_grid > 0]
beta_grid <- beta_grid[beta_grid > 0]

# Compute curvature measures
determinant_surface <- matrix(NA, nrow = length(alpha_grid),
                               ncol = length(beta_grid))
trace_surface <- matrix(NA, nrow = length(alpha_grid),
                         ncol = length(beta_grid))

for (i in seq_along(alpha_grid)) {
  for (j in seq_along(beta_grid)) {
    H <- hskkw(c(alpha_grid[i], beta_grid[j], mle[3], mle[4]), data)
    determinant_surface[i, j] <- det(H)
    trace_surface[i, j] <- sum(diag(H))
  }
}

# Plot

contour(alpha_grid, beta_grid, determinant_surface,
        xlab = expression(alpha), ylab = expression(beta),
        main = "Hessian Determinant", las = 1,
        col = "#2E4057", lwd = 1.5, nlevels = 15)
points(mle[1], mle[2], pch = 19, col = "#8B0000", cex = 1.5)
points(true_params[1], true_params[2], pch = 17, col = "#006400", cex = 1.5)
grid(col = "gray90")


contour(alpha_grid, beta_grid, trace_surface,
        xlab = expression(alpha), ylab = expression(beta),
        main = "Hessian Trace", las = 1,
        col = "#2E4057", lwd = 1.5, nlevels = 15)
points(mle[1], mle[2], pch = 19, col = "#8B0000", cex = 1.5)
points(true_params[1], true_params[2], pch = 17, col = "#006400", cex = 1.5)
grid(col = "gray90")


## Example 6: Confidence Ellipse (Alpha vs Beta)

# Extract 2x2 submatrix for alpha and beta
vcov_2d <- vcov_matrix[1:2, 1:2]

# Create confidence ellipse
theta <- seq(0, 2 * pi, length.out = round(n/2))
chi2_val <- qchisq(0.95, df = 2)

eig_decomp <- eigen(vcov_2d)
ellipse <- matrix(NA, nrow = round(n/2), ncol = 2)
for (i in 1:round(n/2)) {
  v <- c(cos(theta[i]), sin(theta[i]))
  ellipse[i, ] <- mle[1:2] + sqrt(chi2_val) *
    (eig_decomp$vectors %*% diag(sqrt(eig_decomp$values)) %*% v)
}

# Marginal confidence intervals
se_2d <- sqrt(diag(vcov_2d))
ci_alpha <- mle[1] + c(-1, 1) * 1.96 * se_2d[1]
ci_beta <- mle[2] + c(-1, 1) * 1.96 * se_2d[2]

# Plot
plot(ellipse[, 1], ellipse[, 2], type = "l", lwd = 2, col = "#2E4057",
     xlab = expression(alpha), ylab = expression(beta),
     main = "95% Confidence Ellipse (Alpha vs Beta)", las = 1)

# Add marginal CIs
abline(v = ci_alpha, col = "#808080", lty = 3, lwd = 1.5)
abline(h = ci_beta, col = "#808080", lty = 3, lwd = 1.5)

points(mle[1], mle[2], pch = 19, col = "#8B0000", cex = 1.5)
points(true_params[1], true_params[2], pch = 17, col = "#006400", cex = 1.5)

legend("topright",
       legend = c("MLE", "True", "95% CR", "Marginal 95% CI"),
       col = c("#8B0000", "#006400", "#2E4057", "#808080"),
       pch = c(19, 17, NA, NA),
       lty = c(NA, NA, 1, 3),
       lwd = c(NA, NA, 2, 1.5),
       bty = "n")
grid(col = "gray90")



## Example 7: Confidence Ellipse (Delta vs Lambda)

# Extract 2x2 submatrix for delta and lambda
vcov_2d_dl <- vcov_matrix[3:4, 3:4]

# Create confidence ellipse
eig_decomp_dl <- eigen(vcov_2d_dl)
ellipse_dl <- matrix(NA, nrow = round(n/2), ncol = 2)
for (i in 1:round(n/2)) {
  v <- c(cos(theta[i]), sin(theta[i]))
  ellipse_dl[i, ] <- mle[3:4] + sqrt(chi2_val) *
    (eig_decomp_dl$vectors %*% diag(sqrt(eig_decomp_dl$values)) %*% v)
}

# Marginal confidence intervals
se_2d_dl <- sqrt(diag(vcov_2d_dl))
ci_delta <- mle[3] + c(-1, 1) * 1.96 * se_2d_dl[1]
ci_lambda <- mle[4] + c(-1, 1) * 1.96 * se_2d_dl[2]

# Plot
plot(ellipse_dl[, 1], ellipse_dl[, 2], type = "l", lwd = 2, col = "#2E4057",
     xlab = expression(delta), ylab = expression(lambda),
     main = "95% Confidence Ellipse (Delta vs Lambda)", las = 1)

# Add marginal CIs
abline(v = ci_delta, col = "#808080", lty = 3, lwd = 1.5)
abline(h = ci_lambda, col = "#808080", lty = 3, lwd = 1.5)

points(mle[3], mle[4], pch = 19, col = "#8B0000", cex = 1.5)
points(true_params[3], true_params[4], pch = 17, col = "#006400", cex = 1.5)

legend("topright",
       legend = c("MLE", "True", "95% CR", "Marginal 95% CI"),
       col = c("#8B0000", "#006400", "#2E4057", "#808080"),
       pch = c(19, 17, NA, NA),
       lty = c(NA, NA, 1, 3),
       lwd = c(NA, NA, 2, 1.5),
       bty = "n")
grid(col = "gray90")


# }
```
