# Gradient of the Negative Log-Likelihood for the EKw Distribution

Computes the gradient vector (vector of first partial derivatives) of
the negative log-likelihood function for the Exponentiated Kumaraswamy
(EKw) distribution with parameters `alpha` (\\\alpha\\), `beta`
(\\\beta\\), and `lambda` (\\\lambda\\). This distribution is the
special case of the Generalized Kumaraswamy (GKw) distribution where
\\\gamma = 1\\ and \\\delta = 0\\. The gradient is useful for
optimization.

## Usage

``` r
grekw(par, data)
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

Returns a numeric vector of length 3 containing the partial derivatives
of the negative log-likelihood function \\-\ell(\theta \| \mathbf{x})\\
with respect to each parameter: \\(-\partial \ell/\partial \alpha,
-\partial \ell/\partial \beta, -\partial \ell/\partial \lambda)\\.
Returns a vector of `NaN` if any parameter values are invalid according
to their constraints, or if any value in `data` is not in the interval
(0, 1).

## Details

The components of the gradient vector of the negative log-likelihood
(\\-\nabla \ell(\theta \| \mathbf{x})\\) for the EKw (\\\gamma=1,
\delta=0\\) model are:

\$\$ -\frac{\partial \ell}{\partial \alpha} = -\frac{n}{\alpha} -
\sum\_{i=1}^{n}\ln(x_i) + \sum\_{i=1}^{n}\left\[x_i^{\alpha} \ln(x_i)
\left(\frac{\beta-1}{v_i} - \frac{(\lambda-1) \beta
v_i^{\beta-1}}{w_i}\right)\right\] \$\$ \$\$ -\frac{\partial
\ell}{\partial \beta} = -\frac{n}{\beta} - \sum\_{i=1}^{n}\ln(v_i) +
\sum\_{i=1}^{n}\left\[\frac{(\lambda-1) v_i^{\beta}
\ln(v_i)}{w_i}\right\] \$\$ \$\$ -\frac{\partial \ell}{\partial \lambda}
= -\frac{n}{\lambda} - \sum\_{i=1}^{n}\ln(w_i) \$\$

where:

- \\v_i = 1 - x_i^{\alpha}\\

- \\w_i = 1 - v_i^{\beta} = 1 - (1-x_i^{\alpha})^{\beta}\\

These formulas represent the derivatives of \\-\ell(\theta)\\,
consistent with minimizing the negative log-likelihood. They correspond
to the relevant components of the general GKw gradient
([`grgkw`](https://evandeilton.github.io/gkwdist/reference/grgkw.md))
evaluated at \\\gamma=1, \delta=0\\.

## References

Nadarajah, S., Cordeiro, G. M., & Ortega, E. M. (2012). The
exponentiated Kumaraswamy distribution. *Journal of the Franklin
Institute*, *349*(3),

Cordeiro, G. M., & de Castro, M. (2011). A new family of generalized
distributions. *Journal of Statistical Computation and Simulation*,

Kumaraswamy, P. (1980). A generalized probability density function for
double-bounded random processes. *Journal of Hydrology*, *46*(1-2),
79-88.

(Note: Specific gradient formulas might be derived or sourced from
additional references).

## See also

[`grgkw`](https://evandeilton.github.io/gkwdist/reference/grgkw.md)
(parent distribution gradient),
[`llekw`](https://evandeilton.github.io/gkwdist/reference/llekw.md)
(negative log-likelihood for EKw), `hsekw` (Hessian for EKw, if
available),
[`dekw`](https://evandeilton.github.io/gkwdist/reference/dekw.md)
(density for EKw), [`optim`](https://rdrr.io/r/stats/optim.html),
[`grad`](https://rdrr.io/pkg/numDeriv/man/grad.html) (for numerical
gradient comparison).

## Author

Lopes, J. E.

## Examples

``` r
# \donttest{
## Example 1: Basic Gradient Evaluation

# Generate sample data
set.seed(123)
n <- 1000
true_params <- c(alpha = 2.5, beta = 3.5, lambda = 2.0)
data <- rekw(n, alpha = true_params[1], beta = true_params[2],
             lambda = true_params[3])

# Evaluate gradient at true parameters
grad_true <- grekw(par = true_params, data = data)
cat("Gradient at true parameters:\n")
#> Gradient at true parameters:
print(grad_true)
#> [1]  6.172699 -4.918839  3.176895
cat("Norm:", sqrt(sum(grad_true^2)), "\n")
#> Norm: 8.508223 

# Evaluate at different parameter values
test_params <- rbind(
  c(2.0, 3.0, 1.5),
  c(2.5, 3.5, 2.0),
  c(3.0, 4.0, 2.5)
)

grad_norms <- apply(test_params, 1, function(p) {
  g <- grekw(p, data)
  sqrt(sum(g^2))
})

results <- data.frame(
  Alpha = test_params[, 1],
  Beta = test_params[, 2],
  Lambda = test_params[, 3],
  Grad_Norm = grad_norms
)
print(results, digits = 4)
#>   Alpha Beta Lambda Grad_Norm
#> 1   2.0  3.0    1.5   473.068
#> 2   2.5  3.5    2.0     8.508
#> 3   3.0  4.0    2.5   429.054


## Example 2: Gradient in Optimization

# Optimization with analytical gradient
fit_with_grad <- optim(
  par = c(2, 3, 1.5),
  fn = llekw,
  gr = grekw,
  data = data,
  method = "BFGS",
  hessian = TRUE,
  control = list(trace = 0)
)

# Optimization without gradient
fit_no_grad <- optim(
  par = c(2, 3, 1.5),
  fn = llekw,
  data = data,
  method = "BFGS",
  hessian = TRUE,
  control = list(trace = 0)
)

comparison <- data.frame(
  Method = c("With Gradient", "Without Gradient"),
  Alpha = c(fit_with_grad$par[1], fit_no_grad$par[1]),
  Beta = c(fit_with_grad$par[2], fit_no_grad$par[2]),
  Lambda = c(fit_with_grad$par[3], fit_no_grad$par[3]),
  NegLogLik = c(fit_with_grad$value, fit_no_grad$value),
  Iterations = c(fit_with_grad$counts[1], fit_no_grad$counts[1])
)
print(comparison, digits = 4, row.names = FALSE)
#>            Method Alpha  Beta Lambda NegLogLik Iterations
#>     With Gradient 2.663 3.653  1.843    -491.3         59
#>  Without Gradient 2.663 3.653  1.843    -491.3         59


## Example 3: Verifying Gradient at MLE

mle <- fit_with_grad$par
names(mle) <- c("alpha", "beta", "lambda")

# At MLE, gradient should be approximately zero
gradient_at_mle <- grekw(par = mle, data = data)
cat("\nGradient at MLE:\n")
#> 
#> Gradient at MLE:
print(gradient_at_mle)
#> [1] 2.067846e-05 8.446126e-06 1.901243e-05
cat("Max absolute component:", max(abs(gradient_at_mle)), "\n")
#> Max absolute component: 2.067846e-05 
cat("Gradient norm:", sqrt(sum(gradient_at_mle^2)), "\n")
#> Gradient norm: 2.933271e-05 


## Example 4: Numerical vs Analytical Gradient

# Manual finite difference gradient
numerical_gradient <- function(f, x, data, h = 1e-7) {
  grad <- numeric(length(x))
  for (i in seq_along(x)) {
    x_plus <- x_minus <- x
    x_plus[i] <- x[i] + h
    x_minus[i] <- x[i] - h
    grad[i] <- (f(x_plus, data) - f(x_minus, data)) / (2 * h)
  }
  return(grad)
}

# Compare at MLE
grad_analytical <- grekw(par = mle, data = data)
grad_numerical <- numerical_gradient(llekw, mle, data)

comparison_grad <- data.frame(
  Parameter = c("alpha", "beta", "lambda"),
  Analytical = grad_analytical,
  Numerical = grad_numerical,
  Abs_Diff = abs(grad_analytical - grad_numerical),
  Rel_Error = abs(grad_analytical - grad_numerical) /
              (abs(grad_analytical) + 1e-10)
)
print(comparison_grad, digits = 8)
#>   Parameter    Analytical     Numerical      Abs_Diff  Rel_Error
#> 1     alpha 2.0678456e-05 2.6432190e-05 5.7537336e-06 0.27824639
#> 2      beta 8.4461258e-06 1.6484591e-05 8.0384657e-06 0.95172281
#> 3    lambda 1.9012426e-05 2.4726887e-05 5.7144614e-06 0.30056298


## Example 5: Score Test Statistic

# Score test for H0: theta = theta0
theta0 <- c(2.2, 3.2, 1.8)
score_theta0 <- -grekw(par = theta0, data = data)

# Fisher information at theta0
fisher_info <- hsekw(par = theta0, data = data)

# Score test statistic
score_stat <- t(score_theta0) %*% solve(fisher_info) %*% score_theta0
p_value <- pchisq(score_stat, df = 3, lower.tail = FALSE)

cat("\nScore Test:\n")
#> 
#> Score Test:
cat("H0: alpha=2.2, beta=3.2, lambda=1.8\n")
#> H0: alpha=2.2, beta=3.2, lambda=1.8
cat("Test statistic:", score_stat, "\n")
#> Test statistic: 56.50173 
cat("P-value:", format.pval(p_value, digits = 4), "\n")
#> P-value: 3.283e-12 


## Example 6: Confidence Ellipse (Alpha vs Beta)

# Observed information
obs_info <- hsekw(par = mle, data = data)
vcov_full <- solve(obs_info)
vcov_2d <- vcov_full[1:2, 1:2]

# Create confidence ellipse
theta <- seq(0, 2 * pi, length.out = 100)
chi2_val <- qchisq(0.95, df = 2)

eig_decomp <- eigen(vcov_2d)
ellipse <- matrix(NA, nrow = 100, ncol = 2)
for (i in 1:100) {
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
     main = "95% Confidence Region (Alpha vs Beta)", las = 1)

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



## Example 7: Confidence Ellipse (Alpha vs Lambda)

# Extract 2x2 submatrix for alpha and lambda
vcov_2d_al <- vcov_full[c(1, 3), c(1, 3)]

# Create confidence ellipse
eig_decomp_al <- eigen(vcov_2d_al)
ellipse_al <- matrix(NA, nrow = 100, ncol = 2)
for (i in 1:100) {
  v <- c(cos(theta[i]), sin(theta[i]))
  ellipse_al[i, ] <- mle[c(1, 3)] + sqrt(chi2_val) *
    (eig_decomp_al$vectors %*% diag(sqrt(eig_decomp_al$values)) %*% v)
}

# Marginal confidence intervals
se_2d_al <- sqrt(diag(vcov_2d_al))
ci_alpha_2 <- mle[1] + c(-1, 1) * 1.96 * se_2d_al[1]
ci_lambda <- mle[3] + c(-1, 1) * 1.96 * se_2d_al[2]

# Plot

plot(ellipse_al[, 1], ellipse_al[, 2], type = "l", lwd = 2, col = "#2E4057",
     xlab = expression(alpha), ylab = expression(lambda),
     main = "95% Confidence Region (Alpha vs Lambda)", las = 1)

# Add marginal CIs
abline(v = ci_alpha_2, col = "#808080", lty = 3, lwd = 1.5)
abline(h = ci_lambda, col = "#808080", lty = 3, lwd = 1.5)

points(mle[1], mle[3], pch = 19, col = "#8B0000", cex = 1.5)
points(true_params[1], true_params[3], pch = 17, col = "#006400", cex = 1.5)

legend("topright",
       legend = c("MLE", "True", "95% CR", "Marginal 95% CI"),
       col = c("#8B0000", "#006400", "#2E4057", "#808080"),
       pch = c(19, 17, NA, NA),
       lty = c(NA, NA, 1, 3),
       lwd = c(NA, NA, 2, 1.5),
       bty = "n")
grid(col = "gray90")



## Example 8: Confidence Ellipse (Beta vs Lambda)

# Extract 2x2 submatrix for beta and lambda
vcov_2d_bl <- vcov_full[2:3, 2:3]

# Create confidence ellipse
eig_decomp_bl <- eigen(vcov_2d_bl)
ellipse_bl <- matrix(NA, nrow = 100, ncol = 2)
for (i in 1:100) {
  v <- c(cos(theta[i]), sin(theta[i]))
  ellipse_bl[i, ] <- mle[2:3] + sqrt(chi2_val) *
    (eig_decomp_bl$vectors %*% diag(sqrt(eig_decomp_bl$values)) %*% v)
}

# Marginal confidence intervals
se_2d_bl <- sqrt(diag(vcov_2d_bl))
ci_beta_2 <- mle[2] + c(-1, 1) * 1.96 * se_2d_bl[1]
ci_lambda_2 <- mle[3] + c(-1, 1) * 1.96 * se_2d_bl[2]

# Plot

plot(ellipse_bl[, 1], ellipse_bl[, 2], type = "l", lwd = 2, col = "#2E4057",
     xlab = expression(beta), ylab = expression(lambda),
     main = "95% Confidence Region (Beta vs Lambda)", las = 1)

# Add marginal CIs
abline(v = ci_beta_2, col = "#808080", lty = 3, lwd = 1.5)
abline(h = ci_lambda_2, col = "#808080", lty = 3, lwd = 1.5)

points(mle[2], mle[3], pch = 19, col = "#8B0000", cex = 1.5)
points(true_params[2], true_params[3], pch = 17, col = "#006400", cex = 1.5)

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
