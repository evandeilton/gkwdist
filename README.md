# gkwdist: Generalized Kumaraswamy Distribution Family <img src="man/figures/gkwdist.png" align="right" height="138" />

[![CRAN status](https://www.r-pkg.org/badges/version/gkwdist)](https://CRAN.R-project.org/package=gkwdist)
[![R-CMD-check](https://github.com/evandeilton/gkwdist/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/evandeilton/gkwdist/actions/workflows/R-CMD-check.yaml)
[![Downloads](https://cranlogs.r-pkg.org/badges/grand-total/gkwdist)](https://cran.r-project.org/package=gkwdist)
[![License:MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

`gkwdist` implements the **Generalized Kumaraswamy (GKw)** distribution family and its seven nested sub-models for bounded continuous data on $(0,1)$. The package provides complete distribution functions and high-performance analytical derivatives implemented in C++/RcppArmadillo.

**Key Features:**
- Seven flexible distributions for proportions, rates, and bounded data
- Standard R distribution API: `d*`, `p*`, `q*`, `r*`
- Analytical log-likelihood, gradient, and Hessian functions (C++)
- 10-50× faster than numerical alternatives
- Numerically stable implementations

---

## Installation

```r
# install.packages("devtools")
devtools::install_github("evandeilton/gkwdist")
```

---

## The Distribution Family

### Function Table

| Distribution | Code | Parameters | Functions |
|:---|:---:|:---|:---|
| **Generalized Kumaraswamy** | `gkw` | $\alpha, \beta, \gamma, \delta, \lambda$ | `dgkw`, `pgkw`, `qgkw`, `rgkw`, `llgkw`, `grgkw`, `hsgkw` |
| **Beta-Kumaraswamy** | `bkw` | $\alpha, \beta, \gamma, \delta$ | `dbkw`, `pbkw`, `qbkw`, `rbkw`, `llbkw`, `grbkw`, `hsbkw` |
| **Kumaraswamy-Kumaraswamy** | `kkw` | $\alpha, \beta, \delta, \lambda$ | `dkkw`, `pkkw`, `qkkw`, `rkkw`, `llkkw`, `grkkw`, `hskkw` |
| **Exponentiated Kumaraswamy** | `ekw` | $\alpha, \beta, \lambda$ | `dekw`, `pekw`, `qekw`, `rekw`, `llekw`, `grekw`, `hsekw` |
| **McDonald (Beta Power)** | `mc` | $\gamma, \delta, \lambda$ | `dmc`, `pmc`, `qmc`, `rmc`, `llmc`, `grmc`, `hsmc` |
| **Kumaraswamy** | `kw` | $\alpha, \beta$ | `dkw`, `pkw`, `qkw`, `rkw`, `llkw`, `grkw`, `hskw` |
| **Beta** | `beta_` | $\gamma, \delta$ | `dbeta_`, `pbeta_`, `qbeta_`, `rbeta_`, `llbeta_`, `grbeta_`, `hsbeta_` |

### Function Types

**Distribution Functions (R):**

- `d*()` — Probability density function (PDF)
- `p*()` — Cumulative distribution function (CDF)
- `q*()` — Quantile function (inverse CDF)
- `r*()` — Random number generation

**Analytical Functions (C++):**

- `ll*()` — Log-likelihood:

$$\ell(\boldsymbol{\theta}; \mathbf{x}) = \sum_{i=1}^n \log f(x_i; \boldsymbol{\theta})$$

- `gr*()` — Gradient (score vector):

$$\nabla_{\boldsymbol{\theta}} \ell(\boldsymbol{\theta}; \mathbf{x})$$

- `hs*()` — Hessian matrix:

$$\nabla^2_{\boldsymbol{\theta}} \ell(\boldsymbol{\theta}; \mathbf{x})$$

---

## Mathematical Specification

**Notation:** All parameters are positive; support $x \in (0, 1)$. The beta function is $B(a,b) = \Gamma(a)\Gamma(b)/\Gamma(a+b)$, and the regularized incomplete beta function is $I_z(a,b) = B_z(a,b)/B(a,b)$ where $B_z(a,b) = \int_0^z t^{a-1}(1-t)^{b-1}dt$.

### 1. Generalized Kumaraswamy (GKw)

**Parameters:** $\alpha, \beta, \gamma, \delta, \lambda > 0$

**PDF:**

$$f_{\text{GKw}}(x; \alpha, \beta, \gamma, \delta, \lambda) = \frac{\lambda \alpha \beta}{B(\gamma, \delta)} x^{\alpha-1} (1-x^\alpha)^{\beta-1} [1-(1-x^\alpha)^\beta]^{\gamma\lambda-1} \{1-[1-(1-x^\alpha)^\beta]^\lambda\}^{\delta-1}$$

**CDF:**

$$F_{\text{GKw}}(x; \alpha, \beta, \gamma, \delta, \lambda) = I_{[1-(1-x^\alpha)^\beta]^\lambda}(\gamma, \delta)$$

**Quantile:** Numerical inversion of the CDF.

---

### 2. Beta-Kumaraswamy (BKw)

**Relationship:** GKw with $\lambda = 1$

**PDF:**

$$f_{\text{BKw}}(x; \alpha, \beta, \gamma, \delta) = \frac{\alpha \beta}{B(\gamma, \delta)} x^{\alpha-1} (1-x^\alpha)^{\beta\delta-1} [1-(1-x^\alpha)^\beta]^{\gamma-1}$$

**CDF:**

$$F_{\text{BKw}}(x; \alpha, \beta, \gamma, \delta) = I_{1-(1-x^\alpha)^\beta}(\gamma, \delta)$$

**Quantile:** Numerical inversion.

---

### 3. Kumaraswamy-Kumaraswamy (KKw)

**Relationship:** GKw with $\gamma = 1$

**PDF:**

$$f_{\text{KKw}}(x; \alpha, \beta, \delta, \lambda) = \alpha \beta \delta \lambda \, x^{\alpha-1} (1-x^\alpha)^{\beta-1} [1-(1-x^\alpha)^\beta]^{\lambda-1} \{1-[1-(1-x^\alpha)^\beta]^\lambda\}^{\delta-1}$$

**CDF:**

$$F_{\text{KKw}}(x; \alpha, \beta, \delta, \lambda) = 1 - \{1-[1-(1-x^\alpha)^\beta]^\lambda\}^\delta$$

**Quantile (closed-form):**

$$Q_{\text{KKw}}(p; \alpha, \beta, \delta, \lambda) = [1 - (1 - [1-(1-p)^{1/\delta}]^{1/\lambda})^{1/\beta}]^{1/\alpha}$$

---

### 4. Exponentiated Kumaraswamy (EKw)

**Relationship:** GKw with $\gamma = \delta = 1$

**PDF:**

$$f_{\text{EKw}}(x; \alpha, \beta, \lambda) = \lambda \alpha \beta \, x^{\alpha-1} (1-x^\alpha)^{\beta-1} [1-(1-x^\alpha)^\beta]^{\lambda-1}$$

**CDF:**

$$F_{\text{EKw}}(x; \alpha, \beta, \lambda) = [1-(1-x^\alpha)^\beta]^\lambda$$

**Quantile (closed-form):**

$$Q_{\text{EKw}}(p; \alpha, \beta, \lambda) = [1-(1-p^{1/\lambda})^{1/\beta}]^{1/\alpha}$$

---

### 5. McDonald (Beta Power)

**Relationship:** GKw with $\alpha = \beta = 1$

**PDF:**

$$f_{\text{MC}}(x; \gamma, \delta, \lambda) = \frac{\lambda}{B(\gamma, \delta)} x^{\gamma\lambda-1} (1-x^\lambda)^{\delta-1}$$

**CDF:**

$$F_{\text{MC}}(x; \gamma, \delta, \lambda) = I_{x^\lambda}(\gamma, \delta)$$

**Quantile:**

$$Q_{\text{MC}}(p; \gamma, \delta, \lambda) = [I_p^{-1}(\gamma, \delta)]^{1/\lambda}$$

where $I_p^{-1}(\gamma, \delta)$ is the inverse of the regularized incomplete beta function.

---

### 6. Kumaraswamy (Kw)

**Relationship:** GKw with $\gamma = \delta = \lambda = 1$

**PDF:**

$$f_{\text{Kw}}(x; \alpha, \beta) = \alpha \beta \, x^{\alpha-1} (1-x^\alpha)^{\beta-1}$$

**CDF:**

$$F_{\text{Kw}}(x; \alpha, \beta) = 1 - (1-x^\alpha)^\beta$$

**Quantile (closed-form):**

$$Q_{\text{Kw}}(p; \alpha, \beta) = [1-(1-p)^{1/\beta}]^{1/\alpha}$$

**Moments:**

$$\mathbb{E}(X^r) = \beta B(1 + r/\alpha, \beta) = \frac{\beta \, \Gamma(1+r/\alpha) \, \Gamma(\beta)}{\Gamma(1+r/\alpha+\beta)}$$

---

### 7. Beta

**Relationship:** GKw with $\alpha = \beta = \lambda = 1$

**PDF:**

$$f_{\text{Beta}}(x; \gamma, \delta) = \frac{1}{B(\gamma, \delta)} x^{\gamma-1} (1-x)^{\delta-1}$$

**CDF:**

$$F_{\text{Beta}}(x; \gamma, \delta) = I_x(\gamma, \delta)$$

**Quantile:**

$$Q_{\text{Beta}}(p; \gamma, \delta) = I_p^{-1}(\gamma, \delta)$$

**Moments:**

$$\mathbb{E}(X^r) = \frac{B(\gamma+r, \delta)}{B(\gamma, \delta)}, \quad \mathbb{E}(X) = \frac{\gamma}{\gamma+\delta}, \quad \text{Var}(X) = \frac{\gamma\delta}{(\gamma+\delta)^2(\gamma+\delta+1)}$$

---

## Hierarchical Structure

```
                    GKw(α, β, γ, δ, λ)
                     /              \
                λ = 1                γ = 1
                   /                    \
          BKw(α, β, γ, δ)        KKw(α, β, δ, λ)
             |                          |
       α=β=1 |                   δ = 1  |
             |                          |
        MC(γ, δ, λ)              EKw(α, β, λ)
             |                          |
         λ=1 |                          |
             |                          |
        Beta(γ, δ)          γ=δ=1       |
             |                          |
             +----------- λ=1 ----------+
                          |
                    Kw(α, β)
```

---

## Usage Examples

### Example 1: Basic Distribution Functions

```r
library(gkwdist)

alpha <- 2; beta <- 3; gamma <- 1.5; delta <- 2; lambda <- 1.2
x <- seq(0.01, 0.99, length.out = 100)

dens <- dgkw(x, alpha, beta, gamma, delta, lambda)
cdf  <- pgkw(x, alpha, beta, gamma, delta, lambda)

q <- qgkw(c(0.25, 0.5, 0.75), alpha, beta, gamma, delta, lambda)
print(q)

set.seed(123)
random_sample <- rgkw(1000, alpha, beta, gamma, delta, lambda)

par(mfrow = c(1, 2))
plot(x, dens, type = "l", lwd = 2, col = "blue",
     main = "GKw PDF", xlab = "x", ylab = "Density"); grid()
plot(x, cdf, type = "l", lwd = 2, col = "red",
     main = "GKw CDF", xlab = "x", ylab = "F(x)"); grid()
```

### Example 2: Comparing Sub-families

```r
library(gkwdist)

x <- seq(0.001, 0.999, length.out = 500)

d_gkw  <- dgkw(x, 2, 3, 1.5, 2, 1.2)
d_bkw  <- dbkw(x, 2, 3, 1.5, 2)
d_kkw  <- dkkw(x, 2, 3, 2, 1.2)
d_ekw  <- dekw(x, 2, 3, 1.5)
d_mc   <- dmc(x, 2, 3, 1.2)
d_kw   <- dkw(x, 2, 5)
d_beta <- dbeta_(x, 2, 3)

plot(x, d_gkw, type = "l", lwd = 2, col = "black",
     ylim = c(0, max(d_gkw, d_bkw, d_kkw, d_ekw, d_mc, d_kw, d_beta)),
     main = "Distribution Family Comparison",
     xlab = "x", ylab = "Density")
lines(x, d_bkw, lwd = 2, col = "red")
lines(x, d_kkw, lwd = 2, col = "blue")
lines(x, d_ekw, lwd = 2, col = "green")
lines(x, d_mc,  lwd = 2, col = "purple")
lines(x, d_kw,  lwd = 2, col = "orange")
lines(x, d_beta, lwd = 2, col = "brown")
legend("topright",
       legend = c("GKw", "BKw", "KKw", "EKw", "MC", "Kw", "Beta"),
       col = c("black", "red", "blue", "green", "purple", "orange", "brown"),
       lwd = 2, cex = 0.85)
```

### Example 3: Maximum Likelihood Estimation

```r
library(gkwdist)

set.seed(2024)
n <- 200
true_alpha <- 2.5
true_beta  <- 3.5
data <- rkw(n, true_alpha, true_beta)

nll <- function(par, data) {
  if (any(par <= 0)) return(1e10)
  -llkw(data, alpha = par[1], beta = par[2])
}
grad <- function(par, data) {
  if (any(par <= 0)) return(rep(0, length(par)))
  -grkw(data, alpha = par[1], beta = par[2])
}

fit <- optim(par = c(1, 1), fn = nll, gr = grad,
             data = data, method = "BFGS", hessian = TRUE)

se <- sqrt(diag(solve(fit$hessian)))
ci <- cbind(
  Lower    = fit$par - 1.96 * se,
  Estimate = fit$par,
  Upper    = fit$par + 1.96 * se
)
rownames(ci) <- c("alpha", "beta")
print(ci)
```

### Example 4: Diagnostic Plot

```r
library(gkwdist)

x_grid <- seq(0.001, 0.999, length.out = 200)
fitted_dens <- dkw(x_grid, fit$par[1], fit$par[2])
true_dens   <- dkw(x_grid, true_alpha, true_beta)

hist(data, breaks = 30, probability = TRUE,
     col = "lightgray", border = "white",
     main = "Kumaraswamy Fit",
     xlab = "Data", ylab = "Density")
lines(x_grid, fitted_dens, col = "red",  lwd = 2, lty = 1)
lines(x_grid, true_dens,   col = "blue", lwd = 2, lty = 2)
legend("topright", legend = c("Data", "Fitted", "True"),
       col = c("gray", "red", "blue"),
       lwd = c(10, 2, 2), lty = c(1, 1, 2))
```

### Example 5: Model Selection with AIC/BIC

```r
library(gkwdist)

set.seed(456)
n <- 150
data <- rekw(n, alpha = 2, beta = 3, lambda = 1.5)

models <- list(
  Beta = list(
    nll = function(par) -llbeta_(data, par[1], par[2]),
    start = c(1, 1), npar = 2
  ),
  Kw = list(
    nll = function(par) -llkw(data, par[1], par[2]),
    start = c(1, 1), npar = 2
  ),
  EKw = list(
    nll = function(par) -llekw(data, par[1], par[2], par[3]),
    start = c(1, 1, 1), npar = 3
  ),
  MC = list(
    nll = function(par) -llmc(data, par[1], par[2], par[3]),
    start = c(1, 1, 1), npar = 3
  )
)

fits   <- lapply(models, function(m) optim(par = m$start, fn = m$nll, method = "BFGS"))
loglik <- sapply(fits, function(f) -f$value)
k      <- sapply(models, `[[`, "npar")

results <- data.frame(
  Model  = names(models),
  LogLik = loglik,
  nPar   = k,
  AIC    = -2 * loglik + 2 * k,
  BIC    = -2 * loglik + k * log(n)
)
print(results[order(results$AIC), ], row.names = FALSE)
cat("\nBest model (AIC):", results$Model[which.min(results$AIC)], "\n")
```

### Example 6: Using Analytical Functions

```r
library(gkwdist)

set.seed(789)
n <- 100
data <- rekw(n, alpha = 2, beta = 3, lambda = 1.5)
params <- c(2, 3, 1.5)

ll    <- llekw(data, params[1], params[2], params[3])
score <- grekw(data, params[1], params[2], params[3])
hess  <- hsekw(data, params[1], params[2], params[3])

fisher <- -hess
se <- sqrt(diag(solve(fisher)))
names(se) <- c("alpha", "beta", "lambda")

cat("Log-likelihood:", ll, "\n")
cat("\nScore vector:\n"); print(score)
cat("\nHessian matrix:\n"); print(hess)
cat("\nAsymptotic SEs:\n"); print(se)
```

### Example 7: Q-Q Plot

```r
library(gkwdist)

set.seed(101)
n <- 200
data <- rkw(n, alpha = 2, beta = 3)

fit <- optim(par = c(1, 1),
             fn = function(par) -llkw(data, par[1], par[2]),
             method = "BFGS")

p <- ppoints(n)
theoretical_q <- qkw(p, fit$par[1], fit$par[2])
empirical_q   <- sort(data)

plot(theoretical_q, empirical_q,
     xlab = "Theoretical Quantiles", ylab = "Empirical Quantiles",
     main = "Q-Q Plot: Kumaraswamy",
     pch = 19, col = rgb(0, 0, 1, 0.5))
abline(0, 1, col = "red", lwd = 2, lty = 2)
grid()
```

---

## Performance

```r
library(microbenchmark)

n <- 10000
data <- rkw(n, 2, 3)

microbenchmark(
  R_sum_log_d = sum(log(dkw(data, 2, 3))),
  Cpp_ll      = llkw(data, 2, 3),
  times = 100
)
# Typical speedup: 10-50× faster
```

---

## When to Use Each Distribution

| Data Characteristics | Recommended Distribution | Rationale |
|:---|:---|:---|
| Unimodal, symmetric | **Beta** | Parsimony; well-studied |
| Unimodal, asymmetric | **Kumaraswamy** | Closed-form CDF/quantile |
| Bimodal or U-shaped | **GKw** or **BKw** | Maximum flexibility |
| Extreme skewness | **KKw** or **EKw** | Flexible tail control |
| J-shaped (monotonic) | **Kw** or **Beta** | With appropriate parameters |
| Power transformations | **McDonald** | Explicit power parameter |
| Unknown shape | **GKw** | Test nested models |

---

## References

- **Cordeiro, G. M., & de Castro, M. (2011).** A new family of generalized distributions. *Journal of Statistical Computation and Simulation*, 81(7), 883-898. [doi:10.1080/00949650903530745](https://doi.org/10.1080/00949650903530745)

- **Carrasco, J. M. F., Ferrari, S. L. P., & Cordeiro, G. M. (2010).** A new generalized Kumaraswamy distribution. *arXiv:1004.0911*. [arxiv.org/abs/1004.0911](https://arxiv.org/abs/1004.0911)

- **Kumaraswamy, P. (1980).** A generalized probability density function for double-bounded random processes. *Journal of Hydrology*, 46(1-2), 79-88. [doi:10.1016/0022-1694(80)90036-0](https://doi.org/10.1016/0022-1694(80)90036-0)

- **Jones, M. C. (2009).** Kumaraswamy's distribution: A beta-type distribution with some tractability advantages. *Statistical Methodology*, 6(1), 70-81. [doi:10.1016/j.stamet.2008.04.001](https://doi.org/10.1016/j.stamet.2008.04.001)

- **Lemonte, A. J. (2013).** A new exponential-type distribution with constant, decreasing, increasing, upside-down bathtub and bathtub-shaped failure rate function. *Computational Statistics & Data Analysis*, 62, 149-170. [doi:10.1016/j.csda.2013.01.011](https://doi.org/10.1016/j.csda.2013.01.011)

---

## Citation

```r
citation("gkwdist")
```

```bibtex
@Manual{gkwdist2025,
  title  = {gkwdist: Generalized Kumaraswamy Distribution Family},
  author = {J. E. Lopes},
  year   = {2025},
  note   = {R package},
  url    = {https://github.com/evandeilton/gkwdist}
}
```

---

## Author

**J. E. Lopes**  
LEG - Laboratory of Statistics and Geoinformation  
PPGMNE - Graduate Program in Numerical Methods in Engineering  
Federal University of Paraná (UFPR), Brazil  
Email: evandeilton@gmail.com

---

## License

MIT License. See [LICENSE](LICENSE) file for details.

