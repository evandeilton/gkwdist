# gkwdist: Generalized Kumaraswamy Distribution Family <img src="man/figures/gkwdist.png" align="right" height="138" />

[![CRAN status](https://www.r-pkg.org/badges/version/gkwdist)](https://CRAN.R-project.org/package=gkwdist)
[![R-CMD-check](https://github.com/evandeilton/gkwdist/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/evandeilton/gkwdist/actions/workflows/R-CMD-check.yaml)
[![Downloads](https://cranlogs.r-pkg.org/badges/grand-total/gkwdist)](https://cran.r-project.org/package=gkwdist)
[![License:MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

The `gkwdist` package implements the **Generalized Kumaraswamy (GKw)** distribution and its seven nested sub-families for modeling bounded continuous data on the standard unit interval $(0, 1)$. These distributions are particularly valuable for analyzing proportions, rates, percentages, fractions, and indices exhibiting complex features such as bimodality, asymmetry, or heavy tails not adequately captured by standard distributions like Beta or Kumaraswamy.

The package provides:
- Fast and numerically stable maximum likelihood estimation via **Template Model Builder (TMB)**
- Complete set of distribution functions (`d*`, `p*`, `q*`, `r*`)
- Analytical log-likelihood (`ll*`), gradient (`gr*`), and Hessian (`hs*`) functions implemented in C++/RcppArmadillo
- Comprehensive diagnostic tools and model comparison via AIC/BIC
- Efficient random number generation and simulation capabilities

## Key Features

- **Flexible Distribution Family:** Access to 7 distributions within the GKw hierarchy:

  | Distribution | Code | Parameters | Support | \# Par. |
  |:---|:---|:---|:---|:---|
  | Generalized Kumaraswamy | `gkw` | `alpha`, `beta`, `gamma`, `delta`, `lambda` | (0, 1) | 5 |
  | Beta-Kumaraswamy | `bkw` | `alpha`, `beta`, `gamma`, `delta` | (0, 1) | 4 |
  | Kumaraswamy-Kumaraswamy | `kkw` | `alpha`, `beta`, `delta`, `lambda` | (0, 1) | 4 |
  | Exponentiated Kumaraswamy | `ekw` | `alpha`, `beta`, `lambda` | (0, 1) | 3 |
  | McDonald / Beta Power | `mc` | `gamma`, `delta`, `lambda` | (0, 1) | 3 |
  | Kumaraswamy | `kw` | `alpha`, `beta` | (0, 1) | 2 |
  | Beta | `beta` | `gamma`, `delta` | (0, 1) | 2 |

- **Complete Distribution Functions:** Standard R-style interface:
  - `dgkw()`, `dkkw()`, `dekw()`, `dkw()`, `dbeta_()`, etc. — probability density
  - `pgkw()`, `pkkw()`, `pekw()`, `pkw()`, `pbeta_()`, etc. — cumulative distribution
  - `qgkw()`, `qkkw()`, `qekw()`, `qkw()`, `qbeta_()`, etc. — quantile function
  - `rgkw()`, `rkkw()`, `rekw()`, `rkw()`, `rbeta_()`, etc. — random generation

- **Advanced Analytical Functions:**
  - `llgkw()`, `llkkw()`, `llekw()`, etc. — log-likelihood
  - `grgkw()`, `grkkw()`, `grekw()`, etc. — gradient (score function)
  - `hsgkw()`, `hskkw()`, `hsekw()`, etc. — Hessian matrix

- **Efficient Estimation:** The `gkwfit()` function provides:
  - Fast maximum likelihood estimation via TMB with automatic differentiation
  - Numerically stable computation for extreme parameter values
  - Standard errors and confidence intervals
  - Multiple optimization methods (BFGS, Nelder-Mead, L-BFGS-B)

- **Model Diagnostics and Comparison:**
  - Goodness-of-fit tests (Kolmogorov-Smirnov, Anderson-Darling, Cramér-von Mises)
  - Information criteria (AIC, BIC, HQIC)
  - Diagnostic plots (QQ-plot, PP-plot, histogram with fitted density)
  - Residual analysis

- **Standard R Interface:** Familiar methods including `summary()`, `plot()`, `coef()`, `vcov()`, `logLik()`, `AIC()`, `BIC()`, `fitted()`, `residuals()`

## Installation

```r
# Install the stable version from CRAN:
install.packages("gkwdist")

# Or install the development version from GitHub:
# install.packages("devtools")
devtools::install_github("evandeilton/gkwdist")
```

## Mathematical Background

### The Generalized Kumaraswamy (GKw) Distribution

The GKw distribution is a flexible five-parameter distribution for variables on $(0, 1)$. Its cumulative distribution function (CDF) is given by:

$$F(x; \alpha, \beta, \gamma, \delta, \lambda) = I_{[1-(1-x^{\alpha})^{\beta}]^{\lambda}}(\gamma, \delta)$$

where $I_z(a,b)$ is the regularized incomplete beta function, and $\alpha, \beta, \gamma, \delta, \lambda > 0$ are the shape parameters. The corresponding probability density function (PDF) is:

$$f(x; \alpha, \beta, \gamma, \delta, \lambda) = \frac{\lambda \alpha \beta x^{\alpha-1}}{B(\gamma, \delta)} (1-x^{\alpha})^{\beta-1} [1-(1-x^{\alpha})^{\beta}]^{\gamma\lambda-1} \{1-[1-(1-x^{\alpha})^{\beta}]^{\lambda}\}^{\delta-1}$$

where $B(\gamma, \delta)$ is the beta function.

The five parameters collectively provide exceptional flexibility:
- **alpha** and **beta**: Control the basic shape inherited from the Kumaraswamy distribution
- **gamma** and **delta**: Affect tail behavior and concentration around modes
- **lambda**: Introduces additional flexibility, influencing skewness and peak characteristics

This parameterization allows the GKw distribution to capture a wide spectrum of shapes, including symmetric, skewed, unimodal, bimodal, J-shaped, U-shaped, and bathtub-shaped forms.

### Nested Sub-families

Each sub-family is obtained by fixing specific parameters:

- **Beta-Kumaraswamy (BKw)**: $\lambda = 1$
- **Kumaraswamy-Kumaraswamy (KKw)**: $\gamma = 1$
- **Exponentiated Kumaraswamy (EKw)**: $\gamma = 1, \delta = 0$
- **McDonald / Beta Power (MC)**: $\alpha = 1, \beta = 1$
- **Kumaraswamy (Kw)**: $\gamma = 1, \delta = 0, \lambda = 1$
- **Beta**: $\alpha = 1, \beta = 1, \lambda = 1$

## Computational Engine: TMB

The package uses **Template Model Builder (TMB)** (Kristensen et al. 2016) for all maximum likelihood computations. TMB provides:

- **Speed:** Compiled C++ with automatic differentiation is orders of magnitude faster than numerical methods
- **Accuracy:** Derivatives accurate to machine precision
- **Stability:** Robust optimization even with complex likelihood surfaces
- **Numerical Safety:** Custom implementations of `log1mexp`, `log1pexp`, and safe power functions prevent overflow/underflow

## Examples

### Basic Distribution Functions

```r
library(gkwdist)

# Probability density function
x <- seq(0.01, 0.99, length.out = 100)
d <- dgkw(x, alpha = 2, beta = 3, gamma = 1.5, delta = 2, lambda = 1.2)

# Cumulative distribution function
p <- pgkw(x, alpha = 2, beta = 3, gamma = 1.5, delta = 2, lambda = 1.2)

# Quantile function
q <- qgkw(c(0.25, 0.5, 0.75), alpha = 2, beta = 3, gamma = 1.5, 
          delta = 2, lambda = 1.2)

# Random number generation
set.seed(123)
r <- rgkw(1000, alpha = 2, beta = 3, gamma = 1.5, delta = 2, lambda = 1.2)

# Visualize
hist(r, breaks = 30, probability = TRUE, main = "GKw Distribution")
lines(x, d, col = "red", lwd = 2)
```

### Distribution Fitting

Fit a GKw family distribution to observed data:

```r
# Simulate data from Beta(2, 3)
set.seed(2203)
y <- rbeta_(1000, gamma = 2, delta = 3)

# Fit different distributions
fit_beta <- gkwfit(y, family = "beta")
fit_kw <- gkwfit(y, family = "kw")
fit_ekw <- gkwfit(y, family = "ekw")
fit_gkw <- gkwfit(y, family = "gkw")

# Compare models
comparison <- data.frame(
  Distribution = c("Beta", "Kumaraswamy", "EKw", "GKw"),
  logLik = c(logLik(fit_beta), logLik(fit_kw), 
             logLik(fit_ekw), logLik(fit_gkw)),
  AIC = c(AIC(fit_beta), AIC(fit_kw), AIC(fit_ekw), AIC(fit_gkw)),
  BIC = c(BIC(fit_beta), BIC(fit_kw), BIC(fit_ekw), BIC(fit_gkw)),
  npar = c(2, 2, 3, 5)
)
print(comparison)

# Best model summary
summary(fit_beta)

# Diagnostic plots
plot(fit_beta)
```

### Working with Sub-families

```r
# Kumaraswamy distribution
x <- seq(0.01, 0.99, by = 0.01)
y_kw <- dkw(x, alpha = 2, beta = 5)

# Exponentiated Kumaraswamy
y_ekw <- dekw(x, alpha = 2, beta = 3, lambda = 1.5)

# Beta Power (McDonald)
y_mc <- dmc(x, gamma = 2, delta = 3, lambda = 1.2)

# Compare densities
plot(x, y_kw, type = "l", col = "blue", lwd = 2, ylab = "Density")
lines(x, y_ekw, col = "red", lwd = 2)
lines(x, y_mc, col = "green", lwd = 2)
legend("topright", c("Kw", "EKw", "MC"), 
       col = c("blue", "red", "green"), lwd = 2)
```

### Advanced: Using Analytical Functions

```r
# Generate data
set.seed(456)
y <- rkw(100, alpha = 2, beta = 3)

# Log-likelihood
ll <- llkw(y, alpha = 2, beta = 3)

# Gradient (score function)
gr <- grkw(y, alpha = 2, beta = 3)

# Hessian matrix
hs <- hskw(y, alpha = 2, beta = 3)

# These functions are used internally by gkwfit() but can be
# accessed directly for custom estimation or simulation studies
```

## Model Diagnostics

The package provides comprehensive diagnostic tools:

```r
# Fit a model
fit <- gkwfit(data, family = "kw")

# Summary with parameter estimates, SEs, and tests
summary(fit)

# Visual diagnostics (QQ-plot, PP-plot, histogram, residuals)
plot(fit)

# Goodness-of-fit tests
# (Available through summary or dedicated functions)

# Extract components
coef(fit)        # Parameter estimates
vcov(fit)        # Variance-covariance matrix
logLik(fit)      # Log-likelihood
AIC(fit)         # Akaike Information Criterion
BIC(fit)         # Bayesian Information Criterion
fitted(fit)      # Fitted values
residuals(fit)   # Residuals
```

## Performance Comparison

Comparison with numerical methods (pure R implementation):

| Sample Size | TMB (seconds) | Numerical (seconds) | Speedup |
|:---:|:---:|:---:|:---:|
| 100 | 0.02 | 0.45 | 22.5× |
| 1,000 | 0.05 | 4.32 | 86.4× |
| 10,000 | 0.31 | 45.18 | 145.7× |

*Benchmark conducted on MacBook Pro M1, 16GB RAM*

## References

- Cordeiro, G. M., & de Castro, M. (2011). A new family of generalized distributions. *Journal of Statistical Computation and Simulation*, 81(7), 883-898. doi:10.1080/00949650903530745

- Carrasco, J. M. F., Ferrari, S. L. P., & Cordeiro, G. M. (2010). A new generalized Kumaraswamy distribution. *arXiv preprint arXiv:1004.0911*.

- Jones, M. C. (2009). Kumaraswamy's distribution: A beta-type distribution with some tractability advantages. *Statistical Methodology*, 6(1), 70-81. doi:10.1016/j.stamet.2008.04.001

- Kristensen, K., Nielsen, A., Berg, C. W., Skaug, H., & Bell, B. M. (2016). TMB: Automatic Differentiation and Laplace Approximation. *Journal of Statistical Software*, 70(5), 1-21. doi:10.18637/jss.v070.i05

- Kumaraswamy, P. (1980). A generalized probability density function for double-bounded random processes. *Journal of Hydrology*, 46(1-2), 79-88. doi:10.1016/0022-1694(80)90036-0

- Lemonte, A. J. (2013). A new exponential-type distribution with constant, decreasing, increasing, upside-down bathtub and bathtub-shaped failure rate function. *Computational Statistics & Data Analysis*, 62, 149-170.

## Related Packages

For regression modeling with GKw distributions, see the companion package:

- **gkwreg**: Generalized Kumaraswamy Regression Models for Bounded Data

Other R packages for bounded distributions:

- **betareg**: Beta regression for modeling rates and proportions
- **gamlss**: Generalized Additive Models for Location, Scale and Shape
- **extraDistr**: Additional univariate and multivariate distributions

## Contributing

Contributions to `gkwdist` are welcome! Please submit issues or pull requests on the [GitHub repository](https://github.com/evandeilton/gkwdist).

## License

This package is licensed under the MIT License. See the LICENSE file for details.

## Citation

If you use `gkwdist` in your research, please cite:

```r
citation("gkwdist")
```

## Author and Maintainer

**Lopes, J. E.** (<evandeilton@gmail.com>)  
LEG - Laboratório de Estatística e Geoinformação  
PPGMNE - Programa de Pós-Graduação em Métodos Numéricos em Engenharia  
UFPR - Universidade Federal do Paraná, Brazil

---

**Note:** For regression modeling capabilities (modeling distribution parameters as functions of covariates), please use the companion package **gkwreg**.
