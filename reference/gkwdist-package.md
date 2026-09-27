# Generalized Kumaraswamy Distribution Family

The gkwdist package provides a comprehensive, high-performance
implementation of the five-parameter Generalized Kumaraswamy (GKw)
distribution and its seven nested sub-families for modeling bounded
continuous data on the unit interval \\(0,1)\\. All distribution and
likelihood functions are implemented in **C++** via RcppArmadillo for
maximum computational efficiency.

## Details

### Overview

The Generalized Kumaraswamy distribution, proposed by Carrasco, Ferrari
and Cordeiro (2010), extends the Kumaraswamy distribution (Jones, 2009)
by incorporating three additional shape parameters. This generalization
provides remarkable flexibility in accommodating various density shapes
including unimodality, asymmetry, J-shapes, inverted J-shapes, U-shapes,
bathtub shapes, and heavy or light tails.

That flexibility stops short of bimodality. Across roughly 330,000
parameter vectors spanning \\(10^{-3}, 300)\\ in each of the five
parameters, no density with two interior modes was found: every shape
was monotone, unimodal, or U-shaped. A second peak arises only as a
divergence at \\x = 0\\ or \\x = 1\\ — see the boundary limits
documented in
[`dgkw`](https://evandeilton.github.io/gkwdist/reference/dgkw.md) —
never as a second interior hump. Data with two separated interior modes
calls for a mixture, not a larger member of this family.

### Mathematical Specification

The probability density function (PDF) of the GKw distribution with
parameters \\\theta = (\alpha, \beta, \gamma, \delta, \lambda)\\ (all
positive) is: \$\$f(x; \theta) = \frac{\lambda\alpha\beta
x^{\alpha-1}}{B(\gamma, \delta + 1)} (1 - x^\alpha)^{\beta-1} \[1 - (1 -
x^\alpha)^\beta\]^{\gamma\lambda-1} \\1 - \[1 - (1 -
x^\alpha)^\beta\]^\lambda\\^\delta\$\$ for \\0 \< x \< 1\\, where
\\B(a,b) = \Gamma(a)\Gamma(b)/\Gamma(a+b)\\ is the beta function.

The cumulative distribution function (CDF) is: \$\$F(x; \theta) =
I\_{\[1-(1-x^\alpha)^\beta\]^\lambda}(\gamma, \delta + 1)\$\$ where
\\I_y(a,b)\\ denotes the regularized incomplete beta function ratio.

### Distribution Family Hierarchy

The GKw distribution nests seven important sub-models:

- **Beta-Kumaraswamy (BKw)**:

  4 parameters: \\(\alpha, \beta, \gamma, \delta)\\. Set \\\lambda = 1\\
  in GKw.

- **Kumaraswamy-Kumaraswamy (KKw)**:

  4 parameters: \\(\alpha, \beta, \delta, \lambda)\\. Set \\\gamma = 1\\
  in GKw.

- **Exponentiated Kumaraswamy (EKw)**:

  3 parameters: \\(\alpha, \beta, \lambda)\\. Set \\\gamma = 1, \delta =
  0\\ in GKw. Has closed-form quantile function.

- **McDonald (Mc)**:

  3 parameters: \\(\gamma, \delta, \lambda)\\. Set \\\alpha = \beta =
  1\\ in GKw. Also known as Beta-Power distribution.

- **Kumaraswamy (Kw)**:

  2 parameters: \\(\alpha, \beta)\\. Set \\\gamma = 1\\, \\\delta = 0\\
  and \\\lambda = 1\\ in GKw. Has closed-form CDF and quantile function.

- **Beta**:

  2 parameters: \\(\gamma, \delta)\\. Set \\\alpha = \beta = \lambda =
  1\\ in GKw. Classical beta distribution with `shape1 = gamma`,
  `shape2 = delta + 1`.

- **Uniform**:

  0 parameters. Set \\\alpha = \beta = \gamma = \lambda = 1\\ and
  \\\delta = 0\\ in GKw. Note that \\\delta = 0\\, not \\\delta = 1\\,
  is the neutral value: \\\delta\\ enters the density through
  \\B(\gamma, \delta+1)\\ and \\z^{\delta}\\.

## Distribution Functions

The package provides the standard R distribution function API for all
seven distributions. Each distribution has four core functions with
prefix `d`, `p`, `q`, `r`:

**Generalized Kumaraswamy (GKw):**

- [`dgkw`](https://evandeilton.github.io/gkwdist/reference/dgkw.md):
  Density function

- [`pgkw`](https://evandeilton.github.io/gkwdist/reference/pgkw.md):
  Distribution function (CDF)

- [`qgkw`](https://evandeilton.github.io/gkwdist/reference/qgkw.md):
  Quantile function (inverse CDF)

- [`rgkw`](https://evandeilton.github.io/gkwdist/reference/rgkw.md):
  Random generation

**Beta-Kumaraswamy (BKw):**
[`dbkw`](https://evandeilton.github.io/gkwdist/reference/dbkw.md),
[`pbkw`](https://evandeilton.github.io/gkwdist/reference/pbkw.md),
[`qbkw`](https://evandeilton.github.io/gkwdist/reference/qbkw.md),
[`rbkw`](https://evandeilton.github.io/gkwdist/reference/rbkw.md)

**Kumaraswamy-Kumaraswamy (KKw):**
[`dkkw`](https://evandeilton.github.io/gkwdist/reference/dkkw.md),
[`pkkw`](https://evandeilton.github.io/gkwdist/reference/pkkw.md),
[`qkkw`](https://evandeilton.github.io/gkwdist/reference/qkkw.md),
[`rkkw`](https://evandeilton.github.io/gkwdist/reference/rkkw.md)

**Exponentiated Kumaraswamy (EKw):**
[`dekw`](https://evandeilton.github.io/gkwdist/reference/dekw.md),
[`pekw`](https://evandeilton.github.io/gkwdist/reference/pekw.md),
[`qekw`](https://evandeilton.github.io/gkwdist/reference/qekw.md),
[`rekw`](https://evandeilton.github.io/gkwdist/reference/rekw.md)

**McDonald (Mc):**
[`dmc`](https://evandeilton.github.io/gkwdist/reference/dmc.md),
[`pmc`](https://evandeilton.github.io/gkwdist/reference/pmc.md),
[`qmc`](https://evandeilton.github.io/gkwdist/reference/qmc.md),
[`rmc`](https://evandeilton.github.io/gkwdist/reference/rmc.md)

**Kumaraswamy (Kw):**
[`dkw`](https://evandeilton.github.io/gkwdist/reference/dkw.md),
[`pkw`](https://evandeilton.github.io/gkwdist/reference/pkw.md),
[`qkw`](https://evandeilton.github.io/gkwdist/reference/qkw.md),
[`rkw`](https://evandeilton.github.io/gkwdist/reference/rkw.md)

**Beta:**
[`dbeta_`](https://evandeilton.github.io/gkwdist/reference/dbeta_.md),
[`pbeta_`](https://evandeilton.github.io/gkwdist/reference/pbeta_.md),
[`qbeta_`](https://evandeilton.github.io/gkwdist/reference/qbeta_.md),
[`rbeta_`](https://evandeilton.github.io/gkwdist/reference/rbeta_.md)

All distribution functions are implemented in C++ for optimal
performance.

## Likelihood Functions

High-performance analytical log-likelihood, gradient, and Hessian
functions are provided for maximum likelihood estimation. These
functions return **negative** values to facilitate direct use with
optimization routines like
[`optim`](https://rdrr.io/r/stats/optim.html), which perform
minimization by default.

Function signature: `function(par, data)` where `par` is a numeric
vector of parameters and `data` is the observed sample.

- `ll*`:

  Negative log-likelihood: \\-\ell(\theta) = -\sum\_{i=1}^n \log f(x_i;
  \theta)\\

- `gr*`:

  Negative gradient (negative score vector): \\-\nabla\_\theta
  \ell(\theta)\\

- `hs*`:

  Negative Hessian matrix: \\-\nabla^2\_\theta \ell(\theta)\\

Available for all distributions:

- **GKw:**
  [`llgkw`](https://evandeilton.github.io/gkwdist/reference/llgkw.md),
  [`grgkw`](https://evandeilton.github.io/gkwdist/reference/grgkw.md),
  [`hsgkw`](https://evandeilton.github.io/gkwdist/reference/hsgkw.md)

- **BKw:**
  [`llbkw`](https://evandeilton.github.io/gkwdist/reference/llbkw.md),
  [`grbkw`](https://evandeilton.github.io/gkwdist/reference/grbkw.md),
  [`hsbkw`](https://evandeilton.github.io/gkwdist/reference/hsbkw.md)

- **KKw:**
  [`llkkw`](https://evandeilton.github.io/gkwdist/reference/llkkw.md),
  [`grkkw`](https://evandeilton.github.io/gkwdist/reference/grkkw.md),
  [`hskkw`](https://evandeilton.github.io/gkwdist/reference/hskkw.md)

- **EKw:**
  [`llekw`](https://evandeilton.github.io/gkwdist/reference/llekw.md),
  [`grekw`](https://evandeilton.github.io/gkwdist/reference/grekw.md),
  [`hsekw`](https://evandeilton.github.io/gkwdist/reference/hsekw.md)

- **Mc:**
  [`llmc`](https://evandeilton.github.io/gkwdist/reference/llmc.md),
  [`grmc`](https://evandeilton.github.io/gkwdist/reference/grmc.md),
  [`hsmc`](https://evandeilton.github.io/gkwdist/reference/hsmc.md)

- **Kw:**
  [`llkw`](https://evandeilton.github.io/gkwdist/reference/llkw.md),
  [`grkw`](https://evandeilton.github.io/gkwdist/reference/grkw.md),
  [`hskw`](https://evandeilton.github.io/gkwdist/reference/hskw.md)

- **Beta:**
  [`llbeta`](https://evandeilton.github.io/gkwdist/reference/llbeta.md),
  [`grbeta`](https://evandeilton.github.io/gkwdist/reference/grbeta.md),
  [`hsbeta`](https://evandeilton.github.io/gkwdist/reference/hsbeta.md)

These analytical derivatives are exact (up to floating-point precision)
and provide substantial performance gains over numerical approximations.

## Utility Functions

- [`gkwgetstartvalues`](https://evandeilton.github.io/gkwdist/reference/gkwgetstartvalues.md):
  Compute starting values for optimization using method of moments or
  quantile matching

## Applications

The GKw distribution family is particularly suitable for modeling:

- **Proportions and rates**: Bounded continuous data in \\(0,1)\\

- **Percentages**: Economic indices, market shares, conversion rates

- **Fractions**: Parts of a whole, concentration measures

- **Scores and indices**: Normalized measurements, standardized tests

- **Reliability data**: Component lifetimes on bounded domains

- **Hydrological phenomena**: Reservoir levels, soil moisture content

- **Financial ratios**: Debt-to-equity, current ratio, profit margins

- **Quality metrics**: Defect rates, efficiency scores, purity levels

- **Biostatistics**: Survival probabilities, dose-response curves

## Advantages Over Standard Distributions

Compared to the classical Beta distribution, the GKw family offers:

- Greater flexibility in density shapes (U-shaped, bathtub, J and
  reverse J)

- Better accommodation of asymmetry and tail behavior

- Closed-form CDF and quantile for some sub-models (Kw, EKw, KKw)

- Computational efficiency via C++ implementation

- Easy parameter interpretation through nested structure

- Superior performance for extreme parameter values

## Performance

All functions are implemented in C++ using RcppArmadillo, providing:

- 10-100× speedup over pure R implementations

- Linear scaling with sample size

- Optimized memory allocation

- Numerical stability for extreme parameter values

- Efficient vectorized operations

Typical benchmarks on modern hardware:

- Density evaluation: \\\>10^7\\ evaluations per second

- Log-likelihood: \\n = 10^6\\ observations in \\\<100\\ms

- Gradient computation: \\\<5\\× slower than log-likelihood

## Model Selection Workflow

Recommended strategy for choosing among distributions:

1.  **Exploratory Analysis**: Examine histograms, kernel density
    estimates, and summary statistics of your data.

2.  **Start Simple**: Fit Beta and Kumaraswamy distributions (2
    parameters). Use [`optim`](https://rdrr.io/r/stats/optim.html) with
    `method = "BFGS"` and analytical gradients.

3.  **Diagnostic Checking**: Assess fit quality using Q-Q plots,
    probability plots, and goodness-of-fit tests (e.g.,
    Kolmogorov-Smirnov).

4.  **Progressive Complexity**: If simple models inadequate, try
    3-parameter models (EKw or Mc), then 4-parameter models (BKw or
    KKw).

5.  **Information Criteria**: Use AIC, BIC, or AICc to balance
    goodness-of-fit and model parsimony. Lower values indicate better
    models.

6.  **Nested Testing**: Perform likelihood ratio tests when comparing
    nested models (e.g., Kw vs. EKw).

7.  **Cross-Validation**: Validate final model on held-out data or via
    bootstrap procedures.

8.  **Residual Analysis**: Examine probability integral transform
    residuals for uniformity and independence.

## Statistical Inference

Maximum likelihood estimation is performed using numerical optimization:

1.  Obtain starting values via
    [`gkwgetstartvalues`](https://evandeilton.github.io/gkwdist/reference/gkwgetstartvalues.md)
    or manual specification based on sample moments

2.  Minimize negative log-likelihood using
    [`optim`](https://rdrr.io/r/stats/optim.html) with `method = "BFGS"`
    or `"L-BFGS-B"`

3.  Provide analytical gradient via `gr` argument for faster convergence
    and improved accuracy

4.  Set `hessian = TRUE` to obtain observed information matrix

5.  Compute standard errors as `sqrt(diag(solve(hessian)))`

6.  Construct confidence intervals using normal approximation or profile
    likelihood

7.  Perform hypothesis tests using Wald, score, or likelihood ratio
    statistics

For large samples (\\n \> 10^4\\), consider using `method = "L-BFGS-B"`
with box constraints to prevent parameter estimates from drifting to
boundary values or becoming numerically unstable.

## References

Carrasco, J. M. F., Ferrari, S. L. P., and Cordeiro, G. M. (2010). A new
generalized Kumaraswamy distribution. *arXiv preprint arXiv:1004.0911*.
[doi:10.48550/arXiv.1004.0911](https://doi.org/10.48550/arXiv.1004.0911)

Jones, M. C. (2009). Kumaraswamy's distribution: A beta-type
distribution with some tractability advantages. *Statistical
Methodology*, **6**(1), 70-81.
[doi:10.1016/j.stamet.2008.04.001](https://doi.org/10.1016/j.stamet.2008.04.001)

Kumaraswamy, P. (1980). A generalized probability density function for
double-bounded random processes. *Journal of Hydrology*, **46**(1-2),
79-88.
[doi:10.1016/0022-1694(80)90036-0](https://doi.org/10.1016/0022-1694%2880%2990036-0)

Cordeiro, G. M., and de Castro, M. (2011). A new family of generalized
distributions. *Journal of Statistical Computation and Simulation*,
**81**(7), 883-898.
[doi:10.1080/00949650903530745](https://doi.org/10.1080/00949650903530745)

McDonald, J. B. (1984). Some generalized functions for the size
distribution of income. *Econometrica*, **52**(3), 647-663.
[doi:10.2307/1913469](https://doi.org/10.2307/1913469)

Eddelbuettel, D., and Francois, R. (2011). Rcpp: Seamless R and C++
Integration. *Journal of Statistical Software*, **40**(8), 1-18.
[doi:10.18637/jss.v040.i08](https://doi.org/10.18637/jss.v040.i08)

Eddelbuettel, D., and Sanderson, C. (2014). RcppArmadillo: Accelerating
R with high-performance C++ linear algebra. *Computational Statistics &
Data Analysis*, **71**, 1054-1063.
[doi:10.1016/j.csda.2013.02.005](https://doi.org/10.1016/j.csda.2013.02.005)

## See also

[`Beta`](https://rdrr.io/r/stats/Beta.html) for the standard beta
distribution, [`optim`](https://rdrr.io/r/stats/optim.html) for
numerical optimization, [`dbeta`](https://rdrr.io/r/stats/Beta.html) for
beta distribution functions

## Author

Lopes, J. E. <evandeilton@gmail.com> [ORCID:
0009-0007-5887-4084](https://orcid.org/0009-0007-5887-4084)

LEG - Laboratory of Statistics and Geoinformation

PPGMNE - Graduate Program in Numerical Methods in Engineering

Federal University of Paraná (UFPR), Brazil

## Examples

``` r
x <- c(0.1, 0.3, 0.5, 0.7, 0.9)
dgkw(x, alpha = 2, beta = 3, gamma = 1.5, delta = 0.5, lambda = 1.2)
#> [1] 0.10703949 1.33993326 2.30916136 1.18032685 0.05372826

## Every sub-family is GKw with some parameters fixed
all.equal(dkw(x, alpha = 2, beta = 3), dgkw(x, alpha = 2, beta = 3))
#> [1] TRUE
all.equal(dekw(x, 2, 3, 1.2), dgkw(x, 2, 3, gamma = 1, delta = 0, lambda = 1.2))
#> [1] TRUE

## Maximum likelihood with the analytic gradient
set.seed(123)
y <- rkw(300, alpha = 2, beta = 3)
optim(c(alpha = 1, beta = 1), llkw, grkw, data = y, method = "L-BFGS-B",
    lower = 1e-04)$par
#>    alpha     beta 
#> 2.155045 3.387195 
```
