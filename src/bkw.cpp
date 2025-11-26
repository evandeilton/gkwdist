/**
 * @file bkw.cpp
 * @brief Beta-Kumaraswamy (BKw) Distribution Functions
 * 
 * @details
 * This file implements the full suite of distribution functions for the
 * four-parameter Beta-Kumaraswamy (BKw) distribution, which is a sub-family
 * of the Generalized Kumaraswamy (GKw) distribution obtained by setting λ = 1.
 * 
 * **Relationship to GKw:**
 * \deqn{BKw(\alpha, \beta, \gamma, \delta) = GKw(\alpha, \beta, \gamma, \delta, 1)}
 * 
 * The BKw distribution has probability density function:
 * \deqn{
 *   f(x; \alpha, \beta, \gamma, \delta) = 
 *   \frac{\alpha \beta}{B(\gamma, \delta+1)} x^{\alpha-1} (1-x^\alpha)^{\beta(\delta+1)-1}
 *   [1-(1-x^\alpha)^\beta]^{\gamma-1}
 * }
 * for \eqn{x \in (0,1)}, where \eqn{B(\cdot,\cdot)} is the Beta function.
 * 
 * **Derivation of the PDF:**
 * Starting from GKw with λ=1:
 * - The term \eqn{\{1-[1-(1-x^\alpha)^\beta]^\lambda\}^\delta} becomes \eqn{(1-x^\alpha)^{\beta\delta}}
 * - Combined with \eqn{(1-x^\alpha)^{\beta-1}}, this yields \eqn{(1-x^\alpha)^{\beta(\delta+1)-1}}
 * 
 * The cumulative distribution function is:
 * \deqn{
 *   F(x) = I_{1-(1-x^\alpha)^\beta}(\gamma, \delta+1)
 * }
 * where \eqn{I_y(a,b)} is the regularized incomplete Beta function.
 * 
 * The quantile function (inverse CDF) is:
 * \deqn{
 *   Q(p) = \left\{1 - \left[1 - Q_{Beta}(p; \gamma, \delta+1)\right]^{1/\beta}\right\}^{1/\alpha}
 * }
 * 
 * **Parameter Constraints:**
 * - \eqn{\alpha > 0} (shape parameter)
 * - \eqn{\beta > 0} (shape parameter)
 * - \eqn{\gamma > 0} (shape parameter)
 * - \eqn{\delta \geq 0} (shape parameter)
 * 
 * **Special Cases:**
 * | Distribution | Condition | Relation |
 * |--------------|-----------|----------|
 * | Kumaraswamy (Kw) | \eqn{\gamma = 1, \delta = 0} | Standard Kumaraswamy |
 * | Beta | \eqn{\alpha = \beta = 1} | Standard Beta(γ, δ+1) |
 * 
 * **Random Variate Generation:**
 * Uses inverse transform method:
 * 1. Generate \eqn{V \sim Beta(\gamma, \delta+1)}
 * 2. Return \eqn{X = \{1 - (1-V)^{1/\beta}\}^{1/\alpha}}
 * 
 * **Numerical Stability:**
 * All computations use log-space arithmetic and numerically stable helper
 * functions from utils.h to prevent overflow/underflow.
 * 
 * **Implemented Functions:**
 * - dbkw(): Probability density function (PDF)
 * - pbkw(): Cumulative distribution function (CDF)
 * - qbkw(): Quantile function (inverse CDF)
 * - rbkw(): Random variate generation
 * - llbkw(): Negative log-likelihood for MLE
 * - grbkw(): Gradient of negative log-likelihood
 * - hsbkw(): Hessian of negative log-likelihood
 * 
 * @author Lopes, J. E.
 * @date 2025-01-07
 * 
 * @see gkw.cpp for the parent distribution
 * @see utils.h for numerical stability functions and parameter validators
 * 
 * @note All functions use R's vectorization conventions with parameter recycling.
 * @note Thread-safe: No global state is modified.
 */

// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include "utils.h"


// ============================================================================
// PROBABILITY DENSITY FUNCTION
// ============================================================================

/**
 * @brief Probability Density Function of the BKw Distribution
 * 
 * Computes the density (or log-density) for the Beta-Kumaraswamy
 * distribution at specified quantiles.
 * 
 * @param x Vector of quantiles (values in (0,1))
 * @param alpha Shape parameter vector (α > 0)
 * @param beta Shape parameter vector (β > 0)
 * @param gamma Shape parameter vector (γ > 0)
 * @param delta Shape parameter vector (δ ≥ 0)
 * @param log_prob If TRUE, returns log-density; otherwise returns density
 * 
 * @return NumericVector of density values (or log-density if log_prob=TRUE)
 * 
 * @details
 * The log-density is computed as:
 * \deqn{
 *   \log f(x) = \log(\alpha) + \log(\beta) - \log B(\gamma, \delta+1)
 *   + (\alpha-1)\log(x) + (\beta(\delta+1)-1)\log(1-x^\alpha)
 *   + (\gamma-1)\log(1-(1-x^\alpha)^\beta)
 * }
 * 
 * @note Exported as .dbkw_cpp for internal package use
 */
// [[Rcpp::export(.dbkw_cpp)]]
Rcpp::NumericVector dbkw(
    const arma::vec& x,
    const Rcpp::NumericVector& alpha,
    const Rcpp::NumericVector& beta,
    const Rcpp::NumericVector& gamma,
    const Rcpp::NumericVector& delta,
    bool log_prob = false
) {
  // Convert R vectors to Armadillo vectors
  arma::vec alpha_vec(alpha.begin(), alpha.size());
  arma::vec beta_vec(beta.begin(), beta.size());
  arma::vec gamma_vec(gamma.begin(), gamma.size());
  arma::vec delta_vec(delta.begin(), delta.size());
  
  // Determine output length for recycling
  size_t n = std::max({x.n_elem, alpha_vec.n_elem, beta_vec.n_elem,
                      gamma_vec.n_elem, delta_vec.n_elem});
  
  // Initialize result with appropriate default
  arma::vec result(n);
  result.fill(log_prob ? R_NegInf : 0.0);
  
  for (size_t i = 0; i < n; ++i) {
    // Extract recycled parameters
    double a = alpha_vec[i % alpha_vec.n_elem];
    double b = beta_vec[i % beta_vec.n_elem];
    double g = gamma_vec[i % gamma_vec.n_elem];
    double d = delta_vec[i % delta_vec.n_elem];
    double xx = x[i % x.n_elem];
    
    // Validate parameters
    if (!check_bkw_pars(a, b, g, d)) {
      continue;
    }
    
    // Check support: x must be in (0, 1)
    if (xx <= 0.0 || xx >= 1.0 || !R_finite(xx)) {
      continue;
    }
    
    // ---- Log-space computation of density ----
    
    // Normalization constant: log(αβ / B(γ, δ+1))
    double logB = R::lbeta(g, d + 1.0);
    double log_const = safe_log(a) + safe_log(b) - logB;
    
    // Compute log(x) and log(x^α)
    double lx = safe_log(xx);
    double log_xalpha = a * lx;  // log(x^α)
    
    // Compute log(1 - x^α) using stable log1mexp
    double log_v = log1mexp(log_xalpha);
    if (!R_finite(log_v)) {
      continue;
    }
    
    // Term 1: (β(δ+1) - 1) * log(1 - x^α)
    double exponent1 = b * (d + 1.0) - 1.0;
    double term1 = exponent1 * log_v;
    
    // Compute log((1-x^α)^β) = β * log(1-x^α)
    double log_v_beta = b * log_v;
    
    // Compute log(1 - (1-x^α)^β) = log(w) using log1mexp
    double log_w = log1mexp(log_v_beta);
    if (!R_finite(log_w)) {
      continue;
    }
    
    // Term 2: (γ - 1) * log(w)
    double exponent2 = g - 1.0;
    double term2 = exponent2 * log_w;
    
    // Assemble log-density:
    // log(f) = log_const + (α-1)*log(x) + (β(δ+1)-1)*log(v) + (γ-1)*log(w)
    double log_pdf = log_const + (a - 1.0) * lx + term1 + term2;
    
    // Validate result
    if (!R_finite(log_pdf)) {
      continue;
    }
    
    // Return appropriate scale
    result(i) = log_prob ? log_pdf : safe_exp(log_pdf);
  }
  
  return Rcpp::NumericVector(result.memptr(), result.memptr() + result.n_elem);
}


// ============================================================================
// CUMULATIVE DISTRIBUTION FUNCTION
// ============================================================================

/**
 * @brief Cumulative Distribution Function of the BKw Distribution
 * 
 * Computes the cumulative probability for the Beta-Kumaraswamy
 * distribution at specified quantiles.
 * 
 * @param q Vector of quantiles
 * @param alpha Shape parameter vector (α > 0)
 * @param beta Shape parameter vector (β > 0)
 * @param gamma Shape parameter vector (γ > 0)
 * @param delta Shape parameter vector (δ ≥ 0)
 * @param lower_tail If TRUE, returns P(X ≤ q); otherwise P(X > q)
 * @param log_p If TRUE, returns log-probability
 * 
 * @return NumericVector of cumulative probabilities
 * 
 * @details
 * The CDF is computed as:
 * \deqn{F(x) = I_{1-(1-x^\alpha)^\beta}(\gamma, \delta+1)}
 * where \eqn{I_y(a,b)} is the regularized incomplete Beta function.
 * 
 * @note Exported as .pbkw_cpp for internal package use
 */
// [[Rcpp::export(.pbkw_cpp)]]
Rcpp::NumericVector pbkw(
    const arma::vec& q,
    const Rcpp::NumericVector& alpha,
    const Rcpp::NumericVector& beta,
    const Rcpp::NumericVector& gamma,
    const Rcpp::NumericVector& delta,
    bool lower_tail = true,
    bool log_p = false
) {
  // Convert R vectors to Armadillo vectors
  arma::vec alpha_vec(alpha.begin(), alpha.size());
  arma::vec beta_vec(beta.begin(), beta.size());
  arma::vec gamma_vec(gamma.begin(), gamma.size());
  arma::vec delta_vec(delta.begin(), delta.size());
  
  // Determine output length for recycling
  size_t n = std::max({q.n_elem, alpha_vec.n_elem, beta_vec.n_elem,
                      gamma_vec.n_elem, delta_vec.n_elem});
  
  arma::vec res(n);
  
  for (size_t i = 0; i < n; ++i) {
    // Extract recycled parameters
    double a = alpha_vec[i % alpha_vec.n_elem];
    double b = beta_vec[i % beta_vec.n_elem];
    double g = gamma_vec[i % gamma_vec.n_elem];
    double d = delta_vec[i % delta_vec.n_elem];
    double xx = q[i % q.n_elem];
    
    // Validate parameters
    if (!check_bkw_pars(a, b, g, d)) {
      res(i) = NA_REAL;
      continue;
    }
    
    // Handle boundary: q ≤ 0
    if (!R_finite(xx) || xx <= 0.0) {
      double prob0 = lower_tail ? 0.0 : 1.0;
      res(i) = log_p ? safe_log(prob0) : prob0;
      continue;
    }
    
    // Handle boundary: q ≥ 1
    if (xx >= 1.0) {
      double prob1 = lower_tail ? 1.0 : 0.0;
      res(i) = log_p ? safe_log(prob1) : prob1;
      continue;
    }
    
    // ---- Compute CDF ----
    
    // Step 1: x^α
    double lx = safe_log(xx);
    double xalpha = safe_exp(a * lx);
    
    // Step 2: 1 - x^α
    double one_minus_xalpha = 1.0 - xalpha;
    if (one_minus_xalpha <= 0.0) {
      double prob1 = lower_tail ? 1.0 : 0.0;
      res(i) = log_p ? safe_log(prob1) : prob1;
      continue;
    }
    
    // Step 3: (1 - x^α)^β
    double v_beta = safe_pow(one_minus_xalpha, b);
    
    // Step 4: z = 1 - (1 - x^α)^β
    double z = 1.0 - v_beta;
    if (z <= 0.0) {
      double prob0 = lower_tail ? 0.0 : 1.0;
      res(i) = log_p ? safe_log(prob0) : prob0;
      continue;
    }
    if (z >= 1.0) {
      double prob1 = lower_tail ? 1.0 : 0.0;
      res(i) = log_p ? safe_log(prob1) : prob1;
      continue;
    }
    
    // Step 5: F(x) = I_z(γ, δ+1) via pbeta
    double val = R::pbeta(z, g, d + 1.0, true, false);
    
    // Apply tail adjustment
    if (!lower_tail) {
      val = 1.0 - val;
    }
    
    // Apply log transformation
    if (log_p) {
      val = safe_log(val);
    }
    
    res(i) = val;
  }
  
  return Rcpp::NumericVector(res.memptr(), res.memptr() + res.n_elem);
}


// ============================================================================
// QUANTILE FUNCTION
// ============================================================================

/**
 * @brief Quantile Function (Inverse CDF) of the BKw Distribution
 * 
 * Computes quantiles for the Beta-Kumaraswamy distribution
 * given probability values.
 * 
 * @param p Vector of probabilities (values in [0,1])
 * @param alpha Shape parameter vector (α > 0)
 * @param beta Shape parameter vector (β > 0)
 * @param gamma Shape parameter vector (γ > 0)
 * @param delta Shape parameter vector (δ ≥ 0)
 * @param lower_tail If TRUE, probabilities are P(X ≤ x); otherwise P(X > x)
 * @param log_p If TRUE, probabilities are given as log(p)
 * 
 * @return NumericVector of quantiles
 * 
 * @details
 * The quantile function inverts the CDF:
 * \deqn{Q(p) = \left\{1 - \left[1 - Q_{Beta}(p; \gamma, \delta+1)\right]^{1/\beta}\right\}^{1/\alpha}}
 * 
 * @note Exported as .qbkw_cpp for internal package use
 */
// [[Rcpp::export(.qbkw_cpp)]]
Rcpp::NumericVector qbkw(
    const arma::vec& p,
    const Rcpp::NumericVector& alpha,
    const Rcpp::NumericVector& beta,
    const Rcpp::NumericVector& gamma,
    const Rcpp::NumericVector& delta,
    bool lower_tail = true,
    bool log_p = false
) {
  // Convert R vectors to Armadillo vectors
  arma::vec alpha_vec(alpha.begin(), alpha.size());
  arma::vec beta_vec(beta.begin(), beta.size());
  arma::vec gamma_vec(gamma.begin(), gamma.size());
  arma::vec delta_vec(delta.begin(), delta.size());
  
  // Determine output length for recycling
  size_t n = std::max({p.n_elem, alpha_vec.n_elem, beta_vec.n_elem,
                      gamma_vec.n_elem, delta_vec.n_elem});
  
  arma::vec res(n);
  
  for (size_t i = 0; i < n; ++i) {
    // Extract recycled parameters
    double a = alpha_vec[i % alpha_vec.n_elem];
    double b = beta_vec[i % beta_vec.n_elem];
    double g = gamma_vec[i % gamma_vec.n_elem];
    double d = delta_vec[i % delta_vec.n_elem];
    double pp = p[i % p.n_elem];
    
    // Validate parameters
    if (!check_bkw_pars(a, b, g, d)) {
      res(i) = NA_REAL;
      continue;
    }
    
    // ---- Convert probability to linear scale ----
    if (log_p) {
      if (pp > 0.0) {
        res(i) = NA_REAL;
        continue;
      }
      pp = safe_exp(pp);
    }
    
    // Handle upper tail (pp is now always linear scale)
    if (!lower_tail) {
      pp = 1.0 - pp;
    }
    
    // Handle boundary cases
    if (pp <= 0.0) {
      res(i) = 0.0;
      continue;
    }
    if (pp >= 1.0) {
      res(i) = 1.0;
      continue;
    }
    
    // ---- Compute quantile via inverse transformations ----
    
    // Step 1: y = Q_Beta(p, γ, δ+1)
    double y = R::qbeta(pp, g, d + 1.0, true, false);
    
    if (y <= 0.0) {
      res(i) = 0.0;
      continue;
    }
    if (y >= 1.0) {
      res(i) = 1.0;
      continue;
    }
    
    // Step 2: part = 1 - y
    double part = 1.0 - y;
    if (part <= 0.0) {
      res(i) = 1.0;
      continue;
    }
    if (part >= 1.0) {
      res(i) = 0.0;
      continue;
    }
    
    // Step 3: inner = (1 - y)^(1/β)
    double inner = safe_pow(part, 1.0 / b);
    
    // Step 4: xval = 1 - inner
    double xval = 1.0 - inner;
    xval = std::max(0.0, std::min(1.0, xval));
    
    // Step 5: x = xval^(1/α)
    double qv = (a == 1.0) ? xval : safe_pow(xval, 1.0 / a);
    
    // Clamp to valid support
    qv = std::max(0.0, std::min(1.0, qv));
    
    res(i) = qv;
  }
  
  return Rcpp::NumericVector(res.memptr(), res.memptr() + res.n_elem);
}


// ============================================================================
// RANDOM NUMBER GENERATION
// ============================================================================

/**
 * @brief Random Variate Generation for the BKw Distribution
 * 
 * Generates random samples from the Beta-Kumaraswamy distribution
 * using the inverse transform method.
 * 
 * @param n Number of random variates to generate
 * @param alpha Shape parameter vector (α > 0)
 * @param beta Shape parameter vector (β > 0)
 * @param gamma Shape parameter vector (γ > 0)
 * @param delta Shape parameter vector (δ ≥ 0)
 * 
 * @return NumericVector of n random variates from BKw distribution
 * 
 * @details
 * Algorithm:
 * 1. Generate V ~ Beta(γ, δ+1)
 * 2. Return X = {1 - (1-V)^(1/β)}^(1/α)
 * 
 * @note Exported as .rbkw_cpp for internal package use
 */
// [[Rcpp::export(.rbkw_cpp)]]
Rcpp::NumericVector rbkw(
    int n,
    const Rcpp::NumericVector& alpha,
    const Rcpp::NumericVector& beta,
    const Rcpp::NumericVector& gamma,
    const Rcpp::NumericVector& delta
) {
  if (n <= 0) {
    Rcpp::stop("rbkw: n must be positive");
  }
  
  // Convert R vectors to Armadillo vectors
  arma::vec alpha_vec(alpha.begin(), alpha.size());
  arma::vec beta_vec(beta.begin(), beta.size());
  arma::vec gamma_vec(gamma.begin(), gamma.size());
  arma::vec delta_vec(delta.begin(), delta.size());
  
  arma::vec out(n);
  
  for (int i = 0; i < n; ++i) {
    // Extract recycled parameters (direct modulo, no intermediate variable)
    double a = alpha_vec[i % alpha_vec.n_elem];
    double b = beta_vec[i % beta_vec.n_elem];
    double g = gamma_vec[i % gamma_vec.n_elem];
    double d = delta_vec[i % delta_vec.n_elem];
    
    // Validate parameters
    if (!check_bkw_pars(a, b, g, d)) {
      out(i) = NA_REAL;
      Rcpp::warning("rbkw: invalid parameters at index %d", i + 1);
      continue;
    }
    
    // Generate V ~ Beta(γ, δ+1)
    double V = R::rbeta(g, d + 1.0);
    
    // Handle boundary cases
    double one_minus_V = 1.0 - V;
    if (one_minus_V <= 0.0) {
      out(i) = 1.0;
      continue;
    }
    if (one_minus_V >= 1.0) {
      out(i) = 0.0;
      continue;
    }
    
    // Transform: (1 - V)^(1/β)
    double temp = safe_pow(one_minus_V, 1.0 / b);
    
    // Transform: 1 - (1 - V)^(1/β)
    double xval = 1.0 - temp;
    xval = std::max(0.0, std::min(1.0, xval));
    
    // Transform: {1 - (1 - V)^(1/β)}^(1/α)
    double rv;
    if (a == 1.0) {
      rv = xval;
    } else {
      rv = safe_pow(xval, 1.0 / a);
      rv = std::max(0.0, std::min(1.0, rv));
    }
    
    out(i) = rv;
  }
  
  return Rcpp::NumericVector(out.memptr(), out.memptr() + out.n_elem);
}


// ============================================================================
// NEGATIVE LOG-LIKELIHOOD FUNCTION
// ============================================================================

/**
 * @brief Negative Log-Likelihood for BKw Distribution
 * 
 * Computes the negative log-likelihood function for parameter estimation
 * via maximum likelihood.
 * 
 * @param par Parameter vector of length 4: (α, β, γ, δ)
 * @param data Vector of observations (must be in (0,1))
 * 
 * @return Negative log-likelihood value (scalar)
 * 
 * @details
 * The log-likelihood for n observations is:
 * \deqn{
 *   \ell(\theta) = n[\ln\alpha + \ln\beta - \ln B(\gamma,\delta+1)]
 *   + (\alpha-1)\sum\ln x_i + (\beta(\delta+1)-1)\sum\ln v_i
 *   + (\gamma-1)\sum\ln w_i
 * }
 * where:
 * - \eqn{v_i = 1 - x_i^\alpha}
 * - \eqn{w_i = 1 - v_i^\beta}
 * 
 * Returns +Inf for invalid parameters or data outside (0,1).
 * 
 * @note Exported as .llbkw_cpp for internal package use
 */
// [[Rcpp::export(.llbkw_cpp)]]
double llbkw(const Rcpp::NumericVector& par, const Rcpp::NumericVector& data) {
  // Validate parameter vector length
  if (par.size() < 4) {
    return R_PosInf;
  }
  
  // Extract parameters
  double a = par[0];  // alpha
  double b = par[1];  // beta
  double g = par[2];  // gamma
  double d = par[3];  // delta
  
  // Validate parameters using consistent checker
  if (!check_bkw_pars(a, b, g, d)) {
    return R_PosInf;
  }
  
  // Convert and validate data
  arma::vec x = Rcpp::as<arma::vec>(data);
  int n = x.n_elem;
  
  if (n == 0 || arma::any(x <= 0.0) || arma::any(x >= 1.0)) {
    return R_PosInf;
  }
  
  // Numerical stability constant
  const double eps = std::sqrt(std::numeric_limits<double>::epsilon());
  
  // Constant term: n * [log(α) + log(β) - log(B(γ, δ+1))]
  double logB = R::lbeta(g, d + 1.0);
  double ll_const = n * (safe_log(a) + safe_log(b) - logB);
  
  // Term 1: (α - 1) * Σ log(x)
  arma::vec lx = vec_safe_log(x);
  double sum1 = (a - 1.0) * arma::sum(lx);
  
  // Exponent for term 2
  double exp1 = b * (d + 1.0) - 1.0;
  double sum2 = 0.0;
  double sum3 = 0.0;
  
  for (int i = 0; i < n; i++) {
    double xi = x(i);
    
    // Compute x^α stably
    double log_xa = a * safe_log(xi);
    double xa;
    if (log_xa < -700.0) {
      xa = 0.0;
    } else if (log_xa > 700.0) {
      xa = 1e300;
    } else {
      xa = safe_exp(log_xa);
    }
    
    // Compute v = 1 - x^α
    double v;
    if (xa > 0.5) {
      v = std::max(1.0 - xa, eps);
    } else {
      v = 1.0 - xa;
    }
    v = std::max(std::min(v, 1.0 - eps), eps);
    
    // Term 2: (β(δ+1) - 1) * log(v)
    sum2 += exp1 * safe_log(v);
    
    // Compute v^β stably
    double log_vb = b * safe_log(v);
    double vb;
    if (log_vb < -700.0) {
      vb = 0.0;
    } else if (log_vb > 700.0) {
      vb = 1e300;
    } else {
      vb = safe_exp(log_vb);
    }
    
    // Compute w = 1 - v^β
    double w;
    if (vb > 0.5) {
      w = std::max(1.0 - vb, eps);
    } else {
      w = 1.0 - vb;
    }
    w = std::max(std::min(w, 1.0 - eps), eps);
    
    // Term 3: (γ - 1) * log(w) (skip if γ = 1)
    if (g != 1.0) {
      sum3 += (g - 1.0) * safe_log(w);
    }
  }
  
  // Combine all terms
  double ll = ll_const + sum1 + sum2 + sum3;
  
  // Final validity check
  if (!std::isfinite(ll)) {
    return R_PosInf;
  }
  
  // Return negative log-likelihood
  return -ll;
}


// ============================================================================
// GRADIENT OF NEGATIVE LOG-LIKELIHOOD
// ============================================================================

/**
 * @brief Gradient of Negative Log-Likelihood for BKw Distribution
 * 
 * Computes the gradient vector of the negative log-likelihood for
 * optimization-based parameter estimation.
 * 
 * @param par Parameter vector of length 4: (α, β, γ, δ)
 * @param data Vector of observations (must be in (0,1))
 * 
 * @return NumericVector of length 4 containing partial derivatives
 *         with respect to (α, β, γ, δ)
 * 
 * @details
 * The gradient components are:
 * - ∂ℓ/∂α = n/α + Σlog(x) - Σ[x^α log(x) * ((β(δ+1)-1)/v - (γ-1)βv^(β-1)/w)]
 * - ∂ℓ/∂β = n/β + (δ+1)Σlog(v) - (γ-1)Σ[v^β log(v)/w]
 * - ∂ℓ/∂γ = -n[ψ(γ) - ψ(γ+δ+1)] + Σlog(w)
 * - ∂ℓ/∂δ = -n[ψ(δ+1) - ψ(γ+δ+1)] + βΣlog(v)
 * 
 * @note Exported as .grbkw_cpp for internal package use
 */
// [[Rcpp::export(.grbkw_cpp)]]
Rcpp::NumericVector grbkw(const Rcpp::NumericVector& par, const Rcpp::NumericVector& data) {
  // Validate parameter vector length
  if (par.size() < 4) {
    Rcpp::warning("Parameter vector must have at least 4 elements for BKw");
    return Rcpp::NumericVector(4, R_NaN);
  }
  
  // Extract parameters
  double alpha = par[0];
  double beta = par[1];
  double gamma = par[2];
  double delta = par[3];
  
  // Validate parameters using consistent checker
  if (!check_bkw_pars(alpha, beta, gamma, delta)) {
    Rcpp::warning("Invalid parameters in grbkw");
    return Rcpp::NumericVector(4, R_NaN);
  }
  
  // Convert and validate data
  arma::vec x;
  try {
    x = Rcpp::as<arma::vec>(data);
  } catch (...) {
    Rcpp::warning("Failed to convert data to arma::vec in grbkw");
    return Rcpp::NumericVector(4, R_NaN);
  }
  
  if (x.n_elem == 0 || x.has_nan() || arma::any(x <= 0) || arma::any(x >= 1)) {
    Rcpp::warning("Data must be strictly in (0,1) and non-empty for grbkw");
    return Rcpp::NumericVector(4, R_NaN);
  }
  
  int n = x.n_elem;
  Rcpp::NumericVector grad(4, 0.0);
  
  // Numerical stability constant
  const double eps = std::sqrt(std::numeric_limits<double>::epsilon());
  
  // ---- Compute intermediate quantities ----
  
  arma::vec log_x = vec_safe_log(x);
  arma::vec x_alpha = vec_safe_pow(x, alpha);
  arma::vec x_alpha_log_x = x_alpha % log_x;
  
  // Compute v = 1 - x^α
  arma::vec v(n);
  for (int i = 0; i < n; i++) {
    if (x_alpha(i) < 0.5) {
      v(i) = 1.0 - x_alpha(i);
    } else {
      v(i) = -std::expm1(alpha * log_x(i));
    }
    v(i) = std::max(std::min(v(i), 1.0 - eps), eps);
  }
  
  arma::vec log_v = vec_safe_log(v);
  
  // Compute v^(β-1) and v^β
  arma::vec v_beta_m1;
  if (std::abs(beta - 1.0) < eps) {
    v_beta_m1.ones(n);
  } else {
    v_beta_m1 = vec_safe_pow(v, beta - 1.0);
  }
  arma::vec v_beta = v % v_beta_m1;
  arma::vec v_beta_log_v = v_beta % log_v;
  
  // Compute w = 1 - v^β
  arma::vec w(n);
  arma::vec log_w(n);
  for (int i = 0; i < n; i++) {
    if (v_beta(i) < 0.5) {
      w(i) = 1.0 - v_beta(i);
    } else {
      w(i) = -std::expm1(beta * log_v(i));
    }
    w(i) = std::max(std::min(w(i), 1.0 - eps), eps);
    log_w(i) = safe_log(w(i));
  }
  
  // Compute digamma values
  double digamma_gamma = R::digamma(gamma);
  double digamma_delta_plus_1 = R::digamma(delta + 1.0);
  double digamma_sum = R::digamma(gamma + delta + 1.0);
  
  // ---- Calculate gradient components ----
  
  double term_beta_delta = beta * (delta + 1.0) - 1.0;
  double term_gamma = gamma - 1.0;
  
  // d_alpha = n/α + Σlog(x) - Σ[x^α log(x) * ((β(δ+1)-1)/v - (γ-1)βv^(β-1)/w)]
  double d_alpha = n / alpha + arma::sum(log_x);
  for (int i = 0; i < n; i++) {
    double alpha_term = x_alpha_log_x(i) * (
      term_beta_delta / v(i) - term_gamma * beta * v_beta_m1(i) / w(i)
    );
    if (std::isfinite(alpha_term)) {
      d_alpha -= alpha_term;
    }
  }
  
  // d_beta = n/β + (δ+1)Σlog(v) - (γ-1)Σ[v^β log(v)/w]
  double d_beta = n / beta + (delta + 1.0) * arma::sum(log_v);
  if (term_gamma != 0.0) {
    for (int i = 0; i < n; i++) {
      double beta_term = term_gamma * v_beta_log_v(i) / w(i);
      if (std::isfinite(beta_term)) {
        d_beta -= beta_term;
      }
    }
  }
  
  // d_gamma = -n[ψ(γ) - ψ(γ+δ+1)] + Σlog(w)
  double d_gamma = -n * (digamma_gamma - digamma_sum) + arma::sum(log_w);
  
  // d_delta = -n[ψ(δ+1) - ψ(γ+δ+1)] + βΣlog(v)
  double d_delta = -n * (digamma_delta_plus_1 - digamma_sum) + beta * arma::sum(log_v);
  
  // Final validity check
  if (!std::isfinite(d_alpha) || !std::isfinite(d_beta) ||
      !std::isfinite(d_gamma) || !std::isfinite(d_delta)) {
      Rcpp::warning("Gradient calculation produced non-finite values in grbkw");
    return Rcpp::NumericVector(4, R_NaN);
  }
  
  // Return NEGATIVE gradient (for minimization)
  grad[0] = -d_alpha;
  grad[1] = -d_beta;
  grad[2] = -d_gamma;
  grad[3] = -d_delta;
  
  return grad;
}


// ============================================================================
// HESSIAN OF NEGATIVE LOG-LIKELIHOOD
// ============================================================================

/**
 * @brief Hessian Matrix of Negative Log-Likelihood for BKw Distribution
 * 
 * Computes the Hessian matrix (matrix of second partial derivatives) of
 * the negative log-likelihood for standard error estimation and
 * optimization algorithms.
 * 
 * @param par Parameter vector of length 4: (α, β, γ, δ)
 * @param data Vector of observations (must be in (0,1))
 * 
 * @return NumericMatrix of dimension 4×4 containing the Hessian
 * 
 * @details
 * Computes analytical second derivatives. The Hessian is symmetric.
 * Parameter ordering: (α, β, γ, δ) → indices (0, 1, 2, 3).
 * 
 * Returns NaN matrix for invalid inputs.
 * 
 * @note Exported as .hsbkw_cpp for internal package use
 */
// [[Rcpp::export(.hsbkw_cpp)]]
Rcpp::NumericMatrix hsbkw(const Rcpp::NumericVector& par, const Rcpp::NumericVector& data) {
  // Initialize NaN matrix for error cases
  Rcpp::NumericMatrix nanH(4, 4);
  nanH.fill(R_NaN);
  
  // Validate parameter vector length
  if (par.size() < 4) {
    Rcpp::warning("Parameter vector must have at least 4 elements for BKw");
    return nanH;
  }
  
  // Extract parameters
  double alpha = par[0];
  double beta = par[1];
  double gamma = par[2];
  double delta = par[3];
  
  // Validate parameters using consistent checker
  if (!check_bkw_pars(alpha, beta, gamma, delta)) {
    Rcpp::warning("Invalid parameters in hsbkw");
    return nanH;
  }
  
  // Convert and validate data
  arma::vec x;
  try {
    x = Rcpp::as<arma::vec>(data);
  } catch (...) {
    Rcpp::warning("Failed to convert data to arma::vec in hsbkw");
    return nanH;
  }
  
  if (x.n_elem == 0 || x.has_nan() || arma::any(x <= 0) || arma::any(x >= 1)) {
    Rcpp::warning("Data must be strictly in (0,1) and non-empty for hsbkw");
    return nanH;
  }
  
  int n = x.n_elem;
  
  // Initialize Hessian matrix
  arma::mat H(4, 4, arma::fill::zeros);
  
  // Numerical stability constant
  const double eps = std::sqrt(std::numeric_limits<double>::epsilon());
  
  // ---- Constant terms (independent of observations) ----
  
  // H(α,α) from n*ln(α): -n/α²
  H(0, 0) = -n / (alpha * alpha);
  
  // H(β,β) from n*ln(β): -n/β²
  H(1, 1) = -n / (beta * beta);
  
  // Compute trigamma values
  double trigamma_gamma = R::trigamma(gamma);
  double trigamma_delta_plus_1 = R::trigamma(delta + 1.0);
  double trigamma_sum = R::trigamma(gamma + delta + 1.0);
  
  // H(γ,γ): -n*[ψ₁(γ) - ψ₁(γ+δ+1)]
  H(2, 2) = -n * (trigamma_gamma - trigamma_sum);
  
  // H(δ,δ): -n*[ψ₁(δ+1) - ψ₁(γ+δ+1)]
  H(3, 3) = -n * (trigamma_delta_plus_1 - trigamma_sum);
  
  // H(γ,δ) = H(δ,γ): n*ψ₁(γ+δ+1)
  H(2, 3) = n * trigamma_sum;
  H(3, 2) = H(2, 3);
  
  // Precompute common factors
  double beta_delta_factor = beta * (delta + 1.0) - 1.0;
  double gamma_minus_1 = gamma - 1.0;
  
  // ---- Observation-dependent terms ----
  for (int i = 0; i < n; i++) {
    double xi = x(i);
    double ln_xi = safe_log(xi);
    
    // ---- Compute A = x^α and derivatives ----
    double log_A = alpha * ln_xi;
    double A, dA_dalpha, d2A_dalpha2;
    
    if (std::abs(log_A) > 700.0) {
      if (log_A < -700.0) {
        A = 0.0;
        dA_dalpha = 0.0;
        d2A_dalpha2 = 0.0;
      } else {
        A = safe_exp(log_A);
        dA_dalpha = A * ln_xi;
        d2A_dalpha2 = dA_dalpha * ln_xi;
      }
    } else {
      A = std::exp(log_A);
      dA_dalpha = A * ln_xi;
      d2A_dalpha2 = dA_dalpha * ln_xi;
    }
    
    // ---- Compute v = 1 - A and derivatives ----
    double v, ln_v;
    if (A > 0.5) {
      v = -std::expm1(log_A);
    } else {
      v = 1.0 - A;
    }
    v = std::max(std::min(v, 1.0 - eps), eps);
    ln_v = safe_log(v);
    
    double dv_dalpha = -dA_dalpha;
    double d2v_dalpha2 = -d2A_dalpha2;
    
    // ---- Derivatives for L5: (β(δ+1)-1)*ln(v) ----
    double d2L5_dalpha2 = 0.0;
    if (std::abs(beta_delta_factor) > eps) {
      double term = (d2v_dalpha2 * v - dv_dalpha * dv_dalpha) / (v * v);
      if (std::isfinite(term)) {
        d2L5_dalpha2 = beta_delta_factor * term;
      }
    }
    
    double d2L5_dalpha_dbeta = (delta + 1.0) * (dv_dalpha / v);
    double d2L5_dalpha_ddelta = beta * (dv_dalpha / v);
    double d2L5_dbeta_ddelta = ln_v;
    
    // ---- Compute w = 1 - v^β and derivatives ----
    double v_beta = safe_pow(v, beta);
    double w;
    if (v_beta > 0.5) {
      w = -std::expm1(beta * ln_v);
    } else {
      w = 1.0 - v_beta;
    }
    w = std::max(std::min(w, 1.0 - eps), eps);
    
    double v_beta_m1 = (beta > 1.0) ? v_beta / v : 1.0;
    if (beta == 1.0) v_beta_m1 = 1.0;
    
    double dw_dv = -beta * v_beta_m1;
    double dw_dalpha = dw_dv * dv_dalpha;
    
    // ---- Derivatives for L6: (γ-1)*ln(w) ----
    double d2L6_dalpha2 = 0.0;
    double d2L6_dbeta2 = 0.0;
    double d2L6_dalpha_dbeta = 0.0;
    double d2L6_dalpha_dgamma = 0.0;
    double d2L6_dbeta_dgamma = 0.0;
    
    if (std::abs(gamma_minus_1) > eps) {
      // Second derivative of w w.r.t. α
      double d2w_dalpha2 = -beta * ((beta - 1.0) * safe_pow(v, beta - 2.0) *
                                    (dv_dalpha * dv_dalpha) +
                                    v_beta_m1 * d2v_dalpha2);
      
      double term_alpha2 = (d2w_dalpha2 * w - dw_dalpha * dw_dalpha) / (w * w);
      if (std::isfinite(term_alpha2)) {
        d2L6_dalpha2 = gamma_minus_1 * term_alpha2;
      }
      
      // Derivatives w.r.t. β
      double dw_dbeta = -v_beta * ln_v;
      double d2w_dbeta2 = -v_beta * (ln_v * ln_v);
      double term_beta2 = (d2w_dbeta2 * w - dw_dbeta * dw_dbeta) / (w * w);
      if (std::isfinite(term_beta2)) {
        d2L6_dbeta2 = gamma_minus_1 * term_beta2;
      }
      
      // Mixed derivative (α, β)
      double d_dw_dalpha_dbeta = -v_beta_m1 * (1.0 + beta * ln_v) * dv_dalpha;
      double mixed_term = (d_dw_dalpha_dbeta / w) - (dw_dalpha * dw_dbeta) / (w * w);
      if (std::isfinite(mixed_term)) {
        d2L6_dalpha_dbeta = gamma_minus_1 * mixed_term;
      }
      
      // Mixed derivatives with γ
      d2L6_dalpha_dgamma = dw_dalpha / w;
      d2L6_dbeta_dgamma = dw_dbeta / w;
    }
    
    // ---- Accumulate Hessian contributions ----
    if (std::isfinite(d2L5_dalpha2)) H(0, 0) += d2L5_dalpha2;
    if (std::isfinite(d2L6_dalpha2)) H(0, 0) += d2L6_dalpha2;
    
    if (std::isfinite(d2L6_dbeta2)) H(1, 1) += d2L6_dbeta2;
    
    if (std::isfinite(d2L5_dalpha_dbeta)) H(0, 1) += d2L5_dalpha_dbeta;
    if (std::isfinite(d2L6_dalpha_dbeta)) H(0, 1) += d2L6_dalpha_dbeta;
    H(1, 0) = H(0, 1);
    
    if (std::isfinite(d2L6_dalpha_dgamma)) H(0, 2) += d2L6_dalpha_dgamma;
    H(2, 0) = H(0, 2);
    
    if (std::isfinite(d2L5_dalpha_ddelta)) H(0, 3) += d2L5_dalpha_ddelta;
    H(3, 0) = H(0, 3);
    
    if (std::isfinite(d2L6_dbeta_dgamma)) H(1, 2) += d2L6_dbeta_dgamma;
    H(2, 1) = H(1, 2);
    
    if (std::isfinite(d2L5_dbeta_ddelta)) H(1, 3) += d2L5_dbeta_ddelta;
    H(3, 1) = H(1, 3);
  }
  
  // Final validity check
  if (!H.is_finite()) {
    Rcpp::warning("Hessian calculation produced non-finite values");
    return nanH;
  }
  
  // Return NEGATIVE Hessian (for minimization)
  return Rcpp::wrap(-H);
}











// // [[Rcpp::plugins(cpp11)]]
// // [[Rcpp::depends(RcppArmadillo)]]
// #include <RcppArmadillo.h>
// #include "utils.h"
// 
// /*
// ----------------------------------------------------------------------------
// BETA-KUMARASWAMY (BKw) DISTRIBUTION
// ----------------------------------------------------------------------------
// PDF:
// f(x; α, β, γ, δ) = (α β / B(γ, δ+1)) x^(α-1) (1 - x^α)^( β(δ+1) - 1 )
// [ 1 - (1 - x^α)^β ]^(γ - 1)
// 
// CDF:
// F(x; α, β, γ, δ) = I_{ [1 - (1 - x^α)^β ] } ( γ, δ + 1 )
// 
// QUANTILE:
// Q(p; α, β, γ, δ) = { 1 - [1 - ( I^{-1}_{p}(γ, δ+1) ) ]^(1/β) }^(1/α)
// (But see transformations step-by-step in code for numeric stability.)
// 
// RNG:
// If V ~ Beta(γ, δ+1) then
// X = { 1 - [1 - V ]^(1/β) }^(1/α)
// 
// LOG-LIKELIHOOD:
// ℓ(θ) = n log(α β) - n log B(γ, δ+1)
// + Σ { (α-1) log(x_i) + [β(δ+1)-1] log(1 - x_i^α) + (γ - 1) log( 1 - (1 - x_i^α)^β ) }
// 
// This file defines:
// - dbkw()  : density
// - pbkw()  : cumulative distribution
// - qbkw()  : quantile
// - rbkw()  : random generation
// - llbkw() : negative log-likelihood
// */
// 
// 
// // -----------------------------------------------------------------------------
// // 1) dbkw: PDF of Beta-Kumaraswamy
// // -----------------------------------------------------------------------------
// 
// 
// // [[Rcpp::export(.dbkw_cpp)]]
// Rcpp::NumericVector dbkw(
//    const arma::vec& x,
//    const Rcpp::NumericVector& alpha,
//    const Rcpp::NumericVector& beta,
//    const Rcpp::NumericVector& gamma,
//    const Rcpp::NumericVector& delta,
//    bool log_prob = false
// ) {
//  // Convert to arma::vec
//  arma::vec alpha_vec(alpha.begin(), alpha.size());
//  arma::vec beta_vec(beta.begin(), beta.size());
//  arma::vec gamma_vec(gamma.begin(), gamma.size());
//  arma::vec delta_vec(delta.begin(), delta.size());
//  
//  // Broadcast length
//  size_t n = std::max({x.n_elem,
//                      alpha_vec.n_elem,
//                      beta_vec.n_elem,
//                      gamma_vec.n_elem,
//                      delta_vec.n_elem});
//  
//  // Result
//  arma::vec result(n);
//  result.fill(log_prob ? R_NegInf : 0.0);
//  
//  for (size_t i = 0; i < n; ++i) {
//    double a = alpha_vec[i % alpha_vec.n_elem];
//    double b = beta_vec[i % beta_vec.n_elem];
//    double g = gamma_vec[i % gamma_vec.n_elem];
//    double d = delta_vec[i % delta_vec.n_elem];
//    double xx = x[i % x.n_elem];
//    
//    // Check parameter validity
//    if (!check_bkw_pars(a, b, g, d)) {
//      // Invalid params => density = 0 or -Inf
//      continue;
//    }
//    
//    // Outside (0,1) => density=0 or log_density=-Inf
//    if (xx <= 0.0 || xx >= 1.0 || !R_finite(xx)) {
//      continue;
//    }
//    
//    // PDF formula
//    // f(x) = (alpha*beta / B(gamma, delta+1)) *
//    //        x^(alpha-1) * (1 - x^alpha)^(beta*(delta+1) - 1) *
//    //        [1 - (1 - x^alpha)^beta]^(gamma - 1)
//    
//    // Precompute log_B = lbeta(g, d+1)
//    double logB = R::lbeta(g, d + 1.0);
//    double log_const = std::log(a) + std::log(b) - logB;
//    
//    double lx = std::log(xx);
//    double xalpha = a * lx;                    // log(x^alpha) = a * log(x)
//    double log_1_minus_xalpha = log1mexp(xalpha);
//    
//    // (beta*(delta+1) - 1) * log(1 - x^alpha)
//    double exponent1 = b * (d + 1.0) - 1.0;
//    double term1 = exponent1 * log_1_minus_xalpha;
//    
//    // [1 - (1 - x^alpha)^beta]^(gamma - 1)
//    // log(1 - (1 - x^alpha)^beta) = log1mexp( b * log(1 - x^alpha) )
//    double log_1_minus_xalpha_beta = b * log_1_minus_xalpha;
//    double log_bracket = log1mexp(log_1_minus_xalpha_beta);
//    double exponent2 = g - 1.0;
//    double term2 = exponent2 * log_bracket;
//    
//    // Full log pdf
//    double log_pdf = log_const +
//      (a - 1.0) * lx +
//      term1 +
//      term2;
//    
//    if (log_prob) {
//      result(i) = log_pdf;
//    } else {
//      // exp safely
//      result(i) = std::exp(log_pdf);
//    }
//  }
//  
//  return Rcpp::NumericVector(result.memptr(), result.memptr() + result.n_elem);
// }
// 
// 
// // -----------------------------------------------------------------------------
// // 2) pbkw: CDF of Beta-Kumaraswamy
// // -----------------------------------------------------------------------------
// 
// 
// // [[Rcpp::export(.pbkw_cpp)]]
// Rcpp::NumericVector pbkw(
//    const arma::vec& q,
//    const Rcpp::NumericVector& alpha,
//    const Rcpp::NumericVector& beta,
//    const Rcpp::NumericVector& gamma,
//    const Rcpp::NumericVector& delta,
//    bool lower_tail = true,
//    bool log_p = false
// ) {
//  // Convert
//  arma::vec alpha_vec(alpha.begin(), alpha.size());
//  arma::vec beta_vec(beta.begin(), beta.size());
//  arma::vec gamma_vec(gamma.begin(), gamma.size());
//  arma::vec delta_vec(delta.begin(), delta.size());
//  
//  // Broadcast
//  size_t n = std::max({q.n_elem,
//                      alpha_vec.n_elem,
//                      beta_vec.n_elem,
//                      gamma_vec.n_elem,
//                      delta_vec.n_elem});
//  
//  arma::vec res(n);
//  
//  for (size_t i = 0; i < n; ++i) {
//    double a = alpha_vec[i % alpha_vec.n_elem];
//    double b = beta_vec[i % beta_vec.n_elem];
//    double g = gamma_vec[i % gamma_vec.n_elem];
//    double d = delta_vec[i % delta_vec.n_elem];
//    double xx = q[i % q.n_elem];
//    
//    if (!check_bkw_pars(a, b, g, d)) {
//      res(i) = NA_REAL;
//      continue;
//    }
//    
//    if (!R_finite(xx) || xx <= 0.0) {
//      // x=0 => F=0
//      double prob0 = lower_tail ? 0.0 : 1.0;
//      res(i) = log_p ? std::log(prob0) : prob0;
//      continue;
//    }
//    
//    if (xx >= 1.0) {
//      // x=1 => F=1
//      double prob1 = lower_tail ? 1.0 : 0.0;
//      res(i) = log_p ? std::log(prob1) : prob1;
//      continue;
//    }
//    
//    // We want z = 1 - (1 - x^alpha)^beta
//    double lx = std::log(xx);
//    double xalpha = std::exp(a * lx);
//    double one_minus_xalpha = 1.0 - xalpha;
//    
//    if (one_minus_xalpha <= 0.0) {
//      // F(x) ~ 1 if x^alpha>=1
//      double prob1 = lower_tail ? 1.0 : 0.0;
//      res(i) = log_p ? std::log(prob1) : prob1;
//      continue;
//    }
//    
//    double temp = 1.0 - std::pow(one_minus_xalpha, b);
//    if (temp <= 0.0) {
//      double prob0 = lower_tail ? 0.0 : 1.0;
//      res(i) = log_p ? std::log(prob0) : prob0;
//      continue;
//    }
//    
//    if (temp >= 1.0) {
//      double prob1 = lower_tail ? 1.0 : 0.0;
//      res(i) = log_p ? std::log(prob1) : prob1;
//      continue;
//    }
//    
//    // Then F(x) = pbeta(temp, gamma, delta+1, TRUE, FALSE)
//    double val = R::pbeta(temp, g, d+1.0, true, false); // F
//    if (!lower_tail) {
//      val = 1.0 - val;
//    }
//    if (log_p) {
//      val = std::log(val);
//    }
//    res(i) = val;
//  }
//  
//  return Rcpp::NumericVector(res.memptr(), res.memptr() + res.n_elem);
// }
// 
// 
// // -----------------------------------------------------------------------------
// // 3) qbkw: QUANTILE of Beta-Kumaraswamy
// // -----------------------------------------------------------------------------
// 
// 
// // [[Rcpp::export(.qbkw_cpp)]]
// Rcpp::NumericVector qbkw(
//    const arma::vec& p,
//    const Rcpp::NumericVector& alpha,
//    const Rcpp::NumericVector& beta,
//    const Rcpp::NumericVector& gamma,
//    const Rcpp::NumericVector& delta,
//    bool lower_tail = true,
//    bool log_p = false
// ) {
//  arma::vec alpha_vec(alpha.begin(), alpha.size());
//  arma::vec beta_vec(beta.begin(), beta.size());
//  arma::vec gamma_vec(gamma.begin(), gamma.size());
//  arma::vec delta_vec(delta.begin(), delta.size());
//  
//  size_t n = std::max({p.n_elem,
//                      alpha_vec.n_elem,
//                      beta_vec.n_elem,
//                      gamma_vec.n_elem,
//                      delta_vec.n_elem});
//  
//  arma::vec res(n);
//  
//  for (size_t i = 0; i < n; ++i) {
//    double a = alpha_vec[i % alpha_vec.n_elem];
//    double b = beta_vec[i % beta_vec.n_elem];
//    double g = gamma_vec[i % gamma_vec.n_elem];
//    double d = delta_vec[i % delta_vec.n_elem];
//    double pp = p[i % p.n_elem];
//    
//    if (!check_bkw_pars(a, b, g, d)) {
//      res(i) = NA_REAL;
//      continue;
//    }
//    
//    // Convert from log_p if needed
//    if (log_p) {
//      if (pp > 0.0) {
//        // log(p) > 0 => p>1 => invalid
//        res(i) = NA_REAL;
//        continue;
//      }
//      pp = std::exp(pp);
//    }
//    // Convert if upper tail
//    if (!lower_tail) {
//      pp = 1.0 - pp;
//    }
//    
//    // Check boundaries
//    if (pp <= 0.0) {
//      res(i) = 0.0;
//      continue;
//    } else if (pp >= 1.0) {
//      res(i) = 1.0;
//      continue;
//    }
//    
//    // We do: y = qbeta(pp, gamma, delta+1)
//    double y = R::qbeta(pp, g, d+1.0, true, false);
//    if (y <= 0.0) {
//      res(i) = 0.0;
//      continue;
//    } else if (y >= 1.0) {
//      res(i) = 1.0;
//      continue;
//    }
//    
//    // Then x = {1 - [1 - y]^(1/b)}^(1/a)
//    double part = 1.0 - y;
//    if (part <= 0.0) {
//      res(i) = 1.0;
//      continue;
//    } else if (part >= 1.0) {
//      res(i) = 0.0;
//      continue;
//    }
//    
//    double inner = std::pow(part, 1.0/b);
//    double xval = 1.0 - inner;
//    if (xval < 0.0)  xval = 0.0;
//    if (xval > 1.0)  xval = 1.0;
//    
//    if (a == 1.0) {
//      // small optimization
//      res(i) = xval;
//    } else {
//      double qv = std::pow(xval, 1.0/a);
//      if (qv < 0.0)      qv = 0.0;
//      else if (qv > 1.0) qv = 1.0;
//      res(i) = qv;
//    }
//  }
//  
//  return Rcpp::NumericVector(res.memptr(), res.memptr() + res.n_elem);
// }
// 
// 
// // -----------------------------------------------------------------------------
// // 4) rbkw: RNG for Beta-Kumaraswamy
// // -----------------------------------------------------------------------------
// 
// 
// // [[Rcpp::export(.rbkw_cpp)]]
// Rcpp::NumericVector rbkw(
//    int n,
//    const Rcpp::NumericVector& alpha,
//    const Rcpp::NumericVector& beta,
//    const Rcpp::NumericVector& gamma,
//    const Rcpp::NumericVector& delta
// ) {
//  if (n <= 0) {
//    Rcpp::stop("rbkw: n must be positive");
//  }
//  
//  arma::vec alpha_vec(alpha.begin(), alpha.size());
//  arma::vec beta_vec(beta.begin(), beta.size());
//  arma::vec gamma_vec(gamma.begin(), gamma.size());
//  arma::vec delta_vec(delta.begin(), delta.size());
//  
//  size_t k = std::max({alpha_vec.n_elem,
//                      beta_vec.n_elem,
//                      gamma_vec.n_elem,
//                      delta_vec.n_elem});
//  
//  arma::vec out(n);
//  
//  for (int i = 0; i < n; ++i) {
//    size_t idx = i % k;
//    double a = alpha_vec[idx % alpha_vec.n_elem];
//    double b = beta_vec[idx % beta_vec.n_elem];
//    double g = gamma_vec[idx % gamma_vec.n_elem];
//    double d = delta_vec[idx % delta_vec.n_elem];
//    
//    if (!check_bkw_pars(a, b, g, d)) {
//      out(i) = NA_REAL;
//      Rcpp::warning("rbkw: invalid parameters at index %d", i+1);
//      continue;
//    }
//    
//    // V ~ Beta(g, d+1)
//    double V = R::rbeta(g, d + 1.0);
//    // X = {1 - (1 - V)^(1/b)}^(1/a)
//    double one_minus_V = 1.0 - V;
//    if (one_minus_V <= 0.0) {
//      out(i) = 1.0;
//      continue;
//    }
//    if (one_minus_V >= 1.0) {
//      out(i) = 0.0;
//      continue;
//    }
//    
//    double temp = std::pow(one_minus_V, 1.0/b);
//    double xval = 1.0 - temp;
//    if (xval < 0.0)  xval = 0.0;
//    if (xval > 1.0)  xval = 1.0;
//    
//    if (a == 1.0) {
//      out(i) = xval;
//    } else {
//      double rv = std::pow(xval, 1.0/a);
//      if (rv < 0.0) rv = 0.0;
//      if (rv > 1.0) rv = 1.0;
//      out(i) = rv;
//    }
//  }
//  
//  return Rcpp::NumericVector(out.memptr(), out.memptr() + out.n_elem);
// }
// 
// 
// // [[Rcpp::export(.llbkw_cpp)]]
// double llbkw(const Rcpp::NumericVector& par,
//             const Rcpp::NumericVector& data) {
//  // Parameter validation
//  if (par.size() < 4) {
//    return R_PosInf;
//  }
//  
//  double a = par[0];  // alpha > 0
//  double b = par[1];  // beta > 0
//  double g = par[2];  // gamma > 0
//  double d = par[3];  // delta >= 0
//  
//  // Basic parameter validation
//  if (a <= 0.0 || b <= 0.0 || g <= 0.0 || d < 0.0) {
//    return R_PosInf;
//  }
//  
//  // Convert data to armadillo vector
//  arma::vec x = Rcpp::as<arma::vec>(data);
//  int n = x.n_elem;
//  
//  // Basic data validation
//  if (n == 0 || arma::any(x <= 0.0) || arma::any(x >= 1.0)) {
//    return R_PosInf;
//  }
//  
//  // ----- Compute log-likelihood with careful numerical handling -----
//  
//  // Compute log-beta term
//  double logB = R::lbeta(g, d + 1.0);
//  
//  // Constant part: n * (log(a) + log(b) - logB)
//  double ll_const = n * (std::log(a) + std::log(b) - logB);
//  
//  // ----- Term 1: (alpha - 1) * sum(log(x)) -----
//  arma::vec lx = arma::log(x);
//  double sum1 = (a - 1.0) * arma::sum(lx);
//  
//  // ----- Term 2: (beta*(delta+1) - 1) * sum(log(1 - x^alpha)) -----
//  double exp1 = b * (d + 1.0) - 1.0;
//  double sum2 = 0.0;
//  
//  // ----- Term 3: (gamma - 1) * sum(log(1 - (1 - x^alpha)^beta)) -----
//  double sum3 = 0.0;
//  
//  // Small constant for numerical stability
//  double eps = std::sqrt(std::numeric_limits<double>::epsilon());
//  
//  // Process each observation for terms 2 and 3
//  for (int i = 0; i < n; i++) {
//    double xi = x(i);
//    
//    // Compute x^alpha (more accurately in log domain for extreme values)
//    double xa;
//    if (a * std::log(xi) < -700.0) {
//      xa = 0.0;  // Underflow protection
//    } else if (a * std::log(xi) > 700.0) {
//      xa = 1e300;  // Overflow protection - will lead to v ≈ 0
//    } else {
//      xa = std::pow(xi, a);
//    }
//    
//    // Compute v = 1 - x^alpha (more accurately for x^alpha close to 1)
//    double v;
//    if (xa > 0.5) {
//      v = std::max(1.0 - xa, eps);  // Ensure v > 0
//    } else {
//      v = 1.0 - xa;
//    }
//    
//    // Restrict v to valid range for numerical stability
//    v = std::max(std::min(v, 1.0 - eps), eps);
//    
//    // Term 2 contribution: (beta*(delta+1) - 1) * log(v)
//    sum2 += exp1 * std::log(v);
//    
//    // Compute v^beta (more accurately in log domain for extreme values)
//    double vb;
//    if (b * std::log(v) < -700.0) {
//      vb = 0.0;  // Underflow protection
//    } else if (b * std::log(v) > 700.0) {
//      vb = 1e300;  // Overflow protection - will lead to w ≈ 0
//    } else {
//      vb = std::pow(v, b);
//    }
//    
//    // Compute w = 1 - v^beta (more accurately for v^beta close to 1)
//    double w;
//    if (vb > 0.5) {
//      w = std::max(1.0 - vb, eps);  // Ensure w > 0
//    } else {
//      w = 1.0 - vb;
//    }
//    
//    // Restrict w to valid range for numerical stability
//    w = std::max(std::min(w, 1.0 - eps), eps);
//    
//    // Term 3 contribution: (gamma - 1) * log(w)
//    if (g != 1.0) {  // Skip if gamma = 1
//      sum3 += (g - 1.0) * std::log(w);
//    }
//  }
//  
//  // Combine all terms
//  double ll = ll_const + sum1 + sum2 + sum3;
//  
//  // Final validity check
//  if (!std::isfinite(ll)) {
//    return R_PosInf;
//  }
//  
//  // Return negative log-likelihood
//  return -ll;
// }
// 
// 
// // [[Rcpp::export(.grbkw_cpp)]]
// Rcpp::NumericVector grbkw(const Rcpp::NumericVector& par, const Rcpp::NumericVector& data) {
//  // Initialize gradient vector for early return cases
//  Rcpp::NumericVector grad(4, R_NaN);
//  
//  // Parameter extraction and validation
//  if (par.size() < 4) {
//    Rcpp::warning("Parameter vector must have at least 4 elements for BKw");
//    return grad;
//  }
//  
//  double alpha = par[0];   // Shape parameter α > 0
//  double beta = par[1];    // Shape parameter β > 0
//  double gamma = par[2];   // Shape parameter γ > 0
//  double delta = par[3];   // Shape parameter δ ≥ 0
//  
//  // Enhanced parameter validation with NaN checks
//  if (std::isnan(alpha) || std::isnan(beta) || std::isnan(gamma) || std::isnan(delta) ||
//      alpha <= 0 || beta <= 0 || gamma <= 0 || delta < 0) {
//    Rcpp::warning("Invalid parameters in grbkw: all must be positive (delta can be zero)");
//    return grad;
//  }
//  
//  // Data conversion and validation
//  arma::vec x;
//  try {
//    x = Rcpp::as<arma::vec>(data);
//  } catch (...) {
//    Rcpp::warning("Failed to convert data to arma::vec in grbkw");
//    return grad;
//  }
//  
//  // Comprehensive data validation
//  if (x.n_elem == 0 || x.has_nan() || arma::any(x <= 0) || arma::any(x >= 1)) {
//    Rcpp::warning("Data must be strictly in (0,1) and non-empty for grbkw");
//    return grad;
//  }
//  
//  int n = x.n_elem;  // Sample size
//  
//  // Reset gradient to zeros (we'll compute actual values now)
//  grad = Rcpp::NumericVector(4, 0.0);
//  
//  // Small constant for numerical stability - adaptive to machine precision
//  double eps = std::sqrt(std::numeric_limits<double>::epsilon());
//  
//  // -------- Step 1: Compute base transformations safely --------
//  
//  // Compute log(x) safely
//  arma::vec log_x = vec_safe_log(x);
//  
//  // Compute x^α safely
//  arma::vec x_alpha = vec_safe_pow(x, alpha);
//  
//  // Compute x^α * log(x) with check for potential overflow
//  arma::vec x_alpha_log_x = x_alpha % log_x;
//  
//  // ----- Step 2: Compute v_i = 1 - x_i^α and related terms -----
//  
//  // Use log1p and expm1 for better precision near boundaries
//  arma::vec v(n);
//  for (int i = 0; i < n; i++) {
//    // v_i = 1 - x_i^α computed via v_i = -expm1(log(x_i^α)) for better precision
//    if (x_alpha(i) < 0.5) {
//      // Standard calculation is fine for smaller values
//      v(i) = 1.0 - x_alpha(i);
//    } else {
//      // For x_i^α close to 1, use more accurate formula
//      v(i) = -std::expm1(alpha * log_x(i));
//    }
//    
//    // Ensure v is in valid range
//    if (v(i) <= 0.0 || v(i) >= 1.0) {
//      // Apply very cautious clamping only at extremes
//      v(i) = std::max(std::min(v(i), 1.0 - eps), eps);
//    }
//  }
//  
//  // Compute log(v) and v^β terms with proper safeguards
//  arma::vec log_v = vec_safe_log(v);
//  
//  // Compute v^(β-1) with safety for β close to 1
//  arma::vec v_beta_m1;
//  if (std::abs(beta - 1.0) < eps) {
//    // For β ≈ 1, v^(β-1) ≈ 1
//    v_beta_m1.ones(n);
//  } else {
//    v_beta_m1 = vec_safe_pow(v, beta - 1.0);
//  }
//  
//  // Compute v^β safely
//  arma::vec v_beta = v % v_beta_m1;  // More accurate than direct power
//  
//  // Compute v^β * log(v) with check for potential issues
//  arma::vec v_beta_log_v = v_beta % log_v;
//  
//  // ----- Step 3: Compute w_i = 1 - v_i^β safely -----
//  
//  arma::vec w(n);
//  arma::vec log_w(n);
//  
//  for (int i = 0; i < n; i++) {
//    // Compute w_i = 1 - v_i^β more accurately for v_i^β close to 1
//    if (v_beta(i) < 0.5) {
//      w(i) = 1.0 - v_beta(i);
//    } else {
//      // For v_i^β close to 1, use log-domain calculation
//      w(i) = -std::expm1(beta * log_v(i));
//    }
//    
//    // Validate and apply safety bounds
//    if (w(i) <= 0.0 || w(i) >= 1.0) {
//      w(i) = std::max(std::min(w(i), 1.0 - eps), eps);
//    }
//    
//    // Compute log(w) directly for better accuracy
//    log_w(i) = std::log(w(i));
//  }
//  
//  // ----- Step 4: Calculate partial derivatives with careful factoring -----
//  
//  // Compute digamma values once, with checks for large arguments
//  double digamma_gamma, digamma_delta_plus_1, digamma_sum;
//  
//  if (gamma > 1e6 && delta > 1e6) {
//    // For very large parameters, use asymptotic approximation of digamma
//    digamma_gamma = std::log(gamma) - 1.0/(2.0*gamma);
//    digamma_delta_plus_1 = std::log(delta+1.0) - 1.0/(2.0*(delta+1.0));
//    digamma_sum = std::log(gamma+delta+1.0) - 1.0/(2.0*(gamma+delta+1.0));
//  } else {
//    // Use standard digamma for typical values
//    digamma_gamma = R::digamma(gamma);
//    digamma_delta_plus_1 = R::digamma(delta + 1.0);
//    digamma_sum = R::digamma(gamma + delta + 1.0);
//  }
//  
//  // ----- Calculate gradient components -----
//  
//  // d_alpha = n/α + Σᵢlog(xᵢ) - Σᵢ[xᵢ^α * log(xᵢ) * ((β(δ+1)-1)/vᵢ - (γ-1) * β * vᵢ^(β-1) / wᵢ)]
//  double term_beta_delta = beta * (delta + 1.0) - 1.0;
//  double term_gamma = gamma - 1.0;
//  
//  double d_alpha = n / alpha + arma::sum(log_x);
//  
//  for (int i = 0; i < n; i++) {
//    double alpha_term = x_alpha_log_x(i) * (
//      term_beta_delta / v(i) -
//        term_gamma * beta * v_beta_m1(i) / w(i)
//    );
//    
//    // Check for invalid values before adding
//    if (std::isfinite(alpha_term)) {
//      d_alpha -= alpha_term;
//    }
//  }
//  
//  // d_beta = n/β + (δ+1)Σᵢlog(vᵢ) - Σᵢ[(γ-1) * vᵢ^β * log(vᵢ) / wᵢ]
//  double d_beta = n / beta + (delta + 1.0) * arma::sum(log_v);
//  
//  if (term_gamma != 0.0) {  // Skip calculation if γ=1 (term_gamma=0)
//    for (int i = 0; i < n; i++) {
//      double beta_term = term_gamma * v_beta_log_v(i) / w(i);
//      
//      // Check for invalid values before adding
//      if (std::isfinite(beta_term)) {
//        d_beta -= beta_term;
//      }
//    }
//  }
//  
//  // d_gamma = -n[ψ(γ) - ψ(γ+δ+1)] + Σᵢlog(wᵢ)
//  double d_gamma = -n * (digamma_gamma - digamma_sum) + arma::sum(log_w);
//  
//  // d_delta = -n[ψ(δ+1) - ψ(γ+δ+1)] + βΣᵢlog(vᵢ)
//  double d_delta = -n * (digamma_delta_plus_1 - digamma_sum) + beta * arma::sum(log_v);
//  
//  // Final check for NaN/Inf values
//  if (!std::isfinite(d_alpha) || !std::isfinite(d_beta) ||
//      !std::isfinite(d_gamma) || !std::isfinite(d_delta)) {
//      Rcpp::warning("Gradient calculation produced non-finite values in grbkw");
//    return Rcpp::NumericVector(4, R_NaN);
//  }
//  
//  // Since we're optimizing negative log-likelihood, negate all derivatives
//  grad[0] = -d_alpha;
//  grad[1] = -d_beta;
//  grad[2] = -d_gamma;
//  grad[3] = -d_delta;
//  
//  return grad;
// }
// 
// 
// // [[Rcpp::export(.hsbkw_cpp)]]
// Rcpp::NumericMatrix hsbkw(const Rcpp::NumericVector& par, const Rcpp::NumericVector& data) {
//  // Initialize return matrix for error cases
//  Rcpp::NumericMatrix nanH(4, 4);
//  nanH.fill(R_NaN);
//  
//  // Parameter validation
//  if (par.size() < 4) {
//    Rcpp::warning("Parameter vector must have at least 4 elements for BKw");
//    return nanH;
//  }
//  
//  double alpha = par[0];   // Shape parameter α > 0
//  double beta = par[1];    // Shape parameter β > 0
//  double gamma = par[2];   // Shape parameter γ > 0
//  double delta = par[3];   // Shape parameter δ ≥ 0
//  
//  // Enhanced parameter validation with NaN checks
//  if (std::isnan(alpha) || std::isnan(beta) || std::isnan(gamma) || std::isnan(delta) ||
//      alpha <= 0 || beta <= 0 || gamma <= 0 || delta < 0) {
//    Rcpp::warning("Invalid parameters in hsbkw: alpha, beta, gamma must be positive; delta must be non-negative");
//    return nanH;
//  }
//  
//  // Data conversion and validation
//  arma::vec x;
//  try {
//    x = Rcpp::as<arma::vec>(data);
//  } catch (...) {
//    Rcpp::warning("Failed to convert data to arma::vec in hsbkw");
//    return nanH;
//  }
//  
//  // Comprehensive data validation
//  if (x.n_elem == 0 || x.has_nan() || arma::any(x <= 0) || arma::any(x >= 1)) {
//    Rcpp::warning("Data must be strictly in (0,1) and non-empty for hsbkw");
//    return nanH;
//  }
//  
//  int n = x.n_elem;  // Sample size
//  
//  // Initialize Hessian matrix H (of the log-likelihood) as 4x4
//  arma::mat H(4, 4, arma::fill::zeros);
//  
//  // Small constant for numerical stability
//  double eps = std::sqrt(std::numeric_limits<double>::epsilon());
//  
//  // ---------------------------------------------------------------------
//  // STEP 1: Add constant terms (that don't depend on individual data points)
//  // ---------------------------------------------------------------------
//  
//  // Second derivative of n*ln(α) with respect to α: -n/α²
//  H(0, 0) = -n / (alpha * alpha);
//  
//  // Second derivative of n*ln(β) with respect to β: -n/β²
//  H(1, 1) = -n / (beta * beta);
//  
//  // Compute trigamma values with protection for large arguments
//  double trigamma_gamma, trigamma_delta_plus_1, trigamma_sum;
//  
//  if (gamma > 1e6 || delta > 1e6) {
//    // For very large parameters, use asymptotic approximation of trigamma
//    // ψ₁(x) ≈ 1/x + 1/(2x²) + ...
//    trigamma_gamma = 1.0/gamma + 1.0/(2.0*gamma*gamma);
//    trigamma_delta_plus_1 = 1.0/(delta+1.0) + 1.0/(2.0*(delta+1.0)*(delta+1.0));
//    trigamma_sum = 1.0/(gamma+delta+1.0) + 1.0/(2.0*(gamma+delta+1.0)*(gamma+delta+1.0));
//  } else {
//    // Use standard trigamma for typical values
//    trigamma_gamma = R::trigamma(gamma);
//    trigamma_delta_plus_1 = R::trigamma(delta + 1.0);
//    trigamma_sum = R::trigamma(gamma + delta + 1.0);
//  }
//  
//  // Second derivative of -n*ln[B(γ,δ+1)] with respect to γ: -n*[ψ₁(γ) - ψ₁(γ+δ+1)]
//  H(2, 2) = -n * (trigamma_gamma - trigamma_sum);
//  
//  // Second derivative of -n*ln[B(γ,δ+1)] with respect to δ: -n*[ψ₁(δ+1) - ψ₁(γ+δ+1)]
//  H(3, 3) = -n * (trigamma_delta_plus_1 - trigamma_sum);
//  
//  // Mixed derivative (γ,δ): n*ψ₁(γ+δ+1)
//  H(2, 3) = n * trigamma_sum;
//  H(3, 2) = H(2, 3);  // Symmetric matrix
//  
//  // ---------------------------------------------------------------------
//  // STEP 2: Loop through observations to accumulate data-dependent terms
//  // ---------------------------------------------------------------------
//  
//  // Precompute common factor
//  double beta_delta_factor = beta * (delta + 1.0) - 1.0;
//  double gamma_minus_1 = gamma - 1.0;
//  
//  for (int i = 0; i < n; i++) {
//    double xi = x(i);
//    
//    // Compute log(x) safely
//    double ln_xi = safe_log(xi);
//    
//    // ---- Compute x^α and its derivatives more safely ----
//    double A; // A = x^α
//    double dA_dalpha; // dA/dα = x^α * ln(x)
//    double d2A_dalpha2; // d²A/dα² = x^α * (ln(x))²
//    
//    // Use logarithmic domain for more stability
//    double log_A = alpha * ln_xi;
//    
//    if (std::abs(log_A) > 700.0) {
//      // For extreme values, handle specially
//      if (log_A < -700.0) {
//        A = 0.0;
//        dA_dalpha = 0.0;
//        d2A_dalpha2 = 0.0;
//      } else {
//        // Very large - handle with care
//        A = safe_exp(log_A);
//        dA_dalpha = A * ln_xi;
//        d2A_dalpha2 = dA_dalpha * ln_xi;
//      }
//    } else {
//      // Normal range - standard calculation
//      A = std::exp(log_A);
//      dA_dalpha = A * ln_xi;
//      d2A_dalpha2 = dA_dalpha * ln_xi;
//    }
//    
//    // ---- Compute v = 1-x^α and its derivatives safely ----
//    double v; // v = 1 - x^α
//    double ln_v; // ln(v)
//    double dv_dalpha; // dv/dα = -x^α * ln(x)
//    double d2v_dalpha2; // d²v/dα² = -x^α * (ln(x))²
//    
//    if (A > 0.5) {
//      // For A close to 1, use more accurate computation
//      v = -std::expm1(log_A);  // v = 1-e^(α*ln(x)) more accurately
//      dv_dalpha = -dA_dalpha;
//      d2v_dalpha2 = -d2A_dalpha2;
//    } else {
//      // Standard computation is fine for smaller A
//      v = 1.0 - A;
//      dv_dalpha = -dA_dalpha;
//      d2v_dalpha2 = -d2A_dalpha2;
//    }
//    
//    // Safety check - ensure v is strictly in (0,1)
//    if (v <= eps || v >= 1.0 - eps) {
//      v = std::max(std::min(v, 1.0 - eps), eps);
//    }
//    
//    // Compute ln(v) safely
//    ln_v = safe_log(v);
//    
//    // ---- Compute L5 derivatives: (β(δ+1)-1)*ln(v) ----
//    // Second derivative w.r.t. α: (β(δ+1)-1) * [(d²v/dα² * v - (dv/dα)²) / v²]
//    double d2L5_dalpha2 = 0.0;
//    if (std::abs(beta_delta_factor) > eps) {
//      double term = (d2v_dalpha2 * v - dv_dalpha * dv_dalpha) / (v * v);
//      if (std::isfinite(term)) {
//        d2L5_dalpha2 = beta_delta_factor * term;
//      }
//    }
//    
//    // Mixed derivative: d²L5/(dα dβ) = (δ+1) * (dv_dalpha/v)
//    double d2L5_dalpha_dbeta = (delta + 1.0) * (dv_dalpha / v);
//    
//    // Mixed derivative: d²L5/(dα dδ) = β * (dv_dalpha/v)
//    double d2L5_dalpha_ddelta = beta * (dv_dalpha / v);
//    
//    // Mixed derivative: d²L5/(dβ dδ) = ln(v)
//    double d2L5_dbeta_ddelta = ln_v;
//    
//    // ---- Compute w = 1-v^β and its derivatives safely ----
//    double v_beta; // v^β
//    double w; // w = 1 - v^β
//    
//    // Compute v^β safely using log domain when helpful
//    if (beta > 100 || v < 0.01) {
//      double log_v_beta = beta * ln_v;
//      v_beta = safe_exp(log_v_beta);
//    } else {
//      v_beta = safe_pow(v, beta);
//    }
//    
//    // Compute w = 1-v^β carefully
//    if (v_beta > 0.5) {
//      // For v_beta close to 1, use more accurate computation
//      double log_v_beta = beta * ln_v;
//      w = -std::expm1(log_v_beta);  // w = 1-e^(β*ln(v)) more accurately
//    } else {
//      // Standard computation is fine for smaller v_beta
//      w = 1.0 - v_beta;
//    }
//    
//    // Safety check - ensure w is strictly in (0,1)
//    if (w <= eps || w >= 1.0 - eps) {
//      w = std::max(std::min(w, 1.0 - eps), eps);
//    }
//    
//    // Compute ln(w) safely
//    // double ln_w; // ln(w)
//    // double ln_w = safe_log(w);
//    
//    // ---- Derivatives for w ----
//    // dw/dv = -β * v^(β-1)
//    double v_beta_m1 = (beta > 1.0) ? v_beta / v : 1.0;  // v^(β-1)
//    if (beta == 1.0) v_beta_m1 = 1.0;  // Special case
//    
//    double dw_dv = -beta * v_beta_m1;
//    
//    // Chain rule: dw/dα = dw/dv * dv/dα
//    double dw_dalpha = dw_dv * dv_dalpha;
//    
//    // ---- Compute L6 derivatives: (γ-1)*ln(w) ----
//    // Only compute if γ != 1 to avoid unnecessary work
//    double d2L6_dalpha2 = 0.0;
//    double d2L6_dbeta2 = 0.0;
//    double d2L6_dalpha_dbeta = 0.0;
//    double d2L6_dalpha_dgamma = 0.0;
//    double d2L6_dbeta_dgamma = 0.0;
//    
//    if (std::abs(gamma_minus_1) > eps) {
//      // Second derivative of w w.r.t. α
//      double d2w_dalpha2 = -beta * ((beta - 1.0) * safe_pow(v, beta - 2.0) *
//                                    (dv_dalpha * dv_dalpha) +
//                                    v_beta_m1 * d2v_dalpha2);
//      
//      // Second derivative of ln(w) w.r.t. α
//      double term_alpha2 = (d2w_dalpha2 * w - dw_dalpha * dw_dalpha) / (w * w);
//      if (std::isfinite(term_alpha2)) {
//        d2L6_dalpha2 = gamma_minus_1 * term_alpha2;
//      }
//      
//      // Derivative w.r.t. β: d/dβ ln(w)
//      double dw_dbeta = -v_beta * ln_v;
//      
//      // Second derivative of ln(w) w.r.t. β
//      double d2w_dbeta2 = -v_beta * (ln_v * ln_v);
//      double term_beta2 = (d2w_dbeta2 * w - dw_dbeta * dw_dbeta) / (w * w);
//      if (std::isfinite(term_beta2)) {
//        d2L6_dbeta2 = gamma_minus_1 * term_beta2;
//      }
//      
//      // Mixed derivative (α,β)
//      double d_dw_dalpha_dbeta = -v_beta_m1 * (1.0 + beta * ln_v) * dv_dalpha;
//      double mixed_term = (d_dw_dalpha_dbeta / w) - (dw_dalpha * dw_dbeta) / (w * w);
//      if (std::isfinite(mixed_term)) {
//        d2L6_dalpha_dbeta = gamma_minus_1 * mixed_term;
//      }
//      
//      // Mixed derivatives with γ
//      d2L6_dalpha_dgamma = dw_dalpha / w;
//      d2L6_dbeta_dgamma = dw_dbeta / w;
//    }
//    
//    // ---- Accumulate contributions to the Hessian matrix ----
//    // Check each contribution for finiteness before adding
//    
//    // H(α,α): contributions from L5 and L6
//    if (std::isfinite(d2L5_dalpha2)) H(0, 0) += d2L5_dalpha2;
//    if (std::isfinite(d2L6_dalpha2)) H(0, 0) += d2L6_dalpha2;
//    
//    // H(β,β): contributions from L6
//    if (std::isfinite(d2L6_dbeta2)) H(1, 1) += d2L6_dbeta2;
//    
//    // H(α,β): mixed from L5 and L6
//    if (std::isfinite(d2L5_dalpha_dbeta)) H(0, 1) += d2L5_dalpha_dbeta;
//    if (std::isfinite(d2L6_dalpha_dbeta)) H(0, 1) += d2L6_dalpha_dbeta;
//    H(1, 0) = H(0, 1);  // Symmetric
//    
//    // H(α,γ): mixed from L6
//    if (std::isfinite(d2L6_dalpha_dgamma)) H(0, 2) += d2L6_dalpha_dgamma;
//    H(2, 0) = H(0, 2);  // Symmetric
//    
//    // H(α,δ): mixed from L5
//    if (std::isfinite(d2L5_dalpha_ddelta)) H(0, 3) += d2L5_dalpha_ddelta;
//    H(3, 0) = H(0, 3);  // Symmetric
//    
//    // H(β,γ): mixed from L6
//    if (std::isfinite(d2L6_dbeta_dgamma)) H(1, 2) += d2L6_dbeta_dgamma;
//    H(2, 1) = H(1, 2);  // Symmetric
//    
//    // H(β,δ): mixed from L5
//    if (std::isfinite(d2L5_dbeta_ddelta)) H(1, 3) += d2L5_dbeta_ddelta;
//    H(3, 1) = H(1, 3);  // Symmetric
//  }
//  
//  // Final safety check - verify the Hessian is valid
//  if (!H.is_finite()) {
//    Rcpp::warning("Hessian calculation produced non-finite values");
//    return nanH;
//  }
//  
//  // Return the analytic Hessian matrix of the negative log-likelihood
//  return Rcpp::wrap(-H);
// }
