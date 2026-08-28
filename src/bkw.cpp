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
  // Zero-length input: follow R's recycling convention and return an empty
  // vector, as dbeta(numeric(0), 1, 1) does. This also guards the
  // `i % vec.n_elem` recycling below against integer division by zero.
  if (x.n_elem == 0 || alpha_vec.n_elem == 0 || beta_vec.n_elem == 0 || gamma_vec.n_elem == 0 || delta_vec.n_elem == 0) {
    return Rcpp::NumericVector(0);
  }

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
    double log_v = gkw_log1mexp(log_xalpha);
    if (!R_finite(log_v)) {
      continue;
    }
    
    // Term 1: (β(δ+1) - 1) * log(1 - x^α)
    double exponent1 = b * (d + 1.0) - 1.0;
    double term1 = exponent1 * log_v;
    
    // Compute log((1-x^α)^β) = β * log(1-x^α)
    double log_v_beta = b * log_v;
    
    // Compute log(1 - (1-x^α)^β) = log(w) using log1mexp
    double log_w = gkw_log1mexp(log_v_beta);
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
  // Zero-length input: follow R's recycling convention and return an empty
  // vector, as dbeta(numeric(0), 1, 1) does. This also guards the
  // `i % vec.n_elem` recycling below against integer division by zero.
  if (q.n_elem == 0 || alpha_vec.n_elem == 0 || beta_vec.n_elem == 0 || gamma_vec.n_elem == 0 || delta_vec.n_elem == 0) {
    return Rcpp::NumericVector(0);
  }

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
    
    // ---- Cumulative probability, computed entirely in log space ----
    // Every step below is a log(1 - exp(u)), never a 1 - u in linear space.
    // The former chain formed 1 - x^alpha and 1 - (1 - x^alpha)^beta directly:
    // once x^alpha fell below 1.1e-16 the first rounded to exactly 1, the
    // second to exactly 0, and the CDF collapsed to 0 or 1. That is not a
    // last-digit loss -- pekw(5.6e-09, 2, 5, 0.02) returned 0 where the true
    // value is 0.483.
    double log_x_alpha = a * std::log(xx);

    // F = I_z(gamma, delta+1) with z = 1 - (1 - x^alpha)^beta. z is formed
    // with -expm1 so it keeps its digits, and lower_tail/log_p go straight to
    // R::pbeta, which implements both without forming 1 - p or log(p).
    double z = -std::expm1(b * gkw_log1mexp(log_x_alpha));
    res(i) = R::pbeta(z, g, d + 1.0, lower_tail, log_p);
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
  // Zero-length input: follow R's recycling convention and return an empty
  // vector, as dbeta(numeric(0), 1, 1) does. This also guards the
  // `i % vec.n_elem` recycling below against integer division by zero.
  if (p.n_elem == 0 || alpha_vec.n_elem == 0 || beta_vec.n_elem == 0 || gamma_vec.n_elem == 0 || delta_vec.n_elem == 0) {
    return Rcpp::NumericVector(0);
  }

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
  
  // A zero-length parameter cannot be recycled. Match R's convention
  // (rbeta(3, numeric(0), 1) is NA NA NA with a warning) instead of
  // reaching the `i % vec.n_elem` recycling with a zero divisor.
  if (alpha_vec.n_elem == 0 || beta_vec.n_elem == 0 || gamma_vec.n_elem == 0 || delta_vec.n_elem == 0) {
    Rcpp::warning("rbkw: NAs produced");
    return Rcpp::NumericVector(n, NA_REAL);
  }

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
  
  // Constant term: n * [log(alpha) + log(beta) - log(B(gamma, delta+1))]
  double logB = R::lbeta(g, d + 1.0);
  double ll_const = n * (safe_log(a) + safe_log(b) - logB);

  // Term 1: (alpha - 1) * sum log(x)
  arma::vec lx = vec_safe_log(x);
  double sum1 = (a - 1.0) * arma::sum(lx);

  // With lambda = 1 the innermost factor collapses: z = 1 - w = v^beta, so
  // delta*log(z) merges into the log(v) term with exponent beta*(delta+1) - 1.
  double exp1 = b * (d + 1.0) - 1.0;
  double sum2 = 0.0;
  double sum3 = 0.0;

  for (int i = 0; i < n; i++) {
    double log_v = gkw_log1mexp(a * lx(i));
    sum2 += exp1 * log_v;

    double log_w = gkw_log1mexp(b * log_v);
    sum3 += (g - 1.0) * log_w;
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
  arma::vec x = Rcpp::as<arma::vec>(data);

  if (x.n_elem == 0 || x.has_nan() || arma::any(x <= 0) || arma::any(x >= 1)) {
    Rcpp::warning("Data must be strictly in (0,1) and non-empty for grbkw");
    return Rcpp::NumericVector(4, R_NaN);
  }
  
  int n = x.n_elem;
  Rcpp::NumericVector grad(4, 0.0);

  // Digamma values for the Beta-function constant
  double digamma_gamma = R::digamma(gamma);
  double digamma_delta_plus_1 = R::digamma(delta + 1.0);
  double digamma_sum = R::digamma(gamma + delta + 1.0);

  double term_beta_delta = beta * (delta + 1.0) - 1.0;
  double term_gamma = gamma - 1.0;

  double sum_log_x = 0.0, sum_log_v = 0.0, sum_log_w = 0.0;
  double acc_alpha = 0.0, acc_beta = 0.0;

  // Log-space blocks, per observation:
  //   P = dlog(v)/dalpha = -log(x)*exp(log(x^alpha) - log(v))
  //   S = v^beta/w       = exp(beta*log(v) - log(w))
  //   Q = dlog(w)/dalpha = -beta*P*S
  //   R = dlog(w)/dbeta  = -log(v)*S
  for (int i = 0; i < n; i++) {
    double log_xi = std::log(x(i));
    sum_log_x += log_xi;

    double log_x_alpha = alpha * log_xi;
    double log_v = gkw_log1mexp(log_x_alpha);
    sum_log_v += log_v;

    double log_v_beta = beta * log_v;
    double log_w = gkw_log1mexp(log_v_beta);
    sum_log_w += log_w;

    double P = -log_xi * std::exp(log_x_alpha - log_v);
    double S = std::exp(log_v_beta - log_w);
    double Q = -beta * P * S;
    double R = -log_v * S;

    acc_alpha += term_beta_delta * P + term_gamma * Q;
    acc_beta  += term_gamma * R;
  }

  // d_alpha = n/alpha + sum log(x) + (beta(delta+1)-1)*sum P + (gamma-1)*sum Q
  double d_alpha = n / alpha + sum_log_x + acc_alpha;

  // d_beta = n/beta + (delta+1)*sum log(v) + (gamma-1)*sum R
  double d_beta = n / beta + (delta + 1.0) * sum_log_v + acc_beta;

  // d_gamma = -n[psi(gamma) - psi(gamma+delta+1)] + sum log(w)
  double d_gamma = -n * (digamma_gamma - digamma_sum) + sum_log_w;

  // d_delta = -n[psi(delta+1) - psi(gamma+delta+1)] + beta*sum log(v)
  double d_delta = -n * (digamma_delta_plus_1 - digamma_sum) + beta * sum_log_v;

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
  arma::vec x = Rcpp::as<arma::vec>(data);

  if (x.n_elem == 0 || x.has_nan() || arma::any(x <= 0) || arma::any(x >= 1)) {
    Rcpp::warning("Data must be strictly in (0,1) and non-empty for hsbkw");
    return nanH;
  }
  
  int n = x.n_elem;

  // Initialize Hessian matrix
  arma::mat H(4, 4, arma::fill::zeros);

  // Trigamma terms from -n*log(B(gamma, delta+1))
  double trigamma_gamma = R::trigamma(gamma);
  double trigamma_delta_plus_1 = R::trigamma(delta + 1.0);
  double trigamma_sum = R::trigamma(gamma + delta + 1.0);

  H(0, 0) = -n / (alpha * alpha);
  H(1, 1) = -n / (beta * beta);
  H(2, 2) = -n * (trigamma_gamma - trigamma_sum);
  H(3, 3) = -n * (trigamma_delta_plus_1 - trigamma_sum);
  H(2, 3) = n * trigamma_sum;

  double term_beta_delta = beta * (delta + 1.0) - 1.0;
  double term_gamma = gamma - 1.0;

  // Same log-space blocks as the gradient. Note which mixed derivatives keep
  // no vanishing factor and therefore must be accumulated for every parameter
  // value: d2l/dalpha dgamma = sum Q, d2l/dbeta dgamma = sum R,
  // d2l/dalpha ddelta = beta*sum P and d2l/dbeta ddelta = sum log(v).
  for (int i = 0; i < n; i++) {
    double log_xi = std::log(x(i));

    double log_x_alpha = alpha * log_xi;
    double log_v = gkw_log1mexp(log_x_alpha);
    double log_v_beta = beta * log_v;
    double log_w = gkw_log1mexp(log_v_beta);

    double P = -log_xi * std::exp(log_x_alpha - log_v);
    double S = std::exp(log_v_beta - log_w);
    double Q = -beta * P * S;
    double R = -log_v * S;

    double dP_dalpha = P * (log_xi - P);
    double dQ_dalpha = Q * (log_xi - P + beta * P - Q);
    double dQ_dbeta  = -P * S * (1.0 + beta * (log_v - R));
    double dR_dbeta  = R * (log_v - R);

    H(0, 0) += term_beta_delta * dP_dalpha + term_gamma * dQ_dalpha;
    H(0, 1) += (delta + 1.0) * P + term_gamma * dQ_dbeta;
    H(0, 2) += Q;
    H(0, 3) += beta * P;

    H(1, 1) += term_gamma * dR_dbeta;
    H(1, 2) += R;
    H(1, 3) += log_v;
  }

  // Only the upper triangle is accumulated above; mirror it
  H = arma::symmatu(H);

  // Final validity check
  if (!H.is_finite()) {
    Rcpp::warning("Hessian calculation produced non-finite values");
    return nanH;
  }
  
  // Return NEGATIVE Hessian (for minimization)
  return Rcpp::wrap(-H);
}
