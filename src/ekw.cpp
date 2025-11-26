/**
 * @file ekw.cpp
 * @brief Exponentiated Kumaraswamy (EKw) Distribution Functions
 * 
 * @details
 * This file implements the full suite of distribution functions for the
 * three-parameter Exponentiated Kumaraswamy (EKw) distribution, which is a
 * sub-family of the Generalized Kumaraswamy (GKw) distribution obtained
 * by setting γ = 1 and δ = 0.
 * 
 * **Relationship to GKw:**
 * \deqn{EKw(\alpha, \beta, \lambda) = GKw(\alpha, \beta, 1, 0, \lambda)}
 * 
 * The EKw distribution has probability density function:
 * \deqn{
 *   f(x; \alpha, \beta, \lambda) = 
 *   \lambda \alpha \beta x^{\alpha-1} (1-x^\alpha)^{\beta-1}
 *   [1-(1-x^\alpha)^\beta]^{\lambda-1}
 * }
 * for \eqn{x \in (0,1)}.
 * 
 * **Derivation from GKw:**
 * Setting γ=1 and δ=0 in the GKw PDF:
 * - The Beta function term becomes: \eqn{B(1, 0+1) = B(1,1) = 1}
 * - The exponent on the outer bracket: \eqn{\gamma\lambda - 1 = 1\cdot\lambda - 1 = \lambda - 1}
 * - The final term: \eqn{\{1-[...]\}^\delta = \{1-[...]\}^0 = 1}
 * 
 * The cumulative distribution function is:
 * \deqn{
 *   F(x) = [1-(1-x^\alpha)^\beta]^\lambda
 * }
 * 
 * The quantile function (inverse CDF) is:
 * \deqn{
 *   Q(p) = \left\{1 - \left[1 - p^{1/\lambda}\right]^{1/\beta}\right\}^{1/\alpha}
 * }
 * 
 * **Parameter Constraints:**
 * - \eqn{\alpha > 0} (shape parameter)
 * - \eqn{\beta > 0} (shape parameter)
 * - \eqn{\lambda > 0} (exponentiation parameter)
 * 
 * **Special Cases:**
 * | Distribution | Condition | Relation |
 * |--------------|-----------|----------|
 * | Kumaraswamy (Kw) | \eqn{\lambda = 1} | Standard Kumaraswamy |
 * | Generalized Rayleigh | \eqn{\alpha = 2, \beta = 1} | EKw(2, 1, λ) |
 * 
 * **Random Variate Generation:**
 * Uses inverse transform method:
 * 1. Generate \eqn{U \sim Uniform(0,1)}
 * 2. Return \eqn{X = Q(U) = \{1 - [1 - U^{1/\lambda}]^{1/\beta}\}^{1/\alpha}}
 * 
 * **Numerical Stability:**
 * Special attention is given to λ ≈ 1, which can cause numerical cancellation.
 * All computations use log-space arithmetic and numerically stable helper
 * functions from utils.h.
 * 
 * **Implemented Functions:**
 * - dekw(): Probability density function (PDF)
 * - pekw(): Cumulative distribution function (CDF)
 * - qekw(): Quantile function (inverse CDF)
 * - rekw(): Random variate generation
 * - llekw(): Negative log-likelihood for MLE
 * - grekw(): Gradient of negative log-likelihood
 * - hsekw(): Hessian of negative log-likelihood
 * 
 * @author Lopes, J. E.
 * @date 2025-01-07
 * 
 * @see gkw.cpp for the parent distribution
 * @see kkw.cpp for the sister distribution with δ free
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
 * @brief Probability Density Function of the EKw Distribution
 * 
 * Computes the density (or log-density) for the Exponentiated Kumaraswamy
 * distribution at specified quantiles.
 * 
 * @param x Vector of quantiles (values in (0,1))
 * @param alpha Shape parameter vector (α > 0)
 * @param beta Shape parameter vector (β > 0)
 * @param lambda Exponentiation parameter vector (λ > 0)
 * @param log_prob If TRUE, returns log-density; otherwise returns density
 * 
 * @return NumericVector of density values (or log-density if log_prob=TRUE)
 * 
 * @details
 * The log-density is computed as:
 * \deqn{
 *   \log f(x) = \log(\lambda) + \log(\alpha) + \log(\beta)
 *   + (\alpha-1)\log(x) + (\beta-1)\log(1-x^\alpha)
 *   + (\lambda-1)\log(1-(1-x^\alpha)^\beta)
 * }
 * 
 * @note Exported as .dekw_cpp for internal package use
 */
// [[Rcpp::export(.dekw_cpp)]]
Rcpp::NumericVector dekw(
    const arma::vec& x,
    const Rcpp::NumericVector& alpha,
    const Rcpp::NumericVector& beta,
    const Rcpp::NumericVector& lambda,
    bool log_prob = false
) {
  // Convert R vectors to Armadillo vectors
  arma::vec a_vec(alpha.begin(), alpha.size());
  arma::vec b_vec(beta.begin(), beta.size());
  arma::vec l_vec(lambda.begin(), lambda.size());
  
  // Determine output length for recycling
  size_t N = std::max({x.n_elem, a_vec.n_elem, b_vec.n_elem, l_vec.n_elem});
  
  // Initialize result with appropriate default
  arma::vec out(N);
  out.fill(log_prob ? R_NegInf : 0.0);
  
  for (size_t i = 0; i < N; i++) {
    // Extract recycled parameters
    double a = a_vec[i % a_vec.n_elem];
    double b = b_vec[i % b_vec.n_elem];
    double l = l_vec[i % l_vec.n_elem];
    double xx = x[i % x.n_elem];
    
    // Validate parameters
    if (!check_ekw_pars(a, b, l)) {
      continue;
    }
    
    // Check support: x must be in (0, 1)
    if (xx <= 0.0 || xx >= 1.0 || !R_finite(xx)) {
      continue;
    }
    
    // ---- Log-space computation of density ----
    
    // Normalization constant: log(λαβ)
    double ll = safe_log(l);
    double la = safe_log(a);
    double lb = safe_log(b);
    double lx = safe_log(xx);
    
    // Compute log(x^α) = α * log(x)
    double log_xalpha = a * lx;
    
    // Compute log(1 - x^α) using stable log1mexp
    double log_v = log1mexp(log_xalpha);
    if (!R_finite(log_v)) {
      continue;
    }
    
    // Term 1: (β-1) * log(1 - x^α)
    double term2 = (b - 1.0) * log_v;
    
    // Compute log((1-x^α)^β) = β * log(1-x^α)
    double log_v_beta = b * log_v;
    
    // Compute log(1 - (1-x^α)^β) = log(w) using log1mexp
    double log_w = log1mexp(log_v_beta);
    if (!R_finite(log_w)) {
      continue;
    }
    
    // Term 2: (λ-1) * log(w)
    double term3 = (l - 1.0) * log_w;
    
    // Assemble log-density:
    // log(f) = log(λαβ) + (α-1)*log(x) + (β-1)*log(v) + (λ-1)*log(w)
    double log_pdf = ll + la + lb + (a - 1.0) * lx + term2 + term3;
    
    // Validate result
    if (!R_finite(log_pdf)) {
      continue;
    }
    
    // Return appropriate scale
    out(i) = log_prob ? log_pdf : safe_exp(log_pdf);
  }
  
  return Rcpp::NumericVector(out.memptr(), out.memptr() + out.n_elem);
}


// ============================================================================
// CUMULATIVE DISTRIBUTION FUNCTION
// ============================================================================

/**
 * @brief Cumulative Distribution Function of the EKw Distribution
 * 
 * Computes the cumulative probability for the Exponentiated Kumaraswamy
 * distribution at specified quantiles.
 * 
 * @param q Vector of quantiles
 * @param alpha Shape parameter vector (α > 0)
 * @param beta Shape parameter vector (β > 0)
 * @param lambda Exponentiation parameter vector (λ > 0)
 * @param lower_tail If TRUE, returns P(X ≤ q); otherwise P(X > q)
 * @param log_p If TRUE, returns log-probability
 * 
 * @return NumericVector of cumulative probabilities
 * 
 * @details
 * The CDF is computed as:
 * \deqn{F(x) = [1-(1-x^\alpha)^\beta]^\lambda}
 * 
 * @note Exported as .pekw_cpp for internal package use
 */
// [[Rcpp::export(.pekw_cpp)]]
Rcpp::NumericVector pekw(
    const arma::vec& q,
    const Rcpp::NumericVector& alpha,
    const Rcpp::NumericVector& beta,
    const Rcpp::NumericVector& lambda,
    bool lower_tail = true,
    bool log_p = false
) {
  // Convert R vectors to Armadillo vectors
  arma::vec a_vec(alpha.begin(), alpha.size());
  arma::vec b_vec(beta.begin(), beta.size());
  arma::vec l_vec(lambda.begin(), lambda.size());
  
  // Determine output length for recycling
  size_t N = std::max({q.n_elem, a_vec.n_elem, b_vec.n_elem, l_vec.n_elem});
  
  arma::vec out(N);
  
  for (size_t i = 0; i < N; i++) {
    // Extract recycled parameters
    double a = a_vec[i % a_vec.n_elem];
    double b = b_vec[i % b_vec.n_elem];
    double l = l_vec[i % l_vec.n_elem];
    double xx = q[i % q.n_elem];
    
    // Validate parameters
    if (!check_ekw_pars(a, b, l)) {
      out(i) = NA_REAL;
      continue;
    }
    
    // Handle boundary: q ≤ 0
    if (!R_finite(xx) || xx <= 0.0) {
      double val0 = lower_tail ? 0.0 : 1.0;
      out(i) = log_p ? safe_log(val0) : val0;
      continue;
    }
    
    // Handle boundary: q ≥ 1
    if (xx >= 1.0) {
      double val1 = lower_tail ? 1.0 : 0.0;
      out(i) = log_p ? safe_log(val1) : val1;
      continue;
    }
    
    // ---- Compute CDF ----
    
    // Step 1: x^α
    double lx = safe_log(xx);
    double xalpha = safe_exp(a * lx);
    
    // Step 2: 1 - x^α
    double omx = 1.0 - xalpha;
    if (omx <= 0.0) {
      double val1 = lower_tail ? 1.0 : 0.0;
      out(i) = log_p ? safe_log(val1) : val1;
      continue;
    }
    
    // Step 3: (1 - x^α)^β
    double omx_beta = safe_pow(omx, b);
    
    // Step 4: t = 1 - (1 - x^α)^β
    double t = 1.0 - omx_beta;
    if (t <= 0.0) {
      double val0 = lower_tail ? 0.0 : 1.0;
      out(i) = log_p ? safe_log(val0) : val0;
      continue;
    }
    if (t >= 1.0) {
      double val1 = lower_tail ? 1.0 : 0.0;
      out(i) = log_p ? safe_log(val1) : val1;
      continue;
    }
    
    // Step 5: F(x) = t^λ
    double val = safe_pow(t, l);
    
    // Apply tail adjustment
    if (!lower_tail) {
      val = 1.0 - val;
    }
    
    // Apply log transformation
    if (log_p) {
      val = safe_log(val);
    }
    
    out(i) = val;
  }
  
  return Rcpp::NumericVector(out.memptr(), out.memptr() + out.n_elem);
}


// ============================================================================
// QUANTILE FUNCTION
// ============================================================================

/**
 * @brief Quantile Function (Inverse CDF) of the EKw Distribution
 * 
 * Computes quantiles for the Exponentiated Kumaraswamy distribution
 * given probability values.
 * 
 * @param p Vector of probabilities (values in [0,1])
 * @param alpha Shape parameter vector (α > 0)
 * @param beta Shape parameter vector (β > 0)
 * @param lambda Exponentiation parameter vector (λ > 0)
 * @param lower_tail If TRUE, probabilities are P(X ≤ x); otherwise P(X > x)
 * @param log_p If TRUE, probabilities are given as log(p)
 * 
 * @return NumericVector of quantiles
 * 
 * @details
 * The quantile function inverts the CDF:
 * \deqn{Q(p) = \left\{1 - \left[1 - p^{1/\lambda}\right]^{1/\beta}\right\}^{1/\alpha}}
 * 
 * @note Exported as .qekw_cpp for internal package use
 */
// [[Rcpp::export(.qekw_cpp)]]
Rcpp::NumericVector qekw(
    const arma::vec& p,
    const Rcpp::NumericVector& alpha,
    const Rcpp::NumericVector& beta,
    const Rcpp::NumericVector& lambda,
    bool lower_tail = true,
    bool log_p = false
) {
  // Convert R vectors to Armadillo vectors
  arma::vec a_vec(alpha.begin(), alpha.size());
  arma::vec b_vec(beta.begin(), beta.size());
  arma::vec l_vec(lambda.begin(), lambda.size());
  
  // Determine output length for recycling
  size_t N = std::max({p.n_elem, a_vec.n_elem, b_vec.n_elem, l_vec.n_elem});
  
  arma::vec out(N);
  
  for (size_t i = 0; i < N; i++) {
    // Extract recycled parameters
    double a = a_vec[i % a_vec.n_elem];
    double b = b_vec[i % b_vec.n_elem];
    double l = l_vec[i % l_vec.n_elem];
    double pp = p[i % p.n_elem];
    
    // Validate parameters
    if (!check_ekw_pars(a, b, l)) {
      out(i) = NA_REAL;
      continue;
    }
    
    // ---- Convert probability to linear scale ----
    if (log_p) {
      if (pp > 0.0) {
        out(i) = NA_REAL;
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
      out(i) = 0.0;
      continue;
    }
    if (pp >= 1.0) {
      out(i) = 1.0;
      continue;
    }
    
    // ---- Compute quantile via inverse transformations ----
    
    // Step 1: p^(1/λ)
    double step1 = safe_pow(pp, 1.0 / l);
    
    // Step 2: 1 - p^(1/λ)
    double step2 = 1.0 - step1;
    step2 = std::max(0.0, step2);
    
    // Step 3: [1 - p^(1/λ)]^(1/β)
    double step3 = safe_pow(step2, 1.0 / b);
    
    // Step 4: 1 - [1 - p^(1/λ)]^(1/β)
    double step4 = 1.0 - step3;
    step4 = std::max(0.0, step4);
    
    // Step 5: {1 - [1 - p^(1/λ)]^(1/β)}^(1/α)
    double x;
    if (a == 1.0) {
      x = step4;
    } else {
      x = safe_pow(step4, 1.0 / a);
    }
    
    // Clamp to valid support
    x = std::max(0.0, std::min(1.0, x));
    
    out(i) = x;
  }
  
  return Rcpp::NumericVector(out.memptr(), out.memptr() + out.n_elem);
}


// ============================================================================
// RANDOM NUMBER GENERATION
// ============================================================================

/**
 * @brief Random Variate Generation for the EKw Distribution
 * 
 * Generates random samples from the Exponentiated Kumaraswamy distribution
 * using the inverse transform method.
 * 
 * @param n Number of random variates to generate
 * @param alpha Shape parameter vector (α > 0)
 * @param beta Shape parameter vector (β > 0)
 * @param lambda Exponentiation parameter vector (λ > 0)
 * 
 * @return NumericVector of n random variates from EKw distribution
 * 
 * @details
 * Algorithm:
 * 1. Generate U ~ Uniform(0,1)
 * 2. Return X = Q(U) = {1 - [1 - U^(1/λ)]^(1/β)}^(1/α)
 * 
 * @note Exported as .rekw_cpp for internal package use
 */
// [[Rcpp::export(.rekw_cpp)]]
Rcpp::NumericVector rekw(
    int n,
    const Rcpp::NumericVector& alpha,
    const Rcpp::NumericVector& beta,
    const Rcpp::NumericVector& lambda
) {
  if (n <= 0) {
    Rcpp::stop("rekw: n must be positive");
  }
  
  // Convert R vectors to Armadillo vectors
  arma::vec a_vec(alpha.begin(), alpha.size());
  arma::vec b_vec(beta.begin(), beta.size());
  arma::vec l_vec(lambda.begin(), lambda.size());
  
  arma::vec out(n);
  
  for (int i = 0; i < n; i++) {
    // Extract recycled parameters (direct modulo, no intermediate variable)
    double a = a_vec[i % a_vec.n_elem];
    double b = b_vec[i % b_vec.n_elem];
    double l = l_vec[i % l_vec.n_elem];
    
    // Validate parameters
    if (!check_ekw_pars(a, b, l)) {
      out(i) = NA_REAL;
      Rcpp::warning("rekw: invalid parameters at index %d", i + 1);
      continue;
    }
    
    // Generate U ~ Uniform(0,1)
    double U = R::runif(0.0, 1.0);
    
    // Step 1: U^(1/λ)
    double step1 = safe_pow(U, 1.0 / l);
    
    // Step 2: 1 - U^(1/λ)
    double step2 = 1.0 - step1;
    step2 = std::max(0.0, step2);
    
    // Step 3: [1 - U^(1/λ)]^(1/β)
    double step3 = safe_pow(step2, 1.0 / b);
    
    // Step 4: 1 - [1 - U^(1/λ)]^(1/β)
    double step4 = 1.0 - step3;
    step4 = std::max(0.0, step4);
    
    // Step 5: {1 - [1 - U^(1/λ)]^(1/β)}^(1/α)
    double x;
    if (a == 1.0) {
      x = step4;
    } else {
      x = safe_pow(step4, 1.0 / a);
      if (!R_finite(x) || x < 0.0) x = 0.0;
      if (x > 1.0) x = 1.0;
    }
    
    out(i) = x;
  }
  
  return Rcpp::NumericVector(out.memptr(), out.memptr() + out.n_elem);
}


// ============================================================================
// NEGATIVE LOG-LIKELIHOOD FUNCTION
// ============================================================================

/**
 * @brief Negative Log-Likelihood for EKw Distribution
 * 
 * Computes the negative log-likelihood function for parameter estimation
 * via maximum likelihood.
 * 
 * @param par Parameter vector of length 3: (α, β, λ)
 * @param data Vector of observations (must be in (0,1))
 * 
 * @return Negative log-likelihood value (scalar)
 * 
 * @details
 * The log-likelihood for n observations is:
 * \deqn{
 *   \ell(\theta) = n[\ln\lambda + \ln\alpha + \ln\beta]
 *   + (\alpha-1)\sum\ln x_i + (\beta-1)\sum\ln v_i
 *   + (\lambda-1)\sum\ln w_i
 * }
 * where:
 * - \eqn{v_i = 1 - x_i^\alpha}
 * - \eqn{w_i = 1 - v_i^\beta}
 * 
 * Returns +Inf for invalid parameters or data outside (0,1).
 * 
 * **Special handling for λ ≈ 1:**
 * When λ is very close to 1, the term (λ-1)*log(w) can suffer from
 * catastrophic cancellation. Special care is taken in this regime.
 * 
 * @note Exported as .llekw_cpp for internal package use
 */
// [[Rcpp::export(.llekw_cpp)]]
double llekw(const Rcpp::NumericVector& par, const Rcpp::NumericVector& data) {
  // Validate parameter vector length
  if (par.size() < 3) return R_PosInf;
  
  // Extract parameters
  double alpha = par[0];
  double beta = par[1];
  double lambda = par[2];
  
  // Validate parameters using consistent checker
  if (!check_ekw_pars(alpha, beta, lambda)) return R_PosInf;
  
  // Convert and validate data
  arma::vec x = Rcpp::as<arma::vec>(data);
  if (x.n_elem < 1) return R_PosInf;
  if (arma::any(x <= 0.0) || arma::any(x >= 1.0)) return R_PosInf;
  
  int n = x.n_elem;
  
  // Constant term: n * [log(λ) + log(α) + log(β)]
  double log_alpha = safe_log(alpha);
  double log_beta = safe_log(beta);
  double log_lambda = safe_log(lambda);
  double const_term = n * (log_lambda + log_alpha + log_beta);
  
  // Initialize accumulators
  double sum_term1 = 0.0;  // (α-1) * Σlog(x)
  double sum_term2 = 0.0;  // (β-1) * Σlog(v)
  double sum_term3 = 0.0;  // (λ-1) * Σlog(w)
  
  for (int i = 0; i < n; i++) {
    double xi = x(i);
    double log_xi = std::log(xi);
    
    // Term 1: (α-1) * log(x)
    sum_term1 += (alpha - 1.0) * log_xi;
    
    // Compute x^α stably
    double x_alpha;
    if (alpha > 100.0 || (alpha * log_xi < -700.0)) {
      x_alpha = safe_exp(alpha * log_xi);
    } else {
      x_alpha = std::pow(xi, alpha);
    }
    
    // Compute v = 1 - x^α with precision for x^α near 1
    double one_minus_x_alpha;
    double log_one_minus_x_alpha;
    
    if (x_alpha > 0.9995) {
      one_minus_x_alpha = -std::expm1(alpha * log_xi);
      log_one_minus_x_alpha = safe_log(one_minus_x_alpha);
    } else {
      one_minus_x_alpha = 1.0 - x_alpha;
      log_one_minus_x_alpha = safe_log(one_minus_x_alpha);
    }
    
    // Term 2: (β-1) * log(v)
    sum_term2 += (beta - 1.0) * log_one_minus_x_alpha;
    
    // Compute v^β stably
    double v_beta;
    if (beta > 100.0 || (beta * log_one_minus_x_alpha < -700.0)) {
      v_beta = safe_exp(beta * log_one_minus_x_alpha);
    } else {
      v_beta = std::pow(one_minus_x_alpha, beta);
    }
    
    // Compute w = 1 - v^β with precision for v^β near 1
    double one_minus_v_beta;
    double log_one_minus_v_beta;
    
    if (v_beta > 0.9995) {
      one_minus_v_beta = -std::expm1(beta * log_one_minus_x_alpha);
    } else {
      one_minus_v_beta = 1.0 - v_beta;
    }
    
    // Prevent extreme underflow
    if (one_minus_v_beta < 1e-300) {
      one_minus_v_beta = 1e-300;
    }
    
    log_one_minus_v_beta = safe_log(one_minus_v_beta);
    
    // Term 3: (λ-1) * log(w) with special handling for λ ≈ 1
    if (std::abs(lambda - 1.0) < 1e-10) {
      // For λ very close to 1, avoid numerical cancellation
      if (std::abs(lambda - 1.0) > 1e-15) {
        sum_term3 += (lambda - 1.0) * log_one_minus_v_beta;
      }
      // For λ = 1 (machine precision), term is zero
    } else if (lambda > 1000.0 && log_one_minus_v_beta < -0.01) {
      // Special case for very large λ
      double scaled_term = std::max(log_one_minus_v_beta, -700.0 / lambda);
      sum_term3 += (lambda - 1.0) * scaled_term;
    } else {
      // Standard case
      sum_term3 += (lambda - 1.0) * log_one_minus_v_beta;
    }
  }
  
  // Combine all terms
  double loglike = const_term + sum_term1 + sum_term2 + sum_term3;
  
  return -loglike;
}


// ============================================================================
// GRADIENT OF NEGATIVE LOG-LIKELIHOOD
// ============================================================================

/**
 * @brief Gradient of Negative Log-Likelihood for EKw Distribution
 * 
 * Computes the gradient vector of the negative log-likelihood for
 * optimization-based parameter estimation.
 * 
 * @param par Parameter vector of length 3: (α, β, λ)
 * @param data Vector of observations (must be in (0,1))
 * 
 * @return NumericVector of length 3 containing partial derivatives
 *         with respect to (α, β, λ)
 * 
 * @details
 * The gradient components are:
 * - ∂ℓ/∂α = n/α + Σlog(x) - Σ[x^α log(x) * ((β-1)/v - (λ-1)βv^(β-1)/w)]
 * - ∂ℓ/∂β = n/β + Σlog(v) - (λ-1)Σ[v^β log(v)/w]
 * - ∂ℓ/∂λ = n/λ + Σlog(w)
 * 
 * @note Exported as .grekw_cpp for internal package use
 */
// [[Rcpp::export(.grekw_cpp)]]
Rcpp::NumericVector grekw(const Rcpp::NumericVector& par, const Rcpp::NumericVector& data) {
  // Validate parameter vector length
  if (par.size() < 3) {
    return Rcpp::NumericVector(3, R_NaN);
  }
  
  // Extract parameters
  double alpha = par[0];
  double beta = par[1];
  double lambda = par[2];
  
  // Validate parameters using consistent checker
  if (!check_ekw_pars(alpha, beta, lambda)) {
    return Rcpp::NumericVector(3, R_NaN);
  }
  
  // Convert and validate data
  arma::vec x = Rcpp::as<arma::vec>(data);
  if (x.n_elem < 1 || arma::any(x <= 0) || arma::any(x >= 1)) {
    return Rcpp::NumericVector(3, R_NaN);
  }
  
  int n = x.n_elem;
  Rcpp::NumericVector grad(3, 0.0);
  
  // Numerical stability constants
  const double min_val = 1e-10;
  
  // Initialize gradient accumulators
  double d_alpha = n / alpha;
  double d_beta = n / beta;
  double d_lambda = n / lambda;
  
  for (int i = 0; i < n; i++) {
    double xi = x(i);
    double log_xi = std::log(xi);
    d_alpha += log_xi;
    
    // Compute x^α stably
    double x_alpha;
    if (alpha > 100.0 || (alpha * log_xi < -700.0)) {
      x_alpha = safe_exp(alpha * log_xi);
    } else {
      x_alpha = std::pow(xi, alpha);
    }
    
    // Compute v = 1 - x^α with precision
    double v;
    if (x_alpha > 0.9995) {
      v = -std::expm1(alpha * log_xi);
    } else {
      v = 1.0 - x_alpha;
    }
    v = std::max(v, min_val);
    double log_v = safe_log(v);
    d_beta += log_v;
    
    // Compute v^β and v^(β-1) stably
    double v_beta, v_beta_m1;
    if (beta > 100.0 || (beta * log_v < -700.0)) {
      double log_v_beta = beta * log_v;
      v_beta = safe_exp(log_v_beta);
      v_beta_m1 = safe_exp((beta - 1.0) * log_v);
    } else {
      v_beta = std::pow(v, beta);
      v_beta_m1 = std::pow(v, beta - 1.0);
    }
    
    // Compute w = 1 - v^β with precision
    double w;
    if (v_beta > 0.9995) {
      w = -std::expm1(beta * log_v);
    } else {
      w = 1.0 - v_beta;
    }
    w = std::max(w, min_val);
    double log_w = safe_log(w);
    d_lambda += log_w;
    
    // ---- Alpha gradient component ----
    double x_alpha_log_x = x_alpha * log_xi;
    
    // Calculate (β-1)/v term
    double alpha_term1 = 0.0;
    if (std::abs(beta - 1.0) > 1e-14) {
      alpha_term1 = (beta - 1.0) / v;
    }
    
    // Calculate (λ-1) * β * v^(β-1) / w term
    double alpha_term2 = 0.0;
    if (std::abs(lambda - 1.0) > 1e-14) {
      double lambda_factor = lambda - 1.0;
      if (lambda > 1000.0) {
        lambda_factor = std::min(lambda_factor, 1000.0);
      }
      alpha_term2 = lambda_factor * beta * v_beta_m1 / w;
    }
    
    d_alpha -= x_alpha_log_x * (alpha_term1 - alpha_term2);
    
    // ---- Beta gradient component ----
    double beta_term = 0.0;
    if (std::abs(lambda - 1.0) > 1e-14) {
      double lambda_factor = lambda - 1.0;
      if (lambda > 1000.0) {
        lambda_factor = std::min(lambda_factor, 1000.0);
      }
      beta_term = v_beta * log_v * lambda_factor / w;
    }
    
    d_beta -= beta_term;
  }
  
  // Return NEGATIVE gradient (for minimization)
  grad[0] = -d_alpha;
  grad[1] = -d_beta;
  grad[2] = -d_lambda;
  
  return grad;
}


// ============================================================================
// HESSIAN OF NEGATIVE LOG-LIKELIHOOD
// ============================================================================

/**
 * @brief Hessian Matrix of Negative Log-Likelihood for EKw Distribution
 * 
 * Computes the Hessian matrix (matrix of second partial derivatives) of
 * the negative log-likelihood for standard error estimation and
 * optimization algorithms.
 * 
 * @param par Parameter vector of length 3: (α, β, λ)
 * @param data Vector of observations (must be in (0,1))
 * 
 * @return NumericMatrix of dimension 3×3 containing the Hessian
 * 
 * @details
 * Computes analytical second derivatives. The Hessian is symmetric.
 * Parameter ordering: (α, β, λ) → indices (0, 1, 2).
 * 
 * Returns NaN matrix for invalid inputs.
 * 
 * @note Exported as .hsekw_cpp for internal package use
 */
// [[Rcpp::export(.hsekw_cpp)]]
Rcpp::NumericMatrix hsekw(const Rcpp::NumericVector& par, const Rcpp::NumericVector& data) {
  // Initialize NaN matrix for error cases
  Rcpp::NumericMatrix nanH(3, 3);
  nanH.fill(R_NaN);
  
  // Validate parameter vector length
  if (par.size() < 3) {
    return nanH;
  }
  
  // Extract parameters
  double alpha = par[0];
  double beta = par[1];
  double lambda = par[2];
  
  // Validate parameters using consistent checker
  if (!check_ekw_pars(alpha, beta, lambda)) {
    return nanH;
  }
  
  // Convert and validate data
  arma::vec x = Rcpp::as<arma::vec>(data);
  if (x.n_elem < 1 || arma::any(x <= 0) || arma::any(x >= 1)) {
    return nanH;
  }
  
  int n = x.n_elem;
  
  // Initialize Hessian matrix
  arma::mat H(3, 3, arma::fill::zeros);
  
  // Numerical stability constants
  const double min_v = 1e-10;
  const double min_w = 1e-10;
  
  // Constant diagonal terms
  H(0, 0) = -n / (alpha * alpha);   // -n/α²
  H(1, 1) = -n / (beta * beta);     // -n/β²
  H(2, 2) = -n / (lambda * lambda); // -n/λ²
  
  // Special handling for λ ≈ 1
  bool lambda_near_one = std::abs(lambda - 1.0) < 1e-8;
  
  for (int i = 0; i < n; i++) {
    double xi = x(i);
    double log_xi = safe_log(xi);
    
    // ---- Compute A = x^α and derivatives ----
    double A, dA_dalpha, d2A_dalpha2;
    if (alpha > 100.0 || (alpha * log_xi < -700.0)) {
      double log_A = alpha * log_xi;
      A = safe_exp(log_A);
      dA_dalpha = A * log_xi;
      d2A_dalpha2 = A * log_xi * log_xi;
    } else {
      A = std::pow(xi, alpha);
      dA_dalpha = A * log_xi;
      d2A_dalpha2 = A * log_xi * log_xi;
    }
    
    // ---- Compute v = 1 - A and derivatives ----
    double v;
    if (A > 0.9995) {
      v = -std::expm1(alpha * log_xi);
    } else {
      v = 1.0 - A;
    }
    v = std::max(v, min_v);
    double log_v = safe_log(v);
    
    double dv_dalpha = -dA_dalpha;
    double d2v_dalpha2 = -d2A_dalpha2;
    
    // ---- Derivatives for L5: (β-1)*log(v) ----
    double d2L5_dalpha2 = 0.0;
    double d2L5_dalpha_dbeta = 0.0;
    
    if (beta != 1.0) {
      double v_squared = std::max(v * v, 1e-200);
      d2L5_dalpha2 = (beta - 1.0) * ((d2v_dalpha2 * v - dv_dalpha * dv_dalpha) / v_squared);
      d2L5_dalpha_dbeta = dv_dalpha / v;
    }
    
    // ---- Compute w = 1 - v^β and derivatives ----
    double v_beta, v_beta_m1, v_beta_m2;
    if (beta > 100.0 || (beta * log_v < -700.0)) {
      v_beta = safe_exp(beta * log_v);
      v_beta_m1 = safe_exp((beta - 1.0) * log_v);
      v_beta_m2 = safe_exp((beta - 2.0) * log_v);
    } else {
      v_beta = std::pow(v, beta);
      v_beta_m1 = std::pow(v, beta - 1.0);
      v_beta_m2 = std::pow(v, beta - 2.0);
    }
    
    double w;
    if (v_beta > 0.9995) {
      w = -std::expm1(beta * log_v);
    } else {
      w = 1.0 - v_beta;
    }
    w = std::max(w, min_w);
    double w_squared = std::max(w * w, 1e-200);
    
    // First derivatives of w
    double dw_dv = -beta * v_beta_m1;
    double dw_dalpha = dw_dv * dv_dalpha;
    double dw_dbeta = -v_beta * log_v;
    
    // Second derivatives of w
    double d2w_dalpha2 = -beta * ((beta - 1.0) * v_beta_m2 * (dv_dalpha * dv_dalpha) +
                                  v_beta_m1 * d2v_dalpha2);
    double d2w_dbeta2 = -v_beta * (log_v * log_v);
    double d_dw_dalpha_dbeta = -v_beta_m1 * (1.0 + beta * log_v) * dv_dalpha;
    
    // ---- Derivatives for L6: (λ-1)*log(w) ----
    double d2L6_dalpha2 = 0.0;
    double d2L6_dbeta2 = 0.0;
    double d2L6_dalpha_dbeta = 0.0;
    double d2L6_dalpha_dlambda = 0.0;
    double d2L6_dbeta_dlambda = 0.0;
    
    if (lambda_near_one) {
      // For λ ≈ 1, handle carefully
      if (std::abs(lambda - 1.0) > 1e-15) {
        double factor = lambda - 1.0;
        d2L6_dalpha2 = factor * ((d2w_dalpha2 * w - (dw_dalpha * dw_dalpha)) / w_squared);
        d2L6_dbeta2 = factor * ((d2w_dbeta2 * w - (dw_dbeta * dw_dbeta)) / w_squared);
        d2L6_dalpha_dbeta = factor * ((d_dw_dalpha_dbeta / w) - (dw_dalpha * dw_dbeta) / w_squared);
      }
      d2L6_dalpha_dlambda = dw_dalpha / w;
      d2L6_dbeta_dlambda = dw_dbeta / w;
    } else {
      // Standard case
      d2L6_dalpha2 = (lambda - 1.0) * ((d2w_dalpha2 * w - (dw_dalpha * dw_dalpha)) / w_squared);
      d2L6_dbeta2 = (lambda - 1.0) * ((d2w_dbeta2 * w - (dw_dbeta * dw_dbeta)) / w_squared);
      d2L6_dalpha_dbeta = (lambda - 1.0) * ((d_dw_dalpha_dbeta / w) - (dw_dalpha * dw_dbeta) / w_squared);
      d2L6_dalpha_dlambda = dw_dalpha / w;
      d2L6_dbeta_dlambda = dw_dbeta / w;
    }
    
    // Clamp for large λ
    if (lambda > 1000.0) {
      double max_val = 1000.0;
      d2L6_dalpha2 = std::min(std::max(d2L6_dalpha2, -max_val), max_val);
      d2L6_dbeta2 = std::min(std::max(d2L6_dbeta2, -max_val), max_val);
      d2L6_dalpha_dbeta = std::min(std::max(d2L6_dalpha_dbeta, -max_val), max_val);
    }
    
    // ---- Accumulate Hessian contributions ----
    H(0, 0) += d2L5_dalpha2 + d2L6_dalpha2;
    H(0, 1) += d2L5_dalpha_dbeta + d2L6_dalpha_dbeta;
    H(1, 0) = H(0, 1);
    H(1, 1) += d2L6_dbeta2;
    H(0, 2) += d2L6_dalpha_dlambda;
    H(2, 0) = H(0, 2);
    H(1, 2) += d2L6_dbeta_dlambda;
    H(2, 1) = H(1, 2);
  }
  
  // Enforce perfect symmetry
  for (int i = 0; i < 3; i++) {
    for (int j = i + 1; j < 3; j++) {
      double avg = (H(i, j) + H(j, i)) / 2.0;
      H(i, j) = H(j, i) = avg;
    }
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
// EXPONENTIATED KUMARASWAMY (EKw) DISTRIBUTION
// ----------------------------------------------------------------------------
// 
// We interpret EKw(α, β, λ) as the GKw distribution with gamma=1 and delta=0.
// 
// * PDF:
// f(x) = λ * α * β * x^(α-1) * (1 - x^α)^(β - 1) *
// [1 - (1 - x^α)^β ]^(λ - 1),    for 0 < x < 1.
// 
// * CDF:
// F(x) = [1 - (1 - x^α)^β ]^λ,         for 0 < x < 1.
// 
// * QUANTILE:
// If p = F(x) = [1 - (1 - x^α)^β]^λ, then
// p^(1/λ) = 1 - (1 - x^α)^β
// (1 - x^α)^β = 1 - p^(1/λ)
// x^α = 1 - [1 - p^(1/λ)]^(1/β)
// x = {1 - [1 - p^(1/λ)]^(1/β)}^(1/α).
// 
// * RNG:
// We can generate via the quantile method: U ~ Uniform(0,1), X= Q(U).
// 
// X = Q(U) = {1 - [1 - U^(1/λ)]^(1/β)}^(1/α).
// 
// * LOG-LIKELIHOOD:
// The log-density for observation x in (0,1):
// log f(x) = log(λ) + log(α) + log(β)
// + (α-1)*log(x)
// + (β-1)*log(1 - x^α)
// + (λ-1)*log(1 - (1 - x^α)^β).
// 
// Summation of log-likelihood over all x. We return negative of that for 'llekw'.*/
// 
// 
// // [[Rcpp::export(.dekw_cpp)]]
// Rcpp::NumericVector dekw(
//    const arma::vec& x,
//    const Rcpp::NumericVector& alpha,
//    const Rcpp::NumericVector& beta,
//    const Rcpp::NumericVector& lambda,
//    bool log_prob = false
// ) {
//  arma::vec a_vec(alpha.begin(), alpha.size());
//  arma::vec b_vec(beta.begin(), beta.size());
//  arma::vec l_vec(lambda.begin(), lambda.size());
//  
//  size_t N = std::max({ x.n_elem, a_vec.n_elem, b_vec.n_elem, l_vec.n_elem });
//  arma::vec out(N);
//  out.fill(log_prob ? R_NegInf : 0.0);
//  
//  for (size_t i=0; i<N; i++) {
//    double a = a_vec[i % a_vec.n_elem];
//    double b = b_vec[i % b_vec.n_elem];
//    double l = l_vec[i % l_vec.n_elem];
//    double xx = x[i % x.n_elem];
//    
//    if (!check_ekw_pars(a, b, l)) {
//      // invalid => PDF=0 or logPDF=-Inf
//      continue;
//    }
//    // domain check
//    if (xx <= 0.0 || xx >= 1.0 || !R_finite(xx)) {
//      continue;
//    }
//    
//    // log f(x) = log(lambda) + log(a) + log(b) + (a-1)*log(x)
//    //            + (b-1)*log(1 - x^a)
//    //            + (lambda-1)*log(1 - (1 - x^a)^b)
//    double ll  = std::log(l);
//    double la  = std::log(a);
//    double lb  = std::log(b);
//    double lx  = std::log(xx);
//    
//    double xalpha = a*lx; // log(x^a)
//    double log_1_xalpha = log1mexp(xalpha); // log(1 - x^a)
//    if (!R_finite(log_1_xalpha)) {
//      continue;
//    }
//    
//    double term2 = (b - 1.0)*log_1_xalpha; // (b-1)* log(1 - x^a)
//    
//    // let A= (1 - x^a)^b => logA= b*log_1_xalpha
//    double logA = b*log_1_xalpha;
//    double log_1_minus_A = log1mexp(logA); // log(1 - A)
//    if (!R_finite(log_1_minus_A)) {
//      continue;
//    }
//    double term3 = (l - 1.0)* log_1_minus_A;
//    
//    double log_pdf = ll + la + lb
//    + (a - 1.0)* lx
//    + term2
//    + term3;
//    
//    if (log_prob) {
//      out(i)= log_pdf;
//    } else {
//      out(i)= std::exp(log_pdf);
//    }
//  }
//  
//  return Rcpp::NumericVector(out.memptr(), out.memptr() + out.n_elem);
// }
// 
// // -----------------------------------------------------------------------------
// // 2) pekw: CDF of Exponentiated Kumaraswamy
// // -----------------------------------------------------------------------------
// 
// 
// // [[Rcpp::export(.pekw_cpp)]]
// Rcpp::NumericVector pekw(
//    const arma::vec& q,
//    const Rcpp::NumericVector& alpha,
//    const Rcpp::NumericVector& beta,
//    const Rcpp::NumericVector& lambda,
//    bool lower_tail = true,
//    bool log_p = false
// ) {
//  arma::vec a_vec(alpha.begin(), alpha.size());
//  arma::vec b_vec(beta.begin(), beta.size());
//  arma::vec l_vec(lambda.begin(), lambda.size());
//  
//  size_t N = std::max({ q.n_elem, a_vec.n_elem, b_vec.n_elem, l_vec.n_elem });
//  arma::vec out(N);
//  
//  for (size_t i=0; i<N; i++) {
//    double a = a_vec[i % a_vec.n_elem];
//    double b = b_vec[i % b_vec.n_elem];
//    double l = l_vec[i % l_vec.n_elem];
//    double xx = q[i % q.n_elem];
//    
//    if (!check_ekw_pars(a, b, l)) {
//      out(i)= NA_REAL;
//      continue;
//    }
//    
//    // boundary
//    if (!R_finite(xx) || xx <= 0.0) {
//      double val0 = (lower_tail ? 0.0 : 1.0);
//      out(i) = (log_p ? std::log(val0) : val0);
//      continue;
//    }
//    if (xx >= 1.0) {
//      double val1 = (lower_tail ? 1.0 : 0.0);
//      out(i) = (log_p ? std::log(val1) : val1);
//      continue;
//    }
//    
//    // F(x)= [1 - (1 - x^a)^b]^lambda
//    double lx = std::log(xx);
//    double xalpha = std::exp(a*lx);
//    double omx = 1.0 - xalpha;         // (1 - x^α)
//    if (omx <= 0.0) {
//      // => F=1
//      double val1 = (lower_tail ? 1.0 : 0.0);
//      out(i) = (log_p ? std::log(val1) : val1);
//      continue;
//    }
//    double t = 1.0 - std::pow(omx, b);
//    if (t <= 0.0) {
//      // => F=0
//      double val0 = (lower_tail ? 0.0 : 1.0);
//      out(i) = (log_p ? std::log(val0) : val0);
//      continue;
//    }
//    if (t >= 1.0) {
//      // => F=1
//      double val1 = (lower_tail ? 1.0 : 0.0);
//      out(i) = (log_p ? std::log(val1) : val1);
//      continue;
//    }
//    double val = std::pow(t, l);
//    // F(x)=val => if not lower tail => 1-val
//    if (!lower_tail) {
//      val = 1.0 - val;
//    }
//    if (log_p) {
//      val = std::log(val);
//    }
//    out(i) = val;
//  }
//  
//  return Rcpp::NumericVector(out.memptr(), out.memptr() + out.n_elem);
// }
// 
// // -----------------------------------------------------------------------------
// // 3) qekw: Quantile of Exponentiated Kumaraswamy
// // -----------------------------------------------------------------------------
// 
// // [[Rcpp::export(.qekw_cpp)]]
// Rcpp::NumericVector qekw(
//    const arma::vec& p,
//    const Rcpp::NumericVector& alpha,
//    const Rcpp::NumericVector& beta,
//    const Rcpp::NumericVector& lambda,
//    bool lower_tail = true,
//    bool log_p = false
// ) {
//  arma::vec a_vec(alpha.begin(), alpha.size());
//  arma::vec b_vec(beta.begin(), beta.size());
//  arma::vec l_vec(lambda.begin(), lambda.size());
//  
//  size_t N = std::max({ p.n_elem, a_vec.n_elem, b_vec.n_elem, l_vec.n_elem });
//  arma::vec out(N);
//  
//  for (size_t i=0; i<N; i++){
//    double a = a_vec[i % a_vec.n_elem];
//    double b = b_vec[i % b_vec.n_elem];
//    double l = l_vec[i % l_vec.n_elem];
//    double pp = p[i % p.n_elem];
//    
//    if (!check_ekw_pars(a, b, l)) {
//      out(i) = NA_REAL;
//      continue;
//    }
//    
//    // handle log_p
//    if (log_p) {
//      if (pp > 0.0) {
//        // log(p)>0 => p>1 => invalid
//        out(i) = NA_REAL;
//        continue;
//      }
//      pp = std::exp(pp);
//    }
//    // handle tail
//    if (!lower_tail) {
//      pp = 1.0 - pp;
//    }
//    
//    // boundaries
//    if (pp <= 0.0) {
//      out(i) = 0.0;
//      continue;
//    }
//    if (pp >= 1.0) {
//      out(i) = 1.0;
//      continue;
//    }
//    
//    // Q(p)= {1 - [1 - p^(1/λ)]^(1/β)}^(1/α)
//    double step1 = std::pow(pp, 1.0/l);          // p^(1/λ)
//    double step2 = 1.0 - step1;                  // 1 - p^(1/λ)
//    if (step2 < 0.0) step2 = 0.0;
//    double step3 = std::pow(step2, 1.0/b);       // [1 - p^(1/λ)]^(1/β)
//    double step4 = 1.0 - step3;                  // 1 - ...
//    if (step4 < 0.0) step4 = 0.0;
//    
//    double x;
//    if (a == 1.0) {
//      x = step4;
//    } else {
//      x = std::pow(step4, 1.0/a);
//      if (x < 0.0) x = 0.0;
//      if (x > 1.0) x = 1.0;
//    }
//    
//    out(i) = x;
//  }
//  
//  return Rcpp::NumericVector(out.memptr(), out.memptr() + out.n_elem);
// }
// 
// 
// // -----------------------------------------------------------------------------
// // 4) rekw: RNG for Exponentiated Kumaraswamy
// // -----------------------------------------------------------------------------
// 
// 
// // [[Rcpp::export(.rekw_cpp)]]
// Rcpp::NumericVector rekw(
//    int n,
//    const Rcpp::NumericVector& alpha,
//    const Rcpp::NumericVector& beta,
//    const Rcpp::NumericVector& lambda
// ) {
//  if (n <= 0) {
//    Rcpp::stop("rekw: n must be positive");
//  }
//  
//  arma::vec a_vec(alpha.begin(), alpha.size());
//  arma::vec b_vec(beta.begin(), beta.size());
//  arma::vec l_vec(lambda.begin(), lambda.size());
//  
//  size_t k = std::max({ a_vec.n_elem, b_vec.n_elem, l_vec.n_elem });
//  arma::vec out(n);
//  
//  for (int i=0; i<n; i++){
//    size_t idx = i % k;
//    double a = a_vec[idx % a_vec.n_elem];
//    double b = b_vec[idx % b_vec.n_elem];
//    double l = l_vec[idx % l_vec.n_elem];
//    
//    if (!check_ekw_pars(a, b, l)) {
//      out(i) = NA_REAL;
//      Rcpp::warning("rekw: invalid parameters at index %d", i+1);
//      continue;
//    }
//    
//    double U = R::runif(0.0, 1.0);
//    // X = Q(U)
//    double step1 = std::pow(U, 1.0/l);
//    double step2 = 1.0 - step1;
//    if (step2 < 0.0) step2 = 0.0;
//    double step3 = std::pow(step2, 1.0/b);
//    double step4 = 1.0 - step3;
//    if (step4 < 0.0) step4 = 0.0;
//    
//    double x;
//    if (a == 1.0) {
//      x = step4;
//    } else {
//      x = std::pow(step4, 1.0/a);
//      if (!R_finite(x) || x < 0.0) x = 0.0;
//      if (x > 1.0) x = 1.0;
//    }
//    
//    out(i) = x;
//  }
//  
//  return Rcpp::NumericVector(out.memptr(), out.memptr() + out.n_elem);
// }
// 
// 
// // -----------------------------------------------------------------------------
// // 5) llekw: Negative Log-Likelihood of EKw
// // -----------------------------------------------------------------------------
// 
// 
// // [[Rcpp::export(.llekw_cpp)]]
// double llekw(const Rcpp::NumericVector& par,
//             const Rcpp::NumericVector& data) {
//  // Parameter validation
//  if (par.size() < 3) return R_PosInf;
//  
//  double alpha = par[0];
//  double beta = par[1];
//  double lambda = par[2];
//  
//  if (!check_ekw_pars(alpha, beta, lambda)) return R_PosInf;
//  
//  arma::vec x = Rcpp::as<arma::vec>(data);
//  if (x.n_elem < 1) return R_PosInf;
//  if (arma::any(x <= 0.0) || arma::any(x >= 1.0)) return R_PosInf;
//  
//  int n = x.n_elem;
//  
//  // Calculate log parameters for better precision
//  double log_alpha = safe_log(alpha);
//  double log_beta = safe_log(beta);
//  double log_lambda = safe_log(lambda);
//  
//  // Constant term
//  double const_term = n * (log_lambda + log_alpha + log_beta);
//  
//  // Initialize sum terms
//  double sum_term1 = 0.0; // (alpha-1) * sum(log(x))
//  double sum_term2 = 0.0; // (beta-1) * sum(log(1-x^alpha))
//  double sum_term3 = 0.0; // (lambda-1) * sum(log(1-(1-x^alpha)^beta))
//  
//  for (int i = 0; i < n; i++) {
//    double xi = x(i);
//    double log_xi = std::log(xi);
//    
//    // Term 1: (alpha-1) * log(x)
//    sum_term1 += (alpha - 1.0) * log_xi;
//    
//    // Stable calculation of x^alpha for large alpha
//    double x_alpha;
//    if (alpha > 100.0 || (alpha * log_xi < -700.0)) {
//      x_alpha = std::exp(alpha * log_xi);
//    } else {
//      x_alpha = std::pow(xi, alpha);
//    }
//    
//    // Stable calculation of (1-x^alpha) and log(1-x^alpha)
//    double one_minus_x_alpha;
//    double log_one_minus_x_alpha;
//    
//    if (x_alpha > 0.9995) {
//      // For x^alpha close to 1, use complement approach
//      one_minus_x_alpha = -std::expm1(alpha * log_xi);
//      log_one_minus_x_alpha = std::log(one_minus_x_alpha);
//    } else {
//      one_minus_x_alpha = 1.0 - x_alpha;
//      log_one_minus_x_alpha = std::log(one_minus_x_alpha);
//    }
//    
//    // Term 2: (beta-1) * log(1-x^alpha)
//    sum_term2 += (beta - 1.0) * log_one_minus_x_alpha;
//    
//    // Stable calculation of (1-x^alpha)^beta
//    double v_beta;
//    if (beta > 100.0 || (beta * log_one_minus_x_alpha < -700.0)) {
//      v_beta = std::exp(beta * log_one_minus_x_alpha);
//    } else {
//      v_beta = std::pow(one_minus_x_alpha, beta);
//    }
//    
//    // Stable calculation of [1-(1-x^alpha)^beta]
//    double one_minus_v_beta;
//    double log_one_minus_v_beta;
//    
//    if (v_beta > 0.9995) {
//      // When (1-x^alpha)^beta is close to 1
//      one_minus_v_beta = -std::expm1(beta * log_one_minus_x_alpha);
//    } else {
//      one_minus_v_beta = 1.0 - v_beta;
//    }
//    
//    // CRITICAL: Handle extreme lambda values
//    // Prevent underflow when one_minus_v_beta is small and lambda is large
//    if (one_minus_v_beta < 1e-300) {
//      one_minus_v_beta = 1e-300;
//    }
//    
//    log_one_minus_v_beta = std::log(one_minus_v_beta);
//    
//    // Term 3: (lambda-1) * log(1-(1-x^alpha)^beta)
//    // Special handling for lambda near 1
//    if (std::abs(lambda - 1.0) < 1e-10) {
//      // For lambda ≈ 1, avoid numerical cancellation
//      if (std::abs(lambda - 1.0) > 1e-15) {
//        sum_term3 += (lambda - 1.0) * log_one_minus_v_beta;
//      }
//      // For lambda = 1 (machine precision), term is zero
//    } else if (lambda > 1000.0 && log_one_minus_v_beta < -0.01) {
//      // Special case for very large lambda
//      double scaled_term = std::max(log_one_minus_v_beta, -700.0 / lambda);
//      sum_term3 += (lambda - 1.0) * scaled_term;
//    } else {
//      // Standard case
//      sum_term3 += (lambda - 1.0) * log_one_minus_v_beta;
//    }
//  }
//  
//  // Combine all terms
//  double loglike = const_term + sum_term1 + sum_term2 + sum_term3;
//  
//  return -loglike;
// }
// 
// 
// 
// // -----------------------------------------------------------------------------
// // 6) grekw: Gradient of Negative Log-Likelihood of EKw
// // -----------------------------------------------------------------------------
// 
// 
// // [[Rcpp::export(.grekw_cpp)]]
// Rcpp::NumericVector grekw(const Rcpp::NumericVector& par, const Rcpp::NumericVector& data) {
//  // Parameter validation
//  if (par.size() < 3) {
//    Rcpp::NumericVector grad(3, R_NaN);
//    return grad;
//  }
//  
//  double alpha = par[0];
//  double beta = par[1];
//  double lambda = par[2];
//  
//  if (alpha <= 0 || beta <= 0 || lambda <= 0) {
//    Rcpp::NumericVector grad(3, R_NaN);
//    return grad;
//  }
//  
//  arma::vec x = Rcpp::as<arma::vec>(data);
//  if (x.n_elem < 1 || arma::any(x <= 0) || arma::any(x >= 1)) {
//    Rcpp::NumericVector grad(3, R_NaN);
//    return grad;
//  }
//  
//  int n = x.n_elem;
//  Rcpp::NumericVector grad(3, 0.0);
//  
//  // Constants for numerical stability
//  const double min_val = 1e-10;
//  const double exp_threshold = -700.0;
//  
//  // Initialize component accumulators
//  double d_alpha = n / alpha;
//  double d_beta = n / beta;
//  double d_lambda = n / lambda;
//  
//  for (int i = 0; i < n; i++) {
//    double xi = x(i);
//    double log_xi = std::log(xi);
//    d_alpha += log_xi;  // Accumulate (α-1) * log(x_i) term
//    
//    // Compute x^α stably (use log domain for large alpha)
//    double x_alpha;
//    if (alpha > 100.0 || (alpha * log_xi < exp_threshold)) {
//      x_alpha = std::exp(alpha * log_xi);
//    } else {
//      x_alpha = std::pow(xi, alpha);
//    }
//    
//    // Compute v = 1-x^α with precision for x^α near 1
//    double v;
//    if (x_alpha > 0.9995) {
//      v = -std::expm1(alpha * log_xi);  // More precise than 1.0 - x_alpha
//    } else {
//      v = 1.0 - x_alpha;
//    }
//    
//    // Ensure v is not too small
//    v = std::max(v, min_val);
//    double log_v = std::log(v);
//    d_beta += log_v;  // Accumulate (β-1) * log(v_i) term
//    
//    // Compute v^β stably
//    double v_beta, v_beta_m1;
//    if (beta > 100.0 || (beta * log_v < exp_threshold)) {
//      double log_v_beta = beta * log_v;
//      v_beta = std::exp(log_v_beta);
//      v_beta_m1 = std::exp((beta - 1.0) * log_v);
//    } else {
//      v_beta = std::pow(v, beta);
//      v_beta_m1 = std::pow(v, beta - 1.0);
//    }
//    
//    // Compute w = 1-v^β with precision for v^β near 1
//    double w;
//    if (v_beta > 0.9995) {
//      w = -std::expm1(beta * log_v);
//    } else {
//      w = 1.0 - v_beta;
//    }
//    
//    // Ensure w is not too small
//    w = std::max(w, min_val);
//    double log_w = std::log(w);
//    d_lambda += log_w;  // Accumulate (λ-1) * log(w_i) term
//    
//    // --- Alpha gradient component ---
//    // Calculate x^α * log(x) term
//    double x_alpha_log_x = x_alpha * log_xi;
//    
//    // Calculate (β-1)/v term - stable for β ≈ 1
//    double alpha_term1 = 0.0;
//    if (std::abs(beta - 1.0) > 1e-14) {
//      alpha_term1 = (beta - 1.0) / v;
//    }
//    
//    // Calculate (λ-1) * β * v^(β-1) / w term - with λ stability
//    double alpha_term2 = 0.0;
//    if (std::abs(lambda - 1.0) > 1e-14) {
//      double lambda_factor = lambda - 1.0;
//      // Clamp the factor for very large lambda to prevent overflow
//      if (lambda > 1000.0) {
//        lambda_factor = std::min(lambda_factor, 1000.0);
//      }
//      alpha_term2 = lambda_factor * beta * v_beta_m1 / w;
//    }
//    
//    d_alpha -= x_alpha_log_x * (alpha_term1 - alpha_term2);
//    
//    // --- Beta gradient component ---
//    // Calculate v^β * log(v) * (λ-1) / w term - with λ stability
//    double beta_term = 0.0;
//    if (std::abs(lambda - 1.0) > 1e-14) {
//      double lambda_factor = lambda - 1.0;
//      // Clamp the factor for very large lambda
//      if (lambda > 1000.0) {
//        lambda_factor = std::min(lambda_factor, 1000.0);
//      }
//      beta_term = v_beta * log_v * lambda_factor / w;
//    }
//    
//    d_beta -= beta_term;
//  }
//  
//  // Negate for negative log-likelihood
//  grad[0] = -d_alpha;
//  grad[1] = -d_beta;
//  grad[2] = -d_lambda;
//  
//  return grad;
// }
// 
// // -----------------------------------------------------------------------------
// // 7) grekw: Hessian of Negative Log-Likelihood of EKw
// // -----------------------------------------------------------------------------
// 
// 
// // [[Rcpp::export(.hsekw_cpp)]]
// Rcpp::NumericMatrix hsekw(const Rcpp::NumericVector& par, const Rcpp::NumericVector& data) {
//  // Parameter validation
//  if (par.size() < 3) {
//    Rcpp::NumericMatrix nanH(3,3);
//    nanH.fill(R_NaN);
//    return nanH;
//  }
//  
//  double alpha = par[0];
//  double beta = par[1];
//  double lambda = par[2];
//  
//  if (alpha <= 0 || beta <= 0 || lambda <= 0) {
//    Rcpp::NumericMatrix nanH(3,3);
//    nanH.fill(R_NaN);
//    return nanH;
//  }
//  
//  arma::vec x = Rcpp::as<arma::vec>(data);
//  if (x.n_elem < 1 || arma::any(x <= 0) || arma::any(x >= 1)) {
//    Rcpp::NumericMatrix nanH(3,3);
//    nanH.fill(R_NaN);
//    return nanH;
//  }
//  
//  int n = x.n_elem;
//  arma::mat H(3,3, arma::fill::zeros);
//  
//  // Stability constants
//  // const double eps = std::numeric_limits<double>::epsilon() * 100;
//  const double min_v = 1e-10;  // Minimum value for v = 1-x^α
//  const double min_w = 1e-10;  // Minimum value for w = 1-(1-x^α)^β
//  const double exp_threshold = -700.0;  // Threshold for log-domain calculations
//  
//  // Constant terms (diagonal elements)
//  H(0,0) = -n / (alpha * alpha);  // -n/α²
//  H(1,1) = -n / (beta * beta);    // -n/β²
//  H(2,2) = -n / (lambda * lambda); // -n/λ²
//  
//  // Special handling for lambda near 1 (critical case for stability)
//  bool lambda_near_one = std::abs(lambda - 1.0) < 1e-8;
//  
//  for (int i = 0; i < n; i++) {
//    double xi = x(i);
//    double log_xi = std::log(xi);
//    
//    // Calculate x^α (A) and derivatives with log-domain for large alpha
//    double A, dA_dalpha, d2A_dalpha2;
//    if (alpha > 100.0 || (alpha * log_xi < exp_threshold)) {
//      double log_A = alpha * log_xi;
//      A = std::exp(log_A);
//      dA_dalpha = A * log_xi;
//      d2A_dalpha2 = A * log_xi * log_xi;
//    } else {
//      A = std::pow(xi, alpha);
//      dA_dalpha = A * log_xi;
//      d2A_dalpha2 = A * log_xi * log_xi;
//    }
//    
//    // Calculate v = 1 - x^α with precision for x^α near 1
//    double v;
//    if (A > 0.9995) {
//      v = -std::expm1(alpha * log_xi);  // More precise than 1.0 - A
//    } else {
//      v = 1.0 - A;
//    }
//    
//    // Ensure v is not too small
//    v = std::max(v, min_v);
//    double log_v = std::log(v);
//    
//    double dv_dalpha = -dA_dalpha;
//    double d2v_dalpha2 = -d2A_dalpha2;
//    
//    // L5 derivatives: (β-1) log(v)
//    double d2L5_dalpha2 = 0.0;
//    double d2L5_dalpha_dbeta = 0.0;
//    
//    if (beta != 1.0) {
//      double v_squared = std::max(v * v, 1e-200); // Prevent division by zero
//      d2L5_dalpha2 = (beta - 1.0) * ((d2v_dalpha2 * v - dv_dalpha * dv_dalpha) / v_squared);
//      d2L5_dalpha_dbeta = dv_dalpha / v;
//    }
//    
//    // Calculate v^β with log-domain for large beta
//    double v_beta, v_beta_m1, v_beta_m2;
//    if (beta > 100.0 || (beta * log_v < exp_threshold)) {
//      v_beta = std::exp(beta * log_v);
//      v_beta_m1 = std::exp((beta - 1.0) * log_v);
//      v_beta_m2 = std::exp((beta - 2.0) * log_v);
//    } else {
//      v_beta = std::pow(v, beta);
//      v_beta_m1 = std::pow(v, beta - 1.0);
//      v_beta_m2 = std::pow(v, beta - 2.0);
//    }
//    
//    // Calculate w = 1 - v^β precisely for v^β near 1
//    double w;
//    if (v_beta > 0.9995) {
//      w = -std::expm1(beta * log_v);
//    } else {
//      w = 1.0 - v_beta;
//    }
//    
//    w = std::max(w, min_w);
//    double w_squared = std::max(w * w, 1e-200); // Prevent division by zero
//    
//    // First derivatives of w
//    double dw_dv = -beta * v_beta_m1;
//    double dw_dalpha = dw_dv * dv_dalpha;
//    double dw_dbeta = -v_beta * log_v;
//    
//    // Second derivatives of w
//    double d2w_dalpha2 = -beta * ((beta - 1.0) * v_beta_m2 * (dv_dalpha * dv_dalpha) +
//                                  v_beta_m1 * d2v_dalpha2);
//    double d2w_dbeta2 = -v_beta * (log_v * log_v);
//    double d_dw_dalpha_dbeta = -v_beta_m1 * (1.0 + beta * log_v) * dv_dalpha;
//    
//    // L6 derivatives: (λ-1) log(w)
//    double d2L6_dalpha2 = 0.0;
//    double d2L6_dbeta2 = 0.0;
//    double d2L6_dalpha_dbeta = 0.0;
//    double d2L6_dalpha_dlambda = 0.0;
//    double d2L6_dbeta_dlambda = 0.0;
//    
//    // Critical lambda handling for stability
//    if (lambda_near_one) {
//      // For lambda ≈ 1, handle carefully to avoid cancellation errors
//      if (std::abs(lambda - 1.0) > 1e-15) {
//        double factor = lambda - 1.0;
//        d2L6_dalpha2 = factor * ((d2w_dalpha2 * w - (dw_dalpha * dw_dalpha)) / w_squared);
//        d2L6_dbeta2 = factor * ((d2w_dbeta2 * w - (dw_dbeta * dw_dbeta)) / w_squared);
//        d2L6_dalpha_dbeta = factor * ((d_dw_dalpha_dbeta / w) - (dw_dalpha * dw_dbeta) / w_squared);
//      }
//      // When λ = 1 (machine precision), these terms become zero
//      d2L6_dalpha_dlambda = dw_dalpha / w;
//      d2L6_dbeta_dlambda = dw_dbeta / w;
//    } else {
//      // Standard case
//      d2L6_dalpha2 = (lambda - 1.0) * ((d2w_dalpha2 * w - (dw_dalpha * dw_dalpha)) / w_squared);
//      d2L6_dbeta2 = (lambda - 1.0) * ((d2w_dbeta2 * w - (dw_dbeta * dw_dbeta)) / w_squared);
//      d2L6_dalpha_dbeta = (lambda - 1.0) * ((d_dw_dalpha_dbeta / w) - (dw_dalpha * dw_dbeta) / w_squared);
//      d2L6_dalpha_dlambda = dw_dalpha / w;
//      d2L6_dbeta_dlambda = dw_dbeta / w;
//    }
//    
//    // Handle large lambda values (> 1000)
//    if (lambda > 1000.0) {
//      double max_val = 1000.0;
//      d2L6_dalpha2 = std::min(std::max(d2L6_dalpha2, -max_val), max_val);
//      d2L6_dbeta2 = std::min(std::max(d2L6_dbeta2, -max_val), max_val);
//      d2L6_dalpha_dbeta = std::min(std::max(d2L6_dalpha_dbeta, -max_val), max_val);
//    }
//    
//    // Accumulate contributions to Hessian
//    H(0,0) += d2L5_dalpha2 + d2L6_dalpha2;
//    H(0,1) += d2L5_dalpha_dbeta + d2L6_dalpha_dbeta;
//    H(1,0) = H(0,1);
//    H(1,1) += d2L6_dbeta2;
//    H(0,2) += d2L6_dalpha_dlambda;
//    H(2,0) = H(0,2);
//    H(1,2) += d2L6_dbeta_dlambda;
//    H(2,1) = H(1,2);
//  }
//  
//  // Ensure perfect symmetry by averaging
//  for (int i = 0; i < 3; i++) {
//    for (int j = i+1; j < 3; j++) {
//      double avg = (H(i,j) + H(j,i)) / 2.0;
//      H(i,j) = H(j,i) = avg;
//    }
//  }
//  
//  return Rcpp::wrap(-H);
// }
