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
  // Zero-length input: follow R's recycling convention and return an empty
  // vector, as dbeta(numeric(0), 1, 1) does. This also guards the
  // `i % vec.n_elem` recycling below against integer division by zero.
  if (x.n_elem == 0 || a_vec.n_elem == 0 || b_vec.n_elem == 0 || l_vec.n_elem == 0) {
    return Rcpp::NumericVector(0);
  }

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
    
    // Missing and undefined input propagates, as base R does: dbeta(NA) is NA
    // and dbeta(NaN) is NaN. Both used to fall into the "outside the support"
    // branch below -- !R_finite() is true for a NaN -- and were silently
    // replaced by the fill value, 0 or -Inf in log. NA_REAL is itself a NaN
    // carrying a distinguishing payload, so R_IsNA() must be asked first.
    if (ISNAN(xx)) {
      out(i) = R_IsNA(xx) ? NA_REAL : R_NaN;
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
    double log_v = gkw_log1mexp(log_xalpha);
    if (!R_finite(log_v)) {
      continue;
    }
    
    // Term 1: (β-1) * log(1 - x^α)
    double term2 = (b - 1.0) * log_v;
    
    // Compute log((1-x^α)^β) = β * log(1-x^α)
    double log_v_beta = b * log_v;
    
    // Compute log(1 - (1-x^α)^β) = log(w) using log1mexp
    double log_w = gkw_log1mexp(log_v_beta);
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
  // Zero-length input: follow R's recycling convention and return an empty
  // vector, as dbeta(numeric(0), 1, 1) does. This also guards the
  // `i % vec.n_elem` recycling below against integer division by zero.
  if (q.n_elem == 0 || a_vec.n_elem == 0 || b_vec.n_elem == 0 || l_vec.n_elem == 0) {
    return Rcpp::NumericVector(0);
  }

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
    
    // Missing and undefined input propagates, as in d*() above.
    if (ISNAN(xx)) {
      out(i) = R_IsNA(xx) ? NA_REAL : R_NaN;
      continue;
    }
    
    // Handle boundary: q ≤ 0
    if (xx <= 0.0) {
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
    
    // ---- Cumulative probability, computed entirely in log space ----
    // Every step below is a log(1 - exp(u)), never a 1 - u in linear space.
    // The former chain formed 1 - x^alpha and 1 - (1 - x^alpha)^beta directly:
    // once x^alpha fell below 1.1e-16 the first rounded to exactly 1, the
    // second to exactly 0, and the CDF collapsed to 0 or 1. That is not a
    // last-digit loss -- pekw(5.6e-09, 2, 5, 0.02) returned 0 where the true
    // value is 0.483.
    double log_x_alpha = a * std::log(xx);

    // F = [1 - (1 - x^alpha)^beta]^lambda
    double log_t    = gkw_log1mexp(b * gkw_log1mexp(log_x_alpha));
    double log_cdf  = l * log_t;
    double log_surv = gkw_log1mexp(log_cdf);

    // Emit the requested tail on the requested scale without ever forming
    // 1 - p or log(p) from a value that has already lost its digits.
    double lg = lower_tail ? log_cdf : log_surv;
    out(i) = log_p ? lg : std::exp(lg);
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
  // Zero-length input: follow R's recycling convention and return an empty
  // vector, as dbeta(numeric(0), 1, 1) does. This also guards the
  // `i % vec.n_elem` recycling below against integer division by zero.
  if (p.n_elem == 0 || a_vec.n_elem == 0 || b_vec.n_elem == 0 || l_vec.n_elem == 0) {
    return Rcpp::NumericVector(0);
  }

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
    
    // ---- Normalise the probability, without leaving log space ----
    // The former code did exp(log p) and then 1 - p in linear space. The first
    // flushed the deep tail to zero (qbeta_(-1000, 2, 3, log.p = TRUE) gave 0
    // against a true 2.25e-218); the second cost the upper tail.
    // Missing and undefined input propagates.
    if (ISNAN(pp)) {
      out(i) = R_IsNA(pp) ? NA_REAL : R_NaN;
      continue;
    }
    
    // A probability outside its range has no quantile. The wrappers have
    // always warned that such a value "will produce NaN"; the C++ saturated
    // at 0 or 1 instead, which are outside the open support and which
    // defensive code testing is.nan() could not detect.
    if (log_p ? (pp > 0.0) : (pp < 0.0 || pp > 1.0)) {
      out(i) = R_NaN;
      continue;
    }

    // log(u) and log(1-u) for the lower-tail probability u, whichever scale and
    // tail the caller used, so neither has to be recovered by subtraction.
    double log_u, log_1mu;
    if (log_p) {
      if (lower_tail) { log_u = pp;               log_1mu = gkw_log1mexp(pp); }
      else            { log_u = gkw_log1mexp(pp); log_1mu = pp; }
    } else {
      if (lower_tail) { log_u = std::log(pp);     log_1mu = std::log1p(-pp); }
      else            { log_u = std::log1p(-pp);  log_1mu = std::log(pp); }
    }
    if (log_u   == R_NegInf) { out(i) = 0.0; continue; }
    if (log_1mu == R_NegInf) { out(i) = 1.0; continue; }

    // Q(u) = [1 - (1 - u^(1/lambda))^(1/beta)]^(1/alpha)
    out(i) = std::exp(gkw_log1mexp(gkw_log1mexp(log_u / l) / b) / a);
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
  // n = 0 returns numeric(0), matching stats::rbeta(0, 2, 3). Only a
  // negative n is an error.
  if (n < 0) {
    Rcpp::stop("rekw: n must be non-negative");
  }
  if (n == 0) {
    return Rcpp::NumericVector(0);
  }
  
  // Convert R vectors to Armadillo vectors
  arma::vec a_vec(alpha.begin(), alpha.size());
  arma::vec b_vec(beta.begin(), beta.size());
  arma::vec l_vec(lambda.begin(), lambda.size());
  
  // A zero-length parameter cannot be recycled. Match R's convention
  // (rbeta(3, numeric(0), 1) is NA NA NA with a warning) instead of
  // reaching the `i % vec.n_elem` recycling with a zero divisor.
  if (a_vec.n_elem == 0 || b_vec.n_elem == 0 || l_vec.n_elem == 0) {
    Rcpp::warning("rekw: NAs produced");
    return Rcpp::NumericVector(n, NA_REAL);
  }

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

    // x = [1 - (1 - U^(1/lambda))^(1/beta)]^(1/alpha), inverted in log space.
    // The draw itself is untouched, so the RNG stream is identical to
    // before; only the inversion that follows it changes.
    out(i) = std::exp(gkw_log1mexp(gkw_log1mexp(std::log(U) / l) / b) / a);
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

  // Everything below stays in log space. Forming v = 1 - x^α or w = 1 - v^β as
  // doubles loses all significance once x^α or v^β rounds to 1, which happens
  // routinely for small x and moderate α; gkw_log1mexp() keeps full precision there.
  for (int i = 0; i < n; i++) {
    double log_xi = std::log(x(i));

    // Term 1: (α-1) * log(x)
    sum_term1 += (alpha - 1.0) * log_xi;

    // log(v) where v = 1 - x^α
    double log_x_alpha = alpha * log_xi;
    double log_v = gkw_log1mexp(log_x_alpha);

    // Term 2: (β-1) * log(v)
    sum_term2 += (beta - 1.0) * log_v;

    // log(w) where w = 1 - v^β
    double log_w = gkw_log1mexp(beta * log_v);

    // Term 3: (λ-1) * log(w)
    sum_term3 += (lambda - 1.0) * log_w;
  }

  // Combine all terms
  double loglike = const_term + sum_term1 + sum_term2 + sum_term3;

  if (!std::isfinite(loglike)) return R_PosInf;

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

  // Initialize gradient accumulators
  double d_alpha = n / alpha;
  double d_beta = n / beta;
  double d_lambda = n / lambda;

  // Notation, all per observation and all evaluated from logarithms:
  //   P = ∂log(v)/∂α = -log(x) * exp(log(x^α) - log(v))
  //   S = v^β / w    = exp(β*log(v) - log(w))
  //   Q = ∂log(w)/∂α = -β * P * S
  //   R = ∂log(w)/∂β = -log(v) * S
  // Writing each ratio as a single exp() of a difference of logs avoids the
  // 0/0 and tiny/tiny divisions that the direct v, w forms degenerate into.
  for (int i = 0; i < n; i++) {
    double log_xi = std::log(x(i));
    d_alpha += log_xi;

    double log_x_alpha = alpha * log_xi;
    double log_v = gkw_log1mexp(log_x_alpha);
    d_beta += log_v;

    double log_v_beta = beta * log_v;
    double log_w = gkw_log1mexp(log_v_beta);
    d_lambda += log_w;

    double P = -log_xi * std::exp(log_x_alpha - log_v);
    double S = std::exp(log_v_beta - log_w);
    double Q = -beta * P * S;
    double R = -log_v * S;

    // ∂ℓ/∂α += (β-1) * P + (λ-1) * Q
    d_alpha += (beta - 1.0) * P + (lambda - 1.0) * Q;

    // ∂ℓ/∂β += (λ-1) * R
    d_beta += (lambda - 1.0) * R;
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

  // Constant diagonal terms
  H(0, 0) = -n / (alpha * alpha);   // -n/alpha^2
  H(1, 1) = -n / (beta * beta);     // -n/beta^2
  H(2, 2) = -n / (lambda * lambda); // -n/lambda^2

  // Same log-space quantities as the gradient:
  //   P = dlog(v)/dalpha, S = v^beta/w, Q = dlog(w)/dalpha, R = dlog(w)/dbeta
  // and their derivatives, each obtained by differentiating the logarithms:
  //   dP/dalpha = P*(log(x) - P)
  //   dQ/dalpha = Q*(log(x) - P + beta*P - Q)
  //   dQ/dbeta  = -P*S*(1 + beta*(log(v) - R))
  //   dR/dbeta  = R*(log(v) - R)
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

    // d2l/dalpha^2
    H(0, 0) += (beta - 1.0) * dP_dalpha + (lambda - 1.0) * dQ_dalpha;

    // d2l/dalpha dbeta: P survives at beta = 1, it carries no (beta-1) factor
    H(0, 1) += P + (lambda - 1.0) * dQ_dbeta;

    // d2l/dbeta^2
    H(1, 1) += (lambda - 1.0) * dR_dbeta;

    // Mixed derivatives with lambda are the plain log-derivatives of w
    H(0, 2) += Q;
    H(1, 2) += R;
  }

  // Symmetrize once after accumulation
  H = arma::symmatu(H);
  
  // Return NEGATIVE Hessian (for minimization)
  return Rcpp::wrap(-H);
}

