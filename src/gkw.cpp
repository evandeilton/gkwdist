/**
 * @file gkw.cpp
 * @brief Generalized Kumaraswamy (GKw) Distribution Functions
 * 
 * @details
 * This file implements the full suite of distribution functions for the
 * five-parameter Generalized Kumaraswamy (GKw) distribution, which serves
 * as the parent distribution for six sub-families in the gkwdist package.
 * 
 * The GKw distribution has probability density function:
 * \deqn{
 *   f(x; \alpha, \beta, \gamma, \delta, \lambda) = 
 *   \frac{\lambda \alpha \beta}{B(\gamma, \delta+1)} x^{\alpha-1} (1-x^\alpha)^{\beta-1}
 *   [1-(1-x^\alpha)^\beta]^{\gamma\lambda-1} \{1-[1-(1-x^\alpha)^\beta]^\lambda\}^\delta
 * }
 * for \eqn{x \in (0,1)}, where \eqn{B(\cdot,\cdot)} is the Beta function.
 * 
 * **Parameter Constraints:**
 * - \eqn{\alpha > 0} (shape parameter)
 * - \eqn{\beta > 0} (shape parameter)
 * - \eqn{\gamma > 0} (shape parameter)
 * - \eqn{\delta \geq 0} (shape parameter)
 * - \eqn{\lambda > 0} (shape parameter)
 * 
 * **Special Cases (Sub-families):**
 * | Distribution | Parameters | Relation to GKw |
 * |--------------|------------|-----------------|
 * | Beta-Kumaraswamy (BKw) | \eqn{\alpha, \beta, \gamma, \delta} | \eqn{\lambda = 1} |
 * | Kumaraswamy-Kumaraswamy (KKw) | \eqn{\alpha, \beta, \delta, \lambda} | \eqn{\gamma = 1} |
 * | Exponentiated Kumaraswamy (EKw) | \eqn{\alpha, \beta, \lambda} | \eqn{\gamma = 1, \delta = 0} |
 * | McDonald/Beta-Power (Mc/BP) | \eqn{\gamma, \delta, \lambda} | \eqn{\alpha = \beta = 1} |
 * | Kumaraswamy (Kw) | \eqn{\alpha, \beta} | \eqn{\gamma = 1, \delta = 0, \lambda = 1} |
 * | Beta | \eqn{\gamma, \delta} | \eqn{\alpha = \beta = \lambda = 1} |
 * 
 * **Numerical Stability:**
 * All computations use log-space arithmetic and numerically stable helper
 * functions from utils.h to prevent overflow/underflow across the full
 * parameter space. Key techniques include:
 * - gkw_log1mexp() for computing log(1 - exp(x)) stably
 * - safe_pow(), safe_exp(), safe_log() for protected arithmetic
 * - Vectorized operations via Armadillo with element-wise stability
 * 
 * **Implemented Functions:**
 * - dgkw(): Probability density function (PDF)
 * - pgkw(): Cumulative distribution function (CDF)
 * - qgkw(): Quantile function (inverse CDF)
 * - rgkw(): Random variate generation
 * - llgkw(): Negative log-likelihood for MLE
 * - grgkw(): Gradient of negative log-likelihood
 * - hsgkw(): Hessian of negative log-likelihood
 * 
 * @author Lopes, J. E.
 * @date 2025-01-07
 * 
 * @references
 * Cordeiro, G. M., & de Castro, M. (2011). A new family of generalized
 * distributions. Journal of Statistical Computation and Simulation.
 * 
 * Kumaraswamy, P. (1980). A generalized probability density function for
 * double-bounded random processes. Journal of Hydrology, 46(1-2), 79-88.
 * 
 * @see utils.h for numerical stability functions and parameter validators
 * 
 * @note All functions use R's vectorization conventions with parameter recycling.
 * @note Thread-safe: No global state is modified.
 */

// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include "utils.h"

// ----------------------------------------------------------------------------
// Shared log-space chain for the GKw likelihood family
// ----------------------------------------------------------------------------
//
// The GKw density nests three transformations,
//
//     v = 1 - x^alpha        w = 1 - v^beta        z = 1 - w^lambda
//
// and llgkw(), grgkw() and hsgkw() all need their logarithms. Two of the three
// have a regime where the linear quantity underflows to a boundary that
// gkw_log1mexp() cannot recover from, because its argument arrives as exactly 0
// and log(1 - exp(0)) is -Inf:
//
//   x -> 0 :  v -> 1, so log_v = log1p(-x^alpha) ~ -x^alpha, which underflows
//             to 0 once x^alpha < 5e-324. Then log1mexp(beta*log_v) is -Inf.
//             But w = 1 - v^beta -> beta * x^alpha, so
//             log_w = log(beta) + alpha*log(x).
//
//   x -> 1 :  w -> 1 by the same route one level down, and
//             z = 1 - w^lambda -> lambda * v^beta, so
//             log_z = log(lambda) + beta*log_v.
//
// Each substitution is the first-order limit and agrees with log1mexp() to the
// last bits wherever both are representable: at log_w = -1e-300 with lambda = 2
// the two forms give -690.09 and -690.09. The branches fire only where the old
// code produced -Inf, so ordinary data is bit-identical.
//
// This is what made llgkw() return NaN where the likelihood is finite:
// llgkw(c(1, 300, 1, 0, 1), c(.8,.85,.9,.95)) gave NaN for an exact 2609.84,
// because log_z came back -Inf and delta was 0, and 0 * -Inf is NaN.
static inline void gkw_log_chain(double log_x, double alpha, double beta,
                                 double lambda,
                                 double& log_x_alpha, double& log_v,
                                 double& log_v_beta, double& log_w,
                                 double& log_w_lambda, double& log_z) {
  log_x_alpha  = alpha * log_x;
  log_v        = gkw_log1mexp(log_x_alpha);
  log_v_beta   = beta * log_v;
  log_w        = (log_v == 0.0) ? (std::log(beta) + log_x_alpha)
                                : gkw_log1mexp(log_v_beta);
  log_w_lambda = lambda * log_w;
  log_z        = (log_w == 0.0) ? (std::log(lambda) + log_v_beta)
                                : gkw_log1mexp(log_w_lambda);
}

// log_v and log_w each underflow to 0 in the regime where their true values are
// -x^alpha and -v^beta. The quotient they multiply has overflowed by exactly the
// reciprocal amount, so the product is finite while the two factors are 0 and
// +Inf -- and 0 * Inf is NaN. That is how grgkw() returned an all-NaN gradient
// where llgkw() was finite and numDeriv gave an ordinary answer:
//
//   par = (1, 200, 1.5, 2, 1), x = c(.10,.25,.40,.72,.99)
//     grgkw           NaN NaN NaN NaN NaN
//     numDeriv::grad  -1897.99  20.32  -6.76  1354.07  -15.00
//
// Substituting the magnitude back inside the exponential keeps the product
// exact and finite:  log_v * exp(E)  ->  -exp(log_x_alpha + E). The two
// familiar limits fall out of it: with E = log_v_beta - log_w the result is
// -1/beta, and the lambda term's -1/lambda likewise.
static inline double gkw_mul_small_log(double log_small, double log_magnitude,
                                       double E) {
  // log_small is a logarithm of something in (0, 1], so it is <= 0 and the
  // product is <= 0 too.
  //
  // Exactly 0 means the linear quantity underflowed completely and the
  // magnitude has to come from one level up. Short of that, log_small can still
  // be subnormal while exp(E) has already overflowed to +Inf -- and a subnormal
  // times +Inf is -Inf, not the finite product it stands for. Both cases are
  // handled by moving the magnitude inside the exponential, which needs only
  // the sum of the logs to be representable.
  //
  // The direct multiply is kept wherever exp(E) cannot overflow, so ordinary
  // data is bit-identical and pays nothing for the extra log().
  if (log_small == 0.0) return -std::exp(log_magnitude + E);
  if (E < 700.0)        return log_small * std::exp(E);
  return -std::exp(std::log(-log_small) + E);
}


// ============================================================================
// PROBABILITY DENSITY FUNCTION
// ============================================================================

/**
 * @brief Probability Density Function (PDF) of the GKw Distribution
 * 
 * Computes the density (or log-density) for the Generalized Kumaraswamy
 * distribution at specified quantiles.
 * 
 * @param x Vector of quantiles (values in (0,1))
 * @param alpha Shape parameter vector (\eqn{\alpha > 0})
 * @param beta Shape parameter vector (\eqn{\beta > 0})
 * @param gamma Shape parameter vector (\eqn{\gamma > 0})
 * @param delta Shape parameter vector (\eqn{\delta \geq 0})
 * @param lambda Shape parameter vector (\eqn{\lambda > 0})
 * @param log_prob If TRUE, returns log-density; otherwise returns density
 * 
 * @return NumericVector of density values (or log-density if log_prob=TRUE)
 * 
 * @details
 * For x outside (0,1), returns 0 (or -Inf for log-density).
 * Uses log-space computation throughout to ensure numerical stability.
 * Parameters are recycled to match the longest input vector.
 * 
 * @note Exported as .dgkw_cpp for internal package use
 */
// [[Rcpp::export(.dgkw_cpp)]]
Rcpp::NumericVector dgkw(
    const arma::vec& x,
    const Rcpp::NumericVector& alpha,
    const Rcpp::NumericVector& beta,
    const Rcpp::NumericVector& gamma,
    const Rcpp::NumericVector& delta,
    const Rcpp::NumericVector& lambda,
    bool log_prob = false
) {
  // Convert R vectors to Armadillo vectors for efficient computation
  arma::vec alpha_vec(alpha.begin(), alpha.size());
  arma::vec beta_vec(beta.begin(), beta.size());
  arma::vec gamma_vec(gamma.begin(), gamma.size());
  arma::vec delta_vec(delta.begin(), delta.size());
  arma::vec lambda_vec(lambda.begin(), lambda.size());
  
  // Determine output length (maximum of all input lengths for recycling)
  // Zero-length input: follow R's recycling convention and return an empty
  // vector, as dbeta(numeric(0), 1, 1) does. This also guards the
  // `i % vec.n_elem` recycling below against integer division by zero.
  if (x.n_elem == 0 || alpha_vec.n_elem == 0 || beta_vec.n_elem == 0 || gamma_vec.n_elem == 0 || delta_vec.n_elem == 0 || lambda_vec.n_elem == 0) {
    return Rcpp::NumericVector(0);
  }

  size_t n = std::max({x.n_elem, alpha_vec.n_elem, beta_vec.n_elem,
                      gamma_vec.n_elem, delta_vec.n_elem, lambda_vec.n_elem});
  
  // Initialize result vector with appropriate default values
  arma::vec result(n);
  result.fill(log_prob ? R_NegInf : 0.0);
  
  // Process each element with parameter recycling
  for (size_t i = 0; i < n; ++i) {
    // Extract recycled parameter values
    double a = alpha_vec[i % alpha_vec.n_elem];
    double b = beta_vec[i % beta_vec.n_elem];
    double g = gamma_vec[i % gamma_vec.n_elem];
    double d = delta_vec[i % delta_vec.n_elem];
    double l = lambda_vec[i % lambda_vec.n_elem];
    double xi = x[i % x.n_elem];
    
    // Validate parameters
    if (!check_pars(a, b, g, d, l)) {
      Rcpp::warning("dgkw: invalid parameters at index %d (alpha,beta,gamma>0, delta>=0, lambda>0)", i+1);
      continue;
    }
    
    // Check support: x must be in (0, 1)
    if (xi <= 0.0 || xi >= 1.0 || !R_finite(xi)) {
      continue;
    }
    
    // ---- Log-space computation of density ----
    
    // Normalization constant: log(λαβ / B(γ, δ+1))
    double log_beta_val = R::lbeta(g, d + 1.0);
    double log_const = std::log(l) + std::log(a) + std::log(b) - log_beta_val;
    double gamma_lambda = g * l;
    
    // Compute log(x^α) directly. Forming x^α in linear space and taking its
    // logarithm loses the digits that matter as x approaches 1: doubles are
    // spaced 2.2e-16 apart there, so log(x^α) carries a relative error of
    // 2.2e-16 / (1 - x^α), which reaches 4e-6 by 1 - x = 1e-12. The former
    // guard against x^α >= 1 - sqrt(eps) sidestepped that by returning a
    // density of zero, discarding up to 16.5% of the probability mass of some
    // parameterisations. gkw_log1mexp() below is built for exactly this regime.
    //
    // The chain is shared with llgkw(), grgkw() and hsgkw(), and bridges the
    // two points where the linear quantity underflows to a boundary that
    // log1mexp() cannot recover from. The guards that used to sit between these
    // steps -- one `continue` per intermediate -- dropped the observation
    // entirely and left the fill value, 0 or -Inf in log. That is what made
    // -sum(dgkw(c(1e-300, 0.5, 1-1e-15), 1, 70, 1.5, 2, 1, log = TRUE)) return
    // Inf where llbkw() gives the exact 7688.513058854.
    double log_x_alpha, log_v, log_v_beta, log_w, log_w_lambda, log_z;
    gkw_log_chain(std::log(xi), a, b, l,
                  log_x_alpha, log_v, log_v_beta, log_w, log_w_lambda, log_z);
    
    // Assemble log-density:
    // log(f) = log_const + (α-1)*log(x) + (β-1)*log(v) + (γλ-1)*log(w) + δ*log(z)
    //
    // A coefficient that is exactly zero contributes nothing even where its
    // log is -Inf at the boundary of the support; writing 0 * -Inf gives NaN.
    double logdens = log_const;
    if (a != 1.0)             logdens += (a - 1.0) * std::log(xi);
    if (b != 1.0)             logdens += (b - 1.0) * log_v;
    if (gamma_lambda != 1.0)  logdens += (gamma_lambda - 1.0) * log_w;
    if (d != 0.0)             logdens += d * log_z;
    
    // A density that is still not finite here is a genuine boundary value:
    // -Inf in log, 0 on the natural scale, which is what the fill already holds.
    if (!R_finite(logdens)) {
      continue;
    }
    
    // Return appropriate scale
    result(i) = log_prob ? logdens : safe_exp(logdens);
  }
  
  return Rcpp::NumericVector(result.memptr(), result.memptr() + result.n_elem);
}


// ============================================================================
// CUMULATIVE DISTRIBUTION FUNCTION
// ============================================================================

/**
 * @brief Cumulative Distribution Function (CDF) of the GKw Distribution
 * 
 * Computes the cumulative probability for the Generalized Kumaraswamy
 * distribution at specified quantiles.
 * 
 * @param q Vector of quantiles
 * @param alpha Shape parameter vector (\eqn{\alpha > 0})
 * @param beta Shape parameter vector (\eqn{\beta > 0})
 * @param gamma Shape parameter vector (\eqn{\gamma > 0})
 * @param delta Shape parameter vector (\eqn{\delta \geq 0})
 * @param lambda Shape parameter vector (\eqn{\lambda > 0})
 * @param lower_tail If TRUE, returns P(X <= q); otherwise P(X > q)
 * @param log_p If TRUE, returns log-probability
 * 
 * @return NumericVector of cumulative probabilities
 * 
 * @details
 * The CDF is computed as:
 * \deqn{F(x) = I_{[1-(1-x^\alpha)^\beta]^\lambda}(\gamma, \delta+1)}
 * where \eqn{I_y(a,b)} is the regularized incomplete Beta function.
 * 
 * @note Exported as .pgkw_cpp for internal package use
 */
// [[Rcpp::export(.pgkw_cpp)]]
Rcpp::NumericVector pgkw(
    const arma::vec& q,
    const Rcpp::NumericVector& alpha,
    const Rcpp::NumericVector& beta,
    const Rcpp::NumericVector& gamma,
    const Rcpp::NumericVector& delta,
    const Rcpp::NumericVector& lambda,
    bool lower_tail = true,
    bool log_p = false
) {
  // Convert R vectors to Armadillo vectors
  arma::vec alpha_vec(alpha.begin(), alpha.size());
  arma::vec beta_vec(beta.begin(), beta.size());
  arma::vec gamma_vec(gamma.begin(), gamma.size());
  arma::vec delta_vec(delta.begin(), delta.size());
  arma::vec lambda_vec(lambda.begin(), lambda.size());
  
  // Determine output length for recycling
  // Zero-length input: follow R's recycling convention and return an empty
  // vector, as dbeta(numeric(0), 1, 1) does. This also guards the
  // `i % vec.n_elem` recycling below against integer division by zero.
  if (q.n_elem == 0 || alpha_vec.n_elem == 0 || beta_vec.n_elem == 0 || gamma_vec.n_elem == 0 || delta_vec.n_elem == 0 || lambda_vec.n_elem == 0) {
    return Rcpp::NumericVector(0);
  }

  size_t n = std::max({q.n_elem, alpha_vec.n_elem, beta_vec.n_elem,
                      gamma_vec.n_elem, delta_vec.n_elem, lambda_vec.n_elem});
  
  arma::vec result(n);
  
  for (size_t i = 0; i < n; ++i) {
    // Extract recycled parameter values
    double a = alpha_vec[i % alpha_vec.n_elem];
    double b = beta_vec[i % beta_vec.n_elem];
    double g = gamma_vec[i % gamma_vec.n_elem];
    double d = delta_vec[i % delta_vec.n_elem];
    double l = lambda_vec[i % lambda_vec.n_elem];
    double qi = q[i % q.n_elem];
    
    // Validate parameters
    if (!check_pars(a, b, g, d, l)) {
      result(i) = NA_REAL;
      Rcpp::warning("pgkw: invalid parameters at index %d (alpha,beta,gamma>0, delta>=0, lambda>0)", i+1);
      continue;
    }
    
    // Handle boundary cases: q <= 0
    if (!R_finite(qi) || qi <= 0.0) {
      result(i) = lower_tail ? (log_p ? R_NegInf : 0.0) : (log_p ? 0.0 : 1.0);
      continue;
    }
    
    // Handle boundary cases: q >= 1
    if (qi >= 1.0) {
      result(i) = lower_tail ? (log_p ? 0.0 : 1.0) : (log_p ? R_NegInf : 0.0);
      continue;
    }
    
    // ---- Cumulative probability ----
    // Every step is a log(1 - exp(u)); the former chain formed 1 - q^alpha and
    // 1 - (1 - q^alpha)^beta in linear space, which cost the whole lower tail
    // (pgkw at q = 1e-09 returned exactly 0 against a true 4.1e-21).
    // F = I_y(gamma, delta+1) with y = [1 - (1 - q^alpha)^beta]^lambda, and
    // lower_tail/log_p go straight to R::pbeta rather than being applied by
    // forming 1 - p or log(p) afterwards.
    double log_w = gkw_log1mexp(b * gkw_log1mexp(a * std::log(qi)));
    double y = std::exp(l * log_w);
    result(i) = R::pbeta(y, g, d + 1.0, lower_tail, log_p);
  }
  
  return Rcpp::NumericVector(result.memptr(), result.memptr() + result.n_elem);
}


// ============================================================================
// QUANTILE FUNCTION
// ============================================================================

/**
 * @brief Quantile Function (Inverse CDF) of the GKw Distribution
 * 
 * Computes quantiles for the Generalized Kumaraswamy distribution
 * given probability values.
 * 
 * @param p Vector of probabilities (values in [0,1])
 * @param alpha Shape parameter vector (\eqn{\alpha > 0})
 * @param beta Shape parameter vector (\eqn{\beta > 0})
 * @param gamma Shape parameter vector (\eqn{\gamma > 0})
 * @param delta Shape parameter vector (\eqn{\delta \geq 0})
 * @param lambda Shape parameter vector (\eqn{\lambda > 0})
 * @param lower_tail If TRUE, probabilities are P(X <= x); otherwise P(X > x)
 * @param log_p If TRUE, probabilities are given as log(p)
 * 
 * @return NumericVector of quantiles
 * 
 * @details
 * The quantile function is computed by inverting the CDF:
 * \deqn{Q(p) = \{1 - [1 - y^{1/\lambda}]^{1/\beta}\}^{1/\alpha}}
 * where \eqn{y = Q_{Beta}(p; \gamma, \delta+1)}.
 * 
 * @note Exported as .qgkw_cpp for internal package use
 */
// [[Rcpp::export(.qgkw_cpp)]]
Rcpp::NumericVector qgkw(
    const arma::vec& p,
    const Rcpp::NumericVector& alpha,
    const Rcpp::NumericVector& beta,
    const Rcpp::NumericVector& gamma,
    const Rcpp::NumericVector& delta,
    const Rcpp::NumericVector& lambda,
    bool lower_tail = true,
    bool log_p = false
) {
  // Convert R vectors to Armadillo vectors
  arma::vec alpha_vec(alpha.begin(), alpha.size());
  arma::vec beta_vec(beta.begin(), beta.size());
  arma::vec gamma_vec(gamma.begin(), gamma.size());
  arma::vec delta_vec(delta.begin(), delta.size());
  arma::vec lambda_vec(lambda.begin(), lambda.size());
  
  // Determine output length for recycling
  // Zero-length input: follow R's recycling convention and return an empty
  // vector, as dbeta(numeric(0), 1, 1) does. This also guards the
  // `i % vec.n_elem` recycling below against integer division by zero.
  if (p.n_elem == 0 || alpha_vec.n_elem == 0 || beta_vec.n_elem == 0 || gamma_vec.n_elem == 0 || delta_vec.n_elem == 0 || lambda_vec.n_elem == 0) {
    return Rcpp::NumericVector(0);
  }

  size_t n = std::max({p.n_elem, alpha_vec.n_elem, beta_vec.n_elem,
                      gamma_vec.n_elem, delta_vec.n_elem, lambda_vec.n_elem});
  
  arma::vec result(n);
  
  for (size_t i = 0; i < n; ++i) {
    // Extract recycled parameter values
    double a = alpha_vec[i % alpha_vec.n_elem];
    double b = beta_vec[i % beta_vec.n_elem];
    double g = gamma_vec[i % gamma_vec.n_elem];
    double d = delta_vec[i % delta_vec.n_elem];
    double l = lambda_vec[i % lambda_vec.n_elem];
    double pp = p[i % p.n_elem];
    
    // Validate parameters
    if (!check_pars(a, b, g, d, l)) {
      result(i) = NA_REAL;
      Rcpp::warning("qgkw: invalid parameters at index %d (alpha,beta,gamma>0, delta>=0, lambda>0)", i+1);
      continue;
    }
    
    // ---- Normalise the probability, without leaving log space ----
    // The former code did exp(log p) and then 1 - p in linear space. The first
    // flushed the deep tail to zero (qbeta_(-1000, 2, 3, log.p = TRUE) gave 0
    // against a true 2.25e-218); the second cost the upper tail. Out-of-range p
    // keeps the saturating result it has always returned -- whether that should
    // be NaN instead is a separate, still-open question.
    if (log_p && pp > 0.0) { result(i) = NA_REAL; continue; }
    if (!log_p && (pp < 0.0 || pp > 1.0)) {
      result(i) = (lower_tail == (pp > 1.0)) ? 1.0 : 0.0;
      continue;
    }

    // y = I^-1_{gamma,delta+1}(u); w = y^(1/lambda); x = [1-(1-w)^(1/beta)]^(1/alpha).
    // lower_tail and log_p go straight to R::qbeta instead of being undone by
    // exp() and 1 - p first.
    double y = R::qbeta(pp, g, d + 1.0, lower_tail, log_p);
    double log_w = std::log(y) / l;
    result(i) = std::exp(gkw_log1mexp(gkw_log1mexp(log_w) / b) / a);
  }
  
  return Rcpp::NumericVector(result.memptr(), result.memptr() + result.n_elem);
}


// ============================================================================
// RANDOM NUMBER GENERATION
// ============================================================================

/**
 * @brief Random Variate Generation for the GKw Distribution
 * 
 * Generates random samples from the Generalized Kumaraswamy distribution
 * using the inverse transform method.
 * 
 * @param n Number of random variates to generate
 * @param alpha Shape parameter vector (\eqn{\alpha > 0})
 * @param beta Shape parameter vector (\eqn{\beta > 0})
 * @param gamma Shape parameter vector (\eqn{\gamma > 0})
 * @param delta Shape parameter vector (\eqn{\delta \geq 0})
 * @param lambda Shape parameter vector (\eqn{\lambda > 0})
 * 
 * @return NumericVector of n random variates from GKw distribution
 * 
 * @details
 * Uses the representation: if V ~ Beta(γ, δ+1), then
 * \deqn{X = \{1 - [1 - V^{1/\lambda}]^{1/\beta}\}^{1/\alpha} \sim GKw}
 * 
 * @note Exported as .rgkw_cpp for internal package use
 */
// [[Rcpp::export(.rgkw_cpp)]]
Rcpp::NumericVector rgkw(
    int n,
    const Rcpp::NumericVector& alpha,
    const Rcpp::NumericVector& beta,
    const Rcpp::NumericVector& gamma,
    const Rcpp::NumericVector& delta,
    const Rcpp::NumericVector& lambda
) {
  // Convert R vectors to Armadillo vectors
  arma::vec alpha_vec(alpha.begin(), alpha.size());
  arma::vec beta_vec(beta.begin(), beta.size());
  arma::vec gamma_vec(gamma.begin(), gamma.size());
  arma::vec delta_vec(delta.begin(), delta.size());
  arma::vec lambda_vec(lambda.begin(), lambda.size());

  // A zero-length parameter cannot be recycled. Match R's convention
  // (rbeta(3, numeric(0), 1) is NA NA NA with a warning) instead of
  // reaching the `i % vec.n_elem` recycling with a zero divisor.
  if (alpha_vec.n_elem == 0 || beta_vec.n_elem == 0 || gamma_vec.n_elem == 0 ||
      delta_vec.n_elem == 0 || lambda_vec.n_elem == 0) {
    Rcpp::warning("rgkw: NAs produced");
    return Rcpp::NumericVector(n, NA_REAL);
  }

  arma::vec result(n);

  for (int i = 0; i < n; ++i) {
    // Extract recycled parameter values
    double a = alpha_vec[i % alpha_vec.n_elem];
    double b = beta_vec[i % beta_vec.n_elem];
    double g = gamma_vec[i % gamma_vec.n_elem];
    double d = delta_vec[i % delta_vec.n_elem];
    double l = lambda_vec[i % lambda_vec.n_elem];
    
    // Validate parameters
    if (!check_pars(a, b, g, d, l)) {
      result(i) = NA_REAL;
      Rcpp::warning("rgkw: invalid parameters at index %d (alpha,beta,gamma>0, delta>=0, lambda>0)", i+1);
      continue;
    }
    
    // Generate V ~ Beta(gamma, delta+1)
    double vi = R::rbeta(g, d + 1.0);

    // vi is y = [1 - (1 - x^alpha)^beta]^lambda; invert in log space rather
    // than through 1 - v and 1 - v^(1/beta) in linear arithmetic.
    // The draw itself is untouched, so the RNG stream is identical to
    // before; only the inversion that follows it changes.
    double log_w = std::log(vi) / l;
    result(i) = std::exp(gkw_log1mexp(gkw_log1mexp(log_w) / b) / a);
  }
  
  return Rcpp::NumericVector(result.memptr(), result.memptr() + result.n_elem);
}


// ============================================================================
// NEGATIVE LOG-LIKELIHOOD FUNCTION
// ============================================================================

/**
 * @brief Negative Log-Likelihood for GKw Distribution
 * 
 * Computes the negative log-likelihood function for parameter estimation
 * via maximum likelihood.
 * 
 * @param par Parameter vector of length 5: (α, β, γ, δ, λ)
 * @param data Vector of observations (must be in (0,1))
 * 
 * @return Negative log-likelihood value (scalar)
 * 
 * @details
 * The log-likelihood for n observations is:
 * \deqn{\ell(\theta) = n[\ln\lambda + \ln\alpha + \ln\beta - \ln B(\gamma,\delta+1)]
 *       + (\alpha-1)\sum\ln x_i + (\beta-1)\sum\ln(1-x_i^\alpha)
 *       + (\gamma\lambda-1)\sum\ln w_i + \delta\sum\ln z_i}
 * where \eqn{w_i = 1-(1-x_i^\alpha)^\beta} and \eqn{z_i = 1-w_i^\lambda}.
 * 
 * Returns +Inf for invalid parameters or data outside (0,1).
 * 
 * @note Exported as .llgkw_cpp for internal package use
 */
// [[Rcpp::export(.llgkw_cpp)]]
double llgkw(const Rcpp::NumericVector& par, const Rcpp::NumericVector& data) {
  // Extract parameters
  double alpha = par[0];
  double beta = par[1];
  double gamma = par[2];
  double delta = par[3];
  double lambda = par[4];
  
  // Validate parameters
  if (!check_pars(alpha, beta, gamma, delta, lambda)) {
    return R_PosInf;
  }
  
  // Convert data to Armadillo vector
  arma::vec x = Rcpp::as<arma::vec>(data);
  
  // Validate data support
  //
  // llgkw() returns the NEGATIVE log-likelihood, so an invalid point must be
  // +Inf: the value optim() moves away from. Returning -Inf made data outside
  // the open support the global minimum of the objective, and left llgkw() the
  // only one of the seven families with that sign -- llbkw(), llkkw(), llekw(),
  // llkw(), llmc() and llbeta() all return +Inf here. The practical damage was
  // in comparing likelihoods rather than in optim(), which refuses to start at
  // either infinity: on data holding a single 0, the GKw family won every
  // selection by nll = -Inf and AIC = -Inf.
  if (arma::any(x <= 0) || arma::any(x >= 1)) {
    return R_PosInf;
  }
  
  int n = x.n_elem;
  
  // ---- Compute log-likelihood terms ----
  
  // Constant term: n * log(λαβ / B(γ, δ+1))
  double log_beta_term = R::lbeta(gamma, delta + 1);
  double constant_term = n * (std::log(lambda) + std::log(alpha) + std::log(beta) - log_beta_term);
  
  // ---- Accumulate the four data terms ----
  //
  // log(x^alpha) is alpha*log(x) directly. The former code went through
  // vec_safe_log(vec_safe_pow(x, alpha)), a round trip that loses digits and
  // that made llgkw() disagree with dgkw(), which already used std::log(x).
  //
  // Each coefficient is tested against zero before it multiplies its log. At
  // the boundary of the support a log legitimately reaches -Inf, and a zero
  // coefficient must then contribute nothing -- 0 * -Inf is NaN, which is how
  // llgkw(c(1, 300, 1, 0, 1), c(.8,.85,.9,.95)) returned NaN for an exact
  // 2609.84 with delta = 0.
  const double c1 = alpha - 1.0;
  const double c2 = beta - 1.0;
  const double c3 = gamma * lambda - 1.0;
  const double c4 = delta;
  
  double term1 = 0.0, term2 = 0.0, term3 = 0.0, term4 = 0.0;
  
  for (int i = 0; i < n; i++) {
    double log_xi = std::log(x(i));
    double log_x_alpha, log_v, log_v_beta, log_w, log_w_lambda, log_z;
    gkw_log_chain(log_xi, alpha, beta, lambda,
                  log_x_alpha, log_v, log_v_beta, log_w, log_w_lambda, log_z);
    
    if (c1 != 0.0) term1 += c1 * log_xi;
    if (c2 != 0.0) term2 += c2 * log_v;
    if (c3 != 0.0) term3 += c3 * log_w;
    if (c4 != 0.0) term4 += c4 * log_z;
  }
  
  // Return negative log-likelihood
  return -(constant_term + term1 + term2 + term3 + term4);
}


// ============================================================================
// GRADIENT OF NEGATIVE LOG-LIKELIHOOD
// ============================================================================

/**
 * @brief Gradient of Negative Log-Likelihood for GKw Distribution
 * 
 * Computes the gradient vector of the negative log-likelihood for
 * optimization-based parameter estimation.
 * 
 * @param par Parameter vector of length 5: (α, β, γ, δ, λ)
 * @param data Vector of observations (must be in (0,1))
 * 
 * @return NumericVector of length 5 containing partial derivatives
 *         with respect to (α, β, γ, δ, λ)
 * 
 * @details
 * Computes analytical gradients using chain rule and log-space arithmetic
 * for numerical stability. Returns NaN vector for invalid inputs.
 * 
 * @note Exported as .grgkw_cpp for internal package use
 */
// [[Rcpp::export(.grgkw_cpp)]]
Rcpp::NumericVector grgkw(const Rcpp::NumericVector& par, const Rcpp::NumericVector& data) {
  // Extract parameters
  double alpha = par[0];
  double beta = par[1];
  double gamma = par[2];
  double delta = par[3];
  double lambda = par[4];
  
  // Validate parameters
  if (!check_pars(alpha, beta, gamma, delta, lambda)) {
    return Rcpp::NumericVector(5, R_NaN);
  }
  
  // Convert and validate data
  arma::vec x = Rcpp::as<arma::vec>(data);
  
  if (arma::any(x <= 0) || arma::any(x >= 1)) {
    return Rcpp::NumericVector(5, R_NaN);
  }
  
  int n = x.n_elem;
  Rcpp::NumericVector grad(5, 0.0);
  
  // ---- Compute gradient components ----
  //
  // Every ratio below is written as a single exp() of a difference of logs.
  // The former code built the reciprocals separately -- vec_safe_exp(-log_v),
  // vec_safe_exp(-log_w), vec_safe_exp(-log_z) -- and each of those overflowed
  // to +Inf on its own, long before the product it belonged to was large. The
  // whole gradient then came back NaN where llgkw() was finite and correct:
  //
  //   par = (1, 70, 1.5, 2, 1), x = c(.10,.25,.40,.72,.99)
  //     llgkw           1386.78983044      (finite, correct)
  //     grgkw           NaN NaN NaN NaN NaN
  //     numDeriv::grad  -662.27 20.27 -6.76 472.41 -15.00
  //
  // Correcting LOG_DBL_MAX moved that boundary out by a factor of 2.3 but did
  // not remove it: beta = 70 recovered, beta = 200 still broke. Keeping the
  // quotient inside one exponential removes it entirely, since only the
  // difference of the logs has to be representable, not the reciprocal.
  //
  // As in llgkw(), a coefficient that is exactly zero never multiplies a log,
  // so a -Inf at the boundary of the support cannot become NaN.
  double d_alpha  = n / alpha;
  double d_beta   = n / beta;
  double d_lambda = n / lambda;
  double sum_log_w = 0.0, sum_log_z = 0.0;
  
  const double ca2 = (gamma * lambda - 1.0) * beta;   // alpha, w term
  const double ca3 = delta * lambda * beta;           // alpha, z term
  const double cb1 = gamma * lambda - 1.0;            // beta,  w term
  const double cb2 = delta * lambda;                  // beta,  z term
  
  for (int i = 0; i < n; i++) {
    double log_xi = std::log(x(i));
    double log_x_alpha, log_v, log_v_beta, log_w, log_w_lambda, log_z;
    gkw_log_chain(log_xi, alpha, beta, lambda,
                  log_x_alpha, log_v, log_v_beta, log_w, log_w_lambda, log_z);
    
    double log_v_beta_m1  = (beta - 1.0) * log_v;
    double log_w_lambda_m1 = (lambda - 1.0) * log_w;
    
    // dl/dalpha = n/alpha + sum log(x)
    //             - sum x^a log(x) [ (b-1)/v - (gl-1) b v^(b-1)/w
    //                                        + d l b v^(b-1) w^(l-1)/z ]
    d_alpha += log_xi;
    double a_acc = 0.0;
    if (beta != 1.0) a_acc += (beta - 1.0) * std::exp(log_x_alpha - log_v);
    if (ca2 != 0.0)  a_acc -= ca2 * std::exp(log_x_alpha + log_v_beta_m1 - log_w);
    if (ca3 != 0.0)  a_acc += ca3 * std::exp(log_x_alpha + log_v_beta_m1 +
                                             log_w_lambda_m1 - log_z);
    d_alpha -= log_xi * a_acc;
    
    // dl/dbeta = n/beta + sum log(v)
    //            - sum v^b log(v) [ (gl-1)/w - d l w^(l-1)/z ]
    d_beta += log_v;
    if (cb1 != 0.0)
      d_beta -= cb1 * gkw_mul_small_log(log_v, log_x_alpha, log_v_beta - log_w);
    if (cb2 != 0.0)
      d_beta += cb2 * gkw_mul_small_log(log_v, log_x_alpha,
                                        log_v_beta + log_w_lambda_m1 - log_z);
    
    sum_log_w += log_w;
    sum_log_z += log_z;
    
    // dl/dlambda = n/lambda + gamma sum log(w) - delta sum w^l log(w) / z
    if (delta != 0.0) {
      d_lambda -= delta * gkw_mul_small_log(log_w, log_v_beta,
                                            log_w_lambda - log_z);
    }
  }
  
  // dl/dgamma = -n[psi(g) - psi(g+d+1)] + lambda sum log(w)
  double d_gamma = -n * (R::digamma(gamma) - R::digamma(gamma + delta + 1.0)) +
    lambda * sum_log_w;
  
  // dl/ddelta = -n[psi(d+1) - psi(g+d+1)] + sum log(z)
  double d_delta = -n * (R::digamma(delta + 1.0) - R::digamma(gamma + delta + 1.0)) +
    sum_log_z;
  
  d_lambda += gamma * sum_log_w;
  
  // Validate gradient components. A component that is still not finite here is
  // a genuine boundary, not a lost quotient, so say so rather than returning a
  // silent NaN vector.
  if (!R_finite(d_alpha) || !R_finite(d_beta) || !R_finite(d_gamma) || 
      !R_finite(d_delta) || !R_finite(d_lambda)) {
      Rcpp::warning("grgkw: the log-space chain reached the boundary of the "
                    "support for at least one observation; returning NaN");
      return Rcpp::NumericVector(5, R_NaN);
  }
  
  // Return NEGATIVE gradient (for minimization of negative log-likelihood)
  grad[0] = -d_alpha;
  grad[1] = -d_beta;
  grad[2] = -d_gamma;
  grad[3] = -d_delta;
  grad[4] = -d_lambda;
  
  return grad;
}


// ============================================================================
// HESSIAN OF NEGATIVE LOG-LIKELIHOOD
// ============================================================================

/**
 * @brief Hessian Matrix of Negative Log-Likelihood for GKw Distribution
 * 
 * Computes the Hessian matrix (matrix of second partial derivatives) of
 * the negative log-likelihood for standard error estimation and
 * optimization algorithms.
 * 
 * @param par Parameter vector of length 5: (α, β, γ, δ, λ)
 * @param data Vector of observations (must be in (0,1))
 * 
 * @return NumericMatrix of dimension 5×5 containing the Hessian
 * 
 * @details
 * Computes analytical second derivatives. The Hessian is symmetric,
 * so only unique elements are computed. Returns NaN matrix for invalid inputs.
 * 
 * Parameter ordering in matrix: (α, β, γ, δ, λ) corresponding to
 * indices (0, 1, 2, 3, 4).
 * 
 * @note Exported as .hsgkw_cpp for internal package use
 */
// [[Rcpp::export(.hsgkw_cpp)]]
Rcpp::NumericMatrix hsgkw(const Rcpp::NumericVector& par, const Rcpp::NumericVector& data) {
  // Extract parameters
  double alpha  = par[0];
  double beta   = par[1];
  double gamma  = par[2];
  double delta  = par[3];
  double lambda = par[4];
  
  // Validate parameters
  if (!check_pars(alpha, beta, gamma, delta, lambda)) {
    Rcpp::NumericMatrix nanH(5, 5);
    nanH.fill(R_NaN);
    return nanH;
  }
  
  // Convert and validate data
  arma::vec x = Rcpp::as<arma::vec>(data);
  if (arma::any(x <= 0) || arma::any(x >= 1)) {
    Rcpp::NumericMatrix nanH(5, 5);
    nanH.fill(R_NaN);
    return nanH;
  }
  
  int n = x.n_elem;
  arma::mat H(5, 5, arma::fill::zeros);
  
  // ---- Constant terms (independent of observations) ----
  
  // H(λ,λ) from n*ln(λ): ∂²/∂λ² = -n/λ²
  H(4, 4) += -n / (lambda * lambda);
  
  // H(α,α) from n*ln(α): ∂²/∂α² = -n/α²
  H(0, 0) += -n / (alpha * alpha);
  
  // H(β,β) from n*ln(β): ∂²/∂β² = -n/β²
  H(1, 1) += -n / (beta * beta);
  
  // H(γ,γ) from -n*ln(B(γ,δ+1)): involves trigamma
  H(2, 2) += -n * (R::trigamma(gamma) - R::trigamma(gamma + delta + 1));
  
  // H(δ,δ) from -n*ln(B(γ,δ+1)): involves trigamma
  H(3, 3) += -n * (R::trigamma(delta + 1) - R::trigamma(gamma + delta + 1));
  
  // H(γ,δ) = H(δ,γ): mixed derivative
  H(2, 3) += n * R::trigamma(gamma + delta + 1);
  H(3, 2) = H(2, 3);
  
  // Accumulators for mixed derivatives involving λ
  double acc_gamma_lambda = 0.0;
  double acc_delta_lambda = 0.0;
  double acc_alpha_lambda = 0.0;
  double acc_beta_lambda = 0.0;

  // Set when an observation cannot be evaluated. Skipping such an observation
  // would silently compute the Hessian of a smaller sample: with beta = 500 and
  // four observations every one of them was dropped and the result was the
  // constant terms alone, H(alpha,alpha) = n/alpha^2 = 4 against a true 1996.3
  // -- finite, symmetric, free of NaN, and wrong by a factor of 499. A visible
  // failure is the correct answer until the chain is reworked in log space.
  bool degenerate = false;

  // ---- Observation-dependent terms ----
  for (int i = 0; i < n; i++) {
    double xi = x(i);
    
    // Compute A = x^α and derivatives
    double ln_xi = safe_log(xi);
    double A = safe_pow(xi, alpha);
    double dA_dalpha = A * ln_xi;
    double d2A_dalpha2 = A * ln_xi * ln_xi;
    
    // v = 1 - A and derivatives (using log-space for v)
    double log_A = alpha * ln_xi;
    double log_v = gkw_log1mexp(log_A);
    if (!R_finite(log_v)) { degenerate = true; break; }
    double v = safe_exp(log_v);
    double ln_v = log_v;
    double dv_dalpha = -dA_dalpha;
    double d2v_dalpha2 = -d2A_dalpha2;
    
    // --- L6: (β-1) ln(v) contributions ---
    double d2L6_dalpha2 = (beta - 1.0) * ((d2v_dalpha2 * v - dv_dalpha * dv_dalpha) / (v * v));
    double d2L6_dalpha_dbeta = dv_dalpha / v;
    
    // --- L7: (γλ - 1) ln(w), where w = 1 - v^β ---
    double log_v_beta = beta * log_v;
    double log_w = gkw_log1mexp(log_v_beta);
    if (!R_finite(log_w)) { degenerate = true; break; }
    double w = safe_exp(log_w);
    double ln_w = log_w;
    
    // Derivatives of w
    double v_beta_m1 = safe_pow(v, beta - 1.0);
    double dw_dv = -beta * v_beta_m1;
    double dw_dalpha = dw_dv * dv_dalpha;
    
    double d2w_dalpha2 = -beta * ((beta - 1.0) * safe_pow(v, beta - 2.0) * (dv_dalpha * dv_dalpha)
                                    + v_beta_m1 * d2v_dalpha2);
    double d2L7_dalpha2 = (gamma * lambda - 1.0) * ((d2w_dalpha2 * w - (dw_dalpha * dw_dalpha)) / (w * w));
    
    double dw_dbeta = -safe_pow(v, beta) * ln_v;
    double d2w_dbeta2 = -safe_pow(v, beta) * (ln_v * ln_v);
    double d2L7_dbeta2 = (gamma * lambda - 1.0) * ((d2w_dbeta2 * w - (dw_dbeta * dw_dbeta)) / (w * w));
    
    double d_dw_dalpha_dbeta = -safe_pow(v, beta - 1.0) * (1.0 + beta * ln_v) * dv_dalpha;
    double d2L7_dalpha_dbeta = (gamma * lambda - 1.0) * ((d_dw_dalpha_dbeta / w) - (dw_dalpha * dw_dbeta) / (w * w));
    
    // --- L8: δ ln(z), where z = 1 - w^λ ---
    double log_w_lambda = lambda * log_w;
    double log_z = gkw_log1mexp(log_w_lambda);
    if (!R_finite(log_z)) { degenerate = true; break; }
    double z = safe_exp(log_z);
    
    double w_lambda_m1 = safe_pow(w, lambda - 1.0);
    double dz_dalpha = -lambda * w_lambda_m1 * dw_dalpha;
    
    double d2z_dalpha2 = -lambda * ((lambda - 1.0) * safe_pow(w, lambda - 2.0) * (dw_dalpha * dw_dalpha)
                                      + w_lambda_m1 * d2w_dalpha2);
    double d2L8_dalpha2 = delta * ((d2z_dalpha2 * z - dz_dalpha * dz_dalpha) / (z * z));
    
    double dz_dbeta = -lambda * w_lambda_m1 * dw_dbeta;
    double d2z_dbeta2 = -lambda * ((lambda - 1.0) * safe_pow(w, lambda - 2.0) * (dw_dbeta * dw_dbeta)
                                     + w_lambda_m1 * d2w_dbeta2);
    double d2L8_dbeta2 = delta * ((d2z_dbeta2 * z - dz_dbeta * dz_dbeta) / (z * z));
    
    double d_dw_dalpha_dbeta_2 = -lambda * ((lambda - 1.0) * safe_pow(w, lambda - 2.0) * dw_dbeta * dw_dalpha
                                              + w_lambda_m1 * d_dw_dalpha_dbeta);
    double d2L8_dalpha_dbeta = delta * ((d_dw_dalpha_dbeta_2 / z) - (dz_dalpha * dz_dbeta) / (z * z));
    
    double dz_dlambda = -safe_pow(w, lambda) * ln_w;
    double d2z_dlambda2 = -safe_pow(w, lambda) * (ln_w * ln_w);
    double d2L8_dlambda2 = delta * ((d2z_dlambda2 * z - dz_dlambda * dz_dlambda) / (z * z));
    
    double d_dalpha_dz_dlambda = -w_lambda_m1 * dw_dalpha - lambda * ln_w * w_lambda_m1 * dw_dalpha;
    double d2L8_dalpha_dlambda = delta * ((d_dalpha_dz_dlambda / z) - (dz_dlambda * dz_dalpha) / (z * z));
    
    double d_dbeta_dz_dlambda = -w_lambda_m1 * dw_dbeta - lambda * ln_w * w_lambda_m1 * dw_dbeta;
    double d2L8_dbeta_dlambda = delta * ((d_dbeta_dz_dlambda / z) - (dz_dlambda * dz_dbeta) / (z * z));
    
    // Validate intermediate results
    if (!R_finite(d2L6_dalpha2) || !R_finite(d2L7_dalpha2) || !R_finite(d2L8_dalpha2) ||
        !R_finite(d2L6_dalpha_dbeta) || !R_finite(d2L7_dalpha_dbeta) || !R_finite(d2L8_dalpha_dbeta) ||
        !R_finite(d2L7_dbeta2) || !R_finite(d2L8_dbeta2) ||
        !R_finite(d2L8_dlambda2) ||
        !R_finite(dw_dalpha) || !R_finite(dw_dbeta) ||
        !R_finite(dz_dalpha) || !R_finite(dz_dbeta) ||
        !R_finite(dz_dlambda)) {
        Rcpp::NumericMatrix nanH(5, 5);
      nanH.fill(R_NaN);
      return nanH;
    }
    
    // ---- Accumulate upper-triangle Hessian contributions ----
    H(0, 0) += d2L6_dalpha2 + d2L7_dalpha2 + d2L8_dalpha2;
    H(0, 1) += d2L6_dalpha_dbeta + d2L7_dalpha_dbeta + d2L8_dalpha_dbeta;
    H(1, 1) += d2L7_dbeta2 + d2L8_dbeta2;
    H(4, 4) += d2L8_dlambda2;
    H(0, 2) += lambda * (dw_dalpha / w);
    H(1, 2) += lambda * (dw_dbeta / w);
    H(0, 3) += dz_dalpha / z;
    H(1, 3) += dz_dbeta / z;

    // λ mixed derivatives
    acc_alpha_lambda += gamma * (dw_dalpha / w) + d2L8_dalpha_dlambda;
    acc_beta_lambda  += gamma * (dw_dbeta  / w) + d2L8_dbeta_dlambda;
    acc_gamma_lambda += ln_w;
    acc_delta_lambda += dz_dlambda / z;
  }

  // An observation that could not be evaluated must not simply be left out of
  // the sum: the remaining terms would still be returned as a finite, symmetric
  // matrix that silently describes a smaller sample. Report the failure the same
  // way the intermediate-value check below already does.
  if (degenerate) {
    Rcpp::warning("hsgkw: the log-space chain underflowed for at least one observation; returning NaN");
    Rcpp::NumericMatrix nanH(5, 5);
    nanH.fill(R_NaN);
    return nanH;
  }

  // Apply accumulated λ mixed derivatives (upper triangle)
  H(0, 4) = acc_alpha_lambda;
  H(1, 4) = acc_beta_lambda;
  H(2, 4) = acc_gamma_lambda;
  H(3, 4) = acc_delta_lambda;

  // Symmetrize once after all accumulations
  H = arma::symmatu(H);

  // Return NEGATIVE Hessian (for minimization of negative log-likelihood)
  return Rcpp::wrap(-H);
}

