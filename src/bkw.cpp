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

// ----------------------------------------------------------------------------
// Shared log-space chain for the BKw likelihood family
// ----------------------------------------------------------------------------
//
// BKw is GKw at lambda = 1, so the third transformation collapses:
//
//     v = 1 - x^alpha        w = 1 - v^beta        z = 1 - w = v^beta
//
// leaving two logarithms to compute. The second one has a regime where the
// linear quantity underflows to a boundary that gkw_log1mexp() cannot recover
// from, because its argument arrives as exactly 0 and log(1 - exp(0)) is -Inf:
//
//   x -> 0 :  v -> 1, so log_v = log1p(-x^alpha) ~ -x^alpha, which underflows
//             to 0 once x^alpha < 5e-324. Then log1mexp(beta*log_v) is -Inf.
//             But w = 1 - v^beta -> beta * x^alpha, so
//             log_w = log(beta) + alpha*log(x).
//
// The substitution is the first-order limit and agrees with log1mexp() to the
// last bits wherever both are representable, and it fires only where the old
// code produced -Inf, so ordinary data is bit-identical. This is the same
// branch gkw_log_chain() takes in gkw.cpp, which keeps
// BKw(a,b,g,d) == GKw(a,b,g,d,1) exact.
//
// A -Inf log_w is what put an artificial plateau on the BKw likelihood
// surface: alpha*log(min(x)) crosses -745 on entirely ordinary data, and
// llbkw() then returned +Inf where the likelihood is finite. With
// x = c(0.01, 0.3, 0.6, 0.9) and par = c(alpha, 2, 1.5, 1),
//
//   alpha = 161  llbkw = 1515.5261017902   (correct)
//   alpha = 162  llbkw = Inf               (true value 1525.1333577380)
//
// and every larger alpha stayed at Inf, which an optimiser will happily sit on.
static inline void bkw_log_chain(double log_x, double alpha, double beta,
                                 double& log_x_alpha, double& log_v,
                                 double& log_v_beta, double& log_w) {
  log_x_alpha = alpha * log_x;
  log_v       = gkw_log1mexp(log_x_alpha);
  log_v_beta  = beta * log_v;
  log_w       = (log_v == 0.0) ? (std::log(beta) + log_x_alpha)
                               : gkw_log1mexp(log_v_beta);
}

// log_v underflows to 0 in the regime where its true value is -x^alpha. The
// quotient it multiplies has overflowed by exactly the reciprocal amount, so
// the product is finite while the two factors are 0 and +Inf -- and 0 * Inf is
// NaN. That is how grbkw() returned an all-NaN gradient where the true
// gradient is ordinary; see the note above grbkw().
//
// Substituting the magnitude back inside the exponential keeps the product
// exact and finite:  log_v * exp(E)  ->  -exp(log_x_alpha + E). The familiar
// limit falls out of it: with E = log_v_beta - log_w the result is -1/beta.
//
// This mirrors gkw_mul_small_log() in gkw.cpp exactly.
static inline double bkw_mul_small_log(double log_small, double log_magnitude,
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
    
    // Missing and undefined input propagates, as base R does: dbeta(NA) is NA
    // and dbeta(NaN) is NaN. Both used to fall into the "outside the support"
    // branch below -- !R_finite() is true for a NaN -- and were silently
    // replaced by the fill value, 0 or -Inf in log. NA_REAL is itself a NaN
    // carrying a distinguishing payload, so R_IsNA() must be asked first.
    if (ISNAN(xx)) {
      result(i) = R_IsNA(xx) ? NA_REAL : R_NaN;
      continue;
    }
    
    // The closed boundaries carry the limiting density, as base R does:
    // dbeta(0, 0.5, 1) is Inf and dbeta(1, 2, 1) is 2, where this package
    // returned 0 at both ends. Anything strictly outside stays 0.
    if (xx == 0.0 || xx == 1.0) {
      result(i) = gkw_boundary_pdf(xx == 0.0, a, b, g, d, 1.0, log_prob);
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
    
    // Compute log(x) once. Forming x^alpha in linear space and taking its
    // logarithm loses the digits that matter as x approaches 1; the chain below
    // works from alpha*log(x) directly and bridges the point where the linear
    // quantity underflows to a boundary that log1mexp() cannot recover from.
    // The guards that used to sit between the two steps -- one `continue` per
    // intermediate -- dropped the observation entirely and left the fill value,
    // 0 or -Inf in log. That is what made
    // dbkw(1e-300, 2, 1.5, 1.2, 0.5, log = TRUE) return -Inf where
    // dgkw(1e-300, 2, 1.5, 1.2, 0.5, 1, log = TRUE) gives -965.2651.
    double lx = safe_log(xx);
    double log_xalpha, log_v, log_v_beta, log_w;
    bkw_log_chain(lx, a, b, log_xalpha, log_v, log_v_beta, log_w);

    // Exponents: (β(δ+1) - 1) on log(v) and (γ - 1) on log(w)
    double exponent1 = b * (d + 1.0) - 1.0;
    double exponent2 = g - 1.0;

    // Assemble log-density:
    // log(f) = log_const + (α-1)*log(x) + (β(δ+1)-1)*log(v) + (γ-1)*log(w)
    //
    // A coefficient that is exactly zero contributes nothing even where its
    // log is -Inf at the boundary of the support; writing 0 * -Inf gives NaN.
    double log_pdf = log_const;
    if (a != 1.0)         log_pdf += (a - 1.0) * lx;
    if (exponent1 != 0.0) log_pdf += exponent1 * log_v;
    if (exponent2 != 0.0) log_pdf += exponent2 * log_w;

    // A density that is still not finite here is a genuine boundary value:
    // -Inf in log, 0 on the natural scale, which is what the fill already holds.
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
    
    // Missing and undefined input propagates, as in d*() above.
    if (ISNAN(xx)) {
      res(i) = R_IsNA(xx) ? NA_REAL : R_NaN;
      continue;
    }
    
    // Handle boundary: q ≤ 0
    if (xx <= 0.0) {
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
    //
    // The remaining loss was in the argument rather than the tail flag. As
    // x -> 1 the quantity b*log(v) runs off to -Inf, -expm1 of it returns
    // exactly 1, and R::pbeta(1, ., ., lower = FALSE) returns exactly 0 --
    // pbkw(1 - 1e-06, 2, 3, 1.5, 2, lower.tail = FALSE) gave 0 for a tail that
    // is still representable sixteen decades further down.
    //
    // 1 - z is exp(b*log(v)), the same exponent, at full relative accuracy, and
    // I_z(a,b) = 1 - I_{1-z}(b,a) is exact. Reflecting above the crossover sends
    // the small quantity into pbeta. Below it the direct form already holds the
    // small quantity, so the lower tail and any upper tail with z <= 1/2 are
    // untouched. LOG1MEXP_CROSSOVER is -log(2).
    double log_v_beta = b * gkw_log1mexp(log_x_alpha);   // log(1 - z)
    if (!lower_tail && log_v_beta < LOG1MEXP_CROSSOVER) {
      res(i) = R::pbeta(std::exp(log_v_beta), d + 1.0, g, /*lower*/ 1, log_p);
    } else {
      res(i) = R::pbeta(-std::expm1(log_v_beta), g, d + 1.0, lower_tail, log_p);
    }
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
    
    // ---- Normalise the probability, without leaving log space ----
    // The former code did exp(log p) and then 1 - p in linear space. The first
    // flushed the deep tail to zero (qbeta_(-1000, 2, 3, log.p = TRUE) gave 0
    // against a true 2.25e-218); the second cost the upper tail.
    // Missing and undefined input propagates.
    if (ISNAN(pp)) {
      res(i) = R_IsNA(pp) ? NA_REAL : R_NaN;
      continue;
    }
    
    // A probability outside its range has no quantile. The wrappers have
    // always warned that such a value "will produce NaN"; the C++ saturated
    // at 0 or 1 instead, which are outside the open support and which
    // defensive code testing is.nan() could not detect.
    if (log_p ? (pp > 0.0) : (pp < 0.0 || pp > 1.0)) {
      res(i) = R_NaN;
      continue;
    }

    // z = I^-1_{gamma,delta+1}(u), then x = [1 - (1-z)^(1/beta)]^(1/alpha).
    // What the chain needs is log(1-z), and which route keeps its digits
    // depends on where z sits: log1p(-z) is exact while z is small, and once
    // z passes 1/2 the accurate value of 1-z comes straight from R::qbeta via
    // the symmetry I_z(a,b) = 1 - I_{1-z}(b,a). Taking the symmetry in both
    // regimes is what returns 1 for small u and loses the quantile entirely.
    double z = R::qbeta(pp, g, d + 1.0, lower_tail, log_p);
    double log_1mz = (z <= 0.5)
      ? std::log1p(-z)
      : std::log(R::qbeta(pp, d + 1.0, g, !lower_tail, log_p));
    res(i) = std::exp(gkw_log1mexp(log_1mz / b) / a);
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
  // n = 0 returns numeric(0), matching stats::rbeta(0, 2, 3). Only a
  // negative n is an error.
  if (n < 0) {
    Rcpp::stop("rbkw: n must be non-negative");
  }
  if (n == 0) {
    return Rcpp::NumericVector(0);
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

  // One warning per call, not one per element. Warning inside the loop cost 51x
  // on 50,000 values with half the parameters invalid, and under
  // options(warn = 2) each call longjmps out of the loop through C++ frames
  // holding live Armadillo objects. R's own convention is a single
  // "NAs produced" per call.
  bool bad_par = false;

  for (int i = 0; i < n; ++i) {
    // Extract recycled parameters (direct modulo, no intermediate variable)
    double a = alpha_vec[i % alpha_vec.n_elem];
    double b = beta_vec[i % beta_vec.n_elem];
    double g = gamma_vec[i % gamma_vec.n_elem];
    double d = delta_vec[i % delta_vec.n_elem];
    
    // Validate parameters
    if (!check_bkw_pars(a, b, g, d)) {
      out(i) = NA_REAL;
      bad_par = true;
      continue;
    }
    
    // Generate V ~ Beta(gamma, delta+1)
    double V = R::rbeta(g, d + 1.0);

    // V is the incomplete-beta variate z, and x = [1 - (1-z)^(1/beta)]^(1/alpha).
    // The former code formed 1.0 - V: for V below 1.1e-16 that rounds to exactly
    // 1 and the generator returned 0, a value outside the open support that
    // llbkw() then rejects. With gamma = 0.02 it fabricated a zero for 48.6% of
    // the sample, while R::rbeta itself never returned one. log1p(-V) keeps the
    // digits that subtraction threw away.
    // The draw itself is untouched, so the RNG stream is identical to
    // before; only the inversion that follows it changes.
    out(i) = std::exp(gkw_log1mexp(std::log1p(-V) / b) / a);
  }

  if (bad_par) {
    Rcpp::warning("rbkw: NAs produced");
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
  //
  // bkw_log_chain() supplies log(w) in the regime where v underflows to 1 and
  // log1mexp() can only answer -Inf. Without it the whole likelihood collapsed
  // to +Inf as soon as alpha*log(min(x)) crossed -745; see the note on the
  // helper. Each coefficient is tested against zero before it multiplies its
  // log, because a log legitimately reaches -Inf at the boundary of the support
  // and 0 * -Inf is NaN.
  double exp1 = b * (d + 1.0) - 1.0;
  double exp2 = g - 1.0;
  double sum2 = 0.0;
  double sum3 = 0.0;

  for (int i = 0; i < n; i++) {
    double log_x_alpha, log_v, log_v_beta, log_w;
    bkw_log_chain(lx(i), a, b, log_x_alpha, log_v, log_v_beta, log_w);

    if (exp1 != 0.0) sum2 += exp1 * log_v;
    if (exp2 != 0.0) sum3 += exp2 * log_w;
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
  //   Q = dlog(w)/dalpha = beta*log(x)*exp(log(x^alpha) + (beta-1)*log(v) - log(w))
  //   R = dlog(w)/dbeta  = -log(v)*exp(beta*log(v) - log(w))
  //
  // Q and R used to be built from S = v^beta/w = exp(beta*log(v) - log(w)) as
  // -beta*P*S and -log(v)*S. Once x^alpha underflows, log(v) is exactly 0 and
  // log(w) is supplied by bkw_log_chain() as log(beta) + alpha*log(x); S has
  // then overflowed to +Inf while P and log(v) are 0, and 0 * Inf is NaN even
  // though both products are perfectly ordinary -- Q -> log(x) and R -> 1/beta.
  // That is how the entire gradient came back NaN where the likelihood is
  // finite and numDeriv gives an ordinary answer:
  //
  //   par = (200, 2, 1.5, 1), x = c(.01,.30,.60,.90)
  //     grbkw            NaN  NaN  NaN  NaN
  //     grgkw(lambda=1)  9.617994  -3.000000  1278.026571  -2.721489
  //
  // Keeping each quotient inside a single exp() removes the overflow entirely,
  // since only the sum of the logs has to be representable. The direct forms
  // are algebraically identical, so ordinary data is bit-identical.
  for (int i = 0; i < n; i++) {
    double log_xi = std::log(x(i));
    sum_log_x += log_xi;

    double log_x_alpha, log_v, log_v_beta, log_w;
    bkw_log_chain(log_xi, alpha, beta, log_x_alpha, log_v, log_v_beta, log_w);
    sum_log_v += log_v;
    sum_log_w += log_w;

    double log_v_beta_m1 = (beta - 1.0) * log_v;

    double P = -log_xi * std::exp(log_x_alpha - log_v);
    double Q = beta * log_xi * std::exp(log_x_alpha + log_v_beta_m1 - log_w);
    double R = -bkw_mul_small_log(log_v, log_x_alpha, log_v_beta - log_w);

    // A coefficient that is exactly zero must not multiply a log that reaches
    // the boundary: 0 * -Inf is NaN.
    double a_w = (term_beta_delta != 0.0) ? term_beta_delta * P : 0.0;
    double a_g = (term_gamma != 0.0)      ? term_gamma * Q      : 0.0;
    acc_alpha += a_w + a_g;
    acc_beta  += (term_gamma != 0.0) ? term_gamma * R : 0.0;
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

    double log_x_alpha, log_v, log_v_beta, log_w;
    bkw_log_chain(log_xi, alpha, beta, log_x_alpha, log_v, log_v_beta, log_w);

    double log_v_beta_m1 = (beta - 1.0) * log_v;

    // Same rewrite as the gradient: PS stands for -P*S and Q for -beta*P*S,
    // each written as one exp() of a sum of logs so that neither factor has to
    // be representable on its own. Built from S = exp(beta*log(v) - log(w))
    // they were 0 * Inf = NaN once x^alpha underflowed, and the whole Hessian
    // came back NaN through the is_finite() check below.
    double P  = -log_xi * std::exp(log_x_alpha - log_v);
    double PS =  log_xi * std::exp(log_x_alpha + log_v_beta_m1 - log_w);
    double Q  =  beta * PS;
    double R  = -bkw_mul_small_log(log_v, log_x_alpha, log_v_beta - log_w);

    double dP_dalpha = P * (log_xi - P);
    double dQ_dalpha = Q * (log_xi - P + beta * P - Q);
    double dQ_dbeta  = PS * (1.0 + beta * (log_v - R));
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
