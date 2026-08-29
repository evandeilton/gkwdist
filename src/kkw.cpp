/**
 * @file kkw.cpp
 * @brief Kumaraswamy-Kumaraswamy (KKw) Distribution Functions
 * 
 * @details
 * This file implements the full suite of distribution functions for the
 * four-parameter Kumaraswamy-Kumaraswamy (KKw) distribution, which is a
 * sub-family of the Generalized Kumaraswamy (GKw) distribution obtained
 * by setting γ = 1.
 * 
 * **Relationship to GKw:**
 * \deqn{KKw(\alpha, \beta, \delta, \lambda) = GKw(\alpha, \beta, 1, \delta, \lambda)}
 * 
 * The KKw distribution has probability density function:
 * \deqn{
 *   f(x; \alpha, \beta, \delta, \lambda) = 
 *   \lambda \alpha \beta (\delta+1) x^{\alpha-1} (1-x^\alpha)^{\beta-1}
 *   [1-(1-x^\alpha)^\beta]^{\lambda-1} \{1-[1-(1-x^\alpha)^\beta]^\lambda\}^\delta
 * }
 * for \eqn{x \in (0,1)}.
 * 
 * The cumulative distribution function is:
 * \deqn{
 *   F(x) = 1 - \{1 - [1-(1-x^\alpha)^\beta]^\lambda\}^{\delta+1}
 * }
 * 
 * The quantile function (inverse CDF) is:
 * \deqn{
 *   Q(p) = \left\{1 - \left[1 - \left(1 - (1-p)^{1/(\delta+1)}\right)^{1/\lambda}\right]^{1/\beta}\right\}^{1/\alpha}
 * }
 * 
 * **Parameter Constraints:**
 * - \eqn{\alpha > 0} (shape parameter)
 * - \eqn{\beta > 0} (shape parameter)
 * - \eqn{\delta \geq 0} (shape parameter)
 * - \eqn{\lambda > 0} (shape parameter)
 * 
 * **Special Cases:**
 * | Distribution | Condition | Relation |
 * |--------------|-----------|----------|
 * | Exponentiated Kumaraswamy (EKw) | \eqn{\delta = 0} | KKw with δ=0 |
 * | Kumaraswamy (Kw) | \eqn{\delta = 0, \lambda = 1} | Standard Kumaraswamy |
 * 
 * **Random Variate Generation:**
 * Uses inverse transform method:
 * 1. Generate \eqn{V \sim Uniform(0,1)}
 * 2. Compute \eqn{U = 1 - (1-V)^{1/(\delta+1)}}
 * 3. Return \eqn{X = \{1 - [1 - U^{1/\lambda}]^{1/\beta}\}^{1/\alpha}}
 * 
 * **Numerical Stability:**
 * All computations use log-space arithmetic and numerically stable helper
 * functions from utils.h to prevent overflow/underflow.
 * 
 * **Implemented Functions:**
 * - dkkw(): Probability density function (PDF)
 * - pkkw(): Cumulative distribution function (CDF)
 * - qkkw(): Quantile function (inverse CDF)
 * - rkkw(): Random variate generation
 * - llkkw(): Negative log-likelihood for MLE
 * - grkkw(): Gradient of negative log-likelihood
 * - hskkw(): Hessian of negative log-likelihood
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
// Shared log-space chain for the KKw likelihood family
// ----------------------------------------------------------------------------
//
// KKw is GKw at gamma = 1, so all three transformations survive,
//
//     v = 1 - x^alpha        w = 1 - v^beta        z = 1 - w^lambda
//
// and dkkw(), llkkw(), grkkw() and hskkw() all need their logarithms. Two of
// the three have a regime where the linear quantity underflows to a boundary
// that gkw_log1mexp() cannot recover from, because its argument arrives as
// exactly 0 and log(1 - exp(0)) is -Inf:
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
// last bits wherever both are representable. The branches fire only where the
// old code produced -Inf, so ordinary data is bit-identical. This is the same
// chain gkw_log_chain() walks in gkw.cpp -- gamma never enters it -- which
// keeps KKw(a,b,d,l) == GKw(a,b,1,d,l) exact.
//
// A -Inf log_w is what put an artificial plateau on the KKw likelihood surface:
// alpha*log(min(x)) crosses -745 on entirely ordinary data, and llkkw() then
// returned +Inf where the likelihood is finite. With
// x = c(0.01, 0.3, 0.6, 0.9) and par = c(alpha, 2, 1, 1.5),
//
//   alpha = 161  llkkw = 1516.4186759096   (correct)
//   alpha = 162  llkkw = Inf               (true value 1526.0259318659)
//
// and every larger alpha stayed at Inf, which an optimiser will happily sit on.
static inline void kkw_log_chain(double log_x, double alpha, double beta,
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
// -x^alpha and -v^beta. The quotient they multiply has overflowed by exactly
// the reciprocal amount, so the product is finite while the two factors are 0
// and +Inf -- and 0 * Inf is NaN. That is how grkkw() returned a partly-NaN
// gradient where llkkw() was finite; see the note above grkkw().
//
// Substituting the magnitude back inside the exponential keeps the product
// exact and finite:  log_v * exp(E)  ->  -exp(log_x_alpha + E). The two
// familiar limits fall out of it: with E = log_v_beta - log_w the result is
// -1/beta, and the lambda term's -1/lambda likewise.
//
// This mirrors gkw_mul_small_log() in gkw.cpp exactly.
static inline double kkw_mul_small_log(double log_small, double log_magnitude,
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
 * @brief Probability Density Function of the KKw Distribution
 * 
 * Computes the density (or log-density) for the Kumaraswamy-Kumaraswamy
 * distribution at specified quantiles.
 * 
 * @param x Vector of quantiles (values in (0,1))
 * @param alpha Shape parameter vector (α > 0)
 * @param beta Shape parameter vector (β > 0)
 * @param delta Shape parameter vector (δ ≥ 0)
 * @param lambda Shape parameter vector (λ > 0)
 * @param log_prob If TRUE, returns log-density; otherwise returns density
 * 
 * @return NumericVector of density values (or log-density if log_prob=TRUE)
 * 
 * @details
 * The log-density is computed as:
 * \deqn{
 *   \log f(x) = \log(\lambda) + \log(\alpha) + \log(\beta) + \log(\delta+1)
 *   + (\alpha-1)\log(x) + (\beta-1)\log(1-x^\alpha)
 *   + (\lambda-1)\log(1-(1-x^\alpha)^\beta)
 *   + \delta\log(1-[1-(1-x^\alpha)^\beta]^\lambda)
 * }
 * 
 * @note Exported as .dkkw_cpp for internal package use
 */
// [[Rcpp::export(.dkkw_cpp)]]
Rcpp::NumericVector dkkw(
    const arma::vec& x,
    const Rcpp::NumericVector& alpha,
    const Rcpp::NumericVector& beta,
    const Rcpp::NumericVector& delta,
    const Rcpp::NumericVector& lambda,
    bool log_prob = false
) {
  // Convert R vectors to Armadillo vectors
  arma::vec a_vec(alpha.begin(), alpha.size());
  arma::vec b_vec(beta.begin(), beta.size());
  arma::vec d_vec(delta.begin(), delta.size());
  arma::vec l_vec(lambda.begin(), lambda.size());
  
  // Determine output length for recycling
  // Zero-length input: follow R's recycling convention and return an empty
  // vector, as dbeta(numeric(0), 1, 1) does. This also guards the
  // `i % vec.n_elem` recycling below against integer division by zero.
  if (x.n_elem == 0 || a_vec.n_elem == 0 || b_vec.n_elem == 0 || d_vec.n_elem == 0 || l_vec.n_elem == 0) {
    return Rcpp::NumericVector(0);
  }

  size_t N = std::max({x.n_elem, a_vec.n_elem, b_vec.n_elem, 
                      d_vec.n_elem, l_vec.n_elem});
  
  // Initialize result with appropriate default
  arma::vec out(N);
  out.fill(log_prob ? R_NegInf : 0.0);
  
  for (size_t i = 0; i < N; ++i) {
    // Extract recycled parameters
    double a = a_vec[i % a_vec.n_elem];
    double b = b_vec[i % b_vec.n_elem];
    double dd = d_vec[i % d_vec.n_elem];
    double ll = l_vec[i % l_vec.n_elem];
    double xx = x[i % x.n_elem];
    
    // Validate parameters
    if (!check_kkw_pars(a, b, dd, ll)) {
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
    
    // The closed boundaries carry the limiting density, as base R does:
    // dbeta(0, 0.5, 1) is Inf and dbeta(1, 2, 1) is 2, where this package
    // returned 0 at both ends. Anything strictly outside stays 0.
    if (xx == 0.0 || xx == 1.0) {
      out(i) = gkw_boundary_pdf(xx == 0.0, a, b, 1.0, dd, ll, log_prob);
      continue;
    }
    
    // Check support: x must be in (0, 1)
    if (xx <= 0.0 || xx >= 1.0 || !R_finite(xx)) {
      continue;
    }
    
    // ---- Log-space computation of density ----
    
    // Normalization constant: log(λαβ(δ+1))
    double logCst = safe_log(ll) + safe_log(a) + safe_log(b) + safe_log(dd + 1.0);
    
    // Compute log(x) once. Forming x^alpha in linear space and taking its
    // logarithm loses the digits that matter as x approaches 1; the chain below
    // works from alpha*log(x) directly and bridges the two points where the
    // linear quantity underflows to a boundary that log1mexp() cannot recover
    // from. The guards that used to sit between the steps -- one `continue` per
    // intermediate -- dropped the observation entirely and left the fill value,
    // 0 or -Inf in log. That is what made
    // dkkw(1e-300, 2, 1.5, 0.5, 1.2, log = TRUE) return -Inf where
    // dgkw(1e-300, 2, 1.5, 1, 0.5, 1.2, log = TRUE) gives -965.3182.
    double lx = safe_log(xx);
    double log_xalpha, log_v, log_v_beta, log_w, log_w_lambda, log_z;
    kkw_log_chain(lx, a, b, ll,
                  log_xalpha, log_v, log_v_beta, log_w, log_w_lambda, log_z);

    // Assemble log-density:
    // log(f) = logCst + (α-1)*log(x) + (β-1)*log(v) + (λ-1)*log(w) + δ*log(z)
    //
    // A coefficient that is exactly zero contributes nothing even where its
    // log is -Inf at the boundary of the support; writing 0 * -Inf gives NaN.
    double log_pdf = logCst;
    if (a != 1.0)  log_pdf += (a - 1.0) * lx;
    if (b != 1.0)  log_pdf += (b - 1.0) * log_v;
    if (ll != 1.0) log_pdf += (ll - 1.0) * log_w;
    if (dd != 0.0) log_pdf += dd * log_z;

    // A density that is still not finite here is a genuine boundary value:
    // -Inf in log, 0 on the natural scale, which is what the fill already holds.
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
 * @brief Cumulative Distribution Function of the KKw Distribution
 * 
 * Computes the cumulative probability for the Kumaraswamy-Kumaraswamy
 * distribution at specified quantiles.
 * 
 * @param q Vector of quantiles
 * @param alpha Shape parameter vector (α > 0)
 * @param beta Shape parameter vector (β > 0)
 * @param delta Shape parameter vector (δ ≥ 0)
 * @param lambda Shape parameter vector (λ > 0)
 * @param lower_tail If TRUE, returns P(X ≤ q); otherwise P(X > q)
 * @param log_p If TRUE, returns log-probability
 * 
 * @return NumericVector of cumulative probabilities
 * 
 * @details
 * The CDF is computed as:
 * \deqn{F(x) = 1 - \{1 - [1-(1-x^\alpha)^\beta]^\lambda\}^{\delta+1}}
 * 
 * @note Exported as .pkkw_cpp for internal package use
 */
// [[Rcpp::export(.pkkw_cpp)]]
Rcpp::NumericVector pkkw(
    const arma::vec& q,
    const Rcpp::NumericVector& alpha,
    const Rcpp::NumericVector& beta,
    const Rcpp::NumericVector& delta,
    const Rcpp::NumericVector& lambda,
    bool lower_tail = true,
    bool log_p = false
) {
  // Convert R vectors to Armadillo vectors
  arma::vec a_vec(alpha.begin(), alpha.size());
  arma::vec b_vec(beta.begin(), beta.size());
  arma::vec d_vec(delta.begin(), delta.size());
  arma::vec l_vec(lambda.begin(), lambda.size());
  
  // Determine output length for recycling
  // Zero-length input: follow R's recycling convention and return an empty
  // vector, as dbeta(numeric(0), 1, 1) does. This also guards the
  // `i % vec.n_elem` recycling below against integer division by zero.
  if (q.n_elem == 0 || a_vec.n_elem == 0 || b_vec.n_elem == 0 || d_vec.n_elem == 0 || l_vec.n_elem == 0) {
    return Rcpp::NumericVector(0);
  }

  size_t N = std::max({q.n_elem, a_vec.n_elem, b_vec.n_elem, 
                      d_vec.n_elem, l_vec.n_elem});
  
  arma::vec out(N);
  
  for (size_t i = 0; i < N; ++i) {
    // Extract recycled parameters
    double a = a_vec[i % a_vec.n_elem];
    double b = b_vec[i % b_vec.n_elem];
    double dd = d_vec[i % d_vec.n_elem];
    double ll = l_vec[i % l_vec.n_elem];
    double xx = q[i % q.n_elem];
    
    // Validate parameters
    if (!check_kkw_pars(a, b, dd, ll)) {
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

    // F = 1 - (1 - y^lambda)^(delta+1), y = 1 - (1 - x^alpha)^beta
    double log_y    = gkw_log1mexp(b * gkw_log1mexp(log_x_alpha));
    double log_surv = (dd + 1.0) * gkw_log1mexp(ll * log_y);
    double log_cdf  = gkw_log1mexp(log_surv);

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
 * @brief Quantile Function (Inverse CDF) of the KKw Distribution
 * 
 * Computes quantiles for the Kumaraswamy-Kumaraswamy distribution
 * given probability values.
 * 
 * @param p Vector of probabilities (values in [0,1])
 * @param alpha Shape parameter vector (α > 0)
 * @param beta Shape parameter vector (β > 0)
 * @param delta Shape parameter vector (δ ≥ 0)
 * @param lambda Shape parameter vector (λ > 0)
 * @param lower_tail If TRUE, probabilities are P(X ≤ x); otherwise P(X > x)
 * @param log_p If TRUE, probabilities are given as log(p)
 * 
 * @return NumericVector of quantiles
 * 
 * @details
 * The quantile function inverts the CDF:
 * \deqn{Q(p) = \left\{1 - \left[1 - \left(1 - (1-p)^{1/(\delta+1)}\right)^{1/\lambda}\right]^{1/\beta}\right\}^{1/\alpha}}
 * 
 * @note Exported as .qkkw_cpp for internal package use
 */
// [[Rcpp::export(.qkkw_cpp)]]
Rcpp::NumericVector qkkw(
    const arma::vec& p,
    const Rcpp::NumericVector& alpha,
    const Rcpp::NumericVector& beta,
    const Rcpp::NumericVector& delta,
    const Rcpp::NumericVector& lambda,
    bool lower_tail = true,
    bool log_p = false
) {
  // Convert R vectors to Armadillo vectors
  arma::vec a_vec(alpha.begin(), alpha.size());
  arma::vec b_vec(beta.begin(), beta.size());
  arma::vec d_vec(delta.begin(), delta.size());
  arma::vec l_vec(lambda.begin(), lambda.size());
  
  // Determine output length for recycling
  // Zero-length input: follow R's recycling convention and return an empty
  // vector, as dbeta(numeric(0), 1, 1) does. This also guards the
  // `i % vec.n_elem` recycling below against integer division by zero.
  if (p.n_elem == 0 || a_vec.n_elem == 0 || b_vec.n_elem == 0 || d_vec.n_elem == 0 || l_vec.n_elem == 0) {
    return Rcpp::NumericVector(0);
  }

  size_t N = std::max({p.n_elem, a_vec.n_elem, b_vec.n_elem, 
                      d_vec.n_elem, l_vec.n_elem});
  
  arma::vec out(N);
  
  for (size_t i = 0; i < N; ++i) {
    // Extract recycled parameters
    double a = a_vec[i % a_vec.n_elem];
    double b = b_vec[i % b_vec.n_elem];
    double dd = d_vec[i % d_vec.n_elem];
    double ll = l_vec[i % l_vec.n_elem];
    double pp = p[i % p.n_elem];
    
    // Validate parameters
    if (!check_kkw_pars(a, b, dd, ll)) {
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

    // 1 - y^lambda = (1-u)^(1/(delta+1)), then y = 1 - (1 - x^alpha)^beta
    double log_y = gkw_log1mexp(log_1mu / (dd + 1.0)) / ll;
    out(i) = std::exp(gkw_log1mexp(gkw_log1mexp(log_y) / b) / a);
  }
  
  return Rcpp::NumericVector(out.memptr(), out.memptr() + out.n_elem);
}


// ============================================================================
// RANDOM NUMBER GENERATION
// ============================================================================

/**
 * @brief Random Variate Generation for the KKw Distribution
 * 
 * Generates random samples from the Kumaraswamy-Kumaraswamy distribution
 * using the inverse transform method.
 * 
 * @param n Number of random variates to generate
 * @param alpha Shape parameter vector (α > 0)
 * @param beta Shape parameter vector (β > 0)
 * @param delta Shape parameter vector (δ ≥ 0)
 * @param lambda Shape parameter vector (λ > 0)
 * 
 * @return NumericVector of n random variates from KKw distribution
 * 
 * @details
 * Algorithm:
 * 1. Generate V ~ Uniform(0,1)
 * 2. U = 1 - (1-V)^(1/(δ+1))
 * 3. X = {1 - [1 - U^(1/λ)]^(1/β)}^(1/α)
 * 
 * @note Exported as .rkkw_cpp for internal package use
 */
// [[Rcpp::export(.rkkw_cpp)]]
Rcpp::NumericVector rkkw(
    int n,
    const Rcpp::NumericVector& alpha,
    const Rcpp::NumericVector& beta,
    const Rcpp::NumericVector& delta,
    const Rcpp::NumericVector& lambda
) {
  // n = 0 returns numeric(0), matching stats::rbeta(0, 2, 3). Only a
  // negative n is an error.
  if (n < 0) {
    Rcpp::stop("rkkw: n must be non-negative");
  }
  if (n == 0) {
    return Rcpp::NumericVector(0);
  }
  
  // Convert R vectors to Armadillo vectors
  arma::vec a_vec(alpha.begin(), alpha.size());
  arma::vec b_vec(beta.begin(), beta.size());
  arma::vec d_vec(delta.begin(), delta.size());
  arma::vec l_vec(lambda.begin(), lambda.size());
  
  // A zero-length parameter cannot be recycled. Match R's convention
  // (rbeta(3, numeric(0), 1) is NA NA NA with a warning) instead of
  // reaching the `i % vec.n_elem` recycling with a zero divisor.
  if (a_vec.n_elem == 0 || b_vec.n_elem == 0 || d_vec.n_elem == 0 || l_vec.n_elem == 0) {
    Rcpp::warning("rkkw: NAs produced");
    return Rcpp::NumericVector(n, NA_REAL);
  }

  arma::vec out(n);
  
  for (int i = 0; i < n; i++) {
    // Extract recycled parameters (direct modulo, no intermediate variable)
    double a = a_vec[i % a_vec.n_elem];
    double b = b_vec[i % b_vec.n_elem];
    double dd = d_vec[i % d_vec.n_elem];
    double ll = l_vec[i % l_vec.n_elem];
    
    // Validate parameters
    if (!check_kkw_pars(a, b, dd, ll)) {
      out(i) = NA_REAL;
      Rcpp::warning("rkkw: invalid parameters at index %d", i + 1);
      continue;
    }
    
    // Generate V ~ Uniform(0,1)
    double V = R::runif(0.0, 1.0);

    // 1 - y^lambda = (1-V)^(1/(delta+1)), then y = 1 - (1 - x^alpha)^beta.
    // Four linear subtractions used to stand between the draw and the variate.
    // The draw itself is untouched, so the RNG stream is identical to
    // before; only the inversion that follows it changes.
    double log_y = gkw_log1mexp(std::log1p(-V) / (dd + 1.0)) / ll;
    out(i) = std::exp(gkw_log1mexp(gkw_log1mexp(log_y) / b) / a);
  }
  
  return Rcpp::NumericVector(out.memptr(), out.memptr() + out.n_elem);
}


// ============================================================================
// NEGATIVE LOG-LIKELIHOOD FUNCTION
// ============================================================================

/**
 * @brief Negative Log-Likelihood for KKw Distribution
 * 
 * Computes the negative log-likelihood function for parameter estimation
 * via maximum likelihood.
 * 
 * @param par Parameter vector of length 4: (α, β, δ, λ)
 * @param data Vector of observations (must be in (0,1))
 * 
 * @return Negative log-likelihood value (scalar)
 * 
 * @details
 * The log-likelihood for n observations is:
 * \deqn{
 *   \ell(\theta) = n[\ln\lambda + \ln\alpha + \ln\beta + \ln(\delta+1)]
 *   + (\alpha-1)\sum\ln x_i + (\beta-1)\sum\ln v_i
 *   + (\lambda-1)\sum\ln w_i + \delta\sum\ln z_i
 * }
 * where:
 * - \eqn{v_i = 1 - x_i^\alpha}
 * - \eqn{w_i = 1 - v_i^\beta}
 * - \eqn{z_i = 1 - w_i^\lambda}
 * 
 * Returns +Inf for invalid parameters or data outside (0,1).
 * 
 * @note Exported as .llkkw_cpp for internal package use
 */
// [[Rcpp::export(.llkkw_cpp)]]
double llkkw(const Rcpp::NumericVector& par, const Rcpp::NumericVector& data) {
  // Validate parameter vector length
  if (par.size() < 4) return R_PosInf;
  
  // Extract parameters
  double alpha = par[0];
  double beta = par[1];
  double delta = par[2];
  double lambda = par[3];
  
  // Validate parameters using consistent checker
  if (!check_kkw_pars(alpha, beta, delta, lambda)) return R_PosInf;
  
  // Convert and validate data
  arma::vec x = Rcpp::as<arma::vec>(data);
  if (x.n_elem < 1 || arma::any(x <= 0.0) || arma::any(x >= 1.0)) return R_PosInf;
  
  int n = x.n_elem;

  // Constant term: n * [log(lambda) + log(alpha) + log(beta) + log(delta+1)]
  // since B(1, delta+1) = 1/(delta+1) for the KKw case gamma = 1
  double const_term = n * (safe_log(lambda) + safe_log(alpha) + safe_log(beta) +
                           std::log1p(delta));

  double sum_term1 = 0.0;  // (alpha-1) * sum log(x)
  double sum_term2 = 0.0;  // (beta-1)  * sum log(v)
  double sum_term3 = 0.0;  // (lambda-1)* sum log(w)
  double sum_term4 = 0.0;  // delta     * sum log(z)

  // Kept entirely in log space: v = 1-x^alpha, w = 1-v^beta and z = 1-w^lambda
  // all collapse to 0 or 1 in double precision for perfectly ordinary data,
  // and gkw_log1mexp() is the only way to retain their logarithms accurately.
  //
  // kkw_log_chain() supplies log(w) and log(z) in the two regimes where
  // log1mexp() can only answer -Inf. Without it the whole likelihood collapsed
  // to +Inf as soon as alpha*log(min(x)) crossed -745; see the note on the
  // helper. Each coefficient is tested against zero before it multiplies its
  // log, because a log legitimately reaches -Inf at the boundary of the support
  // and 0 * -Inf is NaN.
  const double c1 = alpha - 1.0;
  const double c2 = beta - 1.0;
  const double c3 = lambda - 1.0;
  const double c4 = delta;

  for (int i = 0; i < n; i++) {
    double log_xi = std::log(x(i));
    double log_x_alpha, log_v, log_v_beta, log_w, log_w_lambda, log_z;
    kkw_log_chain(log_xi, alpha, beta, lambda,
                  log_x_alpha, log_v, log_v_beta, log_w, log_w_lambda, log_z);

    if (c1 != 0.0) sum_term1 += c1 * log_xi;
    if (c2 != 0.0) sum_term2 += c2 * log_v;
    if (c3 != 0.0) sum_term3 += c3 * log_w;
    if (c4 != 0.0) sum_term4 += c4 * log_z;
  }

  double loglike = const_term + sum_term1 + sum_term2 + sum_term3 + sum_term4;

  if (!std::isfinite(loglike)) {
    return R_PosInf;
  }

  return -loglike;
}


// ============================================================================
// GRADIENT OF NEGATIVE LOG-LIKELIHOOD
// ============================================================================

/**
 * @brief Gradient of Negative Log-Likelihood for KKw Distribution
 * 
 * Computes the gradient vector of the negative log-likelihood for
 * optimization-based parameter estimation.
 * 
 * @param par Parameter vector of length 4: (α, β, δ, λ)
 * @param data Vector of observations (must be in (0,1))
 * 
 * @return NumericVector of length 4 containing partial derivatives
 *         with respect to (α, β, δ, λ)
 * 
 * @details
 * Computes analytical gradients using chain rule and log-space arithmetic
 * for numerical stability. Returns NaN vector for invalid inputs.
 * 
 * @note Exported as .grkkw_cpp for internal package use
 */
// [[Rcpp::export(.grkkw_cpp)]]
Rcpp::NumericVector grkkw(const Rcpp::NumericVector& par, const Rcpp::NumericVector& data) {
  // Validate parameter vector length
  //
  // Every rejection below says why, as grbkw() has always done. Returning a
  // silent NaN vector left the caller unable to tell a refused input from a
  // genuine boundary.
  if (par.size() < 4) {
    Rcpp::warning("Parameter vector must have at least 4 elements for KKw");
    return Rcpp::NumericVector(4, R_NaN);
  }

  // Extract parameters
  double alpha = par[0];
  double beta = par[1];
  double delta = par[2];
  double lambda = par[3];

  // Validate parameters using consistent checker
  if (!check_kkw_pars(alpha, beta, delta, lambda)) {
    Rcpp::warning("Invalid parameters in grkkw");
    return Rcpp::NumericVector(4, R_NaN);
  }

  // Convert and validate data
  //
  // has_nan() is part of the test: a NaN compares false against both bounds, so
  // without it NaN data passed straight through and the gradient came back NaN
  // with nothing said.
  arma::vec x = Rcpp::as<arma::vec>(data);
  if (x.n_elem < 1 || x.has_nan() || arma::any(x <= 0) || arma::any(x >= 1)) {
    Rcpp::warning("Data must be strictly in (0,1) and non-empty for grkkw");
    return Rcpp::NumericVector(4, R_NaN);
  }

  int n = x.n_elem;
  Rcpp::NumericVector grad(4, 0.0);

  double d_alpha  = n / alpha;
  double d_beta   = n / beta;
  double d_delta  = n / (delta + 1.0);
  double d_lambda = n / lambda;

  // Log-space building blocks, per observation:
  //   P = dlog(v)/dalpha    Q = dlog(w)/dalpha    R = dlog(w)/dbeta
  //   U = dlog(z)/dalpha    V = dlog(z)/dbeta     W = dlog(z)/dlambda
  //
  // These used to be assembled from the two ratios S = v^beta/w and
  // T = w^lambda/z as -beta*P*S, -log(v)*S, -lambda*T*Q, -lambda*T*R and
  // -T*log(w). Once x^alpha underflows, log(v) is exactly 0 and log(w) is
  // supplied by kkw_log_chain() as log(beta) + alpha*log(x); S has then
  // overflowed to +Inf while P and log(v) are 0, and 0 * Inf is NaN even though
  // both products are perfectly ordinary -- Q -> log(x) and R -> 1/beta. The
  // same happens one level down through T as x approaches 1. That is how the
  // gradient came back partly NaN where the likelihood is finite:
  //
  //   par = (200, 2, 1, 1.5), x = c(.01,.30,.60,.90)
  //     grkkw            NaN  NaN  -2  NaN
  //     grgkw(gamma=1)   9.617994  -3.000000  -2.000000  1279.626571
  //
  // Keeping each quotient inside a single exp() removes the overflow entirely,
  // since only the sum of the logs has to be representable. The direct forms
  // are algebraically identical, so ordinary data is bit-identical.
  for (int i = 0; i < n; i++) {
    double log_xi = std::log(x(i));
    d_alpha += log_xi;

    double log_x_alpha, log_v, log_v_beta, log_w, log_w_lambda, log_z;
    kkw_log_chain(log_xi, alpha, beta, lambda,
                  log_x_alpha, log_v, log_v_beta, log_w, log_w_lambda, log_z);
    d_beta += log_v;
    d_delta += log_z;

    double log_v_beta_m1   = (beta - 1.0) * log_v;
    double log_w_lambda_m1 = (lambda - 1.0) * log_w;

    double P = -log_xi * std::exp(log_x_alpha - log_v);
    double Q = beta * log_xi * std::exp(log_x_alpha + log_v_beta_m1 - log_w);
    double R = -kkw_mul_small_log(log_v, log_x_alpha, log_v_beta - log_w);
    double U = -lambda * beta * log_xi *
      std::exp(log_x_alpha + log_v_beta_m1 + log_w_lambda_m1 - log_z);
    double V = lambda * kkw_mul_small_log(log_v, log_x_alpha,
                                          log_v_beta + log_w_lambda_m1 - log_z);
    double W = -kkw_mul_small_log(log_w, log_v_beta, log_w_lambda - log_z);

    d_alpha  += (beta - 1.0) * P + (lambda - 1.0) * Q + delta * U;
    d_beta   += (lambda - 1.0) * R + delta * V;
    d_lambda += log_w + delta * W;
  }

  // Final validity check
  //
  // A component that is still not finite here is a genuine boundary, and the
  // whole vector goes with it. Returning the surviving components alongside a
  // NaN or an Inf is worse than returning nothing: the caller gets a gradient
  // that looks partly usable and is not. grbkw() has always failed uniformly,
  // and grkkw() did not -- with alpha = beta = 1e-8 and lambda = 1e300 it
  // returned c(1.378417e+307, -Inf, -2, 31.43853) and said nothing.
  if (!std::isfinite(d_alpha) || !std::isfinite(d_beta) ||
      !std::isfinite(d_delta) || !std::isfinite(d_lambda)) {
    Rcpp::warning("Gradient calculation produced non-finite values in grkkw");
    return Rcpp::NumericVector(4, R_NaN);
  }

  // Return NEGATIVE gradient (for minimization)
  grad[0] = -d_alpha;
  grad[1] = -d_beta;
  grad[2] = -d_delta;
  grad[3] = -d_lambda;

  return grad;
}


// ============================================================================
// HESSIAN OF NEGATIVE LOG-LIKELIHOOD
// ============================================================================

/**
 * @brief Hessian Matrix of Negative Log-Likelihood for KKw Distribution
 * 
 * Computes the Hessian matrix (matrix of second partial derivatives) of
 * the negative log-likelihood for standard error estimation and
 * optimization algorithms.
 * 
 * @param par Parameter vector of length 4: (α, β, δ, λ)
 * @param data Vector of observations (must be in (0,1))
 * 
 * @return NumericMatrix of dimension 4×4 containing the Hessian
 * 
 * @details
 * Computes analytical second derivatives. The Hessian is symmetric.
 * Parameter ordering: (α, β, δ, λ) → indices (0, 1, 2, 3).
 * 
 * Returns NaN matrix for invalid inputs.
 * 
 * @note Exported as .hskkw_cpp for internal package use
 */
// [[Rcpp::export(.hskkw_cpp)]]
Rcpp::NumericMatrix hskkw(const Rcpp::NumericVector& par, const Rcpp::NumericVector& data) {
  // Initialize NaN matrix for error cases
  Rcpp::NumericMatrix nanH(4, 4);
  nanH.fill(R_NaN);

  // Validate parameter vector length
  //
  // Every rejection below says why, as hsbkw() has always done. Returning a
  // silent NaN matrix left the caller unable to tell a refused input from a
  // genuine boundary.
  if (par.size() < 4) {
    Rcpp::warning("Parameter vector must have at least 4 elements for KKw");
    return nanH;
  }

  // Extract parameters
  double alpha = par[0];
  double beta = par[1];
  double delta = par[2];
  double lambda = par[3];

  // Validate parameters using consistent checker
  if (!check_kkw_pars(alpha, beta, delta, lambda)) {
    Rcpp::warning("Invalid parameters in hskkw");
    return nanH;
  }

  // Convert and validate data
  //
  // has_nan() is part of the test: a NaN compares false against both bounds, so
  // without it NaN data passed straight through and hskkw() returned a matrix
  // with 15 NaN entries and one finite one -- which looks partly usable.
  arma::vec x = Rcpp::as<arma::vec>(data);
  if (x.n_elem < 1 || x.has_nan() || arma::any(x <= 0) || arma::any(x >= 1)) {
    Rcpp::warning("Data must be strictly in (0,1) and non-empty for hskkw");
    return nanH;
  }

  int n = x.n_elem;

  // Initialize Hessian matrix
  arma::mat H(4, 4, arma::fill::zeros);

  // Constant diagonal terms
  H(0, 0) = -n / (alpha * alpha);
  H(1, 1) = -n / (beta * beta);
  H(2, 2) = -n / std::pow(delta + 1.0, 2.0);
  H(3, 3) = -n / (lambda * lambda);

  // Same log-space blocks as the gradient (P, S, Q, R, T, U, V, W), plus the
  // derivatives obtained by differentiating each logarithm once more:
  //   dP/dalpha = P*(log(x) - P)                dQ/dalpha = Q*(log(x) - P + beta*P - Q)
  //   dQ/dbeta  = -P*S*(1 + beta*(log(v) - R))  dR/dbeta  = R*(log(v) - R)
  //   dT/dalpha = T*(lambda*Q - U)              dT/dbeta  = T*(lambda*R - V)
  //   dT/dlambda = T*(log(w) - W)               dW/dlambda = W*(log(w) - W)
  // None of these is skipped at a degenerate parameter value: the mixed
  // derivatives lose the (beta-1), (lambda-1) or delta factor when they are
  // differentiated in that same parameter, so they survive there.
  for (int i = 0; i < n; i++) {
    double log_xi = std::log(x(i));

    double log_x_alpha, log_v, log_v_beta, log_w, log_w_lambda, log_z;
    kkw_log_chain(log_xi, alpha, beta, lambda,
                  log_x_alpha, log_v, log_v_beta, log_w, log_w_lambda, log_z);

    double log_v_beta_m1   = (beta - 1.0) * log_v;
    double log_w_lambda_m1 = (lambda - 1.0) * log_w;

    // Same rewrite as the gradient. The two ratios S = v^beta/w and
    // T = w^lambda/z overflow to +Inf on their own long before any product they
    // belong to is large, and the factors they meet there are 0, so every block
    // built from them became NaN. Each is folded into the exp() of the sum of
    // logs it appears in, which is what the products actually require:
    //
    //   PS  = -P*S     TPS = -P*S*T     TQ = T*Q = beta*TPS
    //   TR  =  T*R
    //
    // and the second derivatives below are rewritten in those terms. They are
    // algebraically identical, so ordinary data is bit-identical.
    double P   = -log_xi * std::exp(log_x_alpha - log_v);
    double PS  =  log_xi * std::exp(log_x_alpha + log_v_beta_m1 - log_w);
    double Q   =  beta * PS;
    double R   = -kkw_mul_small_log(log_v, log_x_alpha, log_v_beta - log_w);
    double TPS =  log_xi * std::exp(log_x_alpha + log_v_beta_m1 +
                                    log_w_lambda_m1 - log_z);
    double TQ  =  beta * TPS;
    double TR  = -kkw_mul_small_log(log_v, log_x_alpha,
                                    log_v_beta + log_w_lambda_m1 - log_z);
    double U   = -lambda * TQ;
    double V   = -lambda * TR;
    double W   = -kkw_mul_small_log(log_w, log_v_beta, log_w_lambda - log_z);

    double dP_dalpha = P * (log_xi - P);
    double dQ_dalpha = Q * (log_xi - P + beta * P - Q);
    double dQ_dbeta  = PS * (1.0 + beta * (log_v - R));
    double dR_dbeta  = R * (log_v - R);

    double dU_dalpha = -lambda * TQ * (lambda * Q - U + log_xi - P + beta * P - Q);
    double dU_dbeta  = -lambda * (TQ * (lambda * R - V) +
                                  TPS * (1.0 + beta * (log_v - R)));
    double dU_dlambda = -TQ * (1.0 + lambda * (log_w - W));
    double dV_dbeta  = -lambda * TR * (lambda * R - V + log_v - R);
    double dV_dlambda = -TR * (1.0 + lambda * (log_w - W));
    double dW_dlambda = W * (log_w - W);

    // alpha row
    H(0, 0) += (beta - 1.0) * dP_dalpha + (lambda - 1.0) * dQ_dalpha + delta * dU_dalpha;
    H(0, 1) += P + (lambda - 1.0) * dQ_dbeta + delta * dU_dbeta;
    H(0, 2) += U;
    H(0, 3) += Q + delta * dU_dlambda;

    // beta row
    H(1, 1) += (lambda - 1.0) * dR_dbeta + delta * dV_dbeta;
    H(1, 2) += V;
    H(1, 3) += R + delta * dV_dlambda;

    // delta and lambda rows
    H(2, 3) += W;
    H(3, 3) += delta * dW_dlambda;
  }

  // Only the upper triangle is accumulated above; mirror it
  H = arma::symmatu(H);

  // Final validity check
  //
  // An entry that is still not finite here is a genuine boundary, and the whole
  // matrix goes with it. A Hessian holding two NaN entries among fourteen finite
  // ones is not a smaller, weaker answer -- it is one that will be inverted for
  // a standard error and silently produce nonsense. hsbkw() has always failed
  // uniformly, and hskkw() did not: with alpha = beta = 1e-8 and lambda = 1e300
  // it returned exactly that, and said nothing.
  if (!H.is_finite()) {
    Rcpp::warning("Hessian calculation produced non-finite values in hskkw");
    return nanH;
  }

  // Return NEGATIVE Hessian (for minimization)
  return Rcpp::wrap(-H);
}

