/**
 * @file bpmc.cpp
 * @brief Beta-Power / McDonald (BP/Mc) Distribution Functions
 * 
 * @details
 * This file implements the full suite of distribution functions for the
 * three-parameter Beta-Power (BP) or McDonald (Mc) distribution, which is
 * a sub-family of the Generalized Kumaraswamy (GKw) distribution obtained
 * by setting α = 1 and β = 1.
 * 
 * **Relationship to GKw:**
 * \deqn{BP(\gamma, \delta, \lambda) = GKw(1, 1, \gamma, \delta, \lambda)}
 * 
 * The BP distribution has probability density function:
 * \deqn{
 *   f(x; \gamma, \delta, \lambda) = 
 *   \frac{\lambda}{B(\gamma, \delta+1)} x^{\gamma\lambda-1} (1-x^\lambda)^\delta
 * }
 * for \eqn{x \in (0,1)}, where \eqn{B(\cdot,\cdot)} is the Beta function.
 * 
 * **Derivation from GKw:**
 * Setting α=1 and β=1 in the GKw PDF:
 * - \eqn{x^{\alpha-1} = x^0 = 1}
 * - \eqn{(1-x^\alpha)^{\beta-1} = (1-x)^0 = 1}
 * - \eqn{[1-(1-x^\alpha)^\beta] = [1-(1-x)] = x}
 * - \eqn{[1-(1-x^\alpha)^\beta]^{\gamma\lambda-1} = x^{\gamma\lambda-1}}
 * - \eqn{\{1-[1-(1-x^\alpha)^\beta]^\lambda\}^\delta = (1-x^\lambda)^\delta}
 * - The Beta function becomes: \eqn{B(\gamma, \delta+1)}
 * 
 * The cumulative distribution function is:
 * \deqn{
 *   F(x) = I_{x^\lambda}(\gamma, \delta+1)
 * }
 * where \eqn{I_y(a,b)} is the regularized incomplete Beta function.
 * 
 * The quantile function (inverse CDF) is:
 * \deqn{
 *   Q(p) = \left[Q_{Beta}(p; \gamma, \delta+1)\right]^{1/\lambda}
 * }
 * 
 * **Parameter Constraints:**
 * - \eqn{\gamma > 0} (shape parameter)
 * - \eqn{\delta \geq 0} (shape parameter)
 * - \eqn{\lambda > 0} (power parameter)
 * 
 * **Special Cases:**
 * | Distribution | Condition | Relation |
 * |--------------|-----------|----------|
 * | Power function | \eqn{\delta = 0} | BP(γ, 0, λ) |
 * | Beta | \eqn{\lambda = 1} | Beta(γ, δ+1) |
 * 
 * **Random Variate Generation:**
 * Uses transformation method:
 * 1. Generate \eqn{U \sim Beta(\gamma, \delta+1)}
 * 2. Return \eqn{X = U^{1/\lambda}}
 * 
 * **Numerical Stability:**
 * All computations use log-space arithmetic and numerically stable helper
 * functions from utils.h to prevent overflow/underflow.
 * 
 * **Implemented Functions:**
 * - dmc(): Probability density function (PDF)
 * - pmc(): Cumulative distribution function (CDF)
 * - qmc(): Quantile function (inverse CDF)
 * - rmc(): Random variate generation
 * - llmc(): Negative log-likelihood for MLE
 * - grmc(): Gradient of negative log-likelihood
 * - hsmc(): Hessian of negative log-likelihood
 * 
 * **Alternative Names:**
 * This distribution is also known as:
 * - McDonald distribution (McDonald, 1984)
 * - Generalized Beta of the first kind
 * - Libby-Novick Beta
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
 * @brief Probability Density Function of the BP/McDonald Distribution
 * 
 * Computes the density (or log-density) for the Beta-Power distribution
 * at specified quantiles.
 * 
 * @param x Vector of quantiles (values in (0,1))
 * @param gamma Shape parameter vector (γ > 0)
 * @param delta Shape parameter vector (δ ≥ 0)
 * @param lambda Power parameter vector (λ > 0)
 * @param log_prob If TRUE, returns log-density; otherwise returns density
 * 
 * @return NumericVector of density values (or log-density if log_prob=TRUE)
 * 
 * @details
 * The log-density is computed as:
 * \deqn{
 *   \log f(x) = \log(\lambda) - \log B(\gamma, \delta+1)
 *   + (\gamma\lambda-1)\log(x) + \delta\log(1-x^\lambda)
 * }
 * 
 * @note Exported as .dmc_cpp for internal package use
 */
// [[Rcpp::export(.dmc_cpp)]]
Rcpp::NumericVector dmc(
    const arma::vec& x,
    const Rcpp::NumericVector& gamma,
    const Rcpp::NumericVector& delta,
    const Rcpp::NumericVector& lambda,
    bool log_prob = false
) {
  // Convert R vectors to Armadillo vectors
  arma::vec g_vec(gamma.begin(), gamma.size());
  arma::vec d_vec(delta.begin(), delta.size());
  arma::vec l_vec(lambda.begin(), lambda.size());
  
  // Determine output length for recycling
  // Zero-length input: follow R's recycling convention and return an empty
  // vector, as dbeta(numeric(0), 1, 1) does. This also guards the
  // `i % vec.n_elem` recycling below against integer division by zero.
  if (x.n_elem == 0 || g_vec.n_elem == 0 || d_vec.n_elem == 0 || l_vec.n_elem == 0) {
    return Rcpp::NumericVector(0);
  }

  size_t N = std::max({x.n_elem, g_vec.n_elem, d_vec.n_elem, l_vec.n_elem});
  
  // Initialize result with appropriate default
  arma::vec out(N);
  out.fill(log_prob ? R_NegInf : 0.0);
  
  for (size_t i = 0; i < N; i++) {
    // Extract recycled parameters
    double gg = g_vec[i % g_vec.n_elem];
    double dd = d_vec[i % d_vec.n_elem];
    double ll = l_vec[i % l_vec.n_elem];
    double xx = x[i % x.n_elem];
    
    // Validate parameters
    if (!check_bp_pars(gg, dd, ll)) {
      continue;
    }
    
    // Check support: x must be in (0, 1)
    if (xx <= 0.0 || xx >= 1.0 || !R_finite(xx)) {
      continue;
    }
    
    // ---- Log-space computation of density ----
    
    // Normalization constant: log(λ / B(γ, δ+1))
    double logB = R::lbeta(gg, dd + 1.0);
    double logCst = safe_log(ll) - logB;
    
    // Exponent: γλ - 1
    double exponent = gg * ll - 1.0;
    double lx = safe_log(xx);
    
    // Term 1: (γλ - 1) * log(x)
    double term1 = exponent * lx;

    // Term 2: δ * log(1 - x^λ), formed from λ·log(x) so the subtraction never
    // happens in linear space. Building x^λ first and then log(1 - x^λ) throws
    // away exactly the digits that matter as x approaches 1: doubles are spaced
    // 2.2e-16 apart there, so 1 - x^λ carries an absolute error of one ulp of 1
    // however small it is, and x^λ rounds to 1 outright at 1 - x ~ 1e-16. The
    // former `if (x_pow_l >= 1.0) continue;` then returned a density of zero
    // where the true density is finite. At γ=1.5, δ=2, λ=0.8 the log-density
    // was off by 0.446 nats at x = 1 - 1e-16 and dmc() disagreed with
    // dgkw(x, 1, 1, γ, δ, λ), which is the same distribution.
    //
    // Guarded on δ because δ = 0 with log(1 - x^λ) at the boundary would give
    // 0 * -Inf = NaN; the same guard llmc(), grmc() and dgkw() already carry.
    double term2 = 0.0;
    if (dd != 0.0) {
      term2 = dd * gkw_log1mexp(ll * lx);
    }

    // Assemble log-density
    double log_pdf = logCst + term1 + term2;
    
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
 * @brief Cumulative Distribution Function of the BP/McDonald Distribution
 * 
 * Computes the cumulative probability for the Beta-Power distribution
 * at specified quantiles.
 * 
 * @param q Vector of quantiles
 * @param gamma Shape parameter vector (γ > 0)
 * @param delta Shape parameter vector (δ ≥ 0)
 * @param lambda Power parameter vector (λ > 0)
 * @param lower_tail If TRUE, returns P(X ≤ q); otherwise P(X > q)
 * @param log_p If TRUE, returns log-probability
 * 
 * @return NumericVector of cumulative probabilities
 * 
 * @details
 * The CDF is computed as:
 * \deqn{F(x) = I_{x^\lambda}(\gamma, \delta+1)}
 * where \eqn{I_y(a,b)} is the regularized incomplete Beta function.
 * 
 * @note Exported as .pmc_cpp for internal package use
 */
// [[Rcpp::export(.pmc_cpp)]]
Rcpp::NumericVector pmc(
    const arma::vec& q,
    const Rcpp::NumericVector& gamma,
    const Rcpp::NumericVector& delta,
    const Rcpp::NumericVector& lambda,
    bool lower_tail = true,
    bool log_p = false
) {
  // Convert R vectors to Armadillo vectors
  arma::vec g_vec(gamma.begin(), gamma.size());
  arma::vec d_vec(delta.begin(), delta.size());
  arma::vec l_vec(lambda.begin(), lambda.size());
  
  // Determine output length for recycling
  // Zero-length input: follow R's recycling convention and return an empty
  // vector, as dbeta(numeric(0), 1, 1) does. This also guards the
  // `i % vec.n_elem` recycling below against integer division by zero.
  if (q.n_elem == 0 || g_vec.n_elem == 0 || d_vec.n_elem == 0 || l_vec.n_elem == 0) {
    return Rcpp::NumericVector(0);
  }

  size_t N = std::max({q.n_elem, g_vec.n_elem, d_vec.n_elem, l_vec.n_elem});
  
  arma::vec out(N);
  
  for (size_t i = 0; i < N; i++) {
    // Extract recycled parameters
    double gg = g_vec[i % g_vec.n_elem];
    double dd = d_vec[i % d_vec.n_elem];
    double ll = l_vec[i % l_vec.n_elem];
    double xx = q[i % q.n_elem];
    
    // Validate parameters
    if (!check_bp_pars(gg, dd, ll)) {
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
    
    // ---- Cumulative probability ----
    // F = I_{x^lambda}(gamma, delta+1). lower_tail and log_p go straight to
    // R::pbeta: computing p first and then forming 1 - p or log(p) lost the
    // whole upper tail (pmc(1 - 1e-06, 2, 3, 2.5, lower.tail = FALSE) returned
    // exactly 0 against a true 1.95e-22).
    double xpow = std::exp(ll * std::log(xx));
    out(i) = R::pbeta(xpow, gg, dd + 1.0, lower_tail, log_p);
  }
  
  return Rcpp::NumericVector(out.memptr(), out.memptr() + out.n_elem);
}


// ============================================================================
// QUANTILE FUNCTION
// ============================================================================

/**
 * @brief Quantile Function (Inverse CDF) of the BP/McDonald Distribution
 * 
 * Computes quantiles for the Beta-Power distribution
 * given probability values.
 * 
 * @param p Vector of probabilities (values in [0,1])
 * @param gamma Shape parameter vector (γ > 0)
 * @param delta Shape parameter vector (δ ≥ 0)
 * @param lambda Power parameter vector (λ > 0)
 * @param lower_tail If TRUE, probabilities are P(X ≤ x); otherwise P(X > x)
 * @param log_p If TRUE, probabilities are given as log(p)
 * 
 * @return NumericVector of quantiles
 * 
 * @details
 * The quantile function inverts the CDF:
 * \deqn{Q(p) = \left[Q_{Beta}(p; \gamma, \delta+1)\right]^{1/\lambda}}
 * 
 * @note Exported as .qmc_cpp for internal package use
 */
// [[Rcpp::export(.qmc_cpp)]]
Rcpp::NumericVector qmc(
    const arma::vec& p,
    const Rcpp::NumericVector& gamma,
    const Rcpp::NumericVector& delta,
    const Rcpp::NumericVector& lambda,
    bool lower_tail = true,
    bool log_p = false
) {
  // Convert R vectors to Armadillo vectors
  arma::vec g_vec(gamma.begin(), gamma.size());
  arma::vec d_vec(delta.begin(), delta.size());
  arma::vec l_vec(lambda.begin(), lambda.size());
  
  // Determine output length for recycling
  // Zero-length input: follow R's recycling convention and return an empty
  // vector, as dbeta(numeric(0), 1, 1) does. This also guards the
  // `i % vec.n_elem` recycling below against integer division by zero.
  if (p.n_elem == 0 || g_vec.n_elem == 0 || d_vec.n_elem == 0 || l_vec.n_elem == 0) {
    return Rcpp::NumericVector(0);
  }

  size_t N = std::max({p.n_elem, g_vec.n_elem, d_vec.n_elem, l_vec.n_elem});
  
  arma::vec out(N);
  
  for (size_t i = 0; i < N; i++) {
    // Extract recycled parameters
    double gg = g_vec[i % g_vec.n_elem];
    double dd = d_vec[i % d_vec.n_elem];
    double ll = l_vec[i % l_vec.n_elem];
    double pp = p[i % p.n_elem];
    
    // Validate parameters
    if (!check_bp_pars(gg, dd, ll)) {
      out(i) = NA_REAL;
      continue;
    }
    
    // ---- Normalise the probability, without leaving log space ----
    // The former code did exp(log p) and then 1 - p in linear space. The first
    // flushed the deep tail to zero (qbeta_(-1000, 2, 3, log.p = TRUE) gave 0
    // against a true 2.25e-218); the second cost the upper tail. Out-of-range p
    // keeps the saturating result it has always returned -- whether that should
    // be NaN instead is a separate, still-open question.
    if (log_p && pp > 0.0) { out(i) = NA_REAL; continue; }
    if (!log_p && (pp < 0.0 || pp > 1.0)) {
      out(i) = (lower_tail == (pp > 1.0)) ? 1.0 : 0.0;
      continue;
    }

    // x = [I^-1_{gamma,delta+1}(u)]^(1/lambda), with both flags passed through.
    double y = R::qbeta(pp, gg, dd + 1.0, lower_tail, log_p);
    out(i) = std::pow(y, 1.0 / ll);
  }
  
  return Rcpp::NumericVector(out.memptr(), out.memptr() + out.n_elem);
}


// ============================================================================
// RANDOM NUMBER GENERATION
// ============================================================================

/**
 * @brief Random Variate Generation for the BP/McDonald Distribution
 * 
 * Generates random samples from the Beta-Power distribution
 * using the transformation method.
 * 
 * @param n Number of random variates to generate
 * @param gamma Shape parameter vector (γ > 0)
 * @param delta Shape parameter vector (δ ≥ 0)
 * @param lambda Power parameter vector (λ > 0)
 * 
 * @return NumericVector of n random variates from BP distribution
 * 
 * @details
 * Algorithm:
 * 1. Generate U ~ Beta(γ, δ+1)
 * 2. Return X = U^(1/λ)
 * 
 * @note Exported as .rmc_cpp for internal package use
 */
// [[Rcpp::export(.rmc_cpp)]]
Rcpp::NumericVector rmc(
    int n,
    const Rcpp::NumericVector& gamma,
    const Rcpp::NumericVector& delta,
    const Rcpp::NumericVector& lambda
) {
  if (n <= 0) {
    Rcpp::stop("rmc: n must be positive");
  }
  
  // Convert R vectors to Armadillo vectors
  arma::vec g_vec(gamma.begin(), gamma.size());
  arma::vec d_vec(delta.begin(), delta.size());
  arma::vec l_vec(lambda.begin(), lambda.size());
  
  // A zero-length parameter cannot be recycled. Match R's convention
  // (rbeta(3, numeric(0), 1) is NA NA NA with a warning) instead of
  // reaching the `i % vec.n_elem` recycling with a zero divisor.
  if (g_vec.n_elem == 0 || d_vec.n_elem == 0 || l_vec.n_elem == 0) {
    Rcpp::warning("rmc: NAs produced");
    return Rcpp::NumericVector(n, NA_REAL);
  }

  arma::vec out(n);
  
  for (int i = 0; i < n; i++) {
    // Extract recycled parameters (direct modulo, no intermediate variable)
    double gg = g_vec[i % g_vec.n_elem];
    double dd = d_vec[i % d_vec.n_elem];
    double ll = l_vec[i % l_vec.n_elem];
    
    // Validate parameters
    if (!check_bp_pars(gg, dd, ll)) {
      out(i) = NA_REAL;
      Rcpp::warning("rmc: invalid parameters at index %d", i + 1);
      continue;
    }
    
    // Generate U ~ Beta(γ, δ+1)
    double U = R::rbeta(gg, dd + 1.0);
    
    // Transform: X = U^(1/λ)
    double xval;
    if (ll == 1.0) {
      xval = U;
    } else {
      xval = safe_pow(U, 1.0 / ll);
    }
    
    // Clamp to valid support
    xval = std::max(0.0, std::min(1.0, xval));
    
    out(i) = xval;
  }
  
  return Rcpp::NumericVector(out.memptr(), out.memptr() + out.n_elem);
}


// ============================================================================
// NEGATIVE LOG-LIKELIHOOD FUNCTION
// ============================================================================

/**
 * @brief Negative Log-Likelihood for BP/McDonald Distribution
 * 
 * Computes the negative log-likelihood function for parameter estimation
 * via maximum likelihood.
 * 
 * @param par Parameter vector of length 3: (γ, δ, λ)
 * @param data Vector of observations (must be in (0,1))
 * 
 * @return Negative log-likelihood value (scalar)
 * 
 * @details
 * The log-likelihood for n observations is:
 * \deqn{
 *   \ell(\theta) = n[\ln\lambda - \ln B(\gamma,\delta+1)]
 *   + (\gamma\lambda-1)\sum\ln x_i + \delta\sum\ln(1-x_i^\lambda)
 * }
 * 
 * Returns +Inf for invalid parameters or data outside (0,1).
 * 
 * @note Exported as .llmc_cpp for internal package use
 */
// [[Rcpp::export(.llmc_cpp)]]
double llmc(const Rcpp::NumericVector& par, const Rcpp::NumericVector& data) {
  // Validate parameter vector length
  if (par.size() < 3) return R_PosInf;
  
  // Extract parameters
  double gamma = par[0];
  double delta = par[1];
  double lambda = par[2];
  
  // Validate parameters using consistent checker
  if (!check_bp_pars(gamma, delta, lambda)) return R_PosInf;
  
  // Convert and validate data
  arma::vec x = Rcpp::as<arma::vec>(data);
  if (x.n_elem < 1) return R_PosInf;
  if (arma::any(x <= 0.0) || arma::any(x >= 1.0)) return R_PosInf;
  
  int n = x.n_elem;

  // Compute log(B(γ, δ+1)) stably
  double log_B;
  if (gamma > 100.0 || delta > 100.0) {
    log_B = lgamma(gamma) + lgamma(delta + 1.0) - lgamma(gamma + delta + 1.0);
  } else {
    log_B = R::lbeta(gamma, delta + 1.0);
  }
  
  // Constant term: n * [log(λ) - log(B(γ, δ+1))]
  double log_lambda = safe_log(lambda);
  double const_term = n * (log_lambda - log_B);
  
  // Calculate γλ - 1
  double gl_minus_1 = gamma * lambda - 1.0;
  
  // Initialize accumulators
  double sum_term1 = 0.0;  // (γλ-1) * Σlog(x)
  double sum_term2 = 0.0;  // δ * Σlog(1-x^λ)
  
  for (int i = 0; i < n; i++) {
    // Observations are used as given. Clamping them to [1e-10, 1-1e-10] moved
    // the likelihood by (γλ-1)(log(1e-10) - log(x)) nats -- 23 nats for a
    // single observation at 1e-20 -- and made llmc() disagree with llbeta()
    // at λ = 1, where the two are the same model. Validation above already
    // guarantees x is strictly inside (0,1), so log(x) is finite.
    double log_xi = std::log(x(i));

    // Term 1: (γλ-1) * log(x)
    sum_term1 += gl_minus_1 * log_xi;

    // Term 2: δ * log(1-x^λ), formed from λ*log(x) so the subtraction never
    // happens in linear space. llmc() used log1p(-x^λ), which cannot recover
    // the digits x^λ has already lost, so the objective and its gradient
    // disagreed as x approached 1.
    //
    // log(-expm1(u)) covers x -> 1 but not x -> 0: there 1 - exp(u) is a value
    // just below 1, which doubles cannot resolve more finely than 1.1e-16, so
    // log(1 - x^λ) came back one ulp of 1 wide however small x^λ actually was.
    // gkw_log1mexp() switches to log1p(-exp(u)) past -log(2) and keeps the full
    // relative accuracy; at δ = 1e12 that ulp was worth 2.6e-05 nats.
    // Guarded on δ because δ = 0 with an underflowing term would give 0 * -Inf.
    if (delta > 0.0) {
      sum_term2 += delta * gkw_log1mexp(lambda * log_xi);
    }
  }
  
  double loglike = const_term + sum_term1 + sum_term2;
  
  // Check for invalid results
  if (!std::isfinite(loglike)) return R_PosInf;
  
  return -loglike;
}


// ============================================================================
// GRADIENT OF NEGATIVE LOG-LIKELIHOOD
// ============================================================================

/**
 * @brief Gradient of Negative Log-Likelihood for BP/McDonald Distribution
 * 
 * Computes the gradient vector of the negative log-likelihood for
 * optimization-based parameter estimation.
 * 
 * @param par Parameter vector of length 3: (γ, δ, λ)
 * @param data Vector of observations (must be in (0,1))
 * 
 * @return NumericVector of length 3 containing partial derivatives
 *         with respect to (γ, δ, λ)
 * 
 * @details
 * The gradient components are:
 * - ∂ℓ/∂γ = -n[ψ(γ) - ψ(γ+δ+1)] + λ Σlog(x)
 * - ∂ℓ/∂δ = -n[ψ(δ+1) - ψ(γ+δ+1)] + Σlog(1-x^λ)
 * - ∂ℓ/∂λ = n/λ + γ Σlog(x) - δ Σ[x^λ log(x)/(1-x^λ)]
 * 
 * @note Exported as .grmc_cpp for internal package use
 */
// [[Rcpp::export(.grmc_cpp)]]
Rcpp::NumericVector grmc(const Rcpp::NumericVector& par, const Rcpp::NumericVector& data) {
  // Validate parameter vector length
  if (par.size() < 3) {
    return Rcpp::NumericVector(3, R_NaN);
  }
  
  // Extract parameters
  double gamma = par[0];
  double delta = par[1];
  double lambda = par[2];
  
  // Validate parameters using consistent checker
  if (!check_bp_pars(gamma, delta, lambda)) {
    return Rcpp::NumericVector(3, R_NaN);
  }
  
  // Convert and validate data
  arma::vec x = Rcpp::as<arma::vec>(data);
  if (x.n_elem < 1 || arma::any(x <= 0) || arma::any(x >= 1)) {
    return Rcpp::NumericVector(3, R_NaN);
  }
  
  int n = x.n_elem;
  Rcpp::NumericVector grad(3, 0.0);

  // Calculate digamma terms stably
  double digamma_gamma, digamma_delta_plus_1, digamma_gamma_delta_plus_1;
  
  if (gamma > 100.0) {
    digamma_gamma = std::log(gamma) - 1.0 / (2.0 * gamma);
  } else {
    digamma_gamma = R::digamma(gamma);
  }
  
  if (delta > 100.0) {
    digamma_delta_plus_1 = std::log(delta + 1.0) - 1.0 / (2.0 * (delta + 1.0));
  } else {
    digamma_delta_plus_1 = R::digamma(delta + 1.0);
  }
  
  if (gamma + delta > 100.0) {
    digamma_gamma_delta_plus_1 = std::log(gamma + delta + 1.0) - 1.0 / (2.0 * (gamma + delta + 1.0));
  } else {
    digamma_gamma_delta_plus_1 = R::digamma(gamma + delta + 1.0);
  }
  
  // Initialize accumulators
  double sum_log_x = 0.0;
  double sum_log_v = 0.0;
  double sum_term_lambda = 0.0;
  
  for (int i = 0; i < n; i++) {
    // Observations are used as given; see the note in llmc().
    double log_xi = std::log(x(i));
    sum_log_x += log_xi;

    // v = 1 - x^λ, always via -expm1 of the same exponent that produces x^λ,
    // so the two never drift apart. The former v = max(v, 1e-10) froze log(v)
    // at -23.03: for x = 1 - 1e-15 the true log(v) is -34.28.
    // log(v) goes through gkw_log1mexp() rather than log(-expm1(u)): the latter
    // has to represent a number just below 1, which costs the whole value of
    // log(v) once x^λ drops under 1.1e-16. This is the same term llmc()
    // accumulates, and the two now agree bit for bit.
    double log_x_lambda = lambda * log_xi;
    double x_lambda = std::exp(log_x_lambda);
    double v = -std::expm1(log_x_lambda);
    sum_log_v += gkw_log1mexp(log_x_lambda);

    // Term for the λ gradient: x^λ log(x) / (1 - x^λ). It tends to -1/λ as
    // x -> 1, so it carries its own ceiling; the former ±1e6 clamp truncated
    // legitimate values once λ dropped below 1e-6.
    sum_term_lambda += (x_lambda * log_xi) / v;
  }
  
  // =========================================================================
  // Compute gradient of LOG-LIKELIHOOD ℓ
  // =========================================================================
  
  // ∂ℓ/∂γ = -n[ψ(γ) - ψ(γ+δ+1)] + λ·Σlog(x)
  double d_gamma = -n * (digamma_gamma - digamma_gamma_delta_plus_1) + lambda * sum_log_x;
  
  // ∂ℓ/∂δ = -n[ψ(δ+1) - ψ(γ+δ+1)] + Σlog(1-x^λ)
  double d_delta = -n * (digamma_delta_plus_1 - digamma_gamma_delta_plus_1) + sum_log_v;
  
  // ∂ℓ/∂λ = n/λ + γ·Σlog(x) - δ·Σ[x^λ*log(x)/(1-x^λ)]
  double d_lambda = n / lambda + gamma * sum_log_x - delta * sum_term_lambda;
  
  // =========================================================================
  // Return NEGATIVE gradient (for minimization of negative log-likelihood)
  // =========================================================================
  grad[0] = -d_gamma;
  grad[1] = -d_delta;
  grad[2] = -d_lambda;
  
  return grad;
}


// ============================================================================
// HESSIAN OF NEGATIVE LOG-LIKELIHOOD
// ============================================================================

/**
 * @brief Hessian Matrix of Negative Log-Likelihood for BP/McDonald Distribution
 * 
 * Computes the Hessian matrix (matrix of second partial derivatives) of
 * the negative log-likelihood for standard error estimation and
 * optimization algorithms.
 * 
 * @param par Parameter vector of length 3: (γ, δ, λ)
 * @param data Vector of observations (must be in (0,1))
 * 
 * @return NumericMatrix of dimension 3×3 containing the Hessian
 * 
 * @details
 * Computes analytical second derivatives. The Hessian is symmetric.
 * Parameter ordering: (γ, δ, λ) → indices (0, 1, 2).
 * 
 * Returns NaN matrix for invalid inputs.
 * 
 * @note Exported as .hsmc_cpp for internal package use
 */
// [[Rcpp::export(.hsmc_cpp)]]
Rcpp::NumericMatrix hsmc(const Rcpp::NumericVector& par, const Rcpp::NumericVector& data) {
  // Initialize NaN matrix for error cases
  Rcpp::NumericMatrix nanHess(3, 3);
  nanHess.fill(R_NaN);
  
  // Validate parameter vector length
  if (par.size() < 3) {
    return nanHess;
  }
  
  // Extract parameters
  double gamma = par[0];
  double delta = par[1];
  double lambda = par[2];
  
  // Validate parameters using consistent checker
  if (!check_bp_pars(gamma, delta, lambda)) {
    return nanHess;
  }
  
  // Convert and validate data
  arma::vec x = Rcpp::as<arma::vec>(data);
  if (x.n_elem < 1 || arma::any(x <= 0) || arma::any(x >= 1)) {
    return nanHess;
  }
  
  int n = x.n_elem;
  Rcpp::NumericMatrix hess(3, 3);

  // Compute trigamma values stably
  double trigamma_gamma, trigamma_delta_plus_1, trigamma_gamma_plus_delta_plus_1;
  
  if (gamma > 100.0) {
    trigamma_gamma = 1.0 / gamma + 1.0 / (2.0 * gamma * gamma);
  } else {
    trigamma_gamma = R::trigamma(gamma);
  }
  
  if (delta > 100.0) {
    trigamma_delta_plus_1 = 1.0 / (delta + 1.0) + 1.0 / (2.0 * (delta + 1.0) * (delta + 1.0));
  } else {
    trigamma_delta_plus_1 = R::trigamma(delta + 1.0);
  }
  
  if (gamma + delta > 100.0) {
    double z = gamma + delta + 1.0;
    trigamma_gamma_plus_delta_plus_1 = 1.0 / z + 1.0 / (2.0 * z * z);
  } else {
    trigamma_gamma_plus_delta_plus_1 = R::trigamma(gamma + delta + 1.0);
  }
  
  // Initialize accumulators for data-dependent terms
  double sum_log_x = 0.0;
  double sum_x_lambda_log_x_div_v = 0.0;
  double sum_lambda_term = 0.0;
  
  for (int i = 0; i < n; i++) {
    // Observations are used as given; see the note in llmc().
    double log_xi = std::log(x(i));
    sum_log_x += log_xi;

    // v = 1 - x^λ, as in grmc(): via -expm1 of the same exponent as x^λ, and
    // no floor, so the Hessian tracks the objective into the upper tail.
    double log_x_lambda = lambda * log_xi;
    double x_lambda = std::exp(log_x_lambda);
    double v = -std::expm1(log_x_lambda);

    // Term for H[δ,λ]: Σ[x^λ*log(x)/(1-x^λ)]
    sum_x_lambda_log_x_div_v += (x_lambda * log_xi) / v;

    // Term for H[λ,λ]: Σ[x^λ*(log(x))²/(1-x^λ)²]
    sum_lambda_term += x_lambda * (log_xi * log_xi) / (v * v);
  }
  
  // =========================================================================
  // Compute Hessian of LOG-LIKELIHOOD ℓ (not negative log-likelihood)
  // =========================================================================
  
  // H[γ,γ] = ∂²ℓ/∂γ² = -n[ψ'(γ) - ψ'(γ+δ+1)]
  double H_gamma_gamma = -n * (trigamma_gamma - trigamma_gamma_plus_delta_plus_1);
  
  // H[γ,δ] = ∂²ℓ/∂γ∂δ = n·ψ'(γ+δ+1)
  double H_gamma_delta = n * trigamma_gamma_plus_delta_plus_1;
  
  // H[γ,λ] = ∂²ℓ/∂γ∂λ = Σlog(x)
  double H_gamma_lambda = sum_log_x;
  
  // H[δ,δ] = ∂²ℓ/∂δ² = -n[ψ'(δ+1) - ψ'(γ+δ+1)]
  double H_delta_delta = -n * (trigamma_delta_plus_1 - trigamma_gamma_plus_delta_plus_1);
  
  // H[δ,λ] = ∂²ℓ/∂δ∂λ = -Σ[x^λ*log(x)/(1-x^λ)]
  double H_delta_lambda = -sum_x_lambda_log_x_div_v;
  
  // H[λ,λ] = ∂²ℓ/∂λ² = -n/λ² - δ·Σ[x^λ*(log(x))²/(1-x^λ)²]
  double H_lambda_lambda = -n / (lambda * lambda) - delta * sum_lambda_term;
  
  // =========================================================================
  // Fill the Hessian matrix for NEGATIVE log-likelihood: -H
  // =========================================================================
  hess(0, 0) = -H_gamma_gamma;
  hess(0, 1) = hess(1, 0) = -H_gamma_delta;
  hess(0, 2) = hess(2, 0) = -H_gamma_lambda;
  hess(1, 1) = -H_delta_delta;
  hess(1, 2) = hess(2, 1) = -H_delta_lambda;
  hess(2, 2) = -H_lambda_lambda;
  
  return hess;
}

