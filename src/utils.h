// utils.h
// Utility functions for Generalized Kumaraswamy distributions package
// Author: Lopes, J. E.
// Date: 2025-10-07
//
// This header provides numerically stable functions and parameter validators
// for various Kumaraswamy-based distributions implemented in the package.

#ifndef GKWDIST_UTILS_H
#define GKWDIST_UTILS_H

// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include <cmath>
#include <limits>
#include <algorithm>
#include <string>
#include <functional>
#include <vector>
#include <random>

/*
 * ===========================================================================
 * COMPILE-TIME MATHEMATICAL CONSTANTS
 * ===========================================================================
 * All constants are computed at compile-time (constexpr) to eliminate
 * runtime initialization overhead. Values are provided with maximum
 * precision available in IEEE 754 double precision (53-bit mantissa).
 */

namespace {  // Anonymous namespace prevents ODR violations across translation units

// Machine precision constants
constexpr double EPSILON      = std::numeric_limits<double>::epsilon();        // ~2.220446e-16
  constexpr double DBL_MIN_SAFE = std::numeric_limits<double>::min() * 10.0;   // ~2.225074e-307
  constexpr double DBL_MAX_SAFE = std::numeric_limits<double>::max() / 10.0;   // ~1.797693e+307
  
  // Logarithmic bounds (pre-calculated for performance). Each name states which
  // quantity it is the natural logarithm of; they are not interchangeable.
  // LOG_DBL_MIN is log(DBL_MIN) and LOG_DBL_MIN_SAFE is log(DBL_MIN_SAFE); the two
  // differ by log(10), so safe_exp() and safe_log() must each use their own.
  constexpr double LOG_DBL_MIN      = -708.3964185322641;  // log(std::numeric_limits<double>::min())
  constexpr double LOG_DBL_MIN_SAFE = -706.09383343927004; // log(DBL_MIN_SAFE)
  constexpr double LOG_DBL_MAX      = 707.48012780038994;  // log(DBL_MAX_SAFE)
  // log(std::numeric_limits<double>::max()). The overflow tests in safe_exp()
  // and safe_pow() used LOG_DBL_MAX above, which is log(DBL_MAX / 10), and so
  // returned +Inf for every result in (1.8e307, 1.8e308] that a double holds:
  // dkw(5e-324, 0.045, 1) is 2.57e307 and came back Inf.
  constexpr double LOG_DBL_MAX_EXACT = 709.78271289338397; // log(DBL_MAX)

  // Mathematical constants (maximum precision)
  constexpr double LN2          = 0.6931471805599453094172321214581766;  // log(2)
  constexpr double SQRT_EPSILON = 1.4901161193847656e-08;  // sqrt(EPSILON) for double
  
  // Optimized thresholds for numerical stability (based on Mächler 2012)
  constexpr double LOG1MEXP_CROSSOVER = -0.6931471805599453;  // -log(2)
  constexpr double LOG1MEXP_TINY      = -1.0e-14;  // Threshold for Taylor expansion
  constexpr double LOG_LOG1MEXP_TINY  = -32.236191301916641;  // log(1e-14) = log(-LOG1MEXP_TINY)
  constexpr double LOG1PEXP_LOWER     = -37.0;     // Below: exp(x) alone suffices
  constexpr double LOG1PEXP_MEDIUM    = 18.0;      // Transition to log1p formulation
  constexpr double LOG1PEXP_UPPER     = 33.3;      // Transition to x + exp(-x)
  constexpr double LOG1PEXP_LARGE     = 700.0;     // Above: x alone suffices
  
  // Parameter validation bounds (strict mode)
  constexpr double STRICT_MIN_PARAM   = 1.0e-8;    // Minimum parameter value (strict)
  constexpr double STRICT_MAX_PARAM   = 1.0e8;     // Maximum parameter value (strict)
  
  // Exponent limits for safe_pow overflow detection
  constexpr double EXTREME_EXPONENT   = 1.0e10;    // Threshold for special exponent handling
  
  // Tolerance for integer detection in negative base powers
  constexpr double INTEGER_TOLERANCE  = 1.0e-12;   // Tolerance for y ≈ round(y)
  
} // namespace

/*
 * ===========================================================================
 * CORE NUMERICAL STABILITY FUNCTIONS
 * ===========================================================================
 * These functions implement numerically stable computations for logarithmic
 * operations that are prone to catastrophic cancellation or overflow.
 * Implementations follow best practices from:
 * - Mächler, M. (2012). "Accurately Computing log(1-exp(-|a|))"
 * - R Core Team. R Mathlib implementation
 */

/**
 * gkw_boundary_pdf: limiting density at the closed boundaries x = 0 and x = 1
 *
 * The GKw support is the open interval, but base R's density functions return
 * the LIMIT at the closed boundary rather than 0 -- dbeta(0, 0.5, 1) is Inf,
 * dbeta(1, 2, 1) is 2 -- and any code that plots a density across [0,1] depends
 * on it. This package returned 0 at both ends for every family.
 *
 * Substituting the same first-order forms the log chain already uses,
 *
 *   x -> 0:  log_v -> 0,  log_w -> log(beta) + alpha*log(x),  log_z -> 0
 *   x -> 1:  log_x -> 0,  log_w -> 0,  log_z -> log(lambda) + beta*log_v
 *
 * the log-density collapses to a constant plus a single power of the vanishing
 * quantity:
 *
 *   at 0:  log_const + (gamma*lambda - 1) log(beta) + (alpha*gamma*lambda - 1) log(x)
 *   at 1:  log_const + delta log(lambda)            + (beta*(delta + 1) - 1) log(v)
 *
 * so the exponent alone decides the answer: positive gives 0, negative gives
 * +Inf, and zero leaves the constant. Every nested family reaches this through
 * its own fixed parameters, and the whole rule was checked against stats::dbeta
 * at each of the ten combinations the Beta parameterisation can express.
 *
 * @param at_zero  true for the x = 0 boundary, false for x = 1
 * @param a,b,g,d,l  the five GKw parameters of the calling family
 * @param log_prob   return the log-density instead of the density
 */
inline double gkw_boundary_pdf(bool at_zero, double a, double b, double g,
                               double d, double l, bool log_prob) {
  double exponent = at_zero ? (a * g * l - 1.0) : (b * (d + 1.0) - 1.0);
  if (exponent > 0.0) return log_prob ? R_NegInf : 0.0;
  if (exponent < 0.0) return R_PosInf;

  double log_const = std::log(l) + std::log(a) + std::log(b) - R::lbeta(g, d + 1.0);
  double lp = at_zero ? (log_const + (g * l - 1.0) * std::log(b))
                      : (log_const + d * std::log(l));
  return log_prob ? lp : std::exp(lp);
}

/**
 * log1mexp: Compute log(1 - exp(u)) with enhanced numerical stability
 * 
 * For u <= 0, computes log(1 - exp(u)) using different approximations
 * depending on the magnitude of u to avoid catastrophic cancellation.
 * 
 * Method selection (Mächler 2012):
 *   u > -1e-14        : Taylor expansion log(-u) + corrections
 *   -log(2) < u <= 0  : log(-expm1(u))
 *   u <= -log(2)      : log1p(-exp(u))
 * 
 * @param u Non-positive value (u <= 0)
 * @return log(1 - exp(u)), or NaN if u > 0
 * 
 * @note Time complexity: O(1)
 * @note Relative error: < 2*EPSILON for all u <= 0
 */
inline double gkw_log1mexp(double u) {
  // Input validation: u must be non-positive
  if (u > 0.0) {
    return R_NaN;
  }
  
  // Region 1: Very small |u| - use Taylor series with correction
  //
  //   1 - exp(u) = -(u + u²/2 + u³/6 + ...) = -u * (1 + u/2 + u²/6 + ...)
  //
  // so log(1 - exp(u)) = log(-u) + log(1 + u/2 + ...) ≈ log(-u) + u/2.
  //
  // The correction is +u/2, not -u/2: the previous line of this derivation
  // read "log(1 + u/2) ≈ -u/2", which drops a sign. Since u < 0 here the two
  // differ by |u|, so the returned value was displaced by up to 1e-14 -- about
  // 1.4 ulp at this magnitude, but in the wrong direction, and it put a step at
  // the boundary with region 2 where the function should be smooth. That also
  // broke the "< 2*EPSILON" guarantee documented above.
  if (u > LOG1MEXP_TINY) {
    double neg_u = -u;
    return std::log(neg_u) + 0.5 * u;
  }
  
  // Region 2: -log(2) < u <= -1e-14 - use expm1 formulation
  // expm1(u) = exp(u) - 1, so -expm1(u) = 1 - exp(u)
  if (u > LOG1MEXP_CROSSOVER) {
    return std::log(-std::expm1(u));
  }
  
  // Region 3: u <= -log(2) - use log1p formulation
  // Most numerically stable for large |u|
  return std::log1p(-std::exp(u));
}

/*
 * ===========================================================================
 * ONE LINK OF THE LOG-SPACE CHAIN
 * ===========================================================================
 * Every family walks some part of the chain
 *
 *     v = 1 - x^alpha        w = 1 - v^beta        z = 1 - w^lambda
 *
 * and each link has the same shape: given t = log(p) and s = log(1 - p) =
 * gkw_log1mexp(t), it needs log(1 - (1 - p)^c) = log(1 - exp(c*s)). The quantile
 * and random-number routines walk the same link backwards, with c = 1/beta.
 *
 * The direct answer gkw_log1mexp(c*s) is exact while s and c*s are normal
 * doubles, and it was always used. It fails in two ways once p is tiny, where
 * s ~ -p. When p < 5e-324, s is exactly 0 and log1mexp(0) = -Inf; the former
 * chains bridged that case alone, on `s == 0.0`. But for p in [5e-324, 2.2e-308)
 * s is a subnormal with only a few significant bits, and log1mexp() takes it at
 * face value. With x = c(.01,.3,.6,.9), llbkw(c(161.8, 2, 1.5, 1), x) came back
 * 1522.8725 against a true 1523.2107 -- 0.34 nats -- and the error jumped about
 * as alpha moved, so an optimiser walking that ridge saw a jagged objective and
 * a gradient 16% off.
 *
 * Both failures have the same cure. -c*s is a product of positive numbers, so
 * its logarithm, log(c) + log(-s), is exact in log space however small the
 * product is, and log1mexp() of a tiny argument is only its logarithm:
 * log(1 - exp(u)) = log(-u) + u/2 for u > -1e-14. log(-s) itself is exact from
 * t: -log(1 - p) = p (1 + p/2 + ...), so log(-s) = t + p/2 + ..., and once s is
 * subnormal p/2 is below the last bit of t.
 *
 * The branch fires only where the direct form had lost its digits, so ordinary
 * data is bit-identical.
 */

/**
 * gkw_log_neg_log1mexp: log(-s) for s = gkw_log1mexp(t), exact even where s
 * itself is subnormal or has underflowed to 0.
 */
inline double gkw_log_neg_log1mexp(double t, double s) {
  const double DBL_MIN_NORMAL = std::numeric_limits<double>::min();
  return (s < -DBL_MIN_NORMAL) ? std::log(-s) : t;
}

/**
 * gkw_log1mexp_pow: log(1 - exp(cs)), where cs = c*s and s = gkw_log1mexp(t).
 *
 * The caller passes cs as it computed it (beta * log_v forwards, log_v / beta
 * backwards) so that the direct branch is bit-identical to the code it
 * replaces. log(c) is formed only on the bridged branch.
 */
inline double gkw_log1mexp_pow(double t, double s, double cs, double c) {
  const double DBL_MIN_NORMAL = std::numeric_limits<double>::min();
  if (s < -DBL_MIN_NORMAL && cs < -DBL_MIN_NORMAL) return gkw_log1mexp(cs);
  if (ISNAN(s) || ISNAN(cs)) return s + cs;

  // m = log(-cs), exact in log space; then the same three regions as
  // gkw_log1mexp(), reached through m instead of through cs.
  const double m = std::log(c) + gkw_log_neg_log1mexp(t, s);
  if (m < LOG_LOG1MEXP_TINY) return m - 0.5 * std::exp(m);
  return gkw_log1mexp(-std::exp(m));
}

/**
 * gkw_log_inv_link: the chain walked backwards,
 *   log(x^alpha) = log(1 - (1 - w)^(1/beta))   given log(w),
 * which is what the quantile and random-number routines of GKw, KKw and EKw
 * invert through. The former form, gkw_log1mexp(gkw_log1mexp(log_w) / beta),
 * lost the whole lower tail once w < 5e-324: qgkw(0.01, 40, 2, .05, .5, .1)
 * returned exactly 0, and rgkw(1e5, 40, 2, .05, .5, .1) drew 2,579 exact zeros
 * -- values outside the open support that llgkw() then rejects, so the
 * sampler's own output could not be fitted.
 */
inline double gkw_log_inv_link_s(double log_w, double log_1mw, double beta) {
  return gkw_log1mexp_pow(log_w, log_1mw, log_1mw / beta, 1.0 / beta);
}

// The same link when only log(w) is known. Callers that also hold log(1 - w)
// more accurately -- the upper tail, where w is near 1 and log(w) is a
// subnormal or 0 -- pass it to gkw_log_inv_link_s() instead: forming it here as
// gkw_log1mexp(log_w) would flush that tail to x = 1.
inline double gkw_log_inv_link(double log_w, double beta) {
  return gkw_log_inv_link_s(log_w, gkw_log1mexp(log_w), beta);  // log(1 - w) = beta * log(v)
}

/**
 * gkw_pbeta_from_log: R::pbeta(y, a, b, lower_tail, log_p) for a y known only
 * through its logarithm.
 *
 * Where y is a normal double this is R::pbeta(exp(log_y), ...) exactly. Below
 * DBL_MIN, exp(log_y) is subnormal or 0 and pbeta returns 0 -- but I_y(a, b) is
 * not small there when a is: I_y(a, b) = y^a / (a B(a, b)) (1 + O(y)), and at
 * y < 2.2e-308 the O(y) term is below the last bit. pgkw(1e-09, 40, 2, 0.05,
 * 0.5, 0.1) returned 0 against a true 0.0163855.
 */
inline double gkw_pbeta_from_log(double log_y, double a, double b,
                                 bool lower_tail, bool log_p) {
  if (!(log_y < LOG_DBL_MIN)) {
    return R::pbeta(std::exp(log_y), a, b, lower_tail, log_p);
  }
  double log_F = a * log_y - std::log(a) - R::lbeta(a, b);
  if (log_F > 0.0) log_F = 0.0;
  if (lower_tail) return log_p ? log_F : std::exp(log_F);
  return log_p ? gkw_log1mexp(log_F) : -std::expm1(log_F);
}

/**
 * gkw_log_qbeta_tiny: log(I^-1(p; a, b)) for a quantile R::qbeta() cannot hold.
 *
 * The inverse of the expansion above, log(y) = [log(F) + log(a) + log B(a,b)]/a,
 * with F the lower-tail probability recovered without subtraction. It is meant
 * for the case where R::qbeta() returned a y below DBL_MIN, including 0: with a
 * small a the quantile y = (F a B)^(1/a) underflows while the x it maps to is
 * perfectly ordinary.
 */
inline double gkw_log_qbeta_tiny(double pp, double a, double b,
                                 bool lower_tail, bool log_p) {
  double log_F;
  if (lower_tail) log_F = log_p ? pp : std::log(pp);
  else            log_F = log_p ? gkw_log1mexp(pp) : std::log1p(-pp);
  return (log_F + std::log(a) + R::lbeta(a, b)) / a;
}

/**
 * gkw_log_qbeta: log(y) and log(1 - y) for y = I^-1(p; a, b), accurate in both
 * tails.
 *
 * Above 1/2 a double holds y no more finely than 1.1e-16, so 1 - y is taken
 * straight from R::qbeta through the symmetry I_y(a, b) = 1 - I_{1-y}(b, a),
 * exactly as qbkw() already did. Without it
 * qgkw(1e-26, 2, 3, 1.5, 0.5, 1.2, lower.tail = FALSE) returned exactly 1, where
 * the true 1 - x is 6.98e-07. Whichever of y and 1 - y is small, R::qbeta
 * saturates it at about 1.1e-308, so below DBL_MIN its logarithm comes from the
 * expansion above -- on the reflected side too, where the saturated value put
 * qgkw(-750, 2, 100, 1, 0, 1, lower.tail = FALSE, log.p = TRUE) on a plateau at
 * 1 - x = 4.16e-04 against a true 2.77e-04.
 */
inline void gkw_log_qbeta(double pp, double a, double b,
                          bool lower_tail, bool log_p,
                          double& log_y, double& log_1my) {
  const double DBL_MIN_NORMAL = std::numeric_limits<double>::min();
  const double y = R::qbeta(pp, a, b, lower_tail, log_p);
  if (y > 0.5) {
    const double omy = R::qbeta(pp, b, a, !lower_tail, log_p);
    log_1my = (omy >= DBL_MIN_NORMAL) ? std::log(omy)
                                      : gkw_log_qbeta_tiny(pp, b, a, !lower_tail, log_p);
    log_y = std::log1p(-omy);
  } else {
    log_y = (y >= DBL_MIN_NORMAL) ? std::log(y)
                                  : gkw_log_qbeta_tiny(pp, a, b, lower_tail, log_p);
    log_1my = std::log1p(-y);
  }
}

/**
 * gkw_warning: raise an R warning without leaking the caller's C++ state.
 *
 * Rcpp::warning() is a bare Rf_warning(). When the warning is turned into an
 * error (options(warn = 2)) or caught by tryCatch(warning = ), R leaves through
 * a longjmp, which skips every C++ destructor on the way out: the Armadillo copy
 * of the data and the Rcpp handles protecting the inputs were never released.
 * Twenty calls of grbkw() on 2e6 observations under
 * tryCatch(warning = function(w) NULL) grew the process by 308 MB.
 *
 * Calling base::warning() through Rcpp::Function evaluates it under
 * R_UnwindProtect: a longjmp is caught, rethrown as a C++ exception that unwinds
 * the stack normally, and resumed by the generated wrapper once every frame has
 * been cleaned up. The message and the call it names -- the exported R function
 * -- are unchanged.
 */
template <typename... Args>
inline void gkw_warning(const char* fmt, Args&&... args) {
  const std::string msg = tfm::format(fmt, std::forward<Args>(args)...);
  Rcpp::Function warning_fn = Rcpp::Environment::base_env()["warning"];
  warning_fn(msg);
}

/**
 * log1pexp: Compute log(1 + exp(x)) with protection against overflow
 * 
 * Handles various regimes of x with appropriate approximations to maintain
 * numerical stability across the entire real line.
 * 
 * Method selection:
 *   x > 700     : x (overflow protection)
 *   x > 33.3    : x + exp(-x) (asymptotic expansion)
 *   x > 18      : x + log1p(exp(-x)) (better for moderate x)
 *   x > -37     : log1p(exp(x)) (standard range)
 *   x > -700    : exp(x) (exp(x) << 1)
 *   x <= -700   : 0 (complete underflow)
 * 
 * @param x Input value (unrestricted)
 * @return log(1 + exp(x)) calculated with numerical stability
 * 
 * @note Time complexity: O(1)
 * @note Relative error: < 2*EPSILON for all x
 */
inline double gkw_log1pexp(double x) {
  // Region 1: Very large x - asymptotic to x
  if (x > LOG1PEXP_LARGE) {
    return x;
  }
  
  // Region 2: Large x - first-order correction
  if (x > LOG1PEXP_UPPER) {
    return x + std::exp(-x);
  }
  
  // Region 3: Moderately large x - use log1p with negative exponent
  if (x > LOG1PEXP_MEDIUM) {
    return x + std::log1p(std::exp(-x));
  }
  
  // Region 4: Standard range - direct log1p
  if (x > LOG1PEXP_LOWER) {
    return std::log1p(std::exp(x));
  }
  
  // Region 5: Large negative x - exp(x) dominates
  if (x > -LOG1PEXP_LARGE) {
    return std::exp(x);
  }
  
  // Region 6: Extreme negative x - complete underflow
  return 0.0;
}

/**
 * safe_log: Compute log(x) with comprehensive error handling
 * 
 * @param x Input value
 * @return log(x), -Inf for x=0, NaN for x<0, or scaled result for tiny x
 * 
 * @note Handles underflow gracefully for very small positive x
 * @note Time complexity: O(1)
 */
inline double safe_log(double x) {
  // Handle invalid inputs
  if (x < 0.0) {
    return R_NaN;
  }
  
  if (x == 0.0) {
    return R_NegInf;
  }
  
  // Handle potential underflow with scaled computation.
  // For x = ε * DBL_MIN_SAFE with ε small,
  // log(x) = log(ε) + log(DBL_MIN_SAFE) = log(ε) + LOG_DBL_MIN_SAFE.
  // The scaling constant must be the logarithm of the divisor used just below.
  if (x < DBL_MIN_SAFE) {
    return LOG_DBL_MIN_SAFE + std::log(x / DBL_MIN_SAFE);
  }
  
  return std::log(x);
}

/**
 * safe_exp: Compute exp(x) with protection against overflow/underflow
 * 
 * @param x Input value
 * @return exp(x), +Inf for overflow, 0 or scaled result for underflow
 * 
 * @note Uses scaled arithmetic near underflow threshold for gradual transition
 * @note Time complexity: O(1)
 */
inline double safe_exp(double x) {
  // Handle overflow
  if (x > LOG_DBL_MAX_EXACT) {
    return R_PosInf;
  }

  // Handle severe underflow
  if (x < LOG_DBL_MIN - 10.0) {
    return 0.0;
  }
  
  // Handle moderate underflow with scaling:
  // DBL_MIN * exp(x - log(DBL_MIN)) = exp(x)  — exact, avoids abrupt flush to zero
  if (x < LOG_DBL_MIN) {
    return std::numeric_limits<double>::min() * std::exp(x - LOG_DBL_MIN);
  }
  
  return std::exp(x);
}

/**
 * safe_pow: Compute x^y with robust error handling and numerical stability
 * 
 * Handles special cases comprehensively:
 * - x = 0: Returns 0 (y>0), 1 (y=0), +Inf (y<0)
 * - x = 1 or y = 0: Returns 1
 * - y = 1: Returns x
 * - x < 0: Requires y to be effectively integer; handles sign correctly
 * - Extreme exponents: Prevents overflow/underflow with early detection
 * 
 * For positive x, uses logarithmic transformation: x^y = exp(y * log(x)).
 * This is what allows the overflow and underflow of an extreme exponent to be
 * detected before it happens, which is the reason the routine exists.
 *
 * It is NOT more accurate than std::pow, and this note used to claim it was.
 * exp(y*log(x)) carries a relative error of roughly |y*log(x)| * EPSILON, while
 * std::pow on a conforming libm is very nearly correctly rounded. Measured
 * against a 60-digit reference:
 *
 *     x     y      exp(y*log x)     std::pow
 *     10    100    1.11e-14         0
 *     10    300    9.00e-14         0
 *     2     1000   6.85e-14         0
 *
 * Callers that do not need the overflow interception should prefer std::pow.
 * 
 * @param x Base value
 * @param y Exponent value
 * @return x^y calculated with numerical stability and comprehensive edge case handling
 * 
 * @note Time complexity: O(1)
 * @note For x < 0, y must satisfy |y - round(y)| < INTEGER_TOLERANCE
 */
inline double safe_pow(double x, double y) {
  // Handle NaN propagation
  if (std::isnan(x) || std::isnan(y)) {
    return R_NaN;
  }
  
  // ===== Handle x = 0 cases =====
  if (x == 0.0) {
    if (y > 0.0)  return 0.0;        // 0^positive = 0
    if (y == 0.0) return 1.0;        // 0^0 = 1 (standard convention in probability)
    return R_PosInf;                  // 0^negative = +Inf
  }
  
  // ===== Trivial cases =====
  if (x == 1.0 || y == 0.0) return 1.0;  // 1^y = 1, x^0 = 1
  if (y == 1.0) return x;                // x^1 = x
  
  // ===== Handle negative base =====
  if (x < 0.0) {
    // Check if y is effectively an integer
    double y_rounded = std::round(y);
    if (std::abs(y - y_rounded) > INTEGER_TOLERANCE) {
      return R_NaN;  // Non-integer power of negative number is undefined in reals
    }
    
    // y is integer - compute |x|^|y| then apply sign.
    //
    // Parity comes from fmod rather than a cast to int. static_cast<int> is
    // undefined behaviour once |y_rounded| exceeds INT_MAX, which UBSan flags,
    // and clamping to int would also answer the parity question wrongly: an odd
    // integer between INT_MAX and 2^53 is exactly representable as a double.
    // fmod is correct at every magnitude -- above 2^53 every double is even,
    // and fmod says so.
    bool y_is_odd = (std::fmod(std::abs(y_rounded), 2.0) == 1.0);
    double abs_x = -x;  // x is negative, so -x is positive
    
    // Compute absolute result using logarithmic method for stability
    double log_abs_x = std::log(abs_x);
    double log_result = std::abs(y) * log_abs_x;
    
    // Check for overflow/underflow
    if (log_result > LOG_DBL_MAX_EXACT) {
      return y_is_odd ? R_NegInf : R_PosInf;
    }
    if (log_result < LOG_DBL_MIN) {
      return 0.0;
    }
    
    double abs_result = std::exp(log_result);
    
    // Apply sign based on whether y is odd and whether we're inverting
    if (y < 0) {
      // Negative exponent: invert result
      if (abs_result == 0.0) return y_is_odd ? R_NegInf : R_PosInf;
      abs_result = 1.0 / abs_result;
    }
    
    return y_is_odd ? -abs_result : abs_result;
  }
  
  // ===== Positive base: use logarithmic transformation =====
  
  // For extreme exponents, check bounds before computation
  if (std::abs(y) > EXTREME_EXPONENT) {
    double log_x = std::log(x);
    double log_result = y * log_x;
    
    // Early overflow/underflow detection
    if (log_result > LOG_DBL_MAX_EXACT) {
      return R_PosInf;
    }
    if (log_result < LOG_DBL_MIN) {
      return 0.0;
    }
    
    return std::exp(log_result);
  }
  
  // Standard case: compute via logarithm for better stability
  double log_x = std::log(x);
  double log_result = y * log_x;
  
  // Use safe_exp for final result
  return safe_exp(log_result);
}

/*
 * ===========================================================================
 * VECTORIZED NUMERICAL STABILITY FUNCTIONS
 * ===========================================================================
 * Element-wise operations on Armadillo vectors with optimized implementation.
 * These functions maintain numerical stability while leveraging SIMD when possible.
 */

/**
 * vec_log1mexp: Vectorized log(1 - exp(u)) computation
 * 
 * @param u Vector of non-positive values
 * @return Vector of log(1 - exp(u)) values
 * 
 * @note Time complexity: O(n)
 * @note Memory complexity: O(n)
 */
inline arma::vec vec_log1mexp(const arma::vec& u) {
  const size_t n = u.n_elem;
  arma::vec result(n);
  
  // Element-wise processing for maximum numerical reliability
  // Each element may fall in different numerical regime
  for (size_t i = 0; i < n; ++i) {
    result(i) = gkw_log1mexp(u(i));
  }
  
  return result;
}

/**
 * vec_log1pexp: Vectorized log(1 + exp(x)) computation
 * 
 * @param x Vector of input values
 * @return Vector of log(1 + exp(x)) values
 * 
 * @note Time complexity: O(n)
 * @note Memory complexity: O(n)
 */
inline arma::vec vec_log1pexp(const arma::vec& x) {
  const size_t n = x.n_elem;
  arma::vec result(n);
  
  for (size_t i = 0; i < n; ++i) {
    result(i) = gkw_log1pexp(x(i));
  }
  
  return result;
}

/**
 * vec_safe_log: Vectorized safe logarithm computation
 * 
 * @param x Vector of input values
 * @return Vector of safe_log(x) values
 * 
 * @note Time complexity: O(n)
 * @note Memory complexity: O(n)
 */
inline arma::vec vec_safe_log(const arma::vec& x) {
  const size_t n = x.n_elem;
  arma::vec result(n);
  
  for (size_t i = 0; i < n; ++i) {
    result(i) = safe_log(x(i));
  }
  
  return result;
}

/**
 * vec_safe_exp: Vectorized safe exponential computation
 * 
 * @param x Vector of input values
 * @return Vector of safe_exp(x) values
 * 
 * @note Time complexity: O(n)
 * @note Memory complexity: O(n)
 */
inline arma::vec vec_safe_exp(const arma::vec& x) {
  const size_t n = x.n_elem;
  arma::vec result(n);
  
  for (size_t i = 0; i < n; ++i) {
    result(i) = safe_exp(x(i));
  }
  
  return result;
}

/**
 * vec_safe_pow: Vectorized safe power computation (scalar exponent)
 * 
 * @param x Vector of base values
 * @param y Scalar exponent value
 * @return Vector of x[i]^y values
 * 
 * @note Time complexity: O(n)
 * @note Memory complexity: O(n)
 * @note Optimized for case where y is constant across all elements
 */
inline arma::vec vec_safe_pow(const arma::vec& x, double y) {
  const size_t n = x.n_elem;
  arma::vec result(n);
  
  // Optimize for common trivial cases (vectorizable)
  if (y == 0.0) {
    result.ones();
    return result;
  }
  
  if (y == 1.0) {
    return x;
  }

  // Fast SIMD path: y > 0, all x positive — typical distribution case (data in (0,1)).
  // arma::exp(y*arma::log(x)) is fully auto-vectorizable; the branch-heavy scalar loop below is not.
  if (y > 0.0 && y < EXTREME_EXPONENT && x.min() > 0.0) {
    return arma::exp(y * arma::log(x));
  }

  // Check if y is effectively an integer (for negative base handling)
  double y_rounded = std::round(y);
  bool y_is_integer = (std::abs(y - y_rounded) <= INTEGER_TOLERANCE);
  // Parity from fmod, matching safe_pow above. The former guard avoided the
  // undefined cast but answered the parity question wrongly in the process: it
  // reported "even" for every |y| above INT_MAX, and an odd integer between
  // INT_MAX and 2^53 is exactly representable as a double.
  bool y_is_odd = y_is_integer && (std::fmod(std::abs(y_rounded), 2.0) == 1.0);
  
  // Element-wise computation with shared exponent logic
  for (size_t i = 0; i < n; ++i) {
    double xi = x(i);
    
    // Handle NaN
    if (std::isnan(xi)) {
      result(i) = R_NaN;
      continue;
    }
    
    // Handle xi = 0
    if (xi == 0.0) {
      if (y > 0.0) {
        result(i) = 0.0;
      } else if (y == 0.0) {
        result(i) = 1.0;
      } else {
        result(i) = R_PosInf;
      }
      continue;
    }
    
    // Handle xi = 1
    if (xi == 1.0) {
      result(i) = 1.0;
      continue;
    }
    
    // Handle negative base
    if (xi < 0.0) {
      if (!y_is_integer) {
        result(i) = R_NaN;
      } else {
        double abs_xi = -xi;
        double log_abs_xi = std::log(abs_xi);
        double log_result = std::abs(y) * log_abs_xi;
        
        if (log_result > LOG_DBL_MAX) {
          result(i) = y_is_odd ? R_NegInf : R_PosInf;
        } else if (log_result < LOG_DBL_MIN) {
          result(i) = 0.0;
        } else {
          double abs_result = std::exp(log_result);
          if (y < 0) abs_result = 1.0 / abs_result;
          result(i) = y_is_odd ? -abs_result : abs_result;
        }
      }
      continue;
    }
    
    // Positive base: logarithmic computation
    double log_xi = std::log(xi);
    double log_result = y * log_xi;
    result(i) = safe_exp(log_result);
  }
  
  return result;
}

/**
 * vec_safe_pow: Vectorized safe power computation (vector exponents)
 * 
 * @param x Vector of base values
 * @param y Vector of exponent values (must match size of x)
 * @return Vector of x[i]^y[i] values
 * 
 * @note Time complexity: O(n)
 * @note Memory complexity: O(n)
 */
inline arma::vec vec_safe_pow(const arma::vec& x, const arma::vec& y) {
  const size_t n = x.n_elem;
  
  // Input validation
  if (y.n_elem != n) {
    Rcpp::stop("vec_safe_pow: vectors must have same length (x: %d, y: %d)", n, y.n_elem);
  }
  
  arma::vec result(n);
  
  // Element-wise computation
  for (size_t i = 0; i < n; ++i) {
    result(i) = safe_pow(x(i), y(i));
  }
  
  return result;
}

/*
 * ===========================================================================
 * PARAMETER VALIDATION FUNCTIONS
 * ===========================================================================
 * These functions verify that distribution parameters satisfy required
 * constraints. Each distribution family has specific requirements.
 * 
 * The 'strict' parameter enables additional bounds checking to prevent
 * numerical instabilities that can arise from extreme parameter values.
 */

/**
 * check_pars: Validate parameters for Generalized Kumaraswamy (GKw) distribution
 * 
 * Parameter constraints:
 *   alpha > 0   (shape parameter)
 *   beta > 0    (shape parameter)
 *   gamma > 0   (shape parameter)
 *   delta >= 0  (shape parameter, allows zero)
 *   lambda > 0  (shape parameter)
 * 
 * Strict mode additionally enforces:
 *   All parameters in [1e-8, 1e8] to prevent numerical issues
 * 
 * @param alpha Shape parameter (must be > 0)
 * @param beta Shape parameter (must be > 0)
 * @param gamma Shape parameter (must be > 0)
 * @param delta Shape parameter (must be >= 0)
 * @param lambda Shape parameter (must be > 0)
 * @param strict Enable strict bounds checking
 * @return true if parameters are valid, false otherwise
 */
inline bool check_pars(double alpha,
                       double beta,
                       double gamma,
                       double delta,
                       double lambda,
                       bool strict = false) {
  // Check for NaN values
  if (std::isnan(alpha) || std::isnan(beta) || std::isnan(gamma) ||
      std::isnan(delta) || std::isnan(lambda)) {
    return false;
  }
  
  // Check for Inf values
  if (std::isinf(alpha) || std::isinf(beta) || std::isinf(gamma) ||
      std::isinf(delta) || std::isinf(lambda)) {
    return false;
  }
  
  // Basic parameter constraints
  if (alpha <= 0.0 || beta <= 0.0 || gamma <= 0.0 || delta < 0.0 || lambda <= 0.0) {
    return false;
  }
  
  // Strict bounds for numerical stability
  if (strict) {
    if (alpha < STRICT_MIN_PARAM || beta < STRICT_MIN_PARAM || 
        gamma < STRICT_MIN_PARAM || lambda < STRICT_MIN_PARAM) {
      return false;
    }
    
    if (alpha > STRICT_MAX_PARAM || beta > STRICT_MAX_PARAM || 
        gamma > STRICT_MAX_PARAM || delta > STRICT_MAX_PARAM || 
        lambda > STRICT_MAX_PARAM) {
      return false;
    }
    
    // Delta can be zero, but if non-zero must satisfy bounds
    if (delta > 0.0 && delta < STRICT_MIN_PARAM) {
      return false;
    }
  }
  
  return true;
}

/**
 * check_pars_vec: Vectorized parameter validation for GKw distribution
 * 
 * Validates all combinations of parameter values using R-style recycling.
 * If vectors have different lengths, shorter ones are recycled.
 * 
 * @param alpha Vector of alpha values
 * @param beta Vector of beta values
 * @param gamma Vector of gamma values
 * @param delta Vector of delta values
 * @param lambda Vector of lambda values
 * @param strict Enable strict bounds checking
 * @return Vector of boolean values (0/1) indicating parameter validity
 * 
 * @note Return type is arma::uvec for compatibility with Armadillo indexing
 */
inline arma::uvec check_pars_vec(const arma::vec& alpha,
                                 const arma::vec& beta,
                                 const arma::vec& gamma,
                                 const arma::vec& delta,
                                 const arma::vec& lambda,
                                 bool strict = false) {
  // Find maximum length for recycling
  size_t n = std::max({alpha.n_elem, beta.n_elem, gamma.n_elem,
                      delta.n_elem, lambda.n_elem});
  
  arma::uvec valid(n);
  
  // Check each combination with proper recycling
  for (size_t i = 0; i < n; ++i) {
    double a = alpha(i % alpha.n_elem);
    double b = beta(i % beta.n_elem);
    double g = gamma(i % gamma.n_elem);
    double d = delta(i % delta.n_elem);
    double l = lambda(i % lambda.n_elem);
    
    valid(i) = check_pars(a, b, g, d, l, strict) ? 1 : 0;
  }
  
  return valid;
}

/**
 * check_kkw_pars: Validate parameters for Kw-Kumaraswamy (kkw) distribution
 * 
 * kkw is GKw with gamma = 1: kkw(α, β, δ, λ) = GKw(α, β, 1, δ, λ)
 * 
 * Parameter constraints:
 *   alpha > 0
 *   beta > 0
 *   delta >= 0
 *   lambda > 0
 * 
 * @param alpha Shape parameter (must be > 0)
 * @param beta Shape parameter (must be > 0)
 * @param delta Shape parameter (must be >= 0)
 * @param lambda Shape parameter (must be > 0)
 * @param strict Enable strict bounds checking
 * @return true if parameters are valid, false otherwise
 */
inline bool check_kkw_pars(double alpha,
                           double beta,
                           double delta,
                           double lambda,
                           bool strict = false) {
  // Check for NaN/Inf
  if (std::isnan(alpha) || std::isnan(beta) || std::isnan(delta) || std::isnan(lambda)) {
    return false;
  }
  if (std::isinf(alpha) || std::isinf(beta) || std::isinf(delta) || std::isinf(lambda)) {
    return false;
  }
  
  // Basic constraints
  if (alpha <= 0.0 || beta <= 0.0 || delta < 0.0 || lambda <= 0.0) {
    return false;
  }
  
  // Strict bounds
  if (strict) {
    if (alpha < STRICT_MIN_PARAM || beta < STRICT_MIN_PARAM || lambda < STRICT_MIN_PARAM) {
      return false;
    }
    if (alpha > STRICT_MAX_PARAM || beta > STRICT_MAX_PARAM || 
        delta > STRICT_MAX_PARAM || lambda > STRICT_MAX_PARAM) {
      return false;
    }
    if (delta > 0.0 && delta < STRICT_MIN_PARAM) {
      return false;
    }
  }
  
  return true;
}

/**
 * check_bkw_pars: Validate parameters for Beta-Kumaraswamy (BKw) distribution
 * 
 * BKw is GKw with lambda = 1: BKw(α, β, γ, δ) = GKw(α, β, γ, δ, 1)
 * 
 * Parameter constraints:
 *   alpha > 0
 *   beta > 0
 *   gamma > 0
 *   delta >= 0
 * 
 * @param alpha Shape parameter (must be > 0)
 * @param beta Shape parameter (must be > 0)
 * @param gamma Shape parameter (must be > 0)
 * @param delta Shape parameter (must be >= 0)
 * @param strict Enable strict bounds checking
 * @return true if parameters are valid, false otherwise
 */
inline bool check_bkw_pars(double alpha,
                           double beta,
                           double gamma,
                           double delta,
                           bool strict = false) {
  // Check for NaN/Inf
  if (std::isnan(alpha) || std::isnan(beta) || std::isnan(gamma) || std::isnan(delta)) {
    return false;
  }
  if (std::isinf(alpha) || std::isinf(beta) || std::isinf(gamma) || std::isinf(delta)) {
    return false;
  }
  
  // Basic constraints
  if (alpha <= 0.0 || beta <= 0.0 || gamma <= 0.0 || delta < 0.0) {
    return false;
  }
  
  // Strict bounds
  if (strict) {
    if (alpha < STRICT_MIN_PARAM || beta < STRICT_MIN_PARAM || gamma < STRICT_MIN_PARAM) {
      return false;
    }
    if (alpha > STRICT_MAX_PARAM || beta > STRICT_MAX_PARAM || 
        gamma > STRICT_MAX_PARAM || delta > STRICT_MAX_PARAM) {
      return false;
    }
    if (delta > 0.0 && delta < STRICT_MIN_PARAM) {
      return false;
    }
  }
  
  return true;
}

/**
 * check_ekw_pars: Validate parameters for Exponentiated-Kumaraswamy (EKw) distribution
 * 
 * EKw is GKw with gamma = 1, delta = 0: EKw(α, β, λ) = GKw(α, β, 1, 0, λ)
 * 
 * Parameter constraints:
 *   alpha > 0
 *   beta > 0
 *   lambda > 0
 * 
 * @param alpha Shape parameter (must be > 0)
 * @param beta Shape parameter (must be > 0)
 * @param lambda Shape parameter (must be > 0)
 * @param strict Enable strict bounds checking
 * @return true if parameters are valid, false otherwise
 */
inline bool check_ekw_pars(double alpha, double beta, double lambda, bool strict = false) {
  // Check for NaN/Inf
  if (std::isnan(alpha) || std::isnan(beta) || std::isnan(lambda)) {
    return false;
  }
  if (std::isinf(alpha) || std::isinf(beta) || std::isinf(lambda)) {
    return false;
  }
  
  // Basic constraints
  if (alpha <= 0.0 || beta <= 0.0 || lambda <= 0.0) {
    return false;
  }
  
  // Strict bounds
  if (strict) {
    if (alpha < STRICT_MIN_PARAM || beta < STRICT_MIN_PARAM || lambda < STRICT_MIN_PARAM) {
      return false;
    }
    if (alpha > STRICT_MAX_PARAM || beta > STRICT_MAX_PARAM || lambda > STRICT_MAX_PARAM) {
      return false;
    }
  }
  
  return true;
}

/**
 * check_bp_pars: Validate parameters for Beta-Power (BP) distribution
 * 
 * BP is GKw with alpha = beta = 1: BP(γ, δ, λ) = GKw(1, 1, γ, δ, λ)
 * 
 * Parameter constraints:
 *   gamma > 0
 *   delta >= 0
 *   lambda > 0
 * 
 * @param gamma Shape parameter (must be > 0)
 * @param delta Shape parameter (must be >= 0)
 * @param lambda Shape parameter (must be > 0)
 * @param strict Enable strict bounds checking
 * @return true if parameters are valid, false otherwise
 */
inline bool check_bp_pars(double gamma, double delta, double lambda, bool strict = false) {
  // Check for NaN/Inf
  if (std::isnan(gamma) || std::isnan(delta) || std::isnan(lambda)) {
    return false;
  }
  if (std::isinf(gamma) || std::isinf(delta) || std::isinf(lambda)) {
    return false;
  }
  
  // Basic constraints
  if (gamma <= 0.0 || delta < 0.0 || lambda <= 0.0) {
    return false;
  }
  
  // Strict bounds
  if (strict) {
    if (gamma < STRICT_MIN_PARAM || lambda < STRICT_MIN_PARAM) {
      return false;
    }
    if (gamma > STRICT_MAX_PARAM || delta > STRICT_MAX_PARAM || lambda > STRICT_MAX_PARAM) {
      return false;
    }
    if (delta > 0.0 && delta < STRICT_MIN_PARAM) {
      return false;
    }
  }
  
  return true;
}

/**
 * check_kw_pars: Validate parameters for Kumaraswamy (Kw) distribution
 * 
 * Kw is GKw with gamma = delta = lambda = 1: Kw(α, β) = GKw(α, β, 1, 0, 1)
 * 
 * Parameter constraints:
 *   alpha > 0
 *   beta > 0
 * 
 * @param alpha Shape parameter (must be > 0)
 * @param beta Shape parameter (must be > 0)
 * @param strict Enable strict bounds checking
 * @return true if parameters are valid, false otherwise
 */
inline bool check_kw_pars(double alpha, double beta, bool strict = false) {
  // Check for NaN/Inf
  if (std::isnan(alpha) || std::isnan(beta)) {
    return false;
  }
  if (std::isinf(alpha) || std::isinf(beta)) {
    return false;
  }
  
  // Basic constraints
  if (alpha <= 0.0 || beta <= 0.0) {
    return false;
  }
  
  // Strict bounds
  if (strict) {
    if (alpha < STRICT_MIN_PARAM || beta < STRICT_MIN_PARAM) {
      return false;
    }
    if (alpha > STRICT_MAX_PARAM || beta > STRICT_MAX_PARAM) {
      return false;
    }
  }
  
  return true;
}

/**
 * check_beta_pars: Validate parameters for Beta distribution
 * 
 * Parameter constraints:
 *   gamma > 0  (shape1)
 *   delta > 0  (shape2, note: must be POSITIVE for Beta, unlike GKw where >= 0)
 * 
 * @param gamma Shape parameter 1 (must be > 0)
 * @param delta Shape parameter 2 (must be > 0)
 * @param strict Enable strict bounds checking
 * @return true if parameters are valid, false otherwise
 */
inline bool check_beta_pars(double gamma, double delta, bool strict = false) {
  // Check for NaN/Inf
  if (std::isnan(gamma) || std::isnan(delta)) {
    return false;
  }
  if (std::isinf(gamma) || std::isinf(delta)) {
    return false;
  }
  
  // The Beta sub-family is Beta(gamma, delta + 1), so the second shape
  // parameter is delta + 1 and delta = 0 is the legitimate Beta(gamma, 1)
  // boundary, not an invalid value. This matches every other validator in
  // this file and the delta >= 0 support of the GKw family itself.
  if (gamma <= 0.0 || delta < 0.0) {
    return false;
  }
  
  // Strict bounds
  if (strict) {
    if (gamma < STRICT_MIN_PARAM || delta < STRICT_MIN_PARAM) {
      return false;
    }
    if (gamma > STRICT_MAX_PARAM || delta > STRICT_MAX_PARAM) {
      return false;
    }
  }
  
  return true;
}

#endif // GKWDIST_UTILS_H
