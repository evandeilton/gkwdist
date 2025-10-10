// utils.h
// Utility functions for Generalized Kumaraswamy distributions package
// Author: Lopes, J. E.
// Date: 2025-10-07
//
// This header provides numerical stability functions and parameter validators
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

/*
 * ===========================================================================
 * NUMERIC STABILITY AUXILIARY FUNCTIONS
 * ===========================================================================
 * These functions ensure accurate numerical calculations even in extreme
 * situations, near distribution boundaries, or with very small/large values.
 */

// Constants for numeric stability and precision
static const double EPSILON      = std::numeric_limits<double>::epsilon();
static const double DBL_MIN_SAFE = std::numeric_limits<double>::min() * 10.0;
static const double LOG_DBL_MIN  = std::log(DBL_MIN_SAFE);
static const double LOG_DBL_MAX  = std::log(std::numeric_limits<double>::max() / 10.0);
static const double LN2          = std::log(2.0); // More direct computation
static const double SQRT_EPSILON = std::sqrt(EPSILON); // More accurate calculation

/**
 * log1mexp(u) calculates log(1 - exp(u)) with enhanced numerical stability
 *
 * This function is crucial for accurate calculations when u is negative and
 * close to zero, where direct computation would suffer catastrophic cancellation.
 * Uses different approximation methods depending on the range of u.
 *
 * @param u A negative value (log(x) where x < 1)
 * @return log(1 - exp(u)), or NaN if u > 0
 */
inline double log1mexp(double u) {
  // Input validation - u must be non-positive
  if (u > 0.0) {
    return R_NaN;  // log(1 - exp(positive)) would yield log of negative number
  }
  
  // For values very close to 0, avoid potential instability
  if (u > -SQRT_EPSILON) {
    return std::log(-u); // Approximation for u ≈ 0-
  }
  
  // For u in (-ln(2), 0], use log(-expm1(u)) for better accuracy
  if (u > -LN2) {
    return std::log(-std::expm1(u));
  }
  
  // For u <= -ln(2), use log1p(-exp(u)) for better accuracy
  return std::log1p(-std::exp(u));
}

/**
 * log1pexp(x) calculates log(1 + exp(x)) with protection against overflow
 *
 * This function handles various regimes of x with appropriate approximations
 * to maintain numerical stability across the entire real line.
 *
 * @param x Input value
 * @return log(1 + exp(x)) calculated with numerical stability
 */
inline double log1pexp(double x) {
  // Improved cutoff points based on numerical analysis
  if (x > 700.0)    return x;                      // For very large x, log(1+exp(x)) ≈ x
  if (x > 37.0)     return x + std::exp(-x);       // For large x, more efficient approximation
  if (x > 18.0)     return x + std::log1p(std::exp(-x)); // For moderately large x
  if (x > -37.0)    return std::log1p(std::exp(x));      // For moderate x
  if (x > -700.0)   return std::exp(x);            // For negative x, where exp(x) is small but not negligible
  return 0.0;                                       // For extremely negative x, where exp(x) ≈ 0
}

/**
 * safe_log(x) computes log(x) with protection against invalid inputs
 *
 * @param x Input value
 * @return log(x) or appropriate limiting value for x <= 0 or very small x
 */
inline double safe_log(double x) {
  // Handle invalid or problematic inputs
  if (x <= 0.0) {
    if (x == 0.0) return R_NegInf;  // Log of zero is -Infinity
    return R_NaN;                   // Log of negative is NaN
  }
  
  // Handle potential underflow
  if (x < DBL_MIN_SAFE) return LOG_DBL_MIN + std::log(x / DBL_MIN_SAFE); // More accurate scaling
  
  return std::log(x);
}

/**
 * safe_exp(x) computes exp(x) with protection against overflow/underflow
 *
 * @param x Input value
 * @return exp(x) or appropriate limiting value for extreme x
 */
inline double safe_exp(double x) {
  // Handle extreme values to prevent overflow/underflow
  if (x > LOG_DBL_MAX) return R_PosInf;  // Prevent overflow
  if (x < LOG_DBL_MIN) {
    if (x < LOG_DBL_MIN - 10.0) return 0.0; // Far below threshold - return 0
    return DBL_MIN_SAFE * std::exp(x - LOG_DBL_MIN); // Scaled computation near threshold
  }
  
  return std::exp(x);
}

/**
 * safe_pow(x, y) computes x^y with robust error handling
 *
 * Handles special cases and applies logarithmic transformation for stability with positive base.
 * Also properly handles edge cases like 0^0, negative bases, and extreme values.
 *
 * @param x Base value
 * @param y Exponent value
 * @return x^y calculated with numerical stability
 */
inline double safe_pow(double x, double y) {
  // Handle special cases
  if (std::isnan(x) || std::isnan(y)) return R_NaN;
  
  // Handle x = 0 cases
  if (x == 0.0) {
    if (y > 0.0)  return 0.0;
    if (y == 0.0) return 1.0;   // 0^0 convention
    return R_PosInf;            // 0^negative is undefined/infinity
  }
  
  // Common trivial cases
  if (x == 1.0 || y == 0.0) return 1.0;
  if (y == 1.0) return x;
  
  // Check for negative base with non-integer exponent (undefined in real domain)
  if (x < 0.0) {
    // Check if y is effectively an integer
    double y_rounded = std::round(y);
    if (std::abs(y - y_rounded) > SQRT_EPSILON) {
      return R_NaN;  // Non-integer power of negative number
    }
    
    // Handle integer powers of negative numbers
    int y_int = static_cast<int>(y_rounded);
    double abs_result = std::pow(std::abs(x), std::abs(y));
    
    // Apply sign: negative^odd = negative, negative^even = positive
    bool negative_result = (y_int % 2 != 0);
    
    // Handle potential over/underflow
    if (y < 0) {
      if (abs_result > 1.0/DBL_MIN_SAFE && negative_result) return -R_PosInf;
      if (abs_result > 1.0/DBL_MIN_SAFE) return R_PosInf;
      return negative_result ? -1.0/abs_result : 1.0/abs_result;
    }
    
    return negative_result ? -abs_result : abs_result;
  }
  
  // For positive base, compute via logarithm for better numerical stability
  if (std::abs(y) > 1e10) {
    // Handle extreme exponents separately
    double lx = std::log(x);
    if (lx < 0.0 && y > 0.0 && y * std::abs(lx) > LOG_DBL_MAX) return 0.0; // Very small result
    if (lx > 0.0 && y > 0.0 && y * lx > LOG_DBL_MAX) return R_PosInf;      // Very large result
    if (lx < 0.0 && y < 0.0 && y * lx < -LOG_DBL_MAX) return R_PosInf;     // Very large result
    if (lx > 0.0 && y < 0.0 && y * std::abs(lx) > LOG_DBL_MAX) return 0.0; // Very small result
  }
  
  // Normal case: compute via logarithm
  double lx = std::log(x);
  double log_result = y * lx;
  return safe_exp(log_result);
}

/**
 * Vector version of log1mexp for element-wise operations on arma::vec
 *
 * @param u Vector of input values
 * @return Vector of log(1 - exp(u)) values
 */
inline arma::vec vec_log1mexp(const arma::vec& u) {
  arma::vec result(u.n_elem);
  
  // Process each element individually for maximum reliability
  for (size_t i = 0; i < u.n_elem; ++i) {
    double ui = u(i);
    
    // Input validation - ui must be non-positive
    if (ui > 0.0) {
      result(i) = arma::datum::nan;
      continue;
    }
    
    // For values very close to 0, avoid potential instability
    if (ui > -SQRT_EPSILON) {
      result(i) = std::log(-ui);
      continue;
    }
    
    // For ui in (-ln(2), 0], use log(-expm1(ui)) for better accuracy
    if (ui > -LN2) {
      result(i) = std::log(-std::expm1(ui));
      continue;
    }
    
    // For ui <= -ln(2), use log1p(-exp(ui)) for better accuracy
    result(i) = std::log1p(-std::exp(ui));
  }
  
  return result;
}

/**
 * Vector version of log1pexp for element-wise operations on arma::vec
 *
 * @param x Vector of input values
 * @return Vector of log(1 + exp(x)) values
 */
inline arma::vec vec_log1pexp(const arma::vec& x) {
  arma::vec result(x.n_elem);
  
  // Process each element individually with optimized computation
  for (size_t i = 0; i < x.n_elem; ++i) {
    double xi = x(i);
    
    // Apply appropriate approximation based on value range
    if (xi > 700.0) {
      result(i) = xi;
    } else if (xi > 37.0) {
      result(i) = xi + std::exp(-xi);
    } else if (xi > 18.0) {
      result(i) = xi + std::log1p(std::exp(-xi));
    } else if (xi > -37.0) {
      result(i) = std::log1p(std::exp(xi));
    } else if (xi > -700.0) {
      result(i) = std::exp(xi);
    } else {
      result(i) = 0.0;  // For extremely negative values
    }
  }
  
  return result;
}

/**
 * Vector version of safe_log for element-wise operations on arma::vec
 *
 * @param x Vector of input values
 * @return Vector of safe_log(x) values
 */
inline arma::vec vec_safe_log(const arma::vec& x) {
  arma::vec result(x.n_elem);
  
  // Process each element individually
  for (size_t i = 0; i < x.n_elem; ++i) {
    double xi = x(i);
    
    // Handle invalid or problematic inputs
    if (xi < 0.0) {
      result(i) = arma::datum::nan;
    } else if (xi == 0.0) {
      result(i) = -arma::datum::inf;
    } else if (xi < DBL_MIN_SAFE) {
      // Handle potential underflow with better scaling
      result(i) = LOG_DBL_MIN + std::log(xi / DBL_MIN_SAFE);
    } else {
      result(i) = std::log(xi);
    }
  }
  
  return result;
}

/**
 * Vector version of safe_exp for element-wise operations on arma::vec
 *
 * @param x Vector of input values
 * @return Vector of safe_exp(x) values
 */
inline arma::vec vec_safe_exp(const arma::vec& x) {
  arma::vec result(x.n_elem);
  
  // Process each element individually
  for (size_t i = 0; i < x.n_elem; ++i) {
    double xi = x(i);
    
    // Handle extreme values to prevent overflow/underflow
    if (xi > LOG_DBL_MAX) {
      result(i) = arma::datum::inf;
    } else if (xi < LOG_DBL_MIN - 10.0) {
      result(i) = 0.0;  // Far below threshold
    } else if (xi < LOG_DBL_MIN) {
      // Scaled computation near threshold for better accuracy
      result(i) = DBL_MIN_SAFE * std::exp(xi - LOG_DBL_MIN);
    } else {
      result(i) = std::exp(xi);
    }
  }
  
  return result;
}

/**
 * Vector version of safe_pow for element-wise operations
 *
 * @param x Vector of base values
 * @param y Single exponent value
 * @return Vector of x[i]^y values
 */
inline arma::vec vec_safe_pow(const arma::vec& x, double y) {
  arma::vec result(x.n_elem);
  
  // Special case handling for trivial exponents
  if (y == 0.0) {
    return arma::vec(x.n_elem, arma::fill::ones);
  }
  
  if (y == 1.0) {
    return x;
  }
  
  // Check if y is effectively an integer for negative base handling
  bool y_is_int = (std::abs(y - std::round(y)) <= SQRT_EPSILON);
  int y_int = static_cast<int>(std::round(y));
  bool y_is_odd = y_is_int && (y_int % 2 != 0);
  
  // Process each element individually
  for (size_t i = 0; i < x.n_elem; ++i) {
    double xi = x(i);
    
    // Handle special cases
    if (std::isnan(xi)) {
      result(i) = arma::datum::nan;
      continue;
    }
    
    // Handle x = 0 cases
    if (xi == 0.0) {
      if (y > 0.0) {
        result(i) = 0.0;
      } else if (y == 0.0) {
        result(i) = 1.0;  // 0^0 convention
      } else {
        result(i) = arma::datum::inf;  // 0^negative
      }
      continue;
    }
    
    // Handle x = 1 case
    if (xi == 1.0) {
      result(i) = 1.0;
      continue;
    }
    
    // Handle negative base cases
    if (xi < 0.0) {
      if (!y_is_int) {
        // Non-integer power of negative not defined in reals
        result(i) = arma::datum::nan;
      } else {
        // Process integer powers of negative numbers
        double abs_xi = std::abs(xi);
        double abs_result = std::pow(abs_xi, std::abs(y));
        
        // Apply sign for odd powers
        if (y < 0) {
          if (y_is_odd) {
            result(i) = -1.0 / abs_result;
          } else {
            result(i) = 1.0 / abs_result;
          }
        } else {
          if (y_is_odd) {
            result(i) = -abs_result;
          } else {
            result(i) = abs_result;
          }
        }
      }
      continue;
    }
    
    // For positive base, compute via logarithm for better numerical stability
    // Handle extreme exponents separately
    if (std::abs(y) > 1e10) {
      double lx = std::log(xi);
      if (lx < 0.0 && y > 0.0 && y * std::abs(lx) > LOG_DBL_MAX) {
        result(i) = 0.0;  // Very small result
      } else if (lx > 0.0 && y > 0.0 && y * lx > LOG_DBL_MAX) {
        result(i) = arma::datum::inf;  // Very large result
      } else if (lx < 0.0 && y < 0.0 && y * lx < -LOG_DBL_MAX) {
        result(i) = arma::datum::inf;  // Very large result
      } else if (lx > 0.0 && y < 0.0 && y * std::abs(lx) > LOG_DBL_MAX) {
        result(i) = 0.0;  // Very small result
      } else {
        double log_result = y * lx;
        result(i) = safe_exp(log_result);
      }
    } else {
      // Normal case: compute via logarithm
      double log_result = y * std::log(xi);
      result(i) = safe_exp(log_result);
    }
  }
  
  return result;
}

/**
 * Vector version of safe_pow with vector exponents
 *
 * @param x Vector of base values
 * @param y Vector of exponent values (must match size of x)
 * @return Vector of x[i]^y[i] values
 */
inline arma::vec vec_safe_pow(const arma::vec& x, const arma::vec& y) {
  if (x.n_elem != y.n_elem) {
    Rcpp::stop("Vectors must have same length in vec_safe_pow");
  }
  
  arma::vec result(x.n_elem);
  
  // Process element-wise with scalar function for maximum reliability
  for (size_t i = 0; i < x.n_elem; ++i) {
    result(i) = safe_pow(x(i), y(i));
  }
  
  return result;
}

/**
 * Checks if GKw parameters are in the valid domain
 *
 * Verifies that all parameters satisfy the constraints:
 * alpha > 0, beta > 0, gamma > 0, delta >= 0, lambda > 0
 *
 * With strict=true, also enforces reasonable bounds to avoid numerical issues.
 *
 * @param alpha Shape parameter
 * @param beta Shape parameter
 * @param gamma Shape parameter
 * @param delta Shape parameter
 * @param lambda Shape parameter
 * @param strict Whether to enforce additional bounds for numerical stability
 * @return true if parameters are valid, false otherwise
 */
inline bool check_pars(double alpha,
                       double beta,
                       double gamma,
                       double delta,
                       double lambda,
                       bool strict = false) {
  // Check for NaN values first
  if (std::isnan(alpha) || std::isnan(beta) || std::isnan(gamma) ||
      std::isnan(delta) || std::isnan(lambda)) {
    return false;
  }
  
  // Basic parameter constraints
  if (alpha <= 0.0 || beta <= 0.0 || gamma <= 0.0 || delta < 0.0 || lambda <= 0.0) {
    return false;
  }
  
  // Optional stricter constraints to avoid numerical issues
  if (strict) {
    const double MIN_PARAM = 1e-5;
    const double MAX_PARAM = 1e5;
    
    if (alpha < MIN_PARAM || beta < MIN_PARAM || gamma < MIN_PARAM || lambda < MIN_PARAM ||
        (delta > 0.0 && delta < MIN_PARAM)) {
      return false;
    }
    if (alpha > MAX_PARAM || beta > MAX_PARAM || gamma > MAX_PARAM ||
        delta > MAX_PARAM || lambda > MAX_PARAM) {
      return false;
    }
  }
  return true;
}

/**
 * Vector version of parameter checker for GKw distribution
 *
 * Checks all combinations of parameter values for validity.
 *
 * @param alpha Vector of alpha values
 * @param beta Vector of beta values
 * @param gamma Vector of gamma values
 * @param delta Vector of delta values
 * @param lambda Vector of lambda values
 * @param strict Whether to enforce additional bounds for numerical stability
 * @return arma::uvec of boolean values indicating parameter validity
 */
inline arma::uvec check_pars_vec(const arma::vec& alpha,
                                 const arma::vec& beta,
                                 const arma::vec& gamma,
                                 const arma::vec& delta,
                                 const arma::vec& lambda,
                                 bool strict = false) {
  // Find maximum length for broadcasting
  size_t n = std::max({alpha.n_elem, beta.n_elem, gamma.n_elem,
                      delta.n_elem, lambda.n_elem});
  
  arma::uvec valid(n, arma::fill::ones);
  
  for (size_t i = 0; i < n; ++i) {
    // Get parameter values with proper cycling/broadcasting
    double a = alpha[i % alpha.n_elem];
    double b = beta[i % beta.n_elem];
    double g = gamma[i % gamma.n_elem];
    double d = delta[i % delta.n_elem];
    double l = lambda[i % lambda.n_elem];
    
    valid[i] = check_pars(a, b, g, d, l, strict);
  }
  
  return valid;
}

/**
 * Parameter Checker for kkw Distribution
 * kkw(α, β, 1, δ, λ) => alpha>0, beta>0, delta≥0, λ>0
 *
 * @param alpha Shape parameter (must be > 0)
 * @param beta Shape parameter (must be > 0)
 * @param delta Shape parameter (must be >= 0)
 * @param lambda Shape parameter (must be > 0)
 * @param strict Whether to enforce additional bounds for numerical stability
 * @return true if parameters are valid, false otherwise
 */
inline bool check_kkw_pars(double alpha,
                           double beta,
                           double delta,
                           double lambda,
                           bool strict = false) {
  if (alpha <= 0.0 || beta <= 0.0 || delta < 0.0 || lambda <= 0.0) {
    return false;
  }
  if (strict) {
    const double MINP = 1e-5;
    const double MAXP = 1e5;
    if (alpha < MINP || beta < MINP || lambda < MINP) {
      return false;
    }
    if (alpha > MAXP || beta > MAXP || delta > MAXP || lambda > MAXP) {
      return false;
    }
  }
  return true;
}

/**
 * Parameter checker for Beta-Kumaraswamy (BKw) distribution
 *
 * @param alpha Shape parameter (must be > 0)
 * @param beta Shape parameter (must be > 0)
 * @param gamma Shape parameter (must be > 0)
 * @param delta Shape parameter (must be >= 0)
 * @param strict Whether to enforce additional bounds for numerical stability
 * @return true if parameters are valid, false otherwise
 */
inline bool check_bkw_pars(double alpha,
                           double beta,
                           double gamma,
                           double delta,
                           bool strict = false) {
  if (alpha <= 0.0 || beta <= 0.0 || gamma <= 0.0 || delta < 0.0) {
    return false;
  }
  
  if (strict) {
    // Optional stricter numeric bounds
    const double MIN_PARAM = 1e-5;
    const double MAX_PARAM = 1e5;
    if (alpha < MIN_PARAM || beta < MIN_PARAM || gamma < MIN_PARAM || delta > MAX_PARAM) {
      return false;
    }
    if (alpha > MAX_PARAM || beta > MAX_PARAM || gamma > MAX_PARAM) {
      return false;
    }
  }
  return true;
}

/**
 * Parameter checker for EKw distribution
 * EKw(α, β, λ):  all must be > 0
 *
 * @param alpha Shape parameter (must be > 0)
 * @param beta Shape parameter (must be > 0)
 * @param lambda Shape parameter (must be > 0)
 * @param strict Whether to enforce additional bounds for numerical stability
 * @return true if parameters are valid, false otherwise
 */
inline bool check_ekw_pars(double alpha, double beta, double lambda, bool strict=false) {
  if (alpha <= 0.0 || beta <= 0.0 || lambda <= 0.0) {
    return false;
  }
  if (strict) {
    const double MINP = 1e-6;
    const double MAXP = 1e6;
    if (alpha < MINP || beta < MINP || lambda < MINP)  return false;
    if (alpha > MAXP || beta > MAXP || lambda > MAXP)  return false;
  }
  return true;
}

/**
 * Parameter checker for Beta Power distribution
 * BP(γ>0, δ≥0, λ>0)
 *
 * @param gamma Shape parameter (must be > 0)
 * @param delta Shape parameter (must be >= 0)
 * @param lambda Shape parameter (must be > 0)
 * @param strict Whether to enforce additional bounds for numerical stability
 * @return true if parameters are valid, false otherwise
 */
inline bool check_bp_pars(double gamma, double delta, double lambda, bool strict = false) {
  if (gamma <= 0.0 || delta < 0.0 || lambda <= 0.0) {
    return false;
  }
  if (strict) {
    const double MINP=1e-8;
    const double MAXP=1e8;
    if (gamma<MINP || lambda<MINP) return false;
    if (gamma>MAXP || delta>MAXP || lambda>MAXP) return false;
  }
  return true;
}

/**
 * Parameter checker for Kumaraswamy distribution
 * alpha>0, beta>0
 *
 * @param alpha Shape parameter (must be > 0)
 * @param beta Shape parameter (must be > 0)
 * @param strict Whether to enforce additional bounds for numerical stability
 * @return true if parameters are valid, false otherwise
 */
inline bool check_kw_pars(double alpha, double beta, bool strict=false) {
  if (alpha <=0.0 || beta <=0.0) {
    return false;
  }
  if (strict) {
    const double MINP=1e-6, MAXP=1e6;
    if (alpha<MINP || beta<MINP) return false;
    if (alpha>MAXP || beta>MAXP) return false;
  }
  return true;
}

/**
 * Parameter checker for Beta distribution
 * Beta(gamma>0, delta>0)
 *
 * @param gamma Shape parameter (must be > 0)
 * @param delta Shape parameter (must be > 0)
 * @param strict Whether to enforce additional bounds for numerical stability
 * @return true if parameters are valid, false otherwise
 */
inline bool check_beta_pars(double gamma, double delta, bool strict=false) {
  if (gamma <= 0.0 || delta <= 0.0) {
    return false;
  }
  if (strict) {
    const double MINP = 1e-7, MAXP = 1e7;
    if (gamma < MINP || delta < MINP) return false;
    if (gamma > MAXP || delta > MAXP) return false;
  }
  return true;
}

#endif // GKWDIST_UTILS_H
