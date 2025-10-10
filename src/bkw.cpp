// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include "utils.h"

/*
----------------------------------------------------------------------------
BETA-KUMARASWAMY (BKw) DISTRIBUTION
----------------------------------------------------------------------------
PDF:
f(x; α, β, γ, δ) = (α β / B(γ, δ+1)) x^(α-1) (1 - x^α)^( β(δ+1) - 1 )
[ 1 - (1 - x^α)^β ]^(γ - 1)

CDF:
F(x; α, β, γ, δ) = I_{ [1 - (1 - x^α)^β ] } ( γ, δ + 1 )

QUANTILE:
Q(p; α, β, γ, δ) = { 1 - [1 - ( I^{-1}_{p}(γ, δ+1) ) ]^(1/β) }^(1/α)
(But see transformations step-by-step in code for numeric stability.)

RNG:
If V ~ Beta(γ, δ+1) then
X = { 1 - [1 - V ]^(1/β) }^(1/α)

LOG-LIKELIHOOD:
ℓ(θ) = n log(α β) - n log B(γ, δ+1)
+ Σ { (α-1) log(x_i) + [β(δ+1)-1] log(1 - x_i^α) + (γ - 1) log( 1 - (1 - x_i^α)^β ) }

This file defines:
- dbkw()  : density
- pbkw()  : cumulative distribution
- qbkw()  : quantile
- rbkw()  : random generation
- llbkw() : negative log-likelihood
*/


// -----------------------------------------------------------------------------
// 1) dbkw: PDF of Beta-Kumaraswamy
// -----------------------------------------------------------------------------

//' @title Density of the Beta-Kumaraswamy (BKw) Distribution
//' @author Lopes, J. E.
//' @keywords distribution density
//'
//' @description
//' Computes the probability density function (PDF) for the Beta-Kumaraswamy
//' (BKw) distribution with parameters \code{alpha} (\eqn{\alpha}), \code{beta}
//' (\eqn{\beta}), \code{gamma} (\eqn{\gamma}), and \code{delta} (\eqn{\delta}).
//' This distribution is defined on the interval (0, 1).
//'
//' @param x Vector of quantiles (values between 0 and 1).
//' @param alpha Shape parameter \code{alpha} > 0. Can be a scalar or a vector.
//'   Default: 1.0.
//' @param beta Shape parameter \code{beta} > 0. Can be a scalar or a vector.
//'   Default: 1.0.
//' @param gamma Shape parameter \code{gamma} > 0. Can be a scalar or a vector.
//'   Default: 1.0.
//' @param delta Shape parameter \code{delta} >= 0. Can be a scalar or a vector.
//'   Default: 0.0.
//' @param log_prob Logical; if \code{TRUE}, the logarithm of the density is
//'   returned (\eqn{\log(f(x))}). Default: \code{FALSE}.
//'
//' @return A vector of density values (\eqn{f(x)}) or log-density values
//'   (\eqn{\log(f(x))}). The length of the result is determined by the recycling
//'   rule applied to the arguments (\code{x}, \code{alpha}, \code{beta},
//'   \code{gamma}, \code{delta}). Returns \code{0} (or \code{-Inf} if
//'   \code{log_prob = TRUE}) for \code{x} outside the interval (0, 1), or
//'   \code{NaN} if parameters are invalid (e.g., \code{alpha <= 0}, \code{beta <= 0},
//'   \code{gamma <= 0}, \code{delta < 0}).
//'
//' @details
//' The probability density function (PDF) of the Beta-Kumaraswamy (BKw)
//' distribution is given by:
//' \deqn{
//' f(x; \alpha, \beta, \gamma, \delta) = \frac{\alpha \beta}{B(\gamma, \delta+1)} x^{\alpha - 1} \bigl(1 - x^\alpha\bigr)^{\beta(\delta+1) - 1} \bigl[1 - \bigl(1 - x^\alpha\bigr)^\beta\bigr]^{\gamma - 1}
//' }
//' for \eqn{0 < x < 1}, where \eqn{B(a,b)} is the Beta function
//' (\code{\link[base]{beta}}).
//'
//' The BKw distribution is a special case of the five-parameter
//' Generalized Kumaraswamy (GKw) distribution (\code{\link{dgkw}}) obtained
//' by setting the parameter \eqn{\lambda = 1}.
//' Numerical evaluation is performed using algorithms similar to those for `dgkw`,
//' ensuring stability.
//'
//' @references
//' Cordeiro, G. M., & de Castro, M. (2011). A new family of generalized
//' distributions. *Journal of Statistical Computation and Simulation*
//'
//' Kumaraswamy, P. (1980). A generalized probability density function for
//' double-bounded random processes. *Journal of Hydrology*, *46*(1-2), 79-88.
//'
//' @seealso
//' \code{\link{dgkw}} (parent distribution density),
//' \code{\link{pbkw}}, \code{\link{qbkw}}, \code{\link{rbkw}} (other BKw functions),
//'
//' @examples
//' \donttest{
//' # Example values
//' x_vals <- c(0.2, 0.5, 0.8)
//' alpha_par <- 2.0
//' beta_par <- 1.5
//' gamma_par <- 1.0 # Equivalent to Kw when gamma=1
//' delta_par <- 0.5
//'
//' # Calculate density
//' densities <- dbkw(x_vals, alpha_par, beta_par, gamma_par, delta_par)
//' print(densities)
//'
//' # Calculate log-density
//' log_densities <- dbkw(x_vals, alpha_par, beta_par, gamma_par, delta_par,
//'                       log_prob = TRUE)
//' print(log_densities)
//' # Check: should match log(densities)
//' print(log(densities))
//'
//' # Compare with dgkw setting lambda = 1
//' densities_gkw <- dgkw(x_vals, alpha_par, beta_par, gamma = gamma_par,
//'                       delta = delta_par, lambda = 1.0)
//' print(paste("Max difference:", max(abs(densities - densities_gkw)))) # Should be near zero
//'
//' # Plot the density for different gamma values
//' curve_x <- seq(0.01, 0.99, length.out = 200)
//' curve_y1 <- dbkw(curve_x, alpha = 2, beta = 3, gamma = 0.5, delta = 1)
//' curve_y2 <- dbkw(curve_x, alpha = 2, beta = 3, gamma = 1.0, delta = 1)
//' curve_y3 <- dbkw(curve_x, alpha = 2, beta = 3, gamma = 2.0, delta = 1)
//'
//' plot(curve_x, curve_y1, type = "l", main = "BKw Density Examples (alpha=2, beta=3, delta=1)",
//'      xlab = "x", ylab = "f(x)", col = "blue", ylim = range(0, curve_y1, curve_y2, curve_y3))
//' lines(curve_x, curve_y2, col = "red")
//' lines(curve_x, curve_y3, col = "green")
//' legend("topright", legend = c("gamma=0.5", "gamma=1.0", "gamma=2.0"),
//'        col = c("blue", "red", "green"), lty = 1, bty = "n")
//' }
//'
//' @export
// [[Rcpp::export]]
Rcpp::NumericVector dbkw(
   const arma::vec& x,
   const Rcpp::NumericVector& alpha,
   const Rcpp::NumericVector& beta,
   const Rcpp::NumericVector& gamma,
   const Rcpp::NumericVector& delta,
   bool log_prob = false
) {
 // Convert to arma::vec
 arma::vec alpha_vec(alpha.begin(), alpha.size());
 arma::vec beta_vec(beta.begin(), beta.size());
 arma::vec gamma_vec(gamma.begin(), gamma.size());
 arma::vec delta_vec(delta.begin(), delta.size());
 
 // Broadcast length
 size_t n = std::max({x.n_elem,
                     alpha_vec.n_elem,
                     beta_vec.n_elem,
                     gamma_vec.n_elem,
                     delta_vec.n_elem});
 
 // Result
 arma::vec result(n);
 result.fill(log_prob ? R_NegInf : 0.0);
 
 for (size_t i = 0; i < n; ++i) {
   double a = alpha_vec[i % alpha_vec.n_elem];
   double b = beta_vec[i % beta_vec.n_elem];
   double g = gamma_vec[i % gamma_vec.n_elem];
   double d = delta_vec[i % delta_vec.n_elem];
   double xx = x[i % x.n_elem];
   
   // Check parameter validity
   if (!check_bkw_pars(a, b, g, d)) {
     // Invalid params => density = 0 or -Inf
     continue;
   }
   
   // Outside (0,1) => density=0 or log_density=-Inf
   if (xx <= 0.0 || xx >= 1.0 || !R_finite(xx)) {
     continue;
   }
   
   // PDF formula
   // f(x) = (alpha*beta / B(gamma, delta+1)) *
   //        x^(alpha-1) * (1 - x^alpha)^(beta*(delta+1) - 1) *
   //        [1 - (1 - x^alpha)^beta]^(gamma - 1)
   
   // Precompute log_B = lbeta(g, d+1)
   double logB = R::lbeta(g, d + 1.0);
   double log_const = std::log(a) + std::log(b) - logB;
   
   double lx = std::log(xx);
   double xalpha = a * lx;                    // log(x^alpha) = a * log(x)
   double log_1_minus_xalpha = log1mexp(xalpha);
   
   // (beta*(delta+1) - 1) * log(1 - x^alpha)
   double exponent1 = b * (d + 1.0) - 1.0;
   double term1 = exponent1 * log_1_minus_xalpha;
   
   // [1 - (1 - x^alpha)^beta]^(gamma - 1)
   // log(1 - (1 - x^alpha)^beta) = log1mexp( b * log(1 - x^alpha) )
   double log_1_minus_xalpha_beta = b * log_1_minus_xalpha;
   double log_bracket = log1mexp(log_1_minus_xalpha_beta);
   double exponent2 = g - 1.0;
   double term2 = exponent2 * log_bracket;
   
   // Full log pdf
   double log_pdf = log_const +
     (a - 1.0) * lx +
     term1 +
     term2;
   
   if (log_prob) {
     result(i) = log_pdf;
   } else {
     // exp safely
     result(i) = std::exp(log_pdf);
   }
 }
 
 return Rcpp::NumericVector(result.memptr(), result.memptr() + result.n_elem);
}


// -----------------------------------------------------------------------------
// 2) pbkw: CDF of Beta-Kumaraswamy
// -----------------------------------------------------------------------------

//' @title Cumulative Distribution Function (CDF) of the Beta-Kumaraswamy (BKw) Distribution
//' @author Lopes, J. E.
//' @keywords distribution cumulative
//'
//' @description
//' Computes the cumulative distribution function (CDF), \eqn{P(X \le q)}, for the
//' Beta-Kumaraswamy (BKw) distribution with parameters \code{alpha} (\eqn{\alpha}),
//' \code{beta} (\eqn{\beta}), \code{gamma} (\eqn{\gamma}), and \code{delta}
//' (\eqn{\delta}). This distribution is defined on the interval (0, 1) and is
//' a special case of the Generalized Kumaraswamy (GKw) distribution where
//' \eqn{\lambda = 1}.
//'
//' @param q Vector of quantiles (values generally between 0 and 1).
//' @param alpha Shape parameter \code{alpha} > 0. Can be a scalar or a vector.
//'   Default: 1.0.
//' @param beta Shape parameter \code{beta} > 0. Can be a scalar or a vector.
//'   Default: 1.0.
//' @param gamma Shape parameter \code{gamma} > 0. Can be a scalar or a vector.
//'   Default: 1.0.
//' @param delta Shape parameter \code{delta} >= 0. Can be a scalar or a vector.
//'   Default: 0.0.
//' @param lower_tail Logical; if \code{TRUE} (default), probabilities are
//'   \eqn{P(X \le q)}, otherwise, \eqn{P(X > q)}.
//' @param log_p Logical; if \code{TRUE}, probabilities \eqn{p} are given as
//'   \eqn{\log(p)}. Default: \code{FALSE}.
//'
//' @return A vector of probabilities, \eqn{F(q)}, or their logarithms/complements
//'   depending on \code{lower_tail} and \code{log_p}. The length of the result
//'   is determined by the recycling rule applied to the arguments (\code{q},
//'   \code{alpha}, \code{beta}, \code{gamma}, \code{delta}). Returns \code{0}
//'   (or \code{-Inf} if \code{log_p = TRUE}) for \code{q <= 0} and \code{1}
//'   (or \code{0} if \code{log_p = TRUE}) for \code{q >= 1}. Returns \code{NaN}
//'   for invalid parameters.
//'
//' @details
//' The Beta-Kumaraswamy (BKw) distribution is a special case of the
//' five-parameter Generalized Kumaraswamy distribution (\code{\link{pgkw}})
//' obtained by setting the shape parameter \eqn{\lambda = 1}.
//'
//' The CDF of the GKw distribution is \eqn{F_{GKw}(q) = I_{y(q)}(\gamma, \delta+1)},
//' where \eqn{y(q) = [1-(1-q^{\alpha})^{\beta}]^{\lambda}} and \eqn{I_x(a,b)}
//' is the regularized incomplete beta function (\code{\link[stats]{pbeta}}).
//' Setting \eqn{\lambda=1} simplifies \eqn{y(q)} to \eqn{1 - (1 - q^\alpha)^\beta},
//' yielding the BKw CDF:
//' \deqn{
//' F(q; \alpha, \beta, \gamma, \delta) = I_{1 - (1 - q^\alpha)^\beta}(\gamma, \delta+1)
//' }
//' This is evaluated using the \code{\link[stats]{pbeta}} function.
//'
//' @references
//' Cordeiro, G. M., & de Castro, M. (2011). A new family of generalized
//' distributions. *Journal of Statistical Computation and Simulation*
//'
//' Kumaraswamy, P. (1980). A generalized probability density function for
//' double-bounded random processes. *Journal of Hydrology*, *46*(1-2), 79-88.
//'
//' @seealso
//' \code{\link{pgkw}} (parent distribution CDF),
//' \code{\link{dbkw}}, \code{\link{qbkw}}, \code{\link{rbkw}} (other BKw functions),
//' \code{\link[stats]{pbeta}}
//'
//' @examples
//' \donttest{
//' # Example values
//' q_vals <- c(0.2, 0.5, 0.8)
//' alpha_par <- 2.0
//' beta_par <- 1.5
//' gamma_par <- 1.0
//' delta_par <- 0.5
//'
//' # Calculate CDF P(X <= q)
//' probs <- pbkw(q_vals, alpha_par, beta_par, gamma_par, delta_par)
//' print(probs)
//'
//' # Calculate upper tail P(X > q)
//' probs_upper <- pbkw(q_vals, alpha_par, beta_par, gamma_par, delta_par,
//'                     lower_tail = FALSE)
//' print(probs_upper)
//' # Check: probs + probs_upper should be 1
//' print(probs + probs_upper)
//'
//' # Calculate log CDF
//' log_probs <- pbkw(q_vals, alpha_par, beta_par, gamma_par, delta_par,
//'                   log_p = TRUE)
//' print(log_probs)
//' # Check: should match log(probs)
//' print(log(probs))
//'
//' # Compare with pgkw setting lambda = 1
//' probs_gkw <- pgkw(q_vals, alpha_par, beta_par, gamma = gamma_par,
//'                  delta = delta_par, lambda = 1.0)
//' print(paste("Max difference:", max(abs(probs - probs_gkw)))) # Should be near zero
//'
//' # Plot the CDF
//' curve_q <- seq(0.01, 0.99, length.out = 200)
//' curve_p <- pbkw(curve_q, alpha = 2, beta = 3, gamma = 0.5, delta = 1)
//' plot(curve_q, curve_p, type = "l", main = "BKw CDF Example",
//'      xlab = "q", ylab = "F(q)", col = "blue", ylim = c(0, 1))
//' }
//'
//' @export
// [[Rcpp::export]]
Rcpp::NumericVector pbkw(
   const arma::vec& q,
   const Rcpp::NumericVector& alpha,
   const Rcpp::NumericVector& beta,
   const Rcpp::NumericVector& gamma,
   const Rcpp::NumericVector& delta,
   bool lower_tail = true,
   bool log_p = false
) {
 // Convert
 arma::vec alpha_vec(alpha.begin(), alpha.size());
 arma::vec beta_vec(beta.begin(), beta.size());
 arma::vec gamma_vec(gamma.begin(), gamma.size());
 arma::vec delta_vec(delta.begin(), delta.size());
 
 // Broadcast
 size_t n = std::max({q.n_elem,
                     alpha_vec.n_elem,
                     beta_vec.n_elem,
                     gamma_vec.n_elem,
                     delta_vec.n_elem});
 
 arma::vec res(n);
 
 for (size_t i = 0; i < n; ++i) {
   double a = alpha_vec[i % alpha_vec.n_elem];
   double b = beta_vec[i % beta_vec.n_elem];
   double g = gamma_vec[i % gamma_vec.n_elem];
   double d = delta_vec[i % delta_vec.n_elem];
   double xx = q[i % q.n_elem];
   
   if (!check_bkw_pars(a, b, g, d)) {
     res(i) = NA_REAL;
     continue;
   }
   
   if (!R_finite(xx) || xx <= 0.0) {
     // x=0 => F=0
     double prob0 = lower_tail ? 0.0 : 1.0;
     res(i) = log_p ? std::log(prob0) : prob0;
     continue;
   }
   
   if (xx >= 1.0) {
     // x=1 => F=1
     double prob1 = lower_tail ? 1.0 : 0.0;
     res(i) = log_p ? std::log(prob1) : prob1;
     continue;
   }
   
   // We want z = 1 - (1 - x^alpha)^beta
   double lx = std::log(xx);
   double xalpha = std::exp(a * lx);
   double one_minus_xalpha = 1.0 - xalpha;
   
   if (one_minus_xalpha <= 0.0) {
     // F(x) ~ 1 if x^alpha>=1
     double prob1 = lower_tail ? 1.0 : 0.0;
     res(i) = log_p ? std::log(prob1) : prob1;
     continue;
   }
   
   double temp = 1.0 - std::pow(one_minus_xalpha, b);
   if (temp <= 0.0) {
     double prob0 = lower_tail ? 0.0 : 1.0;
     res(i) = log_p ? std::log(prob0) : prob0;
     continue;
   }
   
   if (temp >= 1.0) {
     double prob1 = lower_tail ? 1.0 : 0.0;
     res(i) = log_p ? std::log(prob1) : prob1;
     continue;
   }
   
   // Then F(x) = pbeta(temp, gamma, delta+1, TRUE, FALSE)
   double val = R::pbeta(temp, g, d+1.0, true, false); // F
   if (!lower_tail) {
     val = 1.0 - val;
   }
   if (log_p) {
     val = std::log(val);
   }
   res(i) = val;
 }
 
 return Rcpp::NumericVector(res.memptr(), res.memptr() + res.n_elem);
}


// -----------------------------------------------------------------------------
// 3) qbkw: QUANTILE of Beta-Kumaraswamy
// -----------------------------------------------------------------------------

//' @title Quantile Function of the Beta-Kumaraswamy (BKw) Distribution
//' @author Lopes, J. E.
//' @keywords distribution quantile
//'
//' @description
//' Computes the quantile function (inverse CDF) for the Beta-Kumaraswamy (BKw)
//' distribution with parameters \code{alpha} (\eqn{\alpha}), \code{beta}
//' (\eqn{\beta}), \code{gamma} (\eqn{\gamma}), and \code{delta} (\eqn{\delta}).
//' It finds the value \code{q} such that \eqn{P(X \le q) = p}. This distribution
//' is a special case of the Generalized Kumaraswamy (GKw) distribution where
//' the parameter \eqn{\lambda = 1}.
//'
//' @param p Vector of probabilities (values between 0 and 1).
//' @param alpha Shape parameter \code{alpha} > 0. Can be a scalar or a vector.
//'   Default: 1.0.
//' @param beta Shape parameter \code{beta} > 0. Can be a scalar or a vector.
//'   Default: 1.0.
//' @param gamma Shape parameter \code{gamma} > 0. Can be a scalar or a vector.
//'   Default: 1.0.
//' @param delta Shape parameter \code{delta} >= 0. Can be a scalar or a vector.
//'   Default: 0.0.
//' @param lower_tail Logical; if \code{TRUE} (default), probabilities are \eqn{p = P(X \le q)},
//'   otherwise, probabilities are \eqn{p = P(X > q)}.
//' @param log_p Logical; if \code{TRUE}, probabilities \code{p} are given as
//'   \eqn{\log(p)}. Default: \code{FALSE}.
//'
//' @return A vector of quantiles corresponding to the given probabilities \code{p}.
//'   The length of the result is determined by the recycling rule applied to
//'   the arguments (\code{p}, \code{alpha}, \code{beta}, \code{gamma}, \code{delta}).
//'   Returns:
//'   \itemize{
//'     \item \code{0} for \code{p = 0} (or \code{p = -Inf} if \code{log_p = TRUE},
//'           when \code{lower_tail = TRUE}).
//'     \item \code{1} for \code{p = 1} (or \code{p = 0} if \code{log_p = TRUE},
//'           when \code{lower_tail = TRUE}).
//'     \item \code{NaN} for \code{p < 0} or \code{p > 1} (or corresponding log scale).
//'     \item \code{NaN} for invalid parameters (e.g., \code{alpha <= 0},
//'           \code{beta <= 0}, \code{gamma <= 0}, \code{delta < 0}).
//'   }
//'   Boundary return values are adjusted accordingly for \code{lower_tail = FALSE}.
//'
//' @details
//' The quantile function \eqn{Q(p)} is the inverse of the CDF \eqn{F(q)}. The CDF
//' for the BKw (\eqn{\lambda=1}) distribution is \eqn{F(q) = I_{y(q)}(\gamma, \delta+1)},
//' where \eqn{y(q) = 1 - (1 - q^\alpha)^\beta} and \eqn{I_z(a,b)} is the
//' regularized incomplete beta function (see \code{\link{pbkw}}).
//'
//' To find the quantile \eqn{q}, we first invert the outer Beta part: let
//' \eqn{y = I^{-1}_{p}(\gamma, \delta+1)}, where \eqn{I^{-1}_p(a,b)} is the
//' inverse of the regularized incomplete beta function, computed via
//' \code{\link[stats]{qbeta}}. Then, we invert the inner Kumaraswamy part:
//' \eqn{y = 1 - (1 - q^\alpha)^\beta}, which leads to \eqn{q = \{1 - (1-y)^{1/\beta}\}^{1/\alpha}}.
//' Substituting \eqn{y} gives the quantile function:
//' \deqn{
//' Q(p) = \left\{ 1 - \left[ 1 - I^{-1}_{p}(\gamma, \delta+1) \right]^{1/\beta} \right\}^{1/\alpha}
//' }
//' The function uses this formula, calculating \eqn{I^{-1}_{p}(\gamma, \delta+1)}
//' via \code{qbeta(p, gamma, delta + 1, ...)} while respecting the
//' \code{lower_tail} and \code{log_p} arguments.
//'
//' @references
//' Cordeiro, G. M., & de Castro, M. (2011). A new family of generalized
//' distributions. *Journal of Statistical Computation and Simulation*
//'
//' Kumaraswamy, P. (1980). A generalized probability density function for
//' double-bounded random processes. *Journal of Hydrology*, *46*(1-2), 79-88.
//'
//' @seealso
//' \code{\link{qgkw}} (parent distribution quantile function),
//' \code{\link{dbkw}}, \code{\link{pbkw}}, \code{\link{rbkw}} (other BKw functions),
//' \code{\link[stats]{qbeta}}
//'
//' @examples
//' \donttest{
//' # Example values
//' p_vals <- c(0.1, 0.5, 0.9)
//' alpha_par <- 2.0
//' beta_par <- 1.5
//' gamma_par <- 1.0
//' delta_par <- 0.5
//'
//' # Calculate quantiles
//' quantiles <- qbkw(p_vals, alpha_par, beta_par, gamma_par, delta_par)
//' print(quantiles)
//'
//' # Calculate quantiles for upper tail probabilities P(X > q) = p
//' quantiles_upper <- qbkw(p_vals, alpha_par, beta_par, gamma_par, delta_par,
//'                         lower_tail = FALSE)
//' print(quantiles_upper)
//' # Check: qbkw(p, ..., lt=F) == qbkw(1-p, ..., lt=T)
//' print(qbkw(1 - p_vals, alpha_par, beta_par, gamma_par, delta_par))
//'
//' # Calculate quantiles from log probabilities
//' log_p_vals <- log(p_vals)
//' quantiles_logp <- qbkw(log_p_vals, alpha_par, beta_par, gamma_par, delta_par,
//'                        log_p = TRUE)
//' print(quantiles_logp)
//' # Check: should match original quantiles
//' print(quantiles)
//'
//' # Compare with qgkw setting lambda = 1
//' quantiles_gkw <- qgkw(p_vals, alpha_par, beta_par, gamma = gamma_par,
//'                      delta = delta_par, lambda = 1.0)
//' print(paste("Max difference:", max(abs(quantiles - quantiles_gkw)))) # Should be near zero
//'
//' # Verify inverse relationship with pbkw
//' p_check <- 0.75
//' q_calc <- qbkw(p_check, alpha_par, beta_par, gamma_par, delta_par)
//' p_recalc <- pbkw(q_calc, alpha_par, beta_par, gamma_par, delta_par)
//' print(paste("Original p:", p_check, " Recalculated p:", p_recalc))
//' # abs(p_check - p_recalc) < 1e-9 # Should be TRUE
//'
//' # Boundary conditions
//' print(qbkw(c(0, 1), alpha_par, beta_par, gamma_par, delta_par)) # Should be 0, 1
//' print(qbkw(c(-Inf, 0), alpha_par, beta_par, gamma_par, delta_par, log_p = TRUE)) # Should be 0, 1
//'
//' }
//'
//' @export
// [[Rcpp::export]]
Rcpp::NumericVector qbkw(
   const arma::vec& p,
   const Rcpp::NumericVector& alpha,
   const Rcpp::NumericVector& beta,
   const Rcpp::NumericVector& gamma,
   const Rcpp::NumericVector& delta,
   bool lower_tail = true,
   bool log_p = false
) {
 arma::vec alpha_vec(alpha.begin(), alpha.size());
 arma::vec beta_vec(beta.begin(), beta.size());
 arma::vec gamma_vec(gamma.begin(), gamma.size());
 arma::vec delta_vec(delta.begin(), delta.size());
 
 size_t n = std::max({p.n_elem,
                     alpha_vec.n_elem,
                     beta_vec.n_elem,
                     gamma_vec.n_elem,
                     delta_vec.n_elem});
 
 arma::vec res(n);
 
 for (size_t i = 0; i < n; ++i) {
   double a = alpha_vec[i % alpha_vec.n_elem];
   double b = beta_vec[i % beta_vec.n_elem];
   double g = gamma_vec[i % gamma_vec.n_elem];
   double d = delta_vec[i % delta_vec.n_elem];
   double pp = p[i % p.n_elem];
   
   if (!check_bkw_pars(a, b, g, d)) {
     res(i) = NA_REAL;
     continue;
   }
   
   // Convert from log_p if needed
   if (log_p) {
     if (pp > 0.0) {
       // log(p) > 0 => p>1 => invalid
       res(i) = NA_REAL;
       continue;
     }
     pp = std::exp(pp);
   }
   // Convert if upper tail
   if (!lower_tail) {
     pp = 1.0 - pp;
   }
   
   // Check boundaries
   if (pp <= 0.0) {
     res(i) = 0.0;
     continue;
   } else if (pp >= 1.0) {
     res(i) = 1.0;
     continue;
   }
   
   // We do: y = qbeta(pp, gamma, delta+1)
   double y = R::qbeta(pp, g, d+1.0, true, false);
   if (y <= 0.0) {
     res(i) = 0.0;
     continue;
   } else if (y >= 1.0) {
     res(i) = 1.0;
     continue;
   }
   
   // Then x = {1 - [1 - y]^(1/b)}^(1/a)
   double part = 1.0 - y;
   if (part <= 0.0) {
     res(i) = 1.0;
     continue;
   } else if (part >= 1.0) {
     res(i) = 0.0;
     continue;
   }
   
   double inner = std::pow(part, 1.0/b);
   double xval = 1.0 - inner;
   if (xval < 0.0)  xval = 0.0;
   if (xval > 1.0)  xval = 1.0;
   
   if (a == 1.0) {
     // small optimization
     res(i) = xval;
   } else {
     double qv = std::pow(xval, 1.0/a);
     if (qv < 0.0)      qv = 0.0;
     else if (qv > 1.0) qv = 1.0;
     res(i) = qv;
   }
 }
 
 return Rcpp::NumericVector(res.memptr(), res.memptr() + res.n_elem);
}


// -----------------------------------------------------------------------------
// 4) rbkw: RNG for Beta-Kumaraswamy
// -----------------------------------------------------------------------------

//' @title Random Number Generation for the Beta-Kumaraswamy (BKw) Distribution
//' @author Lopes, J. E.
//' @keywords distribution random
//'
//' @description
//' Generates random deviates from the Beta-Kumaraswamy (BKw) distribution
//' with parameters \code{alpha} (\eqn{\alpha}), \code{beta} (\eqn{\beta}),
//' \code{gamma} (\eqn{\gamma}), and \code{delta} (\eqn{\delta}). This distribution
//' is a special case of the Generalized Kumaraswamy (GKw) distribution where
//' the parameter \eqn{\lambda = 1}.
//'
//' @param n Number of observations. If \code{length(n) > 1}, the length is
//'   taken to be the number required. Must be a non-negative integer.
//' @param alpha Shape parameter \code{alpha} > 0. Can be a scalar or a vector.
//'   Default: 1.0.
//' @param beta Shape parameter \code{beta} > 0. Can be a scalar or a vector.
//'   Default: 1.0.
//' @param gamma Shape parameter \code{gamma} > 0. Can be a scalar or a vector.
//'   Default: 1.0.
//' @param delta Shape parameter \code{delta} >= 0. Can be a scalar or a vector.
//'   Default: 0.0.
//'
//' @return A vector of length \code{n} containing random deviates from the BKw
//'   distribution. The length of the result is determined by \code{n} and the
//'   recycling rule applied to the parameters (\code{alpha}, \code{beta},
//'   \code{gamma}, \code{delta}). Returns \code{NaN} if parameters
//'   are invalid (e.g., \code{alpha <= 0}, \code{beta <= 0}, \code{gamma <= 0},
//'   \code{delta < 0}).
//'
//' @details
//' The generation method uses the relationship between the GKw distribution and the
//' Beta distribution. The general procedure for GKw (\code{\link{rgkw}}) is:
//' If \eqn{W \sim \mathrm{Beta}(\gamma, \delta+1)}, then
//' \eqn{X = \{1 - [1 - W^{1/\lambda}]^{1/\beta}\}^{1/\alpha}} follows the
//' GKw(\eqn{\alpha, \beta, \gamma, \delta, \lambda}) distribution.
//'
//' For the BKw distribution, \eqn{\lambda=1}. Therefore, the algorithm simplifies to:
//' \enumerate{
//'   \item Generate \eqn{V \sim \mathrm{Beta}(\gamma, \delta+1)} using
//'         \code{\link[stats]{rbeta}}.
//'   \item Compute the BKw variate \eqn{X = \{1 - (1 - V)^{1/\beta}\}^{1/\alpha}}.
//' }
//' This procedure is implemented efficiently, handling parameter recycling as needed.
//'
//' @references
//' Cordeiro, G. M., & de Castro, M. (2011). A new family of generalized
//' distributions. *Journal of Statistical Computation and Simulation*,
//'
//' Kumaraswamy, P. (1980). A generalized probability density function for
//' double-bounded random processes. *Journal of Hydrology*, *46*(1-2), 79-88.
//'
//'
//' Devroye, L. (1986). *Non-Uniform Random Variate Generation*. Springer-Verlag.
//' (General methods for random variate generation).
//'
//' @seealso
//' \code{\link{rgkw}} (parent distribution random generation),
//' \code{\link{dbkw}}, \code{\link{pbkw}}, \code{\link{qbkw}} (other BKw functions),
//' \code{\link[stats]{rbeta}}
//'
//' @examples
//' \donttest{
//' set.seed(2026) # for reproducibility
//'
//' # Generate 1000 random values from a specific BKw distribution
//' alpha_par <- 2.0
//' beta_par <- 1.5
//' gamma_par <- 1.0
//' delta_par <- 0.5
//'
//' x_sample_bkw <- rbkw(1000, alpha = alpha_par, beta = beta_par,
//'                      gamma = gamma_par, delta = delta_par)
//' summary(x_sample_bkw)
//'
//' # Histogram of generated values compared to theoretical density
//' hist(x_sample_bkw, breaks = 30, freq = FALSE, # freq=FALSE for density
//'      main = "Histogram of BKw Sample", xlab = "x", ylim = c(0, 2.5))
//' curve(dbkw(x, alpha = alpha_par, beta = beta_par, gamma = gamma_par,
//'            delta = delta_par),
//'       add = TRUE, col = "red", lwd = 2, n = 201)
//' legend("topright", legend = "Theoretical PDF", col = "red", lwd = 2, bty = "n")
//'
//' # Comparing empirical and theoretical quantiles (Q-Q plot)
//' prob_points <- seq(0.01, 0.99, by = 0.01)
//' theo_quantiles <- qbkw(prob_points, alpha = alpha_par, beta = beta_par,
//'                        gamma = gamma_par, delta = delta_par)
//' emp_quantiles <- quantile(x_sample_bkw, prob_points, type = 7)
//'
//' plot(theo_quantiles, emp_quantiles, pch = 16, cex = 0.8,
//'      main = "Q-Q Plot for BKw Distribution",
//'      xlab = "Theoretical Quantiles", ylab = "Empirical Quantiles (n=1000)")
//' abline(a = 0, b = 1, col = "blue", lty = 2)
//'
//' # Compare summary stats with rgkw(..., lambda=1, ...)
//' # Note: individual values will differ due to randomness
//' x_sample_gkw <- rgkw(1000, alpha = alpha_par, beta = beta_par, gamma = gamma_par,
//'                      delta = delta_par, lambda = 1.0)
//' print("Summary stats for rbkw sample:")
//' print(summary(x_sample_bkw))
//' print("Summary stats for rgkw(lambda=1) sample:")
//' print(summary(x_sample_gkw)) # Should be similar
//'
//' }
//'
//' @export
// [[Rcpp::export]]
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
 
 arma::vec alpha_vec(alpha.begin(), alpha.size());
 arma::vec beta_vec(beta.begin(), beta.size());
 arma::vec gamma_vec(gamma.begin(), gamma.size());
 arma::vec delta_vec(delta.begin(), delta.size());
 
 size_t k = std::max({alpha_vec.n_elem,
                     beta_vec.n_elem,
                     gamma_vec.n_elem,
                     delta_vec.n_elem});
 
 arma::vec out(n);
 
 for (int i = 0; i < n; ++i) {
   size_t idx = i % k;
   double a = alpha_vec[idx % alpha_vec.n_elem];
   double b = beta_vec[idx % beta_vec.n_elem];
   double g = gamma_vec[idx % gamma_vec.n_elem];
   double d = delta_vec[idx % delta_vec.n_elem];
   
   if (!check_bkw_pars(a, b, g, d)) {
     out(i) = NA_REAL;
     Rcpp::warning("rbkw: invalid parameters at index %d", i+1);
     continue;
   }
   
   // V ~ Beta(g, d+1)
   double V = R::rbeta(g, d + 1.0);
   // X = {1 - (1 - V)^(1/b)}^(1/a)
   double one_minus_V = 1.0 - V;
   if (one_minus_V <= 0.0) {
     out(i) = 1.0;
     continue;
   }
   if (one_minus_V >= 1.0) {
     out(i) = 0.0;
     continue;
   }
   
   double temp = std::pow(one_minus_V, 1.0/b);
   double xval = 1.0 - temp;
   if (xval < 0.0)  xval = 0.0;
   if (xval > 1.0)  xval = 1.0;
   
   if (a == 1.0) {
     out(i) = xval;
   } else {
     double rv = std::pow(xval, 1.0/a);
     if (rv < 0.0) rv = 0.0;
     if (rv > 1.0) rv = 1.0;
     out(i) = rv;
   }
 }
 
 return Rcpp::NumericVector(out.memptr(), out.memptr() + out.n_elem);
}


//' @title Negative Log-Likelihood for Beta-Kumaraswamy (BKw) Distribution
//' @author Lopes, J. E.
//' @keywords distribution likelihood optimize
//'
//' @description
//' Computes the negative log-likelihood function for the Beta-Kumaraswamy (BKw)
//' distribution with parameters \code{alpha} (\eqn{\alpha}), \code{beta}
//' (\eqn{\beta}), \code{gamma} (\eqn{\gamma}), and \code{delta} (\eqn{\delta}),
//' given a vector of observations. This distribution is the special case of the
//' Generalized Kumaraswamy (GKw) distribution where \eqn{\lambda = 1}. This function
//' is typically used for maximum likelihood estimation via numerical optimization.
//'
//' @param par A numeric vector of length 4 containing the distribution parameters
//'   in the order: \code{alpha} (\eqn{\alpha > 0}), \code{beta} (\eqn{\beta > 0}),
//'   \code{gamma} (\eqn{\gamma > 0}), \code{delta} (\eqn{\delta \ge 0}).
//' @param data A numeric vector of observations. All values must be strictly
//'   between 0 and 1 (exclusive).
//'
//' @return Returns a single \code{double} value representing the negative
//'   log-likelihood (\eqn{-\ell(\theta|\mathbf{x})}). Returns \code{Inf}
//'   if any parameter values in \code{par} are invalid according to their
//'   constraints, or if any value in \code{data} is not in the interval (0, 1).
//'
//' @details
//' The Beta-Kumaraswamy (BKw) distribution is the GKw distribution (\code{\link{dgkw}})
//' with \eqn{\lambda=1}. Its probability density function (PDF) is:
//' \deqn{
//' f(x | \theta) = \frac{\alpha \beta}{B(\gamma, \delta+1)} x^{\alpha - 1} \bigl(1 - x^\alpha\bigr)^{\beta(\delta+1) - 1} \bigl[1 - \bigl(1 - x^\alpha\bigr)^\beta\bigr]^{\gamma - 1}
//' }
//' for \eqn{0 < x < 1}, \eqn{\theta = (\alpha, \beta, \gamma, \delta)}, and \eqn{B(a,b)}
//' is the Beta function (\code{\link[base]{beta}}).
//' The log-likelihood function \eqn{\ell(\theta | \mathbf{x})} for a sample
//' \eqn{\mathbf{x} = (x_1, \dots, x_n)} is \eqn{\sum_{i=1}^n \ln f(x_i | \theta)}:
//' \deqn{
//' \ell(\theta | \mathbf{x}) = n[\ln(\alpha) + \ln(\beta) - \ln B(\gamma, \delta+1)]
//' + \sum_{i=1}^{n} [(\alpha-1)\ln(x_i) + (\beta(\delta+1)-1)\ln(v_i) + (\gamma-1)\ln(w_i)]
//' }
//' where:
//' \itemize{
//'   \item \eqn{v_i = 1 - x_i^{\alpha}}
//'   \item \eqn{w_i = 1 - v_i^{\beta} = 1 - (1-x_i^{\alpha})^{\beta}}
//' }
//' This function computes and returns the *negative* log-likelihood, \eqn{-\ell(\theta|\mathbf{x})},
//' suitable for minimization using optimization routines like \code{\link[stats]{optim}}.
//' Numerical stability is maintained similarly to \code{\link{llgkw}}.
//'
//' @references
//' Cordeiro, G. M., & de Castro, M. (2011). A new family of generalized
//' distributions. *Journal of Statistical Computation and Simulation*
//'
//' Kumaraswamy, P. (1980). A generalized probability density function for
//' double-bounded random processes. *Journal of Hydrology*, *46*(1-2), 79-88.
//'
//'
//' @seealso
//' \code{\link{llgkw}} (parent distribution negative log-likelihood),
//' \code{\link{dbkw}}, \code{\link{pbkw}}, \code{\link{qbkw}}, \code{\link{rbkw}},
//' \code{grbkw} (gradient, if available),
//' \code{hsbkw} (Hessian, if available),
//' \code{\link[stats]{optim}}, \code{\link[base]{lbeta}}
//'
//' @examples
//' \donttest{
//'
//' # Generate sample data from a known BKw distribution
//' set.seed(2203)
//' true_par_bkw <- c(alpha = 2.0, beta = 1.5, gamma = 1.5, delta = 0.5)
//' sample_data_bkw <- rbkw(1000, alpha = true_par_bkw[1], beta = true_par_bkw[2],
//'                          gamma = true_par_bkw[3], delta = true_par_bkw[4])
//' hist(sample_data_bkw, breaks = 20, main = "BKw(2, 1.5, 1.5, 0.5) Sample")
//'
//' # --- Maximum Likelihood Estimation using optim ---
//' # Initial parameter guess
//' start_par_bkw <- c(1.8, 1.2, 1.1, 0.3)
//'
//' # Perform optimization (minimizing negative log-likelihood)
//' mle_result_bkw <- stats::optim(par = start_par_bkw,
//'                                fn = llbkw, # Use the BKw neg-log-likelihood
//'                                method = "BFGS", # Needs parameters > 0, consider L-BFGS-B
//'                                hessian = TRUE,
//'                                data = sample_data_bkw)
//'
//' # Check convergence and results
//' if (mle_result_bkw$convergence == 0) {
//'   print("Optimization converged successfully.")
//'   mle_par_bkw <- mle_result_bkw$par
//'   print("Estimated BKw parameters:")
//'   print(mle_par_bkw)
//'   print("True BKw parameters:")
//'   print(true_par_bkw)
//' } else {
//'   warning("Optimization did not converge!")
//'   print(mle_result_bkw$message)
//' }
//'
//' # --- Compare numerical and analytical derivatives (if available) ---
//' # Requires 'numDeriv' package and analytical functions 'grbkw', 'hsbkw'
//' if (mle_result_bkw$convergence == 0 &&
//'     requireNamespace("numDeriv", quietly = TRUE) &&
//'     exists("grbkw") && exists("hsbkw")) {
//'
//'   cat("\nComparing Derivatives at BKw MLE estimates:\n")
//'
//'   # Numerical derivatives of llbkw
//'   num_grad_bkw <- numDeriv::grad(func = llbkw, x = mle_par_bkw, data = sample_data_bkw)
//'   num_hess_bkw <- numDeriv::hessian(func = llbkw, x = mle_par_bkw, data = sample_data_bkw)
//'
//'   # Analytical derivatives (assuming they return derivatives of negative LL)
//'   ana_grad_bkw <- grbkw(par = mle_par_bkw, data = sample_data_bkw)
//'   ana_hess_bkw <- hsbkw(par = mle_par_bkw, data = sample_data_bkw)
//'
//'   # Check differences
//'   cat("Max absolute difference between gradients:\n")
//'   print(max(abs(num_grad_bkw - ana_grad_bkw)))
//'   cat("Max absolute difference between Hessians:\n")
//'   print(max(abs(num_hess_bkw - ana_hess_bkw)))
//'
//' } else {
//'    cat("\nSkipping derivative comparison for BKw.\n")
//'    cat("Requires convergence, 'numDeriv' package and functions 'grbkw', 'hsbkw'.\n")
//' }
//'
//' }
//'
//' @export
// [[Rcpp::export]]
double llbkw(const Rcpp::NumericVector& par,
            const Rcpp::NumericVector& data) {
 // Parameter validation
 if (par.size() < 4) {
   return R_PosInf;
 }
 
 double a = par[0];  // alpha > 0
 double b = par[1];  // beta > 0
 double g = par[2];  // gamma > 0
 double d = par[3];  // delta >= 0
 
 // Basic parameter validation
 if (a <= 0.0 || b <= 0.0 || g <= 0.0 || d < 0.0) {
   return R_PosInf;
 }
 
 // Convert data to armadillo vector
 arma::vec x = Rcpp::as<arma::vec>(data);
 int n = x.n_elem;
 
 // Basic data validation
 if (n == 0 || arma::any(x <= 0.0) || arma::any(x >= 1.0)) {
   return R_PosInf;
 }
 
 // ----- Compute log-likelihood with careful numerical handling -----
 
 // Compute log-beta term
 double logB = R::lbeta(g, d + 1.0);
 
 // Constant part: n * (log(a) + log(b) - logB)
 double ll_const = n * (std::log(a) + std::log(b) - logB);
 
 // ----- Term 1: (alpha - 1) * sum(log(x)) -----
 arma::vec lx = arma::log(x);
 double sum1 = (a - 1.0) * arma::sum(lx);
 
 // ----- Term 2: (beta*(delta+1) - 1) * sum(log(1 - x^alpha)) -----
 double exp1 = b * (d + 1.0) - 1.0;
 double sum2 = 0.0;
 
 // ----- Term 3: (gamma - 1) * sum(log(1 - (1 - x^alpha)^beta)) -----
 double sum3 = 0.0;
 
 // Small constant for numerical stability
 double eps = std::sqrt(std::numeric_limits<double>::epsilon());
 
 // Process each observation for terms 2 and 3
 for (int i = 0; i < n; i++) {
   double xi = x(i);
   
   // Compute x^alpha (more accurately in log domain for extreme values)
   double xa;
   if (a * std::log(xi) < -700.0) {
     xa = 0.0;  // Underflow protection
   } else if (a * std::log(xi) > 700.0) {
     xa = 1e300;  // Overflow protection - will lead to v ≈ 0
   } else {
     xa = std::pow(xi, a);
   }
   
   // Compute v = 1 - x^alpha (more accurately for x^alpha close to 1)
   double v;
   if (xa > 0.5) {
     v = std::max(1.0 - xa, eps);  // Ensure v > 0
   } else {
     v = 1.0 - xa;
   }
   
   // Restrict v to valid range for numerical stability
   v = std::max(std::min(v, 1.0 - eps), eps);
   
   // Term 2 contribution: (beta*(delta+1) - 1) * log(v)
   sum2 += exp1 * std::log(v);
   
   // Compute v^beta (more accurately in log domain for extreme values)
   double vb;
   if (b * std::log(v) < -700.0) {
     vb = 0.0;  // Underflow protection
   } else if (b * std::log(v) > 700.0) {
     vb = 1e300;  // Overflow protection - will lead to w ≈ 0
   } else {
     vb = std::pow(v, b);
   }
   
   // Compute w = 1 - v^beta (more accurately for v^beta close to 1)
   double w;
   if (vb > 0.5) {
     w = std::max(1.0 - vb, eps);  // Ensure w > 0
   } else {
     w = 1.0 - vb;
   }
   
   // Restrict w to valid range for numerical stability
   w = std::max(std::min(w, 1.0 - eps), eps);
   
   // Term 3 contribution: (gamma - 1) * log(w)
   if (g != 1.0) {  // Skip if gamma = 1
     sum3 += (g - 1.0) * std::log(w);
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






//' @title Gradient of the Negative Log-Likelihood for the BKw Distribution
//' @author Lopes, J. E.
//' @keywords distribution likelihood optimize gradient
//'
//' @description
//' Computes the gradient vector (vector of first partial derivatives) of the
//' negative log-likelihood function for the Beta-Kumaraswamy (BKw) distribution
//' with parameters \code{alpha} (\eqn{\alpha}), \code{beta} (\eqn{\beta}),
//' \code{gamma} (\eqn{\gamma}), and \code{delta} (\eqn{\delta}). This distribution
//' is the special case of the Generalized Kumaraswamy (GKw) distribution where
//' \eqn{\lambda = 1}. The gradient is typically used in optimization algorithms
//' for maximum likelihood estimation.
//'
//' @param par A numeric vector of length 4 containing the distribution parameters
//'   in the order: \code{alpha} (\eqn{\alpha > 0}), \code{beta} (\eqn{\beta > 0}),
//'   \code{gamma} (\eqn{\gamma > 0}), \code{delta} (\eqn{\delta \ge 0}).
//' @param data A numeric vector of observations. All values must be strictly
//'   between 0 and 1 (exclusive).
//'
//' @return Returns a numeric vector of length 4 containing the partial derivatives
//'   of the negative log-likelihood function \eqn{-\ell(\theta | \mathbf{x})} with
//'   respect to each parameter:
//'   \eqn{(-\partial \ell/\partial \alpha, -\partial \ell/\partial \beta, -\partial \ell/\partial \gamma, -\partial \ell/\partial \delta)}.
//'   Returns a vector of \code{NaN} if any parameter values are invalid according
//'   to their constraints, or if any value in \code{data} is not in the
//'   interval (0, 1).
//'
//' @details
//' The components of the gradient vector of the negative log-likelihood
//' (\eqn{-\nabla \ell(\theta | \mathbf{x})}) for the BKw (\eqn{\lambda=1}) model are:
//'
//' \deqn{
//' -\frac{\partial \ell}{\partial \alpha} = -\frac{n}{\alpha} - \sum_{i=1}^{n}\ln(x_i)
//' + \sum_{i=1}^{n}\left[x_i^{\alpha} \ln(x_i) \left(\frac{\beta(\delta+1)-1}{v_i} -
//' \frac{(\gamma-1) \beta v_i^{\beta-1}}{w_i}\right)\right]
//' }
//' \deqn{
//' -\frac{\partial \ell}{\partial \beta} = -\frac{n}{\beta} - (\delta+1)\sum_{i=1}^{n}\ln(v_i)
//' + \sum_{i=1}^{n}\left[\frac{(\gamma-1) v_i^{\beta} \ln(v_i)}{w_i}\right]
//' }
//' \deqn{
//' -\frac{\partial \ell}{\partial \gamma} = n[\psi(\gamma) - \psi(\gamma+\delta+1)] -
//' \sum_{i=1}^{n}\ln(w_i)
//' }
//' \deqn{
//' -\frac{\partial \ell}{\partial \delta} = n[\psi(\delta+1) - \psi(\gamma+\delta+1)] -
//' \beta\sum_{i=1}^{n}\ln(v_i)
//' }
//'
//' where:
//' \itemize{
//'   \item \eqn{v_i = 1 - x_i^{\alpha}}
//'   \item \eqn{w_i = 1 - v_i^{\beta} = 1 - (1-x_i^{\alpha})^{\beta}}
//'   \item \eqn{\psi(\cdot)} is the digamma function (\code{\link[base]{digamma}}).
//' }
//' These formulas represent the derivatives of \eqn{-\ell(\theta)}, consistent with
//' minimizing the negative log-likelihood. They correspond to the general GKw
//' gradient (\code{\link{grgkw}}) components for \eqn{\alpha, \beta, \gamma, \delta}
//' evaluated at \eqn{\lambda=1}. Note that the component for \eqn{\lambda} is omitted.
//' Numerical stability is maintained through careful implementation.
//'
//' @references
//' Cordeiro, G. M., & de Castro, M. (2011). A new family of generalized
//' distributions. *Journal of Statistical Computation and Simulation*,
//'
//' Kumaraswamy, P. (1980). A generalized probability density function for
//' double-bounded random processes. *Journal of Hydrology*, *46*(1-2), 79-88.
//'
//' (Note: Specific gradient formulas might be derived or sourced from additional references).
//'
//' @seealso
//' \code{\link{grgkw}} (parent distribution gradient),
//' \code{\link{llbkw}} (negative log-likelihood for BKw),
//' \code{\link{hsbkw}} (Hessian for BKw, if available),
//' \code{\link{dbkw}} (density for BKw),
//' \code{\link[stats]{optim}},
//' \code{\link[numDeriv]{grad}} (for numerical gradient comparison),
//' \code{\link[base]{digamma}}.
//'
//' @examples
//' \donttest{
//' # Assuming existence of rbkw, llbkw, grbkw, hsbkw functions for BKw
//'
//' # Generate sample data
//' set.seed(123)
//' true_par_bkw <- c(alpha = 2, beta = 3, gamma = 1, delta = 0.5)
//' if (exists("rbkw")) {
//'   sample_data_bkw <- rbkw(100, alpha = true_par_bkw[1], beta = true_par_bkw[2],
//'                          gamma = true_par_bkw[3], delta = true_par_bkw[4])
//' } else {
//'   sample_data_bkw <- rgkw(100, alpha = true_par_bkw[1], beta = true_par_bkw[2],
//'                          gamma = true_par_bkw[3], delta = true_par_bkw[4], lambda = 1)
//' }
//' hist(sample_data_bkw, breaks = 20, main = "BKw(2, 3, 1, 0.5) Sample")
//'
//' # --- Find MLE estimates ---
//' start_par_bkw <- c(1.5, 2.5, 0.8, 0.3)
//' mle_result_bkw <- stats::optim(par = start_par_bkw,
//'                                fn = llbkw,
//'                                gr = grbkw, # Use analytical gradient for BKw
//'                                method = "BFGS",
//'                                hessian = TRUE,
//'                                data = sample_data_bkw)
//'
//' # --- Compare analytical gradient to numerical gradient ---
//' if (mle_result_bkw$convergence == 0 &&
//'     requireNamespace("numDeriv", quietly = TRUE)) {
//'
//'   mle_par_bkw <- mle_result_bkw$par
//'   cat("\nComparing Gradients for BKw at MLE estimates:\n")
//'
//'   # Numerical gradient of llbkw
//'   num_grad_bkw <- numDeriv::grad(func = llbkw, x = mle_par_bkw, data = sample_data_bkw)
//'
//'   # Analytical gradient from grbkw
//'   ana_grad_bkw <- grbkw(par = mle_par_bkw, data = sample_data_bkw)
//'
//'   cat("Numerical Gradient (BKw):\n")
//'   print(num_grad_bkw)
//'   cat("Analytical Gradient (BKw):\n")
//'   print(ana_grad_bkw)
//'
//'   # Check differences
//'   cat("Max absolute difference between BKw gradients:\n")
//'   print(max(abs(num_grad_bkw - ana_grad_bkw)))
//'
//' } else {
//'   cat("\nSkipping BKw gradient comparison.\n")
//' }
//'
//' # --- Optional: Compare with relevant components of GKw gradient ---
//' # Requires grgkw function
//' if (mle_result_bkw$convergence == 0 && exists("grgkw")) {
//'   # Create 5-param vector for grgkw (insert lambda=1)
//'   mle_par_gkw_equiv <- c(mle_par_bkw[1:4], lambda = 1.0)
//'   ana_grad_gkw <- grgkw(par = mle_par_gkw_equiv, data = sample_data_bkw)
//'   # Extract components corresponding to alpha, beta, gamma, delta
//'   ana_grad_gkw_subset <- ana_grad_gkw[c(1, 2, 3, 4)]
//'
//'   cat("\nComparison with relevant components of GKw gradient:\n")
//'   cat("Max absolute difference:\n")
//'   print(max(abs(ana_grad_bkw - ana_grad_gkw_subset))) # Should be very small
//' }
//'
//' }
//'
//' @export
//' @title Gradient of the Negative Log-Likelihood for the BKw Distribution
//' @author Lopes, J. E.
//' @keywords distribution likelihood optimize gradient
//'
//' @description
//' Computes the gradient vector (vector of first partial derivatives) of the
//' negative log-likelihood function for the Beta-Kumaraswamy (BKw) distribution
//' with parameters \code{alpha} (\eqn{\alpha}), \code{beta} (\eqn{\beta}),
//' \code{gamma} (\eqn{\gamma}), and \code{delta} (\eqn{\delta}). This distribution
//' is the special case of the Generalized Kumaraswamy (GKw) distribution where
//' \eqn{\lambda = 1}. The gradient is typically used in optimization algorithms
//' for maximum likelihood estimation.
//'
//' @param par A numeric vector of length 4 containing the distribution parameters
//'   in the order: \code{alpha} (\eqn{\alpha > 0}), \code{beta} (\eqn{\beta > 0}),
//'   \code{gamma} (\eqn{\gamma > 0}), \code{delta} (\eqn{\delta \ge 0}).
//' @param data A numeric vector of observations. All values must be strictly
//'   between 0 and 1 (exclusive).
//'
//' @return Returns a numeric vector of length 4 containing the partial derivatives
//'   of the negative log-likelihood function \eqn{-\ell(\theta | \mathbf{x})} with
//'   respect to each parameter:
//'   \eqn{(-\partial \ell/\partial \alpha, -\partial \ell/\partial \beta, -\partial \ell/\partial \gamma, -\partial \ell/\partial \delta)}.
//'   Returns a vector of \code{NaN} if any parameter values are invalid according
//'   to their constraints, or if any value in \code{data} is not in the
//'   interval (0, 1).
//'
//' @details
//' The components of the gradient vector of the negative log-likelihood
//' (\eqn{-\nabla \ell(\theta | \mathbf{x})}) for the BKw (\eqn{\lambda=1}) model are:
//'
//' \deqn{
//' -\frac{\partial \ell}{\partial \alpha} = -\frac{n}{\alpha} - \sum_{i=1}^{n}\ln(x_i)
//' + \sum_{i=1}^{n}\left[x_i^{\alpha} \ln(x_i) \left(\frac{\beta(\delta+1)-1}{v_i} -
//' \frac{(\gamma-1) \beta v_i^{\beta-1}}{w_i}\right)\right]
//' }
//' \deqn{
//' -\frac{\partial \ell}{\partial \beta} = -\frac{n}{\beta} - (\delta+1)\sum_{i=1}^{n}\ln(v_i)
//' + \sum_{i=1}^{n}\left[\frac{(\gamma-1) v_i^{\beta} \ln(v_i)}{w_i}\right]
//' }
//' \deqn{
//' -\frac{\partial \ell}{\partial \gamma} = n[\psi(\gamma) - \psi(\gamma+\delta+1)] -
//' \sum_{i=1}^{n}\ln(w_i)
//' }
//' \deqn{
//' -\frac{\partial \ell}{\partial \delta} = n[\psi(\delta+1) - \psi(\gamma+\delta+1)] -
//' \beta\sum_{i=1}^{n}\ln(v_i)
//' }
//'
//' where:
//' \itemize{
//'   \item \eqn{v_i = 1 - x_i^{\alpha}}
//'   \item \eqn{w_i = 1 - v_i^{\beta} = 1 - (1-x_i^{\alpha})^{\beta}}
//'   \item \eqn{\psi(\cdot)} is the digamma function (\code{\link[base]{digamma}}).
//' }
//' These formulas represent the derivatives of \eqn{-\ell(\theta)}, consistent with
//' minimizing the negative log-likelihood. They correspond to the general GKw
//' gradient (\code{\link{grgkw}}) components for \eqn{\alpha, \beta, \gamma, \delta}
//' evaluated at \eqn{\lambda=1}. Note that the component for \eqn{\lambda} is omitted.
//' Numerical stability is maintained through careful implementation.
//'
//' @references
//' Cordeiro, G. M., & de Castro, M. (2011). A new family of generalized
//' distributions. *Journal of Statistical Computation and Simulation*,
//'
//' Kumaraswamy, P. (1980). A generalized probability density function for
//' double-bounded random processes. *Journal of Hydrology*, *46*(1-2), 79-88.
//'
//' (Note: Specific gradient formulas might be derived or sourced from additional references).
//'
//' @seealso
//' \code{\link{grgkw}} (parent distribution gradient),
//' \code{\link{llbkw}} (negative log-likelihood for BKw),
//' \code{\link{hsbkw}} (Hessian for BKw, if available),
//' \code{\link{dbkw}} (density for BKw),
//' \code{\link[stats]{optim}},
//' \code{\link[numDeriv]{grad}} (for numerical gradient comparison),
//' \code{\link[base]{digamma}}.
//'
//' @examples
//' \donttest{
//' # Assuming existence of rbkw, llbkw, grbkw, hsbkw functions for BKw
//'
//' # Generate sample data
//' set.seed(123)
//' true_par_bkw <- c(alpha = 2, beta = 3, gamma = 1, delta = 0.5)
//' if (exists("rbkw")) {
//'   sample_data_bkw <- rbkw(100, alpha = true_par_bkw[1], beta = true_par_bkw[2],
//'                          gamma = true_par_bkw[3], delta = true_par_bkw[4])
//' } else {
//'   sample_data_bkw <- rgkw(100, alpha = true_par_bkw[1], beta = true_par_bkw[2],
//'                          gamma = true_par_bkw[3], delta = true_par_bkw[4], lambda = 1)
//' }
//' hist(sample_data_bkw, breaks = 20, main = "BKw(2, 3, 1, 0.5) Sample")
//'
//' # --- Find MLE estimates ---
//' start_par_bkw <- c(1.5, 2.5, 0.8, 0.3)
//' mle_result_bkw <- stats::optim(par = start_par_bkw,
//'                                fn = llbkw,
//'                                gr = grbkw, # Use analytical gradient for BKw
//'                                method = "BFGS",
//'                                hessian = TRUE,
//'                                data = sample_data_bkw)
//'
//' # --- Compare analytical gradient to numerical gradient ---
//' if (mle_result_bkw$convergence == 0 &&
//'     requireNamespace("numDeriv", quietly = TRUE)) {
//'
//'   mle_par_bkw <- mle_result_bkw$par
//'   cat("\nComparing Gradients for BKw at MLE estimates:\n")
//'
//'   # Numerical gradient of llbkw
//'   num_grad_bkw <- numDeriv::grad(func = llbkw, x = mle_par_bkw, data = sample_data_bkw)
//'
//'   # Analytical gradient from grbkw
//'   ana_grad_bkw <- grbkw(par = mle_par_bkw, data = sample_data_bkw)
//'
//'   cat("Numerical Gradient (BKw):\n")
//'   print(num_grad_bkw)
//'   cat("Analytical Gradient (BKw):\n")
//'   print(ana_grad_bkw)
//'
//'   # Check differences
//'   cat("Max absolute difference between BKw gradients:\n")
//'   print(max(abs(num_grad_bkw - ana_grad_bkw)))
//'
//' } else {
//'   cat("\nSkipping BKw gradient comparison.\n")
//' }
//'
//' # --- Optional: Compare with relevant components of GKw gradient ---
//' # Requires grgkw function
//' if (mle_result_bkw$convergence == 0 && exists("grgkw")) {
//'   # Create 5-param vector for grgkw (insert lambda=1)
//'   mle_par_gkw_equiv <- c(mle_par_bkw[1:4], lambda = 1.0)
//'   ana_grad_gkw <- grgkw(par = mle_par_gkw_equiv, data = sample_data_bkw)
//'   # Extract components corresponding to alpha, beta, gamma, delta
//'   ana_grad_gkw_subset <- ana_grad_gkw[c(1, 2, 3, 4)]
//'
//'   cat("\nComparison with relevant components of GKw gradient:\n")
//'   cat("Max absolute difference:\n")
//'   print(max(abs(ana_grad_bkw - ana_grad_gkw_subset))) # Should be very small
//' }
//'
//' }
//'
//' @export
// [[Rcpp::export]]
Rcpp::NumericVector grbkw(const Rcpp::NumericVector& par, const Rcpp::NumericVector& data) {
 // Initialize gradient vector for early return cases
 Rcpp::NumericVector grad(4, R_NaN);
 
 // Parameter extraction and validation
 if (par.size() < 4) {
   Rcpp::warning("Parameter vector must have at least 4 elements for BKw");
   return grad;
 }
 
 double alpha = par[0];   // Shape parameter α > 0
 double beta = par[1];    // Shape parameter β > 0
 double gamma = par[2];   // Shape parameter γ > 0
 double delta = par[3];   // Shape parameter δ ≥ 0
 
 // Enhanced parameter validation with NaN checks
 if (std::isnan(alpha) || std::isnan(beta) || std::isnan(gamma) || std::isnan(delta) ||
     alpha <= 0 || beta <= 0 || gamma <= 0 || delta < 0) {
   Rcpp::warning("Invalid parameters in grbkw: all must be positive (delta can be zero)");
   return grad;
 }
 
 // Data conversion and validation
 arma::vec x;
 try {
   x = Rcpp::as<arma::vec>(data);
 } catch (...) {
   Rcpp::warning("Failed to convert data to arma::vec in grbkw");
   return grad;
 }
 
 // Comprehensive data validation
 if (x.n_elem == 0 || x.has_nan() || arma::any(x <= 0) || arma::any(x >= 1)) {
   Rcpp::warning("Data must be strictly in (0,1) and non-empty for grbkw");
   return grad;
 }
 
 int n = x.n_elem;  // Sample size
 
 // Reset gradient to zeros (we'll compute actual values now)
 grad = Rcpp::NumericVector(4, 0.0);
 
 // Small constant for numerical stability - adaptive to machine precision
 double eps = std::sqrt(std::numeric_limits<double>::epsilon());
 
 // -------- Step 1: Compute base transformations safely --------
 
 // Compute log(x) safely
 arma::vec log_x = vec_safe_log(x);
 
 // Compute x^α safely
 arma::vec x_alpha = vec_safe_pow(x, alpha);
 
 // Compute x^α * log(x) with check for potential overflow
 arma::vec x_alpha_log_x = x_alpha % log_x;
 
 // ----- Step 2: Compute v_i = 1 - x_i^α and related terms -----
 
 // Use log1p and expm1 for better precision near boundaries
 arma::vec v(n);
 for (int i = 0; i < n; i++) {
   // v_i = 1 - x_i^α computed via v_i = -expm1(log(x_i^α)) for better precision
   if (x_alpha(i) < 0.5) {
     // Standard calculation is fine for smaller values
     v(i) = 1.0 - x_alpha(i);
   } else {
     // For x_i^α close to 1, use more accurate formula
     v(i) = -std::expm1(alpha * log_x(i));
   }
   
   // Ensure v is in valid range
   if (v(i) <= 0.0 || v(i) >= 1.0) {
     // Apply very cautious clamping only at extremes
     v(i) = std::max(std::min(v(i), 1.0 - eps), eps);
   }
 }
 
 // Compute log(v) and v^β terms with proper safeguards
 arma::vec log_v = vec_safe_log(v);
 
 // Compute v^(β-1) with safety for β close to 1
 arma::vec v_beta_m1;
 if (std::abs(beta - 1.0) < eps) {
   // For β ≈ 1, v^(β-1) ≈ 1
   v_beta_m1.ones(n);
 } else {
   v_beta_m1 = vec_safe_pow(v, beta - 1.0);
 }
 
 // Compute v^β safely
 arma::vec v_beta = v % v_beta_m1;  // More accurate than direct power
 
 // Compute v^β * log(v) with check for potential issues
 arma::vec v_beta_log_v = v_beta % log_v;
 
 // ----- Step 3: Compute w_i = 1 - v_i^β safely -----
 
 arma::vec w(n);
 arma::vec log_w(n);
 
 for (int i = 0; i < n; i++) {
   // Compute w_i = 1 - v_i^β more accurately for v_i^β close to 1
   if (v_beta(i) < 0.5) {
     w(i) = 1.0 - v_beta(i);
   } else {
     // For v_i^β close to 1, use log-domain calculation
     w(i) = -std::expm1(beta * log_v(i));
   }
   
   // Validate and apply safety bounds
   if (w(i) <= 0.0 || w(i) >= 1.0) {
     w(i) = std::max(std::min(w(i), 1.0 - eps), eps);
   }
   
   // Compute log(w) directly for better accuracy
   log_w(i) = std::log(w(i));
 }
 
 // ----- Step 4: Calculate partial derivatives with careful factoring -----
 
 // Compute digamma values once, with checks for large arguments
 double digamma_gamma, digamma_delta_plus_1, digamma_sum;
 
 if (gamma > 1e6 && delta > 1e6) {
   // For very large parameters, use asymptotic approximation of digamma
   digamma_gamma = std::log(gamma) - 1.0/(2.0*gamma);
   digamma_delta_plus_1 = std::log(delta+1.0) - 1.0/(2.0*(delta+1.0));
   digamma_sum = std::log(gamma+delta+1.0) - 1.0/(2.0*(gamma+delta+1.0));
 } else {
   // Use standard digamma for typical values
   digamma_gamma = R::digamma(gamma);
   digamma_delta_plus_1 = R::digamma(delta + 1.0);
   digamma_sum = R::digamma(gamma + delta + 1.0);
 }
 
 // ----- Calculate gradient components -----
 
 // d_alpha = n/α + Σᵢlog(xᵢ) - Σᵢ[xᵢ^α * log(xᵢ) * ((β(δ+1)-1)/vᵢ - (γ-1) * β * vᵢ^(β-1) / wᵢ)]
 double term_beta_delta = beta * (delta + 1.0) - 1.0;
 double term_gamma = gamma - 1.0;
 
 double d_alpha = n / alpha + arma::sum(log_x);
 
 for (int i = 0; i < n; i++) {
   double alpha_term = x_alpha_log_x(i) * (
     term_beta_delta / v(i) -
       term_gamma * beta * v_beta_m1(i) / w(i)
   );
   
   // Check for invalid values before adding
   if (std::isfinite(alpha_term)) {
     d_alpha -= alpha_term;
   }
 }
 
 // d_beta = n/β + (δ+1)Σᵢlog(vᵢ) - Σᵢ[(γ-1) * vᵢ^β * log(vᵢ) / wᵢ]
 double d_beta = n / beta + (delta + 1.0) * arma::sum(log_v);
 
 if (term_gamma != 0.0) {  // Skip calculation if γ=1 (term_gamma=0)
   for (int i = 0; i < n; i++) {
     double beta_term = term_gamma * v_beta_log_v(i) / w(i);
     
     // Check for invalid values before adding
     if (std::isfinite(beta_term)) {
       d_beta -= beta_term;
     }
   }
 }
 
 // d_gamma = -n[ψ(γ) - ψ(γ+δ+1)] + Σᵢlog(wᵢ)
 double d_gamma = -n * (digamma_gamma - digamma_sum) + arma::sum(log_w);
 
 // d_delta = -n[ψ(δ+1) - ψ(γ+δ+1)] + βΣᵢlog(vᵢ)
 double d_delta = -n * (digamma_delta_plus_1 - digamma_sum) + beta * arma::sum(log_v);
 
 // Final check for NaN/Inf values
 if (!std::isfinite(d_alpha) || !std::isfinite(d_beta) ||
     !std::isfinite(d_gamma) || !std::isfinite(d_delta)) {
     Rcpp::warning("Gradient calculation produced non-finite values in grbkw");
   return Rcpp::NumericVector(4, R_NaN);
 }
 
 // Since we're optimizing negative log-likelihood, negate all derivatives
 grad[0] = -d_alpha;
 grad[1] = -d_beta;
 grad[2] = -d_gamma;
 grad[3] = -d_delta;
 
 return grad;
}





//' @title Hessian Matrix of the Negative Log-Likelihood for the BKw Distribution
//' @author Lopes, J. E.
//' @keywords distribution likelihood optimize hessian
//'
//' @description
//' Computes the analytic 4x4 Hessian matrix (matrix of second partial derivatives)
//' of the negative log-likelihood function for the Beta-Kumaraswamy (BKw)
//' distribution with parameters \code{alpha} (\eqn{\alpha}), \code{beta}
//' (\eqn{\beta}), \code{gamma} (\eqn{\gamma}), and \code{delta} (\eqn{\delta}).
//' This distribution is the special case of the Generalized Kumaraswamy (GKw)
//' distribution where \eqn{\lambda = 1}. The Hessian is useful for estimating
//' standard errors and in optimization algorithms.
//'
//' @param par A numeric vector of length 4 containing the distribution parameters
//'   in the order: \code{alpha} (\eqn{\alpha > 0}), \code{beta} (\eqn{\beta > 0}),
//'   \code{gamma} (\eqn{\gamma > 0}), \code{delta} (\eqn{\delta \ge 0}).
//' @param data A numeric vector of observations. All values must be strictly
//'   between 0 and 1 (exclusive).
//'
//' @return Returns a 4x4 numeric matrix representing the Hessian matrix of the
//'   negative log-likelihood function, \eqn{-\partial^2 \ell / (\partial \theta_i \partial \theta_j)},
//'   where \eqn{\theta = (\alpha, \beta, \gamma, \delta)}.
//'   Returns a 4x4 matrix populated with \code{NaN} if any parameter values are
//'   invalid according to their constraints, or if any value in \code{data} is
//'   not in the interval (0, 1).
//'
//' @details
//' This function calculates the analytic second partial derivatives of the
//' negative log-likelihood function based on the BKw log-likelihood
//' (\eqn{\lambda=1} case of GKw, see \code{\link{llbkw}}):
//' \deqn{
//' \ell(\theta | \mathbf{x}) = n[\ln(\alpha) + \ln(\beta) - \ln B(\gamma, \delta+1)]
//' + \sum_{i=1}^{n} [(\alpha-1)\ln(x_i) + (\beta(\delta+1)-1)\ln(v_i) + (\gamma-1)\ln(w_i)]
//' }
//' where \eqn{\theta = (\alpha, \beta, \gamma, \delta)}, \eqn{B(a,b)}
//' is the Beta function (\code{\link[base]{beta}}), and intermediate terms are:
//' \itemize{
//'   \item \eqn{v_i = 1 - x_i^{\alpha}}
//'   \item \eqn{w_i = 1 - v_i^{\beta} = 1 - (1-x_i^{\alpha})^{\beta}}
//' }
//' The Hessian matrix returned contains the elements \eqn{- \frac{\partial^2 \ell(\theta | \mathbf{x})}{\partial \theta_i \partial \theta_j}}
//' for \eqn{\theta_i, \theta_j \in \{\alpha, \beta, \gamma, \delta\}}.
//'
//' Key properties of the returned matrix:
//' \itemize{
//'   \item Dimensions: 4x4.
//'   \item Symmetry: The matrix is symmetric.
//'   \item Ordering: Rows and columns correspond to the parameters in the order
//'     \eqn{\alpha, \beta, \gamma, \delta}.
//'   \item Content: Analytic second derivatives of the *negative* log-likelihood.
//' }
//' This corresponds to the relevant 4x4 submatrix of the 5x5 GKw Hessian (\code{\link{hsgkw}})
//' evaluated at \eqn{\lambda=1}. The exact analytical formulas are implemented directly.
//'
//' @references
//' Cordeiro, G. M., & de Castro, M. (2011). A new family of generalized
//' distributions. *Journal of Statistical Computation and Simulation*,
//'
//' Kumaraswamy, P. (1980). A generalized probability density function for
//' double-bounded random processes. *Journal of Hydrology*, *46*(1-2), 79-88.
//'
//' (Note: Specific Hessian formulas might be derived or sourced from additional references).
//'
//' @seealso
//' \code{\link{hsgkw}} (parent distribution Hessian),
//' \code{\link{llbkw}} (negative log-likelihood for BKw),
//' \code{\link{grbkw}} (gradient for BKw, if available),
//' \code{\link{dbkw}} (density for BKw),
//' \code{\link[stats]{optim}},
//' \code{\link[numDeriv]{hessian}} (for numerical Hessian comparison).
//'
//' @examples
//' \donttest{
//' # Assuming existence of rbkw, llbkw, grbkw, hsbkw functions for BKw
//'
//' # Generate sample data
//' set.seed(123)
//' true_par_bkw <- c(alpha = 2, beta = 3, gamma = 1, delta = 0.5)
//' if (exists("rbkw")) {
//'   sample_data_bkw <- rbkw(100, alpha = true_par_bkw[1], beta = true_par_bkw[2],
//'                          gamma = true_par_bkw[3], delta = true_par_bkw[4])
//' } else {
//'   sample_data_bkw <- rgkw(100, alpha = true_par_bkw[1], beta = true_par_bkw[2],
//'                          gamma = true_par_bkw[3], delta = true_par_bkw[4], lambda = 1)
//' }
//' hist(sample_data_bkw, breaks = 20, main = "BKw(2, 3, 1, 0.5) Sample")
//'
//' # --- Find MLE estimates ---
//' start_par_bkw <- c(1.5, 2.5, 0.8, 0.3)
//' mle_result_bkw <- stats::optim(par = start_par_bkw,
//'                                fn = llbkw,
//'                                gr = if (exists("grbkw")) grbkw else NULL,
//'                                method = "BFGS",
//'                                hessian = TRUE, # Ask optim for numerical Hessian
//'                                data = sample_data_bkw)
//'
//' # --- Compare analytical Hessian to numerical Hessian ---
//' if (mle_result_bkw$convergence == 0 &&
//'     requireNamespace("numDeriv", quietly = TRUE) &&
//'     exists("hsbkw")) {
//'
//'   mle_par_bkw <- mle_result_bkw$par
//'   cat("\nComparing Hessians for BKw at MLE estimates:\n")
//'
//'   # Numerical Hessian of llbkw
//'   num_hess_bkw <- numDeriv::hessian(func = llbkw, x = mle_par_bkw, data = sample_data_bkw)
//'
//'   # Analytical Hessian from hsbkw
//'   ana_hess_bkw <- hsbkw(par = mle_par_bkw, data = sample_data_bkw)
//'
//'   cat("Numerical Hessian (BKw):\n")
//'   print(round(num_hess_bkw, 4))
//'   cat("Analytical Hessian (BKw):\n")
//'   print(round(ana_hess_bkw, 4))
//'
//'   # Check differences
//'   cat("Max absolute difference between BKw Hessians:\n")
//'   print(max(abs(num_hess_bkw - ana_hess_bkw)))
//'
//'   # Optional: Use analytical Hessian for Standard Errors
//'   # tryCatch({
//'   #   cov_matrix_bkw <- solve(ana_hess_bkw)
//'   #   std_errors_bkw <- sqrt(diag(cov_matrix_bkw))
//'   #   cat("Std. Errors from Analytical BKw Hessian:\n")
//'   #   print(std_errors_bkw)
//'   # }, error = function(e) {
//'   #   warning("Could not invert analytical BKw Hessian: ", e$message)
//'   # })
//'
//' } else {
//'   cat("\nSkipping BKw Hessian comparison.\n")
//'   cat("Requires convergence, 'numDeriv' package, and function 'hsbkw'.\n")
//' }
//'
//' }
//'
//' @export
// [[Rcpp::export]]
Rcpp::NumericMatrix hsbkw(const Rcpp::NumericVector& par, const Rcpp::NumericVector& data) {
 // Initialize return matrix for error cases
 Rcpp::NumericMatrix nanH(4, 4);
 nanH.fill(R_NaN);
 
 // Parameter validation
 if (par.size() < 4) {
   Rcpp::warning("Parameter vector must have at least 4 elements for BKw");
   return nanH;
 }
 
 double alpha = par[0];   // Shape parameter α > 0
 double beta = par[1];    // Shape parameter β > 0
 double gamma = par[2];   // Shape parameter γ > 0
 double delta = par[3];   // Shape parameter δ ≥ 0
 
 // Enhanced parameter validation with NaN checks
 if (std::isnan(alpha) || std::isnan(beta) || std::isnan(gamma) || std::isnan(delta) ||
     alpha <= 0 || beta <= 0 || gamma <= 0 || delta < 0) {
   Rcpp::warning("Invalid parameters in hsbkw: alpha, beta, gamma must be positive; delta must be non-negative");
   return nanH;
 }
 
 // Data conversion and validation
 arma::vec x;
 try {
   x = Rcpp::as<arma::vec>(data);
 } catch (...) {
   Rcpp::warning("Failed to convert data to arma::vec in hsbkw");
   return nanH;
 }
 
 // Comprehensive data validation
 if (x.n_elem == 0 || x.has_nan() || arma::any(x <= 0) || arma::any(x >= 1)) {
   Rcpp::warning("Data must be strictly in (0,1) and non-empty for hsbkw");
   return nanH;
 }
 
 int n = x.n_elem;  // Sample size
 
 // Initialize Hessian matrix H (of the log-likelihood) as 4x4
 arma::mat H(4, 4, arma::fill::zeros);
 
 // Small constant for numerical stability
 double eps = std::sqrt(std::numeric_limits<double>::epsilon());
 
 // ---------------------------------------------------------------------
 // STEP 1: Add constant terms (that don't depend on individual data points)
 // ---------------------------------------------------------------------
 
 // Second derivative of n*ln(α) with respect to α: -n/α²
 H(0, 0) = -n / (alpha * alpha);
 
 // Second derivative of n*ln(β) with respect to β: -n/β²
 H(1, 1) = -n / (beta * beta);
 
 // Compute trigamma values with protection for large arguments
 double trigamma_gamma, trigamma_delta_plus_1, trigamma_sum;
 
 if (gamma > 1e6 || delta > 1e6) {
   // For very large parameters, use asymptotic approximation of trigamma
   // ψ₁(x) ≈ 1/x + 1/(2x²) + ...
   trigamma_gamma = 1.0/gamma + 1.0/(2.0*gamma*gamma);
   trigamma_delta_plus_1 = 1.0/(delta+1.0) + 1.0/(2.0*(delta+1.0)*(delta+1.0));
   trigamma_sum = 1.0/(gamma+delta+1.0) + 1.0/(2.0*(gamma+delta+1.0)*(gamma+delta+1.0));
 } else {
   // Use standard trigamma for typical values
   trigamma_gamma = R::trigamma(gamma);
   trigamma_delta_plus_1 = R::trigamma(delta + 1.0);
   trigamma_sum = R::trigamma(gamma + delta + 1.0);
 }
 
 // Second derivative of -n*ln[B(γ,δ+1)] with respect to γ: -n*[ψ₁(γ) - ψ₁(γ+δ+1)]
 H(2, 2) = -n * (trigamma_gamma - trigamma_sum);
 
 // Second derivative of -n*ln[B(γ,δ+1)] with respect to δ: -n*[ψ₁(δ+1) - ψ₁(γ+δ+1)]
 H(3, 3) = -n * (trigamma_delta_plus_1 - trigamma_sum);
 
 // Mixed derivative (γ,δ): n*ψ₁(γ+δ+1)
 H(2, 3) = n * trigamma_sum;
 H(3, 2) = H(2, 3);  // Symmetric matrix
 
 // ---------------------------------------------------------------------
 // STEP 2: Loop through observations to accumulate data-dependent terms
 // ---------------------------------------------------------------------
 
 // Precompute common factor
 double beta_delta_factor = beta * (delta + 1.0) - 1.0;
 double gamma_minus_1 = gamma - 1.0;
 
 for (int i = 0; i < n; i++) {
   double xi = x(i);
   
   // Compute log(x) safely
   double ln_xi = safe_log(xi);
   
   // ---- Compute x^α and its derivatives more safely ----
   double A; // A = x^α
   double dA_dalpha; // dA/dα = x^α * ln(x)
   double d2A_dalpha2; // d²A/dα² = x^α * (ln(x))²
   
   // Use logarithmic domain for more stability
   double log_A = alpha * ln_xi;
   
   if (std::abs(log_A) > 700.0) {
     // For extreme values, handle specially
     if (log_A < -700.0) {
       A = 0.0;
       dA_dalpha = 0.0;
       d2A_dalpha2 = 0.0;
     } else {
       // Very large - handle with care
       A = safe_exp(log_A);
       dA_dalpha = A * ln_xi;
       d2A_dalpha2 = dA_dalpha * ln_xi;
     }
   } else {
     // Normal range - standard calculation
     A = std::exp(log_A);
     dA_dalpha = A * ln_xi;
     d2A_dalpha2 = dA_dalpha * ln_xi;
   }
   
   // ---- Compute v = 1-x^α and its derivatives safely ----
   double v; // v = 1 - x^α
   double ln_v; // ln(v)
   double dv_dalpha; // dv/dα = -x^α * ln(x)
   double d2v_dalpha2; // d²v/dα² = -x^α * (ln(x))²
   
   if (A > 0.5) {
     // For A close to 1, use more accurate computation
     v = -std::expm1(log_A);  // v = 1-e^(α*ln(x)) more accurately
     dv_dalpha = -dA_dalpha;
     d2v_dalpha2 = -d2A_dalpha2;
   } else {
     // Standard computation is fine for smaller A
     v = 1.0 - A;
     dv_dalpha = -dA_dalpha;
     d2v_dalpha2 = -d2A_dalpha2;
   }
   
   // Safety check - ensure v is strictly in (0,1)
   if (v <= eps || v >= 1.0 - eps) {
     v = std::max(std::min(v, 1.0 - eps), eps);
   }
   
   // Compute ln(v) safely
   ln_v = safe_log(v);
   
   // ---- Compute L5 derivatives: (β(δ+1)-1)*ln(v) ----
   // Second derivative w.r.t. α: (β(δ+1)-1) * [(d²v/dα² * v - (dv/dα)²) / v²]
   double d2L5_dalpha2 = 0.0;
   if (std::abs(beta_delta_factor) > eps) {
     double term = (d2v_dalpha2 * v - dv_dalpha * dv_dalpha) / (v * v);
     if (std::isfinite(term)) {
       d2L5_dalpha2 = beta_delta_factor * term;
     }
   }
   
   // Mixed derivative: d²L5/(dα dβ) = (δ+1) * (dv_dalpha/v)
   double d2L5_dalpha_dbeta = (delta + 1.0) * (dv_dalpha / v);
   
   // Mixed derivative: d²L5/(dα dδ) = β * (dv_dalpha/v)
   double d2L5_dalpha_ddelta = beta * (dv_dalpha / v);
   
   // Mixed derivative: d²L5/(dβ dδ) = ln(v)
   double d2L5_dbeta_ddelta = ln_v;
   
   // ---- Compute w = 1-v^β and its derivatives safely ----
   double v_beta; // v^β
   double w; // w = 1 - v^β
   
   // Compute v^β safely using log domain when helpful
   if (beta > 100 || v < 0.01) {
     double log_v_beta = beta * ln_v;
     v_beta = safe_exp(log_v_beta);
   } else {
     v_beta = safe_pow(v, beta);
   }
   
   // Compute w = 1-v^β carefully
   if (v_beta > 0.5) {
     // For v_beta close to 1, use more accurate computation
     double log_v_beta = beta * ln_v;
     w = -std::expm1(log_v_beta);  // w = 1-e^(β*ln(v)) more accurately
   } else {
     // Standard computation is fine for smaller v_beta
     w = 1.0 - v_beta;
   }
   
   // Safety check - ensure w is strictly in (0,1)
   if (w <= eps || w >= 1.0 - eps) {
     w = std::max(std::min(w, 1.0 - eps), eps);
   }
   
   // Compute ln(w) safely
   // double ln_w; // ln(w)
   // double ln_w = safe_log(w);
   
   // ---- Derivatives for w ----
   // dw/dv = -β * v^(β-1)
   double v_beta_m1 = (beta > 1.0) ? v_beta / v : 1.0;  // v^(β-1)
   if (beta == 1.0) v_beta_m1 = 1.0;  // Special case
   
   double dw_dv = -beta * v_beta_m1;
   
   // Chain rule: dw/dα = dw/dv * dv/dα
   double dw_dalpha = dw_dv * dv_dalpha;
   
   // ---- Compute L6 derivatives: (γ-1)*ln(w) ----
   // Only compute if γ != 1 to avoid unnecessary work
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
     
     // Second derivative of ln(w) w.r.t. α
     double term_alpha2 = (d2w_dalpha2 * w - dw_dalpha * dw_dalpha) / (w * w);
     if (std::isfinite(term_alpha2)) {
       d2L6_dalpha2 = gamma_minus_1 * term_alpha2;
     }
     
     // Derivative w.r.t. β: d/dβ ln(w)
     double dw_dbeta = -v_beta * ln_v;
     
     // Second derivative of ln(w) w.r.t. β
     double d2w_dbeta2 = -v_beta * (ln_v * ln_v);
     double term_beta2 = (d2w_dbeta2 * w - dw_dbeta * dw_dbeta) / (w * w);
     if (std::isfinite(term_beta2)) {
       d2L6_dbeta2 = gamma_minus_1 * term_beta2;
     }
     
     // Mixed derivative (α,β)
     double d_dw_dalpha_dbeta = -v_beta_m1 * (1.0 + beta * ln_v) * dv_dalpha;
     double mixed_term = (d_dw_dalpha_dbeta / w) - (dw_dalpha * dw_dbeta) / (w * w);
     if (std::isfinite(mixed_term)) {
       d2L6_dalpha_dbeta = gamma_minus_1 * mixed_term;
     }
     
     // Mixed derivatives with γ
     d2L6_dalpha_dgamma = dw_dalpha / w;
     d2L6_dbeta_dgamma = dw_dbeta / w;
   }
   
   // ---- Accumulate contributions to the Hessian matrix ----
   // Check each contribution for finiteness before adding
   
   // H(α,α): contributions from L5 and L6
   if (std::isfinite(d2L5_dalpha2)) H(0, 0) += d2L5_dalpha2;
   if (std::isfinite(d2L6_dalpha2)) H(0, 0) += d2L6_dalpha2;
   
   // H(β,β): contributions from L6
   if (std::isfinite(d2L6_dbeta2)) H(1, 1) += d2L6_dbeta2;
   
   // H(α,β): mixed from L5 and L6
   if (std::isfinite(d2L5_dalpha_dbeta)) H(0, 1) += d2L5_dalpha_dbeta;
   if (std::isfinite(d2L6_dalpha_dbeta)) H(0, 1) += d2L6_dalpha_dbeta;
   H(1, 0) = H(0, 1);  // Symmetric
   
   // H(α,γ): mixed from L6
   if (std::isfinite(d2L6_dalpha_dgamma)) H(0, 2) += d2L6_dalpha_dgamma;
   H(2, 0) = H(0, 2);  // Symmetric
   
   // H(α,δ): mixed from L5
   if (std::isfinite(d2L5_dalpha_ddelta)) H(0, 3) += d2L5_dalpha_ddelta;
   H(3, 0) = H(0, 3);  // Symmetric
   
   // H(β,γ): mixed from L6
   if (std::isfinite(d2L6_dbeta_dgamma)) H(1, 2) += d2L6_dbeta_dgamma;
   H(2, 1) = H(1, 2);  // Symmetric
   
   // H(β,δ): mixed from L5
   if (std::isfinite(d2L5_dbeta_ddelta)) H(1, 3) += d2L5_dbeta_ddelta;
   H(3, 1) = H(1, 3);  // Symmetric
 }
 
 // Final safety check - verify the Hessian is valid
 if (!H.is_finite()) {
   Rcpp::warning("Hessian calculation produced non-finite values");
   return nanH;
 }
 
 // Return the analytic Hessian matrix of the negative log-likelihood
 return Rcpp::wrap(-H);
}
