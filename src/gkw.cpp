// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include "utils.h"

//' @title Density of the Generalized Kumaraswamy Distribution
//' @author Lopes, J. E.
//' @keywords distribution density
//'
//' @description
//' Computes the probability density function (PDF) for the five-parameter
//' Generalized Kumaraswamy (GKw) distribution, defined on the interval (0, 1).
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
//' @param lambda Shape parameter \code{lambda} > 0. Can be a scalar or a vector.
//'   Default: 1.0.
//' @param log_prob Logical; if \code{TRUE}, the logarithm of the density is
//'   returned. Default: \code{FALSE}.
//'
//' @return A vector of density values (\eqn{f(x)}) or log-density values
//'   (\eqn{\log(f(x))}). The length of the result is determined by the recycling
//'   rule applied to the arguments (\code{x}, \code{alpha}, \code{beta},
//'   \code{gamma}, \code{delta}, \code{lambda}). Returns \code{0} (or \code{-Inf}
//'   if \code{log_prob = TRUE}) for \code{x} outside the interval (0, 1), or
//'   \code{NaN} if parameters are invalid.
//'
//' @details
//' The probability density function of the Generalized Kumaraswamy (GKw)
//' distribution with parameters \code{alpha} (\eqn{\alpha}), \code{beta}
//' (\eqn{\beta}), \code{gamma} (\eqn{\gamma}), \code{delta} (\eqn{\delta}), and
//' \code{lambda} (\eqn{\lambda}) is given by:
//' \deqn{
//' f(x; \alpha, \beta, \gamma, \delta, \lambda) =
//'   \frac{\lambda \alpha \beta x^{\alpha-1}(1-x^{\alpha})^{\beta-1}}
//'        {B(\gamma, \delta+1)}
//'   [1-(1-x^{\alpha})^{\beta}]^{\gamma\lambda-1}
//'   [1-[1-(1-x^{\alpha})^{\beta}]^{\lambda}]^{\delta}
//' }
//' for \eqn{x \in (0,1)}, where \eqn{B(a, b)} is the Beta function
//' \code{\link[base]{beta}}.
//'
//' This distribution was proposed by Cordeiro & de Castro (2011) and includes
//' several other distributions as special cases:
//' \itemize{
//'   \item Kumaraswamy (Kw): \code{gamma = 1}, \code{delta = 0}, \code{lambda = 1}
//'   \item Exponentiated Kumaraswamy (EKw): \code{gamma = 1}, \code{delta = 0}
//'   \item Beta-Kumaraswamy (BKw): \code{lambda = 1}
//'   \item Generalized Beta type 1 (GB1 - implies McDonald): \code{alpha = 1}, \code{beta = 1}
//'   \item Beta distribution: \code{alpha = 1}, \code{beta = 1}, \code{lambda = 1}
//' }
//' The function includes checks for valid parameters and input values \code{x}.
//' It uses numerical stabilization for \code{x} close to 0 or 1.
//'
//' @references
//' Cordeiro, G. M., & de Castro, M. (2011). A new family of generalized
//' distributions. *Journal of Statistical Computation and Simulation*,
//' *81*(7), 883-898.
//'
//' Kumaraswamy, P. (1980). A generalized probability density function for
//' double-bounded random processes. *Journal of Hydrology*, *46*(1-2), 79-88.
//'
//' @seealso
//' \code{\link{pgkw}}, \code{\link{qgkw}}, \code{\link{rgkw}} (if these exist),
//' \code{\link[stats]{dbeta}}, \code{\link[stats]{integrate}}
//'
//' @examples
//' \donttest{
//' # Simple density evaluation at a point
//' dgkw(0.5, alpha = 2, beta = 3, gamma = 1, delta = 0, lambda = 1) # Kw case
//'
//' # Plot the PDF for various parameter sets
//' x_vals <- seq(0.01, 0.99, by = 0.01)
//'
//' # Standard Kumaraswamy (gamma=1, delta=0, lambda=1)
//' pdf_kw <- dgkw(x_vals, alpha = 2, beta = 3, gamma = 1, delta = 0, lambda = 1)
//'
//' # Beta equivalent (alpha=1, beta=1, lambda=1) - Beta(gamma, delta+1)
//' pdf_beta <- dgkw(x_vals, alpha = 1, beta = 1, gamma = 2, delta = 3, lambda = 1)
//' # Compare with stats::dbeta
//' pdf_beta_check <- stats::dbeta(x_vals, shape1 = 2, shape2 = 3 + 1)
//' # max(abs(pdf_beta - pdf_beta_check)) # Should be close to zero
//'
//' # Exponentiated Kumaraswamy (gamma=1, delta=0)
//' pdf_ekw <- dgkw(x_vals, alpha = 2, beta = 3, gamma = 1, delta = 0, lambda = 2)
//'
//' plot(x_vals, pdf_kw, type = "l", ylim = range(c(pdf_kw, pdf_beta, pdf_ekw)),
//'      main = "GKw Densities Examples", ylab = "f(x)", xlab="x", col = "blue")
//' lines(x_vals, pdf_beta, col = "red")
//' lines(x_vals, pdf_ekw, col = "green")
//' legend("topright", legend = c("Kw(2,3)", "Beta(2,4) equivalent", "EKw(2,3, lambda=2)"),
//'        col = c("blue", "red", "green"), lty = 1, bty = "n")
//'
//' # Log-density
//' log_pdf_val <- dgkw(0.5, 2, 3, 1, 0, 1, log_prob = TRUE)
//' print(log_pdf_val)
//' print(log(dgkw(0.5, 2, 3, 1, 0, 1))) # Should match
//'
//' }
//'
//' @export
// [[Rcpp::export]]
Rcpp::NumericVector dgkw(
   const arma::vec& x,
   const Rcpp::NumericVector& alpha,
   const Rcpp::NumericVector& beta,
   const Rcpp::NumericVector& gamma,
   const Rcpp::NumericVector& delta,
   const Rcpp::NumericVector& lambda,
   bool log_prob = false
) {
 // Convert NumericVector to arma::vec
 arma::vec alpha_vec(alpha.begin(), alpha.size());
 arma::vec beta_vec(beta.begin(), beta.size());
 arma::vec gamma_vec(gamma.begin(), gamma.size());
 arma::vec delta_vec(delta.begin(), delta.size());
 arma::vec lambda_vec(lambda.begin(), lambda.size());
 
 // Find the maximum length for broadcasting
 size_t n = std::max({x.n_elem, alpha_vec.n_elem, beta_vec.n_elem,
                     gamma_vec.n_elem, delta_vec.n_elem, lambda_vec.n_elem});
 
 // Initialize result vector
 arma::vec result(n);
 if (log_prob) {
   result.fill(R_NegInf);
 } else {
   result.fill(0.0);
 }
 
 // Process each element
 for (size_t i = 0; i < n; ++i) {
   // Get parameter values with broadcasting/recycling
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
   
   // Check if x is within (0,1)
   if (xi <= 0.0 || xi >= 1.0 || !R_finite(xi)) {
     continue;
   }
   
   // Numerical stability: avoid calculations very close to 0 or 1
   double x_near_zero = std::pow(SQRT_EPSILON, 1.0/a);
   double x_near_one = 1.0 - std::pow(SQRT_EPSILON, 1.0/a);
   
   if (xi < x_near_zero || xi > x_near_one) {
     continue;
   }
   
   // Precalculate common terms used in the PDF formula
   double log_beta_val = R::lgammafn(g) + R::lgammafn(d + 1.0) - R::lgammafn(g + d + 1.0);
   double log_const = std::log(l) + std::log(a) + std::log(b) - log_beta_val;
   double gamma_lambda = g * l;
   
   // Calculate x^α
   double log_xi = std::log(xi);
   double log_x_alpha = a * log_xi;
   double x_alpha = std::exp(log_x_alpha);
   
   // Check if x^α < 1 for numerical stability
   if (x_alpha >= 1.0 - SQRT_EPSILON) {
     continue;
   }
   
   // Calculate (1 - x^α)
   double log_one_minus_x_alpha = log1mexp(log_x_alpha);
   if (!R_finite(log_one_minus_x_alpha)) {
     continue;
   }
   
   // Calculate (1 - x^α)^β
   double log_one_minus_x_alpha_beta = b * log_one_minus_x_alpha;
   
   // Calculate 1 - (1 - x^α)^β
   double log_term1 = log1mexp(log_one_minus_x_alpha_beta);
   if (!R_finite(log_term1)) {
     continue;
   }
   
   // Calculate [1-(1-x^α)^β]^λ
   double log_term1_lambda = l * log_term1;
   
   // Calculate 1 - [1-(1-x^α)^β]^λ
   double log_term2 = log1mexp(log_term1_lambda);
   if (!R_finite(log_term2)) {
     continue;
   }
   
   // Assemble the full log-density expression
   double logdens = log_const +
     (a - 1.0) * log_xi +
     (b - 1.0) * log_one_minus_x_alpha +
     (gamma_lambda - 1.0) * log_term1 +
     d * log_term2;
   
   // Check for invalid result
   if (!R_finite(logdens)) {
     continue;
   }
   
   // Return log-density or density as requested
   result(i) = log_prob ? logdens : safe_exp(logdens);
 }
 
 return Rcpp::NumericVector(result.memptr(), result.memptr() + result.n_elem);
}

//' @title Generalized Kumaraswamy Distribution CDF
//' @author Lopes, J. E.
//' @keywords distribution cumulative
//'
//' @description
//' Computes the cumulative distribution function (CDF) for the five-parameter
//' Generalized Kumaraswamy (GKw) distribution, defined on the interval (0, 1).
//' Calculates \eqn{P(X \le q)}.
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
//' @param lambda Shape parameter \code{lambda} > 0. Can be a scalar or a vector.
//'   Default: 1.0.
//' @param lower_tail Logical; if \code{TRUE} (default), probabilities are
//'   \eqn{P(X \le q)}, otherwise, \eqn{P(X > q)}.
//' @param log_p Logical; if \code{TRUE}, probabilities \eqn{p} are given as
//'   \eqn{\log(p)}. Default: \code{FALSE}.
//'
//' @return A vector of probabilities, \eqn{F(q)}, or their logarithms if
//'   \code{log_p = TRUE}. The length of the result is determined by the recycling
//'   rule applied to the arguments (\code{q}, \code{alpha}, \code{beta},
//'   \code{gamma}, \code{delta}, \code{lambda}). Returns \code{0} (or \code{-Inf}
//'   if \code{log_p = TRUE}) for \code{q <= 0} and \code{1} (or \code{0} if
//'   \code{log_p = TRUE}) for \code{q >= 1}. Returns \code{NaN} for invalid
//'   parameters.
//'
//' @details
//' The cumulative distribution function (CDF) of the Generalized Kumaraswamy (GKw)
//' distribution with parameters \code{alpha} (\eqn{\alpha}), \code{beta}
//' (\eqn{\beta}), \code{gamma} (\eqn{\gamma}), \code{delta} (\eqn{\delta}), and
//' \code{lambda} (\eqn{\lambda}) is given by:
//' \deqn{
//' F(q; \alpha, \beta, \gamma, \delta, \lambda) =
//'   I_{x(q)}(\gamma, \delta+1)
//' }
//' where \eqn{x(q) = [1-(1-q^{\alpha})^{\beta}]^{\lambda}} and \eqn{I_x(a, b)}
//' is the regularized incomplete beta function, defined as:
//' \deqn{
//' I_x(a, b) = \frac{B_x(a, b)}{B(a, b)} = \frac{\int_0^x t^{a-1}(1-t)^{b-1} dt}{\int_0^1 t^{a-1}(1-t)^{b-1} dt}
//' }
//' This corresponds to the \code{\link[stats]{pbeta}} function in R, such that
//' \eqn{F(q; \alpha, \beta, \gamma, \delta, \lambda) = \code{pbeta}(x(q), \code{shape1} = \gamma, \code{shape2} = \delta+1)}.
//'
//' The GKw distribution includes several special cases, such as the Kumaraswamy,
//' Beta, and Exponentiated Kumaraswamy distributions (see \code{\link{dgkw}} for details).
//' The function utilizes numerical algorithms for computing the regularized
//' incomplete beta function accurately, especially near the boundaries.
//'
//' @references
//' Cordeiro, G. M., & de Castro, M. (2011). A new family of generalized
//' distributions. *Journal of Statistical Computation and Simulation*
//'
//' Kumaraswamy, P. (1980). A generalized probability density function for
//' double-bounded random processes. *Journal of Hydrology*, *46*(1-2), 79-88.
//'
//' @seealso
//' \code{\link{dgkw}}, \code{\link{qgkw}}, \code{\link{rgkw}},
//' \code{\link[stats]{pbeta}}
//'
//' @examples
//' \donttest{
//' # Simple CDF evaluation
//' prob <- pgkw(0.5, alpha = 2, beta = 3, gamma = 1, delta = 0, lambda = 1) # Kw case
//' print(prob)
//'
//' # Upper tail probability P(X > q)
//' prob_upper <- pgkw(0.5, alpha = 2, beta = 3, gamma = 1, delta = 0, lambda = 1,
//'                  lower_tail = FALSE)
//' print(prob_upper)
//' # Check: prob + prob_upper should be 1
//' print(prob + prob_upper)
//'
//' # Log probability
//' log_prob <- pgkw(0.5, alpha = 2, beta = 3, gamma = 1, delta = 0, lambda = 1,
//'                  log_p = TRUE)
//' print(log_prob)
//' # Check: exp(log_prob) should be prob
//' print(exp(log_prob))
//'
//' # Use of vectorized parameters
//' q_vals <- c(0.2, 0.5, 0.8)
//' alphas_vec <- c(0.5, 1.0, 2.0)
//' betas_vec <- c(1.0, 2.0, 3.0)
//' # Vectorizes over q, alpha, beta
//' pgkw(q_vals, alpha = alphas_vec, beta = betas_vec, gamma = 1, delta = 0.5, lambda = 0.5)
//'
//' # Plotting the CDF for special cases
//' x_seq <- seq(0.01, 0.99, by = 0.01)
//' # Standard Kumaraswamy CDF
//' cdf_kw <- pgkw(x_seq, alpha = 2, beta = 3, gamma = 1, delta = 0, lambda = 1)
//' # Beta distribution CDF equivalent (Beta(gamma, delta+1))
//' cdf_beta_equiv <- pgkw(x_seq, alpha = 1, beta = 1, gamma = 2, delta = 3, lambda = 1)
//' # Compare with stats::pbeta
//' cdf_beta_check <- stats::pbeta(x_seq, shape1 = 2, shape2 = 3 + 1)
//' # max(abs(cdf_beta_equiv - cdf_beta_check)) # Should be close to zero
//'
//' plot(x_seq, cdf_kw, type = "l", ylim = c(0, 1),
//'      main = "GKw CDF Examples", ylab = "F(x)", xlab = "x", col = "blue")
//' lines(x_seq, cdf_beta_equiv, col = "red", lty = 2)
//' legend("bottomright", legend = c("Kw(2,3)", "Beta(2,4) equivalent"),
//'        col = c("blue", "red"), lty = c(1, 2), bty = "n")
//'}
//' @export
// [[Rcpp::export]]
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
 // Convert NumericVector to arma::vec
 arma::vec alpha_vec(alpha.begin(), alpha.size());
 arma::vec beta_vec(beta.begin(), beta.size());
 arma::vec gamma_vec(gamma.begin(), gamma.size());
 arma::vec delta_vec(delta.begin(), delta.size());
 arma::vec lambda_vec(lambda.begin(), lambda.size());
 
 // Find maximum length for broadcasting
 size_t n = std::max({q.n_elem, alpha_vec.n_elem, beta_vec.n_elem,
                     gamma_vec.n_elem, delta_vec.n_elem, lambda_vec.n_elem});
 
 // Initialize result vector
 arma::vec result(n);
 
 // Process each element
 for (size_t i = 0; i < n; ++i) {
   // Get parameter values with broadcasting/recycling
   double a = alpha_vec[i % alpha_vec.n_elem];
   double b = beta_vec[i % beta_vec.n_elem];
   double g = gamma_vec[i % gamma_vec.n_elem];
   double d = delta_vec[i % delta_vec.n_elem];
   double l = lambda_vec[i % lambda_vec.n_elem];
   double qi = q[i % q.n_elem];
   
   // Check parameter validity
   if (!check_pars(a, b, g, d, l)) {
     result(i) = NA_REAL;
     Rcpp::warning("pgkw: invalid parameters at index %d (alpha,beta,gamma>0, delta>=0, lambda>0)", i+1);
     continue;
   }
   
   // Check domain boundaries
   if (!R_finite(qi) || qi <= 0.0) {
     result(i) = lower_tail ? (log_p ? R_NegInf : 0.0) : (log_p ? 0.0 : 1.0);
     continue;
   }
   
   if (qi >= 1.0) {
     result(i) = lower_tail ? (log_p ? 0.0 : 1.0) : (log_p ? R_NegInf : 0.0);
     continue;
   }
   
   // Compute CDF using stable numerical methods
   double log_qi = std::log(qi);
   double log_qi_alpha = a * log_qi;
   double qi_alpha = safe_exp(log_qi_alpha);
   
   // Calculate (1 - q^alpha) with numerical stability
   double one_minus_qi_alpha;
   if (qi_alpha < 0.5) {
     one_minus_qi_alpha = 1.0 - qi_alpha;
   } else {
     // If close to 1, use expm1 for better precision
     one_minus_qi_alpha = -std::expm1(log_qi_alpha);
   }
   
   // Boundary checks
   if (one_minus_qi_alpha <= 0.0) {
     result(i) = lower_tail ? (log_p ? 0.0 : 1.0) : (log_p ? R_NegInf : 0.0);
     continue;
   }
   
   if (one_minus_qi_alpha >= 1.0) {
     result(i) = lower_tail ? (log_p ? R_NegInf : 0.0) : (log_p ? 0.0 : 1.0);
     continue;
   }
   
   // Calculate (1 - q^alpha)^beta
   double log_oma = std::log(one_minus_qi_alpha);
   double log_oma_beta = b * log_oma;
   double oma_beta = safe_exp(log_oma_beta);
   
   // Calculate 1 - (1 - q^alpha)^beta
   double term = 1.0 - oma_beta;
   
   // Boundary checks
   if (term <= 0.0) {
     result(i) = lower_tail ? (log_p ? R_NegInf : 0.0) : (log_p ? 0.0 : 1.0);
     continue;
   }
   
   if (term >= 1.0) {
     result(i) = lower_tail ? (log_p ? 0.0 : 1.0) : (log_p ? R_NegInf : 0.0);
     continue;
   }
   
   // Calculate [1 - (1 - q^alpha)^beta]^lambda
   double log_term = std::log(term);
   double log_y = l * log_term;
   double y = safe_exp(log_y);
   
   // Boundary checks
   if (y <= 0.0) {
     result(i) = lower_tail ? (log_p ? R_NegInf : 0.0) : (log_p ? 0.0 : 1.0);
     continue;
   }
   
   if (y >= 1.0) {
     result(i) = lower_tail ? (log_p ? 0.0 : 1.0) : (log_p ? R_NegInf : 0.0);
     continue;
   }
   
   // Use pbeta for the final calculation
   double prob = R::pbeta(y, g, d + 1.0, /*lower_tail=*/true, /*log_p=*/false);
   
   // Adjust for upper tail if requested
   if (!lower_tail) {
     prob = 1.0 - prob;
   }
   
   // Convert to log scale if requested
   if (log_p) {
     if (prob <= 0.0) {
       prob = R_NegInf;
     } else if (prob >= 1.0) {
       prob = 0.0;
     } else {
       prob = std::log(prob);
     }
   }
   
   result(i) = prob;
 }
 
 return Rcpp::NumericVector(result.memptr(), result.memptr() + result.n_elem);
}


//' @title Generalized Kumaraswamy Distribution Quantile Function
//' @author Lopes, J. E.
//' @keywords distribution quantile
//'
//' @description
//' Computes the quantile function (inverse CDF) for the five-parameter
//' Generalized Kumaraswamy (GKw) distribution. Finds the value \code{x} such
//' that \eqn{P(X \le x) = p}, where \code{X} follows the GKw distribution.
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
//' @param lambda Shape parameter \code{lambda} > 0. Can be a scalar or a vector.
//'   Default: 1.0.
//' @param lower_tail Logical; if \code{TRUE} (default), probabilities are
//'   \eqn{P(X \le x)}, otherwise, \eqn{P(X > x)}.
//' @param log_p Logical; if \code{TRUE}, probabilities \code{p} are given as
//'   \eqn{\log(p)}. Default: \code{FALSE}.
//'
//' @return A vector of quantiles corresponding to the given probabilities \code{p}.
//'   The length of the result is determined by the recycling rule applied to
//'   the arguments (\code{p}, \code{alpha}, \code{beta}, \code{gamma},
//'   \code{delta}, \code{lambda}). Returns:
//'   \itemize{
//'     \item \code{0} for \code{p = 0} (or \code{p = -Inf} if \code{log_p = TRUE}).
//'     \item \code{1} for \code{p = 1} (or \code{p = 0} if \code{log_p = TRUE}).
//'     \item \code{NaN} for \code{p < 0} or \code{p > 1} (or corresponding log scale).
//'     \item \code{NaN} for invalid parameters (e.g., \code{alpha <= 0},
//'           \code{beta <= 0}, \code{gamma <= 0}, \code{delta < 0},
//'           \code{lambda <= 0}).
//'   }
//'
//' @details
//' The quantile function \eqn{Q(p)} is the inverse of the CDF \eqn{F(x)}.
//' Given \eqn{F(x) = I_{y(x)}(\gamma, \delta+1)} where
//' \eqn{y(x) = [1-(1-x^{\alpha})^{\beta}]^{\lambda}}, the quantile function is:
//' \deqn{
//' Q(p) = x = \left\{ 1 - \left[ 1 - \left( I^{-1}_{p}(\gamma, \delta+1) \right)^{1/\lambda} \right]^{1/\beta} \right\}^{1/\alpha}
//' }
//' where \eqn{I^{-1}_{p}(a, b)} is the inverse of the regularized incomplete beta
//' function, which corresponds to the quantile function of the Beta distribution,
//' \code{\link[stats]{qbeta}}.
//'
//' The computation proceeds as follows:
//' \enumerate{
//'   \item Calculate \code{y = stats::qbeta(p, shape1 = gamma, shape2 = delta + 1, lower.tail = lower_tail, log.p = log_p)}.
//'   \item Calculate \eqn{v = y^{1/\lambda}}.
//'   \item Calculate \eqn{w = (1 - v)^{1/\beta}}. Note: Requires \eqn{v \le 1}.
//'   \item Calculate \eqn{q = (1 - w)^{1/\alpha}}. Note: Requires \eqn{w \le 1}.
//' }
//' Numerical stability is maintained by handling boundary cases (\code{p = 0},
//' \code{p = 1}) directly and checking intermediate results (e.g., ensuring
//' arguments to powers are non-negative).
//'
//' @references
//' Cordeiro, G. M., & de Castro, M. (2011). A new family of generalized
//' distributions. *Journal of Statistical Computation and Simulation*
//'
//' Kumaraswamy, P. (1980). A generalized probability density function for
//' double-bounded random processes. *Journal of Hydrology*, *46*(1-2), 79-88.
//'
//' @seealso
//' \code{\link{dgkw}}, \code{\link{pgkw}}, \code{\link{rgkw}},
//' \code{\link[stats]{qbeta}}
//'
//' @examples
//' \donttest{
//' # Basic quantile calculation (median)
//' median_val <- qgkw(0.5, alpha = 2, beta = 3, gamma = 1, delta = 0, lambda = 1)
//' print(median_val)
//'
//' # Computing multiple quantiles
//' probs <- c(0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)
//' quantiles <- qgkw(probs, alpha = 2, beta = 3, gamma = 1, delta = 0, lambda = 1)
//' print(quantiles)
//'
//' # Upper tail quantile (e.g., find x such that P(X > x) = 0.1, which is 90th percentile)
//' q90 <- qgkw(0.1, alpha = 2, beta = 3, gamma = 1, delta = 0, lambda = 1,
//'             lower_tail = FALSE)
//' print(q90)
//' # Check: should match quantile for p = 0.9 with lower_tail = TRUE
//' print(qgkw(0.9, alpha = 2, beta = 3, gamma = 1, delta = 0, lambda = 1))
//'
//' # Log probabilities
//' median_logp <- qgkw(log(0.5), alpha = 2, beta = 3, gamma = 1, delta = 0, lambda = 1,
//'                     log_p = TRUE)
//' print(median_logp) # Should match median_val
//'
//' # Vectorized parameters
//' alphas_vec <- c(0.5, 1.0, 2.0)
//' betas_vec <- c(1.0, 2.0, 3.0)
//' # Get median for 3 different GKw distributions
//' medians_vec <- qgkw(0.5, alpha = alphas_vec, beta = betas_vec, gamma = 1, delta = 0, lambda = 1)
//' print(medians_vec)
//'
//' # Verify inverse relationship with pgkw
//' p_val <- 0.75
//' x_val <- qgkw(p_val, alpha = 2, beta = 3, gamma = 1, delta = 0, lambda = 1)
//' p_check <- pgkw(x_val, alpha = 2, beta = 3, gamma = 1, delta = 0, lambda = 1)
//' print(paste("Calculated p:", p_check, " (Expected:", p_val, ")"))
//'}
//' @export
// [[Rcpp::export]]
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
 // Convert NumericVector to arma::vec
 arma::vec alpha_vec(alpha.begin(), alpha.size());
 arma::vec beta_vec(beta.begin(), beta.size());
 arma::vec gamma_vec(gamma.begin(), gamma.size());
 arma::vec delta_vec(delta.begin(), delta.size());
 arma::vec lambda_vec(lambda.begin(), lambda.size());
 
 // Find maximum length for broadcasting
 size_t n = std::max({p.n_elem, alpha_vec.n_elem, beta_vec.n_elem,
                     gamma_vec.n_elem, delta_vec.n_elem, lambda_vec.n_elem});
 
 // Initialize result vector
 arma::vec result(n);
 
 // Process each element
 for (size_t i = 0; i < n; ++i) {
   // Get parameter values with broadcasting/recycling
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
   
   // Process log_p and lower_tail
   if (log_p) {
     if (pp > 0.0) {
       // log(p) > 0 implies p > 1, which is invalid
       result(i) = NA_REAL;
       continue;
     }
     pp = std::exp(pp);  // Convert from log-scale
   }
   
   if (!lower_tail) {
     pp = 1.0 - pp;  // Convert from upper tail to lower tail
   }
   
   // Check probability bounds
   if (!R_finite(pp) || pp < 0.0) {
     result(i) = 0.0;  // For p ≤ 0, quantile is 0
     continue;
   }
   
   if (pp > 1.0) {
     result(i) = 1.0;  // For p > 1, quantile is 1
     continue;
   }
   
   if (pp <= 0.0) {
     result(i) = 0.0;  // For p = 0, quantile is 0
     continue;
   }
   
   if (pp >= 1.0) {
     result(i) = 1.0;  // For p = 1, quantile is 1
     continue;
   }
   
   // Step 1: Find y = qbeta(p, γ, δ+1)
   double y = R::qbeta(pp, g, d + 1.0, /*lower_tail=*/true, /*log_p=*/false);
   
   // Check for boundary conditions
   if (y <= 0.0) {
     result(i) = 0.0;
     continue;
   }
   
   if (y >= 1.0) {
     result(i) = 1.0;
     continue;
   }
   
   // Step 2: Compute v = y^(1/λ)
   double v;
   if (l == 1.0) {
     v = y;  // Optimization for λ=1
   } else {
     v = safe_pow(y, 1.0/l);
   }
   
   // Step 3: Compute tmp = 1 - v
   double tmp = 1.0 - v;
   
   // Check for boundary conditions
   if (tmp <= 0.0) {
     result(i) = 1.0;
     continue;
   }
   
   if (tmp >= 1.0) {
     result(i) = 0.0;
     continue;
   }
   
   // Step 4: Compute tmp2 = tmp^(1/β)
   double tmp2;
   if (b == 1.0) {
     tmp2 = tmp;  // Optimization for β=1
   } else {
     tmp2 = safe_pow(tmp, 1.0/b);
   }
   
   // Check for boundary conditions
   if (tmp2 <= 0.0) {
     result(i) = 1.0;
     continue;
   }
   
   if (tmp2 >= 1.0) {
     result(i) = 0.0;
     continue;
   }
   
   // Step 5: Compute q = (1 - tmp2)^(1/α)
   double one_minus_tmp2 = 1.0 - tmp2;
   double qq;
   
   if (one_minus_tmp2 <= 0.0) {
     qq = 0.0;
   } else if (one_minus_tmp2 >= 1.0) {
     qq = 1.0;
   } else if (a == 1.0) {
     qq = one_minus_tmp2;  // Optimization for α=1
   } else {
     qq = safe_pow(one_minus_tmp2, 1.0/a);
   }
   
   // Final boundary check to ensure result is in (0,1)
   if (qq < 0.0) {
     qq = 0.0;
   } else if (qq > 1.0) {
     qq = 1.0;
   }
   
   result(i) = qq;
 }
 
 return Rcpp::NumericVector(result.memptr(), result.memptr() + result.n_elem);
}


//' @title Generalized Kumaraswamy Distribution Random Generation
//' @author Lopes, J. E.
//' @keywords distribution random
//'
//' @description
//' Generates random deviates from the five-parameter Generalized Kumaraswamy (GKw)
//' distribution defined on the interval (0, 1).
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
//' @param lambda Shape parameter \code{lambda} > 0. Can be a scalar or a vector.
//'   Default: 1.0.
//'
//' @return A vector of length \code{n} containing random deviates from the GKw
//'   distribution. The length of the result is determined by \code{n} and the
//'   recycling rule applied to the parameters (\code{alpha}, \code{beta},
//'   \code{gamma}, \code{delta}, \code{lambda}). Returns \code{NaN} if parameters
//'   are invalid (e.g., \code{alpha <= 0}, \code{beta <= 0}, \code{gamma <= 0},
//'   \code{delta < 0}, \code{lambda <= 0}).
//'
//' @details
//' The generation method relies on the transformation property: if
//' \eqn{V \sim \mathrm{Beta}(\gamma, \delta+1)}, then the random variable \code{X}
//' defined as
//' \deqn{
//' X = \left\{ 1 - \left[ 1 - V^{1/\lambda} \right]^{1/\beta} \right\}^{1/\alpha}
//' }
//' follows the GKw(\eqn{\alpha, \beta, \gamma, \delta, \lambda}) distribution.
//'
//' The algorithm proceeds as follows:
//' \enumerate{
//'   \item Generate \code{V} from \code{stats::rbeta(n, shape1 = gamma, shape2 = delta + 1)}.
//'   \item Calculate \eqn{v = V^{1/\lambda}}.
//'   \item Calculate \eqn{w = (1 - v)^{1/\beta}}.
//'   \item Calculate \eqn{x = (1 - w)^{1/\alpha}}.
//' }
//' Parameters (\code{alpha}, \code{beta}, \code{gamma}, \code{delta}, \code{lambda})
//' are recycled to match the length required by \code{n}. Numerical stability is
//' maintained by handling potential edge cases during the transformations.
//'
//' @references
//' Cordeiro, G. M., & de Castro, M. (2011). A new family of generalized
//' distributions. *Journal of Statistical Computation and Simulation*
//'
//' Kumaraswamy, P. (1980). A generalized probability density function for
//' double-bounded random processes. *Journal of Hydrology*, *46*(1-2), 79-88.
//'
//' @seealso
//' \code{\link{dgkw}}, \code{\link{pgkw}}, \code{\link{qgkw}},
//' \code{\link[stats]{rbeta}}, \code{\link[base]{set.seed}}
//'
//' @examples
//' \donttest{
//' set.seed(1234) # for reproducibility
//'
//' # Generate 1000 random values from a specific GKw distribution (Kw case)
//' x_sample <- rgkw(1000, alpha = 2, beta = 3, gamma = 1, delta = 0, lambda = 1)
//' summary(x_sample)
//'
//' # Histogram of generated values compared to theoretical density
//' hist(x_sample, breaks = 30, freq = FALSE, # freq=FALSE for density scale
//'      main = "Histogram of GKw(2,3,1,0,1) Sample", xlab = "x", ylim = c(0, 2.5))
//' curve(dgkw(x, alpha = 2, beta = 3, gamma = 1, delta = 0, lambda = 1),
//'       add = TRUE, col = "red", lwd = 2, n = 201)
//' legend("topright", legend = "Theoretical PDF", col = "red", lwd = 2, bty = "n")
//'
//' # Comparing empirical and theoretical quantiles (Q-Q plot)
//' prob_points <- seq(0.01, 0.99, by = 0.01)
//' theo_quantiles <- qgkw(prob_points, alpha = 2, beta = 3, gamma = 1, delta = 0, lambda = 1)
//' emp_quantiles <- quantile(x_sample, prob_points)
//'
//' plot(theo_quantiles, emp_quantiles, pch = 16, cex = 0.8,
//'      main = "Q-Q Plot for GKw(2,3,1,0,1)",
//'      xlab = "Theoretical Quantiles", ylab = "Empirical Quantiles (n=1000)")
//' abline(a = 0, b = 1, col = "blue", lty = 2)
//'
//' # Using vectorized parameters: generate 1 value for each alpha
//' alphas_vec <- c(0.5, 1.0, 2.0)
//' n_param <- length(alphas_vec)
//' samples_vec <- rgkw(n_param, alpha = alphas_vec, beta = 2, gamma = 1, delta = 0, lambda = 1)
//' print(samples_vec) # One sample for each alpha value
//' # Result length matches n=3, parameters alpha recycled accordingly
//'
//' # Example with invalid parameters (should produce NaN)
//' invalid_sample <- rgkw(1, alpha = -1, beta = 2, gamma = 1, delta = 0, lambda = 1)
//' print(invalid_sample)
//'}
//' @export
// [[Rcpp::export]]
Rcpp::NumericVector rgkw(
   int n,
   const Rcpp::NumericVector& alpha,
   const Rcpp::NumericVector& beta,
   const Rcpp::NumericVector& gamma,
   const Rcpp::NumericVector& delta,
   const Rcpp::NumericVector& lambda
) {
 // Convert NumericVector to arma::vec
 arma::vec alpha_vec(alpha.begin(), alpha.size());
 arma::vec beta_vec(beta.begin(), beta.size());
 arma::vec gamma_vec(gamma.begin(), gamma.size());
 arma::vec delta_vec(delta.begin(), delta.size());
 arma::vec lambda_vec(lambda.begin(), lambda.size());
 
 // Count of parameter combinations for vectorization
 size_t k = std::max({alpha_vec.n_elem, beta_vec.n_elem, gamma_vec.n_elem,
                     delta_vec.n_elem, lambda_vec.n_elem});
 
 // Initialize result vector
 arma::vec result(n);
 
 // Process each element
 for (int i = 0; i < n; ++i) {
   // Index for parameter combination (cycling through k combinations)
   size_t idx = i % k;
   
   // Get parameter values with broadcasting/recycling
   double a = alpha_vec[idx % alpha_vec.n_elem];
   double b = beta_vec[idx % beta_vec.n_elem];
   double g = gamma_vec[idx % gamma_vec.n_elem];
   double d = delta_vec[idx % delta_vec.n_elem];
   double l = lambda_vec[idx % lambda_vec.n_elem];
   
   // Validate parameters
   if (!check_pars(a, b, g, d, l)) {
     result(i) = NA_REAL;
     Rcpp::warning("rgkw: invalid parameters at index %d (alpha,beta,gamma>0, delta>=0, lambda>0)", idx+1);
     continue;
   }
   
   // Generate Beta(γ, δ+1) random value
   double vi = R::rbeta(g, d + 1.0);
   
   // Check for boundary conditions
   if (vi <= 0.0) {
     result(i) = 0.0;
     continue;
   }
   
   if (vi >= 1.0) {
     result(i) = 1.0;
     continue;
   }
   
   // Compute v = V^(1/λ)
   double vl;
   if (l == 1.0) {
     vl = vi;  // Optimization for λ=1
   } else {
     vl = safe_pow(vi, 1.0/l);
   }
   
   // Compute tmp = 1 - v
   double tmp = 1.0 - vl;
   
   // Check for boundary conditions
   if (tmp <= 0.0) {
     result(i) = 1.0;
     continue;
   }
   
   if (tmp >= 1.0) {
     result(i) = 0.0;
     continue;
   }
   
   // Compute tmp2 = tmp^(1/β)
   double tmp2;
   if (b == 1.0) {
     tmp2 = tmp;  // Optimization for β=1
   } else {
     tmp2 = safe_pow(tmp, 1.0/b);
   }
   
   // Check for boundary conditions
   if (tmp2 <= 0.0) {
     result(i) = 1.0;
     continue;
   }
   
   if (tmp2 >= 1.0) {
     result(i) = 0.0;
     continue;
   }
   
   // Compute x = (1 - tmp2)^(1/α)
   double one_minus_tmp2 = 1.0 - tmp2;
   double xx;
   
   if (one_minus_tmp2 <= 0.0) {
     xx = 0.0;
   } else if (one_minus_tmp2 >= 1.0) {
     xx = 1.0;
   } else if (a == 1.0) {
     xx = one_minus_tmp2;  // Optimization for α=1
   } else {
     xx = safe_pow(one_minus_tmp2, 1.0/a);
   }
   
   // Final boundary check to ensure result is in (0,1)
   if (xx < 0.0) {
     xx = 0.0;
   } else if (xx > 1.0) {
     xx = 1.0;
   }
   
   result(i) = xx;
 }
 
 return Rcpp::NumericVector(result.memptr(), result.memptr() + result.n_elem);
}





//' @title Negative Log-Likelihood for the Generalized Kumaraswamy Distribution
//' @author Lopes, J. E.
//' @keywords distribution likelihood optimize
//'
//' @description
//' Computes the negative log-likelihood function for the five-parameter
//' Generalized Kumaraswamy (GKw) distribution given a vector of observations.
//' This function is designed for use in optimization routines (e.g., maximum
//' likelihood estimation).
//'
//' @param par A numeric vector of length 5 containing the distribution parameters
//'   in the order: \code{alpha} (\eqn{\alpha > 0}), \code{beta} (\eqn{\beta > 0}),
//'   \code{gamma} (\eqn{\gamma > 0}), \code{delta} (\eqn{\delta \ge 0}),
//'   \code{lambda} (\eqn{\lambda > 0}).
//' @param data A numeric vector of observations. All values must be strictly
//'   between 0 and 1 (exclusive).
//'
//' @return Returns a single \code{double} value representing the negative
//'   log-likelihood (\eqn{-\ell(\theta|\mathbf{x})}). Returns a large positive
//'   value (e.g., \code{Inf}) if any parameter values in \code{par} are invalid
//'   according to their constraints, or if any value in \code{data} is not in
//'   the interval (0, 1).
//'
//' @details
//' The probability density function (PDF) of the GKw distribution is given in
//' \code{\link{dgkw}}. The log-likelihood function \eqn{\ell(\theta)} for a sample
//' \eqn{\mathbf{x} = (x_1, \dots, x_n)} is:
//' \deqn{
//' \ell(\theta | \mathbf{x}) = n\ln(\lambda\alpha\beta) - n\ln B(\gamma,\delta+1) +
//'   \sum_{i=1}^{n} [(\alpha-1)\ln(x_i) + (\beta-1)\ln(v_i) + (\gamma\lambda-1)\ln(w_i) + \delta\ln(z_i)]
//' }
//' where \eqn{\theta = (\alpha, \beta, \gamma, \delta, \lambda)}, \eqn{B(a,b)}
//' is the Beta function (\code{\link[base]{beta}}), and:
//' \itemize{
//'   \item \eqn{v_i = 1 - x_i^{\alpha}}
//'   \item \eqn{w_i = 1 - v_i^{\beta} = 1 - (1-x_i^{\alpha})^{\beta}}
//'   \item \eqn{z_i = 1 - w_i^{\lambda} = 1 - [1-(1-x_i^{\alpha})^{\beta}]^{\lambda}}
//' }
//' This function computes \eqn{-\ell(\theta|\mathbf{x})}.
//'
//' Numerical stability is prioritized using:
//' \itemize{
//'   \item \code{\link[base]{lbeta}} function for the log-Beta term.
//'   \item Log-transformations of intermediate terms (\eqn{v_i, w_i, z_i}) and
//'         use of \code{\link[base]{log1p}} where appropriate to handle values
//'         close to 0 or 1 accurately.
//'   \item Checks for invalid parameters and data.
//' }
//'
//' @references
//' Cordeiro, G. M., & de Castro, M. (2011). A new family of generalized
//' distributions. *Journal of Statistical Computation and Simulation*
//'
//' Kumaraswamy, P. (1980). A generalized probability density function for
//' double-bounded random processes. *Journal of Hydrology*, *46*(1-2), 79-88.
//'
//' @seealso
//' \code{\link{dgkw}}, \code{\link{pgkw}}, \code{\link{qgkw}}, \code{\link{rgkw}},
//' \code{\link{grgkw}}, \code{\link{hsgkw}} (gradient and Hessian functions, if available),
//' \code{\link[stats]{optim}}, \code{\link[base]{lbeta}}, \code{\link[base]{log1p}}
//'
//' @examples
//' \donttest{
//' # Generate sample data from a known GKw distribution
//' set.seed(123)
//' true_par <- c(alpha = 2, beta = 3, gamma = 1.0, delta = 0.5, lambda = 0.5)
//' sample_data <- rgkw(100, alpha = true_par[1], beta = true_par[2],
//'                    gamma = true_par[3], delta = true_par[4], lambda = true_par[5])
//' hist(sample_data, breaks = 20, main = "Sample GKw Data")
//'
//' # --- Maximum Likelihood Estimation using optim ---
//' # Initial parameter guess (can be crucial)
//' start_par <- c(1.5, 2.5, 1.2, 0.3, 0.6)
//'
//' # Perform optimization (minimizing negative log-likelihood)
//' # Ensure data is passed correctly to llgkw
//' mle_result <- stats::optim(par = start_par,
//'                            fn = llgkw,
//'                            method = "BFGS", # Method supporting bounds might be safer
//'                            hessian = TRUE,
//'                            data = sample_data)
//'
//' # Check convergence and results
//' if (mle_result$convergence == 0) {
//'   print("Optimization converged successfully.")
//'   mle_par <- mle_result$par
//'   print("Estimated parameters:")
//'   print(mle_par)
//'   print("True parameters:")
//'   print(true_par)
//'
//'   # Standard errors from Hessian (optional)
//'   # fisher_info <- solve(mle_result$hessian) # Need positive definite Hessian
//'   # std_errors <- sqrt(diag(fisher_info))
//'   # print("Approximate Standard Errors:")
//'   # print(std_errors)
//'
//' } else {
//'   warning("Optimization did not converge!")
//'   print(mle_result$message)
//' }
//'
//' # --- Compare numerical and analytical derivatives (if available) ---
//' # Requires the 'numDeriv' package and analytical functions 'grgkw', 'hsgkw'
//' if (requireNamespace("numDeriv", quietly = TRUE) &&
//'     exists("grgkw") && exists("hsgkw") && mle_result$convergence == 0) {
//'
//'   cat("\nComparing Derivatives at MLE estimates:\n")
//'
//'   # Numerical derivatives
//'   num_grad <- numDeriv::grad(func = llgkw, x = mle_par, data = sample_data)
//'   num_hess <- numDeriv::hessian(func = llgkw, x = mle_par, data = sample_data)
//'
//'   # Analytical derivatives (assuming they exist)
//'   # Note: grgkw/hsgkw might compute derivatives of log-likelihood,
//'   # while llgkw is negative log-likelihood. Adjust signs if needed.
//'   # Assuming grgkw/hsgkw compute derivatives of NEGATIVE log-likelihood here:
//'   ana_grad <- grgkw(par = mle_par, data = sample_data)
//'   ana_hess <- hsgkw(par = mle_par, data = sample_data)
//'
//'   # Check differences (should be small if analytical functions are correct)
//'   cat("Difference between numerical and analytical gradient:\n")
//'   print(summary(abs(num_grad - ana_grad)))
//'
//'   cat("Difference between numerical and analytical Hessian:\n")
//'   print(summary(abs(num_hess - ana_hess)))
//'
//' } else {
//'    cat("\nSkipping derivative comparison.\n")
//'    cat("Requires 'numDeriv' package and functions 'grgkw', 'hsgkw'.\n")
//' }
//'
//' }
//'
//' @export
// [[Rcpp::export]]
double llgkw(const Rcpp::NumericVector& par, const Rcpp::NumericVector& data) {
 // Parameter extraction
 double alpha = par[0];   // Shape parameter α > 0
 double beta = par[1];    // Shape parameter β > 0
 double gamma = par[2];   // Shape parameter γ > 0
 double delta = par[3];   // Shape parameter δ > 0
 double lambda = par[4];  // Shape parameter λ > 0
 
 // Parameter validation - all parameters must be positive
 if (alpha <= 0 || beta <= 0 || gamma <= 0 || delta <= 0 || lambda <= 0) {
   return R_NegInf;  // Return negative infinity for invalid parameters
 }
 
 // Convert data to arma::vec for more efficient operations
 // Use aliasing (false) to avoid copying the data
 // arma::vec x(data.begin(), data.size(), false);
 arma::vec x = Rcpp::as<arma::vec>(data);
 
 // Data validation - all values must be in the range (0,1)
 if (arma::any(x <= 0) || arma::any(x >= 1)) {
   return R_NegInf;  // Return negative infinity for invalid data
 }
 
 int n = x.n_elem;  // Sample size
 
 // Calculate log of Beta function for constant term
 // Use R's lbeta function for better numerical stability
 double log_beta_term = R::lbeta(gamma, delta + 1);
 
 // Calculate the constant term: n*log(λαβ/B(γ,δ+1))
 double constant_term = n * (std::log(lambda) + std::log(alpha) + std::log(beta) - log_beta_term);
 
 // Calculate log(x) and sum (α-1)*log(x) terms
 arma::vec log_x = arma::log(x);
 double term1 = arma::sum((alpha - 1.0) * log_x);
 
 // Calculate v = 1-x^α and sum (β-1)*log(v) terms
 arma::vec x_alpha = arma::pow(x, alpha);
 arma::vec v = 1.0 - x_alpha;
 arma::vec log_v = arma::log(v);
 double term2 = arma::sum((beta - 1.0) * log_v);
 
 // Calculate w = 1-v^β = 1-(1-x^α)^β and sum (γλ-1)*log(w) terms
 arma::vec v_beta = arma::pow(v, beta);
 arma::vec w = 1.0 - v_beta;
 
 // Handle numerical stability for log(w) when w is close to zero
 arma::vec log_w(n);
 for (int i = 0; i < n; i++) {
   if (w(i) < 1e-10) {
     // Use log1p for numerical stability: log(w) = log(1-v^β) = log1p(-v^β)
     log_w(i) = std::log1p(-v_beta(i));
   } else {
     log_w(i) = std::log(w(i));
   }
 }
 double term3 = arma::sum((gamma * lambda - 1.0) * log_w);
 
 // Calculate z = 1-w^λ = 1-[1-(1-x^α)^β]^λ and sum δ*log(z) terms
 arma::vec w_lambda = arma::pow(w, lambda);
 arma::vec z = 1.0 - w_lambda;
 
 // Handle numerical stability for log(z) when z is close to zero
 arma::vec log_z(n);
 for (int i = 0; i < n; i++) {
   if (z(i) < 1e-10) {
     // Use log1p for numerical stability: log(z) = log(1-w^λ) = log1p(-w^λ)
     log_z(i) = std::log1p(-w_lambda(i));
   } else {
     log_z(i) = std::log(z(i));
   }
 }
 double term4 = arma::sum(delta * log_z);
 
 // Return final minus-log-likelihood: constant term + sum of all individual terms
 return -(constant_term + term1 + term2 + term3 + term4);
}



//' @title Gradient of the Negative Log-Likelihood for the GKw Distribution
//' @author Lopes, J. E.
//' @keywords distribution likelihood optimize gradient
//'
//' @description
//' Computes the gradient vector (vector of partial derivatives) of the negative
//' log-likelihood function for the five-parameter Generalized Kumaraswamy (GKw)
//' distribution. This provides the analytical gradient, often used for efficient
//' optimization via maximum likelihood estimation.
//'
//' @param par A numeric vector of length 5 containing the distribution parameters
//'   in the order: \code{alpha} (\eqn{\alpha > 0}), \code{beta} (\eqn{\beta > 0}),
//'   \code{gamma} (\eqn{\gamma > 0}), \code{delta} (\eqn{\delta \ge 0}),
//'   \code{lambda} (\eqn{\lambda > 0}).
//' @param data A numeric vector of observations. All values must be strictly
//'   between 0 and 1 (exclusive).
//'
//' @return Returns a numeric vector of length 5 containing the partial derivatives
//'   of the negative log-likelihood function \eqn{-\ell(\theta | \mathbf{x})} with
//'   respect to each parameter:
//'   \eqn{(-\partial \ell/\partial \alpha, -\partial \ell/\partial \beta, -\partial \ell/\partial \gamma, -\partial \ell/\partial \delta, -\partial \ell/\partial \lambda)}.
//'   Returns a vector of \code{NaN} if any parameter values are invalid according
//'   to their constraints, or if any value in \code{data} is not in the
//'   interval (0, 1).
//'
//' @details
//' The components of the gradient vector of the negative log-likelihood
//' (\eqn{-\nabla \ell(\theta | \mathbf{x})}) are:
//'
//' \deqn{
//' -\frac{\partial \ell}{\partial \alpha} = -\frac{n}{\alpha} - \sum_{i=1}^{n}\ln(x_i) +
//' \sum_{i=1}^{n}\left[x_i^{\alpha} \ln(x_i) \left(\frac{\beta-1}{v_i} -
//' \frac{(\gamma\lambda-1) \beta v_i^{\beta-1}}{w_i} +
//' \frac{\delta \lambda \beta v_i^{\beta-1} w_i^{\lambda-1}}{z_i}\right)\right]
//' }
//' \deqn{
//' -\frac{\partial \ell}{\partial \beta} = -\frac{n}{\beta} - \sum_{i=1}^{n}\ln(v_i) +
//' \sum_{i=1}^{n}\left[v_i^{\beta} \ln(v_i) \left(\frac{\gamma\lambda-1}{w_i} -
//' \frac{\delta \lambda w_i^{\lambda-1}}{z_i}\right)\right]
//' }
//' \deqn{
//' -\frac{\partial \ell}{\partial \gamma} = n[\psi(\gamma) - \psi(\gamma+\delta+1)] -
//' \lambda\sum_{i=1}^{n}\ln(w_i)
//' }
//' \deqn{
//' -\frac{\partial \ell}{\partial \delta} = n[\psi(\delta+1) - \psi(\gamma+\delta+1)] -
//' \sum_{i=1}^{n}\ln(z_i)
//' }
//' \deqn{
//' -\frac{\partial \ell}{\partial \lambda} = -\frac{n}{\lambda} -
//' \gamma\sum_{i=1}^{n}\ln(w_i) + \delta\sum_{i=1}^{n}\frac{w_i^{\lambda}\ln(w_i)}{z_i}
//' }
//'
//' where:
//' \itemize{
//'   \item \eqn{v_i = 1 - x_i^{\alpha}}
//'   \item \eqn{w_i = 1 - v_i^{\beta} = 1 - (1-x_i^{\alpha})^{\beta}}
//'   \item \eqn{z_i = 1 - w_i^{\lambda} = 1 - [1-(1-x_i^{\alpha})^{\beta}]^{\lambda}}
//'   \item \eqn{\psi(\cdot)} is the digamma function (\code{\link[base]{digamma}}).
//' }
//'
//' Numerical stability is ensured through careful implementation, including checks
//' for valid inputs and handling of intermediate calculations involving potentially
//' small or large numbers, often leveraging the Armadillo C++ library for efficiency.
//'
//' @references
//' Cordeiro, G. M., & de Castro, M. (2011). A new family of generalized
//' distributions. *Journal of Statistical Computation and Simulation*
//'
//' Kumaraswamy, P. (1980). A generalized probability density function for
//' double-bounded random processes. *Journal of Hydrology*, *46*(1-2), 79-88.
//'
//' @seealso
//' \code{\link{llgkw}} (negative log-likelihood),
//' \code{\link{hsgkw}} (Hessian matrix),
//' \code{\link{dgkw}} (density),
//' \code{\link[stats]{optim}},
//' \code{\link[numDeriv]{grad}} (for numerical gradient comparison),
//' \code{\link[base]{digamma}}
//'
//' @examples
//' \donttest{
//' # Generate sample data from a known GKw distribution
//' set.seed(123)
//' true_par <- c(alpha = 2, beta = 3, gamma = 1.0, delta = 0.5, lambda = 0.5)
//' sample_data <- rgkw(100, alpha = true_par[1], beta = true_par[2],
//'                    gamma = true_par[3], delta = true_par[4], lambda = true_par[5])
//'
//' # --- Use in Optimization (e.g., with optim using analytical gradient) ---
//' start_par <- c(1.5, 2.5, 1.2, 0.3, 0.6) # Initial guess
//'
//' # Optimization using analytical gradient
//' mle_result_gr <- stats::optim(par = start_par,
//'                               fn = llgkw,  # Objective function (Negative LL)
//'                               gr = grgkw,  # Gradient function
//'                               method = "BFGS", # Method using gradient
//'                               hessian = TRUE,
//'                               data = sample_data)
//'
//' if (mle_result_gr$convergence == 0) {
//'   print("Optimization with analytical gradient converged.")
//'   mle_par_gr <- mle_result_gr$par
//'   print("Estimated parameters:")
//'   print(mle_par_gr)
//' } else {
//'   warning("Optimization with analytical gradient failed!")
//' }
//'
//' # --- Compare analytical gradient to numerical gradient ---
//' # Requires the 'numDeriv' package
//' if (requireNamespace("numDeriv", quietly = TRUE) && mle_result_gr$convergence == 0) {
//'
//'   cat("\nComparing Gradients at MLE estimates:\n")
//'
//'   # Numerical gradient of the negative log-likelihood function
//'   num_grad <- numDeriv::grad(func = llgkw, x = mle_par_gr, data = sample_data)
//'
//'   # Analytical gradient (output of grgkw)
//'   ana_grad <- grgkw(par = mle_par_gr, data = sample_data)
//'
//'   cat("Numerical Gradient:\n")
//'   print(num_grad)
//'   cat("Analytical Gradient:\n")
//'   print(ana_grad)
//'
//'   # Check differences (should be small)
//'   cat("Max absolute difference between gradients:\n")
//'   print(max(abs(num_grad - ana_grad)))
//'
//' } else {
//'   cat("\nSkipping gradient comparison (requires 'numDeriv' package or convergence).\n")
//' }
//'
//' }
//'
//' @export
// [[Rcpp::export]]
Rcpp::NumericVector grgkw(const Rcpp::NumericVector& par, const Rcpp::NumericVector& data) {
 // Parameter extraction
 double alpha = par[0];   // Shape parameter α > 0
 double beta = par[1];    // Shape parameter β > 0
 double gamma = par[2];   // Shape parameter γ > 0
 double delta = par[3];   // Shape parameter δ > 0
 double lambda = par[4];  // Shape parameter λ > 0
 
 // Parameter validation
 if (alpha <= 0 || beta <= 0 || gamma <= 0 || delta <= 0 || lambda <= 0) {
   Rcpp::NumericVector grad(5, R_NaN);
   return grad;
 }
 
 // Data conversion and validation
 arma::vec x = Rcpp::as<arma::vec>(data);
 
 if (arma::any(x <= 0) || arma::any(x >= 1)) {
   Rcpp::NumericVector grad(5, R_NaN);
   return grad;
 }
 
 int n = x.n_elem;  // Sample size
 
 // Initialize gradient vector
 Rcpp::NumericVector grad(5, 0.0);
 
 // Small constant to avoid numerical issues
 double eps = std::numeric_limits<double>::epsilon() * 100;
 
 // Compute transformations and intermediate values
 arma::vec log_x = arma::log(x);                // log(x_i)
 arma::vec x_alpha = arma::pow(x, alpha);       // x_i^α
 arma::vec x_alpha_log_x = x_alpha % log_x;     // x_i^α * log(x_i)
 
 // v_i = 1 - x_i^α
 arma::vec v = 1.0 - x_alpha;
 v = arma::clamp(v, eps, 1.0 - eps);            // Prevent numerical issues
 
 arma::vec log_v = arma::log(v);                // log(v_i)
 arma::vec v_beta_m1 = arma::pow(v, beta - 1.0); // v_i^(β-1)
 arma::vec v_beta = arma::pow(v, beta);          // v_i^β
 arma::vec v_beta_log_v = v_beta % log_v;        // v_i^β * log(v_i)
 
 // w_i = 1 - v_i^β = 1 - (1-x_i^α)^β
 arma::vec w = 1.0 - v_beta;
 w = arma::clamp(w, eps, 1.0 - eps);            // Prevent numerical issues
 
 arma::vec log_w = arma::log(w);                // log(w_i)
 arma::vec w_lambda_m1 = arma::pow(w, lambda - 1.0); // w_i^(λ-1)
 arma::vec w_lambda = arma::pow(w, lambda);          // w_i^λ
 arma::vec w_lambda_log_w = w_lambda % log_w;        // w_i^λ * log(w_i)
 
 // z_i = 1 - w_i^λ = 1 - [1-(1-x_i^α)^β]^λ
 arma::vec z = 1.0 - w_lambda;
 z = arma::clamp(z, eps, 1.0 - eps);            // Prevent numerical issues
 
 arma::vec log_z = arma::log(z);                // log(z_i)
 
 // Calculate partial derivatives for each parameter (for log-likelihood)
 
 // ∂ℓ/∂α = n/α + Σᵢlog(xᵢ) - Σᵢ[xᵢ^α * log(xᵢ) * ((β-1)/vᵢ - (γλ-1) * β * vᵢ^(β-1) / wᵢ + δ * λ * β * vᵢ^(β-1) * wᵢ^(λ-1) / zᵢ)]
 double d_alpha = n / alpha + arma::sum(log_x);
 
 // Calculate the complex term in the α gradient
 arma::vec alpha_term2 = (beta - 1.0) / v;                // (β-1)/v_i
 arma::vec alpha_term3 = (gamma * lambda - 1.0) * beta * v_beta_m1 / w;  // (γλ-1) * β * v_i^(β-1) / w_i
 arma::vec alpha_term4 = delta * lambda * beta * v_beta_m1 % w_lambda_m1 / z;  // δ * λ * β * v_i^(β-1) * w_i^(λ-1) / z_i
 
 d_alpha -= arma::sum(x_alpha_log_x % (alpha_term2 - alpha_term3 + alpha_term4));
 
 // ∂ℓ/∂β = n/β + Σᵢlog(vᵢ) - Σᵢ[vᵢ^β * log(vᵢ) * ((γλ-1) / wᵢ - δ * λ * wᵢ^(λ-1) / zᵢ)]
 double d_beta = n / beta + arma::sum(log_v);
 
 // Calculate the complex term in the β gradient
 arma::vec beta_term2 = (gamma * lambda - 1.0) / w;       // (γλ-1) / w_i
 arma::vec beta_term3 = delta * lambda * w_lambda_m1 / z; // δ * λ * w_i^(λ-1) / z_i
 
 d_beta -= arma::sum(v_beta_log_v % (beta_term2 - beta_term3));
 
 // ∂ℓ/∂γ = -n[ψ(γ) - ψ(γ+δ+1)] + λΣᵢlog(wᵢ)
 double d_gamma = -n * (R::digamma(gamma) - R::digamma(gamma + delta + 1)) + lambda * arma::sum(log_w);
 
 // ∂ℓ/∂δ = -n[ψ(δ+1) - ψ(γ+δ+1)] + Σᵢlog(zᵢ)
 double d_delta = -n * (R::digamma(delta + 1) - R::digamma(gamma + delta + 1)) + arma::sum(log_z);
 
 // ∂ℓ/∂λ = n/λ + γΣᵢlog(wᵢ) - δΣᵢ[(wᵢ^λ*log(wᵢ))/zᵢ]
 double d_lambda = n / lambda + gamma * arma::sum(log_w) - delta * arma::sum(w_lambda_log_w / z);
 
 // Since we're optimizing negative log-likelihood, negate all derivatives
 grad[0] = -d_alpha;
 grad[1] = -d_beta;
 grad[2] = -d_gamma;
 grad[3] = -d_delta;
 grad[4] = -d_lambda;
 
 return grad;
}




//' @title Hessian Matrix of the Negative Log-Likelihood for the GKw Distribution
//' @author Lopes, J. E.
//' @keywords distribution likelihood optimize hessian
//'
//' @description
//' Computes the analytic Hessian matrix (matrix of second partial derivatives)
//' of the negative log-likelihood function for the five-parameter Generalized
//' Kumaraswamy (GKw) distribution. This is typically used to estimate standard
//' errors of maximum likelihood estimates or in optimization algorithms.
//'
//' @param par A numeric vector of length 5 containing the distribution parameters
//'   in the order: \code{alpha} (\eqn{\alpha > 0}), \code{beta} (\eqn{\beta > 0}),
//'   \code{gamma} (\eqn{\gamma > 0}), \code{delta} (\eqn{\delta \ge 0}),
//'   \code{lambda} (\eqn{\lambda > 0}).
//' @param data A numeric vector of observations. All values must be strictly
//'   between 0 and 1 (exclusive).
//'
//' @return Returns a 5x5 numeric matrix representing the Hessian matrix of the
//'   negative log-likelihood function, i.e., the matrix of second partial
//'   derivatives \eqn{-\partial^2 \ell / (\partial \theta_i \partial \theta_j)}.
//'   Returns a 5x5 matrix populated with \code{NaN} if any parameter values are
//'   invalid according to their constraints, or if any value in \code{data} is
//'   not in the interval (0, 1).
//'
//' @details
//' This function calculates the analytic second partial derivatives of the
//' negative log-likelihood function based on the GKw PDF (see \code{\link{dgkw}}).
//' The log-likelihood function \eqn{\ell(\theta | \mathbf{x})} is given by:
//' \deqn{
//' \ell(\theta) = n \ln(\lambda\alpha\beta) - n \ln B(\gamma, \delta+1)
//' + \sum_{i=1}^{n} [(\alpha-1) \ln(x_i)
//' + (\beta-1) \ln(v_i)
//' + (\gamma\lambda - 1) \ln(w_i)
//' + \delta \ln(z_i)]
//' }
//' where \eqn{\theta = (\alpha, \beta, \gamma, \delta, \lambda)}, \eqn{B(a,b)}
//' is the Beta function (\code{\link[base]{beta}}), and intermediate terms are:
//' \itemize{
//'   \item \eqn{v_i = 1 - x_i^{\alpha}}
//'   \item \eqn{w_i = 1 - v_i^{\beta} = 1 - (1-x_i^{\alpha})^{\beta}}
//'   \item \eqn{z_i = 1 - w_i^{\lambda} = 1 - [1-(1-x_i^{\alpha})^{\beta}]^{\lambda}}
//' }
//' The Hessian matrix returned contains the elements \eqn{- \frac{\partial^2 \ell(\theta | \mathbf{x})}{\partial \theta_i \partial \theta_j}}.
//'
//' Key properties of the returned matrix:
//' \itemize{
//'   \item Dimensions: 5x5.
//'   \item Symmetry: The matrix is symmetric.
//'   \item Ordering: Rows and columns correspond to the parameters in the order
//'     \eqn{\alpha, \beta, \gamma, \delta, \lambda}.
//'   \item Content: Analytic second derivatives of the *negative* log-likelihood.
//' }
//' The exact analytical formulas for the second derivatives are implemented
//' directly (often derived using symbolic differentiation) for accuracy and
//' efficiency, typically using C++.
//'
//' @references
//' Cordeiro, G. M., & de Castro, M. (2011). A new family of generalized
//' distributions. *Journal of Statistical Computation and Simulation*
//'
//' Kumaraswamy, P. (1980). A generalized probability density function for
//' double-bounded random processes. *Journal of Hydrology*, *46*(1-2), 79-88.
//' @seealso
//' \code{\link{llgkw}} (negative log-likelihood function),
//' \code{\link{grgkw}} (gradient vector),
//' \code{\link{dgkw}} (density function),
//' \code{\link[stats]{optim}},
//' \code{\link[numDeriv]{hessian}} (for numerical Hessian comparison).
//'
//' @examples
//' \donttest{
//' # Generate sample data from a known GKw distribution
//' set.seed(123)
//' true_par <- c(alpha = 2, beta = 3, gamma = 1.0, delta = 0.5, lambda = 0.5)
//' sample_data <- rgkw(100, alpha = true_par[1], beta = true_par[2],
//'                    gamma = true_par[3], delta = true_par[4], lambda = true_par[5])
//'
//' # --- Find MLE estimates (e.g., using optim) ---
//' start_par <- c(1.5, 2.5, 1.2, 0.3, 0.6) # Initial guess
//' mle_result <- stats::optim(par = start_par,
//'                            fn = llgkw,
//'                            method = "BFGS",
//'                            hessian = TRUE, # Ask optim for numerical Hessian
//'                            data = sample_data)
//'
//' if (mle_result$convergence == 0) {
//'   mle_par <- mle_result$par
//'   print("MLE parameters found:")
//'   print(mle_par)
//'
//'   # --- Compare analytical Hessian to numerical Hessian ---
//'   # Requires the 'numDeriv' package
//'   if (requireNamespace("numDeriv", quietly = TRUE)) {
//'
//'     cat("\nComparing Hessians at MLE estimates:\n")
//'
//'     # Numerical Hessian from numDeriv applied to negative LL function
//'     num_hess <- numDeriv::hessian(func = llgkw, x = mle_par, data = sample_data)
//'
//'     # Analytical Hessian (output of hsgkw)
//'     ana_hess <- hsgkw(par = mle_par, data = sample_data)
//'
//'     cat("Numerical Hessian (from numDeriv):\n")
//'     print(round(num_hess, 4))
//'     cat("Analytical Hessian (from hsgkw):\n")
//'     print(round(ana_hess, 4))
//'
//'     # Check differences (should be small)
//'     cat("Max absolute difference between Hessians:\n")
//'     print(max(abs(num_hess - ana_hess)))
//'
//'     # Optional: Use analytical Hessian to estimate standard errors
//'     # Ensure Hessian is positive definite before inverting
//'     # fisher_info_analytic <- ana_hess # Hessian of negative LL
//'     # tryCatch({
//'     #   cov_matrix <- solve(fisher_info_analytic)
//'     #   std_errors <- sqrt(diag(cov_matrix))
//'     #   cat("Std. Errors from Analytical Hessian:\n")
//'     #   print(std_errors)
//'     # }, error = function(e) {
//'     #   warning("Could not invert analytical Hessian to get standard errors: ", e$message)
//'     # })
//'
//'   } else {
//'     cat("\nSkipping Hessian comparison (requires 'numDeriv' package).\n")
//'   }
//' } else {
//'   warning("Optimization did not converge. Hessian comparison skipped.")
//' }
//'
//' }
//'
//' @export
// [[Rcpp::export]]
Rcpp::NumericMatrix hsgkw(const Rcpp::NumericVector& par, const Rcpp::NumericVector& data) {
 // Parameter extraction
 double alpha  = par[0];   // θ[0] = α
 double beta   = par[1];   // θ[1] = β
 double gamma  = par[2];   // θ[2] = γ
 double delta  = par[3];   // θ[3] = δ
 double lambda = par[4];   // θ[4] = λ
 
 // Simple parameter validation (all > 0)
 if(alpha <= 0 || beta <= 0 || gamma <= 0 || delta <= 0 || lambda <= 0) {
   Rcpp::NumericMatrix nanH(5,5);
   nanH.fill(R_NaN);
   return nanH;
 }
 
 // Data conversion and basic validation
 arma::vec x = Rcpp::as<arma::vec>(data);
 if(arma::any(x <= 0) || arma::any(x >= 1)) {
   Rcpp::NumericMatrix nanH(5,5);
   nanH.fill(R_NaN);
   return nanH;
 }
 
 int n = x.n_elem;  // sample size
 
 // Initialize Hessian matrix H (of ℓ(θ)) as 5x5
 arma::mat H(5,5, arma::fill::zeros);
 
 // --- CONSTANT TERMS (do not depend on x) ---
 // L1: n ln(λ)  => d²/dλ² = -n/λ²
 H(4,4) += - n/(lambda*lambda);
 // L2: n ln(α)  => d²/dα² = -n/α²
 H(0,0) += - n/(alpha*alpha);
 // L3: n ln(β)  => d²/dβ² = -n/β²
 H(1,1) += - n/(beta*beta);
 // L4: - n ln[B(γ, δ+1)]
 //   d²/dγ² = -n [ψ₁(γ) - ψ₁(γ+δ+1)]  where ψ₁ is the trigamma function
 H(2,2) += - n * ( R::trigamma(gamma) - R::trigamma(gamma+delta+1) );
 //   d²/dδ² = -n [ψ₁(δ+1) - ψ₁(γ+δ+1)]
 H(3,3) += - n * ( R::trigamma(delta+1) - R::trigamma(gamma+delta+1) );
 //   Mixed derivative (γ,δ): = n ψ₁(γ+δ+1)
 H(2,3) += n * R::trigamma(gamma+delta+1);
 H(3,2) = H(2,3);
 // L5: (α-1) Σ ln(x_i)  --> contributes only to first derivatives
 
 // Accumulators for mixed derivatives with λ
 double acc_gamma_lambda = 0.0;  // Sum of ln(w)
 double acc_delta_lambda = 0.0;  // Sum of dz_dlambda / z
 double acc_alpha_lambda = 0.0;  // For α,λ contributions
 double acc_beta_lambda = 0.0;   // For β,λ contributions
 
 // --- TERMS THAT INVOLVE THE OBSERVATIONS ---
 // Loop over each observation to accumulate contributions from:
 // L6: (β-1) Σ ln(v), where v = 1 - x^α
 // L7: (γλ-1) Σ ln(w), where w = 1 - v^β
 // L8: δ Σ ln(z), where z = 1 - w^λ
 for (int i = 0; i < n; i++) {
   double xi    = x(i);
   double ln_xi = std::log(xi);
   
   // -- Compute A = x^α and its derivatives --
   double A = std::pow(xi, alpha);                  // A = x^α
   double dA_dalpha = A * ln_xi;                    // dA/dα = x^α ln(x)
   double d2A_dalpha2 = A * ln_xi * ln_xi;          // d²A/dα² = x^α (ln(x))²
   
   // -- v = 1 - A and its derivatives --
   double v = 1.0 - A;                              // v = 1 - x^α
   double ln_v = std::log(v);                       // ln(v)
   double dv_dalpha = -dA_dalpha;                   // dv/dα = -dA/dα = -x^α ln(x)
   double d2v_dalpha2 = -d2A_dalpha2;               // d²v/dα² = -d²A/dα² = -x^α (ln(x))²
   
   // --- L6: (β-1) ln(v) ---
   // Second derivative w.r.t. α: (β-1)*[(d²v/dα²*v - (dv/dα)²)/v²]
   double d2L6_dalpha2 = (beta - 1.0) * ((d2v_dalpha2 * v - dv_dalpha * dv_dalpha) / (v*v));
   // Mixed derivative: d²L6/(dα dβ) = d/dβ[(β-1)*(dv_dalpha/v)] = (dv_dalpha/v)
   double d2L6_dalpha_dbeta = dv_dalpha / v;
   
   // --- L7: (γλ - 1) ln(w), where w = 1 - v^β ---
   double v_beta = std::pow(v, beta);               // v^β
   double w = 1.0 - v_beta;                         // w = 1 - v^β
   double ln_w = std::log(w);                       // ln(w)
   // Derivative of w w.r.t. v: dw/dv = -β * v^(β-1)
   double dw_dv = -beta * std::pow(v, beta - 1.0);
   // Chain rule: dw/dα = dw/dv * dv/dα
   double dw_dalpha = dw_dv * dv_dalpha;
   // Second derivative w.r.t. α for L7:
   // d²/dα² ln(w) = [d²w/dα² * w - (dw/dα)²] / w²
   // Computing d²w/dα²:
   //   dw/dα = -β * v^(β-1)*dv_dalpha,
   //   d²w/dα² = -β * [(β-1)*v^(β-2)*(dv_dalpha)² + v^(β-1)*d²v_dalpha²]
   double d2w_dalpha2 = -beta * ((beta - 1.0) * std::pow(v, beta-2.0) * (dv_dalpha * dv_dalpha)
                                   + std::pow(v, beta-1.0) * d2v_dalpha2);
   double d2L7_dalpha2 = (gamma * lambda - 1.0) * ((d2w_dalpha2 * w - (dw_dalpha * dw_dalpha)) / (w*w));
   // Derivative w.r.t. β: d/dβ ln(w). Note: d/dβ(v^β) = v^β ln(v) => d/dβ w = -v^β ln(v)
   double dw_dbeta = -v_beta * ln_v;
   // Second derivative w.r.t. β for L7:
   // d²/dβ² ln(w) = [d²w/dβ² * w - (dw/dβ)²]/w², where d²w/dβ² = -v^β (ln(v))²
   double d2w_dbeta2 = -v_beta * (ln_v * ln_v);
   double d2L7_dbeta2 = (gamma * lambda - 1.0) * ((d2w_dbeta2 * w - (dw_dbeta * dw_dbeta))/(w*w));
   // Mixed derivative L7 (α,β): d²/(dα dβ) ln(w) =
   //   = d/dβ[(dw_dalpha)/w] = (d/dβ dw_dalpha)/w - (dw_dalpha*dw_dbeta)/(w*w)
   // Approximate d/dβ dw_dalpha:
   double d_dw_dalpha_dbeta = -std::pow(v, beta-1.0) * (1.0 + beta * ln_v) * dv_dalpha;
   double d2L7_dalpha_dbeta = (gamma * lambda - 1.0) * ((d_dw_dalpha_dbeta / w) - (dw_dalpha * dw_dbeta)/(w*w));
   
   // --- L8: δ ln(z), where z = 1 - w^λ ---
   double w_lambda_val = std::pow(w, lambda);       // w^λ
   double z = 1.0 - w_lambda_val;                   // z = 1 - w^λ
   // Derivative w.r.t. α: dz/dα = -λ * w^(λ-1) * dw/dα
   double dz_dalpha = -lambda * std::pow(w, lambda-1.0) * dw_dalpha;
   // Second derivative w.r.t. α for L8:
   // d²z/dα² = -λ * [(λ-1)*w^(λ-2)*(dw/dα)² + w^(λ-1)*d²w/dα²]
   double d2z_dalpha2 = -lambda * ((lambda - 1.0) * std::pow(w, lambda-2.0) * (dw_dalpha*dw_dalpha)
                                     + std::pow(w, lambda-1.0) * d2w_dalpha2);
   double d2L8_dalpha2 = delta * ((d2z_dalpha2 * z - dz_dalpha*dz_dalpha)/(z*z));
   
   // Derivative w.r.t. β: dz/dβ = -λ * w^(λ-1) * dw/dβ
   double dz_dbeta = -lambda * std::pow(w, lambda-1.0) * dw_dbeta;
   // Second derivative w.r.t. β for L8:
   // d²z/dβ² = -λ * [(λ-1)*w^(λ-2)*(dw/dβ)² + w^(λ-1)*d²w/dβ²]
   double d2z_dbeta2 = -lambda * ((lambda - 1.0) * std::pow(w, lambda-2.0) * (dw_dbeta*dw_dbeta)
                                    + std::pow(w, lambda-1.0) * d2w_dbeta2);
   double d2L8_dbeta2 = delta * ((d2z_dbeta2 * z - dz_dbeta*dz_dbeta)/(z*z));
   
   // Mixed derivative L8 (α,β): d²/(dα dβ) ln(z)
   // = (d/dβ dz_dalpha)/z - (dz_dalpha*dz_dbeta)/(z*z)
   // Approximate d/dβ dz_dalpha = -λ * [(λ-1)*w^(λ-2)*(dw_dβ*dw_dα) + w^(λ-1)*(d/dβ dw_dalpha)]
   double d_dw_dalpha_dbeta_2 = -lambda * ((lambda - 1.0) * std::pow(w, lambda-2.0) * dw_dbeta * dw_dalpha
                                             + std::pow(w, lambda-1.0) * d_dw_dalpha_dbeta);
   double d2L8_dalpha_dbeta = delta * ((d_dw_dalpha_dbeta_2 / z) - (dz_dalpha*dz_dbeta)/(z*z));
   
   // Derivatives of L8 with respect to λ:
   // d/dλ ln(z) = (1/z)*dz/dλ, with dz/dλ = -w^λ ln(w)
   double dz_dlambda = -w_lambda_val * ln_w;
   // d²/dλ² ln(z) = [d²z/dλ² * z - (dz_dlambda)²]/z² (assuming w constant in λ)
   double d2z_dlambda2 = -w_lambda_val * (ln_w * ln_w);
   double d2L8_dlambda2 = delta * ((d2z_dlambda2 * z - dz_dlambda*dz_dlambda)/(z*z));
   
   // Mixed derivative L8 (α,λ): d²/(dα dλ) ln(z) = (d/dα dz_dλ)/z - (dz_dλ*dz_dalpha)/(z*z)
   // Correct formula: sum of two terms, not multiplication
   double d_dalpha_dz_dlambda = -std::pow(w, lambda-1.0) * dw_dalpha -
     lambda * ln_w * std::pow(w, lambda-1.0) * dw_dalpha;
   double d2L8_dalpha_dlambda = delta * ((d_dalpha_dz_dlambda / z) - (dz_dlambda*dz_dalpha)/(z*z));
   
   // Mixed derivative L8 (β,λ): d²/(dβ dλ) ln(z) = (d/dβ dz_dλ)/z - (dz_dlambda*dz_dbeta)/(z*z)
   // Correct formula: sum of two terms, not multiplication
   double d_dbeta_dz_dlambda = -std::pow(w, lambda-1.0) * dw_dbeta -
     lambda * ln_w * std::pow(w, lambda-1.0) * dw_dbeta;
   double d2L8_dbeta_dlambda = delta * ((d_dbeta_dz_dlambda / z) - (dz_dlambda*dz_dbeta)/(z*z));
   
   // --- ACCUMULATING CONTRIBUTIONS TO THE HESSIAN MATRIX ---
   // Index: 0 = α, 1 = β, 2 = γ, 3 = δ, 4 = λ
   
   // H(α,α): sum of L2, L6, L7, and L8 (constants already added)
   H(0,0) += d2L6_dalpha2 + d2L7_dalpha2 + d2L8_dalpha2;
   
   // H(α,β): mixed from L6, L7, and L8
   H(0,1) += d2L6_dalpha_dbeta + d2L7_dalpha_dbeta + d2L8_dalpha_dbeta;
   H(1,0) = H(0,1);
   
   // H(β,β): contributions from L3, L7, and L8
   H(1,1) += d2L7_dbeta2 + d2L8_dbeta2;
   
   // H(λ,λ): contains L1 and L8 (L1 already added)
   H(4,4) += d2L8_dlambda2;
   
   // H(γ,α): from L7 - derivative of ln(w) in α multiplied by λ factor of (γλ-1)
   H(2,0) += lambda * (dw_dalpha / w);
   H(0,2) = H(2,0);
   
   // H(γ,β): from L7 - derivative of ln(w) in β multiplied by λ
   H(2,1) += lambda * (dw_dbeta / w);
   H(1,2) = H(2,1);
   
   // H(δ,α): L8 - mixed derivative: d/dα ln(z)
   H(3,0) += dz_dalpha / z;
   H(0,3) = H(3,0);
   
   // H(δ,β): L8 - d/dβ ln(z)
   H(3,1) += dz_dbeta / z;
   H(1,3) = H(3,1);
   
   // Accumulating terms for mixed derivatives with λ
   // (α,λ): Term from L7 (γ contribution) + term from L8 (δ contribution)
   double term1_alpha_lambda = gamma * (dw_dalpha / w);
   double term2_alpha_lambda = d2L8_dalpha_dlambda;
   acc_alpha_lambda += term1_alpha_lambda + term2_alpha_lambda;
   
   // (β,λ): Term from L7 (γ contribution) + term from L8 (δ contribution)
   double term1_beta_lambda = gamma * (dw_dbeta / w);
   double term2_beta_lambda = d2L8_dbeta_dlambda;
   acc_beta_lambda += term1_beta_lambda + term2_beta_lambda;
   
   // (γ,λ): Contribution from L7 (γλ-1)*ln(w)
   acc_gamma_lambda += ln_w;
   
   // (δ,λ): Contribution from L8 δ*ln(z)
   acc_delta_lambda += dz_dlambda / z;
 } // end of loop
 
 // Applying mixed derivatives with λ
 // Note: All signs are positive for log-likelihood (not negative log-likelihood)
 
 // H(α,λ): Positive sign for log-likelihood
 H(0,4) = acc_alpha_lambda;
 H(4,0) = H(0,4);
 
 // H(β,λ): Positive sign for log-likelihood
 H(1,4) = acc_beta_lambda;
 H(4,1) = H(1,4);
 
 // H(γ,λ): Positive sign for log-likelihood
 H(2,4) = acc_gamma_lambda;
 H(4,2) = H(2,4);
 
 // H(δ,λ): Positive sign for log-likelihood
 H(3,4) = acc_delta_lambda;
 H(4,3) = H(3,4);
 
 // Returns the analytic Hessian matrix of the log-likelihood
 return Rcpp::wrap(-H);
}
