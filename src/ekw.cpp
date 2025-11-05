// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include "utils.h"

/*
----------------------------------------------------------------------------
EXPONENTIATED KUMARASWAMY (EKw) DISTRIBUTION
----------------------------------------------------------------------------

We interpret EKw(α, β, λ) as the GKw distribution with gamma=1 and delta=0.

* PDF:
f(x) = λ * α * β * x^(α-1) * (1 - x^α)^(β - 1) *
[1 - (1 - x^α)^β ]^(λ - 1),    for 0 < x < 1.

* CDF:
F(x) = [1 - (1 - x^α)^β ]^λ,         for 0 < x < 1.

* QUANTILE:
If p = F(x) = [1 - (1 - x^α)^β]^λ, then
p^(1/λ) = 1 - (1 - x^α)^β
(1 - x^α)^β = 1 - p^(1/λ)
x^α = 1 - [1 - p^(1/λ)]^(1/β)
x = {1 - [1 - p^(1/λ)]^(1/β)}^(1/α).

* RNG:
We can generate via the quantile method: U ~ Uniform(0,1), X= Q(U).

X = Q(U) = {1 - [1 - U^(1/λ)]^(1/β)}^(1/α).

* LOG-LIKELIHOOD:
The log-density for observation x in (0,1):
log f(x) = log(λ) + log(α) + log(β)
+ (α-1)*log(x)
+ (β-1)*log(1 - x^α)
+ (λ-1)*log(1 - (1 - x^α)^β).

Summation of log-likelihood over all x. We return negative of that for 'llekw'.*/


//' @title Density of the Exponentiated Kumaraswamy (EKw) Distribution
//'
//' @author Lopes, J. E.
//' @keywords distribution density
//'
//' @description
//' Computes the probability density function (PDF) for the Exponentiated
//' Kumaraswamy (EKw) distribution with parameters \code{alpha} (\eqn{\alpha}),
//' \code{beta} (\eqn{\beta}), and \code{lambda} (\eqn{\lambda}).
//' This distribution is defined on the interval (0, 1).
//'
//' @param x Vector of quantiles (values between 0 and 1).
//' @param alpha Shape parameter \code{alpha} > 0. Can be a scalar or a vector.
//'   Default: 1.0.
//' @param beta Shape parameter \code{beta} > 0. Can be a scalar or a vector.
//'   Default: 1.0.
//' @param lambda Shape parameter \code{lambda} > 0 (exponent parameter).
//'   Can be a scalar or a vector. Default: 1.0.
//' @param log_prob Logical; if \code{TRUE}, the logarithm of the density is
//'   returned (\eqn{\log(f(x))}). Default: \code{FALSE}.
//'
//' @return A vector of density values (\eqn{f(x)}) or log-density values
//'   (\eqn{\log(f(x))}). The length of the result is determined by the recycling
//'   rule applied to the arguments (\code{x}, \code{alpha}, \code{beta},
//'   \code{lambda}). Returns \code{0} (or \code{-Inf} if
//'   \code{log_prob = TRUE}) for \code{x} outside the interval (0, 1), or
//'   \code{NaN} if parameters are invalid (e.g., \code{alpha <= 0},
//'   \code{beta <= 0}, \code{lambda <= 0}).
//'
//' @details
//' The probability density function (PDF) of the Exponentiated Kumaraswamy (EKw)
//' distribution is given by:
//' \deqn{
//' f(x; \alpha, \beta, \lambda) = \lambda \alpha \beta x^{\alpha-1} (1 - x^\alpha)^{\beta-1} \bigl[1 - (1 - x^\alpha)^\beta \bigr]^{\lambda - 1}
//' }
//' for \eqn{0 < x < 1}.
//'
//' The EKw distribution is a special case of the five-parameter
//' Generalized Kumaraswamy (GKw) distribution (\code{\link{dgkw}}) obtained
//' by setting the parameters \eqn{\gamma = 1} and \eqn{\delta = 0}.
//' When \eqn{\lambda = 1}, the EKw distribution reduces to the standard
//' Kumaraswamy distribution.
//'
//' @references
//' Nadarajah, S., Cordeiro, G. M., & Ortega, E. M. (2012). The exponentiated
//' Kumaraswamy distribution. *Journal of the Franklin Institute*, *349*(3),
//'
//' Cordeiro, G. M., & de Castro, M. (2011). A new family of generalized
//' distributions. *Journal of Statistical Computation and Simulation*,
//'
//' Kumaraswamy, P. (1980). A generalized probability density function for
//' double-bounded random processes. *Journal of Hydrology*, *46*(1-2), 79-88.
//'
//' @seealso
//' \code{\link{dgkw}} (parent distribution density),
//' \code{\link{pekw}}, \code{\link{qekw}}, \code{\link{rekw}} (other EKw functions),
//'
//' @examples
//' \donttest{
//' # Example values
//' x_vals <- c(0.2, 0.5, 0.8)
//' alpha_par <- 2.0
//' beta_par <- 3.0
//' lambda_par <- 1.5 # Exponent parameter
//'
//' # Calculate density
//' densities <- dekw(x_vals, alpha_par, beta_par, lambda_par)
//' print(densities)
//'
//' # Calculate log-density
//' log_densities <- dekw(x_vals, alpha_par, beta_par, lambda_par, log_prob = TRUE)
//' print(log_densities)
//' # Check: should match log(densities)
//' print(log(densities))
//'
//' # Compare with dgkw setting gamma = 1, delta = 0
//' densities_gkw <- dgkw(x_vals, alpha_par, beta_par, gamma = 1.0, delta = 0.0,
//'                       lambda = lambda_par)
//' print(paste("Max difference:", max(abs(densities - densities_gkw)))) # Should be near zero
//'
//' # Plot the density for different lambda values
//' curve_x <- seq(0.01, 0.99, length.out = 200)
//' curve_y1 <- dekw(curve_x, alpha = 2, beta = 3, lambda = 0.5) # less peaked
//' curve_y2 <- dekw(curve_x, alpha = 2, beta = 3, lambda = 1.0) # standard Kw
//' curve_y3 <- dekw(curve_x, alpha = 2, beta = 3, lambda = 2.0) # more peaked
//'
//' plot(curve_x, curve_y2, type = "l", main = "EKw Density Examples (alpha=2, beta=3)",
//'      xlab = "x", ylab = "f(x)", col = "red", ylim = range(0, curve_y1, curve_y2, curve_y3))
//' lines(curve_x, curve_y1, col = "blue")
//' lines(curve_x, curve_y3, col = "green")
//' legend("topright", legend = c("lambda=0.5", "lambda=1.0 (Kw)", "lambda=2.0"),
//'        col = c("blue", "red", "green"), lty = 1, bty = "n")
//' }
//'
//' @export
// [[Rcpp::export]]
Rcpp::NumericVector dekw(
   const arma::vec& x,
   const Rcpp::NumericVector& alpha,
   const Rcpp::NumericVector& beta,
   const Rcpp::NumericVector& lambda,
   bool log_prob = false
) {
 arma::vec a_vec(alpha.begin(), alpha.size());
 arma::vec b_vec(beta.begin(), beta.size());
 arma::vec l_vec(lambda.begin(), lambda.size());
 
 size_t N = std::max({ x.n_elem, a_vec.n_elem, b_vec.n_elem, l_vec.n_elem });
 arma::vec out(N);
 out.fill(log_prob ? R_NegInf : 0.0);
 
 for (size_t i=0; i<N; i++) {
   double a = a_vec[i % a_vec.n_elem];
   double b = b_vec[i % b_vec.n_elem];
   double l = l_vec[i % l_vec.n_elem];
   double xx = x[i % x.n_elem];
   
   if (!check_ekw_pars(a, b, l)) {
     // invalid => PDF=0 or logPDF=-Inf
     continue;
   }
   // domain check
   if (xx <= 0.0 || xx >= 1.0 || !R_finite(xx)) {
     continue;
   }
   
   // log f(x) = log(lambda) + log(a) + log(b) + (a-1)*log(x)
   //            + (b-1)*log(1 - x^a)
   //            + (lambda-1)*log(1 - (1 - x^a)^b)
   double ll  = std::log(l);
   double la  = std::log(a);
   double lb  = std::log(b);
   double lx  = std::log(xx);
   
   double xalpha = a*lx; // log(x^a)
   double log_1_xalpha = log1mexp(xalpha); // log(1 - x^a)
   if (!R_finite(log_1_xalpha)) {
     continue;
   }
   
   double term2 = (b - 1.0)*log_1_xalpha; // (b-1)* log(1 - x^a)
   
   // let A= (1 - x^a)^b => logA= b*log_1_xalpha
   double logA = b*log_1_xalpha;
   double log_1_minus_A = log1mexp(logA); // log(1 - A)
   if (!R_finite(log_1_minus_A)) {
     continue;
   }
   double term3 = (l - 1.0)* log_1_minus_A;
   
   double log_pdf = ll + la + lb
   + (a - 1.0)* lx
   + term2
   + term3;
   
   if (log_prob) {
     out(i)= log_pdf;
   } else {
     out(i)= std::exp(log_pdf);
   }
 }
 
 return Rcpp::NumericVector(out.memptr(), out.memptr() + out.n_elem);
}

// -----------------------------------------------------------------------------
// 2) pekw: CDF of Exponentiated Kumaraswamy
// -----------------------------------------------------------------------------

//' @title Cumulative Distribution Function (CDF) of the EKw Distribution
//' @author Lopes, J. E.
//' @keywords distribution cumulative
//'
//' @description
//' Computes the cumulative distribution function (CDF), \eqn{P(X \le q)}, for the
//' Exponentiated Kumaraswamy (EKw) distribution with parameters \code{alpha}
//' (\eqn{\alpha}), \code{beta} (\eqn{\beta}), and \code{lambda} (\eqn{\lambda}).
//' This distribution is defined on the interval (0, 1) and is a special case
//' of the Generalized Kumaraswamy (GKw) distribution where \eqn{\gamma = 1}
//' and \eqn{\delta = 0}.
//'
//' @param q Vector of quantiles (values generally between 0 and 1).
//' @param alpha Shape parameter \code{alpha} > 0. Can be a scalar or a vector.
//'   Default: 1.0.
//' @param beta Shape parameter \code{beta} > 0. Can be a scalar or a vector.
//'   Default: 1.0.
//' @param lambda Shape parameter \code{lambda} > 0 (exponent parameter).
//'   Can be a scalar or a vector. Default: 1.0.
//' @param lower_tail Logical; if \code{TRUE} (default), probabilities are
//'   \eqn{P(X \le q)}, otherwise, \eqn{P(X > q)}.
//' @param log_p Logical; if \code{TRUE}, probabilities \eqn{p} are given as
//'   \eqn{\log(p)}. Default: \code{FALSE}.
//'
//' @return A vector of probabilities, \eqn{F(q)}, or their logarithms/complements
//'   depending on \code{lower_tail} and \code{log_p}. The length of the result
//'   is determined by the recycling rule applied to the arguments (\code{q},
//'   \code{alpha}, \code{beta}, \code{lambda}). Returns \code{0} (or \code{-Inf}
//'   if \code{log_p = TRUE}) for \code{q <= 0} and \code{1} (or \code{0} if
//'   \code{log_p = TRUE}) for \code{q >= 1}. Returns \code{NaN} for invalid
//'   parameters.
//'
//' @details
//' The Exponentiated Kumaraswamy (EKw) distribution is a special case of the
//' five-parameter Generalized Kumaraswamy distribution (\code{\link{pgkw}})
//' obtained by setting parameters \eqn{\gamma = 1} and \eqn{\delta = 0}.
//'
//' The CDF of the GKw distribution is \eqn{F_{GKw}(q) = I_{y(q)}(\gamma, \delta+1)},
//' where \eqn{y(q) = [1-(1-q^{\alpha})^{\beta}]^{\lambda}} and \eqn{I_x(a,b)}
//' is the regularized incomplete beta function (\code{\link[stats]{pbeta}}).
//' Setting \eqn{\gamma=1} and \eqn{\delta=0} gives \eqn{I_{y(q)}(1, 1)}. Since
//' \eqn{I_x(1, 1) = x}, the CDF simplifies to \eqn{y(q)}:
//' \deqn{
//' F(q; \alpha, \beta, \lambda) = \bigl[1 - (1 - q^\alpha)^\beta \bigr]^\lambda
//' }
//' for \eqn{0 < q < 1}.
//' The implementation uses this closed-form expression for efficiency and handles
//' \code{lower_tail} and \code{log_p} arguments appropriately.
//'
//' @references
//' Nadarajah, S., Cordeiro, G. M., & Ortega, E. M. (2012). The exponentiated
//' Kumaraswamy distribution. *Journal of the Franklin Institute*, *349*(3),
//'
//'
//' Cordeiro, G. M., & de Castro, M. (2011). A new family of generalized
//' distributions. *Journal of Statistical Computation and Simulation*,
//'
//' Kumaraswamy, P. (1980). A generalized probability density function for
//' double-bounded random processes. *Journal of Hydrology*, *46*(1-2), 79-88.
//'
//'
//' @seealso
//' \code{\link{pgkw}} (parent distribution CDF),
//' \code{\link{dekw}}, \code{\link{qekw}}, \code{\link{rekw}} (other EKw functions),
//'
//' @examples
//' \donttest{
//' # Example values
//' q_vals <- c(0.2, 0.5, 0.8)
//' alpha_par <- 2.0
//' beta_par <- 3.0
//' lambda_par <- 1.5
//'
//' # Calculate CDF P(X <= q)
//' probs <- pekw(q_vals, alpha_par, beta_par, lambda_par)
//' print(probs)
//'
//' # Calculate upper tail P(X > q)
//' probs_upper <- pekw(q_vals, alpha_par, beta_par, lambda_par,
//'                     lower_tail = FALSE)
//' print(probs_upper)
//' # Check: probs + probs_upper should be 1
//' print(probs + probs_upper)
//'
//' # Calculate log CDF
//' log_probs <- pekw(q_vals, alpha_par, beta_par, lambda_par, log_p = TRUE)
//' print(log_probs)
//' # Check: should match log(probs)
//' print(log(probs))
//'
//' # Compare with pgkw setting gamma = 1, delta = 0
//' probs_gkw <- pgkw(q_vals, alpha_par, beta_par, gamma = 1.0, delta = 0.0,
//'                  lambda = lambda_par)
//' print(paste("Max difference:", max(abs(probs - probs_gkw)))) # Should be near zero
//'
//' # Plot the CDF for different lambda values
//' curve_q <- seq(0.01, 0.99, length.out = 200)
//' curve_p1 <- pekw(curve_q, alpha = 2, beta = 3, lambda = 0.5)
//' curve_p2 <- pekw(curve_q, alpha = 2, beta = 3, lambda = 1.0) # standard Kw
//' curve_p3 <- pekw(curve_q, alpha = 2, beta = 3, lambda = 2.0)
//'
//' plot(curve_q, curve_p2, type = "l", main = "EKw CDF Examples (alpha=2, beta=3)",
//'      xlab = "q", ylab = "F(q)", col = "red", ylim = c(0, 1))
//' lines(curve_q, curve_p1, col = "blue")
//' lines(curve_q, curve_p3, col = "green")
//' legend("bottomright", legend = c("lambda=0.5", "lambda=1.0 (Kw)", "lambda=2.0"),
//'        col = c("blue", "red", "green"), lty = 1, bty = "n")
//' }
//'
//' @export
// [[Rcpp::export]]
Rcpp::NumericVector pekw(
   const arma::vec& q,
   const Rcpp::NumericVector& alpha,
   const Rcpp::NumericVector& beta,
   const Rcpp::NumericVector& lambda,
   bool lower_tail = true,
   bool log_p = false
) {
 arma::vec a_vec(alpha.begin(), alpha.size());
 arma::vec b_vec(beta.begin(), beta.size());
 arma::vec l_vec(lambda.begin(), lambda.size());
 
 size_t N = std::max({ q.n_elem, a_vec.n_elem, b_vec.n_elem, l_vec.n_elem });
 arma::vec out(N);
 
 for (size_t i=0; i<N; i++) {
   double a = a_vec[i % a_vec.n_elem];
   double b = b_vec[i % b_vec.n_elem];
   double l = l_vec[i % l_vec.n_elem];
   double xx = q[i % q.n_elem];
   
   if (!check_ekw_pars(a, b, l)) {
     out(i)= NA_REAL;
     continue;
   }
   
   // boundary
   if (!R_finite(xx) || xx <= 0.0) {
     double val0 = (lower_tail ? 0.0 : 1.0);
     out(i) = (log_p ? std::log(val0) : val0);
     continue;
   }
   if (xx >= 1.0) {
     double val1 = (lower_tail ? 1.0 : 0.0);
     out(i) = (log_p ? std::log(val1) : val1);
     continue;
   }
   
   // F(x)= [1 - (1 - x^a)^b]^lambda
   double lx = std::log(xx);
   double xalpha = std::exp(a*lx);
   double omx = 1.0 - xalpha;         // (1 - x^α)
   if (omx <= 0.0) {
     // => F=1
     double val1 = (lower_tail ? 1.0 : 0.0);
     out(i) = (log_p ? std::log(val1) : val1);
     continue;
   }
   double t = 1.0 - std::pow(omx, b);
   if (t <= 0.0) {
     // => F=0
     double val0 = (lower_tail ? 0.0 : 1.0);
     out(i) = (log_p ? std::log(val0) : val0);
     continue;
   }
   if (t >= 1.0) {
     // => F=1
     double val1 = (lower_tail ? 1.0 : 0.0);
     out(i) = (log_p ? std::log(val1) : val1);
     continue;
   }
   double val = std::pow(t, l);
   // F(x)=val => if not lower tail => 1-val
   if (!lower_tail) {
     val = 1.0 - val;
   }
   if (log_p) {
     val = std::log(val);
   }
   out(i) = val;
 }
 
 return Rcpp::NumericVector(out.memptr(), out.memptr() + out.n_elem);
}


// -----------------------------------------------------------------------------
// 3) qekw: Quantile of Exponentiated Kumaraswamy
// -----------------------------------------------------------------------------

//' @title Quantile Function of the Exponentiated Kumaraswamy (EKw) Distribution
//' @author Lopes, J. E.
//' @keywords distribution quantile
//'
//' @description
//' Computes the quantile function (inverse CDF) for the Exponentiated
//' Kumaraswamy (EKw) distribution with parameters \code{alpha} (\eqn{\alpha}),
//' \code{beta} (\eqn{\beta}), and \code{lambda} (\eqn{\lambda}).
//' It finds the value \code{q} such that \eqn{P(X \le q) = p}. This distribution
//' is a special case of the Generalized Kumaraswamy (GKw) distribution where
//' \eqn{\gamma = 1} and \eqn{\delta = 0}.
//'
//' @param p Vector of probabilities (values between 0 and 1).
//' @param alpha Shape parameter \code{alpha} > 0. Can be a scalar or a vector.
//'   Default: 1.0.
//' @param beta Shape parameter \code{beta} > 0. Can be a scalar or a vector.
//'   Default: 1.0.
//' @param lambda Shape parameter \code{lambda} > 0 (exponent parameter).
//'   Can be a scalar or a vector. Default: 1.0.
//' @param lower_tail Logical; if \code{TRUE} (default), probabilities are \eqn{p = P(X \le q)},
//'   otherwise, probabilities are \eqn{p = P(X > q)}.
//' @param log_p Logical; if \code{TRUE}, probabilities \code{p} are given as
//'   \eqn{\log(p)}. Default: \code{FALSE}.
//'
//' @return A vector of quantiles corresponding to the given probabilities \code{p}.
//'   The length of the result is determined by the recycling rule applied to
//'   the arguments (\code{p}, \code{alpha}, \code{beta}, \code{lambda}).
//'   Returns:
//'   \itemize{
//'     \item \code{0} for \code{p = 0} (or \code{p = -Inf} if \code{log_p = TRUE},
//'           when \code{lower_tail = TRUE}).
//'     \item \code{1} for \code{p = 1} (or \code{p = 0} if \code{log_p = TRUE},
//'           when \code{lower_tail = TRUE}).
//'     \item \code{NaN} for \code{p < 0} or \code{p > 1} (or corresponding log scale).
//'     \item \code{NaN} for invalid parameters (e.g., \code{alpha <= 0},
//'           \code{beta <= 0}, \code{lambda <= 0}).
//'   }
//'   Boundary return values are adjusted accordingly for \code{lower_tail = FALSE}.
//'
//' @details
//' The quantile function \eqn{Q(p)} is the inverse of the CDF \eqn{F(q)}. The CDF
//' for the EKw (\eqn{\gamma=1, \delta=0}) distribution is \eqn{F(q) = [1 - (1 - q^\alpha)^\beta ]^\lambda}
//' (see \code{\link{pekw}}). Inverting this equation for \eqn{q} yields the
//' quantile function:
//' \deqn{
//' Q(p) = \left\{ 1 - \left[ 1 - p^{1/\lambda} \right]^{1/\beta} \right\}^{1/\alpha}
//' }
//' The function uses this closed-form expression and correctly handles the
//' \code{lower_tail} and \code{log_p} arguments by transforming \code{p}
//' appropriately before applying the formula. This is equivalent to the general
//' GKw quantile function (\code{\link{qgkw}}) evaluated with \eqn{\gamma=1, \delta=0}.
//'
//' @references
//' Nadarajah, S., Cordeiro, G. M., & Ortega, E. M. (2012). The exponentiated
//' Kumaraswamy distribution. *Journal of the Franklin Institute*, *349*(3),
//'
//'
//' Cordeiro, G. M., & de Castro, M. (2011). A new family of generalized
//' distributions. *Journal of Statistical Computation and Simulation*,
//'
//' Kumaraswamy, P. (1980). A generalized probability density function for
//' double-bounded random processes. *Journal of Hydrology*, *46*(1-2), 79-88.
//'
//'
//' @seealso
//' \code{\link{qgkw}} (parent distribution quantile function),
//' \code{\link{dekw}}, \code{\link{pekw}}, \code{\link{rekw}} (other EKw functions),
//' \code{\link[stats]{qunif}}
//'
//' @examples
//' \donttest{
//' # Example values
//' p_vals <- c(0.1, 0.5, 0.9)
//' alpha_par <- 2.0
//' beta_par <- 3.0
//' lambda_par <- 1.5
//'
//' # Calculate quantiles
//' quantiles <- qekw(p_vals, alpha_par, beta_par, lambda_par)
//' print(quantiles)
//'
//' # Calculate quantiles for upper tail probabilities P(X > q) = p
//' quantiles_upper <- qekw(p_vals, alpha_par, beta_par, lambda_par,
//'                         lower_tail = FALSE)
//' print(quantiles_upper)
//' # Check: qekw(p, ..., lt=F) == qekw(1-p, ..., lt=T)
//' print(qekw(1 - p_vals, alpha_par, beta_par, lambda_par))
//'
//' # Calculate quantiles from log probabilities
//' log_p_vals <- log(p_vals)
//' quantiles_logp <- qekw(log_p_vals, alpha_par, beta_par, lambda_par,
//'                        log_p = TRUE)
//' print(quantiles_logp)
//' # Check: should match original quantiles
//' print(quantiles)
//'
//' # Compare with qgkw setting gamma = 1, delta = 0
//' quantiles_gkw <- qgkw(p_vals, alpha = alpha_par, beta = beta_par,
//'                      gamma = 1.0, delta = 0.0, lambda = lambda_par)
//' print(paste("Max difference:", max(abs(quantiles - quantiles_gkw)))) # Should be near zero
//'
//' # Verify inverse relationship with pekw
//' p_check <- 0.75
//' q_calc <- qekw(p_check, alpha_par, beta_par, lambda_par)
//' p_recalc <- pekw(q_calc, alpha_par, beta_par, lambda_par)
//' print(paste("Original p:", p_check, " Recalculated p:", p_recalc))
//' # abs(p_check - p_recalc) < 1e-9 # Should be TRUE
//'
//' # Boundary conditions
//' print(qekw(c(0, 1), alpha_par, beta_par, lambda_par)) # Should be 0, 1
//' print(qekw(c(-Inf, 0), alpha_par, beta_par, lambda_par, log_p = TRUE)) # Should be 0, 1
//' }
//'
//' @export
// [[Rcpp::export]]
Rcpp::NumericVector qekw(
   const arma::vec& p,
   const Rcpp::NumericVector& alpha,
   const Rcpp::NumericVector& beta,
   const Rcpp::NumericVector& lambda,
   bool lower_tail = true,
   bool log_p = false
) {
 arma::vec a_vec(alpha.begin(), alpha.size());
 arma::vec b_vec(beta.begin(), beta.size());
 arma::vec l_vec(lambda.begin(), lambda.size());
 
 size_t N = std::max({ p.n_elem, a_vec.n_elem, b_vec.n_elem, l_vec.n_elem });
 arma::vec out(N);
 
 for (size_t i=0; i<N; i++){
   double a = a_vec[i % a_vec.n_elem];
   double b = b_vec[i % b_vec.n_elem];
   double l = l_vec[i % l_vec.n_elem];
   double pp = p[i % p.n_elem];
   
   if (!check_ekw_pars(a, b, l)) {
     out(i) = NA_REAL;
     continue;
   }
   
   // handle log_p
   if (log_p) {
     if (pp > 0.0) {
       // log(p)>0 => p>1 => invalid
       out(i) = NA_REAL;
       continue;
     }
     pp = std::exp(pp);
   }
   // handle tail
   if (!lower_tail) {
     pp = 1.0 - pp;
   }
   
   // boundaries
   if (pp <= 0.0) {
     out(i) = 0.0;
     continue;
   }
   if (pp >= 1.0) {
     out(i) = 1.0;
     continue;
   }
   
   // Q(p)= {1 - [1 - p^(1/λ)]^(1/β)}^(1/α)
   double step1 = std::pow(pp, 1.0/l);          // p^(1/λ)
   double step2 = 1.0 - step1;                  // 1 - p^(1/λ)
   if (step2 < 0.0) step2 = 0.0;
   double step3 = std::pow(step2, 1.0/b);       // [1 - p^(1/λ)]^(1/β)
   double step4 = 1.0 - step3;                  // 1 - ...
   if (step4 < 0.0) step4 = 0.0;
   
   double x;
   if (a == 1.0) {
     x = step4;
   } else {
     x = std::pow(step4, 1.0/a);
     if (x < 0.0) x = 0.0;
     if (x > 1.0) x = 1.0;
   }
   
   out(i) = x;
 }
 
 return Rcpp::NumericVector(out.memptr(), out.memptr() + out.n_elem);
}


// -----------------------------------------------------------------------------
// 4) rekw: RNG for Exponentiated Kumaraswamy
// -----------------------------------------------------------------------------

//' @title Random Number Generation for the Exponentiated Kumaraswamy (EKw) Distribution
//' @author Lopes, J. E.
//' @keywords distribution random
//'
//' @description
//' Generates random deviates from the Exponentiated Kumaraswamy (EKw)
//' distribution with parameters \code{alpha} (\eqn{\alpha}), \code{beta}
//' (\eqn{\beta}), and \code{lambda} (\eqn{\lambda}). This distribution is a
//' special case of the Generalized Kumaraswamy (GKw) distribution where
//' \eqn{\gamma = 1} and \eqn{\delta = 0}.
//'
//' @param n Number of observations. If \code{length(n) > 1}, the length is
//'   taken to be the number required. Must be a non-negative integer.
//' @param alpha Shape parameter \code{alpha} > 0. Can be a scalar or a vector.
//'   Default: 1.0.
//' @param beta Shape parameter \code{beta} > 0. Can be a scalar or a vector.
//'   Default: 1.0.
//' @param lambda Shape parameter \code{lambda} > 0 (exponent parameter).
//'   Can be a scalar or a vector. Default: 1.0.
//'
//' @return A vector of length \code{n} containing random deviates from the EKw
//'   distribution. The length of the result is determined by \code{n} and the
//'   recycling rule applied to the parameters (\code{alpha}, \code{beta},
//'   \code{lambda}). Returns \code{NaN} if parameters
//'   are invalid (e.g., \code{alpha <= 0}, \code{beta <= 0}, \code{lambda <= 0}).
//'
//' @details
//' The generation method uses the inverse transform (quantile) method.
//' That is, if \eqn{U} is a random variable following a standard Uniform
//' distribution on (0, 1), then \eqn{X = Q(U)} follows the EKw distribution,
//' where \eqn{Q(u)} is the EKw quantile function (\code{\link{qekw}}):
//' \deqn{
//' Q(u) = \left\{ 1 - \left[ 1 - u^{1/\lambda} \right]^{1/\beta} \right\}^{1/\alpha}
//' }
//' This is computationally equivalent to the general GKw generation method
//' (\code{\link{rgkw}}) when specialized for \eqn{\gamma=1, \delta=0}, as the
//' required Beta(1, 1) random variate is equivalent to a standard Uniform(0, 1)
//' variate. The implementation generates \eqn{U} using \code{\link[stats]{runif}}
//' and applies the transformation above.
//'
//' @references
//' Nadarajah, S., Cordeiro, G. M., & Ortega, E. M. (2012). The exponentiated
//' Kumaraswamy distribution. *Journal of the Franklin Institute*, *349*(3),
//'
//'
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
//' \code{\link{dekw}}, \code{\link{pekw}}, \code{\link{qekw}} (other EKw functions),
//' \code{\link[stats]{runif}}
//'
//' @examples
//' \donttest{
//' set.seed(2027) # for reproducibility
//'
//' # Generate 1000 random values from a specific EKw distribution
//' alpha_par <- 2.0
//' beta_par <- 3.0
//' lambda_par <- 1.5
//'
//' x_sample_ekw <- rekw(1000, alpha = alpha_par, beta = beta_par, lambda = lambda_par)
//' summary(x_sample_ekw)
//'
//' # Histogram of generated values compared to theoretical density
//' hist(x_sample_ekw, breaks = 30, freq = FALSE, # freq=FALSE for density
//'      main = "Histogram of EKw Sample", xlab = "x", ylim = c(0, 3.0))
//' curve(dekw(x, alpha = alpha_par, beta = beta_par, lambda = lambda_par),
//'       add = TRUE, col = "red", lwd = 2, n = 201)
//' legend("topright", legend = "Theoretical PDF", col = "red", lwd = 2, bty = "n")
//'
//' # Comparing empirical and theoretical quantiles (Q-Q plot)
//' prob_points <- seq(0.01, 0.99, by = 0.01)
//' theo_quantiles <- qekw(prob_points, alpha = alpha_par, beta = beta_par,
//'                        lambda = lambda_par)
//' emp_quantiles <- quantile(x_sample_ekw, prob_points, type = 7)
//'
//' plot(theo_quantiles, emp_quantiles, pch = 16, cex = 0.8,
//'      main = "Q-Q Plot for EKw Distribution",
//'      xlab = "Theoretical Quantiles", ylab = "Empirical Quantiles (n=1000)")
//' abline(a = 0, b = 1, col = "blue", lty = 2)
//'
//' # Compare summary stats with rgkw(..., gamma=1, delta=0, ...)
//' # Note: individual values will differ due to randomness
//' x_sample_gkw <- rgkw(1000, alpha = alpha_par, beta = beta_par, gamma = 1.0,
//'                      delta = 0.0, lambda = lambda_par)
//' print("Summary stats for rekw sample:")
//' print(summary(x_sample_ekw))
//' print("Summary stats for rgkw(gamma=1, delta=0) sample:")
//' print(summary(x_sample_gkw)) # Should be similar
//'
//' }
//'
//' @export
// [[Rcpp::export]]
Rcpp::NumericVector rekw(
   int n,
   const Rcpp::NumericVector& alpha,
   const Rcpp::NumericVector& beta,
   const Rcpp::NumericVector& lambda
) {
 if (n <= 0) {
   Rcpp::stop("rekw: n must be positive");
 }
 
 arma::vec a_vec(alpha.begin(), alpha.size());
 arma::vec b_vec(beta.begin(), beta.size());
 arma::vec l_vec(lambda.begin(), lambda.size());
 
 size_t k = std::max({ a_vec.n_elem, b_vec.n_elem, l_vec.n_elem });
 arma::vec out(n);
 
 for (int i=0; i<n; i++){
   size_t idx = i % k;
   double a = a_vec[idx % a_vec.n_elem];
   double b = b_vec[idx % b_vec.n_elem];
   double l = l_vec[idx % l_vec.n_elem];
   
   if (!check_ekw_pars(a, b, l)) {
     out(i) = NA_REAL;
     Rcpp::warning("rekw: invalid parameters at index %d", i+1);
     continue;
   }
   
   double U = R::runif(0.0, 1.0);
   // X = Q(U)
   double step1 = std::pow(U, 1.0/l);
   double step2 = 1.0 - step1;
   if (step2 < 0.0) step2 = 0.0;
   double step3 = std::pow(step2, 1.0/b);
   double step4 = 1.0 - step3;
   if (step4 < 0.0) step4 = 0.0;
   
   double x;
   if (a == 1.0) {
     x = step4;
   } else {
     x = std::pow(step4, 1.0/a);
     if (!R_finite(x) || x < 0.0) x = 0.0;
     if (x > 1.0) x = 1.0;
   }
   
   out(i) = x;
 }
 
 return Rcpp::NumericVector(out.memptr(), out.memptr() + out.n_elem);
}


// -----------------------------------------------------------------------------
// 5) llekw: Negative Log-Likelihood of EKw
// -----------------------------------------------------------------------------



//' @title Negative Log-Likelihood for the Exponentiated Kumaraswamy (EKw) Distribution
//' @author Lopes, J. E.
//' @keywords distribution likelihood optimize
//'
//' @description
//' Computes the negative log-likelihood function for the Exponentiated
//' Kumaraswamy (EKw) distribution with parameters \code{alpha} (\eqn{\alpha}),
//' \code{beta} (\eqn{\beta}), and \code{lambda} (\eqn{\lambda}), given a vector
//' of observations. This distribution is the special case of the Generalized
//' Kumaraswamy (GKw) distribution where \eqn{\gamma = 1} and \eqn{\delta = 0}.
//' This function is suitable for maximum likelihood estimation.
//'
//' @param par A numeric vector of length 3 containing the distribution parameters
//'   in the order: \code{alpha} (\eqn{\alpha > 0}), \code{beta} (\eqn{\beta > 0}),
//'   \code{lambda} (\eqn{\lambda > 0}).
//' @param data A numeric vector of observations. All values must be strictly
//'   between 0 and 1 (exclusive).
//'
//' @return Returns a single \code{double} value representing the negative
//'   log-likelihood (\eqn{-\ell(\theta|\mathbf{x})}). Returns \code{Inf}
//'   if any parameter values in \code{par} are invalid according to their
//'   constraints, or if any value in \code{data} is not in the interval (0, 1).
//'
//' @details
//' The Exponentiated Kumaraswamy (EKw) distribution is the GKw distribution
//' (\code{\link{dekw}}) with \eqn{\gamma=1} and \eqn{\delta=0}. Its probability
//' density function (PDF) is:
//' \deqn{
//' f(x | \theta) = \lambda \alpha \beta x^{\alpha-1} (1 - x^\alpha)^{\beta-1} \bigl[1 - (1 - x^\alpha)^\beta \bigr]^{\lambda - 1}
//' }
//' for \eqn{0 < x < 1} and \eqn{\theta = (\alpha, \beta, \lambda)}.
//' The log-likelihood function \eqn{\ell(\theta | \mathbf{x})} for a sample
//' \eqn{\mathbf{x} = (x_1, \dots, x_n)} is \eqn{\sum_{i=1}^n \ln f(x_i | \theta)}:
//' \deqn{
//' \ell(\theta | \mathbf{x}) = n[\ln(\lambda) + \ln(\alpha) + \ln(\beta)]
//' + \sum_{i=1}^{n} [(\alpha-1)\ln(x_i) + (\beta-1)\ln(v_i) + (\lambda-1)\ln(w_i)]
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
//' Nadarajah, S., Cordeiro, G. M., & Ortega, E. M. (2012). The exponentiated
//' Kumaraswamy distribution. *Journal of the Franklin Institute*, *349*(3),
//'
//'
//' Cordeiro, G. M., & de Castro, M. (2011). A new family of generalized
//' distributions. *Journal of Statistical Computation and Simulation*,
//'
//' Kumaraswamy, P. (1980). A generalized probability density function for
//' double-bounded random processes. *Journal of Hydrology*, *46*(1-2), 79-88.
//'
//'
//' @seealso
//' \code{\link{llgkw}} (parent distribution negative log-likelihood),
//' \code{\link{dekw}}, \code{\link{pekw}}, \code{\link{qekw}}, \code{\link{rekw}},
//' \code{grekw} (gradient, if available),
//' \code{hsekw} (Hessian, if available),
//' \code{\link[stats]{optim}}
//'
//' @examples
//' \donttest{
//' ## Example 1: Basic Log-Likelihood Evaluation
//' 
//' par_ <- par()
//' 
//' # Generate sample data
//' set.seed(123)
//' n <- 1000
//' true_params <- c(alpha = 2.5, beta = 3.5, lambda = 2.0)
//' data <- rekw(n, alpha = true_params[1], beta = true_params[2],
//'              lambda = true_params[3])
//' 
//' # Evaluate negative log-likelihood at true parameters
//' nll_true <- llekw(par = true_params, data = data)
//' cat("Negative log-likelihood at true parameters:", nll_true, "\n")
//' 
//' # Evaluate at different parameter values
//' test_params <- rbind(
//'   c(2.0, 3.0, 1.5),
//'   c(2.5, 3.5, 2.0),
//'   c(3.0, 4.0, 2.5)
//' )
//' 
//' nll_values <- apply(test_params, 1, function(p) llekw(p, data))
//' results <- data.frame(
//'   Alpha = test_params[, 1],
//'   Beta = test_params[, 2],
//'   Lambda = test_params[, 3],
//'   NegLogLik = nll_values
//' )
//' print(results, digits = 4)
//' 
//' 
//' ## Example 2: Maximum Likelihood Estimation
//' 
//' # Optimization using BFGS with analytical gradient
//' fit <- optim(
//'   par = c(2, 3, 1.5),
//'   fn = llekw,
//'   gr = grekw,
//'   data = data,
//'   method = "BFGS",
//'   hessian = TRUE
//' )
//' 
//' mle <- fit$par
//' names(mle) <- c("alpha", "beta", "lambda")
//' se <- sqrt(diag(solve(fit$hessian)))
//' 
//' results <- data.frame(
//'   Parameter = c("alpha", "beta", "lambda"),
//'   True = true_params,
//'   MLE = mle,
//'   SE = se,
//'   CI_Lower = mle - 1.96 * se,
//'   CI_Upper = mle + 1.96 * se
//' )
//' print(results, digits = 4)
//' 
//' cat("\nNegative log-likelihood at MLE:", fit$value, "\n")
//' cat("AIC:", 2 * fit$value + 2 * length(mle), "\n")
//' cat("BIC:", 2 * fit$value + length(mle) * log(n), "\n")
//' 
//' 
//' ## Example 3: Comparing Optimization Methods
//' 
//' methods <- c("BFGS", "L-BFGS-B", "Nelder-Mead", "CG")
//' start_params <- c(2, 3, 1.5)
//' 
//' comparison <- data.frame(
//'   Method = character(),
//'   Alpha = numeric(),
//'   Beta = numeric(),
//'   Lambda = numeric(),
//'   NegLogLik = numeric(),
//'   Convergence = integer(),
//'   stringsAsFactors = FALSE
//' )
//' 
//' for (method in methods) {
//'   if (method %in% c("BFGS", "CG")) {
//'     fit_temp <- optim(
//'       par = start_params,
//'       fn = llekw,
//'       gr = grekw,
//'       data = data,
//'       method = method
//'     )
//'   } else if (method == "L-BFGS-B") {
//'     fit_temp <- optim(
//'       par = start_params,
//'       fn = llekw,
//'       gr = grekw,
//'       data = data,
//'       method = method,
//'       lower = c(0.01, 0.01, 0.01),
//'       upper = c(100, 100, 100)
//'     )
//'   } else {
//'     fit_temp <- optim(
//'       par = start_params,
//'       fn = llekw,
//'       data = data,
//'       method = method
//'     )
//'   }
//' 
//'   comparison <- rbind(comparison, data.frame(
//'     Method = method,
//'     Alpha = fit_temp$par[1],
//'     Beta = fit_temp$par[2],
//'     Lambda = fit_temp$par[3],
//'     NegLogLik = fit_temp$value,
//'     Convergence = fit_temp$convergence,
//'     stringsAsFactors = FALSE
//'   ))
//' }
//' 
//' print(comparison, digits = 4, row.names = FALSE)
//' 
//' 
//' ## Example 4: Likelihood Ratio Test
//' 
//' # Test H0: lambda = 2 vs H1: lambda free
//' loglik_full <- -fit$value
//' 
//' restricted_ll <- function(params_restricted, data, lambda_fixed) {
//'   llekw(par = c(params_restricted[1], params_restricted[2],
//'                 lambda_fixed), data = data)
//' }
//' 
//' fit_restricted <- optim(
//'   par = c(mle[1], mle[2]),
//'   fn = restricted_ll,
//'   data = data,
//'   lambda_fixed = 2,
//'   method = "BFGS"
//' )
//' 
//' loglik_restricted <- -fit_restricted$value
//' lr_stat <- 2 * (loglik_full - loglik_restricted)
//' p_value <- pchisq(lr_stat, df = 1, lower.tail = FALSE)
//' 
//' cat("LR Statistic:", round(lr_stat, 4), "\n")
//' cat("P-value:", format.pval(p_value, digits = 4), "\n")
//' 
//' 
//' ## Example 5: Univariate Profile Likelihoods
//' 
//' # Profile for alpha
//' alpha_grid <- seq(mle[1] - 1, mle[1] + 1, length.out = 50)
//' alpha_grid <- alpha_grid[alpha_grid > 0]
//' profile_ll_alpha <- numeric(length(alpha_grid))
//' 
//' for (i in seq_along(alpha_grid)) {
//'   profile_fit <- optim(
//'     par = mle[-1],
//'     fn = function(p) llekw(c(alpha_grid[i], p), data),
//'     method = "BFGS"
//'   )
//'   profile_ll_alpha[i] <- -profile_fit$value
//' }
//' 
//' # Profile for beta
//' beta_grid <- seq(mle[2] - 1, mle[2] + 1, length.out = 50)
//' beta_grid <- beta_grid[beta_grid > 0]
//' profile_ll_beta <- numeric(length(beta_grid))
//' 
//' for (i in seq_along(beta_grid)) {
//'   profile_fit <- optim(
//'     par = mle[-2],
//'     fn = function(p) llekw(c(p[1], beta_grid[i], p[2]), data),
//'     method = "BFGS"
//'   )
//'   profile_ll_beta[i] <- -profile_fit$value
//' }
//' 
//' # Profile for lambda
//' lambda_grid <- seq(mle[3] - 1, mle[3] + 1, length.out = 50)
//' lambda_grid <- lambda_grid[lambda_grid > 0]
//' profile_ll_lambda <- numeric(length(lambda_grid))
//' 
//' for (i in seq_along(lambda_grid)) {
//'   profile_fit <- optim(
//'     par = mle[-3],
//'     fn = function(p) llekw(c(p[1], p[2], lambda_grid[i]), data),
//'     method = "BFGS"
//'   )
//'   profile_ll_lambda[i] <- -profile_fit$value
//' }
//' 
//' # 95% confidence threshold
//' chi_crit <- qchisq(0.95, df = 1)
//' threshold <- max(profile_ll_alpha) - chi_crit / 2
//' 
//' # Plot all profiles
//' par(mfrow = c(1, 3), mar = c(4, 4, 3, 1))
//' 
//' plot(alpha_grid, profile_ll_alpha, type = "l", lwd = 2, col = "#2E4057",
//'      xlab = expression(alpha), ylab = "Profile Log-Likelihood",
//'      main = expression(paste("Profile: ", alpha)), las = 1)
//' abline(v = mle[1], col = "#8B0000", lty = 2, lwd = 2)
//' abline(v = true_params[1], col = "#006400", lty = 2, lwd = 2)
//' abline(h = threshold, col = "#808080", lty = 3, lwd = 1.5)
//' legend("topright", legend = c("MLE", "True", "95% CI"),
//'        col = c("#8B0000", "#006400", "#808080"),
//'        lty = c(2, 2, 3), lwd = 2, bty = "n", cex = 0.8)
//' grid(col = "gray90")
//' 
//' plot(beta_grid, profile_ll_beta, type = "l", lwd = 2, col = "#2E4057",
//'      xlab = expression(beta), ylab = "Profile Log-Likelihood",
//'      main = expression(paste("Profile: ", beta)), las = 1)
//' abline(v = mle[2], col = "#8B0000", lty = 2, lwd = 2)
//' abline(v = true_params[2], col = "#006400", lty = 2, lwd = 2)
//' abline(h = threshold, col = "#808080", lty = 3, lwd = 1.5)
//' legend("topright", legend = c("MLE", "True", "95% CI"),
//'        col = c("#8B0000", "#006400", "#808080"),
//'        lty = c(2, 2, 3), lwd = 2, bty = "n", cex = 0.8)
//' grid(col = "gray90")
//' 
//' plot(lambda_grid, profile_ll_lambda, type = "l", lwd = 2, col = "#2E4057",
//'      xlab = expression(lambda), ylab = "Profile Log-Likelihood",
//'      main = expression(paste("Profile: ", lambda)), las = 1)
//' abline(v = mle[3], col = "#8B0000", lty = 2, lwd = 2)
//' abline(v = true_params[3], col = "#006400", lty = 2, lwd = 2)
//' abline(h = threshold, col = "#808080", lty = 3, lwd = 1.5)
//' legend("topright", legend = c("MLE", "True", "95% CI"),
//'        col = c("#8B0000", "#006400", "#808080"),
//'        lty = c(2, 2, 3), lwd = 2, bty = "n", cex = 0.8)
//' grid(col = "gray90")
//' 
//' par(mfrow = c(1, 1))
//' 
//' 
//' ## Example 6: 2D Log-Likelihood Surface (Alpha vs Beta)
//' 
//' par(mfrow = c(1, 3), mar = c(4, 4, 3, 1))
//' 
//' # Create 2D grid
//' alpha_2d <- seq(mle[1] - 0.8, mle[1] + 0.8, length.out = round(n/25))
//' beta_2d <- seq(mle[2] - 0.8, mle[2] + 0.8, length.out = round(n/25))
//' alpha_2d <- alpha_2d[alpha_2d > 0]
//' beta_2d <- beta_2d[beta_2d > 0]
//' 
//' # Compute log-likelihood surface
//' ll_surface_ab <- matrix(NA, nrow = length(alpha_2d), ncol = length(beta_2d))
//' 
//' for (i in seq_along(alpha_2d)) {
//'   for (j in seq_along(beta_2d)) {
//'     ll_surface_ab[i, j] <- -llekw(c(alpha_2d[i], beta_2d[j], mle[3]), data)
//'   }
//' }
//' 
//' # Confidence region levels
//' max_ll_ab <- max(ll_surface_ab, na.rm = TRUE)
//' levels_90_ab <- max_ll_ab - qchisq(0.90, df = 2) / 2
//' levels_95_ab <- max_ll_ab - qchisq(0.95, df = 2) / 2
//' levels_99_ab <- max_ll_ab - qchisq(0.99, df = 2) / 2
//' 
//' # Plot contour
//' contour(alpha_2d, beta_2d, ll_surface_ab,
//'         xlab = expression(alpha), ylab = expression(beta),
//'         main = "2D Log-Likelihood: Alpha vs Beta",
//'         levels = seq(min(ll_surface_ab, na.rm = TRUE), max_ll_ab, length.out = 20),
//'         col = "#2E4057", las = 1, lwd = 1)
//' 
//' contour(alpha_2d, beta_2d, ll_surface_ab,
//'         levels = c(levels_90_ab, levels_95_ab, levels_99_ab),
//'         col = c("#FFA07A", "#FF6347", "#8B0000"),
//'         lwd = c(2, 2.5, 3), lty = c(3, 2, 1),
//'         add = TRUE, labcex = 0.8)
//' 
//' points(mle[1], mle[2], pch = 19, col = "#8B0000", cex = 1.5)
//' points(true_params[1], true_params[2], pch = 17, col = "#006400", cex = 1.5)
//' 
//' legend("topright",
//'        legend = c("MLE", "True", "90% CR", "95% CR", "99% CR"),
//'        col = c("#8B0000", "#006400", "#FFA07A", "#FF6347", "#8B0000"),
//'        pch = c(19, 17, NA, NA, NA),
//'        lty = c(NA, NA, 3, 2, 1),
//'        lwd = c(NA, NA, 2, 2.5, 3),
//'        bty = "n", cex = 0.8)
//' grid(col = "gray90")
//' 
//' 
//' ## Example 7: 2D Log-Likelihood Surface (Alpha vs Lambda)
//' 
//' # Create 2D grid
//' alpha_2d_2 <- seq(mle[1] - 0.8, mle[1] + 0.8, length.out = round(n/25))
//' lambda_2d <- seq(mle[3] - 0.8, mle[3] + 0.8, length.out = round(n/25))
//' alpha_2d_2 <- alpha_2d_2[alpha_2d_2 > 0]
//' lambda_2d <- lambda_2d[lambda_2d > 0]
//' 
//' # Compute log-likelihood surface
//' ll_surface_al <- matrix(NA, nrow = length(alpha_2d_2), ncol = length(lambda_2d))
//' 
//' for (i in seq_along(alpha_2d_2)) {
//'   for (j in seq_along(lambda_2d)) {
//'     ll_surface_al[i, j] <- -llekw(c(alpha_2d_2[i], mle[2], lambda_2d[j]), data)
//'   }
//' }
//' 
//' # Confidence region levels
//' max_ll_al <- max(ll_surface_al, na.rm = TRUE)
//' levels_90_al <- max_ll_al - qchisq(0.90, df = 2) / 2
//' levels_95_al <- max_ll_al - qchisq(0.95, df = 2) / 2
//' levels_99_al <- max_ll_al - qchisq(0.99, df = 2) / 2
//' 
//' # Plot contour
//' contour(alpha_2d_2, lambda_2d, ll_surface_al,
//'         xlab = expression(alpha), ylab = expression(lambda),
//'         main = "2D Log-Likelihood: Alpha vs Lambda",
//'         levels = seq(min(ll_surface_al, na.rm = TRUE), max_ll_al, length.out = 20),
//'         col = "#2E4057", las = 1, lwd = 1)
//' 
//' contour(alpha_2d_2, lambda_2d, ll_surface_al,
//'         levels = c(levels_90_al, levels_95_al, levels_99_al),
//'         col = c("#FFA07A", "#FF6347", "#8B0000"),
//'         lwd = c(2, 2.5, 3), lty = c(3, 2, 1),
//'         add = TRUE, labcex = 0.8)
//' 
//' points(mle[1], mle[3], pch = 19, col = "#8B0000", cex = 1.5)
//' points(true_params[1], true_params[3], pch = 17, col = "#006400", cex = 1.5)
//' 
//' legend("topright",
//'        legend = c("MLE", "True", "90% CR", "95% CR", "99% CR"),
//'        col = c("#8B0000", "#006400", "#FFA07A", "#FF6347", "#8B0000"),
//'        pch = c(19, 17, NA, NA, NA),
//'        lty = c(NA, NA, 3, 2, 1),
//'        lwd = c(NA, NA, 2, 2.5, 3),
//'        bty = "n", cex = 0.8)
//' grid(col = "gray90")
//' 
//' 
//' ## Example 8: 2D Log-Likelihood Surface (Beta vs Lambda)
//' 
//' # Create 2D grid
//' beta_2d_2 <- seq(mle[2] - 0.8, mle[2] + 0.8, length.out = round(n/25))
//' lambda_2d_2 <- seq(mle[3] - 0.8, mle[3] + 0.8, length.out = round(n/25))
//' beta_2d_2 <- beta_2d_2[beta_2d_2 > 0]
//' lambda_2d_2 <- lambda_2d_2[lambda_2d_2 > 0]
//' 
//' # Compute log-likelihood surface
//' ll_surface_bl <- matrix(NA, nrow = length(beta_2d_2), ncol = length(lambda_2d_2))
//' 
//' for (i in seq_along(beta_2d_2)) {
//'   for (j in seq_along(lambda_2d_2)) {
//'     ll_surface_bl[i, j] <- -llekw(c(mle[1], beta_2d_2[i], lambda_2d_2[j]), data)
//'   }
//' }
//' 
//' # Confidence region levels
//' max_ll_bl <- max(ll_surface_bl, na.rm = TRUE)
//' levels_90_bl <- max_ll_bl - qchisq(0.90, df = 2) / 2
//' levels_95_bl <- max_ll_bl - qchisq(0.95, df = 2) / 2
//' levels_99_bl <- max_ll_bl - qchisq(0.99, df = 2) / 2
//' 
//' # Plot contour
//' contour(beta_2d_2, lambda_2d_2, ll_surface_bl,
//'         xlab = expression(beta), ylab = expression(lambda),
//'         main = "2D Log-Likelihood: Beta vs Lambda",
//'         levels = seq(min(ll_surface_bl, na.rm = TRUE), max_ll_bl, length.out = 20),
//'         col = "#2E4057", las = 1, lwd = 1)
//' 
//' contour(beta_2d_2, lambda_2d_2, ll_surface_bl,
//'         levels = c(levels_90_bl, levels_95_bl, levels_99_bl),
//'         col = c("#FFA07A", "#FF6347", "#8B0000"),
//'         lwd = c(2, 2.5, 3), lty = c(3, 2, 1),
//'         add = TRUE, labcex = 0.8)
//' 
//' points(mle[2], mle[3], pch = 19, col = "#8B0000", cex = 1.5)
//' points(true_params[2], true_params[3], pch = 17, col = "#006400", cex = 1.5)
//' 
//' legend("topright",
//'        legend = c("MLE", "True", "90% CR", "95% CR", "99% CR"),
//'        col = c("#8B0000", "#006400", "#FFA07A", "#FF6347", "#8B0000"),
//'        pch = c(19, 17, NA, NA, NA),
//'        lty = c(NA, NA, 3, 2, 1),
//'        lwd = c(NA, NA, 2, 2.5, 3),
//'        bty = "n", cex = 0.8)
//' grid(col = "gray90")
//' 
//' par(par_)
//' 
//' }
//'
//' @export
// [[Rcpp::export]]
double llekw(const Rcpp::NumericVector& par,
            const Rcpp::NumericVector& data) {
 // Parameter validation
 if (par.size() < 3) return R_PosInf;
 
 double alpha = par[0];
 double beta = par[1];
 double lambda = par[2];
 
 if (!check_ekw_pars(alpha, beta, lambda)) return R_PosInf;
 
 arma::vec x = Rcpp::as<arma::vec>(data);
 if (x.n_elem < 1) return R_PosInf;
 if (arma::any(x <= 0.0) || arma::any(x >= 1.0)) return R_PosInf;
 
 int n = x.n_elem;
 
 // Calculate log parameters for better precision
 double log_alpha = safe_log(alpha);
 double log_beta = safe_log(beta);
 double log_lambda = safe_log(lambda);
 
 // Constant term
 double const_term = n * (log_lambda + log_alpha + log_beta);
 
 // Initialize sum terms
 double sum_term1 = 0.0; // (alpha-1) * sum(log(x))
 double sum_term2 = 0.0; // (beta-1) * sum(log(1-x^alpha))
 double sum_term3 = 0.0; // (lambda-1) * sum(log(1-(1-x^alpha)^beta))
 
 for (int i = 0; i < n; i++) {
   double xi = x(i);
   double log_xi = std::log(xi);
   
   // Term 1: (alpha-1) * log(x)
   sum_term1 += (alpha - 1.0) * log_xi;
   
   // Stable calculation of x^alpha for large alpha
   double x_alpha;
   if (alpha > 100.0 || (alpha * log_xi < -700.0)) {
     x_alpha = std::exp(alpha * log_xi);
   } else {
     x_alpha = std::pow(xi, alpha);
   }
   
   // Stable calculation of (1-x^alpha) and log(1-x^alpha)
   double one_minus_x_alpha;
   double log_one_minus_x_alpha;
   
   if (x_alpha > 0.9995) {
     // For x^alpha close to 1, use complement approach
     one_minus_x_alpha = -std::expm1(alpha * log_xi);
     log_one_minus_x_alpha = std::log(one_minus_x_alpha);
   } else {
     one_minus_x_alpha = 1.0 - x_alpha;
     log_one_minus_x_alpha = std::log(one_minus_x_alpha);
   }
   
   // Term 2: (beta-1) * log(1-x^alpha)
   sum_term2 += (beta - 1.0) * log_one_minus_x_alpha;
   
   // Stable calculation of (1-x^alpha)^beta
   double v_beta;
   if (beta > 100.0 || (beta * log_one_minus_x_alpha < -700.0)) {
     v_beta = std::exp(beta * log_one_minus_x_alpha);
   } else {
     v_beta = std::pow(one_minus_x_alpha, beta);
   }
   
   // Stable calculation of [1-(1-x^alpha)^beta]
   double one_minus_v_beta;
   double log_one_minus_v_beta;
   
   if (v_beta > 0.9995) {
     // When (1-x^alpha)^beta is close to 1
     one_minus_v_beta = -std::expm1(beta * log_one_minus_x_alpha);
   } else {
     one_minus_v_beta = 1.0 - v_beta;
   }
   
   // CRITICAL: Handle extreme lambda values
   // Prevent underflow when one_minus_v_beta is small and lambda is large
   if (one_minus_v_beta < 1e-300) {
     one_minus_v_beta = 1e-300;
   }
   
   log_one_minus_v_beta = std::log(one_minus_v_beta);
   
   // Term 3: (lambda-1) * log(1-(1-x^alpha)^beta)
   // Special handling for lambda near 1
   if (std::abs(lambda - 1.0) < 1e-10) {
     // For lambda ≈ 1, avoid numerical cancellation
     if (std::abs(lambda - 1.0) > 1e-15) {
       sum_term3 += (lambda - 1.0) * log_one_minus_v_beta;
     }
     // For lambda = 1 (machine precision), term is zero
   } else if (lambda > 1000.0 && log_one_minus_v_beta < -0.01) {
     // Special case for very large lambda
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



// -----------------------------------------------------------------------------
// 6) grekw: Gradient of Negative Log-Likelihood of EKw
// -----------------------------------------------------------------------------

//' @title Gradient of the Negative Log-Likelihood for the EKw Distribution
//' @author Lopes, J. E.
//' @keywords distribution likelihood optimize gradient
//'
//' @description
//' Computes the gradient vector (vector of first partial derivatives) of the
//' negative log-likelihood function for the Exponentiated Kumaraswamy (EKw)
//' distribution with parameters \code{alpha} (\eqn{\alpha}), \code{beta}
//' (\eqn{\beta}), and \code{lambda} (\eqn{\lambda}). This distribution is the
//' special case of the Generalized Kumaraswamy (GKw) distribution where
//' \eqn{\gamma = 1} and \eqn{\delta = 0}. The gradient is useful for optimization.
//'
//' @param par A numeric vector of length 3 containing the distribution parameters
//'   in the order: \code{alpha} (\eqn{\alpha > 0}), \code{beta} (\eqn{\beta > 0}),
//'   \code{lambda} (\eqn{\lambda > 0}).
//' @param data A numeric vector of observations. All values must be strictly
//'   between 0 and 1 (exclusive).
//'
//' @return Returns a numeric vector of length 3 containing the partial derivatives
//'   of the negative log-likelihood function \eqn{-\ell(\theta | \mathbf{x})} with
//'   respect to each parameter: \eqn{(-\partial \ell/\partial \alpha, -\partial \ell/\partial \beta, -\partial \ell/\partial \lambda)}.
//'   Returns a vector of \code{NaN} if any parameter values are invalid according
//'   to their constraints, or if any value in \code{data} is not in the
//'   interval (0, 1).
//'
//' @details
//' The components of the gradient vector of the negative log-likelihood
//' (\eqn{-\nabla \ell(\theta | \mathbf{x})}) for the EKw (\eqn{\gamma=1, \delta=0})
//' model are:
//'
//' \deqn{
//' -\frac{\partial \ell}{\partial \alpha} = -\frac{n}{\alpha} - \sum_{i=1}^{n}\ln(x_i)
//' + \sum_{i=1}^{n}\left[x_i^{\alpha} \ln(x_i) \left(\frac{\beta-1}{v_i} -
//' \frac{(\lambda-1) \beta v_i^{\beta-1}}{w_i}\right)\right]
//' }
//' \deqn{
//' -\frac{\partial \ell}{\partial \beta} = -\frac{n}{\beta} - \sum_{i=1}^{n}\ln(v_i)
//' + \sum_{i=1}^{n}\left[\frac{(\lambda-1) v_i^{\beta} \ln(v_i)}{w_i}\right]
//' }
//' \deqn{
//' -\frac{\partial \ell}{\partial \lambda} = -\frac{n}{\lambda} - \sum_{i=1}^{n}\ln(w_i)
//' }
//'
//' where:
//' \itemize{
//'   \item \eqn{v_i = 1 - x_i^{\alpha}}
//'   \item \eqn{w_i = 1 - v_i^{\beta} = 1 - (1-x_i^{\alpha})^{\beta}}
//' }
//' These formulas represent the derivatives of \eqn{-\ell(\theta)}, consistent with
//' minimizing the negative log-likelihood. They correspond to the relevant components
//' of the general GKw gradient (\code{\link{grgkw}}) evaluated at \eqn{\gamma=1, \delta=0}.
//'
//' @references
//' Nadarajah, S., Cordeiro, G. M., & Ortega, E. M. (2012). The exponentiated
//' Kumaraswamy distribution. *Journal of the Franklin Institute*, *349*(3),
//'
//'
//' Cordeiro, G. M., & de Castro, M. (2011). A new family of generalized
//' distributions. *Journal of Statistical Computation and Simulation*,
//'
//'
//' Kumaraswamy, P. (1980). A generalized probability density function for
//' double-bounded random processes. *Journal of Hydrology*, *46*(1-2), 79-88.
//'
//' (Note: Specific gradient formulas might be derived or sourced from additional references).
//'
//' @seealso
//' \code{\link{grgkw}} (parent distribution gradient),
//' \code{\link{llekw}} (negative log-likelihood for EKw),
//' \code{hsekw} (Hessian for EKw, if available),
//' \code{\link{dekw}} (density for EKw),
//' \code{\link[stats]{optim}},
//' \code{\link[numDeriv]{grad}} (for numerical gradient comparison).
//'
//' @examples
//' \donttest{
//' ## Example 1: Basic Gradient Evaluation
//' 
//' par_ <- par()
//' 
//' # Generate sample data
//' set.seed(123)
//' n <- 1000
//' true_params <- c(alpha = 2.5, beta = 3.5, lambda = 2.0)
//' data <- rekw(n, alpha = true_params[1], beta = true_params[2],
//'              lambda = true_params[3])
//' 
//' # Evaluate gradient at true parameters
//' grad_true <- grekw(par = true_params, data = data)
//' cat("Gradient at true parameters:\n")
//' print(grad_true)
//' cat("Norm:", sqrt(sum(grad_true^2)), "\n")
//' 
//' # Evaluate at different parameter values
//' test_params <- rbind(
//'   c(2.0, 3.0, 1.5),
//'   c(2.5, 3.5, 2.0),
//'   c(3.0, 4.0, 2.5)
//' )
//' 
//' grad_norms <- apply(test_params, 1, function(p) {
//'   g <- grekw(p, data)
//'   sqrt(sum(g^2))
//' })
//' 
//' results <- data.frame(
//'   Alpha = test_params[, 1],
//'   Beta = test_params[, 2],
//'   Lambda = test_params[, 3],
//'   Grad_Norm = grad_norms
//' )
//' print(results, digits = 4)
//' 
//' 
//' ## Example 2: Gradient in Optimization
//' 
//' # Optimization with analytical gradient
//' fit_with_grad <- optim(
//'   par = c(2, 3, 1.5),
//'   fn = llekw,
//'   gr = grekw,
//'   data = data,
//'   method = "BFGS",
//'   hessian = TRUE,
//'   control = list(trace = 0)
//' )
//' 
//' # Optimization without gradient
//' fit_no_grad <- optim(
//'   par = c(2, 3, 1.5),
//'   fn = llekw,
//'   data = data,
//'   method = "BFGS",
//'   hessian = TRUE,
//'   control = list(trace = 0)
//' )
//' 
//' comparison <- data.frame(
//'   Method = c("With Gradient", "Without Gradient"),
//'   Alpha = c(fit_with_grad$par[1], fit_no_grad$par[1]),
//'   Beta = c(fit_with_grad$par[2], fit_no_grad$par[2]),
//'   Lambda = c(fit_with_grad$par[3], fit_no_grad$par[3]),
//'   NegLogLik = c(fit_with_grad$value, fit_no_grad$value),
//'   Iterations = c(fit_with_grad$counts[1], fit_no_grad$counts[1])
//' )
//' print(comparison, digits = 4, row.names = FALSE)
//' 
//' 
//' ## Example 3: Verifying Gradient at MLE
//' 
//' mle <- fit_with_grad$par
//' names(mle) <- c("alpha", "beta", "lambda")
//' 
//' # At MLE, gradient should be approximately zero
//' gradient_at_mle <- grekw(par = mle, data = data)
//' cat("\nGradient at MLE:\n")
//' print(gradient_at_mle)
//' cat("Max absolute component:", max(abs(gradient_at_mle)), "\n")
//' cat("Gradient norm:", sqrt(sum(gradient_at_mle^2)), "\n")
//' 
//' 
//' ## Example 4: Numerical vs Analytical Gradient
//' 
//' # Manual finite difference gradient
//' numerical_gradient <- function(f, x, data, h = 1e-7) {
//'   grad <- numeric(length(x))
//'   for (i in seq_along(x)) {
//'     x_plus <- x_minus <- x
//'     x_plus[i] <- x[i] + h
//'     x_minus[i] <- x[i] - h
//'     grad[i] <- (f(x_plus, data) - f(x_minus, data)) / (2 * h)
//'   }
//'   return(grad)
//' }
//' 
//' # Compare at MLE
//' grad_analytical <- grekw(par = mle, data = data)
//' grad_numerical <- numerical_gradient(llekw, mle, data)
//' 
//' comparison_grad <- data.frame(
//'   Parameter = c("alpha", "beta", "lambda"),
//'   Analytical = grad_analytical,
//'   Numerical = grad_numerical,
//'   Abs_Diff = abs(grad_analytical - grad_numerical),
//'   Rel_Error = abs(grad_analytical - grad_numerical) /
//'               (abs(grad_analytical) + 1e-10)
//' )
//' print(comparison_grad, digits = 8)
//' 
//' 
//' ## Example 5: Score Test Statistic
//' 
//' # Score test for H0: theta = theta0
//' theta0 <- c(2.2, 3.2, 1.8)
//' score_theta0 <- -grekw(par = theta0, data = data)
//' 
//' # Fisher information at theta0
//' fisher_info <- hsekw(par = theta0, data = data)
//' 
//' # Score test statistic
//' score_stat <- t(score_theta0) %*% solve(fisher_info) %*% score_theta0
//' p_value <- pchisq(score_stat, df = 3, lower.tail = FALSE)
//' 
//' cat("\nScore Test:\n")
//' cat("H0: alpha=2.2, beta=3.2, lambda=1.8\n")
//' cat("Test statistic:", score_stat, "\n")
//' cat("P-value:", format.pval(p_value, digits = 4), "\n")
//' 
//' 
//' ## Example 6: Confidence Ellipse (Alpha vs Beta)
//' 
//' # Observed information
//' obs_info <- hsekw(par = mle, data = data)
//' vcov_full <- solve(obs_info)
//' vcov_2d <- vcov_full[1:2, 1:2]
//' 
//' # Create confidence ellipse
//' theta <- seq(0, 2 * pi, length.out = 100)
//' chi2_val <- qchisq(0.95, df = 2)
//' 
//' eig_decomp <- eigen(vcov_2d)
//' ellipse <- matrix(NA, nrow = 100, ncol = 2)
//' for (i in 1:100) {
//'   v <- c(cos(theta[i]), sin(theta[i]))
//'   ellipse[i, ] <- mle[1:2] + sqrt(chi2_val) *
//'     (eig_decomp$vectors %*% diag(sqrt(eig_decomp$values)) %*% v)
//' }
//' 
//' # Marginal confidence intervals
//' se_2d <- sqrt(diag(vcov_2d))
//' ci_alpha <- mle[1] + c(-1, 1) * 1.96 * se_2d[1]
//' ci_beta <- mle[2] + c(-1, 1) * 1.96 * se_2d[2]
//' 
//' # Plot
//' par(mfrow = c(1,3), mar = c(4, 4, 3, 1))
//' plot(ellipse[, 1], ellipse[, 2], type = "l", lwd = 2, col = "#2E4057",
//'      xlab = expression(alpha), ylab = expression(beta),
//'      main = "95% Confidence Region (Alpha vs Beta)", las = 1)
//' 
//' # Add marginal CIs
//' abline(v = ci_alpha, col = "#808080", lty = 3, lwd = 1.5)
//' abline(h = ci_beta, col = "#808080", lty = 3, lwd = 1.5)
//' 
//' points(mle[1], mle[2], pch = 19, col = "#8B0000", cex = 1.5)
//' points(true_params[1], true_params[2], pch = 17, col = "#006400", cex = 1.5)
//' 
//' legend("topright",
//'        legend = c("MLE", "True", "95% CR", "Marginal 95% CI"),
//'        col = c("#8B0000", "#006400", "#2E4057", "#808080"),
//'        pch = c(19, 17, NA, NA),
//'        lty = c(NA, NA, 1, 3),
//'        lwd = c(NA, NA, 2, 1.5),
//'        bty = "n")
//' grid(col = "gray90")
//' 
//' 
//' ## Example 7: Confidence Ellipse (Alpha vs Lambda)
//' 
//' # Extract 2x2 submatrix for alpha and lambda
//' vcov_2d_al <- vcov_full[c(1, 3), c(1, 3)]
//' 
//' # Create confidence ellipse
//' eig_decomp_al <- eigen(vcov_2d_al)
//' ellipse_al <- matrix(NA, nrow = 100, ncol = 2)
//' for (i in 1:100) {
//'   v <- c(cos(theta[i]), sin(theta[i]))
//'   ellipse_al[i, ] <- mle[c(1, 3)] + sqrt(chi2_val) *
//'     (eig_decomp_al$vectors %*% diag(sqrt(eig_decomp_al$values)) %*% v)
//' }
//' 
//' # Marginal confidence intervals
//' se_2d_al <- sqrt(diag(vcov_2d_al))
//' ci_alpha_2 <- mle[1] + c(-1, 1) * 1.96 * se_2d_al[1]
//' ci_lambda <- mle[3] + c(-1, 1) * 1.96 * se_2d_al[2]
//' 
//' # Plot
//' par(mar = c(4, 4, 3, 1))
//' plot(ellipse_al[, 1], ellipse_al[, 2], type = "l", lwd = 2, col = "#2E4057",
//'      xlab = expression(alpha), ylab = expression(lambda),
//'      main = "95% Confidence Region (Alpha vs Lambda)", las = 1)
//' 
//' # Add marginal CIs
//' abline(v = ci_alpha_2, col = "#808080", lty = 3, lwd = 1.5)
//' abline(h = ci_lambda, col = "#808080", lty = 3, lwd = 1.5)
//' 
//' points(mle[1], mle[3], pch = 19, col = "#8B0000", cex = 1.5)
//' points(true_params[1], true_params[3], pch = 17, col = "#006400", cex = 1.5)
//' 
//' legend("topright",
//'        legend = c("MLE", "True", "95% CR", "Marginal 95% CI"),
//'        col = c("#8B0000", "#006400", "#2E4057", "#808080"),
//'        pch = c(19, 17, NA, NA),
//'        lty = c(NA, NA, 1, 3),
//'        lwd = c(NA, NA, 2, 1.5),
//'        bty = "n")
//' grid(col = "gray90")
//' 
//' 
//' ## Example 8: Confidence Ellipse (Beta vs Lambda)
//' 
//' # Extract 2x2 submatrix for beta and lambda
//' vcov_2d_bl <- vcov_full[2:3, 2:3]
//' 
//' # Create confidence ellipse
//' eig_decomp_bl <- eigen(vcov_2d_bl)
//' ellipse_bl <- matrix(NA, nrow = 100, ncol = 2)
//' for (i in 1:100) {
//'   v <- c(cos(theta[i]), sin(theta[i]))
//'   ellipse_bl[i, ] <- mle[2:3] + sqrt(chi2_val) *
//'     (eig_decomp_bl$vectors %*% diag(sqrt(eig_decomp_bl$values)) %*% v)
//' }
//' 
//' # Marginal confidence intervals
//' se_2d_bl <- sqrt(diag(vcov_2d_bl))
//' ci_beta_2 <- mle[2] + c(-1, 1) * 1.96 * se_2d_bl[1]
//' ci_lambda_2 <- mle[3] + c(-1, 1) * 1.96 * se_2d_bl[2]
//' 
//' # Plot
//' par(mar = c(4, 4, 3, 1))
//' plot(ellipse_bl[, 1], ellipse_bl[, 2], type = "l", lwd = 2, col = "#2E4057",
//'      xlab = expression(beta), ylab = expression(lambda),
//'      main = "95% Confidence Region (Beta vs Lambda)", las = 1)
//' 
//' # Add marginal CIs
//' abline(v = ci_beta_2, col = "#808080", lty = 3, lwd = 1.5)
//' abline(h = ci_lambda_2, col = "#808080", lty = 3, lwd = 1.5)
//' 
//' points(mle[2], mle[3], pch = 19, col = "#8B0000", cex = 1.5)
//' points(true_params[2], true_params[3], pch = 17, col = "#006400", cex = 1.5)
//' 
//' legend("topright",
//'        legend = c("MLE", "True", "95% CR", "Marginal 95% CI"),
//'        col = c("#8B0000", "#006400", "#2E4057", "#808080"),
//'        pch = c(19, 17, NA, NA),
//'        lty = c(NA, NA, 1, 3),
//'        lwd = c(NA, NA, 2, 1.5),
//'        bty = "n")
//' grid(col = "gray90")
//' 
//' par(par_)
//' 
//' }
//'
//' @export
// [[Rcpp::export]]
Rcpp::NumericVector grekw(const Rcpp::NumericVector& par, const Rcpp::NumericVector& data) {
 // Parameter validation
 if (par.size() < 3) {
   Rcpp::NumericVector grad(3, R_NaN);
   return grad;
 }
 
 double alpha = par[0];
 double beta = par[1];
 double lambda = par[2];
 
 if (alpha <= 0 || beta <= 0 || lambda <= 0) {
   Rcpp::NumericVector grad(3, R_NaN);
   return grad;
 }
 
 arma::vec x = Rcpp::as<arma::vec>(data);
 if (x.n_elem < 1 || arma::any(x <= 0) || arma::any(x >= 1)) {
   Rcpp::NumericVector grad(3, R_NaN);
   return grad;
 }
 
 int n = x.n_elem;
 Rcpp::NumericVector grad(3, 0.0);
 
 // Constants for numerical stability
 const double min_val = 1e-10;
 const double exp_threshold = -700.0;
 
 // Initialize component accumulators
 double d_alpha = n / alpha;
 double d_beta = n / beta;
 double d_lambda = n / lambda;
 
 for (int i = 0; i < n; i++) {
   double xi = x(i);
   double log_xi = std::log(xi);
   d_alpha += log_xi;  // Accumulate (α-1) * log(x_i) term
   
   // Compute x^α stably (use log domain for large alpha)
   double x_alpha;
   if (alpha > 100.0 || (alpha * log_xi < exp_threshold)) {
     x_alpha = std::exp(alpha * log_xi);
   } else {
     x_alpha = std::pow(xi, alpha);
   }
   
   // Compute v = 1-x^α with precision for x^α near 1
   double v;
   if (x_alpha > 0.9995) {
     v = -std::expm1(alpha * log_xi);  // More precise than 1.0 - x_alpha
   } else {
     v = 1.0 - x_alpha;
   }
   
   // Ensure v is not too small
   v = std::max(v, min_val);
   double log_v = std::log(v);
   d_beta += log_v;  // Accumulate (β-1) * log(v_i) term
   
   // Compute v^β stably
   double v_beta, v_beta_m1;
   if (beta > 100.0 || (beta * log_v < exp_threshold)) {
     double log_v_beta = beta * log_v;
     v_beta = std::exp(log_v_beta);
     v_beta_m1 = std::exp((beta - 1.0) * log_v);
   } else {
     v_beta = std::pow(v, beta);
     v_beta_m1 = std::pow(v, beta - 1.0);
   }
   
   // Compute w = 1-v^β with precision for v^β near 1
   double w;
   if (v_beta > 0.9995) {
     w = -std::expm1(beta * log_v);
   } else {
     w = 1.0 - v_beta;
   }
   
   // Ensure w is not too small
   w = std::max(w, min_val);
   double log_w = std::log(w);
   d_lambda += log_w;  // Accumulate (λ-1) * log(w_i) term
   
   // --- Alpha gradient component ---
   // Calculate x^α * log(x) term
   double x_alpha_log_x = x_alpha * log_xi;
   
   // Calculate (β-1)/v term - stable for β ≈ 1
   double alpha_term1 = 0.0;
   if (std::abs(beta - 1.0) > 1e-14) {
     alpha_term1 = (beta - 1.0) / v;
   }
   
   // Calculate (λ-1) * β * v^(β-1) / w term - with λ stability
   double alpha_term2 = 0.0;
   if (std::abs(lambda - 1.0) > 1e-14) {
     double lambda_factor = lambda - 1.0;
     // Clamp the factor for very large lambda to prevent overflow
     if (lambda > 1000.0) {
       lambda_factor = std::min(lambda_factor, 1000.0);
     }
     alpha_term2 = lambda_factor * beta * v_beta_m1 / w;
   }
   
   d_alpha -= x_alpha_log_x * (alpha_term1 - alpha_term2);
   
   // --- Beta gradient component ---
   // Calculate v^β * log(v) * (λ-1) / w term - with λ stability
   double beta_term = 0.0;
   if (std::abs(lambda - 1.0) > 1e-14) {
     double lambda_factor = lambda - 1.0;
     // Clamp the factor for very large lambda
     if (lambda > 1000.0) {
       lambda_factor = std::min(lambda_factor, 1000.0);
     }
     beta_term = v_beta * log_v * lambda_factor / w;
   }
   
   d_beta -= beta_term;
 }
 
 // Negate for negative log-likelihood
 grad[0] = -d_alpha;
 grad[1] = -d_beta;
 grad[2] = -d_lambda;
 
 return grad;
}



//' @title Hessian Matrix of the Negative Log-Likelihood for the EKw Distribution
//' @author Lopes, J. E.
//' @keywords distribution likelihood optimize hessian
//'
//' @description
//' Computes the analytic 3x3 Hessian matrix (matrix of second partial derivatives)
//' of the negative log-likelihood function for the Exponentiated Kumaraswamy (EKw)
//' distribution with parameters \code{alpha} (\eqn{\alpha}), \code{beta}
//' (\eqn{\beta}), and \code{lambda} (\eqn{\lambda}). This distribution is the
//' special case of the Generalized Kumaraswamy (GKw) distribution where
//' \eqn{\gamma = 1} and \eqn{\delta = 0}. The Hessian is useful for estimating
//' standard errors and in optimization algorithms.
//'
//' @param par A numeric vector of length 3 containing the distribution parameters
//'   in the order: \code{alpha} (\eqn{\alpha > 0}), \code{beta} (\eqn{\beta > 0}),
//'   \code{lambda} (\eqn{\lambda > 0}).
//' @param data A numeric vector of observations. All values must be strictly
//'   between 0 and 1 (exclusive).
//'
//' @return Returns a 3x3 numeric matrix representing the Hessian matrix of the
//'   negative log-likelihood function, \eqn{-\partial^2 \ell / (\partial \theta_i \partial \theta_j)},
//'   where \eqn{\theta = (\alpha, \beta, \lambda)}.
//'   Returns a 3x3 matrix populated with \code{NaN} if any parameter values are
//'   invalid according to their constraints, or if any value in \code{data} is
//'   not in the interval (0, 1).
//'
//' @details
//' This function calculates the analytic second partial derivatives of the
//' negative log-likelihood function based on the EKw log-likelihood
//' (\eqn{\gamma=1, \delta=0} case of GKw, see \code{\link{llekw}}):
//' \deqn{
//' \ell(\theta | \mathbf{x}) = n[\ln(\lambda) + \ln(\alpha) + \ln(\beta)]
//' + \sum_{i=1}^{n} [(\alpha-1)\ln(x_i) + (\beta-1)\ln(v_i) + (\lambda-1)\ln(w_i)]
//' }
//' where \eqn{\theta = (\alpha, \beta, \lambda)} and intermediate terms are:
//' \itemize{
//'   \item \eqn{v_i = 1 - x_i^{\alpha}}
//'   \item \eqn{w_i = 1 - v_i^{\beta} = 1 - (1-x_i^{\alpha})^{\beta}}
//' }
//' The Hessian matrix returned contains the elements \eqn{- \frac{\partial^2 \ell(\theta | \mathbf{x})}{\partial \theta_i \partial \theta_j}}
//' for \eqn{\theta_i, \theta_j \in \{\alpha, \beta, \lambda\}}.
//'
//' Key properties of the returned matrix:
//' \itemize{
//'   \item Dimensions: 3x3.
//'   \item Symmetry: The matrix is symmetric.
//'   \item Ordering: Rows and columns correspond to the parameters in the order
//'     \eqn{\alpha, \beta, \lambda}.
//'   \item Content: Analytic second derivatives of the *negative* log-likelihood.
//' }
//' This corresponds to the relevant 3x3 submatrix of the 5x5 GKw Hessian (\code{\link{hsgkw}})
//' evaluated at \eqn{\gamma=1, \delta=0}. The exact analytical formulas are implemented directly.
//'
//' @references
//' Nadarajah, S., Cordeiro, G. M., & Ortega, E. M. (2012). The exponentiated
//' Kumaraswamy distribution. *Journal of the Franklin Institute*, *349*(3),
//'
//'
//' Cordeiro, G. M., & de Castro, M. (2011). A new family of generalized
//' distributions. *Journal of Statistical Computation and Simulation*,
//'
//'
//' Kumaraswamy, P. (1980). A generalized probability density function for
//' double-bounded random processes. *Journal of Hydrology*, *46*(1-2), 79-88.
//'
//' (Note: Specific Hessian formulas might be derived or sourced from additional references).
//'
//' @seealso
//' \code{\link{hsgkw}} (parent distribution Hessian),
//' \code{\link{llekw}} (negative log-likelihood for EKw),
//' \code{grekw} (gradient for EKw, if available),
//' \code{\link{dekw}} (density for EKw),
//' \code{\link[stats]{optim}},
//' \code{\link[numDeriv]{hessian}} (for numerical Hessian comparison).
//'
//' @examples
//' \donttest{
//' ## Example 1: Basic Hessian Evaluation
//' 
//' par_ <- par()
//' 
//' # Generate sample data
//' set.seed(123)
//' n <- 1000
//' true_params <- c(alpha = 2.5, beta = 3.5, lambda = 2.0)
//' data <- rekw(n, alpha = true_params[1], beta = true_params[2],
//'              lambda = true_params[3])
//' 
//' # Evaluate Hessian at true parameters
//' hess_true <- hsekw(par = true_params, data = data)
//' cat("Hessian matrix at true parameters:\n")
//' print(hess_true, digits = 4)
//' 
//' # Check symmetry
//' cat("\nSymmetry check (max |H - H^T|):",
//'     max(abs(hess_true - t(hess_true))), "\n")
//' 
//' 
//' ## Example 2: Hessian Properties at MLE
//' 
//' # Fit model
//' fit <- optim(
//'   par = c(2, 3, 1.5),
//'   fn = llekw,
//'   gr = grekw,
//'   data = data,
//'   method = "BFGS",
//'   hessian = TRUE
//' )
//' 
//' mle <- fit$par
//' names(mle) <- c("alpha", "beta", "lambda")
//' 
//' # Hessian at MLE
//' hessian_at_mle <- hsekw(par = mle, data = data)
//' cat("\nHessian at MLE:\n")
//' print(hessian_at_mle, digits = 4)
//' 
//' # Compare with optim's numerical Hessian
//' cat("\nComparison with optim Hessian:\n")
//' cat("Max absolute difference:",
//'     max(abs(hessian_at_mle - fit$hessian)), "\n")
//' 
//' # Eigenvalue analysis
//' eigenvals <- eigen(hessian_at_mle, only.values = TRUE)$values
//' cat("\nEigenvalues:\n")
//' print(eigenvals)
//' 
//' cat("\nPositive definite:", all(eigenvals > 0), "\n")
//' cat("Condition number:", max(eigenvals) / min(eigenvals), "\n")
//' 
//' 
//' ## Example 3: Standard Errors and Confidence Intervals
//' 
//' # Observed information matrix
//' obs_info <- hessian_at_mle
//' 
//' # Variance-covariance matrix
//' vcov_matrix <- solve(obs_info)
//' cat("\nVariance-Covariance Matrix:\n")
//' print(vcov_matrix, digits = 6)
//' 
//' # Standard errors
//' se <- sqrt(diag(vcov_matrix))
//' names(se) <- c("alpha", "beta", "lambda")
//' 
//' # Correlation matrix
//' corr_matrix <- cov2cor(vcov_matrix)
//' cat("\nCorrelation Matrix:\n")
//' print(corr_matrix, digits = 4)
//' 
//' # Confidence intervals
//' z_crit <- qnorm(0.975)
//' results <- data.frame(
//'   Parameter = c("alpha", "beta", "lambda"),
//'   True = true_params,
//'   MLE = mle,
//'   SE = se,
//'   CI_Lower = mle - z_crit * se,
//'   CI_Upper = mle + z_crit * se
//' )
//' print(results, digits = 4)
//' 
//' 
//' ## Example 4: Determinant and Trace Analysis
//' 
//' # Compute at different points
//' test_params <- rbind(
//'   c(2.0, 3.0, 1.5),
//'   c(2.5, 3.5, 2.0),
//'   mle,
//'   c(3.0, 4.0, 2.5)
//' )
//' 
//' hess_properties <- data.frame(
//'   Alpha = numeric(),
//'   Beta = numeric(),
//'   Lambda = numeric(),
//'   Determinant = numeric(),
//'   Trace = numeric(),
//'   Min_Eigenval = numeric(),
//'   Max_Eigenval = numeric(),
//'   Cond_Number = numeric(),
//'   stringsAsFactors = FALSE
//' )
//' 
//' for (i in 1:nrow(test_params)) {
//'   H <- hsekw(par = test_params[i, ], data = data)
//'   eigs <- eigen(H, only.values = TRUE)$values
//' 
//'   hess_properties <- rbind(hess_properties, data.frame(
//'     Alpha = test_params[i, 1],
//'     Beta = test_params[i, 2],
//'     Lambda = test_params[i, 3],
//'     Determinant = det(H),
//'     Trace = sum(diag(H)),
//'     Min_Eigenval = min(eigs),
//'     Max_Eigenval = max(eigs),
//'     Cond_Number = max(eigs) / min(eigs)
//'   ))
//' }
//' 
//' cat("\nHessian Properties at Different Points:\n")
//' print(hess_properties, digits = 4, row.names = FALSE)
//' 
//' 
//' ## Example 5: Curvature Visualization (Alpha vs Beta)
//' 
//' # Create grid around MLE
//' alpha_grid <- seq(mle[1] - 0.5, mle[1] + 0.5, length.out = 25)
//' beta_grid <- seq(mle[2] - 0.5, mle[2] + 0.5, length.out = 25)
//' alpha_grid <- alpha_grid[alpha_grid > 0]
//' beta_grid <- beta_grid[beta_grid > 0]
//' 
//' # Compute curvature measures
//' determinant_surface <- matrix(NA, nrow = length(alpha_grid),
//'                                ncol = length(beta_grid))
//' trace_surface <- matrix(NA, nrow = length(alpha_grid),
//'                          ncol = length(beta_grid))
//' 
//' for (i in seq_along(alpha_grid)) {
//'   for (j in seq_along(beta_grid)) {
//'     H <- hsekw(c(alpha_grid[i], beta_grid[j], mle[3]), data)
//'     determinant_surface[i, j] <- det(H)
//'     trace_surface[i, j] <- sum(diag(H))
//'   }
//' }
//' 
//' # Plot
//' par(mfrow = c(1, 2), mar = c(4, 4, 3, 1))
//' 
//' contour(alpha_grid, beta_grid, determinant_surface,
//'         xlab = expression(alpha), ylab = expression(beta),
//'         main = "Hessian Determinant", las = 1,
//'         col = "#2E4057", lwd = 1.5, nlevels = 15)
//' points(mle[1], mle[2], pch = 19, col = "#8B0000", cex = 1.5)
//' points(true_params[1], true_params[2], pch = 17, col = "#006400", cex = 1.5)
//' grid(col = "gray90")
//' 
//' contour(alpha_grid, beta_grid, trace_surface,
//'         xlab = expression(alpha), ylab = expression(beta),
//'         main = "Hessian Trace", las = 1,
//'         col = "#2E4057", lwd = 1.5, nlevels = 15)
//' points(mle[1], mle[2], pch = 19, col = "#8B0000", cex = 1.5)
//' points(true_params[1], true_params[2], pch = 17, col = "#006400", cex = 1.5)
//' grid(col = "gray90")
//' par(mfrow = c(1,1))
//' 
//' ## Example 6: Confidence Ellipse (Alpha vs Beta)
//' par(mfrow = c(1, 3), mar = c(4, 4, 3, 1))
//' 
//' # Extract 2x2 submatrix for alpha and beta
//' vcov_2d <- vcov_matrix[1:2, 1:2]
//' 
//' # Create confidence ellipse
//' theta <- seq(0, 2 * pi, length.out = 100)
//' chi2_val <- qchisq(0.95, df = 2)
//' 
//' eig_decomp <- eigen(vcov_2d)
//' ellipse <- matrix(NA, nrow = 100, ncol = 2)
//' for (i in 1:100) {
//'   v <- c(cos(theta[i]), sin(theta[i]))
//'   ellipse[i, ] <- mle[1:2] + sqrt(chi2_val) *
//'     (eig_decomp$vectors %*% diag(sqrt(eig_decomp$values)) %*% v)
//' }
//' 
//' # Marginal confidence intervals
//' se_2d <- sqrt(diag(vcov_2d))
//' ci_alpha <- mle[1] + c(-1, 1) * 1.96 * se_2d[1]
//' ci_beta <- mle[2] + c(-1, 1) * 1.96 * se_2d[2]
//' 
//' # Plot
//' # par(mar = c(4, 4, 3, 1))
//' plot(ellipse[, 1], ellipse[, 2], type = "l", lwd = 2, col = "#2E4057",
//'      xlab = expression(alpha), ylab = expression(beta),
//'      main = "95% Confidence Ellipse (Alpha vs Beta)", las = 1)
//' 
//' # Add marginal CIs
//' abline(v = ci_alpha, col = "#808080", lty = 3, lwd = 1.5)
//' abline(h = ci_beta, col = "#808080", lty = 3, lwd = 1.5)
//' 
//' points(mle[1], mle[2], pch = 19, col = "#8B0000", cex = 1.5)
//' points(true_params[1], true_params[2], pch = 17, col = "#006400", cex = 1.5)
//' 
//' legend("topright",
//'        legend = c("MLE", "True", "95% CR", "Marginal 95% CI"),
//'        col = c("#8B0000", "#006400", "#2E4057", "#808080"),
//'        pch = c(19, 17, NA, NA),
//'        lty = c(NA, NA, 1, 3),
//'        lwd = c(NA, NA, 2, 1.5),
//'        bty = "n")
//' grid(col = "gray90")
//' 
//' 
//' ## Example 7: Confidence Ellipse (Alpha vs Lambda)
//' 
//' # Extract 2x2 submatrix for alpha and lambda
//' vcov_2d_al <- vcov_matrix[c(1, 3), c(1, 3)]
//' 
//' # Create confidence ellipse
//' eig_decomp_al <- eigen(vcov_2d_al)
//' ellipse_al <- matrix(NA, nrow = 100, ncol = 2)
//' for (i in 1:100) {
//'   v <- c(cos(theta[i]), sin(theta[i]))
//'   ellipse_al[i, ] <- mle[c(1, 3)] + sqrt(chi2_val) *
//'     (eig_decomp_al$vectors %*% diag(sqrt(eig_decomp_al$values)) %*% v)
//' }
//' 
//' # Marginal confidence intervals
//' se_2d_al <- sqrt(diag(vcov_2d_al))
//' ci_alpha_2 <- mle[1] + c(-1, 1) * 1.96 * se_2d_al[1]
//' ci_lambda <- mle[3] + c(-1, 1) * 1.96 * se_2d_al[2]
//' 
//' # Plot
//' par(mar = c(4, 4, 3, 1))
//' plot(ellipse_al[, 1], ellipse_al[, 2], type = "l", lwd = 2, col = "#2E4057",
//'      xlab = expression(alpha), ylab = expression(lambda),
//'      main = "95% Confidence Ellipse (Alpha vs Lambda)", las = 1)
//' 
//' # Add marginal CIs
//' abline(v = ci_alpha_2, col = "#808080", lty = 3, lwd = 1.5)
//' abline(h = ci_lambda, col = "#808080", lty = 3, lwd = 1.5)
//' 
//' points(mle[1], mle[3], pch = 19, col = "#8B0000", cex = 1.5)
//' points(true_params[1], true_params[3], pch = 17, col = "#006400", cex = 1.5)
//' 
//' legend("topright",
//'        legend = c("MLE", "True", "95% CR", "Marginal 95% CI"),
//'        col = c("#8B0000", "#006400", "#2E4057", "#808080"),
//'        pch = c(19, 17, NA, NA),
//'        lty = c(NA, NA, 1, 3),
//'        lwd = c(NA, NA, 2, 1.5),
//'        bty = "n")
//' grid(col = "gray90")
//' 
//' 
//' ## Example 8: Confidence Ellipse (Beta vs Lambda)
//' 
//' # Extract 2x2 submatrix for beta and lambda
//' vcov_2d_bl <- vcov_matrix[2:3, 2:3]
//' 
//' # Create confidence ellipse
//' eig_decomp_bl <- eigen(vcov_2d_bl)
//' ellipse_bl <- matrix(NA, nrow = 100, ncol = 2)
//' for (i in 1:100) {
//'   v <- c(cos(theta[i]), sin(theta[i]))
//'   ellipse_bl[i, ] <- mle[2:3] + sqrt(chi2_val) *
//'     (eig_decomp_bl$vectors %*% diag(sqrt(eig_decomp_bl$values)) %*% v)
//' }
//' 
//' # Marginal confidence intervals
//' se_2d_bl <- sqrt(diag(vcov_2d_bl))
//' ci_beta_2 <- mle[2] + c(-1, 1) * 1.96 * se_2d_bl[1]
//' ci_lambda_2 <- mle[3] + c(-1, 1) * 1.96 * se_2d_bl[2]
//' 
//' # Plot
//' par(mar = c(4, 4, 3, 1))
//' plot(ellipse_bl[, 1], ellipse_bl[, 2], type = "l", lwd = 2, col = "#2E4057",
//'      xlab = expression(beta), ylab = expression(lambda),
//'      main = "95% Confidence Ellipse (Beta vs Lambda)", las = 1)
//' 
//' # Add marginal CIs
//' abline(v = ci_beta_2, col = "#808080", lty = 3, lwd = 1.5)
//' abline(h = ci_lambda_2, col = "#808080", lty = 3, lwd = 1.5)
//' 
//' points(mle[2], mle[3], pch = 19, col = "#8B0000", cex = 1.5)
//' points(true_params[2], true_params[3], pch = 17, col = "#006400", cex = 1.5)
//' 
//' legend("topright",
//'        legend = c("MLE", "True", "95% CR", "Marginal 95% CI"),
//'        col = c("#8B0000", "#006400", "#2E4057", "#808080"),
//'        pch = c(19, 17, NA, NA),
//'        lty = c(NA, NA, 1, 3),
//'        lwd = c(NA, NA, 2, 1.5),
//'        bty = "n")
//' grid(col = "gray90")
//' 
//' par(par_)
//' 
//' }
//'
//' @export
// [[Rcpp::export]]
Rcpp::NumericMatrix hsekw(const Rcpp::NumericVector& par, const Rcpp::NumericVector& data) {
 // Parameter validation
 if (par.size() < 3) {
   Rcpp::NumericMatrix nanH(3,3);
   nanH.fill(R_NaN);
   return nanH;
 }
 
 double alpha = par[0];
 double beta = par[1];
 double lambda = par[2];
 
 if (alpha <= 0 || beta <= 0 || lambda <= 0) {
   Rcpp::NumericMatrix nanH(3,3);
   nanH.fill(R_NaN);
   return nanH;
 }
 
 arma::vec x = Rcpp::as<arma::vec>(data);
 if (x.n_elem < 1 || arma::any(x <= 0) || arma::any(x >= 1)) {
   Rcpp::NumericMatrix nanH(3,3);
   nanH.fill(R_NaN);
   return nanH;
 }
 
 int n = x.n_elem;
 arma::mat H(3,3, arma::fill::zeros);
 
 // Stability constants
 // const double eps = std::numeric_limits<double>::epsilon() * 100;
 const double min_v = 1e-10;  // Minimum value for v = 1-x^α
 const double min_w = 1e-10;  // Minimum value for w = 1-(1-x^α)^β
 const double exp_threshold = -700.0;  // Threshold for log-domain calculations
 
 // Constant terms (diagonal elements)
 H(0,0) = -n / (alpha * alpha);  // -n/α²
 H(1,1) = -n / (beta * beta);    // -n/β²
 H(2,2) = -n / (lambda * lambda); // -n/λ²
 
 // Special handling for lambda near 1 (critical case for stability)
 bool lambda_near_one = std::abs(lambda - 1.0) < 1e-8;
 
 for (int i = 0; i < n; i++) {
   double xi = x(i);
   double log_xi = std::log(xi);
   
   // Calculate x^α (A) and derivatives with log-domain for large alpha
   double A, dA_dalpha, d2A_dalpha2;
   if (alpha > 100.0 || (alpha * log_xi < exp_threshold)) {
     double log_A = alpha * log_xi;
     A = std::exp(log_A);
     dA_dalpha = A * log_xi;
     d2A_dalpha2 = A * log_xi * log_xi;
   } else {
     A = std::pow(xi, alpha);
     dA_dalpha = A * log_xi;
     d2A_dalpha2 = A * log_xi * log_xi;
   }
   
   // Calculate v = 1 - x^α with precision for x^α near 1
   double v;
   if (A > 0.9995) {
     v = -std::expm1(alpha * log_xi);  // More precise than 1.0 - A
   } else {
     v = 1.0 - A;
   }
   
   // Ensure v is not too small
   v = std::max(v, min_v);
   double log_v = std::log(v);
   
   double dv_dalpha = -dA_dalpha;
   double d2v_dalpha2 = -d2A_dalpha2;
   
   // L5 derivatives: (β-1) log(v)
   double d2L5_dalpha2 = 0.0;
   double d2L5_dalpha_dbeta = 0.0;
   
   if (beta != 1.0) {
     double v_squared = std::max(v * v, 1e-200); // Prevent division by zero
     d2L5_dalpha2 = (beta - 1.0) * ((d2v_dalpha2 * v - dv_dalpha * dv_dalpha) / v_squared);
     d2L5_dalpha_dbeta = dv_dalpha / v;
   }
   
   // Calculate v^β with log-domain for large beta
   double v_beta, v_beta_m1, v_beta_m2;
   if (beta > 100.0 || (beta * log_v < exp_threshold)) {
     v_beta = std::exp(beta * log_v);
     v_beta_m1 = std::exp((beta - 1.0) * log_v);
     v_beta_m2 = std::exp((beta - 2.0) * log_v);
   } else {
     v_beta = std::pow(v, beta);
     v_beta_m1 = std::pow(v, beta - 1.0);
     v_beta_m2 = std::pow(v, beta - 2.0);
   }
   
   // Calculate w = 1 - v^β precisely for v^β near 1
   double w;
   if (v_beta > 0.9995) {
     w = -std::expm1(beta * log_v);
   } else {
     w = 1.0 - v_beta;
   }
   
   w = std::max(w, min_w);
   double w_squared = std::max(w * w, 1e-200); // Prevent division by zero
   
   // First derivatives of w
   double dw_dv = -beta * v_beta_m1;
   double dw_dalpha = dw_dv * dv_dalpha;
   double dw_dbeta = -v_beta * log_v;
   
   // Second derivatives of w
   double d2w_dalpha2 = -beta * ((beta - 1.0) * v_beta_m2 * (dv_dalpha * dv_dalpha) +
                                 v_beta_m1 * d2v_dalpha2);
   double d2w_dbeta2 = -v_beta * (log_v * log_v);
   double d_dw_dalpha_dbeta = -v_beta_m1 * (1.0 + beta * log_v) * dv_dalpha;
   
   // L6 derivatives: (λ-1) log(w)
   double d2L6_dalpha2 = 0.0;
   double d2L6_dbeta2 = 0.0;
   double d2L6_dalpha_dbeta = 0.0;
   double d2L6_dalpha_dlambda = 0.0;
   double d2L6_dbeta_dlambda = 0.0;
   
   // Critical lambda handling for stability
   if (lambda_near_one) {
     // For lambda ≈ 1, handle carefully to avoid cancellation errors
     if (std::abs(lambda - 1.0) > 1e-15) {
       double factor = lambda - 1.0;
       d2L6_dalpha2 = factor * ((d2w_dalpha2 * w - (dw_dalpha * dw_dalpha)) / w_squared);
       d2L6_dbeta2 = factor * ((d2w_dbeta2 * w - (dw_dbeta * dw_dbeta)) / w_squared);
       d2L6_dalpha_dbeta = factor * ((d_dw_dalpha_dbeta / w) - (dw_dalpha * dw_dbeta) / w_squared);
     }
     // When λ = 1 (machine precision), these terms become zero
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
   
   // Handle large lambda values (> 1000)
   if (lambda > 1000.0) {
     double max_val = 1000.0;
     d2L6_dalpha2 = std::min(std::max(d2L6_dalpha2, -max_val), max_val);
     d2L6_dbeta2 = std::min(std::max(d2L6_dbeta2, -max_val), max_val);
     d2L6_dalpha_dbeta = std::min(std::max(d2L6_dalpha_dbeta, -max_val), max_val);
   }
   
   // Accumulate contributions to Hessian
   H(0,0) += d2L5_dalpha2 + d2L6_dalpha2;
   H(0,1) += d2L5_dalpha_dbeta + d2L6_dalpha_dbeta;
   H(1,0) = H(0,1);
   H(1,1) += d2L6_dbeta2;
   H(0,2) += d2L6_dalpha_dlambda;
   H(2,0) = H(0,2);
   H(1,2) += d2L6_dbeta_dlambda;
   H(2,1) = H(1,2);
 }
 
 // Ensure perfect symmetry by averaging
 for (int i = 0; i < 3; i++) {
   for (int j = i+1; j < 3; j++) {
     double avg = (H(i,j) + H(j,i)) / 2.0;
     H(i,j) = H(j,i) = avg;
   }
 }
 
 return Rcpp::wrap(-H);
}
