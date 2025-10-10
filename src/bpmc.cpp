// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include "utils.h"


/*
----------------------------------------------------------------------------
BETA POWER (BP) DISTRIBUTION: BP(γ, δ, λ)
----------------------------------------------------------------------------

This arises from GKw with α=1 and β=1, leaving three parameters: (γ>0, δ≥0, λ>0).

* PDF:
f(x; γ, δ, λ) = [ λ / B(γ, δ+1) ] * x^(γλ - 1) * (1 - x^λ)^δ,   0<x<1.

* CDF:
F(x; γ, δ, λ) = I_{x^λ}(γ, δ+1) = pbeta(x^λ, γ, δ+1).

* QUANTILE:
Q(p; γ, δ, λ) = [ qbeta(p, γ, δ+1) ]^(1/λ).

* RNG:
If U ~ Beta(γ, δ+1), then X = U^(1/λ).

* NEGATIVE LOG-LIKELIHOOD:
sum( -log f(x_i) )
where
log f(x) = log(λ) - log B(γ, δ+1)
+ (γ λ -1)* log(x)
+ δ * log(1 - x^λ).

We'll define five functions:
- dmc() : PDF
- pmc() : CDF
- qmc() : quantile
- rmc() : random generator
- llmc(): negative log-likelihood

We'll also define a param-checker for (γ, δ, λ).
*/

// -----------------------------------------------------------------------------
// 1) dmc: PDF of Beta Power McDonald
// -----------------------------------------------------------------------------

//' @title Density of the McDonald (Mc)/Beta Power Distribution Distribution
//' @author Lopes, J. E.
//' @keywords distribution density mcdonald
//'
//' @description
//' Computes the probability density function (PDF) for the McDonald (Mc)
//' distribution (also previously referred to as Beta Power) with parameters
//' \code{gamma} (\eqn{\gamma}), \code{delta} (\eqn{\delta}), and \code{lambda}
//' (\eqn{\lambda}). This distribution is defined on the interval (0, 1).
//'
//' @param x Vector of quantiles (values between 0 and 1).
//' @param gamma Shape parameter \code{gamma} > 0. Can be a scalar or a vector.
//'   Default: 1.0.
//' @param delta Shape parameter \code{delta} >= 0. Can be a scalar or a vector.
//'   Default: 0.0.
//' @param lambda Shape parameter \code{lambda} > 0. Can be a scalar or a vector.
//'   Default: 1.0.
//' @param log_prob Logical; if \code{TRUE}, the logarithm of the density is
//'   returned (\eqn{\log(f(x))}). Default: \code{FALSE}.
//'
//' @return A vector of density values (\eqn{f(x)}) or log-density values
//'   (\eqn{\log(f(x))}). The length of the result is determined by the recycling
//'   rule applied to the arguments (\code{x}, \code{gamma}, \code{delta},
//'   \code{lambda}). Returns \code{0} (or \code{-Inf} if
//'   \code{log_prob = TRUE}) for \code{x} outside the interval (0, 1), or
//'   \code{NaN} if parameters are invalid (e.g., \code{gamma <= 0},
//'   \code{delta < 0}, \code{lambda <= 0}).
//'
//' @details
//' The probability density function (PDF) of the McDonald (Mc) distribution
//' is given by:
//' \deqn{
//' f(x; \gamma, \delta, \lambda) = \frac{\lambda}{B(\gamma,\delta+1)} x^{\gamma \lambda - 1} (1 - x^\lambda)^\delta
//' }
//' for \eqn{0 < x < 1}, where \eqn{B(a,b)} is the Beta function
//' (\code{\link[base]{beta}}).
//'
//' The Mc distribution is a special case of the five-parameter
//' Generalized Kumaraswamy (GKw) distribution (\code{\link{dgkw}}) obtained
//' by setting the parameters \eqn{\alpha = 1} and \eqn{\beta = 1}.
//' It was introduced by McDonald (1984) and is related to the Generalized Beta
//' distribution of the first kind (GB1). When \eqn{\lambda=1}, it simplifies
//' to the standard Beta distribution with parameters \eqn{\gamma} and
//' \eqn{\delta+1}.
//'
//' @references
//' McDonald, J. B. (1984). Some generalized functions for the size distribution
//' of income. *Econometrica*, 52(3), 647-663.
//'
//' Cordeiro, G. M., & de Castro, M. (2011). A new family of generalized
//' distributions. *Journal of Statistical Computation and Simulation*,
//'
//'
//' Kumaraswamy, P. (1980). A generalized probability density function for
//' double-bounded random processes. *Journal of Hydrology*, *46*(1-2), 79-88.
//'
//'
//' @seealso
//' \code{\link{dgkw}} (parent distribution density),
//' \code{\link{pmc}}, \code{\link{qmc}}, \code{\link{rmc}} (other Mc functions),
//' \code{\link[stats]{dbeta}}
//'
//' @examples
//' \donttest{
//' # Example values
//' x_vals <- c(0.2, 0.5, 0.8)
//' gamma_par <- 2.0
//' delta_par <- 1.5
//' lambda_par <- 1.0 # Equivalent to Beta(gamma, delta+1)
//'
//' # Calculate density using dmc
//' densities <- dmc(x_vals, gamma_par, delta_par, lambda_par)
//' print(densities)
//' # Compare with Beta density
//' print(stats::dbeta(x_vals, shape1 = gamma_par, shape2 = delta_par + 1))
//'
//' # Calculate log-density
//' log_densities <- dmc(x_vals, gamma_par, delta_par, lambda_par, log_prob = TRUE)
//' print(log_densities)
//'
//' # Compare with dgkw setting alpha = 1, beta = 1
//' densities_gkw <- dgkw(x_vals, alpha = 1.0, beta = 1.0, gamma = gamma_par,
//'                       delta = delta_par, lambda = lambda_par)
//' print(paste("Max difference:", max(abs(densities - densities_gkw)))) # Should be near zero
//'
//' # Plot the density for different lambda values
//' curve_x <- seq(0.01, 0.99, length.out = 200)
//' curve_y1 <- dmc(curve_x, gamma = 2, delta = 3, lambda = 0.5)
//' curve_y2 <- dmc(curve_x, gamma = 2, delta = 3, lambda = 1.0) # Beta(2, 4)
//' curve_y3 <- dmc(curve_x, gamma = 2, delta = 3, lambda = 2.0)
//'
//' plot(curve_x, curve_y2, type = "l", main = "McDonald (Mc) Density (gamma=2, delta=3)",
//'      xlab = "x", ylab = "f(x)", col = "red", ylim = range(0, curve_y1, curve_y2, curve_y3))
//' lines(curve_x, curve_y1, col = "blue")
//' lines(curve_x, curve_y3, col = "green")
//' legend("topright", legend = c("lambda=0.5", "lambda=1.0 (Beta)", "lambda=2.0"),
//'        col = c("blue", "red", "green"), lty = 1, bty = "n")
//' }
//'
//' @export
// [[Rcpp::export]]
Rcpp::NumericVector dmc(
   const arma::vec& x,
   const Rcpp::NumericVector& gamma,
   const Rcpp::NumericVector& delta,
   const Rcpp::NumericVector& lambda,
   bool log_prob = false
) {
 arma::vec g_vec(gamma.begin(), gamma.size());
 arma::vec d_vec(delta.begin(), delta.size());
 arma::vec l_vec(lambda.begin(), lambda.size());
 
 size_t N= std::max({ x.n_elem, g_vec.n_elem, d_vec.n_elem, l_vec.n_elem });
 arma::vec out(N);
 
 // Pre-fill
 out.fill(log_prob ? R_NegInf : 0.0);
 
 for (size_t i=0; i<N; i++){
   double gg= g_vec[i % g_vec.n_elem];
   double dd= d_vec[i % d_vec.n_elem];
   double ll= l_vec[i % l_vec.n_elem];
   double xx= x[i % x.n_elem];
   
   if (!check_bp_pars(gg,dd,ll)) {
     // invalid => pdf=0 or logpdf=-Inf
     continue;
   }
   // domain
   if (xx<=0.0 || xx>=1.0 || !R_finite(xx)) {
     continue;
   }
   
   // log f(x)= log(λ) - log( B(γ, δ+1) )
   //           + (γλ -1)* log(x)
   //           + δ * log(1 - x^λ)
   double logB = R::lbeta(gg, dd+1.0);
   double logCst= std::log(ll) - logB;
   
   // (γ λ -1)* log(x)
   double exponent= gg*ll - 1.0;
   double lx= std::log(xx);
   double term1= exponent* lx;
   
   // δ * log(1 - x^λ)
   double x_pow_l= std::pow(xx, ll);
   if (x_pow_l>=1.0) {
     // => pdf=0
     continue;
   }
   double log_1_minus_xpow= std::log(1.0 - x_pow_l);
   double term2= dd * log_1_minus_xpow;
   
   double log_pdf= logCst + term1 + term2;
   if (log_prob) {
     out(i)= log_pdf;
   } else {
     out(i)= std::exp(log_pdf);
   }
 }
 
 return Rcpp::NumericVector(out.memptr(), out.memptr() + out.n_elem);
}


// -----------------------------------------------------------------------------
// 2) pmc: CDF of Beta Power
// -----------------------------------------------------------------------------

//' @title CDF of the McDonald (Mc)/Beta Power Distribution
//' @author Lopes, J. E.
//' @keywords distribution cumulative mcdonald
//'
//' @description
//' Computes the cumulative distribution function (CDF), \eqn{F(q) = P(X \le q)},
//' for the McDonald (Mc) distribution (also known as Beta Power) with
//' parameters \code{gamma} (\eqn{\gamma}), \code{delta} (\eqn{\delta}), and
//' \code{lambda} (\eqn{\lambda}). This distribution is defined on the interval
//' (0, 1) and is a special case of the Generalized Kumaraswamy (GKw)
//' distribution where \eqn{\alpha = 1} and \eqn{\beta = 1}.
//'
//' @param q Vector of quantiles (values generally between 0 and 1).
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
//' @return A vector of probabilities, \eqn{F(q)}, or their logarithms/complements
//'   depending on \code{lower_tail} and \code{log_p}. The length of the result
//'   is determined by the recycling rule applied to the arguments (\code{q},
//'   \code{gamma}, \code{delta}, \code{lambda}). Returns \code{0} (or \code{-Inf}
//'   if \code{log_p = TRUE}) for \code{q <= 0} and \code{1} (or \code{0} if
//'   \code{log_p = TRUE}) for \code{q >= 1}. Returns \code{NaN} for invalid
//'   parameters.
//'
//' @details
//' The McDonald (Mc) distribution is a special case of the five-parameter
//' Generalized Kumaraswamy (GKw) distribution (\code{\link{pgkw}}) obtained
//' by setting parameters \eqn{\alpha = 1} and \eqn{\beta = 1}.
//'
//' The CDF of the GKw distribution is \eqn{F_{GKw}(q) = I_{y(q)}(\gamma, \delta+1)},
//' where \eqn{y(q) = [1-(1-q^{\alpha})^{\beta}]^{\lambda}} and \eqn{I_x(a,b)}
//' is the regularized incomplete beta function (\code{\link[stats]{pbeta}}).
//' Setting \eqn{\alpha=1} and \eqn{\beta=1} simplifies \eqn{y(q)} to \eqn{q^\lambda},
//' yielding the Mc CDF:
//' \deqn{
//' F(q; \gamma, \delta, \lambda) = I_{q^\lambda}(\gamma, \delta+1)
//' }
//' This is evaluated using the \code{\link[stats]{pbeta}} function as
//' \code{pbeta(q^lambda, shape1 = gamma, shape2 = delta + 1)}.
//'
//' @references
//' McDonald, J. B. (1984). Some generalized functions for the size distribution
//' of income. *Econometrica*, 52(3), 647-663.
//'
//' Cordeiro, G. M., & de Castro, M. (2011). A new family of generalized
//' distributions. *Journal of Statistical Computation and Simulation*,
//'
//'
//' Kumaraswamy, P. (1980). A generalized probability density function for
//' double-bounded random processes. *Journal of Hydrology*, *46*(1-2), 79-88.
//'
//'
//' @seealso
//' \code{\link{pgkw}} (parent distribution CDF),
//' \code{\link{dmc}}, \code{\link{qmc}}, \code{\link{rmc}} (other Mc functions),
//' \code{\link[stats]{pbeta}}
//'
//' @examples
//' \donttest{
//' # Example values
//' q_vals <- c(0.2, 0.5, 0.8)
//' gamma_par <- 2.0
//' delta_par <- 1.5
//' lambda_par <- 1.0 # Equivalent to Beta(gamma, delta+1)
//'
//' # Calculate CDF P(X <= q) using pmc
//' probs <- pmc(q_vals, gamma_par, delta_par, lambda_par)
//' print(probs)
//' # Compare with Beta CDF
//' print(stats::pbeta(q_vals, shape1 = gamma_par, shape2 = delta_par + 1))
//'
//' # Calculate upper tail P(X > q)
//' probs_upper <- pmc(q_vals, gamma_par, delta_par, lambda_par,
//'                    lower_tail = FALSE)
//' print(probs_upper)
//' # Check: probs + probs_upper should be 1
//' print(probs + probs_upper)
//'
//' # Calculate log CDF
//' log_probs <- pmc(q_vals, gamma_par, delta_par, lambda_par, log_p = TRUE)
//' print(log_probs)
//' # Check: should match log(probs)
//' print(log(probs))
//'
//' # Compare with pgkw setting alpha = 1, beta = 1
//' probs_gkw <- pgkw(q_vals, alpha = 1.0, beta = 1.0, gamma = gamma_par,
//'                   delta = delta_par, lambda = lambda_par)
//' print(paste("Max difference:", max(abs(probs - probs_gkw)))) # Should be near zero
//'
//' # Plot the CDF for different lambda values
//' curve_q <- seq(0.01, 0.99, length.out = 200)
//' curve_p1 <- pmc(curve_q, gamma = 2, delta = 3, lambda = 0.5)
//' curve_p2 <- pmc(curve_q, gamma = 2, delta = 3, lambda = 1.0) # Beta(2, 4)
//' curve_p3 <- pmc(curve_q, gamma = 2, delta = 3, lambda = 2.0)
//'
//' plot(curve_q, curve_p2, type = "l", main = "Mc/Beta Power CDF (gamma=2, delta=3)",
//'      xlab = "q", ylab = "F(q)", col = "red", ylim = c(0, 1))
//' lines(curve_q, curve_p1, col = "blue")
//' lines(curve_q, curve_p3, col = "green")
//' legend("bottomright", legend = c("lambda=0.5", "lambda=1.0 (Beta)", "lambda=2.0"),
//'        col = c("blue", "red", "green"), lty = 1, bty = "n")
//' }
//'
//' @export
// [[Rcpp::export]]
Rcpp::NumericVector pmc(
   const arma::vec& q,
   const Rcpp::NumericVector& gamma,
   const Rcpp::NumericVector& delta,
   const Rcpp::NumericVector& lambda,
   bool lower_tail = true,
   bool log_p = false
) {
 arma::vec g_vec(gamma.begin(), gamma.size());
 arma::vec d_vec(delta.begin(), delta.size());
 arma::vec l_vec(lambda.begin(), lambda.size());
 
 size_t N= std::max({ q.n_elem, g_vec.n_elem, d_vec.n_elem, l_vec.n_elem });
 arma::vec out(N);
 
 for (size_t i=0; i<N; i++){
   double gg= g_vec[i % g_vec.n_elem];
   double dd= d_vec[i % d_vec.n_elem];
   double ll= l_vec[i % l_vec.n_elem];
   double xx= q[i % q.n_elem];
   
   if (!check_bp_pars(gg,dd,ll)) {
     out(i)= NA_REAL;
     continue;
   }
   
   // boundaries
   if (!R_finite(xx) || xx<=0.0) {
     double val0= (lower_tail ? 0.0 : 1.0);
     out(i)= log_p ? std::log(val0) : val0;
     continue;
   }
   if (xx>=1.0) {
     double val1= (lower_tail ? 1.0 : 0.0);
     out(i)= log_p ? std::log(val1) : val1;
     continue;
   }
   
   double xpow= std::pow(xx, ll);
   // pbeta(xpow, gg, dd+1, TRUE, FALSE)
   double val= R::pbeta( xpow, gg, dd+1.0, true, false );
   if (!lower_tail) {
     val= 1.0 - val;
   }
   if (log_p) {
     val= std::log(val);
   }
   out(i)= val;
 }
 
 return Rcpp::NumericVector(out.memptr(), out.memptr() + out.n_elem);
}


// -----------------------------------------------------------------------------
// 3) qmc: Quantile of Beta Power
// -----------------------------------------------------------------------------

//' @title Quantile Function of the McDonald (Mc)/Beta Power Distribution
//' @author Lopes, J. E.
//' @keywords distribution quantile mcdonald
//'
//' @description
//' Computes the quantile function (inverse CDF) for the McDonald (Mc) distribution
//' (also known as Beta Power) with parameters \code{gamma} (\eqn{\gamma}),
//' \code{delta} (\eqn{\delta}), and \code{lambda} (\eqn{\lambda}). It finds the
//' value \code{q} such that \eqn{P(X \le q) = p}. This distribution is a special
//' case of the Generalized Kumaraswamy (GKw) distribution where \eqn{\alpha = 1}
//' and \eqn{\beta = 1}.
//'
//' @param p Vector of probabilities (values between 0 and 1).
//' @param gamma Shape parameter \code{gamma} > 0. Can be a scalar or a vector.
//'   Default: 1.0.
//' @param delta Shape parameter \code{delta} >= 0. Can be a scalar or a vector.
//'   Default: 0.0.
//' @param lambda Shape parameter \code{lambda} > 0. Can be a scalar or a vector.
//'   Default: 1.0.
//' @param lower_tail Logical; if \code{TRUE} (default), probabilities are \eqn{p = P(X \le q)},
//'   otherwise, probabilities are \eqn{p = P(X > q)}.
//' @param log_p Logical; if \code{TRUE}, probabilities \code{p} are given as
//'   \eqn{\log(p)}. Default: \code{FALSE}.
//'
//' @return A vector of quantiles corresponding to the given probabilities \code{p}.
//'   The length of the result is determined by the recycling rule applied to
//'   the arguments (\code{p}, \code{gamma}, \code{delta}, \code{lambda}).
//'   Returns:
//'   \itemize{
//'     \item \code{0} for \code{p = 0} (or \code{p = -Inf} if \code{log_p = TRUE},
//'           when \code{lower_tail = TRUE}).
//'     \item \code{1} for \code{p = 1} (or \code{p = 0} if \code{log_p = TRUE},
//'           when \code{lower_tail = TRUE}).
//'     \item \code{NaN} for \code{p < 0} or \code{p > 1} (or corresponding log scale).
//'     \item \code{NaN} for invalid parameters (e.g., \code{gamma <= 0},
//'           \code{delta < 0}, \code{lambda <= 0}).
//'   }
//'   Boundary return values are adjusted accordingly for \code{lower_tail = FALSE}.
//'
//' @details
//' The quantile function \eqn{Q(p)} is the inverse of the CDF \eqn{F(q)}. The CDF
//' for the Mc (\eqn{\alpha=1, \beta=1}) distribution is \eqn{F(q) = I_{q^\lambda}(\gamma, \delta+1)},
//' where \eqn{I_z(a,b)} is the regularized incomplete beta function (see \code{\link{pmc}}).
//'
//' To find the quantile \eqn{q}, we first invert the Beta function part: let
//' \eqn{y = I^{-1}_{p}(\gamma, \delta+1)}, where \eqn{I^{-1}_p(a,b)} is the
//' inverse computed via \code{\link[stats]{qbeta}}. We then solve \eqn{q^\lambda = y}
//' for \eqn{q}, yielding the quantile function:
//' \deqn{
//' Q(p) = \left[ I^{-1}_{p}(\gamma, \delta+1) \right]^{1/\lambda}
//' }
//' The function uses this formula, calculating \eqn{I^{-1}_{p}(\gamma, \delta+1)}
//' via \code{qbeta(p, gamma, delta + 1, ...)} while respecting the
//' \code{lower_tail} and \code{log_p} arguments. This is equivalent to the general
//' GKw quantile function (\code{\link{qgkw}}) evaluated with \eqn{\alpha=1, \beta=1}.
//'
//' @references
//' McDonald, J. B. (1984). Some generalized functions for the size distribution
//' of income. *Econometrica*, 52(3), 647-663.
//'
//' Cordeiro, G. M., & de Castro, M. (2011). A new family of generalized
//' distributions. *Journal of Statistical Computation and Simulation*,
//'
//'
//' Kumaraswamy, P. (1980). A generalized probability density function for
//' double-bounded random processes. *Journal of Hydrology*, *46*(1-2), 79-88.
//'
//'
//' @seealso
//' \code{\link{qgkw}} (parent distribution quantile function),
//' \code{\link{dmc}}, \code{\link{pmc}}, \code{\link{rmc}} (other Mc functions),
//' \code{\link[stats]{qbeta}}
//'
//' @examples
//' \donttest{
//' # Example values
//' p_vals <- c(0.1, 0.5, 0.9)
//' gamma_par <- 2.0
//' delta_par <- 1.5
//' lambda_par <- 1.0 # Equivalent to Beta(gamma, delta+1)
//'
//' # Calculate quantiles using qmc
//' quantiles <- qmc(p_vals, gamma_par, delta_par, lambda_par)
//' print(quantiles)
//' # Compare with Beta quantiles
//' print(stats::qbeta(p_vals, shape1 = gamma_par, shape2 = delta_par + 1))
//'
//' # Calculate quantiles for upper tail probabilities P(X > q) = p
//' quantiles_upper <- qmc(p_vals, gamma_par, delta_par, lambda_par,
//'                        lower_tail = FALSE)
//' print(quantiles_upper)
//' # Check: qmc(p, ..., lt=F) == qmc(1-p, ..., lt=T)
//' print(qmc(1 - p_vals, gamma_par, delta_par, lambda_par))
//'
//' # Calculate quantiles from log probabilities
//' log_p_vals <- log(p_vals)
//' quantiles_logp <- qmc(log_p_vals, gamma_par, delta_par, lambda_par, log_p = TRUE)
//' print(quantiles_logp)
//' # Check: should match original quantiles
//' print(quantiles)
//'
//' # Compare with qgkw setting alpha = 1, beta = 1
//' quantiles_gkw <- qgkw(p_vals, alpha = 1.0, beta = 1.0, gamma = gamma_par,
//'                       delta = delta_par, lambda = lambda_par)
//' print(paste("Max difference:", max(abs(quantiles - quantiles_gkw)))) # Should be near zero
//'
//' # Verify inverse relationship with pmc
//' p_check <- 0.75
//' q_calc <- qmc(p_check, gamma_par, delta_par, lambda_par) # Use lambda != 1
//' p_recalc <- pmc(q_calc, gamma_par, delta_par, lambda_par)
//' print(paste("Original p:", p_check, " Recalculated p:", p_recalc))
//' # abs(p_check - p_recalc) < 1e-9 # Should be TRUE
//'
//' # Boundary conditions
//' print(qmc(c(0, 1), gamma_par, delta_par, lambda_par)) # Should be 0, 1
//' print(qmc(c(-Inf, 0), gamma_par, delta_par, lambda_par, log_p = TRUE)) # Should be 0, 1
//'
//' }
//'
//' @export
// [[Rcpp::export]]
Rcpp::NumericVector qmc(
   const arma::vec& p,
   const Rcpp::NumericVector& gamma,
   const Rcpp::NumericVector& delta,
   const Rcpp::NumericVector& lambda,
   bool lower_tail=true,
   bool log_p=false
) {
 arma::vec g_vec(gamma.begin(), gamma.size());
 arma::vec d_vec(delta.begin(), delta.size());
 arma::vec l_vec(lambda.begin(), lambda.size());
 
 size_t N= std::max({ p.n_elem, g_vec.n_elem, d_vec.n_elem, l_vec.n_elem });
 arma::vec out(N);
 
 for (size_t i=0; i<N; i++){
   double gg= g_vec[i % g_vec.n_elem];
   double dd= d_vec[i % d_vec.n_elem];
   double ll= l_vec[i % l_vec.n_elem];
   double pp= p[i % p.n_elem];
   
   if (!check_bp_pars(gg,dd,ll)) {
     out(i)= NA_REAL;
     continue;
   }
   
   // handle log_p
   if (log_p) {
     if (pp>0.0) {
       // log(p)>0 => p>1 => invalid
       out(i)= NA_REAL;
       continue;
     }
     pp= std::exp(pp);
   }
   // handle tail
   if (!lower_tail) {
     pp= 1.0 - pp;
   }
   
   // boundary
   if (pp<=0.0) {
     out(i)= 0.0;
     continue;
   }
   if (pp>=1.0) {
     out(i)= 1.0;
     continue;
   }
   
   // step1= R::qbeta(pp, gg, dd+1)
   double y= R::qbeta(pp, gg, dd+1.0, true, false);
   // step2= y^(1/λ)
   double xval;
   if (ll==1.0) {
     xval= y;
   } else {
     xval= std::pow(y, 1.0/ll);
   }
   if (!R_finite(xval) || xval<0.0) xval=0.0;
   if (xval>1.0) xval=1.0;
   out(i)= xval;
 }
 
 return Rcpp::NumericVector(out.memptr(), out.memptr() + out.n_elem);
}


// -----------------------------------------------------------------------------
// 4) rmc: RNG for Beta Power
// -----------------------------------------------------------------------------

//' @title Random Number Generation for the McDonald (Mc)/Beta Power Distribution
//' @author Lopes, J. E.
//' @keywords distribution random mcdonald
//'
//' @description
//' Generates random deviates from the McDonald (Mc) distribution (also known as
//' Beta Power) with parameters \code{gamma} (\eqn{\gamma}), \code{delta}
//' (\eqn{\delta}), and \code{lambda} (\eqn{\lambda}). This distribution is a
//' special case of the Generalized Kumaraswamy (GKw) distribution where
//' \eqn{\alpha = 1} and \eqn{\beta = 1}.
//'
//' @param n Number of observations. If \code{length(n) > 1}, the length is
//'   taken to be the number required. Must be a non-negative integer.
//' @param gamma Shape parameter \code{gamma} > 0. Can be a scalar or a vector.
//'   Default: 1.0.
//' @param delta Shape parameter \code{delta} >= 0. Can be a scalar or a vector.
//'   Default: 0.0.
//' @param lambda Shape parameter \code{lambda} > 0. Can be a scalar or a vector.
//'   Default: 1.0.
//'
//' @return A vector of length \code{n} containing random deviates from the Mc
//'   distribution, with values in (0, 1). The length of the result is determined
//'   by \code{n} and the recycling rule applied to the parameters (\code{gamma},
//'   \code{delta}, \code{lambda}). Returns \code{NaN} if parameters
//'   are invalid (e.g., \code{gamma <= 0}, \code{delta < 0}, \code{lambda <= 0}).
//'
//' @details
//' The generation method uses the relationship between the GKw distribution and the
//' Beta distribution. The general procedure for GKw (\code{\link{rgkw}}) is:
//' If \eqn{W \sim \mathrm{Beta}(\gamma, \delta+1)}, then
//' \eqn{X = \{1 - [1 - W^{1/\lambda}]^{1/\beta}\}^{1/\alpha}} follows the
//' GKw(\eqn{\alpha, \beta, \gamma, \delta, \lambda}) distribution.
//'
//' For the Mc distribution, \eqn{\alpha=1} and \eqn{\beta=1}. Therefore, the
//' algorithm simplifies significantly:
//' \enumerate{
//'   \item Generate \eqn{U \sim \mathrm{Beta}(\gamma, \delta+1)} using
//'         \code{\link[stats]{rbeta}}.
//'   \item Compute the Mc variate \eqn{X = U^{1/\lambda}}.
//' }
//' This procedure is implemented efficiently, handling parameter recycling as needed.
//'
//' @references
//' McDonald, J. B. (1984). Some generalized functions for the size distribution
//' of income. *Econometrica*, 52(3), 647-663.
//'
//' Cordeiro, G. M., & de Castro, M. (2011). A new family of generalized
//' distributions. *Journal of Statistical Computation and Simulation*,
//'
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
//' \code{\link{dmc}}, \code{\link{pmc}}, \code{\link{qmc}} (other Mc functions),
//' \code{\link[stats]{rbeta}}
//'
//' @examples
//' \donttest{
//' set.seed(2028) # for reproducibility
//'
//' # Generate 1000 random values from a specific Mc distribution
//' gamma_par <- 2.0
//' delta_par <- 1.5
//' lambda_par <- 1.0 # Equivalent to Beta(gamma, delta+1)
//'
//' x_sample_mc <- rmc(1000, gamma = gamma_par, delta = delta_par,
//'                    lambda = lambda_par)
//' summary(x_sample_mc)
//'
//' # Histogram of generated values compared to theoretical density
//' hist(x_sample_mc, breaks = 30, freq = FALSE, # freq=FALSE for density
//'      main = "Histogram of Mc Sample (Beta Case)", xlab = "x")
//' curve(dmc(x, gamma = gamma_par, delta = delta_par, lambda = lambda_par),
//'       add = TRUE, col = "red", lwd = 2, n = 201)
//' curve(stats::dbeta(x, gamma_par, delta_par + 1), add=TRUE, col="blue", lty=2)
//' legend("topright", legend = c("Theoretical Mc PDF", "Theoretical Beta PDF"),
//'        col = c("red", "blue"), lwd = c(2,1), lty=c(1,2), bty = "n")
//'
//' # Comparing empirical and theoretical quantiles (Q-Q plot)
//' lambda_par_qq <- 0.7 # Use lambda != 1 for non-Beta case
//' x_sample_mc_qq <- rmc(1000, gamma = gamma_par, delta = delta_par,
//'                       lambda = lambda_par_qq)
//' prob_points <- seq(0.01, 0.99, by = 0.01)
//' theo_quantiles <- qmc(prob_points, gamma = gamma_par, delta = delta_par,
//'                       lambda = lambda_par_qq)
//' emp_quantiles <- quantile(x_sample_mc_qq, prob_points, type = 7)
//'
//' plot(theo_quantiles, emp_quantiles, pch = 16, cex = 0.8,
//'      main = "Q-Q Plot for Mc Distribution",
//'      xlab = "Theoretical Quantiles", ylab = "Empirical Quantiles (n=1000)")
//' abline(a = 0, b = 1, col = "blue", lty = 2)
//'
//' # Compare summary stats with rgkw(..., alpha=1, beta=1, ...)
//' # Note: individual values will differ due to randomness
//' x_sample_gkw <- rgkw(1000, alpha = 1.0, beta = 1.0, gamma = gamma_par,
//'                      delta = delta_par, lambda = lambda_par_qq)
//' print("Summary stats for rmc sample:")
//' print(summary(x_sample_mc_qq))
//' print("Summary stats for rgkw(alpha=1, beta=1) sample:")
//' print(summary(x_sample_gkw)) # Should be similar
//'
//' }
//'
//' @export
// [[Rcpp::export]]
Rcpp::NumericVector rmc(
   int n,
   const Rcpp::NumericVector& gamma,
   const Rcpp::NumericVector& delta,
   const Rcpp::NumericVector& lambda
) {
 if (n<=0) {
   Rcpp::stop("rmc: n must be positive");
 }
 
 arma::vec g_vec(gamma.begin(), gamma.size());
 arma::vec d_vec(delta.begin(), delta.size());
 arma::vec l_vec(lambda.begin(), lambda.size());
 
 size_t k= std::max({ g_vec.n_elem, d_vec.n_elem, l_vec.n_elem });
 arma::vec out(n);
 
 for(int i=0; i<n; i++){
   size_t idx= i%k;
   double gg= g_vec[idx % g_vec.n_elem];
   double dd= d_vec[idx % d_vec.n_elem];
   double ll= l_vec[idx % l_vec.n_elem];
   
   if(!check_bp_pars(gg,dd,ll)) {
     out(i)= NA_REAL;
     Rcpp::warning("rmc: invalid parameters at index %d", i+1);
     continue;
   }
   
   double U= R::rbeta(gg, dd+1.0);
   double xval;
   if (ll==1.0) {
     xval= U;
   } else {
     xval= std::pow(U, 1.0/ll);
   }
   if (!R_finite(xval) || xval<0.0) xval=0.0;
   if (xval>1.0) xval=1.0;
   out(i)= xval;
 }
 
 return Rcpp::NumericVector(out.memptr(), out.memptr() + out.n_elem);
}


// -----------------------------------------------------------------------------
// 5) llmc: Negative Log-Likelihood for Beta Power
// -----------------------------------------------------------------------------


//' @title Negative Log-Likelihood for the McDonald (Mc)/Beta Power Distribution
//' @author Lopes, J. E.
//' @keywords distribution likelihood optimize mcdonald
//'
//' @description
//' Computes the negative log-likelihood function for the McDonald (Mc)
//' distribution (also known as Beta Power) with parameters \code{gamma}
//' (\eqn{\gamma}), \code{delta} (\eqn{\delta}), and \code{lambda} (\eqn{\lambda}),
//' given a vector of observations. This distribution is the special case of the
//' Generalized Kumaraswamy (GKw) distribution where \eqn{\alpha = 1} and
//' \eqn{\beta = 1}. This function is suitable for maximum likelihood estimation.
//'
//' @param par A numeric vector of length 3 containing the distribution parameters
//'   in the order: \code{gamma} (\eqn{\gamma > 0}), \code{delta} (\eqn{\delta \ge 0}),
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
//' The McDonald (Mc) distribution is the GKw distribution (\code{\link{dmc}})
//' with \eqn{\alpha=1} and \eqn{\beta=1}. Its probability density function (PDF) is:
//' \deqn{
//' f(x | \theta) = \frac{\lambda}{B(\gamma,\delta+1)} x^{\gamma \lambda - 1} (1 - x^\lambda)^\delta
//' }
//' for \eqn{0 < x < 1}, \eqn{\theta = (\gamma, \delta, \lambda)}, and \eqn{B(a,b)}
//' is the Beta function (\code{\link[base]{beta}}).
//' The log-likelihood function \eqn{\ell(\theta | \mathbf{x})} for a sample
//' \eqn{\mathbf{x} = (x_1, \dots, x_n)} is \eqn{\sum_{i=1}^n \ln f(x_i | \theta)}:
//' \deqn{
//' \ell(\theta | \mathbf{x}) = n[\ln(\lambda) - \ln B(\gamma, \delta+1)]
//' + \sum_{i=1}^{n} [(\gamma\lambda - 1)\ln(x_i) + \delta\ln(1 - x_i^\lambda)]
//' }
//' This function computes and returns the *negative* log-likelihood, \eqn{-\ell(\theta|\mathbf{x})},
//' suitable for minimization using optimization routines like \code{\link[stats]{optim}}.
//' Numerical stability is maintained, including using the log-gamma function
//' (\code{\link[base]{lgamma}}) for the Beta function term.
//'
//' @references
//' McDonald, J. B. (1984). Some generalized functions for the size distribution
//' of income. *Econometrica*, 52(3), 647-663.
//'
//' Cordeiro, G. M., & de Castro, M. (2011). A new family of generalized
//' distributions. *Journal of Statistical Computation and Simulation*,
//'
//'
//' Kumaraswamy, P. (1980). A generalized probability density function for
//' double-bounded random processes. *Journal of Hydrology*, *46*(1-2), 79-88.
//'
//'
//' @seealso
//' \code{\link{llgkw}} (parent distribution negative log-likelihood),
//' \code{\link{dmc}}, \code{\link{pmc}}, \code{\link{qmc}}, \code{\link{rmc}},
//' \code{grmc} (gradient, if available),
//' \code{hsmc} (Hessian, if available),
//' \code{\link[stats]{optim}}, \code{\link[base]{lbeta}}
//'
//' @examples
//' \donttest{
//' # Assuming existence of rmc, grmc, hsmc functions for Mc distribution
//'
//' # Generate sample data from a known Mc distribution
//' set.seed(123)
//' true_par_mc <- c(gamma = 2, delta = 3, lambda = 0.5)
//' # Use rmc for data generation
//' sample_data_mc <- rmc(100, gamma = true_par_mc[1], delta = true_par_mc[2],
//'                       lambda = true_par_mc[3])
//' hist(sample_data_mc, breaks = 20, main = "Mc(2, 3, 0.5) Sample")
//'
//' # --- Maximum Likelihood Estimation using optim ---
//' # Initial parameter guess
//' start_par_mc <- c(1.5, 2.5, 0.8)
//'
//' # Perform optimization (minimizing negative log-likelihood)
//' mle_result_mc <- stats::optim(par = start_par_mc,
//'                               fn = llmc, # Use the Mc neg-log-likelihood
//'                               method = "BFGS", # Or "L-BFGS-B" with lower=1e-6
//'                               hessian = TRUE,
//'                               data = sample_data_mc)
//'
//' # Check convergence and results
//' if (mle_result_mc$convergence == 0) {
//'   print("Optimization converged successfully.")
//'   mle_par_mc <- mle_result_mc$par
//'   print("Estimated Mc parameters:")
//'   print(mle_par_mc)
//'   print("True Mc parameters:")
//'   print(true_par_mc)
//' } else {
//'   warning("Optimization did not converge!")
//'   print(mle_result_mc$message)
//' }
//'
//' # --- Compare numerical and analytical derivatives (if available) ---
//' # Requires 'numDeriv' package and analytical functions 'grmc', 'hsmc'
//' if (mle_result_mc$convergence == 0 &&
//'     requireNamespace("numDeriv", quietly = TRUE) &&
//'     exists("grmc") && exists("hsmc")) {
//'
//'   cat("\nComparing Derivatives at Mc MLE estimates:\n")
//'
//'   # Numerical derivatives of llmc
//'   num_grad_mc <- numDeriv::grad(func = llmc, x = mle_par_mc, data = sample_data_mc)
//'   num_hess_mc <- numDeriv::hessian(func = llmc, x = mle_par_mc, data = sample_data_mc)
//'
//'   # Analytical derivatives (assuming they return derivatives of negative LL)
//'   ana_grad_mc <- grmc(par = mle_par_mc, data = sample_data_mc)
//'   ana_hess_mc <- hsmc(par = mle_par_mc, data = sample_data_mc)
//'
//'   # Check differences
//'   cat("Max absolute difference between gradients:\n")
//'   print(max(abs(num_grad_mc - ana_grad_mc)))
//'   cat("Max absolute difference between Hessians:\n")
//'   print(max(abs(num_hess_mc - ana_hess_mc)))
//'
//' } else {
//'    cat("\nSkipping derivative comparison for Mc.\n")
//'    cat("Requires convergence, 'numDeriv' package and functions 'grmc', 'hsmc'.\n")
//' }
//'
//' }
//'
//' @export
// [[Rcpp::export]]
double llmc(const Rcpp::NumericVector& par, const Rcpp::NumericVector& data) {
 // Parameter validation
 if (par.size() < 3) return R_PosInf;
 
 double gamma = par[0];
 double delta = par[1];
 double lambda = par[2];
 
 if (!check_bp_pars(gamma, delta, lambda)) return R_PosInf;
 
 arma::vec x = Rcpp::as<arma::vec>(data);
 if (x.n_elem < 1) return R_PosInf;
 
 // Data boundary check
 if (arma::any(x <= 0.0) || arma::any(x >= 1.0)) return R_PosInf;
 
 int n = x.n_elem;
 double loglike = 0.0;
 
 // Stability constants
 const double eps = 1e-10;
 // const double exp_threshold = -700.0;
 
 // Compute log(Beta(gamma, delta+1)) stably
 double log_B;
 if (gamma > 100.0 || delta > 100.0) {
   // For large parameters, use Stirling's approximation
   log_B = lgamma(gamma) + lgamma(delta + 1.0) - lgamma(gamma + delta + 1.0);
 } else {
   log_B = R::lbeta(gamma, delta + 1.0);
 }
 
 // Constant term: n*(log(lambda) - log(B(gamma, delta+1)))
 double log_lambda = safe_log(lambda);
 double const_term = n * (log_lambda - log_B);
 
 // Calculate gamma*lambda - 1.0 with precision for values near 1.0
 double gl_minus_1 = gamma * lambda - 1.0;
 
 // Initialize accumulators for sum terms
 double sum_term1 = 0.0;  // Sum of (gamma*lambda-1)*log(x)
 double sum_term2 = 0.0;  // Sum of delta*log(1-x^lambda)
 
 // Process each observation with careful numerical treatment
 for (int i = 0; i < n; i++) {
   double xi = x(i);
   
   // Handle observations near boundaries
   if (xi < eps) xi = eps;
   if (xi > 1.0 - eps) xi = 1.0 - eps;
   
   double log_xi = std::log(xi);
   
   // Term 1: (gamma*lambda-1)*log(x)
   // Special handling for large gamma*lambda
   sum_term1 += gl_minus_1 * log_xi;
   
   // Calculate x^lambda stably
   double x_lambda;
   if (lambda * std::abs(log_xi) > 1.0) {
     // Use log domain for potential overflow/underflow
     x_lambda = std::exp(lambda * log_xi);
   } else {
     x_lambda = std::pow(xi, lambda);
   }
   
   // Term 2: delta*log(1-x^lambda)
   // Use log1p for x^lambda close to 1 for better precision
   double log_1_minus_x_lambda;
   if (x_lambda > 0.9995) {
     // For x^lambda near 1, use complementary calculation
     log_1_minus_x_lambda = log1p(-x_lambda);
   } else {
     log_1_minus_x_lambda = std::log(1.0 - x_lambda);
   }
   
   // Special handling for large delta values
   if (delta > 1000.0 && log_1_minus_x_lambda < -0.01) {
     // Scale to prevent overflow with large delta
     double scaled_term = std::max(log_1_minus_x_lambda, -700.0 / delta);
     sum_term2 += delta * scaled_term;
   } else {
     sum_term2 += delta * log_1_minus_x_lambda;
   }
 }
 
 loglike = const_term + sum_term1 + sum_term2;
 
 // Check for invalid results
 if (!std::isfinite(loglike)) return R_PosInf;
 
 return -loglike;  // Return negative log-likelihood
}




//' @title Gradient of the Negative Log-Likelihood for the McDonald (Mc)/Beta Power Distribution
//' @author Lopes, J. E.
//' @keywords distribution likelihood optimize gradient mcdonald
//'
//' @description
//' Computes the gradient vector (vector of first partial derivatives) of the
//' negative log-likelihood function for the McDonald (Mc) distribution (also
//' known as Beta Power) with parameters \code{gamma} (\eqn{\gamma}), \code{delta}
//' (\eqn{\delta}), and \code{lambda} (\eqn{\lambda}). This distribution is the
//' special case of the Generalized Kumaraswamy (GKw) distribution where
//' \eqn{\alpha = 1} and \eqn{\beta = 1}. The gradient is useful for optimization.
//'
//' @param par A numeric vector of length 3 containing the distribution parameters
//'   in the order: \code{gamma} (\eqn{\gamma > 0}), \code{delta} (\eqn{\delta \ge 0}),
//'   \code{lambda} (\eqn{\lambda > 0}).
//' @param data A numeric vector of observations. All values must be strictly
//'   between 0 and 1 (exclusive).
//'
//' @return Returns a numeric vector of length 3 containing the partial derivatives
//'   of the negative log-likelihood function \eqn{-\ell(\theta | \mathbf{x})} with
//'   respect to each parameter:
//'   \eqn{(-\partial \ell/\partial \gamma, -\partial \ell/\partial \delta, -\partial \ell/\partial \lambda)}.
//'   Returns a vector of \code{NaN} if any parameter values are invalid according
//'   to their constraints, or if any value in \code{data} is not in the
//'   interval (0, 1).
//'
//' @details
//' The components of the gradient vector of the negative log-likelihood
//' (\eqn{-\nabla \ell(\theta | \mathbf{x})}) for the Mc (\eqn{\alpha=1, \beta=1})
//' model are:
//'
//' \deqn{
//' -\frac{\partial \ell}{\partial \gamma} = n[\psi(\gamma+\delta+1) - \psi(\gamma)] -
//' \lambda\sum_{i=1}^{n}\ln(x_i)
//' }
//' \deqn{
//' -\frac{\partial \ell}{\partial \delta} = n[\psi(\gamma+\delta+1) - \psi(\delta+1)] -
//' \sum_{i=1}^{n}\ln(1-x_i^{\lambda})
//' }
//' \deqn{
//' -\frac{\partial \ell}{\partial \lambda} = -\frac{n}{\lambda} - \gamma\sum_{i=1}^{n}\ln(x_i) +
//' \delta\sum_{i=1}^{n}\frac{x_i^{\lambda}\ln(x_i)}{1-x_i^{\lambda}}
//' }
//'
//' where \eqn{\psi(\cdot)} is the digamma function (\code{\link[base]{digamma}}).
//' These formulas represent the derivatives of \eqn{-\ell(\theta)}, consistent with
//' minimizing the negative log-likelihood. They correspond to the relevant components
//' of the general GKw gradient (\code{\link{grgkw}}) evaluated at \eqn{\alpha=1, \beta=1}.
//'
//' @references
//' McDonald, J. B. (1984). Some generalized functions for the size distribution
//' of income. *Econometrica*, 52(3), 647-663.
//'
//' Cordeiro, G. M., & de Castro, M. (2011). A new family of generalized
//' distributions. *Journal of Statistical Computation and Simulation*,
//'
//' (Note: Specific gradient formulas might be derived or sourced from additional references).
//'
//' @seealso
//' \code{\link{grgkw}} (parent distribution gradient),
//' \code{\link{llmc}} (negative log-likelihood for Mc),
//' \code{hsmc} (Hessian for Mc, if available),
//' \code{\link{dmc}} (density for Mc),
//' \code{\link[stats]{optim}},
//' \code{\link[numDeriv]{grad}} (for numerical gradient comparison),
//' \code{\link[base]{digamma}}.
//'
//' @examples
//' \donttest{
//' # Assuming existence of rmc, llmc, grmc, hsmc functions for Mc distribution
//'
//' # Generate sample data
//' set.seed(123)
//' true_par_mc <- c(gamma = 2, delta = 3, lambda = 0.5)
//' sample_data_mc <- rmc(100, gamma = true_par_mc[1], delta = true_par_mc[2],
//'                       lambda = true_par_mc[3])
//' hist(sample_data_mc, breaks = 20, main = "Mc(2, 3, 0.5) Sample")
//'
//' # --- Find MLE estimates ---
//' start_par_mc <- c(1.5, 2.5, 0.8)
//' mle_result_mc <- stats::optim(par = start_par_mc,
//'                               fn = llmc,
//'                               gr = grmc, # Use analytical gradient for Mc
//'                               method = "BFGS",
//'                               hessian = TRUE,
//'                               data = sample_data_mc)
//'
//' # --- Compare analytical gradient to numerical gradient ---
//' if (mle_result_mc$convergence == 0 &&
//'     requireNamespace("numDeriv", quietly = TRUE)) {
//'
//'   mle_par_mc <- mle_result_mc$par
//'   cat("\nComparing Gradients for Mc at MLE estimates:\n")
//'
//'   # Numerical gradient of llmc
//'   num_grad_mc <- numDeriv::grad(func = llmc, x = mle_par_mc, data = sample_data_mc)
//'
//'   # Analytical gradient from grmc
//'   ana_grad_mc <- grmc(par = mle_par_mc, data = sample_data_mc)
//'
//'   cat("Numerical Gradient (Mc):\n")
//'   print(num_grad_mc)
//'   cat("Analytical Gradient (Mc):\n")
//'   print(ana_grad_mc)
//'
//'   # Check differences
//'   cat("Max absolute difference between Mc gradients:\n")
//'   print(max(abs(num_grad_mc - ana_grad_mc)))
//'
//' } else {
//'   cat("\nSkipping Mc gradient comparison.\n")
//' }
//'
//' # Example with Hessian comparison (if hsmc exists)
//' if (mle_result_mc$convergence == 0 &&
//'     requireNamespace("numDeriv", quietly = TRUE) && exists("hsmc")) {
//'
//'   num_hess_mc <- numDeriv::hessian(func = llmc, x = mle_par_mc, data = sample_data_mc)
//'   ana_hess_mc <- hsmc(par = mle_par_mc, data = sample_data_mc)
//'   cat("\nMax absolute difference between Mc Hessians:\n")
//'   print(max(abs(num_hess_mc - ana_hess_mc)))
//'
//' }
//'
//' }
//'
//' @export
// [[Rcpp::export]]
Rcpp::NumericVector grmc(const Rcpp::NumericVector& par, const Rcpp::NumericVector& data) {
 // Parameter validation
 if (par.size() < 3) {
   Rcpp::NumericVector grad(3, R_NaN);
   return grad;
 }
 
 double gamma = par[0];
 double delta = par[1];
 double lambda = par[2];
 
 if (gamma <= 0 || delta < 0 || lambda <= 0) {
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
 
 // Stability constants
 const double eps = 1e-10;
 
 // Calculate digamma terms stably
 double digamma_gamma_delta_plus_1, digamma_gamma, digamma_delta_plus_1;
 
 // For large arguments, use asymptotic approximation of digamma
 if (gamma + delta > 100.0) {
   digamma_gamma_delta_plus_1 = std::log(gamma + delta + 1.0) - 1.0/(2.0*(gamma + delta + 1.0));
 } else {
   digamma_gamma_delta_plus_1 = R::digamma(gamma + delta + 1.0);
 }
 
 if (gamma > 100.0) {
   digamma_gamma = std::log(gamma) - 1.0/(2.0*gamma);
 } else {
   digamma_gamma = R::digamma(gamma);
 }
 
 if (delta > 100.0) {
   digamma_delta_plus_1 = std::log(delta + 1.0) - 1.0/(2.0*(delta + 1.0));
 } else {
   digamma_delta_plus_1 = R::digamma(delta + 1.0);
 }
 
 // Initialize accumulators
 double sum_log_x = 0.0;
 double sum_log_v = 0.0;
 double sum_term_lambda = 0.0;
 
 for (int i = 0; i < n; i++) {
   double xi = x(i);
   
   // Handle boundary values
   if (xi < eps) xi = eps;
   if (xi > 1.0 - eps) xi = 1.0 - eps;
   
   double log_xi = std::log(xi);
   sum_log_x += log_xi;
   
   // Calculate x^lambda stably
   double x_lambda;
   if (lambda > 100.0 || lambda * std::abs(log_xi) > 1.0) {
     x_lambda = std::exp(lambda * log_xi);
   } else {
     x_lambda = std::pow(xi, lambda);
   }
   
   // Calculate 1-x^lambda with precision for x^lambda near 1
   double v;
   if (x_lambda > 0.9995) {
     v = -std::expm1(lambda * log_xi);  // More precise than 1.0 - x_lambda
   } else {
     v = 1.0 - x_lambda;
   }
   
   // Ensure v is not too small
   v = std::max(v, eps);
   double log_v = std::log(v);
   sum_log_v += log_v;
   
   // Calculate term for lambda gradient: (x^lambda * log(x)) / (1-x^lambda)
   double lambda_term = (x_lambda * log_xi) / v;
   
   // Prevent extreme values that might lead to instability
   if (std::abs(lambda_term) > 1e6) {
     lambda_term = std::copysign(1e6, lambda_term);
   }
   
   sum_term_lambda += lambda_term;
 }
 
 // Compute gradient components
 double d_gamma = -n * (digamma_gamma_delta_plus_1 - digamma_gamma) - lambda * sum_log_x;
 double d_delta = -n * (digamma_gamma_delta_plus_1 - digamma_delta_plus_1) - sum_log_v;
 double d_lambda = -n / lambda - gamma * sum_log_x + delta * sum_term_lambda;
 
 // Alread negative gradient for negative log-likelihood
 grad[0] = d_gamma;
 grad[1] = d_delta;
 grad[2] = d_lambda;
 
 return grad;
}



//' @title Hessian Matrix of the Negative Log-Likelihood for the McDonald (Mc)/Beta Power Distribution
//' @author Lopes, J. E.
//' @keywords distribution likelihood optimize hessian mcdonald
//'
//' @description
//' Computes the analytic 3x3 Hessian matrix (matrix of second partial derivatives)
//' of the negative log-likelihood function for the McDonald (Mc) distribution
//' (also known as Beta Power) with parameters \code{gamma} (\eqn{\gamma}),
//' \code{delta} (\eqn{\delta}), and \code{lambda} (\eqn{\lambda}). This distribution
//' is the special case of the Generalized Kumaraswamy (GKw) distribution where
//' \eqn{\alpha = 1} and \eqn{\beta = 1}. The Hessian is useful for estimating
//' standard errors and in optimization algorithms.
//'
//' @param par A numeric vector of length 3 containing the distribution parameters
//'   in the order: \code{gamma} (\eqn{\gamma > 0}), \code{delta} (\eqn{\delta \ge 0}),
//'   \code{lambda} (\eqn{\lambda > 0}).
//' @param data A numeric vector of observations. All values must be strictly
//'   between 0 and 1 (exclusive).
//'
//' @return Returns a 3x3 numeric matrix representing the Hessian matrix of the
//'   negative log-likelihood function, \eqn{-\partial^2 \ell / (\partial \theta_i \partial \theta_j)},
//'   where \eqn{\theta = (\gamma, \delta, \lambda)}.
//'   Returns a 3x3 matrix populated with \code{NaN} if any parameter values are
//'   invalid according to their constraints, or if any value in \code{data} is
//'   not in the interval (0, 1).
//'
//' @details
//' This function calculates the analytic second partial derivatives of the
//' negative log-likelihood function (\eqn{-\ell(\theta|\mathbf{x})}).
//' The components are based on the second derivatives of the log-likelihood \eqn{\ell}
//' (derived from the PDF in \code{\link{dmc}}).
//'
//' **Note:** The formulas below represent the second derivatives of the positive
//' log-likelihood (\eqn{\ell}). The function returns the **negative** of these values.
//' Users should verify these formulas independently if using for critical applications.
//'
//' \deqn{
//' \frac{\partial^2 \ell}{\partial \gamma^2} = -n[\psi'(\gamma) - \psi'(\gamma+\delta+1)]
//' }
//' \deqn{
//' \frac{\partial^2 \ell}{\partial \gamma \partial \delta} = -n\psi'(\gamma+\delta+1)
//' }
//' \deqn{
//' \frac{\partial^2 \ell}{\partial \gamma \partial \lambda} = \sum_{i=1}^{n}\ln(x_i)
//' }
//' \deqn{
//' \frac{\partial^2 \ell}{\partial \delta^2} = -n[\psi'(\delta+1) - \psi'(\gamma+\delta+1)]
//' }
//' \deqn{
//' \frac{\partial^2 \ell}{\partial \delta \partial \lambda} = -\sum_{i=1}^{n}\frac{x_i^{\lambda}\ln(x_i)}{1-x_i^{\lambda}}
//' }
//' \deqn{
//' \frac{\partial^2 \ell}{\partial \lambda^2} = -\frac{n}{\lambda^2} -
//' \delta\sum_{i=1}^{n}\frac{x_i^{\lambda}[\ln(x_i)]^2}{(1-x_i^{\lambda})^2}
//' }
//'
//' where \eqn{\psi'(\cdot)} is the trigamma function (\code{\link[base]{trigamma}}).
//' (*Note: The formula for \eqn{\partial^2 \ell / \partial \lambda^2} provided in the source
//' comment was different and potentially related to the expected information matrix;
//' the formula shown here is derived from the gradient provided earlier. Verification
//' is recommended.*)
//'
//' The returned matrix is symmetric, with rows/columns corresponding to
//' \eqn{\gamma, \delta, \lambda}.
//'
//' @references
//' McDonald, J. B. (1984). Some generalized functions for the size distribution
//' of income. *Econometrica*, 52(3), 647-663.
//'
//' Cordeiro, G. M., & de Castro, M. (2011). A new family of generalized
//' distributions. *Journal of Statistical Computation and Simulation*,
//'
//' (Note: Specific Hessian formulas might be derived or sourced from additional references).
//'
//' @seealso
//' \code{\link{hsgkw}} (parent distribution Hessian),
//' \code{\link{llmc}} (negative log-likelihood for Mc),
//' \code{\link{grmc}} (gradient for Mc, if available),
//' \code{\link{dmc}} (density for Mc),
//' \code{\link[stats]{optim}},
//' \code{\link[numDeriv]{hessian}} (for numerical Hessian comparison),
//' \code{\link[base]{trigamma}}.
//'
//' @examples
//' \donttest{
//' # Assuming existence of rmc, llmc, grmc, hsmc functions for Mc distribution
//'
//' # Generate sample data
//' set.seed(123)
//' true_par_mc <- c(gamma = 2, delta = 3, lambda = 0.5)
//' sample_data_mc <- rmc(100, gamma = true_par_mc[1], delta = true_par_mc[2],
//'                       lambda = true_par_mc[3])
//' hist(sample_data_mc, breaks = 20, main = "Mc(2, 3, 0.5) Sample")
//'
//' # --- Find MLE estimates ---
//' start_par_mc <- c(1.5, 2.5, 0.8)
//' mle_result_mc <- stats::optim(par = start_par_mc,
//'                               fn = llmc,
//'                               gr = if (exists("grmc")) grmc else NULL,
//'                               method = "BFGS",
//'                               hessian = TRUE, # Ask optim for numerical Hessian
//'                               data = sample_data_mc)
//'
//' # --- Compare analytical Hessian to numerical Hessian ---
//' if (mle_result_mc$convergence == 0 &&
//'     requireNamespace("numDeriv", quietly = TRUE) &&
//'     exists("hsmc")) {
//'
//'   mle_par_mc <- mle_result_mc$par
//'   cat("\nComparing Hessians for Mc at MLE estimates:\n")
//'
//'   # Numerical Hessian of llmc
//'   num_hess_mc <- numDeriv::hessian(func = llmc, x = mle_par_mc, data = sample_data_mc)
//'
//'   # Analytical Hessian from hsmc
//'   ana_hess_mc <- hsmc(par = mle_par_mc, data = sample_data_mc)
//'
//'   cat("Numerical Hessian (Mc):\n")
//'   print(round(num_hess_mc, 4))
//'   cat("Analytical Hessian (Mc):\n")
//'   print(round(ana_hess_mc, 4))
//'
//'   # Check differences (monitor sign consistency)
//'   cat("Max absolute difference between Mc Hessians:\n")
//'   print(max(abs(num_hess_mc - ana_hess_mc)))
//'
//'   # Optional: Use analytical Hessian for Standard Errors
//'   # tryCatch({
//'   #   cov_matrix_mc <- solve(ana_hess_mc) # ana_hess_mc is already Hessian of negative LL
//'   #   std_errors_mc <- sqrt(diag(cov_matrix_mc))
//'   #   cat("Std. Errors from Analytical Mc Hessian:\n")
//'   #   print(std_errors_mc)
//'   # }, error = function(e) {
//'   #   warning("Could not invert analytical Mc Hessian: ", e$message)
//'   # })
//'
//' } else {
//'   cat("\nSkipping Mc Hessian comparison.\n")
//'   cat("Requires convergence, 'numDeriv' package, and function 'hsmc'.\n")
//' }
//'
//' }
//'
//' @export
// [[Rcpp::export]]
Rcpp::NumericMatrix hsmc(const Rcpp::NumericVector& par, const Rcpp::NumericVector& data) {
 // Parameter extraction and validation
 if (par.size() < 3) {
   Rcpp::NumericMatrix hess(3, 3);
   hess.fill(R_NaN);
   return hess;
 }
 
 double gamma = par[0];
 double delta = par[1];
 double lambda = par[2];
 
 // Parameter validation
 if (gamma <= 0 || delta < 0 || lambda <= 0) {
   Rcpp::NumericMatrix hess(3, 3);
   hess.fill(R_NaN);
   return hess;
 }
 
 arma::vec x = Rcpp::as<arma::vec>(data);
 if (x.n_elem < 1 || arma::any(x <= 0) || arma::any(x >= 1)) {
   Rcpp::NumericMatrix hess(3, 3);
   hess.fill(R_NaN);
   return hess;
 }
 
 int n = x.n_elem;
 Rcpp::NumericMatrix hess(3, 3);
 
 // Stability constants
 const double eps = 1e-10;
 const double max_contrib = 1e6;  // Limit for individual contributions
 
 // Compute trigamma values stably
 // For large z: trigamma(z) ≈ 1/z + 1/(2z²) + O(1/z⁴)
 double trigamma_gamma, trigamma_delta_plus_1, trigamma_gamma_plus_delta_plus_1;
 
 if (gamma > 100.0) {
   // Asymptotic approximation: ψ'(z) ≈ 1/z + 1/(2z²)
   trigamma_gamma = 1.0/gamma + 1.0/(2.0*gamma*gamma);
 } else {
   trigamma_gamma = R::trigamma(gamma);
 }
 
 if (delta > 100.0) {
   // Asymptotic approximation for ψ'(δ+1)
   trigamma_delta_plus_1 = 1.0/(delta+1.0) + 1.0/(2.0*(delta+1.0)*(delta+1.0));
 } else {
   trigamma_delta_plus_1 = R::trigamma(delta + 1.0);
 }
 
 if (gamma + delta > 100.0) {
   // Asymptotic approximation for ψ'(γ+δ+1)
   double z = gamma + delta + 1.0;
   trigamma_gamma_plus_delta_plus_1 = 1.0/z + 1.0/(2.0*z*z);
 } else {
   trigamma_gamma_plus_delta_plus_1 = R::trigamma(gamma + delta + 1.0);
 }
 
 // Initialize accumulators for sums
 double sum_log_x = 0.0;
 double sum_x_lambda_log_x_div_v = 0.0;
 double sum_lambda_term = 0.0;
 
 // Calculate term-by-term to control numerical stability
 for (int i = 0; i < n; i++) {
   double xi = x(i);
   
   // Handle boundary values
   if (xi < eps) xi = eps;
   if (xi > 1.0 - eps) xi = 1.0 - eps;
   
   // Calculate log(x) stably
   double log_xi = std::log(xi);
   sum_log_x += log_xi;
   
   // Calculate x^lambda stably using log domain for large lambda
   double x_lambda;
   if (lambda > 100.0 || lambda * std::abs(log_xi) > 1.0) {
     double log_x_lambda = lambda * log_xi;
     x_lambda = std::exp(log_x_lambda);
   } else {
     x_lambda = std::pow(xi, lambda);
   }
   
   // Calculate v = 1-x^lambda with precision for x^lambda near 1
   double v;
   if (x_lambda > 0.9995) {
     // Use complementary calculation: 1-exp(a) = -expm1(a)
     v = -std::expm1(lambda * log_xi);
   } else {
     v = 1.0 - x_lambda;
   }
   
   // Ensure v is not too small
   v = std::max(v, eps);
   
   // Term for H[1,2] = ∂²ℓ/∂δ∂λ = Σ[x^λ*log(x)/(1-x^λ)]
   double term1 = (x_lambda * log_xi) / v;
   // Prevent extreme values
   term1 = std::min(std::max(term1, -max_contrib), max_contrib);
   sum_x_lambda_log_x_div_v += term1;
   
   // Calculate squared log with safe scaling
   double log_xi_squared = log_xi * log_xi;
   
   // Term for H[2,2] = ∂²ℓ/∂λ²
   // = n/λ² + δ*Σ[x^λ*(log(x))²/(1-x^λ)*(1 + x^λ/(1-x^λ))]
   // = n/λ² + δ*Σ[x^λ*(log(x))²/(1-x^λ)²]
   double term_ratio = x_lambda / v;
   double term_combined = 1.0 + term_ratio;  // = 1/(1-x^λ)
   
   // Prevent overflow in combined term for x^λ near 1
   if (term_combined > 1e6) {
     term_combined = 1e6;
   }
   
   double lambda_term = delta * x_lambda * log_xi_squared * term_combined / v;
   
   // Prevent extreme values
   lambda_term = std::min(std::max(lambda_term, -max_contrib), max_contrib);
   sum_lambda_term += lambda_term;
 }
 
 // Calculate Hessian components
 
 // H[0,0] = -∂²ℓ/∂γ² = n[ψ'(γ+δ+1) - ψ'(γ)]
 double h_gamma_gamma = n * (trigamma_gamma_plus_delta_plus_1 - trigamma_gamma);
 
 // H[0,1] = H[1,0] = -∂²ℓ/∂γ∂δ = n*ψ'(γ+δ+1)
 double h_gamma_delta = n * trigamma_gamma_plus_delta_plus_1;
 
 // H[0,2] = H[2,0] = -∂²ℓ/∂γ∂λ = Σlog(x)
 double h_gamma_lambda = sum_log_x;
 
 // H[1,1] = -∂²ℓ/∂δ² = n[ψ'(γ+δ+1) - ψ'(δ+1)]
 double h_delta_delta = n * (trigamma_gamma_plus_delta_plus_1 - trigamma_delta_plus_1);
 
 // H[1,2] = H[2,1] = -∂²ℓ/∂δ∂λ = -Σ[x^λ*log(x)/(1-x^λ)]
 double h_delta_lambda = -sum_x_lambda_log_x_div_v;
 
 // H[2,2] = -∂²ℓ/∂λ² = -n/λ² - δ*Σ[x^λ*(log(x))²/(1-x^λ)²]
 double h_lambda_lambda = -n / (lambda * lambda) - sum_lambda_term;
 
 // Fill the Hessian matrix (symmetric)
 hess(0, 0) = -h_gamma_gamma;
 hess(0, 1) = hess(1, 0) = -h_gamma_delta;
 hess(0, 2) = hess(2, 0) = -h_gamma_lambda;
 hess(1, 1) = -h_delta_delta;
 hess(1, 2) = hess(2, 1) = -h_delta_lambda;
 hess(2, 2) = -h_lambda_lambda;
 
 return hess;
}

