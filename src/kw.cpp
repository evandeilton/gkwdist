// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include "utils.h"

/*
----------------------------------------------------------------------------
KUMARASWAMY (Kw) DISTRIBUTION
----------------------------------------------------------------------------

Parameters: alpha>0, beta>0.

* PDF:
f(x) = alpha * beta * x^(alpha -1) * (1 - x^alpha)^(beta -1),  for 0<x<1.

* CDF:
F(x)= 1 - (1 - x^alpha)^beta.

* QUANTILE:
Q(p)= {1 - [1 - p]^(1/beta)}^(1/alpha).

* RANDOM GENERATION:
If V ~ Uniform(0,1), then X= {1 - [1 - V]^(1/beta)}^(1/alpha).

* NEGATIVE LOG-LIKELIHOOD:
sum over i of -log( f(x_i) ).
log f(x_i)= log(alpha) + log(beta) + (alpha-1)*log(x_i) + (beta-1)*log(1 - x_i^alpha).
----------------------------------------------------------------------------*/






// -----------------------------------------------------------------------------
// 1) dkw: PDF of Kumaraswamy
// -----------------------------------------------------------------------------


//' @title Density of the Kumaraswamy (Kw) Distribution
//' @author Lopes, J. E.
//' @keywords distribution density kumaraswamy
//'
//' @description
//' Computes the probability density function (PDF) for the two-parameter
//' Kumaraswamy (Kw) distribution with shape parameters \code{alpha} (\eqn{\alpha})
//' and \code{beta} (\eqn{\beta}). This distribution is defined on the interval (0, 1).
//'
//' @param x Vector of quantiles (values between 0 and 1).
//' @param alpha Shape parameter \code{alpha} > 0. Can be a scalar or a vector.
//'   Default: 1.0.
//' @param beta Shape parameter \code{beta} > 0. Can be a scalar or a vector.
//'   Default: 1.0.
//' @param log_prob Logical; if \code{TRUE}, the logarithm of the density is
//'   returned (\eqn{\log(f(x))}). Default: \code{FALSE}.
//'
//' @return A vector of density values (\eqn{f(x)}) or log-density values
//'   (\eqn{\log(f(x))}). The length of the result is determined by the recycling
//'   rule applied to the arguments (\code{x}, \code{alpha}, \code{beta}).
//'   Returns \code{0} (or \code{-Inf} if \code{log_prob = TRUE}) for \code{x}
//'   outside the interval (0, 1), or \code{NaN} if parameters are invalid
//'   (e.g., \code{alpha <= 0}, \code{beta <= 0}).
//'
//' @details
//' The probability density function (PDF) of the Kumaraswamy (Kw) distribution
//' is given by:
//' \deqn{
//' f(x; \alpha, \beta) = \alpha \beta x^{\alpha-1} (1 - x^\alpha)^{\beta-1}
//' }
//' for \eqn{0 < x < 1}, \eqn{\alpha > 0}, and \eqn{\beta > 0}.
//'
//' The Kumaraswamy distribution is identical to the Generalized Kumaraswamy (GKw)
//' distribution (\code{\link{dgkw}}) with parameters \eqn{\gamma = 1},
//' \eqn{\delta = 0}, and \eqn{\lambda = 1}. It is also a special case of the
//' Exponentiated Kumaraswamy (\code{\link{dekw}}) with \eqn{\lambda = 1}, and
//' the Kumaraswamy-Kumaraswamy (\code{\link{dkkw}}) with \eqn{\delta = 0}
//' and \eqn{\lambda = 1}.
//'
//' @references
//' Kumaraswamy, P. (1980). A generalized probability density function for
//' double-bounded random processes. *Journal of Hydrology*, *46*(1-2), 79-88.
//'
//'
//' Jones, M. C. (2009). Kumaraswamy's distribution: A beta-type distribution
//' with some tractability advantages. *Statistical Methodology*, *6*(1), 70-81.
//'
//'
//' @seealso
//' \code{\link{dgkw}} (parent distribution density),
//' \code{\link{dekw}}, \code{\link{dkkw}},
//' \code{\link{pkw}}, \code{\link{qkw}}, \code{\link{rkw}} (other Kw functions),
//' \code{\link[stats]{dbeta}}
//'
//' @examples
//' \donttest{
//' # Example values
//' x_vals <- c(0.2, 0.5, 0.8)
//' alpha_par <- 2.0
//' beta_par <- 3.0
//'
//' # Calculate density using dkw
//' densities <- dkw(x_vals, alpha_par, beta_par)
//' print(densities)
//'
//' # Calculate log-density
//' log_densities <- dkw(x_vals, alpha_par, beta_par, log_prob = TRUE)
//' print(log_densities)
//' # Check: should match log(densities)
//' print(log(densities))
//'
//' # Compare with dgkw setting gamma = 1, delta = 0, lambda = 1
//' densities_gkw <- dgkw(x_vals, alpha_par, beta_par, gamma = 1.0, delta = 0.0,
//'                       lambda = 1.0)
//' print(paste("Max difference:", max(abs(densities - densities_gkw)))) # Should be near zero
//'
//' # Plot the density for different shape parameter combinations
//' curve_x <- seq(0.001, 0.999, length.out = 200)
//' plot(curve_x, dkw(curve_x, alpha = 2, beta = 3), type = "l",
//'      main = "Kumaraswamy Density Examples", xlab = "x", ylab = "f(x)",
//'      col = "blue", ylim = c(0, 4))
//' lines(curve_x, dkw(curve_x, alpha = 3, beta = 2), col = "red")
//' lines(curve_x, dkw(curve_x, alpha = 0.5, beta = 0.5), col = "green") # U-shaped
//' lines(curve_x, dkw(curve_x, alpha = 5, beta = 1), col = "purple") # J-shaped
//' lines(curve_x, dkw(curve_x, alpha = 1, beta = 3), col = "orange") # J-shaped (reversed)
//' legend("top", legend = c("a=2, b=3", "a=3, b=2", "a=0.5, b=0.5", "a=5, b=1", "a=1, b=3"),
//'        col = c("blue", "red", "green", "purple", "orange"), lty = 1, bty = "n", ncol = 2)
//' }
//'
//' @export
// [[Rcpp::export]]
Rcpp::NumericVector dkw(
   const arma::vec& x,
   const Rcpp::NumericVector& alpha,
   const Rcpp::NumericVector& beta,
   bool log_prob=false
) {
 arma::vec a_vec(alpha.begin(), alpha.size());
 arma::vec b_vec(beta.begin(), beta.size());
 
 size_t N= std::max({ x.n_elem, a_vec.n_elem, b_vec.n_elem });
 arma::vec out(N);
 
 out.fill(log_prob ? R_NegInf : 0.0);
 
 for (size_t i=0; i<N; i++){
   double a= a_vec[i % a_vec.n_elem];
   double b= b_vec[i % b_vec.n_elem];
   double xx= x[i % x.n_elem];
   
   if (!check_kw_pars(a,b)) {
     // invalid => pdf=0 or logpdf=-Inf
     continue;
   }
   if (xx<=0.0 || xx>=1.0 || !R_finite(xx)) {
     // outside domain => 0 or -Inf
     continue;
   }
   
   // log f(x)= log(a)+ log(b) + (a-1)* log(x) + (b-1)* log(1- x^a)
   double la= std::log(a);
   double lb= std::log(b);
   
   double lx= std::log(xx);
   double xalpha= a* lx; // log(x^a)
   // log(1- x^a)= log1mexp(xalpha)
   double log_1_xalpha= log1mexp(xalpha);
   if (!R_finite(log_1_xalpha)) {
     continue;
   }
   
   double log_pdf= la + lb + (a-1.0)* lx + (b-1.0)* log_1_xalpha;
   
   if (log_prob) {
     out(i)= log_pdf;
   } else {
     out(i)= std::exp(log_pdf);
   }
 }
 
 return Rcpp::NumericVector(out.memptr(), out.memptr() + out.n_elem);
}


// -----------------------------------------------------------------------------
// 2) pkw: CDF of Kumaraswamy
// -----------------------------------------------------------------------------

//' @title Cumulative Distribution Function (CDF) of the Kumaraswamy (Kw) Distribution
//' @author Lopes, J. E.
//' @keywords distribution cumulative kumaraswamy
//'
//' @description
//' Computes the cumulative distribution function (CDF), \eqn{P(X \le q)}, for the
//' two-parameter Kumaraswamy (Kw) distribution with shape parameters \code{alpha}
//' (\eqn{\alpha}) and \code{beta} (\eqn{\beta}). This distribution is defined
//' on the interval (0, 1).
//'
//' @param q Vector of quantiles (values generally between 0 and 1).
//' @param alpha Shape parameter \code{alpha} > 0. Can be a scalar or a vector.
//'   Default: 1.0.
//' @param beta Shape parameter \code{beta} > 0. Can be a scalar or a vector.
//'   Default: 1.0.
//' @param lower_tail Logical; if \code{TRUE} (default), probabilities are
//'   \eqn{P(X \le q)}, otherwise, \eqn{P(X > q)}.
//' @param log_p Logical; if \code{TRUE}, probabilities \eqn{p} are given as
//'   \eqn{\log(p)}. Default: \code{FALSE}.
//'
//' @return A vector of probabilities, \eqn{F(q)}, or their logarithms/complements
//'   depending on \code{lower_tail} and \code{log_p}. The length of the result
//'   is determined by the recycling rule applied to the arguments (\code{q},
//'   \code{alpha}, \code{beta}). Returns \code{0} (or \code{-Inf} if
//'   \code{log_p = TRUE}) for \code{q <= 0} and \code{1} (or \code{0} if
//'   \code{log_p = TRUE}) for \code{q >= 1}. Returns \code{NaN} for invalid
//'   parameters.
//'
//' @details
//' The cumulative distribution function (CDF) of the Kumaraswamy (Kw)
//' distribution is given by:
//' \deqn{
//' F(x; \alpha, \beta) = 1 - (1 - x^\alpha)^\beta
//' }
//' for \eqn{0 < x < 1}, \eqn{\alpha > 0}, and \eqn{\beta > 0}.
//'
//' The Kw distribution is a special case of several generalized distributions:
//' \itemize{
//'  \item Generalized Kumaraswamy (\code{\link{pgkw}}) with \eqn{\gamma=1, \delta=0, \lambda=1}.
//'  \item Exponentiated Kumaraswamy (\code{\link{pekw}}) with \eqn{\lambda=1}.
//'  \item Kumaraswamy-Kumaraswamy (\code{\link{pkkw}}) with \eqn{\delta=0, \lambda=1}.
//' }
//' The implementation uses the closed-form expression for efficiency.
//'
//' @references
//' Kumaraswamy, P. (1980). A generalized probability density function for
//' double-bounded random processes. *Journal of Hydrology*, *46*(1-2), 79-88.
//'
//'
//' Jones, M. C. (2009). Kumaraswamy's distribution: A beta-type distribution
//' with some tractability advantages. *Statistical Methodology*, *6*(1), 70-81.
//'
//'
//' @seealso
//' \code{\link{pgkw}}, \code{\link{pekw}}, \code{\link{pkkw}} (related generalized CDFs),
//' \code{\link{dkw}}, \code{\link{qkw}}, \code{\link{rkw}} (other Kw functions),
//' \code{\link[stats]{pbeta}}
//'
//' @examples
//' \donttest{
//' # Example values
//' q_vals <- c(0.2, 0.5, 0.8)
//' alpha_par <- 2.0
//' beta_par <- 3.0
//'
//' # Calculate CDF P(X <= q) using pkw
//' probs <- pkw(q_vals, alpha_par, beta_par)
//' print(probs)
//'
//' # Calculate upper tail P(X > q)
//' probs_upper <- pkw(q_vals, alpha_par, beta_par, lower_tail = FALSE)
//' print(probs_upper)
//' # Check: probs + probs_upper should be 1
//' print(probs + probs_upper)
//'
//' # Calculate log CDF
//' log_probs <- pkw(q_vals, alpha_par, beta_par, log_p = TRUE)
//' print(log_probs)
//' # Check: should match log(probs)
//' print(log(probs))
//'
//' # Compare with pgkw setting gamma = 1, delta = 0, lambda = 1
//' probs_gkw <- pgkw(q_vals, alpha_par, beta_par, gamma = 1.0, delta = 0.0,
//'                   lambda = 1.0)
//' print(paste("Max difference:", max(abs(probs - probs_gkw)))) # Should be near zero
//'
//' # Plot the CDF for different shape parameter combinations
//' curve_q <- seq(0.001, 0.999, length.out = 200)
//' plot(curve_q, pkw(curve_q, alpha = 2, beta = 3), type = "l",
//'      main = "Kumaraswamy CDF Examples", xlab = "q", ylab = "F(q)",
//'      col = "blue", ylim = c(0, 1))
//' lines(curve_q, pkw(curve_q, alpha = 3, beta = 2), col = "red")
//' lines(curve_q, pkw(curve_q, alpha = 0.5, beta = 0.5), col = "green")
//' lines(curve_q, pkw(curve_q, alpha = 5, beta = 1), col = "purple")
//' lines(curve_q, pkw(curve_q, alpha = 1, beta = 3), col = "orange")
//' legend("bottomright", legend = c("a=2, b=3", "a=3, b=2", "a=0.5, b=0.5", "a=5, b=1", "a=1, b=3"),
//'        col = c("blue", "red", "green", "purple", "orange"), lty = 1, bty = "n", ncol = 2)
//'
//' }
//'
//' @export
// [[Rcpp::export]]
Rcpp::NumericVector pkw(
   const arma::vec& q,
   const Rcpp::NumericVector& alpha,
   const Rcpp::NumericVector& beta,
   bool lower_tail=true,
   bool log_p=false
) {
 arma::vec a_vec(alpha.begin(), alpha.size());
 arma::vec b_vec(beta.begin(), beta.size());
 
 size_t N= std::max({ q.n_elem, a_vec.n_elem, b_vec.n_elem });
 arma::vec out(N);
 
 for (size_t i=0; i<N; i++){
   double a= a_vec[i % a_vec.n_elem];
   double b= b_vec[i % b_vec.n_elem];
   double xx= q[i % q.n_elem];
   
   if (!check_kw_pars(a,b)) {
     out(i)= NA_REAL;
     continue;
   }
   
   // boundary
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
   
   double xalpha= std::pow(xx, a);
   double tmp= 1.0 - std::pow( (1.0 - xalpha), b );
   if (tmp<=0.0) {
     double val0= (lower_tail ? 0.0 : 1.0);
     out(i)= log_p ? std::log(val0) : val0;
     continue;
   }
   if (tmp>=1.0) {
     double val1= (lower_tail ? 1.0 : 0.0);
     out(i)= log_p ? std::log(val1) : val1;
     continue;
   }
   
   double val= tmp;
   if (!lower_tail) {
     val= 1.0- val;
   }
   if (log_p) {
     val= std::log(val);
   }
   out(i)= val;
 }
 
 return Rcpp::NumericVector(out.memptr(), out.memptr() + out.n_elem);
}

// -----------------------------------------------------------------------------
// 3) qkw: Quantile of Kumaraswamy
// -----------------------------------------------------------------------------

//' @title Quantile Function of the Kumaraswamy (Kw) Distribution
//' @author Lopes, J. E.
//' @keywords distribution quantile kumaraswamy
//'
//' @description
//' Computes the quantile function (inverse CDF) for the two-parameter
//' Kumaraswamy (Kw) distribution with shape parameters \code{alpha} (\eqn{\alpha})
//' and \code{beta} (\eqn{\beta}). It finds the value \code{q} such that
//' \eqn{P(X \le q) = p}.
//'
//' @param p Vector of probabilities (values between 0 and 1).
//' @param alpha Shape parameter \code{alpha} > 0. Can be a scalar or a vector.
//'   Default: 1.0.
//' @param beta Shape parameter \code{beta} > 0. Can be a scalar or a vector.
//'   Default: 1.0.
//' @param lower_tail Logical; if \code{TRUE} (default), probabilities are \eqn{p = P(X \le q)},
//'   otherwise, probabilities are \eqn{p = P(X > q)}.
//' @param log_p Logical; if \code{TRUE}, probabilities \code{p} are given as
//'   \eqn{\log(p)}. Default: \code{FALSE}.
//'
//' @return A vector of quantiles corresponding to the given probabilities \code{p}.
//'   The length of the result is determined by the recycling rule applied to
//'   the arguments (\code{p}, \code{alpha}, \code{beta}).
//'   Returns:
//'   \itemize{
//'     \item \code{0} for \code{p = 0} (or \code{p = -Inf} if \code{log_p = TRUE},
//'           when \code{lower_tail = TRUE}).
//'     \item \code{1} for \code{p = 1} (or \code{p = 0} if \code{log_p = TRUE},
//'           when \code{lower_tail = TRUE}).
//'     \item \code{NaN} for \code{p < 0} or \code{p > 1} (or corresponding log scale).
//'     \item \code{NaN} for invalid parameters (e.g., \code{alpha <= 0},
//'           \code{beta <= 0}).
//'   }
//'   Boundary return values are adjusted accordingly for \code{lower_tail = FALSE}.
//'
//' @details
//' The quantile function \eqn{Q(p)} is the inverse of the CDF \eqn{F(q)}. The CDF
//' for the Kumaraswamy distribution is \eqn{F(q) = 1 - (1 - q^\alpha)^\beta}
//' (see \code{\link{pkw}}). Inverting this equation for \eqn{q} yields the
//' quantile function:
//' \deqn{
//' Q(p) = \left\{ 1 - (1 - p)^{1/\beta} \right\}^{1/\alpha}
//' }
//' The function uses this closed-form expression and correctly handles the
//' \code{lower_tail} and \code{log_p} arguments by transforming \code{p}
//' appropriately before applying the formula. This is equivalent to the general
//' GKw quantile function (\code{\link{qgkw}}) evaluated with \eqn{\gamma=1, \delta=0, \lambda=1}.
//'
//' @references
//' Kumaraswamy, P. (1980). A generalized probability density function for
//' double-bounded random processes. *Journal of Hydrology*, *46*(1-2), 79-88.
//'
//'
//' Jones, M. C. (2009). Kumaraswamy's distribution: A beta-type distribution
//' with some tractability advantages. *Statistical Methodology*, *6*(1), 70-81.
//'
//'
//' @seealso
//' \code{\link{qgkw}} (parent distribution quantile function),
//' \code{\link{dkw}}, \code{\link{pkw}}, \code{\link{rkw}} (other Kw functions),
//' \code{\link[stats]{qbeta}}, \code{\link[stats]{qunif}}
//'
//' @examples
//' \donttest{
//' # Example values
//' p_vals <- c(0.1, 0.5, 0.9)
//' alpha_par <- 2.0
//' beta_par <- 3.0
//'
//' # Calculate quantiles using qkw
//' quantiles <- qkw(p_vals, alpha_par, beta_par)
//' print(quantiles)
//'
//' # Calculate quantiles for upper tail probabilities P(X > q) = p
//' quantiles_upper <- qkw(p_vals, alpha_par, beta_par, lower_tail = FALSE)
//' print(quantiles_upper)
//' # Check: qkw(p, ..., lt=F) == qkw(1-p, ..., lt=T)
//' print(qkw(1 - p_vals, alpha_par, beta_par))
//'
//' # Calculate quantiles from log probabilities
//' log_p_vals <- log(p_vals)
//' quantiles_logp <- qkw(log_p_vals, alpha_par, beta_par, log_p = TRUE)
//' print(quantiles_logp)
//' # Check: should match original quantiles
//' print(quantiles)
//'
//' # Compare with qgkw setting gamma = 1, delta = 0, lambda = 1
//' quantiles_gkw <- qgkw(p_vals, alpha = alpha_par, beta = beta_par,
//'                      gamma = 1.0, delta = 0.0, lambda = 1.0)
//' print(paste("Max difference:", max(abs(quantiles - quantiles_gkw)))) # Should be near zero
//'
//' # Verify inverse relationship with pkw
//' p_check <- 0.75
//' q_calc <- qkw(p_check, alpha_par, beta_par)
//' p_recalc <- pkw(q_calc, alpha_par, beta_par)
//' print(paste("Original p:", p_check, " Recalculated p:", p_recalc))
//' # abs(p_check - p_recalc) < 1e-9 # Should be TRUE
//'
//' # Boundary conditions
//' print(qkw(c(0, 1), alpha_par, beta_par)) # Should be 0, 1
//' print(qkw(c(-Inf, 0), alpha_par, beta_par, log_p = TRUE)) # Should be 0, 1
//'
//' }
//'
//' @export
// [[Rcpp::export]]
Rcpp::NumericVector qkw(
   const arma::vec& p,
   const Rcpp::NumericVector& alpha,
   const Rcpp::NumericVector& beta,
   bool lower_tail=true,
   bool log_p=false
) {
 arma::vec a_vec(alpha.begin(), alpha.size());
 arma::vec b_vec(beta.begin(), beta.size());
 
 size_t N= std::max({ p.n_elem, a_vec.n_elem, b_vec.n_elem });
 arma::vec out(N);
 
 for (size_t i=0; i<N; i++){
   double a= a_vec[i % a_vec.n_elem];
   double b= b_vec[i % b_vec.n_elem];
   double pp= p[i % p.n_elem];
   
   if (!check_kw_pars(a,b)) {
     out(i)= NA_REAL;
     continue;
   }
   
   // convert if log
   if (log_p) {
     if (pp>0.0) {
       // invalid => p>1
       out(i)= NA_REAL;
       continue;
     }
     pp= std::exp(pp);
   }
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
   
   // x= {1 - [1 - p]^(1/beta)}^(1/alpha)
   double step1= 1.0 - pp;
   if (step1<0.0) step1=0.0;
   double step2= std::pow(step1, 1.0/b);
   double step3= 1.0 - step2;
   if (step3<0.0) step3=0.0;
   
   double xval;
   if (a==1.0) {
     xval= step3;
   } else {
     xval= std::pow(step3, 1.0/a);
     if (!R_finite(xval)|| xval<0.0) xval=0.0;
     if (xval>1.0) xval=1.0;
   }
   out(i)= xval;
 }
 
 return Rcpp::NumericVector(out.memptr(), out.memptr() + out.n_elem);
}


// -----------------------------------------------------------------------------
// 4) rkw: Random Generation from Kumaraswamy
// -----------------------------------------------------------------------------

//' @title Random Number Generation for the Kumaraswamy (Kw) Distribution
//' @author Lopes, J. E.
//' @keywords distribution random kumaraswamy
//'
//' @description
//' Generates random deviates from the two-parameter Kumaraswamy (Kw)
//' distribution with shape parameters \code{alpha} (\eqn{\alpha}) and
//' \code{beta} (\eqn{\beta}).
//'
//' @param n Number of observations. If \code{length(n) > 1}, the length is
//'   taken to be the number required. Must be a non-negative integer.
//' @param alpha Shape parameter \code{alpha} > 0. Can be a scalar or a vector.
//'   Default: 1.0.
//' @param beta Shape parameter \code{beta} > 0. Can be a scalar or a vector.
//'   Default: 1.0.
//'
//' @return A vector of length \code{n} containing random deviates from the Kw
//'   distribution, with values in (0, 1). The length of the result is determined
//'   by \code{n} and the recycling rule applied to the parameters (\code{alpha},
//'   \code{beta}). Returns \code{NaN} if parameters are invalid (e.g.,
//'   \code{alpha <= 0}, \code{beta <= 0}).
//'
//' @details
//' The generation method uses the inverse transform (quantile) method.
//' That is, if \eqn{U} is a random variable following a standard Uniform
//' distribution on (0, 1), then \eqn{X = Q(U)} follows the Kw distribution,
//' where \eqn{Q(p)} is the Kw quantile function (\code{\link{qkw}}):
//' \deqn{
//' Q(p) = \left\{ 1 - (1 - p)^{1/\beta} \right\}^{1/\alpha}
//' }
//' The implementation generates \eqn{U} using \code{\link[stats]{runif}}
//' and applies this transformation. This is equivalent to the general GKw
//' generation method (\code{\link{rgkw}}) evaluated at \eqn{\gamma=1, \delta=0, \lambda=1}.
//'
//' @references
//' Kumaraswamy, P. (1980). A generalized probability density function for
//' double-bounded random processes. *Journal of Hydrology*, *46*(1-2), 79-88.
//'
//'
//' Jones, M. C. (2009). Kumaraswamy's distribution: A beta-type distribution
//' with some tractability advantages. *Statistical Methodology*, *6*(1), 70-81.
//'
//'
//' Devroye, L. (1986). *Non-Uniform Random Variate Generation*. Springer-Verlag.
//' (General methods for random variate generation).
//'
//' @seealso
//' \code{\link{rgkw}} (parent distribution random generation),
//' \code{\link{dkw}}, \code{\link{pkw}}, \code{\link{qkw}} (other Kw functions),
//' \code{\link[stats]{runif}}
//'
//' @examples
//' \donttest{
//' set.seed(2029) # for reproducibility
//'
//' # Generate 1000 random values from a specific Kw distribution
//' alpha_par <- 2.0
//' beta_par <- 3.0
//'
//' x_sample_kw <- rkw(1000, alpha = alpha_par, beta = beta_par)
//' summary(x_sample_kw)
//'
//' # Histogram of generated values compared to theoretical density
//' hist(x_sample_kw, breaks = 30, freq = FALSE, # freq=FALSE for density
//'      main = "Histogram of Kw Sample", xlab = "x", ylim = c(0, 2.5))
//' curve(dkw(x, alpha = alpha_par, beta = beta_par),
//'       add = TRUE, col = "red", lwd = 2, n = 201)
//' legend("top", legend = "Theoretical PDF", col = "red", lwd = 2, bty = "n")
//'
//' # Comparing empirical and theoretical quantiles (Q-Q plot)
//' prob_points <- seq(0.01, 0.99, by = 0.01)
//' theo_quantiles <- qkw(prob_points, alpha = alpha_par, beta = beta_par)
//' emp_quantiles <- quantile(x_sample_kw, prob_points, type = 7)
//'
//' plot(theo_quantiles, emp_quantiles, pch = 16, cex = 0.8,
//'      main = "Q-Q Plot for Kw Distribution",
//'      xlab = "Theoretical Quantiles", ylab = "Empirical Quantiles (n=1000)")
//' abline(a = 0, b = 1, col = "blue", lty = 2)
//'
//' # Compare summary stats with rgkw(..., gamma=1, delta=0, lambda=1)
//' # Note: individual values will differ due to randomness
//' x_sample_gkw <- rgkw(1000, alpha = alpha_par, beta = beta_par, gamma = 1.0,
//'                      delta = 0.0, lambda = 1.0)
//' print("Summary stats for rkw sample:")
//' print(summary(x_sample_kw))
//' print("Summary stats for rgkw(gamma=1, delta=0, lambda=1) sample:")
//' print(summary(x_sample_gkw)) # Should be similar
//'
//' }
//'
//' @export
// [[Rcpp::export]]
Rcpp::NumericVector rkw(
   int n,
   const Rcpp::NumericVector& alpha,
   const Rcpp::NumericVector& beta
) {
 if (n<=0) {
   Rcpp::stop("rkw: n must be positive");
 }
 
 arma::vec a_vec(alpha.begin(), alpha.size());
 arma::vec b_vec(beta.begin(), beta.size());
 
 size_t k= std::max({ a_vec.n_elem, b_vec.n_elem });
 arma::vec out(n);
 
 for (int i=0; i<n; i++){
   size_t idx= i % k;
   double a= a_vec[idx % a_vec.n_elem];
   double b= b_vec[idx % b_vec.n_elem];
   
   if (!check_kw_pars(a,b)) {
     out(i)= NA_REAL;
     Rcpp::warning("rkw: invalid parameters at index %d", i+1);
     continue;
   }
   
   double U= R::runif(0.0,1.0);
   // X= {1 - [1 - U]^(1/beta)}^(1/alpha)
   double step1= 1.0 - U;
   if (step1<0.0) step1=0.0;
   double step2= std::pow(step1, 1.0/b);
   double step3= 1.0 - step2;
   if (step3<0.0) step3=0.0;
   
   double x;
   if (a==1.0) {
     x= step3;
   } else {
     x= std::pow(step3, 1.0/a);
     if (!R_finite(x)|| x<0.0) x=0.0;
     if (x>1.0) x=1.0;
   }
   out(i)= x;
 }
 
 return Rcpp::NumericVector(out.memptr(), out.memptr() + out.n_elem);
}

// -----------------------------------------------------------------------------
// 5) llkw: Negative Log-Likelihood for Kumaraswamy
// -----------------------------------------------------------------------------

//' @title Negative Log-Likelihood of the Kumaraswamy (Kw) Distribution
//' @author Lopes, J. E.
//' @keywords distribution likelihood optimize kumaraswamy
//'
//' @description
//' Computes the negative log-likelihood function for the two-parameter
//' Kumaraswamy (Kw) distribution with parameters \code{alpha} (\eqn{\alpha})
//' and \code{beta} (\eqn{\beta}), given a vector of observations. This function
//' is suitable for maximum likelihood estimation.
//'
//' @param par A numeric vector of length 2 containing the distribution parameters
//'   in the order: \code{alpha} (\eqn{\alpha > 0}), \code{beta} (\eqn{\beta > 0}).
//' @param data A numeric vector of observations. All values must be strictly
//'   between 0 and 1 (exclusive).
//'
//' @return Returns a single \code{double} value representing the negative
//'   log-likelihood (\eqn{-\ell(\theta|\mathbf{x})}). Returns \code{Inf}
//'   if any parameter values in \code{par} are invalid according to their
//'   constraints, or if any value in \code{data} is not in the interval (0, 1).
//'
//' @details
//' The Kumaraswamy (Kw) distribution's probability density function (PDF) is
//' (see \code{\link{dkw}}):
//' \deqn{
//' f(x | \theta) = \alpha \beta x^{\alpha-1} (1 - x^\alpha)^{\beta-1}
//' }
//' for \eqn{0 < x < 1} and \eqn{\theta = (\alpha, \beta)}.
//' The log-likelihood function \eqn{\ell(\theta | \mathbf{x})} for a sample
//' \eqn{\mathbf{x} = (x_1, \dots, x_n)} is \eqn{\sum_{i=1}^n \ln f(x_i | \theta)}:
//' \deqn{
//' \ell(\theta | \mathbf{x}) = n[\ln(\alpha) + \ln(\beta)]
//' + \sum_{i=1}^{n} [(\alpha-1)\ln(x_i) + (\beta-1)\ln(v_i)]
//' }
//' where \eqn{v_i = 1 - x_i^{\alpha}}.
//' This function computes and returns the *negative* log-likelihood, \eqn{-\ell(\theta|\mathbf{x})},
//' suitable for minimization using optimization routines like \code{\link[stats]{optim}}.
//' It is equivalent to the negative log-likelihood of the GKw distribution
//' (\code{\link{llgkw}}) evaluated at \eqn{\gamma=1, \delta=0, \lambda=1}.
//'
//' @references
//' Kumaraswamy, P. (1980). A generalized probability density function for
//' double-bounded random processes. *Journal of Hydrology*, *46*(1-2), 79-88.
//'
//'
//' Jones, M. C. (2009). Kumaraswamy's distribution: A beta-type distribution
//' with some tractability advantages. *Statistical Methodology*, *6*(1), 70-81.
//'
//'
//' @seealso
//' \code{\link{llgkw}} (parent distribution negative log-likelihood),
//' \code{\link{dkw}}, \code{\link{pkw}}, \code{\link{qkw}}, \code{\link{rkw}},
//' \code{grkw} (gradient, if available),
//' \code{hskw} (Hessian, if available),
//' \code{\link[stats]{optim}}
//'
//' @examples
//' \donttest{
//' # Assuming existence of rkw, grkw, hskw functions for Kw distribution
//'
//' # Generate sample data from a known Kw distribution
//' set.seed(123)
//' true_par_kw <- c(alpha = 2, beta = 3)
//' sample_data_kw <- rkw(100, alpha = true_par_kw[1], beta = true_par_kw[2])
//' hist(sample_data_kw, breaks = 20, main = "Kw(2, 3) Sample")
//'
//' # --- Maximum Likelihood Estimation using optim ---
//' # Initial parameter guess
//' start_par_kw <- c(1.5, 2.5)
//'
//' # Perform optimization (minimizing negative log-likelihood)
//' # Use method="L-BFGS-B" for box constraints (params > 0)
//' mle_result_kw <- stats::optim(par = start_par_kw,
//'                               fn = llkw, # Use the Kw neg-log-likelihood
//'                               method = "L-BFGS-B",
//'                               lower = c(1e-6, 1e-6), # Lower bounds > 0
//'                               hessian = TRUE,
//'                               data = sample_data_kw)
//'
//' # Check convergence and results
//' if (mle_result_kw$convergence == 0) {
//'   print("Optimization converged successfully.")
//'   mle_par_kw <- mle_result_kw$par
//'   print("Estimated Kw parameters:")
//'   print(mle_par_kw)
//'   print("True Kw parameters:")
//'   print(true_par_kw)
//' } else {
//'   warning("Optimization did not converge!")
//'   print(mle_result_kw$message)
//' }
//'
//' # --- Compare numerical and analytical derivatives (if available) ---
//' # Requires 'numDeriv' package and analytical functions 'grkw', 'hskw'
//' if (mle_result_kw$convergence == 0 &&
//'     requireNamespace("numDeriv", quietly = TRUE) &&
//'     exists("grkw") && exists("hskw")) {
//'
//'   cat("\nComparing Derivatives at Kw MLE estimates:\n")
//'
//'   # Numerical derivatives of llkw
//'   num_grad_kw <- numDeriv::grad(func = llkw, x = mle_par_kw, data = sample_data_kw)
//'   num_hess_kw <- numDeriv::hessian(func = llkw, x = mle_par_kw, data = sample_data_kw)
//'
//'   # Analytical derivatives (assuming they return derivatives of negative LL)
//'   ana_grad_kw <- grkw(par = mle_par_kw, data = sample_data_kw)
//'   ana_hess_kw <- hskw(par = mle_par_kw, data = sample_data_kw)
//'
//'   # Check differences
//'   cat("Max absolute difference between gradients:\n")
//'   print(max(abs(num_grad_kw - ana_grad_kw)))
//'   cat("Max absolute difference between Hessians:\n")
//'   print(max(abs(num_hess_kw - ana_hess_kw)))
//'
//' } else {
//'    cat("\nSkipping derivative comparison for Kw.\n")
//'    cat("Requires convergence, 'numDeriv' package and functions 'grkw', 'hskw'.\n")
//' }
//'
//' }
//'
//' @export
// [[Rcpp::export]]
double llkw(const Rcpp::NumericVector& par,
           const Rcpp::NumericVector& data) {
 if (par.size()<2) {
   return R_PosInf;
 }
 double a= par[0];
 double b= par[1];
 
 if (!check_kw_pars(a,b)) {
   return R_PosInf;
 }
 
 arma::vec x= Rcpp::as<arma::vec>(data);
 if (x.n_elem<1) {
   return R_PosInf;
 }
 if (arma::any(x<=0.0) || arma::any(x>=1.0)) {
   return R_PosInf;
 }
 
 int n= x.n_elem;
 // constant: n*( log(a)+ log(b) )
 double cst= n*( std::log(a) + std::log(b) );
 
 // sum( (a-1)* log(x_i ) )
 arma::vec lx= arma::log(x);
 double sum1= (a-1.0)* arma::sum(lx);
 
 // sum( (b-1)* log(1- x^a) )
 arma::vec xalpha= arma::pow(x,a);
 arma::vec log_1_xalpha= arma::log(1.0 - xalpha);
 double sum2= (b-1.0)* arma::sum(log_1_xalpha);
 
 double loglike= cst + sum1 + sum2;
 // negative
 return -loglike;
}



//' @title Gradient of the Negative Log-Likelihood for the Kumaraswamy (Kw) Distribution
//' @author Lopes, J. E.
//' @keywords distribution likelihood optimize gradient kumaraswamy
//'
//' @description
//' Computes the gradient vector (vector of first partial derivatives) of the
//' negative log-likelihood function for the two-parameter Kumaraswamy (Kw)
//' distribution with parameters \code{alpha} (\eqn{\alpha}) and \code{beta}
//' (\eqn{\beta}). This provides the analytical gradient often used for efficient
//' optimization via maximum likelihood estimation.
//'
//' @param par A numeric vector of length 2 containing the distribution parameters
//'   in the order: \code{alpha} (\eqn{\alpha > 0}), \code{beta} (\eqn{\beta > 0}).
//' @param data A numeric vector of observations. All values must be strictly
//'   between 0 and 1 (exclusive).
//'
//' @return Returns a numeric vector of length 2 containing the partial derivatives
//'   of the negative log-likelihood function \eqn{-\ell(\theta | \mathbf{x})} with
//'   respect to each parameter: \eqn{(-\partial \ell/\partial \alpha, -\partial \ell/\partial \beta)}.
//'   Returns a vector of \code{NaN} if any parameter values are invalid according
//'   to their constraints, or if any value in \code{data} is not in the
//'   interval (0, 1).
//'
//' @details
//' The components of the gradient vector of the negative log-likelihood
//' (\eqn{-\nabla \ell(\theta | \mathbf{x})}) for the Kw model are:
//'
//' \deqn{
//' -\frac{\partial \ell}{\partial \alpha} = -\frac{n}{\alpha} - \sum_{i=1}^{n}\ln(x_i)
//' + (\beta-1)\sum_{i=1}^{n}\frac{x_i^{\alpha}\ln(x_i)}{v_i}
//' }
//' \deqn{
//' -\frac{\partial \ell}{\partial \beta} = -\frac{n}{\beta} - \sum_{i=1}^{n}\ln(v_i)
//' }
//'
//' where \eqn{v_i = 1 - x_i^{\alpha}}.
//' These formulas represent the derivatives of \eqn{-\ell(\theta)}, consistent with
//' minimizing the negative log-likelihood. They correspond to the relevant components
//' of the general GKw gradient (\code{\link{grgkw}}) evaluated at \eqn{\gamma=1, \delta=0, \lambda=1}.
//'
//' @references
//' Kumaraswamy, P. (1980). A generalized probability density function for
//' double-bounded random processes. *Journal of Hydrology*, *46*(1-2), 79-88.
//'
//'
//' Jones, M. C. (2009). Kumaraswamy's distribution: A beta-type distribution
//' with some tractability advantages. *Statistical Methodology*, *6*(1), 70-81.
//'
//' (Note: Specific gradient formulas might be derived or sourced from additional references).
//'
//' @seealso
//' \code{\link{grgkw}} (parent distribution gradient),
//' \code{\link{llkw}} (negative log-likelihood for Kw),
//' \code{hskw} (Hessian for Kw, if available),
//' \code{\link{dkw}} (density for Kw),
//' \code{\link[stats]{optim}},
//' \code{\link[numDeriv]{grad}} (for numerical gradient comparison).
//'
//' @examples
//' \donttest{
//' # Assuming existence of rkw, llkw, grkw, hskw functions for Kw
//'
//' # Generate sample data
//' set.seed(123)
//' true_par_kw <- c(alpha = 2, beta = 3)
//' sample_data_kw <- rkw(100, alpha = true_par_kw[1], beta = true_par_kw[2])
//' hist(sample_data_kw, breaks = 20, main = "Kw(2, 3) Sample")
//'
//' # --- Find MLE estimates ---
//' start_par_kw <- c(1.5, 2.5)
//' mle_result_kw <- stats::optim(par = start_par_kw,
//'                               fn = llkw,
//'                               gr = grkw, # Use analytical gradient for Kw
//'                               method = "L-BFGS-B", # Recommended for bounds
//'                               lower = c(1e-6, 1e-6),
//'                               hessian = TRUE,
//'                               data = sample_data_kw)
//'
//' # --- Compare analytical gradient to numerical gradient ---
//' if (mle_result_kw$convergence == 0 &&
//'     requireNamespace("numDeriv", quietly = TRUE)) {
//'
//'   mle_par_kw <- mle_result_kw$par
//'   cat("\nComparing Gradients for Kw at MLE estimates:\n")
//'
//'   # Numerical gradient of llkw
//'   num_grad_kw <- numDeriv::grad(func = llkw, x = mle_par_kw, data = sample_data_kw)
//'
//'   # Analytical gradient from grkw
//'   ana_grad_kw <- grkw(par = mle_par_kw, data = sample_data_kw)
//'
//'   cat("Numerical Gradient (Kw):\n")
//'   print(num_grad_kw)
//'   cat("Analytical Gradient (Kw):\n")
//'   print(ana_grad_kw)
//'
//'   # Check differences
//'   cat("Max absolute difference between Kw gradients:\n")
//'   print(max(abs(num_grad_kw - ana_grad_kw)))
//'
//' } else {
//'   cat("\nSkipping Kw gradient comparison.\n")
//' }
//'
//' # Example with Hessian comparison (if hskw exists)
//' if (mle_result_kw$convergence == 0 &&
//'     requireNamespace("numDeriv", quietly = TRUE) && exists("hskw")) {
//'
//'   num_hess_kw <- numDeriv::hessian(func = llkw, x = mle_par_kw, data = sample_data_kw)
//'   ana_hess_kw <- hskw(par = mle_par_kw, data = sample_data_kw)
//'   cat("\nMax absolute difference between Kw Hessians:\n")
//'   print(max(abs(num_hess_kw - ana_hess_kw)))
//'
//' }
//'
//' }
//'
//' @export
// [[Rcpp::export]]
Rcpp::NumericVector grkw(const Rcpp::NumericVector& par, const Rcpp::NumericVector& data) {
 // Parameter extraction
 double alpha = par[0];   // Shape parameter α > 0
 double beta = par[1];    // Shape parameter β > 0
 
 // Parameter validation
 if (alpha <= 0 || beta <= 0) {
   Rcpp::NumericVector grad(2, R_NaN);
   return grad;
 }
 
 // Data conversion and validation
 arma::vec x = Rcpp::as<arma::vec>(data);
 
 if (arma::any(x <= 0) || arma::any(x >= 1)) {
   Rcpp::NumericVector grad(2, R_NaN);
   return grad;
 }
 
 int n = x.n_elem;  // Sample size
 
 // Initialize gradient vector
 Rcpp::NumericVector grad(2, 0.0);
 
 // Small constant to avoid numerical issues
 double eps = std::numeric_limits<double>::epsilon() * 100;
 
 // Compute transformations and intermediate values
 arma::vec log_x = arma::log(x);                // log(x_i)
 arma::vec x_alpha = arma::pow(x, alpha);       // x_i^α
 arma::vec x_alpha_log_x = x_alpha % log_x;     // x_i^α * log(x_i)
 
 // v_i = 1 - x_i^α
 arma::vec v = 1.0 - x_alpha;
 v = arma::clamp(v, eps, 1.0 - eps);            // Prevent numerical issues
 
 arma::vec log_v = arma::log(v);                // log(1-x_i^α)
 
 // Calculate partial derivatives for each parameter (for log-likelihood)
 
 // ∂ℓ/∂α = n/α + Σᵢlog(xᵢ) - Σᵢ[(β-1)xᵢ^α*log(xᵢ)/(1-xᵢ^α)]
 double d_alpha = n / alpha + arma::sum(log_x);
 
 // Calculate the term for α gradient
 arma::vec alpha_term = (beta - 1.0) * x_alpha_log_x / v;
 
 d_alpha -= arma::sum(alpha_term);
 
 // ∂ℓ/∂β = n/β + Σᵢlog(1-xᵢ^α)
 double d_beta = n / beta + arma::sum(log_v);
 
 // Since we're optimizing negative log-likelihood, negate all derivatives
 grad[0] = -d_alpha;
 grad[1] = -d_beta;
 
 return grad;
}



//' @title Hessian Matrix of the Negative Log-Likelihood for the Kw Distribution
//' @author Lopes, J. E.
//' @keywords distribution likelihood optimize hessian kumaraswamy
//'
//' @description
//' Computes the analytic 2x2 Hessian matrix (matrix of second partial derivatives)
//' of the negative log-likelihood function for the two-parameter Kumaraswamy (Kw)
//' distribution with parameters \code{alpha} (\eqn{\alpha}) and \code{beta}
//' (\eqn{\beta}). The Hessian is useful for estimating standard errors and in
//' optimization algorithms.
//'
//' @param par A numeric vector of length 2 containing the distribution parameters
//'   in the order: \code{alpha} (\eqn{\alpha > 0}), \code{beta} (\eqn{\beta > 0}).
//' @param data A numeric vector of observations. All values must be strictly
//'   between 0 and 1 (exclusive).
//'
//' @return Returns a 2x2 numeric matrix representing the Hessian matrix of the
//'   negative log-likelihood function, \eqn{-\partial^2 \ell / (\partial \theta_i \partial \theta_j)},
//'   where \eqn{\theta = (\alpha, \beta)}.
//'   Returns a 2x2 matrix populated with \code{NaN} if any parameter values are
//'   invalid according to their constraints, or if any value in \code{data} is
//'   not in the interval (0, 1).
//'
//' @details
//' This function calculates the analytic second partial derivatives of the
//' negative log-likelihood function (\eqn{-\ell(\theta|\mathbf{x})}). The components
//' are the negative of the second derivatives of the log-likelihood \eqn{\ell}
//' (derived from the PDF in \code{\link{dkw}}).
//'
//' Let \eqn{v_i = 1 - x_i^{\alpha}}. The second derivatives of the positive log-likelihood (\eqn{\ell}) are:
//' \deqn{
//' \frac{\partial^2 \ell}{\partial \alpha^2} = -\frac{n}{\alpha^2} -
//' (\beta-1)\sum_{i=1}^{n}\frac{x_i^{\alpha}(\ln(x_i))^2}{v_i^2}
//' }
//' \deqn{
//' \frac{\partial^2 \ell}{\partial \alpha \partial \beta} = -
//' \sum_{i=1}^{n}\frac{x_i^{\alpha}\ln(x_i)}{v_i}
//' }
//' \deqn{
//' \frac{\partial^2 \ell}{\partial \beta^2} = -\frac{n}{\beta^2}
//' }
//' The function returns the Hessian matrix containing the negative of these values.
//'
//' Key properties of the returned matrix:
//' \itemize{
//'   \item Dimensions: 2x2.
//'   \item Symmetry: The matrix is symmetric.
//'   \item Ordering: Rows and columns correspond to the parameters in the order
//'     \eqn{\alpha, \beta}.
//'   \item Content: Analytic second derivatives of the *negative* log-likelihood.
//' }
//' This corresponds to the relevant 2x2 submatrix of the 5x5 GKw Hessian (\code{\link{hsgkw}})
//' evaluated at \eqn{\gamma=1, \delta=0, \lambda=1}.
//'
//' @references
//' Kumaraswamy, P. (1980). A generalized probability density function for
//' double-bounded random processes. *Journal of Hydrology*, *46*(1-2), 79-88.
//'
//'
//' Jones, M. C. (2009). Kumaraswamy's distribution: A beta-type distribution
//' with some tractability advantages. *Statistical Methodology*, *6*(1), 70-81.
//'
//' (Note: Specific Hessian formulas might be derived or sourced from additional references).
//'
//' @seealso
//' \code{\link{hsgkw}} (parent distribution Hessian),
//' \code{\link{llkw}} (negative log-likelihood for Kw),
//' \code{grkw} (gradient for Kw, if available),
//' \code{\link{dkw}} (density for Kw),
//' \code{\link[stats]{optim}},
//' \code{\link[numDeriv]{hessian}} (for numerical Hessian comparison).
//'
//' @examples
//' \donttest{
//' # Assuming existence of rkw, llkw, grkw, hskw functions for Kw
//'
//' # Generate sample data
//' set.seed(123)
//' true_par_kw <- c(alpha = 2, beta = 3)
//' sample_data_kw <- rkw(100, alpha = true_par_kw[1], beta = true_par_kw[2])
//' hist(sample_data_kw, breaks = 20, main = "Kw(2, 3) Sample")
//'
//' # --- Find MLE estimates ---
//' start_par_kw <- c(1.5, 2.5)
//' mle_result_kw <- stats::optim(par = start_par_kw,
//'                               fn = llkw,
//'                               gr = if (exists("grkw")) grkw else NULL,
//'                               method = "L-BFGS-B",
//'                               lower = c(1e-6, 1e-6),
//'                               hessian = TRUE, # Ask optim for numerical Hessian
//'                               data = sample_data_kw)
//'
//' # --- Compare analytical Hessian to numerical Hessian ---
//' if (mle_result_kw$convergence == 0 &&
//'     requireNamespace("numDeriv", quietly = TRUE) &&
//'     exists("hskw")) {
//'
//'   mle_par_kw <- mle_result_kw$par
//'   cat("\nComparing Hessians for Kw at MLE estimates:\n")
//'
//'   # Numerical Hessian of llkw
//'   num_hess_kw <- numDeriv::hessian(func = llkw, x = mle_par_kw, data = sample_data_kw)
//'
//'   # Analytical Hessian from hskw
//'   ana_hess_kw <- hskw(par = mle_par_kw, data = sample_data_kw)
//'
//'   cat("Numerical Hessian (Kw):\n")
//'   print(round(num_hess_kw, 4))
//'   cat("Analytical Hessian (Kw):\n")
//'   print(round(ana_hess_kw, 4))
//'
//'   # Check differences
//'   cat("Max absolute difference between Kw Hessians:\n")
//'   print(max(abs(num_hess_kw - ana_hess_kw)))
//'
//'   # Optional: Use analytical Hessian for Standard Errors
//'   # tryCatch({
//'   #   cov_matrix_kw <- solve(ana_hess_kw) # ana_hess_kw is already Hessian of negative LL
//'   #   std_errors_kw <- sqrt(diag(cov_matrix_kw))
//'   #   cat("Std. Errors from Analytical Kw Hessian:\n")
//'   #   print(std_errors_kw)
//'   # }, error = function(e) {
//'   #   warning("Could not invert analytical Kw Hessian: ", e$message)
//'   # })
//'
//' } else {
//'   cat("\nSkipping Kw Hessian comparison.\n")
//'   cat("Requires convergence, 'numDeriv' package, and function 'hskw'.\n")
//' }
//'
//' }
//'
//' @export
// [[Rcpp::export]]
Rcpp::NumericMatrix hskw(const Rcpp::NumericVector& par, const Rcpp::NumericVector& data) {
 // Parameter extraction
 double alpha = par[0];   // Shape parameter α > 0
 double beta = par[1];    // Shape parameter β > 0
 
 // Initialize Hessian matrix
 Rcpp::NumericMatrix hess(2, 2);
 
 // Parameter validation
 if (alpha <= 0 || beta <= 0) {
   hess.fill(R_NaN);
   return hess;
 }
 
 // Data conversion and validation
 arma::vec x = Rcpp::as<arma::vec>(data);
 
 if (arma::any(x <= 0) || arma::any(x >= 1)) {
   hess.fill(R_NaN);
   return hess;
 }
 
 int n = x.n_elem;  // Sample size
 
 // Small constant to avoid numerical issues
 double eps = std::numeric_limits<double>::epsilon() * 100;
 
 // Compute transformations and intermediate values
 arma::vec log_x = arma::log(x);                  // log(x_i)
 arma::vec log_x_squared = arma::square(log_x);   // (log(x_i))²
 arma::vec x_alpha = arma::pow(x, alpha);         // x_i^α
 arma::vec x_alpha_log_x = x_alpha % log_x;       // x_i^α * log(x_i)
 
 // v_i = 1 - x_i^α
 arma::vec v = 1.0 - x_alpha;
 v = arma::clamp(v, eps, 1.0 - eps);              // Prevent numerical issues
 
 // Additional terms for second derivatives
 arma::vec term_ratio = x_alpha / v;              // x_i^α / (1-x_i^α)
 arma::vec term_combined = 1.0 + term_ratio;      // 1 + x_i^α/(1-x_i^α)
 
 // Calculate the Hessian components for negative log-likelihood
 
 // H[0,0] = ∂²ℓ/∂α² = -n/α² - Σᵢ[(β-1)x_i^α(log(x_i))²/(1-x_i^α)(1 + x_i^α/(1-x_i^α))]
 double h_alpha_alpha = -n / (alpha * alpha);
 arma::vec d2a_terms = (beta - 1.0) * x_alpha % log_x_squared % term_combined / v;
 h_alpha_alpha -= arma::sum(d2a_terms);
 
 // H[0,1] = H[1,0] = ∂²ℓ/∂α∂β = -Σᵢ[x_i^α*log(x_i)/(1-x_i^α)]
 double h_alpha_beta = -arma::sum(x_alpha_log_x / v);
 
 // H[1,1] = ∂²ℓ/∂β² = -n/β²
 double h_beta_beta = -n / (beta * beta);
 
 // Fill the Hessian matrix (symmetric)
 hess(0, 0) = -h_alpha_alpha;
 hess(0, 1) = hess(1, 0) = -h_alpha_beta;
 hess(1, 1) = -h_beta_beta;
 
 return hess;
}
