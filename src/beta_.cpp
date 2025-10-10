// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include "utils.h"

/*
----------------------------------------------------------------------------
BETA DISTRIBUTION: Beta(γ, δ)
----------------------------------------------------------------------------

We use parameters gamma (γ) and delta (δ), both > 0, consistent with GKw family.
Domain: x in (0,1).

* PDF:
f(x;γ,δ) = x^(γ-1) * (1-x)^δ / B(γ,δ+1),  for 0<x<1.

* CDF:
F(x;γ,δ) = pbeta(x, γ, δ+1).

* QUANTILE:
Q(p;γ,δ) = qbeta(p, γ, δ+1).

* RNG:
X = rbeta(γ, δ+1).

* NEGATIVE LOG-LIKELIHOOD:
For data x_i in (0,1),
log f(x_i) = (γ-1)*log(x_i) + δ*log(1-x_i) - ln B(γ,δ+1).
Summation => negative => used in MLE.
*/

// -----------------------------------------------------------------------------
// 1) dbeta_: PDF for Beta distribution
// -----------------------------------------------------------------------------

//' @title Density of the Beta Distribution (gamma, delta+1 Parameterization)
//' @author Lopes, J. E.
//' @keywords distribution density beta
//'
//' @description
//' Computes the probability density function (PDF) for the standard Beta
//' distribution, using a parameterization common in generalized distribution
//' families. The distribution is parameterized by \code{gamma} (\eqn{\gamma}) and
//' \code{delta} (\eqn{\delta}), corresponding to the standard Beta distribution
//' with shape parameters \code{shape1 = gamma} and \code{shape2 = delta + 1}.
//' The distribution is defined on the interval (0, 1).
//'
//' @param x Vector of quantiles (values between 0 and 1).
//' @param gamma First shape parameter (\code{shape1}), \eqn{\gamma > 0}. Can be a
//'   scalar or a vector. Default: 1.0.
//' @param delta Second shape parameter is \code{delta + 1} (\code{shape2}), requires
//'   \eqn{\delta \ge 0} so that \code{shape2 >= 1}. Can be a scalar or a vector.
//'   Default: 0.0 (leading to \code{shape2 = 1}).
//' @param log_prob Logical; if \code{TRUE}, the logarithm of the density is
//'   returned (\eqn{\log(f(x))}). Default: \code{FALSE}.
//'
//' @return A vector of density values (\eqn{f(x)}) or log-density values
//'   (\eqn{\log(f(x))}). The length of the result is determined by the recycling
//'   rule applied to the arguments (\code{x}, \code{gamma}, \code{delta}).
//'   Returns \code{0} (or \code{-Inf} if \code{log_prob = TRUE}) for \code{x}
//'   outside the interval (0, 1), or \code{NaN} if parameters are invalid
//'   (e.g., \code{gamma <= 0}, \code{delta < 0}).
//'
//' @details
//' The probability density function (PDF) calculated by this function corresponds
//' to a standard Beta distribution \eqn{Beta(\gamma, \delta+1)}:
//' \deqn{
//' f(x; \gamma, \delta) = \frac{x^{\gamma-1} (1-x)^{(\delta+1)-1}}{B(\gamma, \delta+1)} = \frac{x^{\gamma-1} (1-x)^{\delta}}{B(\gamma, \delta+1)}
//' }
//' for \eqn{0 < x < 1}, where \eqn{B(a,b)} is the Beta function
//' (\code{\link[base]{beta}}).
//'
//' This specific parameterization arises as a special case of the five-parameter
//' Generalized Kumaraswamy (GKw) distribution (\code{\link{dgkw}}) obtained
//' by setting the parameters \eqn{\alpha = 1}, \eqn{\beta = 1}, and \eqn{\lambda = 1}.
//' It is therefore equivalent to the McDonald (Mc)/Beta Power distribution
//' (\code{\link{dmc}}) with \eqn{\lambda = 1}.
//'
//' Note the difference in the second parameter compared to \code{\link[stats]{dbeta}},
//' where \code{dbeta(x, shape1, shape2)} uses \code{shape2} directly. Here,
//' \code{shape1 = gamma} and \code{shape2 = delta + 1}.
//'
//' @references
//' Johnson, N. L., Kotz, S., & Balakrishnan, N. (1995). *Continuous Univariate
//' Distributions, Volume 2* (2nd ed.). Wiley.
//'
//' Cordeiro, G. M., & de Castro, M. (2011). A new family of generalized
//' distributions. *Journal of Statistical Computation and Simulation*,
//'
//'
//' @seealso
//' \code{\link[stats]{dbeta}} (standard R implementation),
//' \code{\link{dgkw}} (parent distribution density),
//' \code{\link{dmc}} (McDonald/Beta Power density),
//' \code{pbeta_}, \code{qbeta_}, \code{rbeta_} (other functions for this parameterization, if they exist).
//'
//' @examples
//' \donttest{
//' # Example values
//' x_vals <- c(0.2, 0.5, 0.8)
//' gamma_par <- 2.0 # Corresponds to shape1
//' delta_par <- 3.0 # Corresponds to shape2 - 1
//' shape1 <- gamma_par
//' shape2 <- delta_par + 1
//'
//' # Calculate density using dbeta_
//' densities <- dbeta_(x_vals, gamma_par, delta_par)
//' print(densities)
//'
//' # Compare with stats::dbeta
//' densities_stats <- stats::dbeta(x_vals, shape1 = shape1, shape2 = shape2)
//' print(paste("Max difference vs stats::dbeta:", max(abs(densities - densities_stats))))
//'
//' # Compare with dgkw setting alpha=1, beta=1, lambda=1
//' densities_gkw <- dgkw(x_vals, alpha = 1.0, beta = 1.0, gamma = gamma_par,
//'                       delta = delta_par, lambda = 1.0)
//' print(paste("Max difference vs dgkw:", max(abs(densities - densities_gkw))))
//'
//' # Compare with dmc setting lambda=1
//' densities_mc <- dmc(x_vals, gamma = gamma_par, delta = delta_par, lambda = 1.0)
//' print(paste("Max difference vs dmc:", max(abs(densities - densities_mc))))
//'
//' # Calculate log-density
//' log_densities <- dbeta_(x_vals, gamma_par, delta_par, log_prob = TRUE)
//' print(log_densities)
//' print(stats::dbeta(x_vals, shape1 = shape1, shape2 = shape2, log = TRUE))
//'
//' # Plot the density
//' curve_x <- seq(0.001, 0.999, length.out = 200)
//' curve_y <- dbeta_(curve_x, gamma = 2, delta = 3) # Beta(2, 4)
//' plot(curve_x, curve_y, type = "l", main = "Beta(2, 4) Density via dbeta_",
//'      xlab = "x", ylab = "f(x)", col = "blue")
//' curve(stats::dbeta(x, 2, 4), add=TRUE, col="red", lty=2)
//' legend("topright", legend=c("dbeta_(gamma=2, delta=3)", "stats::dbeta(shape1=2, shape2=4)"),
//'        col=c("blue", "red"), lty=c(1,2), bty="n")
//'
//' }
//'
//' @export
// [[Rcpp::export]]
Rcpp::NumericVector dbeta_(
   const arma::vec& x,
   const Rcpp::NumericVector& gamma,
   const Rcpp::NumericVector& delta,
   bool log_prob = false
) {
 arma::vec g_vec(gamma.begin(), gamma.size());
 arma::vec d_vec(delta.begin(), delta.size());
 
 size_t N = std::max({x.n_elem, g_vec.n_elem, d_vec.n_elem});
 arma::vec out(N);
 out.fill(log_prob ? R_NegInf : 0.0);
 
 for (size_t i = 0; i < N; i++) {
   double g = g_vec[i % g_vec.n_elem];
   double d = d_vec[i % d_vec.n_elem];
   double xx = x[i % x.n_elem];
   
   if (!check_beta_pars(g, d)) {
     continue; // => 0 or -Inf
   }
   if (xx <= 0.0 || xx >= 1.0 || !R_finite(xx)) {
     continue;
   }
   
   // logBeta = R::lbeta(g, d+1)
   double lB = R::lbeta(g, d + 1.0);
   // log pdf = (g-1)*log(x) + d*log(1-x) - lB
   double lx = std::log(xx);
   double one_minus_x = 1.0 - xx;
   if (one_minus_x <= 0.0) {
     // => out of domain, effectively => 0
     continue;
   }
   double log_1_minus_x = std::log(one_minus_x);
   
   double log_pdf = (g - 1.0) * lx + d * log_1_minus_x - lB;
   
   if (log_prob) {
     out(i) = log_pdf;
   } else {
     out(i) = std::exp(log_pdf);
   }
 }
 
 return Rcpp::NumericVector(out.memptr(), out.memptr() + out.n_elem);
}


// -----------------------------------------------------------------------------
// 2) pbeta_: CDF for Beta
// -----------------------------------------------------------------------------


//' @title CDF of the Beta Distribution (gamma, delta+1 Parameterization)
//' @author Lopes, J. E.
//' @keywords distribution cumulative beta
//'
//' @description
//' Computes the cumulative distribution function (CDF), \eqn{F(q) = P(X \le q)},
//' for the standard Beta distribution, using a parameterization common in
//' generalized distribution families. The distribution is parameterized by
//' \code{gamma} (\eqn{\gamma}) and \code{delta} (\eqn{\delta}), corresponding to
//' the standard Beta distribution with shape parameters \code{shape1 = gamma}
//' and \code{shape2 = delta + 1}.
//'
//' @param q Vector of quantiles (values generally between 0 and 1).
//' @param gamma First shape parameter (\code{shape1}), \eqn{\gamma > 0}. Can be a
//'   scalar or a vector. Default: 1.0.
//' @param delta Second shape parameter is \code{delta + 1} (\code{shape2}), requires
//'   \eqn{\delta \ge 0} so that \code{shape2 >= 1}. Can be a scalar or a vector.
//'   Default: 0.0 (leading to \code{shape2 = 1}).
//' @param lower_tail Logical; if \code{TRUE} (default), probabilities are
//'   \eqn{P(X \le q)}, otherwise, \eqn{P(X > q)}.
//' @param log_p Logical; if \code{TRUE}, probabilities \eqn{p} are given as
//'   \eqn{\log(p)}. Default: \code{FALSE}.
//'
//' @return A vector of probabilities, \eqn{F(q)}, or their logarithms/complements
//'   depending on \code{lower_tail} and \code{log_p}. The length of the result
//'   is determined by the recycling rule applied to the arguments (\code{q},
//'   \code{gamma}, \code{delta}). Returns \code{0} (or \code{-Inf} if
//'   \code{log_p = TRUE}) for \code{q <= 0} and \code{1} (or \code{0} if
//'   \code{log_p = TRUE}) for \code{q >= 1}. Returns \code{NaN} for invalid
//'   parameters.
//'
//' @details
//' This function computes the CDF of a Beta distribution with parameters
//' \code{shape1 = gamma} and \code{shape2 = delta + 1}. It is equivalent to
//' calling \code{stats::pbeta(q, shape1 = gamma, shape2 = delta + 1,
//' lower.tail = lower_tail, log.p = log_p)}.
//'
//' This distribution arises as a special case of the five-parameter
//' Generalized Kumaraswamy (GKw) distribution (\code{\link{pgkw}}) obtained
//' by setting \eqn{\alpha = 1}, \eqn{\beta = 1}, and \eqn{\lambda = 1}.
//' It is therefore also equivalent to the McDonald (Mc)/Beta Power distribution
//' (\code{\link{pmc}}) with \eqn{\lambda = 1}.
//'
//' The function likely calls R's underlying \code{pbeta} function but ensures
//' consistent parameter recycling and handling within the C++ environment,
//' matching the style of other functions in the related families.
//'
//' @references
//' Johnson, N. L., Kotz, S., & Balakrishnan, N. (1995). *Continuous Univariate
//' Distributions, Volume 2* (2nd ed.). Wiley.
//'
//' Cordeiro, G. M., & de Castro, M. (2011). A new family of generalized
//' distributions. *Journal of Statistical Computation and Simulation*,
//'
//'
//' @seealso
//' \code{\link[stats]{pbeta}} (standard R implementation),
//' \code{\link{pgkw}} (parent distribution CDF),
//' \code{\link{pmc}} (McDonald/Beta Power CDF),
//' \code{dbeta_}, \code{qbeta_}, \code{rbeta_} (other functions for this parameterization, if they exist).
//'
//' @examples
//' \donttest{
//' # Example values
//' q_vals <- c(0.2, 0.5, 0.8)
//' gamma_par <- 2.0 # Corresponds to shape1
//' delta_par <- 3.0 # Corresponds to shape2 - 1
//' shape1 <- gamma_par
//' shape2 <- delta_par + 1
//'
//' # Calculate CDF using pbeta_
//' probs <- pbeta_(q_vals, gamma_par, delta_par)
//' print(probs)
//'
//' # Compare with stats::pbeta
//' probs_stats <- stats::pbeta(q_vals, shape1 = shape1, shape2 = shape2)
//' print(paste("Max difference vs stats::pbeta:", max(abs(probs - probs_stats))))
//'
//' # Compare with pgkw setting alpha=1, beta=1, lambda=1
//' probs_gkw <- pgkw(q_vals, alpha = 1.0, beta = 1.0, gamma = gamma_par,
//'                   delta = delta_par, lambda = 1.0)
//' print(paste("Max difference vs pgkw:", max(abs(probs - probs_gkw))))
//'
//' # Compare with pmc setting lambda=1
//' probs_mc <- pmc(q_vals, gamma = gamma_par, delta = delta_par, lambda = 1.0)
//' print(paste("Max difference vs pmc:", max(abs(probs - probs_mc))))
//'
//' # Calculate upper tail P(X > q)
//' probs_upper <- pbeta_(q_vals, gamma_par, delta_par, lower_tail = FALSE)
//' print(probs_upper)
//' print(stats::pbeta(q_vals, shape1, shape2, lower.tail = FALSE))
//'
//' # Calculate log CDF
//' log_probs <- pbeta_(q_vals, gamma_par, delta_par, log_p = TRUE)
//' print(log_probs)
//' print(stats::pbeta(q_vals, shape1, shape2, log.p = TRUE))
//'
//' # Plot the CDF
//' curve_q <- seq(0.001, 0.999, length.out = 200)
//' curve_p <- pbeta_(curve_q, gamma = 2, delta = 3) # Beta(2, 4)
//' plot(curve_q, curve_p, type = "l", main = "Beta(2, 4) CDF via pbeta_",
//'      xlab = "q", ylab = "F(q)", col = "blue")
//' curve(stats::pbeta(x, 2, 4), add=TRUE, col="red", lty=2)
//' legend("bottomright", legend=c("pbeta_(gamma=2, delta=3)", "stats::pbeta(shape1=2, shape2=4)"),
//'        col=c("blue", "red"), lty=c(1,2), bty="n")
//'
//' }
//'
//' @export
// [[Rcpp::export]]
Rcpp::NumericVector pbeta_(
   const arma::vec& q,
   const Rcpp::NumericVector& gamma,
   const Rcpp::NumericVector& delta,
   bool lower_tail = true,
   bool log_p = false
) {
 arma::vec g_vec(gamma.begin(), gamma.size());
 arma::vec d_vec(delta.begin(), delta.size());
 
 size_t N = std::max({q.n_elem, g_vec.n_elem, d_vec.n_elem});
 arma::vec out(N);
 
 for (size_t i = 0; i < N; i++) {
   double g = g_vec[i % g_vec.n_elem];
   double d = d_vec[i % d_vec.n_elem];
   double qq = q[i % q.n_elem];
   
   if (!check_beta_pars(g, d)) {
     out(i) = NA_REAL;
     continue;
   }
   
   // boundary
   if (!R_finite(qq) || qq <= 0.0) {
     double v0 = lower_tail ? 0.0 : 1.0;
     out(i) = (log_p ? std::log(v0) : v0);
     continue;
   }
   if (qq >= 1.0) {
     double v1 = lower_tail ? 1.0 : 0.0;
     out(i) = (log_p ? std::log(v1) : v1);
     continue;
   }
   
   // call R's pbeta with adjusted parameters for GKw-style Beta
   double val = R::pbeta(qq, g, d + 1.0, lower_tail, false); // not log
   if (log_p) {
     val = std::log(val);
   }
   out(i) = val;
 }
 
 return Rcpp::NumericVector(out.memptr(), out.memptr() + out.n_elem);
}


// -----------------------------------------------------------------------------
// 3) qbeta_: Quantile function for Beta
// -----------------------------------------------------------------------------

//' @title Quantile Function of the Beta Distribution (gamma, delta+1 Parameterization)
//' @author Lopes, J. E.
//' @keywords distribution quantile beta
//'
//' @description
//' Computes the quantile function (inverse CDF) for the standard Beta
//' distribution, using a parameterization common in generalized distribution
//' families. It finds the value \code{q} such that \eqn{P(X \le q) = p}. The
//' distribution is parameterized by \code{gamma} (\eqn{\gamma}) and \code{delta}
//' (\eqn{\delta}), corresponding to the standard Beta distribution with shape
//' parameters \code{shape1 = gamma} and \code{shape2 = delta + 1}.
//'
//' @param p Vector of probabilities (values between 0 and 1).
//' @param gamma First shape parameter (\code{shape1}), \eqn{\gamma > 0}. Can be a
//'   scalar or a vector. Default: 1.0.
//' @param delta Second shape parameter is \code{delta + 1} (\code{shape2}), requires
//'   \eqn{\delta \ge 0} so that \code{shape2 >= 1}. Can be a scalar or a vector.
//'   Default: 0.0 (leading to \code{shape2 = 1}).
//' @param lower_tail Logical; if \code{TRUE} (default), probabilities are \eqn{p = P(X \le q)},
//'   otherwise, probabilities are \eqn{p = P(X > q)}.
//' @param log_p Logical; if \code{TRUE}, probabilities \code{p} are given as
//'   \eqn{\log(p)}. Default: \code{FALSE}.
//'
//' @return A vector of quantiles corresponding to the given probabilities \code{p}.
//'   The length of the result is determined by the recycling rule applied to
//'   the arguments (\code{p}, \code{gamma}, \code{delta}).
//'   Returns:
//'   \itemize{
//'     \item \code{0} for \code{p = 0} (or \code{p = -Inf} if \code{log_p = TRUE},
//'           when \code{lower_tail = TRUE}).
//'     \item \code{1} for \code{p = 1} (or \code{p = 0} if \code{log_p = TRUE},
//'           when \code{lower_tail = TRUE}).
//'     \item \code{NaN} for \code{p < 0} or \code{p > 1} (or corresponding log scale).
//'     \item \code{NaN} for invalid parameters (e.g., \code{gamma <= 0},
//'           \code{delta < 0}).
//'   }
//'   Boundary return values are adjusted accordingly for \code{lower_tail = FALSE}.
//'
//' @details
//' This function computes the quantiles of a Beta distribution with parameters
//' \code{shape1 = gamma} and \code{shape2 = delta + 1}. It is equivalent to
//' calling \code{stats::qbeta(p, shape1 = gamma, shape2 = delta + 1,
//' lower.tail = lower_tail, log.p = log_p)}.
//'
//' This distribution arises as a special case of the five-parameter
//' Generalized Kumaraswamy (GKw) distribution (\code{\link{qgkw}}) obtained
//' by setting \eqn{\alpha = 1}, \eqn{\beta = 1}, and \eqn{\lambda = 1}.
//' It is therefore also equivalent to the McDonald (Mc)/Beta Power distribution
//' (\code{\link{qmc}}) with \eqn{\lambda = 1}.
//'
//' The function likely calls R's underlying \code{qbeta} function but ensures
//' consistent parameter recycling and handling within the C++ environment,
//' matching the style of other functions in the related families. Boundary
//' conditions (p=0, p=1) are handled explicitly.
//'
//' @references
//' Johnson, N. L., Kotz, S., & Balakrishnan, N. (1995). *Continuous Univariate
//' Distributions, Volume 2* (2nd ed.). Wiley.
//'
//' Cordeiro, G. M., & de Castro, M. (2011). A new family of generalized
//' distributions. *Journal of Statistical Computation and Simulation*,
//'
//'
//' @seealso
//' \code{\link[stats]{qbeta}} (standard R implementation),
//' \code{\link{qgkw}} (parent distribution quantile function),
//' \code{\link{qmc}} (McDonald/Beta Power quantile function),
//' \code{dbeta_}, \code{pbeta_}, \code{rbeta_} (other functions for this parameterization, if they exist).
//'
//' @examples
//' \donttest{
//' # Example values
//' p_vals <- c(0.1, 0.5, 0.9)
//' gamma_par <- 2.0 # Corresponds to shape1
//' delta_par <- 3.0 # Corresponds to shape2 - 1
//' shape1 <- gamma_par
//' shape2 <- delta_par + 1
//'
//' # Calculate quantiles using qbeta_
//' quantiles <- qbeta_(p_vals, gamma_par, delta_par)
//' print(quantiles)
//'
//' # Compare with stats::qbeta
//' quantiles_stats <- stats::qbeta(p_vals, shape1 = shape1, shape2 = shape2)
//' print(paste("Max difference vs stats::qbeta:", max(abs(quantiles - quantiles_stats))))
//'
//' # Compare with qgkw setting alpha=1, beta=1, lambda=1
//' quantiles_gkw <- qgkw(p_vals, alpha = 1.0, beta = 1.0, gamma = gamma_par,
//'                       delta = delta_par, lambda = 1.0)
//' print(paste("Max difference vs qgkw:", max(abs(quantiles - quantiles_gkw))))
//'
//' # Compare with qmc setting lambda=1
//' quantiles_mc <- qmc(p_vals, gamma = gamma_par, delta = delta_par, lambda = 1.0)
//' print(paste("Max difference vs qmc:", max(abs(quantiles - quantiles_mc))))
//'
//' # Calculate quantiles for upper tail
//' quantiles_upper <- qbeta_(p_vals, gamma_par, delta_par, lower_tail = FALSE)
//' print(quantiles_upper)
//' print(stats::qbeta(p_vals, shape1, shape2, lower.tail = FALSE))
//'
//' # Calculate quantiles from log probabilities
//' log_p_vals <- log(p_vals)
//' quantiles_logp <- qbeta_(log_p_vals, gamma_par, delta_par, log_p = TRUE)
//' print(quantiles_logp)
//' print(stats::qbeta(log_p_vals, shape1, shape2, log.p = TRUE))
//'
//' # Verify inverse relationship with pbeta_
//' p_check <- 0.75
//' q_calc <- qbeta_(p_check, gamma_par, delta_par)
//' p_recalc <- pbeta_(q_calc, gamma_par, delta_par)
//' print(paste("Original p:", p_check, " Recalculated p:", p_recalc))
//' # abs(p_check - p_recalc) < 1e-9 # Should be TRUE
//'
//' # Boundary conditions
//' print(qbeta_(c(0, 1), gamma_par, delta_par)) # Should be 0, 1
//' print(qbeta_(c(-Inf, 0), gamma_par, delta_par, log_p = TRUE)) # Should be 0, 1
//'
//' }
//'
//' @export
// [[Rcpp::export]]
Rcpp::NumericVector qbeta_(
   const arma::vec& p,
   const Rcpp::NumericVector& gamma,
   const Rcpp::NumericVector& delta,
   bool lower_tail = true,
   bool log_p = false
) {
 arma::vec g_vec(gamma.begin(), gamma.size());
 arma::vec d_vec(delta.begin(), delta.size());
 
 size_t N = std::max({p.n_elem, g_vec.n_elem, d_vec.n_elem});
 arma::vec out(N);
 
 for (size_t i = 0; i < N; i++) {
   double g = g_vec[i % g_vec.n_elem];
   double d = d_vec[i % d_vec.n_elem];
   double pp = p[i % p.n_elem];
   
   if (!check_beta_pars(g, d)) {
     out(i) = NA_REAL;
     continue;
   }
   
   // handle log_p
   if (log_p) {
     if (pp > 0.0) {
       // => p>1
       out(i) = NA_REAL;
       continue;
     }
     pp = std::exp(pp);
   }
   // handle lower_tail
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
   
   // Use adjusted parameters for GKw-style Beta
   double val = R::qbeta(pp, g, d + 1.0, true, false); // returns not log
   out(i) = val;
 }
 
 return Rcpp::NumericVector(out.memptr(), out.memptr() + out.n_elem);
}


// -----------------------------------------------------------------------------
// 4) rbeta_: RNG for Beta distribution
// -----------------------------------------------------------------------------

//' @title Random Generation for the Beta Distribution (gamma, delta+1 Parameterization)
//' @author Lopes, J. E.
//' @keywords distribution random beta
//'
//' @description
//' Generates random deviates from the standard Beta distribution, using a
//' parameterization common in generalized distribution families. The distribution
//' is parameterized by \code{gamma} (\eqn{\gamma}) and \code{delta} (\eqn{\delta}),
//' corresponding to the standard Beta distribution with shape parameters
//' \code{shape1 = gamma} and \code{shape2 = delta + 1}. This is a special case
//' of the Generalized Kumaraswamy (GKw) distribution where \eqn{\alpha = 1},
//' \eqn{\beta = 1}, and \eqn{\lambda = 1}.
//'
//' @param n Number of observations. If \code{length(n) > 1}, the length is
//'   taken to be the number required. Must be a non-negative integer.
//' @param gamma First shape parameter (\code{shape1}), \eqn{\gamma > 0}. Can be a
//'   scalar or a vector. Default: 1.0.
//' @param delta Second shape parameter is \code{delta + 1} (\code{shape2}), requires
//'   \eqn{\delta \ge 0} so that \code{shape2 >= 1}. Can be a scalar or a vector.
//'   Default: 0.0 (leading to \code{shape2 = 1}, i.e., Uniform).
//'
//' @return A numeric vector of length \code{n} containing random deviates from the
//'   Beta(\eqn{\gamma, \delta+1}) distribution, with values in (0, 1). The length
//'   of the result is determined by \code{n} and the recycling rule applied to
//'   the parameters (\code{gamma}, \code{delta}). Returns \code{NaN} if parameters
//'   are invalid (e.g., \code{gamma <= 0}, \code{delta < 0}).
//'
//' @details
//' This function generates samples from a Beta distribution with parameters
//' \code{shape1 = gamma} and \code{shape2 = delta + 1}. It is equivalent to
//' calling \code{stats::rbeta(n, shape1 = gamma, shape2 = delta + 1)}.
//'
//' This distribution arises as a special case of the five-parameter
//' Generalized Kumaraswamy (GKw) distribution (\code{\link{rgkw}}) obtained
//' by setting \eqn{\alpha = 1}, \eqn{\beta = 1}, and \eqn{\lambda = 1}.
//' It is therefore also equivalent to the McDonald (Mc)/Beta Power distribution
//' (\code{\link{rmc}}) with \eqn{\lambda = 1}.
//'
//' The function likely calls R's underlying \code{rbeta} function but ensures
//' consistent parameter recycling and handling within the C++ environment,
//' matching the style of other functions in the related families.
//'
//' @references
//' Johnson, N. L., Kotz, S., & Balakrishnan, N. (1995). *Continuous Univariate
//' Distributions, Volume 2* (2nd ed.). Wiley.
//'
//' Cordeiro, G. M., & de Castro, M. (2011). A new family of generalized
//' distributions. *Journal of Statistical Computation and Simulation*,
//'
//'
//' Devroye, L. (1986). *Non-Uniform Random Variate Generation*. Springer-Verlag.
//'
//' @seealso
//' \code{\link[stats]{rbeta}} (standard R implementation),
//' \code{\link{rgkw}} (parent distribution random generation),
//' \code{\link{rmc}} (McDonald/Beta Power random generation),
//' \code{dbeta_}, \code{pbeta_}, \code{qbeta_} (other functions for this parameterization, if they exist).
//'
//' @examples
//' \donttest{
//' set.seed(2030) # for reproducibility
//'
//' # Generate 1000 samples using rbeta_
//' gamma_par <- 2.0 # Corresponds to shape1
//' delta_par <- 3.0 # Corresponds to shape2 - 1
//' shape1 <- gamma_par
//' shape2 <- delta_par + 1
//'
//' x_sample <- rbeta_(1000, gamma = gamma_par, delta = delta_par)
//' summary(x_sample)
//'
//' # Compare with stats::rbeta
//' x_sample_stats <- stats::rbeta(1000, shape1 = shape1, shape2 = shape2)
//' # Visually compare histograms or QQ-plots
//' hist(x_sample, main="rbeta_ Sample", freq=FALSE, breaks=30)
//' curve(dbeta_(x, gamma_par, delta_par), add=TRUE, col="red", lwd=2)
//' hist(x_sample_stats, main="stats::rbeta Sample", freq=FALSE, breaks=30)
//' curve(stats::dbeta(x, shape1, shape2), add=TRUE, col="blue", lwd=2)
//' # Compare summary stats (should be similar due to randomness)
//' print(summary(x_sample))
//' print(summary(x_sample_stats))
//'
//' # Compare summary stats with rgkw(alpha=1, beta=1, lambda=1)
//' x_sample_gkw <- rgkw(1000, alpha = 1.0, beta = 1.0, gamma = gamma_par,
//'                      delta = delta_par, lambda = 1.0)
//' print("Summary stats for rgkw(a=1,b=1,l=1) sample:")
//' print(summary(x_sample_gkw))
//'
//' # Compare summary stats with rmc(lambda=1)
//' x_sample_mc <- rmc(1000, gamma = gamma_par, delta = delta_par, lambda = 1.0)
//' print("Summary stats for rmc(l=1) sample:")
//' print(summary(x_sample_mc))
//'
//' }
//'
//' @export
// [[Rcpp::export]]
Rcpp::NumericVector rbeta_(
   int n,
   const Rcpp::NumericVector& gamma,
   const Rcpp::NumericVector& delta
) {
 if (n <= 0) {
   Rcpp::stop("rbeta_: n must be positive");
 }
 
 arma::vec g_vec(gamma.begin(), gamma.size());
 arma::vec d_vec(delta.begin(), delta.size());
 
 size_t k = std::max({g_vec.n_elem, d_vec.n_elem});
 arma::vec out(n);
 
 for (int i = 0; i < n; i++) {
   size_t idx = i % k;
   double g = g_vec[idx % g_vec.n_elem];
   double d = d_vec[idx % d_vec.n_elem];
   
   if (!check_beta_pars(g, d)) {
     out(i) = NA_REAL;
     Rcpp::warning("rbeta_: invalid parameters at index %d", i+1);
     continue;
   }
   
   // Use adjusted parameters for GKw-style Beta
   double val = R::rbeta(g, d + 1.0);
   out(i) = val;
 }
 return Rcpp::NumericVector(out.memptr(), out.memptr() + out.n_elem);
}


// -----------------------------------------------------------------------------
// 5) llbeta: Negative Log-Likelihood for Beta
// -----------------------------------------------------------------------------

//' @title Negative Log-Likelihood for the Beta Distribution (gamma, delta+1 Parameterization)
//' @author Lopes, J. E.
//' @keywords distribution likelihood optimize beta
//'
//' @description
//' Computes the negative log-likelihood function for the standard Beta
//' distribution, using a parameterization common in generalized distribution
//' families. The distribution is parameterized by \code{gamma} (\eqn{\gamma}) and
//' \code{delta} (\eqn{\delta}), corresponding to the standard Beta distribution
//' with shape parameters \code{shape1 = gamma} and \code{shape2 = delta + 1}.
//' This function is suitable for maximum likelihood estimation.
//'
//' @param par A numeric vector of length 2 containing the distribution parameters
//'   in the order: \code{gamma} (\eqn{\gamma > 0}), \code{delta} (\eqn{\delta \ge 0}).
//' @param data A numeric vector of observations. All values must be strictly
//'   between 0 and 1 (exclusive).
//'
//' @return Returns a single \code{double} value representing the negative
//'   log-likelihood (\eqn{-\ell(\theta|\mathbf{x})}). Returns \code{Inf}
//'   if any parameter values in \code{par} are invalid according to their
//'   constraints, or if any value in \code{data} is not in the interval (0, 1).
//'
//' @details
//' This function calculates the negative log-likelihood for a Beta distribution
//' with parameters \code{shape1 = gamma} (\eqn{\gamma}) and \code{shape2 = delta + 1} (\eqn{\delta+1}).
//' The probability density function (PDF) is:
//' \deqn{
//' f(x | \gamma, \delta) = \frac{x^{\gamma-1} (1-x)^{\delta}}{B(\gamma, \delta+1)}
//' }
//' for \eqn{0 < x < 1}, where \eqn{B(a,b)} is the Beta function (\code{\link[base]{beta}}).
//' The log-likelihood function \eqn{\ell(\theta | \mathbf{x})} for a sample
//' \eqn{\mathbf{x} = (x_1, \dots, x_n)} is \eqn{\sum_{i=1}^n \ln f(x_i | \theta)}:
//' \deqn{
//' \ell(\theta | \mathbf{x}) = \sum_{i=1}^{n} [(\gamma-1)\ln(x_i) + \delta\ln(1-x_i)] - n \ln B(\gamma, \delta+1)
//' }
//' where \eqn{\theta = (\gamma, \delta)}.
//' This function computes and returns the *negative* log-likelihood, \eqn{-\ell(\theta|\mathbf{x})},
//' suitable for minimization using optimization routines like \code{\link[stats]{optim}}.
//' It is equivalent to the negative log-likelihood of the GKw distribution
//' (\code{\link{llgkw}}) evaluated at \eqn{\alpha=1, \beta=1, \lambda=1}, and also
//' to the negative log-likelihood of the McDonald distribution (\code{\link{llmc}})
//' evaluated at \eqn{\lambda=1}. The term \eqn{\ln B(\gamma, \delta+1)} is typically
//' computed using log-gamma functions (\code{\link[base]{lgamma}}) for numerical stability.
//'
//' @references
//' Johnson, N. L., Kotz, S., & Balakrishnan, N. (1995). *Continuous Univariate
//' Distributions, Volume 2* (2nd ed.). Wiley.
//'
//' Cordeiro, G. M., & de Castro, M. (2011). A new family of generalized
//' distributions. *Journal of Statistical Computation and Simulation*,
//'
//'
//' @seealso
//' \code{\link{llgkw}}, \code{\link{llmc}} (related negative log-likelihoods),
//' \code{dbeta_}, \code{pbeta_}, \code{qbeta_}, \code{rbeta_},
//' \code{grbeta} (gradient, if available),
//' \code{hsbeta} (Hessian, if available),
//' \code{\link[stats]{optim}}, \code{\link[base]{lbeta}}.
//'
//' @examples
//' \donttest{
//' # Assuming existence of rbeta_, llbeta, grbeta, hsbeta functions
//'
//' # Generate sample data from a Beta(2, 4) distribution
//' # (gamma=2, delta=3 in this parameterization)
//' set.seed(123)
//' true_par_beta <- c(gamma = 2, delta = 3)
//' sample_data_beta <- rbeta_(100, gamma = true_par_beta[1], delta = true_par_beta[2])
//' hist(sample_data_beta, breaks = 20, main = "Beta(2, 4) Sample")
//'
//' # --- Maximum Likelihood Estimation using optim ---
//' # Initial parameter guess
//' start_par_beta <- c(1.5, 2.5)
//'
//' # Perform optimization (minimizing negative log-likelihood)
//' # Use method="L-BFGS-B" for box constraints (params > 0 / >= 0)
//' mle_result_beta <- stats::optim(par = start_par_beta,
//'                                fn = llbeta, # Use the custom Beta neg-log-likelihood
//'                                method = "L-BFGS-B",
//'                                lower = c(1e-6, 1e-6), # Bounds: gamma>0, delta>=0
//'                                hessian = TRUE,
//'                                data = sample_data_beta)
//'
//' # Check convergence and results
//' if (mle_result_beta$convergence == 0) {
//'   print("Optimization converged successfully.")
//'   mle_par_beta <- mle_result_beta$par
//'   print("Estimated Beta parameters (gamma, delta):")
//'   print(mle_par_beta)
//'   print("True Beta parameters (gamma, delta):")
//'   print(true_par_beta)
//'   cat(sprintf("MLE corresponds approx to Beta(%.2f, %.2f)\n",
//'       mle_par_beta[1], mle_par_beta[2] + 1))
//'
//' } else {
//'   warning("Optimization did not converge!")
//'   print(mle_result_beta$message)
//' }
//'
//' # --- Compare numerical and analytical derivatives (if available) ---
//' # Requires 'numDeriv' package and analytical functions 'grbeta', 'hsbeta'
//' if (mle_result_beta$convergence == 0 &&
//'     requireNamespace("numDeriv", quietly = TRUE) &&
//'     exists("grbeta") && exists("hsbeta")) {
//'
//'   cat("\nComparing Derivatives at Beta MLE estimates:\n")
//'
//'   # Numerical derivatives of llbeta
//'   num_grad_beta <- numDeriv::grad(func = llbeta, x = mle_par_beta, data = sample_data_beta)
//'   num_hess_beta <- numDeriv::hessian(func = llbeta, x = mle_par_beta, data = sample_data_beta)
//'
//'   # Analytical derivatives (assuming they return derivatives of negative LL)
//'   ana_grad_beta <- grbeta(par = mle_par_beta, data = sample_data_beta)
//'   ana_hess_beta <- hsbeta(par = mle_par_beta, data = sample_data_beta)
//'
//'   # Check differences
//'   cat("Max absolute difference between gradients:\n")
//'   print(max(abs(num_grad_beta - ana_grad_beta)))
//'   cat("Max absolute difference between Hessians:\n")
//'   print(max(abs(num_hess_beta - ana_hess_beta)))
//'
//' } else {
//'    cat("\nSkipping derivative comparison for Beta.\n")
//'    cat("Requires convergence, 'numDeriv' pkg & functions 'grbeta', 'hsbeta'.\n")
//' }
//'
//' }
//'
//' @export
// [[Rcpp::export]]
double llbeta(const Rcpp::NumericVector& par,
             const Rcpp::NumericVector& data) {
 if (par.size() < 2) {
   return R_PosInf;
 }
 double gamma = par[0]; // gamma > 0
 double delta = par[1]; // delta > 0
 
 if (!check_beta_pars(gamma, delta)) {
   return R_PosInf;
 }
 
 arma::vec x = Rcpp::as<arma::vec>(data);
 if (x.n_elem < 1) {
   return R_PosInf;
 }
 // domain check
 if (arma::any(x <= 0.0) || arma::any(x >= 1.0)) {
   return R_PosInf;
 }
 
 int n = x.n_elem;
 // Use correct parameterization for GKw-style Beta
 double logB = R::lbeta(gamma, delta + 1.0);
 // constant => -n*logB
 double cst = -double(n) * logB;
 
 // sum((gamma-1)*log(x_i) + delta*log(1-x_i)), i=1..n
 arma::vec lx = arma::log(x);
 arma::vec l1mx = arma::log(1.0 - x);
 
 double sum1 = (gamma - 1.0) * arma::sum(lx);
 double sum2 = delta * arma::sum(l1mx);  // Corrected: no subtraction of 1.0
 
 double loglike = cst + sum1 + sum2; // that's the log-likelihood
 
 // We must return negative
 return -loglike;
}


//' @title Gradient of the Negative Log-Likelihood for the Beta Distribution (gamma, delta+1 Parameterization)
//' @author Lopes, J. E.
//' @keywords distribution likelihood optimize gradient beta
//'
//' @description
//' Computes the gradient vector (vector of first partial derivatives) of the
//' negative log-likelihood function for the standard Beta distribution, using
//' a parameterization common in generalized distribution families. The
//' distribution is parameterized by \code{gamma} (\eqn{\gamma}) and \code{delta}
//' (\eqn{\delta}), corresponding to the standard Beta distribution with shape
//' parameters \code{shape1 = gamma} and \code{shape2 = delta + 1}.
//' The gradient is useful for optimization algorithms.
//'
//' @param par A numeric vector of length 2 containing the distribution parameters
//'   in the order: \code{gamma} (\eqn{\gamma > 0}), \code{delta} (\eqn{\delta \ge 0}).
//' @param data A numeric vector of observations. All values must be strictly
//'   between 0 and 1 (exclusive).
//'
//' @return Returns a numeric vector of length 2 containing the partial derivatives
//'   of the negative log-likelihood function \eqn{-\ell(\theta | \mathbf{x})} with
//'   respect to each parameter: \eqn{(-\partial \ell/\partial \gamma, -\partial \ell/\partial \delta)}.
//'   Returns a vector of \code{NaN} if any parameter values are invalid according
//'   to their constraints, or if any value in \code{data} is not in the
//'   interval (0, 1).
//'
//' @details
//' This function calculates the gradient of the negative log-likelihood for a
//' Beta distribution with parameters \code{shape1 = gamma} (\eqn{\gamma}) and
//' \code{shape2 = delta + 1} (\eqn{\delta+1}). The components of the gradient
//' vector (\eqn{-\nabla \ell(\theta | \mathbf{x})}) are:
//'
//' \deqn{
//' -\frac{\partial \ell}{\partial \gamma} = n[\psi(\gamma) - \psi(\gamma+\delta+1)] -
//' \sum_{i=1}^{n}\ln(x_i)
//' }
//' \deqn{
//' -\frac{\partial \ell}{\partial \delta} = n[\psi(\delta+1) - \psi(\gamma+\delta+1)] -
//' \sum_{i=1}^{n}\ln(1-x_i)
//' }
//'
//' where \eqn{\psi(\cdot)} is the digamma function (\code{\link[base]{digamma}}).
//' These formulas represent the derivatives of \eqn{-\ell(\theta)}, consistent with
//' minimizing the negative log-likelihood. They correspond to the relevant components
//' of the general GKw gradient (\code{\link{grgkw}}) evaluated at \eqn{\alpha=1, \beta=1, \lambda=1}.
//' Note the parameterization: the standard Beta shape parameters are \eqn{\gamma} and \eqn{\delta+1}.
//'
//' @references
//' Johnson, N. L., Kotz, S., & Balakrishnan, N. (1995). *Continuous Univariate
//' Distributions, Volume 2* (2nd ed.). Wiley.
//'
//' Cordeiro, G. M., & de Castro, M. (2011). A new family of generalized
//' distributions. *Journal of Statistical Computation and Simulation*,
//'
//' (Note: Specific gradient formulas might be derived or sourced from additional references).
//'
//' @seealso
//' \code{\link{grgkw}}, \code{\link{grmc}} (related gradients),
//' \code{\link{llbeta}} (negative log-likelihood function),
//' \code{hsbeta} (Hessian, if available),
//' \code{dbeta_}, \code{pbeta_}, \code{qbeta_}, \code{rbeta_},
//' \code{\link[stats]{optim}},
//' \code{\link[numDeriv]{grad}} (for numerical gradient comparison),
//' \code{\link[base]{digamma}}.
//'
//' @examples
//' \donttest{
//' # Assuming existence of rbeta_, llbeta, grbeta, hsbeta functions
//'
//' # Generate sample data from a Beta(2, 4) distribution
//' # (gamma=2, delta=3 in this parameterization)
//' set.seed(123)
//' true_par_beta <- c(gamma = 2, delta = 3)
//' sample_data_beta <- rbeta_(100, gamma = true_par_beta[1], delta = true_par_beta[2])
//' hist(sample_data_beta, breaks = 20, main = "Beta(2, 4) Sample")
//'
//' # --- Find MLE estimates ---
//' start_par_beta <- c(1.5, 2.5)
//' mle_result_beta <- stats::optim(par = start_par_beta,
//'                                fn = llbeta,
//'                                gr = grbeta, # Use analytical gradient
//'                                method = "L-BFGS-B",
//'                                lower = c(1e-6, 1e-6), # Bounds: gamma>0, delta>=0
//'                                hessian = TRUE,
//'                                data = sample_data_beta)
//'
//' # --- Compare analytical gradient to numerical gradient ---
//' if (mle_result_beta$convergence == 0 &&
//'     requireNamespace("numDeriv", quietly = TRUE)) {
//'
//'   mle_par_beta <- mle_result_beta$par
//'   cat("\nComparing Gradients for Beta at MLE estimates:\n")
//'
//'   # Numerical gradient of llbeta
//'   num_grad_beta <- numDeriv::grad(func = llbeta, x = mle_par_beta, data = sample_data_beta)
//'
//'   # Analytical gradient from grbeta
//'   ana_grad_beta <- grbeta(par = mle_par_beta, data = sample_data_beta)
//'
//'   cat("Numerical Gradient (Beta):\n")
//'   print(num_grad_beta)
//'   cat("Analytical Gradient (Beta):\n")
//'   print(ana_grad_beta)
//'
//'   # Check differences
//'   cat("Max absolute difference between Beta gradients:\n")
//'   print(max(abs(num_grad_beta - ana_grad_beta)))
//'
//' } else {
//'   cat("\nSkipping Beta gradient comparison.\n")
//' }
//'
//' # Example with Hessian comparison (if hsbeta exists)
//' if (mle_result_beta$convergence == 0 &&
//'     requireNamespace("numDeriv", quietly = TRUE) && exists("hsbeta")) {
//'
//'   num_hess_beta <- numDeriv::hessian(func = llbeta, x = mle_par_beta, data = sample_data_beta)
//'   ana_hess_beta <- hsbeta(par = mle_par_beta, data = sample_data_beta)
//'   cat("\nMax absolute difference between Beta Hessians:\n")
//'   print(max(abs(num_hess_beta - ana_hess_beta)))
//'
//' }
//'
//' }
//'
//' @export
// [[Rcpp::export]]
Rcpp::NumericVector grbeta(const Rcpp::NumericVector& par,
                          const Rcpp::NumericVector& data) {
 Rcpp::NumericVector grad(2, R_PosInf); // initialize with Inf
 
 if (par.size() < 2) {
   return grad;
 }
 
 double gamma = par[0];
 double delta = par[1];
 
 if (!check_beta_pars(gamma, delta)) {
   return grad;
 }
 
 arma::vec x = Rcpp::as<arma::vec>(data);
 if (x.n_elem < 1) {
   return grad;
 }
 
 // domain check
 if (arma::any(x <= 0.0) || arma::any(x >= 1.0)) {
   return grad;
 }
 
 int n = x.n_elem;
 
 // Calculate digamma terms with correct parameterization for GKw-style Beta
 double dig_g = R::digamma(gamma);
 double dig_d = R::digamma(delta + 1.0);  // Corrected: digamma(δ+1)
 double dig_gd = R::digamma(gamma + delta + 1.0);  // Corrected: digamma(γ+δ+1)
 
 // Sum of log terms
 arma::vec lx = arma::log(x);
 arma::vec l1mx = arma::log(1.0 - x);
 double sum_lx = arma::sum(lx);
 double sum_l1mx = arma::sum(l1mx);
 
 // Partial derivatives for negative log-likelihood
 grad[0] = n * (dig_g - dig_gd) - sum_lx; // wrt gamma
 grad[1] = n * (dig_d - dig_gd) - sum_l1mx; // wrt delta
 
 return grad; // Already negated for negative log-likelihood
}


//' @title Hessian Matrix of the Negative Log-Likelihood for the Beta Distribution (gamma, delta+1 Parameterization)
//' @author Lopes, J. E.
//' @keywords distribution likelihood optimize hessian beta
//'
//' @description
//' Computes the analytic 2x2 Hessian matrix (matrix of second partial derivatives)
//' of the negative log-likelihood function for the standard Beta distribution,
//' using a parameterization common in generalized distribution families. The
//' distribution is parameterized by \code{gamma} (\eqn{\gamma}) and \code{delta}
//' (\eqn{\delta}), corresponding to the standard Beta distribution with shape
//' parameters \code{shape1 = gamma} and \code{shape2 = delta + 1}. The Hessian
//' is useful for estimating standard errors and in optimization algorithms.
//'
//' @param par A numeric vector of length 2 containing the distribution parameters
//'   in the order: \code{gamma} (\eqn{\gamma > 0}), \code{delta} (\eqn{\delta \ge 0}).
//' @param data A numeric vector of observations. All values must be strictly
//'   between 0 and 1 (exclusive).
//'
//' @return Returns a 2x2 numeric matrix representing the Hessian matrix of the
//'   negative log-likelihood function, \eqn{-\partial^2 \ell / (\partial \theta_i \partial \theta_j)},
//'   where \eqn{\theta = (\gamma, \delta)}.
//'   Returns a 2x2 matrix populated with \code{NaN} if any parameter values are
//'   invalid according to their constraints, or if any value in \code{data} is
//'   not in the interval (0, 1).
//'
//' @details
//' This function calculates the analytic second partial derivatives of the
//' negative log-likelihood function (\eqn{-\ell(\theta|\mathbf{x})}) for a Beta
//' distribution with parameters \code{shape1 = gamma} (\eqn{\gamma}) and
//' \code{shape2 = delta + 1} (\eqn{\delta+1}). The components of the Hessian
//' matrix (\eqn{-\mathbf{H}(\theta)}) are:
//'
//' \deqn{
//' -\frac{\partial^2 \ell}{\partial \gamma^2} = n[\psi'(\gamma) - \psi'(\gamma+\delta+1)]
//' }
//' \deqn{
//' -\frac{\partial^2 \ell}{\partial \gamma \partial \delta} = -n\psi'(\gamma+\delta+1)
//' }
//' \deqn{
//' -\frac{\partial^2 \ell}{\partial \delta^2} = n[\psi'(\delta+1) - \psi'(\gamma+\delta+1)]
//' }
//'
//' where \eqn{\psi'(\cdot)} is the trigamma function (\code{\link[base]{trigamma}}).
//' These formulas represent the second derivatives of \eqn{-\ell(\theta)},
//' consistent with minimizing the negative log-likelihood. They correspond to
//' the relevant 2x2 submatrix of the general GKw Hessian (\code{\link{hsgkw}})
//' evaluated at \eqn{\alpha=1, \beta=1, \lambda=1}. Note the parameterization
//' difference from the standard Beta distribution (\code{shape2 = delta + 1}).
//'
//' The returned matrix is symmetric.
//'
//' @references
//' Johnson, N. L., Kotz, S., & Balakrishnan, N. (1995). *Continuous Univariate
//' Distributions, Volume 2* (2nd ed.). Wiley.
//'
//' Cordeiro, G. M., & de Castro, M. (2011). A new family of generalized
//' distributions. *Journal of Statistical Computation and Simulation*,
//'
//' (Note: Specific Hessian formulas might be derived or sourced from additional references).
//'
//' @seealso
//' \code{\link{hsgkw}}, \code{\link{hsmc}} (related Hessians),
//' \code{\link{llbeta}} (negative log-likelihood function),
//' \code{grbeta} (gradient, if available),
//' \code{dbeta_}, \code{pbeta_}, \code{qbeta_}, \code{rbeta_},
//' \code{\link[stats]{optim}},
//' \code{\link[numDeriv]{hessian}} (for numerical Hessian comparison),
//' \code{\link[base]{trigamma}}.
//'
//' @examples
//' \donttest{
//' # Assuming existence of rbeta_, llbeta, grbeta, hsbeta functions
//'
//' # Generate sample data from a Beta(2, 4) distribution
//' # (gamma=2, delta=3 in this parameterization)
//' set.seed(123)
//' true_par_beta <- c(gamma = 2, delta = 3)
//' sample_data_beta <- rbeta_(100, gamma = true_par_beta[1], delta = true_par_beta[2])
//' hist(sample_data_beta, breaks = 20, main = "Beta(2, 4) Sample")
//'
//' # --- Find MLE estimates ---
//' start_par_beta <- c(1.5, 2.5)
//' mle_result_beta <- stats::optim(par = start_par_beta,
//'                                fn = llbeta,
//'                                gr = if (exists("grbeta")) grbeta else NULL,
//'                                method = "L-BFGS-B",
//'                                lower = c(1e-6, 1e-6), # Bounds: gamma>0, delta>=0
//'                                hessian = TRUE, # Ask optim for numerical Hessian
//'                                data = sample_data_beta)
//'
//' # --- Compare analytical Hessian to numerical Hessian ---
//' if (mle_result_beta$convergence == 0 &&
//'     requireNamespace("numDeriv", quietly = TRUE) &&
//'     exists("hsbeta")) {
//'
//'   mle_par_beta <- mle_result_beta$par
//'   cat("\nComparing Hessians for Beta at MLE estimates:\n")
//'
//'   # Numerical Hessian of llbeta
//'   num_hess_beta <- numDeriv::hessian(func = llbeta, x = mle_par_beta, data = sample_data_beta)
//'
//'   # Analytical Hessian from hsbeta
//'   ana_hess_beta <- hsbeta(par = mle_par_beta, data = sample_data_beta)
//'
//'   cat("Numerical Hessian (Beta):\n")
//'   print(round(num_hess_beta, 4))
//'   cat("Analytical Hessian (Beta):\n")
//'   print(round(ana_hess_beta, 4))
//'
//'   # Check differences
//'   cat("Max absolute difference between Beta Hessians:\n")
//'   print(max(abs(num_hess_beta - ana_hess_beta)))
//'
//'   # Optional: Use analytical Hessian for Standard Errors
//'   # tryCatch({
//'   #   cov_matrix_beta <- solve(ana_hess_beta) # ana_hess_beta is already Hessian of negative LL
//'   #   std_errors_beta <- sqrt(diag(cov_matrix_beta))
//'   #   cat("Std. Errors from Analytical Beta Hessian:\n")
//'   #   print(std_errors_beta)
//'   # }, error = function(e) {
//'   #   warning("Could not invert analytical Beta Hessian: ", e$message)
//'   # })
//'
//' } else {
//'   cat("\nSkipping Beta Hessian comparison.\n")
//'   cat("Requires convergence, 'numDeriv' package, and function 'hsbeta'.\n")
//' }
//'
//' }
//'
//' @export
// [[Rcpp::export]]
Rcpp::NumericMatrix hsbeta(const Rcpp::NumericVector& par,
                          const Rcpp::NumericVector& data) {
 Rcpp::NumericMatrix hess(2, 2);
 // Initialize with Inf
 for (int i = 0; i < 2; i++) {
   for (int j = 0; j < 2; j++) {
     hess(i, j) = R_PosInf;
   }
 }
 
 if (par.size() < 2) {
   return hess;
 }
 
 double gamma = par[0];
 double delta = par[1];
 
 if (!check_beta_pars(gamma, delta)) {
   return hess;
 }
 
 arma::vec x = Rcpp::as<arma::vec>(data);
 if (x.n_elem < 1) {
   return hess;
 }
 
 // domain check
 if (arma::any(x <= 0.0) || arma::any(x >= 1.0)) {
   return hess;
 }
 
 int n = x.n_elem;
 
 // Calculate trigamma terms with correct parameterization for GKw-style Beta
 double trig_g = R::trigamma(gamma);
 double trig_d = R::trigamma(delta + 1.0);  // Corrected: trigamma(δ+1)
 double trig_gd = R::trigamma(gamma + delta + 1.0);  // Corrected: trigamma(γ+δ+1)
 
 // Second derivatives for negative log-likelihood
 hess(0, 0) = n * (trig_g - trig_gd);  // d²/dγ²
 hess(1, 1) = n * (trig_d - trig_gd);  // d²/dδ²
 hess(0, 1) = hess(1, 0) = -n * trig_gd;  // d²/dγdδ
 
 return hess; // Already for negative log-likelihood
}
