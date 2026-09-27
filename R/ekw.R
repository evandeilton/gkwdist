# ============================================================================#
# EXPONENTIATED KUMARASWAMY (EKw) DISTRIBUTION - R WRAPPERS
# ============================================================================#
#
# Wrapper functions for the EXPONENTIATED KUMARASWAMY (EKw) DISTRIBUTION.
# C++ implementations are in src/mc.cpp
#
# Functions:
#   - dekq: Probability density function (PDF)
#   - pekq: Cumulative distribution function (CDF)
#   - qekq: Quantile function (inverse CDF)
#   - rekq: Random number generation
#   - llekq: Negative log-likelihood
#   - grekq: Gradient of negative log-likelihood
#   - hsekq: Hessian of negative log-likelihood
# ============================================================================#

#' @title Density of the Exponentiated Kumaraswamy (EKw) Distribution
#'
#' @author Lopes, J. E.
#' @family density functions
#' @concept exponentiated kumaraswamy
#' @keywords distribution
#'
#' @description
#' Computes the probability density function (PDF) for the Exponentiated
#' Kumaraswamy (EKw) distribution with parameters \code{alpha} (\eqn{\alpha}),
#' \code{beta} (\eqn{\beta}), and \code{lambda} (\eqn{\lambda}).
#' This distribution is defined on the interval (0, 1).
#'
#' @param x Vector of quantiles (values between 0 and 1).
#' @param alpha Shape parameter \code{alpha} > 0. Can be a scalar or a vector.
#'   Default: 1.0.
#' @param beta Shape parameter \code{beta} > 0. Can be a scalar or a vector.
#'   Default: 1.0.
#' @param lambda Shape parameter \code{lambda} > 0 (exponent parameter).
#'   Can be a scalar or a vector. Default: 1.0.
#' @param log Logical; if \code{TRUE}, the logarithm of the density is
#'   returned (\eqn{\log(f(x))}). Default: \code{FALSE}.
#'
#' @return A vector of density values (\eqn{f(x)}) or log-density values
#'   (\eqn{\log(f(x))}). The length of the result is determined by the recycling
#'   rule applied to the arguments (\code{x}, \code{alpha}, \code{beta},
#'   \code{lambda}). Returns \code{0} (or \code{-Inf} if
#'   \code{log = TRUE}) for \code{x} strictly outside the interval \[0, 1\]. At the
#'   closed boundaries \code{x = 0} and \code{x = 1} the limiting density is
#'   returned rather than \code{0}, following the convention of base R's density
#'   functions (compare \code{\link[stats]{dbeta}}); depending on the parameters
#'   that limit is \code{0}, a finite positive value, or \code{Inf}.
#'   An out-of-bound or missing parameter is an error, not a return value: the
#'   wrapper stops with a message naming the parameter. An infinite parameter is
#'   not currently intercepted there and reaches the C++ layer, which treats it
#'   as invalid.
#'
#' @details
#' The probability density function (PDF) of the Exponentiated Kumaraswamy (EKw)
#' distribution is given by:
#' \deqn{
#' f(x; \alpha, \beta, \lambda) = \lambda \alpha \beta x^{\alpha-1} (1 - x^\alpha)^{\beta-1} \bigl[1 - (1 - x^\alpha)^\beta \bigr]^{\lambda - 1}
#' }
#' for \eqn{0 < x < 1}.
#'
#' The EKw distribution is a special case of the five-parameter
#' Generalized Kumaraswamy (GKw) distribution (\code{\link{dgkw}}) obtained
#' by setting the parameters \eqn{\gamma = 1} and \eqn{\delta = 0}.
#' When \eqn{\lambda = 1}, the EKw distribution reduces to the standard
#' Kumaraswamy distribution.
#'
#' @references
#' Nadarajah, S., Cordeiro, G. M., & Ortega, E. M. (2012). The exponentiated
#' Kumaraswamy distribution. *Journal of the Franklin Institute*, *349*(3),
#'
#' Cordeiro, G. M., & de Castro, M. (2011). A new family of generalized
#' distributions. *Journal of Statistical Computation and Simulation*,
#' *81*(7), 883-898.
#' \doi{10.1080/00949650903530745}
#'
#' Kumaraswamy, P. (1980). A generalized probability density function for
#' double-bounded random processes. *Journal of Hydrology*, *46*(1-2), 79-88.
#' \doi{10.1016/0022-1694(80)90036-0}
#'
#' @seealso
#' \code{\link{dgkw}} (parent distribution density),
#' \code{\link{pekw}}, \code{\link{qekw}}, \code{\link{rekw}} (other EKw functions),
#'
#' @examples
#' x <- c(0.1, 0.3, 0.5, 0.7, 0.9)
#' dekw(x, alpha = 2, beta = 3, lambda = 1.2)
#' dekw(x, alpha = 2, beta = 3, lambda = 1.2, log = TRUE)
#'
#' ## EKw is GKw with gamma = 1, delta = 0
#' all.equal(dekw(x, 2, 3, 1.2), dgkw(x, 2, 3, gamma = 1, delta = 0, lambda = 1.2))
#'
#' ## The density integrates to one
#' integrate(dekw, 0, 1, alpha = 2, beta = 3, lambda = 1.2, rel.tol = 1e-10)
#'
#' curve(dekw(x, alpha = 2, beta = 3, lambda = 1.2), from = 0, to = 1, ylab = "density")
#'
#' @export
dekw <- function(x, alpha = 1, beta = 1, lambda = 1, log = FALSE) {
  if (!is.numeric(x)) stop("'x' must be numeric")
  if (!is.numeric(alpha) || anyNA(alpha) || any(alpha <= 0)) {
    stop("'alpha' must be positive (alpha > 0)")
  }
  if (!is.numeric(beta) || anyNA(beta) || any(beta <= 0)) {
    stop("'beta' must be positive (beta > 0)")
  }
  if (!is.numeric(lambda) || anyNA(lambda) || any(lambda <= 0)) {
    stop("'lambda' must be positive (lambda > 0)")
  }
  if (!is.logical(log) || length(log) != 1 || is.na(log)) {
    stop("'log' must be a single logical value")
  }

  .shape_like(.Call("_gkwdist_dekw",
    as.numeric(x),
    as.numeric(alpha),
    as.numeric(beta),
    as.numeric(lambda),
    as.logical(log),
    PACKAGE = "gkwdist"
  ), x)
}


#' @title Cumulative Distribution Function (CDF) of the EKw Distribution
#' @author Lopes, J. E.
#' @family cumulative distribution functions
#' @concept exponentiated kumaraswamy
#' @keywords distribution
#'
#' @description
#' Computes the cumulative distribution function (CDF), \eqn{P(X \le q)}, for the
#' Exponentiated Kumaraswamy (EKw) distribution with parameters \code{alpha}
#' (\eqn{\alpha}), \code{beta} (\eqn{\beta}), and \code{lambda} (\eqn{\lambda}).
#' This distribution is defined on the interval (0, 1) and is a special case
#' of the Generalized Kumaraswamy (GKw) distribution where \eqn{\gamma = 1}
#' and \eqn{\delta = 0}.
#'
#' @param q Vector of quantiles (values generally between 0 and 1).
#' @param alpha Shape parameter \code{alpha} > 0. Can be a scalar or a vector.
#'   Default: 1.0.
#' @param beta Shape parameter \code{beta} > 0. Can be a scalar or a vector.
#'   Default: 1.0.
#' @param lambda Shape parameter \code{lambda} > 0 (exponent parameter).
#'   Can be a scalar or a vector. Default: 1.0.
#' @param lower.tail Logical; if \code{TRUE} (default), probabilities are
#'   \eqn{P(X \le q)}, otherwise, \eqn{P(X > q)}.
#' @param log.p Logical; if \code{TRUE}, probabilities \eqn{p} are given as
#'   \eqn{\log(p)}. Default: \code{FALSE}.
#'
#' @return A vector of probabilities, \eqn{F(q)}, or their logarithms/complements
#'   depending on \code{lower.tail} and \code{log.p}. The length of the result
#'   is determined by the recycling rule applied to the arguments (\code{q},
#'   \code{alpha}, \code{beta}, \code{lambda}). When \code{lower.tail = TRUE},
#'   returns \code{0} (or \code{-Inf} if \code{log.p = TRUE}) for \code{q <= 0}
#'   and \code{1} (or \code{0} if \code{log.p = TRUE}) for \code{q >= 1}. An
#'   out-of-bound or missing parameter is an error, not a return value: the
#'   wrapper stops with a message naming the parameter. An infinite parameter
#'   is not currently intercepted there and reaches the C++ layer, which
#'   treats it as invalid.
#'   Boundary return values are adjusted accordingly for \code{lower.tail = FALSE}.
#'
#' @details
#' The Exponentiated Kumaraswamy (EKw) distribution is a special case of the
#' five-parameter Generalized Kumaraswamy distribution (\code{\link{pgkw}})
#' obtained by setting parameters \eqn{\gamma = 1} and \eqn{\delta = 0}.
#'
#' The CDF of the GKw distribution is \eqn{F_{GKw}(q) = I_{y(q)}(\gamma, \delta+1)},
#' where \eqn{y(q) = [1-(1-q^{\alpha})^{\beta}]^{\lambda}} and \eqn{I_x(a,b)}
#' is the regularized incomplete beta function (\code{\link[stats]{pbeta}}).
#' Setting \eqn{\gamma=1} and \eqn{\delta=0} gives \eqn{I_{y(q)}(1, 1)}. Since
#' \eqn{I_x(1, 1) = x}, the CDF simplifies to \eqn{y(q)}:
#' \deqn{
#' F(q; \alpha, \beta, \lambda) = \bigl[1 - (1 - q^\alpha)^\beta \bigr]^\lambda
#' }
#' for \eqn{0 < q < 1}.
#' The implementation uses this closed-form expression for efficiency and handles
#' \code{lower.tail} and \code{log.p} arguments appropriately.
#'
#' @references
#' Nadarajah, S., Cordeiro, G. M., & Ortega, E. M. (2012). The exponentiated
#' Kumaraswamy distribution. *Journal of the Franklin Institute*, *349*(3),
#'
#'
#' Cordeiro, G. M., & de Castro, M. (2011). A new family of generalized
#' distributions. *Journal of Statistical Computation and Simulation*,
#' *81*(7), 883-898.
#' \doi{10.1080/00949650903530745}
#'
#' Kumaraswamy, P. (1980). A generalized probability density function for
#' double-bounded random processes. *Journal of Hydrology*, *46*(1-2), 79-88.
#' \doi{10.1016/0022-1694(80)90036-0}
#'
#'
#' @seealso
#' \code{\link{pgkw}} (parent distribution CDF),
#' \code{\link{dekw}}, \code{\link{qekw}}, \code{\link{rekw}} (other EKw functions),
#'
#' @examples
#' q <- c(0.2, 0.5, 0.8)
#' pekw(q, alpha = 2, beta = 3, lambda = 1.2)
#' pekw(q, alpha = 2, beta = 3, lambda = 1.2, lower.tail = FALSE)  # P(X > q)
#' pekw(q, alpha = 2, beta = 3, lambda = 1.2, log.p = TRUE)
#'
#' ## pekw() is the integral of dekw()
#' Fq <- pekw(0.5, alpha = 2, beta = 3, lambda = 1.2)
#' all.equal(Fq, integrate(dekw, 0, 0.5, alpha = 2, beta = 3, lambda = 1.2,
#'     rel.tol = 1e-10)$value)
#'
#' @export
pekw <- function(q, alpha = 1, beta = 1, lambda = 1, lower.tail = TRUE, log.p = FALSE) {
  if (!is.numeric(q)) stop("'q' must be numeric")
  if (!is.numeric(alpha) || anyNA(alpha) || any(alpha <= 0)) {
    stop("'alpha' must be positive (alpha > 0)")
  }
  if (!is.numeric(beta) || anyNA(beta) || any(beta <= 0)) {
    stop("'beta' must be positive (beta > 0)")
  }
  if (!is.numeric(lambda) || anyNA(lambda) || any(lambda <= 0)) {
    stop("'lambda' must be positive (lambda > 0)")
  }
  if (!is.logical(lower.tail) || length(lower.tail) != 1 || is.na(lower.tail)) {
    stop("'lower.tail' must be a single logical value")
  }
  if (!is.logical(log.p) || length(log.p) != 1 || is.na(log.p)) {
    stop("'log.p' must be a single logical value")
  }

  .shape_like(.Call("_gkwdist_pekw",
    as.numeric(q),
    as.numeric(alpha),
    as.numeric(beta),
    as.numeric(lambda),
    as.logical(lower.tail),
    as.logical(log.p),
    PACKAGE = "gkwdist"
  ), q)
}


# 3) qekw: Quantile of Exponentiated Kumaraswamy

#' @title Quantile Function of the Exponentiated Kumaraswamy (EKw) Distribution
#' @author Lopes, J. E.
#' @family quantile functions
#' @concept exponentiated kumaraswamy
#' @keywords distribution
#'
#' @description
#' Computes the quantile function (inverse CDF) for the Exponentiated
#' Kumaraswamy (EKw) distribution with parameters \code{alpha} (\eqn{\alpha}),
#' \code{beta} (\eqn{\beta}), and \code{lambda} (\eqn{\lambda}).
#' It finds the value \code{q} such that \eqn{P(X \le q) = p}. This distribution
#' is a special case of the Generalized Kumaraswamy (GKw) distribution where
#' \eqn{\gamma = 1} and \eqn{\delta = 0}.
#'
#' @param p Vector of probabilities (values between 0 and 1).
#' @param alpha Shape parameter \code{alpha} > 0. Can be a scalar or a vector.
#'   Default: 1.0.
#' @param beta Shape parameter \code{beta} > 0. Can be a scalar or a vector.
#'   Default: 1.0.
#' @param lambda Shape parameter \code{lambda} > 0 (exponent parameter).
#'   Can be a scalar or a vector. Default: 1.0.
#' @param lower.tail Logical; if \code{TRUE} (default), probabilities are \eqn{p = P(X \le q)},
#'   otherwise, probabilities are \eqn{p = P(X > q)}.
#' @param log.p Logical; if \code{TRUE}, probabilities \code{p} are given as
#'   \eqn{\log(p)}. Default: \code{FALSE}.
#'
#' @return A vector of quantiles corresponding to the given probabilities \code{p}.
#'   The length of the result is determined by the recycling rule applied to
#'   the arguments (\code{p}, \code{alpha}, \code{beta}, \code{lambda}).
#'   Returns:
#'   \itemize{
#'     \item \code{0} for \code{p = 0} (or \code{p = -Inf} if \code{log.p = TRUE},
#'           when \code{lower.tail = TRUE}).
#'     \item \code{1} for \code{p = 1} (or \code{p = 0} if \code{log.p = TRUE},
#'           when \code{lower.tail = TRUE}).
#'     \item \code{NaN} for \code{p < 0} or \code{p > 1} (or corresponding log scale).
#'     \item An out-of-bound or missing parameter is an error, not a
#'       return value: the wrapper stops with a message naming the parameter.
#'       An infinite parameter is not currently intercepted there and reaches
#'       the C++ layer, which treats it as invalid.
#'   }
#'   Boundary return values are adjusted accordingly for \code{lower.tail = FALSE}.
#'
#' @details
#' The quantile function \eqn{Q(p)} is the inverse of the CDF \eqn{F(q)}. The CDF
#' for the EKw (\eqn{\gamma=1, \delta=0}) distribution is \eqn{F(q) = [1 - (1 - q^\alpha)^\beta ]^\lambda}
#' (see \code{\link{pekw}}). Inverting this equation for \eqn{q} yields the
#' quantile function:
#' \deqn{
#' Q(p) = \left\{ 1 - \left[ 1 - p^{1/\lambda} \right]^{1/\beta} \right\}^{1/\alpha}
#' }
#' The function uses this closed-form expression and correctly handles the
#' \code{lower.tail} and \code{log.p} arguments by transforming \code{p}
#' appropriately before applying the formula. This is equivalent to the general
#' GKw quantile function (\code{\link{qgkw}}) evaluated with \eqn{\gamma=1, \delta=0}.
#'
#' @references
#' Nadarajah, S., Cordeiro, G. M., & Ortega, E. M. (2012). The exponentiated
#' Kumaraswamy distribution. *Journal of the Franklin Institute*, *349*(3),
#'
#'
#' Cordeiro, G. M., & de Castro, M. (2011). A new family of generalized
#' distributions. *Journal of Statistical Computation and Simulation*,
#' *81*(7), 883-898.
#' \doi{10.1080/00949650903530745}
#'
#' Kumaraswamy, P. (1980). A generalized probability density function for
#' double-bounded random processes. *Journal of Hydrology*, *46*(1-2), 79-88.
#' \doi{10.1016/0022-1694(80)90036-0}
#'
#'
#' @seealso
#' \code{\link{qgkw}} (parent distribution quantile function),
#' \code{\link{dekw}}, \code{\link{pekw}}, \code{\link{rekw}} (other EKw functions),
#' \code{\link[stats]{qunif}}
#'
#' @examples
#' p <- c(0.1, 0.5, 0.9)
#' qekw(p, alpha = 2, beta = 3, lambda = 1.2)
#' # upper-tail quantiles
#' qekw(p, alpha = 2, beta = 3, lambda = 1.2, lower.tail = FALSE)
#'
#' ## qekw() inverts pekw()
#' all.equal(pekw(qekw(p, alpha = 2, beta = 3, lambda = 1.2), alpha = 2,
#'     beta = 3, lambda = 1.2), p)
#'
#' @export
qekw <- function(p, alpha = 1, beta = 1, lambda = 1, lower.tail = TRUE, log.p = FALSE) {
  if (!is.numeric(p)) stop("'p' must be numeric")
  if (!is.numeric(alpha) || anyNA(alpha) || any(alpha <= 0)) {
    stop("'alpha' must be positive (alpha > 0)")
  }
  if (!is.numeric(beta) || anyNA(beta) || any(beta <= 0)) {
    stop("'beta' must be positive (beta > 0)")
  }
  if (!is.numeric(lambda) || anyNA(lambda) || any(lambda <= 0)) {
    stop("'lambda' must be positive (lambda > 0)")
  }
  if (!is.logical(lower.tail) || length(lower.tail) != 1 || is.na(lower.tail)) {
    stop("'lower.tail' must be a single logical value")
  }
  if (!is.logical(log.p) || length(log.p) != 1 || is.na(log.p)) {
    stop("'log.p' must be a single logical value")
  }
  if (!log.p && any(p < 0 | p > 1, na.rm = TRUE)) {
    warning("'p' values outside [0, 1] will produce NaN")
  }

  .shape_like(.Call("_gkwdist_qekw",
    as.numeric(p),
    as.numeric(alpha),
    as.numeric(beta),
    as.numeric(lambda),
    as.logical(lower.tail),
    as.logical(log.p),
    PACKAGE = "gkwdist"
  ), p)
}


#' @title Random Number Generation for the Exponentiated Kumaraswamy (EKw) Distribution
#' @author Lopes, J. E.
#' @family random generation functions
#' @concept exponentiated kumaraswamy
#' @keywords distribution
#'
#' @description
#' Generates random deviates from the Exponentiated Kumaraswamy (EKw)
#' distribution with parameters \code{alpha} (\eqn{\alpha}), \code{beta}
#' (\eqn{\beta}), and \code{lambda} (\eqn{\lambda}). This distribution is a
#' special case of the Generalized Kumaraswamy (GKw) distribution where
#' \eqn{\gamma = 1} and \eqn{\delta = 0}.
#'
#' @param n Number of observations. If \code{length(n) > 1}, the length is
#'   taken to be the number required. Must be a non-negative integer.
#' @param alpha Shape parameter \code{alpha} > 0. Can be a scalar or a vector.
#'   Default: 1.0.
#' @param beta Shape parameter \code{beta} > 0. Can be a scalar or a vector.
#'   Default: 1.0.
#' @param lambda Shape parameter \code{lambda} > 0 (exponent parameter).
#'   Can be a scalar or a vector. Default: 1.0.
#'
#' @return A vector of length \code{n} containing random deviates from the EKw
#'   distribution. The length of the result is determined by \code{n} and the
#'   recycling rule applied to the parameters (\code{alpha}, \code{beta},
#'   \code{lambda}). An out-of-bound or missing parameter is an
#'   error, not a return value: the wrapper stops with a message naming the
#'   parameter. An infinite parameter is not currently intercepted there and
#'   reaches the C++ layer, which treats it as invalid.
#'
#' @details
#' The generation method uses the inverse transform (quantile) method.
#' That is, if \eqn{U} is a random variable following a standard Uniform
#' distribution on (0, 1), then \eqn{X = Q(U)} follows the EKw distribution,
#' where \eqn{Q(u)} is the EKw quantile function (\code{\link{qekw}}):
#' \deqn{
#' Q(u) = \left\{ 1 - \left[ 1 - u^{1/\lambda} \right]^{1/\beta} \right\}^{1/\alpha}
#' }
#' This is computationally equivalent to the general GKw generation method
#' (\code{\link{rgkw}}) when specialized for \eqn{\gamma=1, \delta=0}, as the
#' required Beta(1, 1) random variate is equivalent to a standard Uniform(0, 1)
#' variate. The implementation generates \eqn{U} using \code{\link[stats]{runif}}
#' and applies the transformation above.
#'
#' @references
#' Nadarajah, S., Cordeiro, G. M., & Ortega, E. M. (2012). The exponentiated
#' Kumaraswamy distribution. *Journal of the Franklin Institute*, *349*(3),
#'
#'
#' Cordeiro, G. M., & de Castro, M. (2011). A new family of generalized
#' distributions. *Journal of Statistical Computation and Simulation*,
#' *81*(7), 883-898.
#' \doi{10.1080/00949650903530745}
#'
#' Kumaraswamy, P. (1980). A generalized probability density function for
#' double-bounded random processes. *Journal of Hydrology*, *46*(1-2), 79-88.
#' \doi{10.1016/0022-1694(80)90036-0}
#'
#'
#' Devroye, L. (1986). *Non-Uniform Random Variate Generation*. Springer-Verlag.
#' (General methods for random variate generation).
#'
#' @seealso
#' \code{\link{rgkw}} (parent distribution random generation),
#' \code{\link{dekw}}, \code{\link{pekw}}, \code{\link{qekw}} (other EKw functions),
#' \code{\link[stats]{runif}}
#'
#' @examples
#' set.seed(123)
#' x <- rekw(1000, alpha = 2, beta = 3, lambda = 1.2)
#' summary(x)
#'
#' ## The sample follows the distribution
#' hist(x, breaks = 30, freq = FALSE, main = "")
#' curve(dekw(x, alpha = 2, beta = 3, lambda = 1.2), add = TRUE)
#' ks.test(x, pekw, alpha = 2, beta = 3, lambda = 1.2)
#'
#' @export
rekw <- function(n, alpha = 1, beta = 1, lambda = 1) {
  if (length(n) > 1) n <- length(n)
  # n = 0 is legal and yields numeric(0), as stats::rbeta(0, 2, 3) does.
  # Negative and missing n stay errors, which is also what base R does.
  if (!is.numeric(n) || length(n) != 1 || is.na(n) || n < 0) {
    stop("'n' must be a single non-negative integer")
  }
  n <- as.integer(n)

  if (!is.numeric(alpha) || anyNA(alpha) || any(alpha <= 0)) {
    stop("'alpha' must be positive (alpha > 0)")
  }
  if (!is.numeric(beta) || anyNA(beta) || any(beta <= 0)) {
    stop("'beta' must be positive (beta > 0)")
  }
  if (!is.numeric(lambda) || anyNA(lambda) || any(lambda <= 0)) {
    stop("'lambda' must be positive (lambda > 0)")
  }

  .Call("_gkwdist_rekw",
    as.integer(n),
    as.numeric(alpha),
    as.numeric(beta),
    as.numeric(lambda),
    PACKAGE = "gkwdist"
  )
}


#' @title Negative Log-Likelihood for the Exponentiated Kumaraswamy (EKw) Distribution
#' @author Lopes, J. E.
#' @family log-likelihood functions
#' @concept exponentiated kumaraswamy
#' @keywords distribution optimize
#'
#' @description
#' Computes the negative log-likelihood function for the Exponentiated
#' Kumaraswamy (EKw) distribution with parameters \code{alpha} (\eqn{\alpha}),
#' \code{beta} (\eqn{\beta}), and \code{lambda} (\eqn{\lambda}), given a vector
#' of observations. This distribution is the special case of the Generalized
#' Kumaraswamy (GKw) distribution where \eqn{\gamma = 1} and \eqn{\delta = 0}.
#' This function is suitable for maximum likelihood estimation.
#'
#' @param par A numeric vector of length 3 containing the distribution parameters
#'   in the order: \code{alpha} (\eqn{\alpha > 0}), \code{beta} (\eqn{\beta > 0}),
#'   \code{lambda} (\eqn{\lambda > 0}).
#' @param data A numeric vector of observations. All values must be strictly
#'   between 0 and 1 (exclusive).
#'
#' @return Returns a single \code{double} value representing the negative
#'   log-likelihood (\eqn{-\ell(\theta|\mathbf{x})}). Returns \code{Inf}
#'   if any parameter values in \code{par} are invalid according to their
#'   constraints, or if any value in \code{data} is not in the interval (0, 1);
#'   in the latter case a warning naming \code{data} is also signaled, because
#'   an infinite objective offers an optimizer no gradient direction to follow
#'   and more often means a sample on the wrong scale than a genuine fit
#'   failure.
#'
#' @details
#' The Exponentiated Kumaraswamy (EKw) distribution is the GKw distribution
#' (\code{\link{dekw}}) with \eqn{\gamma=1} and \eqn{\delta=0}. Its probability
#' density function (PDF) is:
#' \deqn{
#' f(x | \theta) = \lambda \alpha \beta x^{\alpha-1} (1 - x^\alpha)^{\beta-1} \bigl[1 - (1 - x^\alpha)^\beta \bigr]^{\lambda - 1}
#' }
#' for \eqn{0 < x < 1} and \eqn{\theta = (\alpha, \beta, \lambda)}.
#' The log-likelihood function \eqn{\ell(\theta | \mathbf{x})} for a sample
#' \eqn{\mathbf{x} = (x_1, \dots, x_n)} is \eqn{\sum_{i=1}^n \ln f(x_i | \theta)}:
#' \deqn{
#' \ell(\theta | \mathbf{x}) = n[\ln(\lambda) + \ln(\alpha) + \ln(\beta)]
#' + \sum_{i=1}^{n} [(\alpha-1)\ln(x_i) + (\beta-1)\ln(v_i) + (\lambda-1)\ln(w_i)]
#' }
#' where:
#' \itemize{
#'   \item \eqn{v_i = 1 - x_i^{\alpha}}
#'   \item \eqn{w_i = 1 - v_i^{\beta} = 1 - (1-x_i^{\alpha})^{\beta}}
#' }
#' This function computes and returns the *negative* log-likelihood, \eqn{-\ell(\theta|\mathbf{x})},
#' suitable for minimization using optimization routines like \code{\link[stats]{optim}}.
#' Numerical stability is maintained similarly to \code{\link{llgkw}}.
#'
#' @references
#' Nadarajah, S., Cordeiro, G. M., & Ortega, E. M. (2012). The exponentiated
#' Kumaraswamy distribution. *Journal of the Franklin Institute*, *349*(3),
#'
#'
#' Cordeiro, G. M., & de Castro, M. (2011). A new family of generalized
#' distributions. *Journal of Statistical Computation and Simulation*,
#' *81*(7), 883-898.
#' \doi{10.1080/00949650903530745}
#'
#' Kumaraswamy, P. (1980). A generalized probability density function for
#' double-bounded random processes. *Journal of Hydrology*, *46*(1-2), 79-88.
#' \doi{10.1016/0022-1694(80)90036-0}
#'
#'
#' @seealso
#' \code{\link{llgkw}} (parent distribution negative log-likelihood),
#' \code{\link{dekw}}, \code{\link{pekw}}, \code{\link{qekw}}, \code{\link{rekw}},
#' \code{\link{grekw}} (gradient),
#' \code{\link{hsekw}} (Hessian),
#' \code{\link[stats]{optim}}
#'
#' @examples
#' set.seed(123)
#' x <- rekw(1000, alpha = 2, beta = 3, lambda = 1.2)
#' par <- c(alpha = 2, beta = 3, lambda = 1.2)
#'
#' ## llekw() is the negative log-likelihood, -sum(log f(x))
#' llekw(par, x)
#' -sum(dekw(x, alpha = 2, beta = 3, lambda = 1.2, log = TRUE))
#'
#' ## Maximum likelihood: minimize llekw(), with grekw() as its gradient
#' start <- gkwgetstartvalues(x, family = "ekw")
#' fit <- optim(start, llekw, grekw, data = x, method = "L-BFGS-B", lower = 1e-4)
#' fit$convergence  # 0: converged
#' fit$par
#' fit$value <= llekw(par, x)  # at least as good as the true values
#'
#' @export
llekw <- function(par, data) {
  if (!is.numeric(par) || length(par) != 3) {
    stop("'par' must be a numeric vector of length 3")
  }
  if (!is.numeric(data)) {
    stop("'data' must be numeric")
  }
  if (length(data) < 1) {
    stop("'data' must have at least one observation")
  }
  # ll*() returns +Inf for data outside the open support -- an infinite objective
  # with no gradient direction, which an optimiser can only sit on.
  if (any(data <= 0 | data >= 1, na.rm = TRUE)) {
    warning("'data' contains values outside (0, 1)")
  }

  .Call("_gkwdist_llekw",
    as.numeric(par),
    as.numeric(data),
    PACKAGE = "gkwdist"
  )
}


#' @title Gradient of the Negative Log-Likelihood for the EKw Distribution
#' @author Lopes, J. E.
#' @family gradient functions
#' @concept exponentiated kumaraswamy
#' @keywords distribution optimize
#'
#' @description
#' Computes the gradient vector (vector of first partial derivatives) of the
#' negative log-likelihood function for the Exponentiated Kumaraswamy (EKw)
#' distribution with parameters \code{alpha} (\eqn{\alpha}), \code{beta}
#' (\eqn{\beta}), and \code{lambda} (\eqn{\lambda}). This distribution is the
#' special case of the Generalized Kumaraswamy (GKw) distribution where
#' \eqn{\gamma = 1} and \eqn{\delta = 0}. The gradient is useful for optimization.
#'
#' @param par A numeric vector of length 3 containing the distribution parameters
#'   in the order: \code{alpha} (\eqn{\alpha > 0}), \code{beta} (\eqn{\beta > 0}),
#'   \code{lambda} (\eqn{\lambda > 0}).
#' @param data A numeric vector of observations. All values must be strictly
#'   between 0 and 1 (exclusive).
#'
#' @return Returns a numeric vector of length 3 containing the partial derivatives
#'   of the negative log-likelihood function \eqn{-\ell(\theta | \mathbf{x})} with
#'   respect to each parameter: \eqn{(-\partial \ell/\partial \alpha, -\partial \ell/\partial \beta, -\partial \ell/\partial \lambda)}.
#'   Returns a vector of \code{NaN} if any parameter values are invalid according
#'   to their constraints, or if any value in \code{data} is not in the
#'   interval (0, 1).
#'
#' @details
#' The components of the gradient vector of the negative log-likelihood
#' (\eqn{-\nabla \ell(\theta | \mathbf{x})}) for the EKw (\eqn{\gamma=1, \delta=0})
#' model are:
#'
#' \deqn{
#' -\frac{\partial \ell}{\partial \alpha} = -\frac{n}{\alpha} - \sum_{i=1}^{n}\ln(x_i)
#' + \sum_{i=1}^{n}\left[x_i^{\alpha} \ln(x_i) \left(\frac{\beta-1}{v_i} -
#' \frac{(\lambda-1) \beta v_i^{\beta-1}}{w_i}\right)\right]
#' }
#' \deqn{
#' -\frac{\partial \ell}{\partial \beta} = -\frac{n}{\beta} - \sum_{i=1}^{n}\ln(v_i)
#' + \sum_{i=1}^{n}\left[\frac{(\lambda-1) v_i^{\beta} \ln(v_i)}{w_i}\right]
#' }
#' \deqn{
#' -\frac{\partial \ell}{\partial \lambda} = -\frac{n}{\lambda} - \sum_{i=1}^{n}\ln(w_i)
#' }
#'
#' where:
#' \itemize{
#'   \item \eqn{v_i = 1 - x_i^{\alpha}}
#'   \item \eqn{w_i = 1 - v_i^{\beta} = 1 - (1-x_i^{\alpha})^{\beta}}
#' }
#' These formulas represent the derivatives of \eqn{-\ell(\theta)}, consistent with
#' minimizing the negative log-likelihood. They correspond to the relevant components
#' of the general GKw gradient (\code{\link{grgkw}}) evaluated at \eqn{\gamma=1, \delta=0}.
#'
#' @references
#' Nadarajah, S., Cordeiro, G. M., & Ortega, E. M. (2012). The exponentiated
#' Kumaraswamy distribution. *Journal of the Franklin Institute*, *349*(3),
#'
#'
#' Cordeiro, G. M., & de Castro, M. (2011). A new family of generalized
#' distributions. *Journal of Statistical Computation and Simulation*,
#' *81*(7), 883-898.
#' \doi{10.1080/00949650903530745}
#'
#'
#' Kumaraswamy, P. (1980). A generalized probability density function for
#' double-bounded random processes. *Journal of Hydrology*, *46*(1-2), 79-88.
#' \doi{10.1016/0022-1694(80)90036-0}
#'
#' (Note: Specific gradient formulas might be derived or sourced from additional references).
#'
#' @seealso
#' \code{\link{grgkw}} (parent distribution gradient),
#' \code{\link{llekw}} (negative log-likelihood for EKw),
#' \code{\link{hsekw}} (Hessian for EKw),
#' \code{\link{dekw}} (density for EKw),
#' \code{\link[stats]{optim}},
#' \code{\link[numDeriv]{grad}} (for numerical gradient comparison).
#'
#' @examples
#' set.seed(123)
#' x <- rekw(200, alpha = 2, beta = 3, lambda = 1.2)
#' par <- c(alpha = 2, beta = 3, lambda = 1.2)
#'
#' ## Gradient of the negative log-likelihood llekw(), not of the log-likelihood
#' g <- grekw(par, x)
#' g
#'
#' ## A small step against the gradient lowers llekw()
#' llekw(par - 1e-4 * g, x) < llekw(par, x)
#'
#' ## Agrees with a numerical derivative of llekw()
#' if (requireNamespace("numDeriv", quietly = TRUE))
#'   all.equal(g, numDeriv::grad(llekw, par, data = x), tolerance = 1e-6)
#'
#' @export
grekw <- function(par, data) {
  if (!is.numeric(par) || length(par) != 3) {
    stop("'par' must be a numeric vector of length 3")
  }
  if (!is.numeric(data)) {
    stop("'data' must be numeric")
  }
  if (length(data) < 1) {
    stop("'data' must have at least one observation")
  }

  .Call("_gkwdist_grekw",
    as.numeric(par),
    as.numeric(data),
    PACKAGE = "gkwdist"
  )
}

#' @title Hessian Matrix of the Negative Log-Likelihood for the EKw Distribution
#' @author Lopes, J. E.
#' @family Hessian functions
#' @concept exponentiated kumaraswamy
#' @keywords distribution optimize
#'
#' @description
#' Computes the analytic 3x3 Hessian matrix (matrix of second partial derivatives)
#' of the negative log-likelihood function for the Exponentiated Kumaraswamy (EKw)
#' distribution with parameters \code{alpha} (\eqn{\alpha}), \code{beta}
#' (\eqn{\beta}), and \code{lambda} (\eqn{\lambda}). This distribution is the
#' special case of the Generalized Kumaraswamy (GKw) distribution where
#' \eqn{\gamma = 1} and \eqn{\delta = 0}. The Hessian is useful for estimating
#' standard errors and in optimization algorithms.
#'
#' @param par A numeric vector of length 3 containing the distribution parameters
#'   in the order: \code{alpha} (\eqn{\alpha > 0}), \code{beta} (\eqn{\beta > 0}),
#'   \code{lambda} (\eqn{\lambda > 0}).
#' @param data A numeric vector of observations. All values must be strictly
#'   between 0 and 1 (exclusive).
#'
#' @return Returns a 3x3 numeric matrix representing the Hessian matrix of the
#'   negative log-likelihood function, \eqn{-\partial^2 \ell / (\partial \theta_i \partial \theta_j)},
#'   where \eqn{\theta = (\alpha, \beta, \lambda)}.
#'   Returns a 3x3 matrix populated with \code{NaN} if any parameter values are
#'   invalid according to their constraints, or if any value in \code{data} is
#'   not in the interval (0, 1).
#'
#' @details
#' This function calculates the analytic second partial derivatives of the
#' negative log-likelihood function based on the EKw log-likelihood
#' (\eqn{\gamma=1, \delta=0} case of GKw, see \code{\link{llekw}}):
#' \deqn{
#' \ell(\theta | \mathbf{x}) = n[\ln(\lambda) + \ln(\alpha) + \ln(\beta)]
#' + \sum_{i=1}^{n} [(\alpha-1)\ln(x_i) + (\beta-1)\ln(v_i) + (\lambda-1)\ln(w_i)]
#' }
#' where \eqn{\theta = (\alpha, \beta, \lambda)} and intermediate terms are:
#' \itemize{
#'   \item \eqn{v_i = 1 - x_i^{\alpha}}
#'   \item \eqn{w_i = 1 - v_i^{\beta} = 1 - (1-x_i^{\alpha})^{\beta}}
#' }
#' The Hessian matrix returned contains the elements \eqn{- \frac{\partial^2 \ell(\theta | \mathbf{x})}{\partial \theta_i \partial \theta_j}}
#' for \eqn{\theta_i, \theta_j \in \{\alpha, \beta, \lambda\}}.
#'
#' Key properties of the returned matrix:
#' \itemize{
#'   \item Dimensions: 3x3.
#'   \item Symmetry: The matrix is symmetric.
#'   \item Ordering: Rows and columns correspond to the parameters in the order
#'     \eqn{\alpha, \beta, \lambda}.
#'   \item Content: Analytic second derivatives of the *negative* log-likelihood.
#' }
#' This corresponds to the relevant 3x3 submatrix of the 5x5 GKw Hessian (\code{\link{hsgkw}})
#' evaluated at \eqn{\gamma=1, \delta=0}. The exact analytical formulas are implemented directly.
#'
#' @references
#' Nadarajah, S., Cordeiro, G. M., & Ortega, E. M. (2012). The exponentiated
#' Kumaraswamy distribution. *Journal of the Franklin Institute*, *349*(3),
#'
#'
#' Cordeiro, G. M., & de Castro, M. (2011). A new family of generalized
#' distributions. *Journal of Statistical Computation and Simulation*,
#' *81*(7), 883-898.
#' \doi{10.1080/00949650903530745}
#'
#'
#' Kumaraswamy, P. (1980). A generalized probability density function for
#' double-bounded random processes. *Journal of Hydrology*, *46*(1-2), 79-88.
#' \doi{10.1016/0022-1694(80)90036-0}
#'
#' (Note: Specific Hessian formulas might be derived or sourced from additional references).
#'
#' @seealso
#' \code{\link{hsgkw}} (parent distribution Hessian),
#' \code{\link{llekw}} (negative log-likelihood for EKw),
#' \code{\link{grekw}} (gradient for EKw),
#' \code{\link{dekw}} (density for EKw),
#' \code{\link[stats]{optim}},
#' \code{\link[numDeriv]{hessian}} (for numerical Hessian comparison).
#'
#' @examples
#' set.seed(123)
#' x <- rekw(1000, alpha = 2, beta = 3, lambda = 1.2)
#' par <- c(alpha = 2, beta = 3, lambda = 1.2)
#'
#' ## Hessian of the negative log-likelihood llekw()
#' H <- hsekw(par, x)
#' isSymmetric(H)
#'
#' ## Agrees with a numerical Hessian of llekw()
#' if (requireNamespace("numDeriv", quietly = TRUE))
#'   all.equal(H, numDeriv::hessian(llekw, par, data = x), tolerance = 1e-8)
#'
#' ## At the MLE it is the observed information; its inverse estimates the
#' ## covariance of the estimates
#' fit <- optim(par, llekw, grekw, data = x, method = "L-BFGS-B", lower = 1e-4)
#' sqrt(diag(solve(hsekw(fit$par, x))))  # standard errors
#'
#' @export
hsekw <- function(par, data) {
  if (!is.numeric(par) || length(par) != 3) {
    stop("'par' must be a numeric vector of length 3")
  }
  if (!is.numeric(data)) {
    stop("'data' must be numeric")
  }
  if (length(data) < 1) {
    stop("'data' must have at least one observation")
  }

  .Call("_gkwdist_hsekw",
    as.numeric(par),
    as.numeric(data),
    PACKAGE = "gkwdist"
  )
}
