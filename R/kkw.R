# ============================================================================#
# KUMARASWAMY-KUMARASWAMY (KKw) DISTRIBUTION
# ============================================================================#
#
# Wrapper functions for the Kumaraswamy-Kumaraswamy distribution.
# C++ implementations are in src/kkw.cpp
#
# Functions:
#   - dkkw: Probability density function (PDF)
#   - pkkw: Cumulative distribution function (CDF)
#   - qkkw: Quantile function (inverse CDF)
#   - rkkw: Random number generation
#   - llkkw: Negative log-likelihood
#   - grkkw: Gradient of negative log-likelihood
#   - hskkw: Hessian of negative log-likelihood
# ============================================================================#


# ----------------------------------------------------------------------------#
# 1. DENSITY FUNCTION (dkkw)
# ----------------------------------------------------------------------------#

#' @title Density of the Kumaraswamy-Kumaraswamy (KKw) Distribution
#' @author Lopes, J. E.
#' @family density functions
#' @concept kumaraswamy-kumaraswamy
#' @keywords distribution
#'
#' @description
#' Computes the probability density function (PDF) for the Kumaraswamy-Kumaraswamy
#' (KKw) distribution with parameters \code{alpha} (\eqn{\alpha}), \code{beta}
#' (\eqn{\beta}), \code{delta} (\eqn{\delta}), and \code{lambda} (\eqn{\lambda}).
#' This distribution is defined on the interval (0, 1).
#'
#' @param x Vector of quantiles (values between 0 and 1).
#' @param alpha Shape parameter \code{alpha} > 0. Can be a scalar or a vector.
#'   Default: 1.0.
#' @param beta Shape parameter \code{beta} > 0. Can be a scalar or a vector.
#'   Default: 1.0.
#' @param delta Shape parameter \code{delta} >= 0. Can be a scalar or a vector.
#'   Default: 0.0.
#' @param lambda Shape parameter \code{lambda} > 0. Can be a scalar or a vector.
#'   Default: 1.0.
#' @param log Logical; if \code{TRUE}, the logarithm of the density is
#'   returned (\eqn{\log(f(x))}). Default: \code{FALSE}.
#'
#' @return A vector of density values (\eqn{f(x)}) or log-density values
#'   (\eqn{\log(f(x))}). The length of the result is determined by the recycling
#'   rule applied to the arguments (\code{x}, \code{alpha}, \code{beta},
#'   \code{delta}, \code{lambda}). Returns \code{0} (or \code{-Inf} if
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
#' The Kumaraswamy-Kumaraswamy (KKw) distribution is a special case of the
#' five-parameter Generalized Kumaraswamy distribution (\code{\link{dgkw}})
#' obtained by setting the parameter \eqn{\gamma = 1}.
#'
#' The probability density function is given by:
#' \deqn{
#' f(x; \alpha, \beta, \delta, \lambda) = (\delta + 1) \lambda \alpha \beta x^{\alpha - 1} (1 - x^\alpha)^{\beta - 1} \bigl[1 - (1 - x^\alpha)^\beta\bigr]^{\lambda - 1} \bigl\{1 - \bigl[1 - (1 - x^\alpha)^\beta\bigr]^\lambda\bigr\}^{\delta}
#' }
#' for \eqn{0 < x < 1}. Note that \eqn{1/(\delta+1)} corresponds to the Beta function
#' term \eqn{B(1, \delta+1)} when \eqn{\gamma=1}.
#'
#' Numerical evaluation follows similar stability considerations as \code{\link{dgkw}}.
#'
#' @references
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
#' \code{\link{pkkw}}, \code{\link{qkkw}}, \code{\link{rkkw}},
#' \code{\link[stats]{dbeta}}
#'
#' @examples
#' x <- c(0.1, 0.3, 0.5, 0.7, 0.9)
#' dkkw(x, alpha = 2, beta = 3, delta = 0.5, lambda = 1.2)
#' dkkw(x, alpha = 2, beta = 3, delta = 0.5, lambda = 1.2, log = TRUE)
#'
#' ## KKw is GKw with gamma = 1
#' all.equal(dkkw(x, 2, 3, 0.5, 1.2), dgkw(x, 2, 3, gamma = 1, delta = 0.5,
#'     lambda = 1.2))
#'
#' ## The density integrates to one
#' integrate(dkkw, 0, 1, alpha = 2, beta = 3, delta = 0.5, lambda = 1.2,
#'     rel.tol = 1e-10)
#'
#' curve(dkkw(x, alpha = 2, beta = 3, delta = 0.5, lambda = 1.2), from = 0,
#'     to = 1, ylab = "density")
#'
#' @export
dkkw <- function(x, alpha = 1, beta = 1, delta = 0, lambda = 1, log = FALSE) {
  # Input validation
  if (!is.numeric(x)) stop("'x' must be numeric")
  if (!is.numeric(alpha) || anyNA(alpha) || any(alpha <= 0)) {
    stop("'alpha' must be positive")
  }
  if (!is.numeric(beta) || anyNA(beta) || any(beta <= 0)) {
    stop("'beta' must be positive")
  }
  if (!is.numeric(delta) || anyNA(delta) || any(delta < 0)) {
    stop("'delta' must be non-negative")
  }
  if (!is.numeric(lambda) || anyNA(lambda) || any(lambda <= 0)) {
    stop("'lambda' must be positive")
  }
  if (!is.logical(log) || length(log) != 1 || is.na(log)) {
    stop("'log' must be a single logical value")
  }

  # Call C++ implementation
  .shape_like(.Call("_gkwdist_dkkw",
    as.numeric(x),
    as.numeric(alpha),
    as.numeric(beta),
    as.numeric(delta),
    as.numeric(lambda),
    as.logical(log),
    PACKAGE = "gkwdist"
  ), x)
}


# ----------------------------------------------------------------------------#
# 2. DISTRIBUTION FUNCTION (pkkw)
# ----------------------------------------------------------------------------#

#' @title Cumulative Distribution Function (CDF) of the KKw Distribution
#' @author Lopes, J. E.
#' @family cumulative distribution functions
#' @concept kumaraswamy-kumaraswamy
#' @keywords distribution
#'
#' @description
#' Computes the cumulative distribution function (CDF), \eqn{P(X \le q)}, for the
#' Kumaraswamy-Kumaraswamy (KKw) distribution with parameters \code{alpha}
#' (\eqn{\alpha}), \code{beta} (\eqn{\beta}), \code{delta} (\eqn{\delta}),
#' and \code{lambda} (\eqn{\lambda}). This distribution is defined on the
#' interval (0, 1).
#'
#' @param q Vector of quantiles (values generally between 0 and 1).
#' @param alpha Shape parameter \code{alpha} > 0. Can be a scalar or a vector.
#'   Default: 1.0.
#' @param beta Shape parameter \code{beta} > 0. Can be a scalar or a vector.
#'   Default: 1.0.
#' @param delta Shape parameter \code{delta} >= 0. Can be a scalar or a vector.
#'   Default: 0.0.
#' @param lambda Shape parameter \code{lambda} > 0. Can be a scalar or a vector.
#'   Default: 1.0.
#' @param lower.tail Logical; if \code{TRUE} (default), probabilities are
#'   \eqn{P(X \le q)}, otherwise, \eqn{P(X > q)}.
#' @param log.p Logical; if \code{TRUE}, probabilities \eqn{p} are given as
#'   \eqn{\log(p)}. Default: \code{FALSE}.
#'
#' @return A vector of probabilities, \eqn{F(q)}, or their logarithms/complements
#'   depending on \code{lower.tail} and \code{log.p}. The length of the result
#'   is determined by the recycling rule applied to the arguments (\code{q},
#'   \code{alpha}, \code{beta}, \code{delta}, \code{lambda}). When
#'   \code{lower.tail = TRUE}, returns \code{0} (or \code{-Inf} if
#'   \code{log.p = TRUE}) for \code{q <= 0} and \code{1} (or \code{0} if
#'   \code{log.p = TRUE}) for \code{q >= 1}. An out-of-bound or missing
#'   parameter is an error, not a return value: the wrapper stops with a
#'   message naming the parameter. An infinite parameter is not currently
#'   intercepted there and reaches the C++ layer, which treats it as invalid.
#'   Boundary return values are adjusted accordingly for \code{lower.tail = FALSE}.
#'
#' @details
#' The Kumaraswamy-Kumaraswamy (KKw) distribution is a special case of the
#' five-parameter Generalized Kumaraswamy distribution (\code{\link{pgkw}})
#' obtained by setting the shape parameter \eqn{\gamma = 1}.
#'
#' The CDF of the GKw distribution is \eqn{F_{GKw}(q) = I_{y(q)}(\gamma, \delta+1)},
#' where \eqn{y(q) = [1-(1-q^{\alpha})^{\beta}]^{\lambda}} and \eqn{I_x(a,b)}
#' is the regularized incomplete beta function (\code{\link[stats]{pbeta}}).
#' Setting \eqn{\gamma=1} utilizes the property \eqn{I_x(1, b) = 1 - (1-x)^b},
#' yielding the KKw CDF:
#' \deqn{
#' F(q; \alpha, \beta, \delta, \lambda) = 1 - \bigl\{1 - \bigl[1 - (1 - q^\alpha)^\beta\bigr]^\lambda\bigr\}^{\delta + 1}
#' }
#' for \eqn{0 < q < 1}.
#'
#' The implementation uses this closed-form expression for efficiency and handles
#' \code{lower.tail} and \code{log.p} arguments appropriately.
#'
#' @references
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
#' \code{\link{pgkw}} (parent distribution CDF),
#' \code{\link{dkkw}}, \code{\link{qkkw}}, \code{\link{rkkw}},
#' \code{\link[stats]{pbeta}}
#'
#' @examples
#' q <- c(0.2, 0.5, 0.8)
#' pkkw(q, alpha = 2, beta = 3, delta = 0.5, lambda = 1.2)
#' # P(X > q)
#' pkkw(q, alpha = 2, beta = 3, delta = 0.5, lambda = 1.2, lower.tail = FALSE)
#' pkkw(q, alpha = 2, beta = 3, delta = 0.5, lambda = 1.2, log.p = TRUE)
#'
#' ## pkkw() is the integral of dkkw()
#' Fq <- pkkw(0.5, alpha = 2, beta = 3, delta = 0.5, lambda = 1.2)
#' all.equal(Fq, integrate(dkkw, 0, 0.5, alpha = 2, beta = 3, delta = 0.5,
#'     lambda = 1.2, rel.tol = 1e-10)$value)
#'
#' @export
pkkw <- function(q, alpha = 1, beta = 1, delta = 0, lambda = 1, lower.tail = TRUE, log.p = FALSE) {
  # Input validation
  if (!is.numeric(q)) stop("'q' must be numeric")
  if (!is.numeric(alpha) || anyNA(alpha) || any(alpha <= 0)) {
    stop("'alpha' must be positive")
  }
  if (!is.numeric(beta) || anyNA(beta) || any(beta <= 0)) {
    stop("'beta' must be positive")
  }
  if (!is.numeric(delta) || anyNA(delta) || any(delta < 0)) {
    stop("'delta' must be non-negative")
  }
  if (!is.numeric(lambda) || anyNA(lambda) || any(lambda <= 0)) {
    stop("'lambda' must be positive")
  }
  if (!is.logical(lower.tail) || length(lower.tail) != 1 || is.na(lower.tail)) {
    stop("'lower.tail' must be a single logical value")
  }
  if (!is.logical(log.p) || length(log.p) != 1 || is.na(log.p)) {
    stop("'log.p' must be a single logical value")
  }

  # Call C++ implementation
  .shape_like(.Call("_gkwdist_pkkw",
    as.numeric(q),
    as.numeric(alpha),
    as.numeric(beta),
    as.numeric(delta),
    as.numeric(lambda),
    as.logical(lower.tail),
    as.logical(log.p),
    PACKAGE = "gkwdist"
  ), q)
}


# ----------------------------------------------------------------------------#
# 3. QUANTILE FUNCTION (qkkw)
# ----------------------------------------------------------------------------#

#' @title Quantile Function of the Kumaraswamy-Kumaraswamy (KKw) Distribution
#' @author Lopes, J. E.
#' @family quantile functions
#' @concept kumaraswamy-kumaraswamy
#' @keywords distribution
#'
#' @description
#' Computes the quantile function (inverse CDF) for the Kumaraswamy-Kumaraswamy
#' (KKw) distribution with parameters \code{alpha} (\eqn{\alpha}), \code{beta}
#' (\eqn{\beta}), \code{delta} (\eqn{\delta}), and \code{lambda} (\eqn{\lambda}).
#' It finds the value \code{q} such that \eqn{P(X \le q) = p}. This distribution
#' is a special case of the Generalized Kumaraswamy (GKw) distribution where
#' the parameter \eqn{\gamma = 1}.
#'
#' @param p Vector of probabilities (values between 0 and 1).
#' @param alpha Shape parameter \code{alpha} > 0. Can be a scalar or a vector.
#'   Default: 1.0.
#' @param beta Shape parameter \code{beta} > 0. Can be a scalar or a vector.
#'   Default: 1.0.
#' @param delta Shape parameter \code{delta} >= 0. Can be a scalar or a vector.
#'   Default: 0.0.
#' @param lambda Shape parameter \code{lambda} > 0. Can be a scalar or a vector.
#'   Default: 1.0.
#' @param lower.tail Logical; if \code{TRUE} (default), probabilities are \eqn{p = P(X \le q)},
#'   otherwise, probabilities are \eqn{p = P(X > q)}.
#' @param log.p Logical; if \code{TRUE}, probabilities \code{p} are given as
#'   \eqn{\log(p)}. Default: \code{FALSE}.
#'
#' @return A vector of quantiles corresponding to the given probabilities \code{p}.
#'   The length of the result is determined by the recycling rule applied to
#'   the arguments (\code{p}, \code{alpha}, \code{beta}, \code{delta},
#'   \code{lambda}). Returns:
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
#' for the KKw (\eqn{\gamma=1}) distribution is (see \code{\link{pkkw}}):
#' \deqn{
#' F(q) = 1 - \bigl\{1 - \bigl[1 - (1 - q^\alpha)^\beta\bigr]^\lambda\bigr\}^{\delta + 1}
#' }
#' Inverting this equation for \eqn{q} yields the quantile function:
#' \deqn{
#' Q(p) = \left[ 1 - \left\{ 1 - \left[ 1 - (1 - p)^{1/(\delta+1)} \right]^{1/\lambda} \right\}^{1/\beta} \right]^{1/\alpha}
#' }
#' The function uses this closed-form expression and correctly handles the
#' \code{lower.tail} and \code{log.p} arguments by transforming \code{p}
#' appropriately before applying the formula.
#'
#' @references
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
#' \code{\link{qgkw}} (parent distribution quantile function),
#' \code{\link{dkkw}}, \code{\link{pkkw}}, \code{\link{rkkw}},
#' \code{\link[stats]{qbeta}}
#'
#' @examples
#' p <- c(0.1, 0.5, 0.9)
#' qkkw(p, alpha = 2, beta = 3, delta = 0.5, lambda = 1.2)
#' # upper-tail quantiles
#' qkkw(p, alpha = 2, beta = 3, delta = 0.5, lambda = 1.2, lower.tail = FALSE)
#'
#' ## qkkw() inverts pkkw()
#' all.equal(pkkw(qkkw(p, alpha = 2, beta = 3, delta = 0.5, lambda = 1.2),
#'     alpha = 2, beta = 3, delta = 0.5, lambda = 1.2), p)
#'
#' @export
qkkw <- function(p, alpha = 1, beta = 1, delta = 0, lambda = 1, lower.tail = TRUE, log.p = FALSE) {
  # Input validation
  if (!is.numeric(p)) stop("'p' must be numeric")
  if (!is.numeric(alpha) || anyNA(alpha) || any(alpha <= 0)) {
    stop("'alpha' must be positive")
  }
  if (!is.numeric(beta) || anyNA(beta) || any(beta <= 0)) {
    stop("'beta' must be positive")
  }
  if (!is.numeric(delta) || anyNA(delta) || any(delta < 0)) {
    stop("'delta' must be non-negative")
  }
  if (!is.numeric(lambda) || anyNA(lambda) || any(lambda <= 0)) {
    stop("'lambda' must be positive")
  }
  if (!is.logical(lower.tail) || length(lower.tail) != 1 || is.na(lower.tail)) {
    stop("'lower.tail' must be a single logical value")
  }
  if (!is.logical(log.p) || length(log.p) != 1 || is.na(log.p)) {
    stop("'log.p' must be a single logical value")
  }

  # Additional validation for probabilities
  if (!log.p && any(p < 0 | p > 1, na.rm = TRUE)) {
    warning("'p' values outside [0, 1] will produce NaN")
  }

  # Call C++ implementation
  .shape_like(.Call("_gkwdist_qkkw",
    as.numeric(p),
    as.numeric(alpha),
    as.numeric(beta),
    as.numeric(delta),
    as.numeric(lambda),
    as.logical(lower.tail),
    as.logical(log.p),
    PACKAGE = "gkwdist"
  ), p)
}


# ----------------------------------------------------------------------------#
# 4. RANDOM GENERATION (rkkw)
# ----------------------------------------------------------------------------#

#' @title Random Number Generation for the KKw Distribution
#' @author Lopes, J. E.
#' @family random generation functions
#' @concept kumaraswamy-kumaraswamy
#' @keywords distribution
#'
#' @description
#' Generates random deviates from the Kumaraswamy-Kumaraswamy (KKw)
#' distribution with parameters \code{alpha} (\eqn{\alpha}), \code{beta}
#' (\eqn{\beta}), \code{delta} (\eqn{\delta}), and \code{lambda} (\eqn{\lambda}).
#' This distribution is a special case of the Generalized Kumaraswamy (GKw)
#' distribution where the parameter \eqn{\gamma = 1}.
#'
#' @param n Number of observations. If \code{length(n) > 1}, the length is
#'   taken to be the number required. Must be a non-negative integer.
#' @param alpha Shape parameter \code{alpha} > 0. Can be a scalar or a vector.
#'   Default: 1.0.
#' @param beta Shape parameter \code{beta} > 0. Can be a scalar or a vector.
#'   Default: 1.0.
#' @param delta Shape parameter \code{delta} >= 0. Can be a scalar or a vector.
#'   Default: 0.0.
#' @param lambda Shape parameter \code{lambda} > 0. Can be a scalar or a vector.
#'   Default: 1.0.
#'
#' @return A vector of length \code{n} containing random deviates from the KKw
#'   distribution. The length of the result is determined by \code{n} and the
#'   recycling rule applied to the parameters (\code{alpha}, \code{beta},
#'   \code{delta}, \code{lambda}). An out-of-bound or missing parameter is an
#'   error, not a return value: the wrapper stops with a message naming the
#'   parameter. An infinite parameter is not currently intercepted there and
#'   reaches the C++ layer, which treats it as invalid.
#'
#' @details
#' The generation method uses the inverse transform method based on the quantile
#' function (\code{\link{qkkw}}). The KKw quantile function is:
#' \deqn{
#' Q(p) = \left[ 1 - \left\{ 1 - \left[ 1 - (1 - p)^{1/(\delta+1)} \right]^{1/\lambda} \right\}^{1/\beta} \right]^{1/\alpha}
#' }
#' Random deviates are generated by evaluating \eqn{Q(p)} where \eqn{p} is a
#' random variable following the standard Uniform distribution on (0, 1)
#' (\code{\link[stats]{runif}}).
#'
#' This is equivalent to the general method for the GKw distribution
#' (\code{\link{rgkw}}) specialized for \eqn{\gamma=1}. The GKw method generates
#' \eqn{W \sim \mathrm{Beta}(\gamma, \delta+1)} and then applies transformations.
#' When \eqn{\gamma=1}, \eqn{W \sim \mathrm{Beta}(1, \delta+1)}, which can be
#' generated via \eqn{W = 1 - V^{1/(\delta+1)}} where \eqn{V \sim \mathrm{Unif}(0,1)}.
#' Substituting this \eqn{W} into the GKw transformation yields the same result
#' as evaluating \eqn{Q(1-V)} above (noting \eqn{p = 1-V} is also Uniform).
#'
#' @references
#' Cordeiro, G. M., & de Castro, M. (2011). A new family of generalized
#' distributions. *Journal of Statistical Computation and Simulation*,
#' *81*(7), 883-898.
#' \doi{10.1080/00949650903530745}
#'
#' Kumaraswamy, P. (1980). A generalized probability density function for
#' double-bounded random processes. *Journal of Hydrology*, *46*(1-2), 79-88.
#' \doi{10.1016/0022-1694(80)90036-0}
#'
#' Devroye, L. (1986). *Non-Uniform Random Variate Generation*. Springer-Verlag.
#' (General methods for random variate generation).
#'
#' @seealso
#' \code{\link{rgkw}} (parent distribution random generation),
#' \code{\link{dkkw}}, \code{\link{pkkw}}, \code{\link{qkkw}},
#' \code{\link[stats]{runif}}, \code{\link[stats]{rbeta}}
#'
#' @examples
#' set.seed(123)
#' x <- rkkw(1000, alpha = 2, beta = 3, delta = 0.5, lambda = 1.2)
#' summary(x)
#'
#' ## The sample follows the distribution
#' hist(x, breaks = 30, freq = FALSE, main = "")
#' curve(dkkw(x, alpha = 2, beta = 3, delta = 0.5, lambda = 1.2), add = TRUE)
#' ks.test(x, pkkw, alpha = 2, beta = 3, delta = 0.5, lambda = 1.2)
#'
#' @export
rkkw <- function(n, alpha = 1, beta = 1, delta = 0, lambda = 1) {
  # Input validation
  if (length(n) > 1) n <- length(n)
  # n = 0 is legal and yields numeric(0), as stats::rbeta(0, 2, 3) does.
  # Negative and missing n stay errors, which is also what base R does.
  if (!is.numeric(n) || length(n) != 1 || is.na(n) || n < 0) {
    stop("'n' must be a single non-negative integer")
  }
  n <- as.integer(n)

  if (!is.numeric(alpha) || anyNA(alpha) || any(alpha <= 0)) {
    stop("'alpha' must be positive")
  }
  if (!is.numeric(beta) || anyNA(beta) || any(beta <= 0)) {
    stop("'beta' must be positive")
  }
  if (!is.numeric(delta) || anyNA(delta) || any(delta < 0)) {
    stop("'delta' must be non-negative")
  }
  if (!is.numeric(lambda) || anyNA(lambda) || any(lambda <= 0)) {
    stop("'lambda' must be positive")
  }

  # Call C++ implementation
  .Call("_gkwdist_rkkw",
    as.integer(n),
    as.numeric(alpha),
    as.numeric(beta),
    as.numeric(delta),
    as.numeric(lambda),
    PACKAGE = "gkwdist"
  )
}


# ============================================================================#
# MAXIMUM LIKELIHOOD ESTIMATION FUNCTIONS
# ============================================================================#

# ----------------------------------------------------------------------------#
# 5. NEGATIVE LOG-LIKELIHOOD (llkkw)
# ----------------------------------------------------------------------------#

#' @title Negative Log-Likelihood for the KKw Distribution
#' @author Lopes, J. E.
#' @family log-likelihood functions
#' @concept kumaraswamy-kumaraswamy
#' @keywords distribution optimize
#'
#' @description
#' Computes the negative log-likelihood function for the Kumaraswamy-Kumaraswamy
#' (KKw) distribution with parameters \code{alpha} (\eqn{\alpha}), \code{beta}
#' (\eqn{\beta}), \code{delta} (\eqn{\delta}), and \code{lambda} (\eqn{\lambda}),
#' given a vector of observations. This distribution is a special case of the
#' Generalized Kumaraswamy (GKw) distribution where \eqn{\gamma = 1}.
#'
#' @param par A numeric vector of length 4 containing the distribution parameters
#'   in the order: \code{alpha} (\eqn{\alpha > 0}), \code{beta} (\eqn{\beta > 0}),
#'   \code{delta} (\eqn{\delta \ge 0}), \code{lambda} (\eqn{\lambda > 0}).
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
#' The KKw distribution is the GKw distribution (\code{\link{dgkw}}) with \eqn{\gamma=1}.
#' Its probability density function (PDF) is:
#' \deqn{
#' f(x | \theta) = (\delta + 1) \lambda \alpha \beta x^{\alpha - 1} (1 - x^\alpha)^{\beta - 1} \bigl[1 - (1 - x^\alpha)^\beta\bigr]^{\lambda - 1} \bigl\{1 - \bigl[1 - (1 - x^\alpha)^\beta\bigr]^\lambda\bigr\}^{\delta}
#' }
#' for \eqn{0 < x < 1} and \eqn{\theta = (\alpha, \beta, \delta, \lambda)}.
#' The log-likelihood function \eqn{\ell(\theta | \mathbf{x})} for a sample
#' \eqn{\mathbf{x} = (x_1, \dots, x_n)} is \eqn{\sum_{i=1}^n \ln f(x_i | \theta)}:
#' \deqn{
#' \ell(\theta | \mathbf{x}) = n[\ln(\delta+1) + \ln(\lambda) + \ln(\alpha) + \ln(\beta)]
#' + \sum_{i=1}^{n} [(\alpha-1)\ln(x_i) + (\beta-1)\ln(v_i) + (\lambda-1)\ln(w_i) + \delta\ln(z_i)]
#' }
#' where:
#' \itemize{
#'   \item \eqn{v_i = 1 - x_i^{\alpha}}
#'   \item \eqn{w_i = 1 - v_i^{\beta} = 1 - (1-x_i^{\alpha})^{\beta}}
#'   \item \eqn{z_i = 1 - w_i^{\lambda} = 1 - [1-(1-x_i^{\alpha})^{\beta}]^{\lambda}}
#' }
#' This function computes and returns the *negative* log-likelihood, \eqn{-\ell(\theta|\mathbf{x})},
#' suitable for minimization using optimization routines like \code{\link[stats]{optim}}.
#' Numerical stability is maintained similarly to \code{\link{llgkw}}.
#'
#' @references
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
#' \code{\link{llgkw}} (parent distribution negative log-likelihood),
#' \code{\link{dkkw}}, \code{\link{pkkw}}, \code{\link{qkkw}}, \code{\link{rkkw}},
#' \code{\link{grkkw}} (gradient),
#' \code{\link{hskkw}} (Hessian),
#' \code{\link[stats]{optim}}
#'
#' @examples
#' set.seed(123)
#' x <- rkkw(1000, alpha = 2, beta = 3, delta = 0.5, lambda = 1.2)
#' par <- c(alpha = 2, beta = 3, delta = 0.5, lambda = 1.2)
#'
#' ## llkkw() is the negative log-likelihood, -sum(log f(x))
#' llkkw(par, x)
#' -sum(dkkw(x, alpha = 2, beta = 3, delta = 0.5, lambda = 1.2, log = TRUE))
#'
#' ## Maximum likelihood: minimize llkkw(), with grkkw() as its gradient
#' start <- gkwgetstartvalues(x, family = "kkw")
#' fit <- optim(start, llkkw, grkkw, data = x, method = "L-BFGS-B", lower = 1e-4)
#' fit$convergence  # 0: converged
#' ## The parameters of this family are weakly identified: compare likelihoods
#' fit$value <= llkkw(par, x)  # at least as good as the true values
#'
#' @export
llkkw <- function(par, data) {
  # Input validation
  if (!is.numeric(par) || length(par) != 4) {
    stop("'par' must be a numeric vector of length 4")
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

  # Call C++ implementation
  .Call("_gkwdist_llkkw",
    as.numeric(par),
    as.numeric(data),
    PACKAGE = "gkwdist"
  )
}


# ----------------------------------------------------------------------------#
# 6. GRADIENT (grkkw)
# ----------------------------------------------------------------------------#

#' @title Gradient of the Negative Log-Likelihood for the KKw Distribution
#' @author Lopes, J. E.
#' @family gradient functions
#' @concept kumaraswamy-kumaraswamy
#' @keywords distribution optimize
#'
#' @description
#' Computes the gradient vector (vector of first partial derivatives) of the
#' negative log-likelihood function for the Kumaraswamy-Kumaraswamy (KKw)
#' distribution with parameters \code{alpha} (\eqn{\alpha}), \code{beta}
#' (\eqn{\beta}), \code{delta} (\eqn{\delta}), and \code{lambda} (\eqn{\lambda}).
#' This distribution is the special case of the Generalized Kumaraswamy (GKw)
#' distribution where \eqn{\gamma = 1}. The gradient is typically used in
#' optimization algorithms for maximum likelihood estimation.
#'
#' @param par A numeric vector of length 4 containing the distribution parameters
#'   in the order: \code{alpha} (\eqn{\alpha > 0}), \code{beta} (\eqn{\beta > 0}),
#'   \code{delta} (\eqn{\delta \ge 0}), \code{lambda} (\eqn{\lambda > 0}).
#' @param data A numeric vector of observations. All values must be strictly
#'   between 0 and 1 (exclusive).
#'
#' @return Returns a numeric vector of length 4 containing the partial derivatives
#'   of the negative log-likelihood function \eqn{-\ell(\theta | \mathbf{x})} with
#'   respect to each parameter:
#'   \eqn{(-\partial \ell/\partial \alpha, -\partial \ell/\partial \beta, -\partial \ell/\partial \delta, -\partial \ell/\partial \lambda)}.
#'   Returns a vector of \code{NaN} if any parameter values are invalid according
#'   to their constraints, or if any value in \code{data} is not in the
#'   interval (0, 1).
#'
#' @details
#' The components of the gradient vector of the negative log-likelihood
#' (\eqn{-\nabla \ell(\theta | \mathbf{x})}) for the KKw (\eqn{\gamma=1}) model are:
#'
#' \deqn{
#' -\frac{\partial \ell}{\partial \alpha} = -\frac{n}{\alpha} - \sum_{i=1}^{n}\ln(x_i)
#' + (\beta-1)\sum_{i=1}^{n}\frac{x_i^{\alpha}\ln(x_i)}{v_i}
#' - (\lambda-1)\sum_{i=1}^{n}\frac{\beta v_i^{\beta-1} x_i^{\alpha}\ln(x_i)}{w_i}
#' + \delta\sum_{i=1}^{n}\frac{\lambda w_i^{\lambda-1} \beta v_i^{\beta-1} x_i^{\alpha}\ln(x_i)}{z_i}
#' }
#' \deqn{
#' -\frac{\partial \ell}{\partial \beta} = -\frac{n}{\beta} - \sum_{i=1}^{n}\ln(v_i)
#' + (\lambda-1)\sum_{i=1}^{n}\frac{v_i^{\beta}\ln(v_i)}{w_i}
#' - \delta\sum_{i=1}^{n}\frac{\lambda w_i^{\lambda-1} v_i^{\beta}\ln(v_i)}{z_i}
#' }
#' \deqn{
#' -\frac{\partial \ell}{\partial \delta} = -\frac{n}{\delta+1} - \sum_{i=1}^{n}\ln(z_i)
#' }
#' \deqn{
#' -\frac{\partial \ell}{\partial \lambda} = -\frac{n}{\lambda} - \sum_{i=1}^{n}\ln(w_i)
#' + \delta\sum_{i=1}^{n}\frac{w_i^{\lambda}\ln(w_i)}{z_i}
#' }
#'
#' where:
#' \itemize{
#'   \item \eqn{v_i = 1 - x_i^{\alpha}}
#'   \item \eqn{w_i = 1 - v_i^{\beta} = 1 - (1-x_i^{\alpha})^{\beta}}
#'   \item \eqn{z_i = 1 - w_i^{\lambda} = 1 - [1-(1-x_i^{\alpha})^{\beta}]^{\lambda}}
#' }
#' These formulas represent the derivatives of \eqn{-\ell(\theta)}, consistent with
#' minimizing the negative log-likelihood. They correspond to the general GKw
#' gradient (\code{\link{grgkw}}) components for \eqn{\alpha, \beta, \delta, \lambda}
#' evaluated at \eqn{\gamma=1}. Note that the component for \eqn{\gamma} is omitted.
#' Numerical stability is maintained through careful implementation.
#'
#' @references
#' Cordeiro, G. M., & de Castro, M. (2011). A new family of generalized
#' distributions. *Journal of Statistical Computation and Simulation*,
#' *81*(7), 883-898.
#' \doi{10.1080/00949650903530745}
#'
#' Kumaraswamy, P. (1980). A generalized probability density function for
#' double-bounded random processes. *Journal of Hydrology*, *46*(1-2), 79-88.
#' \doi{10.1016/0022-1694(80)90036-0}
#' @seealso
#' \code{\link{grgkw}} (parent distribution gradient),
#' \code{\link{llkkw}} (negative log-likelihood for KKw),
#' \code{\link{hskkw}} (Hessian for KKw),
#' \code{\link{dkkw}} (density for KKw),
#' \code{\link[stats]{optim}},
#' \code{\link[numDeriv]{grad}} (for numerical gradient comparison).
#'
#' @examples
#' set.seed(123)
#' x <- rkkw(200, alpha = 2, beta = 3, delta = 0.5, lambda = 1.2)
#' par <- c(alpha = 2, beta = 3, delta = 0.5, lambda = 1.2)
#'
#' ## Gradient of the negative log-likelihood llkkw(), not of the log-likelihood
#' g <- grkkw(par, x)
#' g
#'
#' ## A small step against the gradient lowers llkkw()
#' llkkw(par - 1e-4 * g, x) < llkkw(par, x)
#'
#' ## Agrees with a numerical derivative of llkkw()
#' if (requireNamespace("numDeriv", quietly = TRUE))
#'   all.equal(g, numDeriv::grad(llkkw, par, data = x), tolerance = 1e-6)
#'
#' @export
grkkw <- function(par, data) {
  # Input validation
  if (!is.numeric(par) || length(par) != 4) {
    stop("'par' must be a numeric vector of length 4")
  }
  if (!is.numeric(data)) {
    stop("'data' must be numeric")
  }
  if (length(data) < 1) {
    stop("'data' must have at least one observation")
  }

  # Call C++ implementation
  .Call("_gkwdist_grkkw",
    as.numeric(par),
    as.numeric(data),
    PACKAGE = "gkwdist"
  )
}


# ----------------------------------------------------------------------------#
# 7. HESSIAN (hskkw)
# ----------------------------------------------------------------------------#

#' @title Hessian Matrix of the Negative Log-Likelihood for the KKw Distribution
#' @author Lopes, J. E.
#' @family Hessian functions
#' @concept kumaraswamy-kumaraswamy
#' @keywords distribution optimize
#'
#' @description
#' Computes the analytic 4x4 Hessian matrix (matrix of second partial derivatives)
#' of the negative log-likelihood function for the Kumaraswamy-Kumaraswamy (KKw)
#' distribution with parameters \code{alpha} (\eqn{\alpha}), \code{beta}
#' (\eqn{\beta}), \code{delta} (\eqn{\delta}), and \code{lambda} (\eqn{\lambda}).
#' This distribution is the special case of the Generalized Kumaraswamy (GKw)
#' distribution where \eqn{\gamma = 1}. The Hessian is useful for estimating
#' standard errors and in optimization algorithms.
#'
#' @param par A numeric vector of length 4 containing the distribution parameters
#'   in the order: \code{alpha} (\eqn{\alpha > 0}), \code{beta} (\eqn{\beta > 0}),
#'   \code{delta} (\eqn{\delta \ge 0}), \code{lambda} (\eqn{\lambda > 0}).
#' @param data A numeric vector of observations. All values must be strictly
#'   between 0 and 1 (exclusive).
#'
#' @return Returns a 4x4 numeric matrix representing the Hessian matrix of the
#'   negative log-likelihood function, \eqn{-\partial^2 \ell / (\partial \theta_i \partial \theta_j)},
#'   where \eqn{\theta = (\alpha, \beta, \delta, \lambda)}.
#'   Returns a 4x4 matrix populated with \code{NaN} if any parameter values are
#'   invalid according to their constraints, or if any value in \code{data} is
#'   not in the interval (0, 1).
#'
#' @details
#' This function calculates the analytic second partial derivatives of the
#' negative log-likelihood function based on the KKw log-likelihood
#' (\eqn{\gamma=1} case of GKw, see \code{\link{llkkw}}):
#' \deqn{
#' \ell(\theta | \mathbf{x}) = n[\ln(\delta+1) + \ln(\lambda) + \ln(\alpha) + \ln(\beta)]
#' + \sum_{i=1}^{n} [(\alpha-1)\ln(x_i) + (\beta-1)\ln(v_i) + (\lambda-1)\ln(w_i) + \delta\ln(z_i)]
#' }
#' where \eqn{\theta = (\alpha, \beta, \delta, \lambda)} and intermediate terms are:
#' \itemize{
#'   \item \eqn{v_i = 1 - x_i^{\alpha}}
#'   \item \eqn{w_i = 1 - v_i^{\beta} = 1 - (1-x_i^{\alpha})^{\beta}}
#'   \item \eqn{z_i = 1 - w_i^{\lambda} = 1 - [1-(1-x_i^{\alpha})^{\beta}]^{\lambda}}
#' }
#' The Hessian matrix returned contains the elements \eqn{- \frac{\partial^2 \ell(\theta | \mathbf{x})}{\partial \theta_i \partial \theta_j}}
#' for \eqn{\theta_i, \theta_j \in \{\alpha, \beta, \delta, \lambda\}}.
#'
#' Key properties of the returned matrix:
#' \itemize{
#'   \item Dimensions: 4x4.
#'   \item Symmetry: The matrix is symmetric.
#'   \item Ordering: Rows and columns correspond to the parameters in the order
#'     \eqn{\alpha, \beta, \delta, \lambda}.
#'   \item Content: Analytic second derivatives of the *negative* log-likelihood.
#' }
#' This corresponds to the relevant submatrix of the 5x5 GKw Hessian (\code{\link{hsgkw}})
#' evaluated at \eqn{\gamma=1}. The exact analytical formulas are implemented directly.
#'
#' @references
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
#' \code{\link{hsgkw}} (parent distribution Hessian),
#' \code{\link{llkkw}} (negative log-likelihood for KKw),
#' \code{\link{grkkw}} (gradient for KKw),
#' \code{\link{dkkw}} (density for KKw),
#' \code{\link[stats]{optim}},
#' \code{\link[numDeriv]{hessian}} (for numerical Hessian comparison).
#'
#' @examples
#' set.seed(123)
#' x <- rkkw(1000, alpha = 2, beta = 3, delta = 0.5, lambda = 1.2)
#' par <- c(alpha = 2, beta = 3, delta = 0.5, lambda = 1.2)
#'
#' ## Hessian of the negative log-likelihood llkkw()
#' H <- hskkw(par, x)
#' isSymmetric(H)
#'
#' ## Agrees with a numerical Hessian of llkkw()
#' if (requireNamespace("numDeriv", quietly = TRUE))
#'   all.equal(H, numDeriv::hessian(llkkw, par, data = x), tolerance = 1e-8)
#'
#' @export
hskkw <- function(par, data) {
  # Input validation
  if (!is.numeric(par) || length(par) != 4) {
    stop("'par' must be a numeric vector of length 4")
  }
  if (!is.numeric(data)) {
    stop("'data' must be numeric")
  }
  if (length(data) < 1) {
    stop("'data' must have at least one observation")
  }

  # Call C++ implementation
  .Call("_gkwdist_hskkw",
    as.numeric(par),
    as.numeric(data),
    PACKAGE = "gkwdist"
  )
}
