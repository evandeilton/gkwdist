# ============================================================================#
# BETA-KUMARASWAMY (BKw) DISTRIBUTION
# ============================================================================#
#
# Wrapper functions for the Beta-Kumaraswamy distribution.
# C++ implementations are in src/bkw.cpp
#
# Functions:
#   - dbkw: Probability density function (PDF)
#   - pbkw: Cumulative distribution function (CDF)
#   - qbkw: Quantile function (inverse CDF)
#   - rbkw: Random number generation
#   - llbkw: Negative log-likelihood
#   - grbkw: Gradient of negative log-likelihood
#   - hsbkw: Hessian of negative log-likelihood
# ============================================================================#


# ----------------------------------------------------------------------------#
# 1. DENSITY FUNCTION (dbkw)
# ----------------------------------------------------------------------------#

#' @title Density of the Beta-Kumaraswamy (BKw) Distribution
#' @author Lopes, J. E.
#' @family density functions
#' @concept beta-kumaraswamy
#' @keywords distribution
#'
#' @description
#' Computes the probability density function (PDF) for the Beta-Kumaraswamy
#' (BKw) distribution with parameters \code{alpha} (\eqn{\alpha}), \code{beta}
#' (\eqn{\beta}), \code{gamma} (\eqn{\gamma}), and \code{delta} (\eqn{\delta}).
#' This distribution is defined on the interval (0, 1).
#'
#' @param x Vector of quantiles (values between 0 and 1).
#' @param alpha Shape parameter \code{alpha} > 0. Can be a scalar or a vector.
#'   Default: 1.0.
#' @param beta Shape parameter \code{beta} > 0. Can be a scalar or a vector.
#'   Default: 1.0.
#' @param gamma Shape parameter \code{gamma} > 0. Can be a scalar or a vector.
#'   Default: 1.0.
#' @param delta Shape parameter \code{delta} >= 0. Can be a scalar or a vector.
#'   Default: 0.0.
#' @param log Logical; if \code{TRUE}, the logarithm of the density is
#'   returned (\eqn{\log(f(x))}). Default: \code{FALSE}.
#'
#' @return A vector of density values (\eqn{f(x)}) or log-density values
#'   (\eqn{\log(f(x))}). The length of the result is determined by the recycling
#'   rule applied to the arguments (\code{x}, \code{alpha}, \code{beta},
#'   \code{gamma}, \code{delta}). Returns \code{0} (or \code{-Inf} if
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
#' The probability density function (PDF) of the Beta-Kumaraswamy (BKw)
#' distribution is given by:
#' \deqn{
#' f(x; \alpha, \beta, \gamma, \delta) = \frac{\alpha \beta}{B(\gamma, \delta+1)} x^{\alpha - 1} \bigl(1 - x^\alpha\bigr)^{\beta(\delta+1) - 1} \bigl[1 - \bigl(1 - x^\alpha\bigr)^\beta\bigr]^{\gamma - 1}
#' }
#' for \eqn{0 < x < 1}, where \eqn{B(a,b)} is the Beta function
#' (\code{\link[base]{beta}}).
#'
#' The BKw distribution is a special case of the five-parameter
#' Generalized Kumaraswamy (GKw) distribution (\code{\link{dgkw}}) obtained
#' by setting the parameter \eqn{\lambda = 1}.
#' Numerical evaluation is performed using algorithms similar to those for `dgkw`,
#' ensuring stability.
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
#' \code{\link{pbkw}}, \code{\link{qbkw}}, \code{\link{rbkw}} (other BKw functions),
#'
#' @examples
#' x <- c(0.1, 0.3, 0.5, 0.7, 0.9)
#' dbkw(x, alpha = 2, beta = 3, gamma = 1.5, delta = 0.5)
#' dbkw(x, alpha = 2, beta = 3, gamma = 1.5, delta = 0.5, log = TRUE)
#'
#' ## BKw is GKw with lambda = 1
#' all.equal(dbkw(x, 2, 3, 1.5, 0.5), dgkw(x, 2, 3, 1.5, 0.5, lambda = 1))
#'
#' ## The density integrates to one
#' integrate(dbkw, 0, 1, alpha = 2, beta = 3, gamma = 1.5, delta = 0.5,
#'     rel.tol = 1e-10)
#'
#' curve(dbkw(x, alpha = 2, beta = 3, gamma = 1.5, delta = 0.5), from = 0,
#'     to = 1, ylab = "density")
#'
#' @export
dbkw <- function(x, alpha = 1, beta = 1, gamma = 1, delta = 0, log = FALSE) {
  # Input validation
  if (!is.numeric(x)) stop("'x' must be numeric")
  if (!is.numeric(alpha) || anyNA(alpha) || any(alpha <= 0)) {
    stop("'alpha' must be positive")
  }
  if (!is.numeric(beta) || anyNA(beta) || any(beta <= 0)) {
    stop("'beta' must be positive")
  }
  if (!is.numeric(gamma) || anyNA(gamma) || any(gamma <= 0)) {
    stop("'gamma' must be positive")
  }
  if (!is.numeric(delta) || anyNA(delta) || any(delta < 0)) {
    stop("'delta' must be non-negative")
  }
  if (!is.logical(log) || length(log) != 1 || is.na(log)) {
    stop("'log' must be a single logical value")
  }

  # Call C++ implementation
  .shape_like(.Call("_gkwdist_dbkw",
    as.numeric(x),
    as.numeric(alpha),
    as.numeric(beta),
    as.numeric(gamma),
    as.numeric(delta),
    as.logical(log),
    PACKAGE = "gkwdist"
  ), x)
}


# ----------------------------------------------------------------------------#
# 2. DISTRIBUTION FUNCTION (pbkw)
# ----------------------------------------------------------------------------#

#' @title Cumulative Distribution Function (CDF) of the Beta-Kumaraswamy (BKw) Distribution
#' @author Lopes, J. E.
#' @family cumulative distribution functions
#' @concept beta-kumaraswamy
#' @keywords distribution
#'
#' @description
#' Computes the cumulative distribution function (CDF), \eqn{P(X \le q)}, for the
#' Beta-Kumaraswamy (BKw) distribution with parameters \code{alpha} (\eqn{\alpha}),
#' \code{beta} (\eqn{\beta}), \code{gamma} (\eqn{\gamma}), and \code{delta}
#' (\eqn{\delta}). This distribution is defined on the interval (0, 1) and is
#' a special case of the Generalized Kumaraswamy (GKw) distribution where
#' \eqn{\lambda = 1}.
#'
#' @param q Vector of quantiles (values generally between 0 and 1).
#' @param alpha Shape parameter \code{alpha} > 0. Can be a scalar or a vector.
#'   Default: 1.0.
#' @param beta Shape parameter \code{beta} > 0. Can be a scalar or a vector.
#'   Default: 1.0.
#' @param gamma Shape parameter \code{gamma} > 0. Can be a scalar or a vector.
#'   Default: 1.0.
#' @param delta Shape parameter \code{delta} >= 0. Can be a scalar or a vector.
#'   Default: 0.0.
#' @param lower.tail Logical; if \code{TRUE} (default), probabilities are
#'   \eqn{P(X \le q)}, otherwise, \eqn{P(X > q)}.
#' @param log.p Logical; if \code{TRUE}, probabilities \eqn{p} are given as
#'   \eqn{\log(p)}. Default: \code{FALSE}.
#'
#' @return A vector of probabilities, \eqn{F(q)}, or their logarithms/complements
#'   depending on \code{lower.tail} and \code{log.p}. The length of the result
#'   is determined by the recycling rule applied to the arguments (\code{q},
#'   \code{alpha}, \code{beta}, \code{gamma}, \code{delta}). When
#'   \code{lower.tail = TRUE}, returns \code{0} (or \code{-Inf} if
#'   \code{log.p = TRUE}) for \code{q <= 0} and \code{1} (or \code{0} if
#'   \code{log.p = TRUE}) for \code{q >= 1}. An out-of-bound or missing
#'   parameter is an error, not a return value: the wrapper stops with a
#'   message naming the parameter. An infinite parameter is not currently
#'   intercepted there and reaches the C++ layer, which treats it as invalid.
#'   Boundary return values are adjusted accordingly for \code{lower.tail = FALSE}.
#'
#' @details
#' The Beta-Kumaraswamy (BKw) distribution is a special case of the
#' five-parameter Generalized Kumaraswamy distribution (\code{\link{pgkw}})
#' obtained by setting the shape parameter \eqn{\lambda = 1}.
#'
#' The CDF of the GKw distribution is \eqn{F_{GKw}(q) = I_{y(q)}(\gamma, \delta+1)},
#' where \eqn{y(q) = [1-(1-q^{\alpha})^{\beta}]^{\lambda}} and \eqn{I_x(a,b)}
#' is the regularized incomplete beta function (\code{\link[stats]{pbeta}}).
#' Setting \eqn{\lambda=1} simplifies \eqn{y(q)} to \eqn{1 - (1 - q^\alpha)^\beta},
#' yielding the BKw CDF:
#' \deqn{
#' F(q; \alpha, \beta, \gamma, \delta) = I_{1 - (1 - q^\alpha)^\beta}(\gamma, \delta+1)
#' }
#' This is evaluated using the \code{\link[stats]{pbeta}} function.
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
#' \code{\link{dbkw}}, \code{\link{qbkw}}, \code{\link{rbkw}} (other BKw functions),
#' \code{\link[stats]{pbeta}}
#'
#' @examples
#' q <- c(0.2, 0.5, 0.8)
#' pbkw(q, alpha = 2, beta = 3, gamma = 1.5, delta = 0.5)
#' # P(X > q)
#' pbkw(q, alpha = 2, beta = 3, gamma = 1.5, delta = 0.5, lower.tail = FALSE)
#' pbkw(q, alpha = 2, beta = 3, gamma = 1.5, delta = 0.5, log.p = TRUE)
#'
#' ## pbkw() is the integral of dbkw()
#' Fq <- pbkw(0.5, alpha = 2, beta = 3, gamma = 1.5, delta = 0.5)
#' all.equal(Fq, integrate(dbkw, 0, 0.5, alpha = 2, beta = 3, gamma = 1.5,
#'     delta = 0.5, rel.tol = 1e-10)$value)
#'
#' @export
pbkw <- function(q, alpha = 1, beta = 1, gamma = 1, delta = 0, lower.tail = TRUE, log.p = FALSE) {
  # Input validation
  if (!is.numeric(q)) stop("'q' must be numeric")
  if (!is.numeric(alpha) || anyNA(alpha) || any(alpha <= 0)) {
    stop("'alpha' must be positive")
  }
  if (!is.numeric(beta) || anyNA(beta) || any(beta <= 0)) {
    stop("'beta' must be positive")
  }
  if (!is.numeric(gamma) || anyNA(gamma) || any(gamma <= 0)) {
    stop("'gamma' must be positive")
  }
  if (!is.numeric(delta) || anyNA(delta) || any(delta < 0)) {
    stop("'delta' must be non-negative")
  }
  if (!is.logical(lower.tail) || length(lower.tail) != 1 || is.na(lower.tail)) {
    stop("'lower.tail' must be a single logical value")
  }
  if (!is.logical(log.p) || length(log.p) != 1 || is.na(log.p)) {
    stop("'log.p' must be a single logical value")
  }

  # Call C++ implementation
  .shape_like(.Call("_gkwdist_pbkw",
    as.numeric(q),
    as.numeric(alpha),
    as.numeric(beta),
    as.numeric(gamma),
    as.numeric(delta),
    as.logical(lower.tail),
    as.logical(log.p),
    PACKAGE = "gkwdist"
  ), q)
}


# ----------------------------------------------------------------------------#
# 3. QUANTILE FUNCTION (qbkw)
# ----------------------------------------------------------------------------#


#' @title Quantile Function of the Beta-Kumaraswamy (BKw) Distribution
#' @author Lopes, J. E.
#' @family quantile functions
#' @concept beta-kumaraswamy
#' @keywords distribution
#'
#' @description
#' Computes the quantile function (inverse CDF) for the Beta-Kumaraswamy (BKw)
#' distribution with parameters \code{alpha} (\eqn{\alpha}), \code{beta}
#' (\eqn{\beta}), \code{gamma} (\eqn{\gamma}), and \code{delta} (\eqn{\delta}).
#' It finds the value \code{q} such that \eqn{P(X \le q) = p}. This distribution
#' is a special case of the Generalized Kumaraswamy (GKw) distribution where
#' the parameter \eqn{\lambda = 1}.
#'
#' @param p Vector of probabilities (values between 0 and 1).
#' @param alpha Shape parameter \code{alpha} > 0. Can be a scalar or a vector.
#'   Default: 1.0.
#' @param beta Shape parameter \code{beta} > 0. Can be a scalar or a vector.
#'   Default: 1.0.
#' @param gamma Shape parameter \code{gamma} > 0. Can be a scalar or a vector.
#'   Default: 1.0.
#' @param delta Shape parameter \code{delta} >= 0. Can be a scalar or a vector.
#'   Default: 0.0.
#' @param lower.tail Logical; if \code{TRUE} (default), probabilities are \eqn{p = P(X \le q)},
#'   otherwise, probabilities are \eqn{p = P(X > q)}.
#' @param log.p Logical; if \code{TRUE}, probabilities \code{p} are given as
#'   \eqn{\log(p)}. Default: \code{FALSE}.
#'
#' @return A vector of quantiles corresponding to the given probabilities \code{p}.
#'   The length of the result is determined by the recycling rule applied to
#'   the arguments (\code{p}, \code{alpha}, \code{beta}, \code{gamma}, \code{delta}).
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
#' for the BKw (\eqn{\lambda=1}) distribution is \eqn{F(q) = I_{y(q)}(\gamma, \delta+1)},
#' where \eqn{y(q) = 1 - (1 - q^\alpha)^\beta} and \eqn{I_z(a,b)} is the
#' regularized incomplete beta function (see \code{\link{pbkw}}).
#'
#' To find the quantile \eqn{q}, we first invert the outer Beta part: let
#' \eqn{y = I^{-1}_{p}(\gamma, \delta+1)}, where \eqn{I^{-1}_p(a,b)} is the
#' inverse of the regularized incomplete beta function, computed via
#' \code{\link[stats]{qbeta}}. Then, we invert the inner Kumaraswamy part:
#' \eqn{y = 1 - (1 - q^\alpha)^\beta}, which leads to \eqn{q = \{1 - (1-y)^{1/\beta}\}^{1/\alpha}}.
#' Substituting \eqn{y} gives the quantile function:
#' \deqn{
#' Q(p) = \left\{ 1 - \left[ 1 - I^{-1}_{p}(\gamma, \delta+1) \right]^{1/\beta} \right\}^{1/\alpha}
#' }
#' The function uses this formula, calculating \eqn{I^{-1}_{p}(\gamma, \delta+1)}
#' via \code{qbeta(p, gamma, delta + 1, ...)} while respecting the
#' \code{lower.tail} and \code{log.p} arguments.
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
#' \code{\link{dbkw}}, \code{\link{pbkw}}, \code{\link{rbkw}} (other BKw functions),
#' \code{\link[stats]{qbeta}}
#'
#' @examples
#' p <- c(0.1, 0.5, 0.9)
#' qbkw(p, alpha = 2, beta = 3, gamma = 1.5, delta = 0.5)
#' # upper-tail quantiles
#' qbkw(p, alpha = 2, beta = 3, gamma = 1.5, delta = 0.5, lower.tail = FALSE)
#'
#' ## qbkw() inverts pbkw()
#' all.equal(pbkw(qbkw(p, alpha = 2, beta = 3, gamma = 1.5, delta = 0.5),
#'     alpha = 2, beta = 3, gamma = 1.5, delta = 0.5), p)
#'
#' @export
qbkw <- function(p, alpha = 1, beta = 1, gamma = 1, delta = 0, lower.tail = TRUE, log.p = FALSE) {
  # Input validation
  if (!is.numeric(p)) stop("'p' must be numeric")
  if (!is.numeric(alpha) || anyNA(alpha) || any(alpha <= 0)) {
    stop("'alpha' must be positive")
  }
  if (!is.numeric(beta) || anyNA(beta) || any(beta <= 0)) {
    stop("'beta' must be positive")
  }
  if (!is.numeric(gamma) || anyNA(gamma) || any(gamma <= 0)) {
    stop("'gamma' must be positive")
  }
  if (!is.numeric(delta) || anyNA(delta) || any(delta < 0)) {
    stop("'delta' must be non-negative")
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
  .shape_like(.Call("_gkwdist_qbkw",
    as.numeric(p),
    as.numeric(alpha),
    as.numeric(beta),
    as.numeric(gamma),
    as.numeric(delta),
    as.logical(lower.tail),
    as.logical(log.p),
    PACKAGE = "gkwdist"
  ), p)
}


# ----------------------------------------------------------------------------#
# 4. RANDOM GENERATION (rbkw)
# ----------------------------------------------------------------------------#

#' @title Random Number Generation for the Beta-Kumaraswamy (BKw) Distribution
#' @author Lopes, J. E.
#' @family random generation functions
#' @concept beta-kumaraswamy
#' @keywords distribution
#'
#' @description
#' Generates random deviates from the Beta-Kumaraswamy (BKw) distribution
#' with parameters \code{alpha} (\eqn{\alpha}), \code{beta} (\eqn{\beta}),
#' \code{gamma} (\eqn{\gamma}), and \code{delta} (\eqn{\delta}). This distribution
#' is a special case of the Generalized Kumaraswamy (GKw) distribution where
#' the parameter \eqn{\lambda = 1}.
#'
#' @param n Number of observations. If \code{length(n) > 1}, the length is
#'   taken to be the number required. Must be a non-negative integer.
#' @param alpha Shape parameter \code{alpha} > 0. Can be a scalar or a vector.
#'   Default: 1.0.
#' @param beta Shape parameter \code{beta} > 0. Can be a scalar or a vector.
#'   Default: 1.0.
#' @param gamma Shape parameter \code{gamma} > 0. Can be a scalar or a vector.
#'   Default: 1.0.
#' @param delta Shape parameter \code{delta} >= 0. Can be a scalar or a vector.
#'   Default: 0.0.
#'
#' @return A vector of length \code{n} containing random deviates from the BKw
#'   distribution. The length of the result is determined by \code{n} and the
#'   recycling rule applied to the parameters (\code{alpha}, \code{beta},
#'   \code{gamma}, \code{delta}). An out-of-bound or missing parameter is an
#'   error, not a return value: the wrapper stops with a message naming the
#'   parameter. An infinite parameter is not currently intercepted there and
#'   reaches the C++ layer, which treats it as invalid.
#'
#' @details
#' The generation method uses the relationship between the GKw distribution and the
#' Beta distribution. The general procedure for GKw (\code{\link{rgkw}}) is:
#' If \eqn{W \sim \mathrm{Beta}(\gamma, \delta+1)}, then
#' \eqn{X = \{1 - [1 - W^{1/\lambda}]^{1/\beta}\}^{1/\alpha}} follows the
#' GKw(\eqn{\alpha, \beta, \gamma, \delta, \lambda}) distribution.
#'
#' For the BKw distribution, \eqn{\lambda=1}. Therefore, the algorithm simplifies to:
#' \enumerate{
#'   \item Generate \eqn{V \sim \mathrm{Beta}(\gamma, \delta+1)} using
#'         \code{\link[stats]{rbeta}}.
#'   \item Compute the BKw variate \eqn{X = \{1 - (1 - V)^{1/\beta}\}^{1/\alpha}}.
#' }
#' This procedure is implemented efficiently, handling parameter recycling as needed.
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
#'
#' Devroye, L. (1986). *Non-Uniform Random Variate Generation*. Springer-Verlag.
#' (General methods for random variate generation).
#'
#' @seealso
#' \code{\link{rgkw}} (parent distribution random generation),
#' \code{\link{dbkw}}, \code{\link{pbkw}}, \code{\link{qbkw}} (other BKw functions),
#' \code{\link[stats]{rbeta}}
#'
#' @examples
#' set.seed(123)
#' x <- rbkw(1000, alpha = 2, beta = 3, gamma = 1.5, delta = 0.5)
#' summary(x)
#'
#' ## The sample follows the distribution
#' hist(x, breaks = 30, freq = FALSE, main = "")
#' curve(dbkw(x, alpha = 2, beta = 3, gamma = 1.5, delta = 0.5), add = TRUE)
#' ks.test(x, pbkw, alpha = 2, beta = 3, gamma = 1.5, delta = 0.5)
#'
#' @export
rbkw <- function(n, alpha = 1, beta = 1, gamma = 1, delta = 0) {
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
  if (!is.numeric(gamma) || anyNA(gamma) || any(gamma <= 0)) {
    stop("'gamma' must be positive")
  }
  if (!is.numeric(delta) || anyNA(delta) || any(delta < 0)) {
    stop("'delta' must be non-negative")
  }

  # Call C++ implementation
  .Call("_gkwdist_rbkw",
    as.integer(n),
    as.numeric(alpha),
    as.numeric(beta),
    as.numeric(gamma),
    as.numeric(delta),
    PACKAGE = "gkwdist"
  )
}


# ============================================================================#
# MAXIMUM LIKELIHOOD ESTIMATION FUNCTIONS
# ============================================================================#

# ----------------------------------------------------------------------------#
# 5. NEGATIVE LOG-LIKELIHOOD (llbkw)
# ----------------------------------------------------------------------------#

#' @title Negative Log-Likelihood for the Beta-Kumaraswamy (BKw) Distribution
#' @author Lopes, J. E.
#' @family log-likelihood functions
#' @concept beta-kumaraswamy
#' @keywords distribution optimize
#'
#' @description
#' Computes the negative log-likelihood function for the Beta-Kumaraswamy (BKw)
#' distribution with parameters \code{alpha} (\eqn{\alpha}), \code{beta}
#' (\eqn{\beta}), \code{gamma} (\eqn{\gamma}), and \code{delta} (\eqn{\delta}),
#' given a vector of observations. This distribution is the special case of the
#' Generalized Kumaraswamy (GKw) distribution where \eqn{\lambda = 1}. This function
#' is typically used for maximum likelihood estimation via numerical optimization.
#'
#' @param par A numeric vector of length 4 containing the distribution parameters
#'   in the order: \code{alpha} (\eqn{\alpha > 0}), \code{beta} (\eqn{\beta > 0}),
#'   \code{gamma} (\eqn{\gamma > 0}), \code{delta} (\eqn{\delta \ge 0}).
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
#' The Beta-Kumaraswamy (BKw) distribution is the GKw distribution (\code{\link{dgkw}})
#' with \eqn{\lambda=1}. Its probability density function (PDF) is:
#' \deqn{
#' f(x | \theta) = \frac{\alpha \beta}{B(\gamma, \delta+1)} x^{\alpha - 1} \bigl(1 - x^\alpha\bigr)^{\beta(\delta+1) - 1} \bigl[1 - \bigl(1 - x^\alpha\bigr)^\beta\bigr]^{\gamma - 1}
#' }
#' for \eqn{0 < x < 1}, \eqn{\theta = (\alpha, \beta, \gamma, \delta)}, and \eqn{B(a,b)}
#' is the Beta function (\code{\link[base]{beta}}).
#' The log-likelihood function \eqn{\ell(\theta | \mathbf{x})} for a sample
#' \eqn{\mathbf{x} = (x_1, \dots, x_n)} is \eqn{\sum_{i=1}^n \ln f(x_i | \theta)}:
#' \deqn{
#' \ell(\theta | \mathbf{x}) = n[\ln(\alpha) + \ln(\beta) - \ln B(\gamma, \delta+1)]
#' + \sum_{i=1}^{n} [(\alpha-1)\ln(x_i) + (\beta(\delta+1)-1)\ln(v_i) + (\gamma-1)\ln(w_i)]
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
#' \code{\link{dbkw}}, \code{\link{pbkw}}, \code{\link{qbkw}}, \code{\link{rbkw}},
#' \code{\link{grbkw}} (gradient),
#' \code{\link{hsbkw}} (Hessian),
#' \code{\link[stats]{optim}}, \code{\link[base]{lbeta}}
#'
#' @examples
#' set.seed(123)
#' x <- rbkw(1000, alpha = 2, beta = 3, gamma = 1.5, delta = 0.5)
#' par <- c(alpha = 2, beta = 3, gamma = 1.5, delta = 0.5)
#'
#' ## llbkw() is the negative log-likelihood, -sum(log f(x))
#' llbkw(par, x)
#' -sum(dbkw(x, alpha = 2, beta = 3, gamma = 1.5, delta = 0.5, log = TRUE))
#'
#' ## Maximum likelihood: minimize llbkw(), with grbkw() as its gradient
#' start <- gkwgetstartvalues(x, family = "bkw")
#' fit <- optim(start, llbkw, grbkw, data = x, method = "L-BFGS-B", lower = 1e-4)
#' fit$convergence  # 0: converged
#' ## The parameters of this family are weakly identified: compare likelihoods
#' fit$value <= llbkw(par, x)  # at least as good as the true values
#'
#' @export
llbkw <- function(par, data) {
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
  .Call("_gkwdist_llbkw",
    as.numeric(par),
    as.numeric(data),
    PACKAGE = "gkwdist"
  )
}


# ----------------------------------------------------------------------------#
# 6. GRADIENT (grbkw)
# ----------------------------------------------------------------------------#

#' @title Gradient of the Negative Log-Likelihood for the BKw Distribution
#' @author Lopes, J. E.
#' @family gradient functions
#' @concept beta-kumaraswamy
#' @keywords distribution optimize
#'
#' @description
#' Computes the gradient vector (vector of first partial derivatives) of the
#' negative log-likelihood function for the Beta-Kumaraswamy (BKw) distribution
#' with parameters \code{alpha} (\eqn{\alpha}), \code{beta} (\eqn{\beta}),
#' \code{gamma} (\eqn{\gamma}), and \code{delta} (\eqn{\delta}). This distribution
#' is the special case of the Generalized Kumaraswamy (GKw) distribution where
#' \eqn{\lambda = 1}. The gradient is typically used in optimization algorithms
#' for maximum likelihood estimation.
#'
#' @param par A numeric vector of length 4 containing the distribution parameters
#'   in the order: \code{alpha} (\eqn{\alpha > 0}), \code{beta} (\eqn{\beta > 0}),
#'   \code{gamma} (\eqn{\gamma > 0}), \code{delta} (\eqn{\delta \ge 0}).
#' @param data A numeric vector of observations. All values must be strictly
#'   between 0 and 1 (exclusive).
#'
#' @return Returns a numeric vector of length 4 containing the partial derivatives
#'   of the negative log-likelihood function \eqn{-\ell(\theta | \mathbf{x})} with
#'   respect to each parameter:
#'   \eqn{(-\partial \ell/\partial \alpha, -\partial \ell/\partial \beta, -\partial \ell/\partial \gamma, -\partial \ell/\partial \delta)}.
#'   Returns a vector of \code{NaN} if any parameter values are invalid according
#'   to their constraints, or if any value in \code{data} is not in the
#'   interval (0, 1).
#'
#' @details
#' The components of the gradient vector of the negative log-likelihood
#' (\eqn{-\nabla \ell(\theta | \mathbf{x})}) for the BKw (\eqn{\lambda=1}) model are:
#'
#' \deqn{
#' -\frac{\partial \ell}{\partial \alpha} = -\frac{n}{\alpha} - \sum_{i=1}^{n}\ln(x_i)
#' + \sum_{i=1}^{n}\left[x_i^{\alpha} \ln(x_i) \left(\frac{\beta(\delta+1)-1}{v_i} -
#' \frac{(\gamma-1) \beta v_i^{\beta-1}}{w_i}\right)\right]
#' }
#' \deqn{
#' -\frac{\partial \ell}{\partial \beta} = -\frac{n}{\beta} - (\delta+1)\sum_{i=1}^{n}\ln(v_i)
#' + \sum_{i=1}^{n}\left[\frac{(\gamma-1) v_i^{\beta} \ln(v_i)}{w_i}\right]
#' }
#' \deqn{
#' -\frac{\partial \ell}{\partial \gamma} = n[\psi(\gamma) - \psi(\gamma+\delta+1)] -
#' \sum_{i=1}^{n}\ln(w_i)
#' }
#' \deqn{
#' -\frac{\partial \ell}{\partial \delta} = n[\psi(\delta+1) - \psi(\gamma+\delta+1)] -
#' \beta\sum_{i=1}^{n}\ln(v_i)
#' }
#'
#' where:
#' \itemize{
#'   \item \eqn{v_i = 1 - x_i^{\alpha}}
#'   \item \eqn{w_i = 1 - v_i^{\beta} = 1 - (1-x_i^{\alpha})^{\beta}}
#'   \item \eqn{\psi(\cdot)} is the digamma function (\code{\link[base]{digamma}}).
#' }
#' These formulas represent the derivatives of \eqn{-\ell(\theta)}, consistent with
#' minimizing the negative log-likelihood. They correspond to the general GKw
#' gradient (\code{\link{grgkw}}) components for \eqn{\alpha, \beta, \gamma, \delta}
#' evaluated at \eqn{\lambda=1}. Note that the component for \eqn{\lambda} is omitted.
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
#'
#' (Note: Specific gradient formulas might be derived or sourced from additional references).
#'
#' @seealso
#' \code{\link{grgkw}} (parent distribution gradient),
#' \code{\link{llbkw}} (negative log-likelihood for BKw),
#' \code{\link{hsbkw}} (Hessian for BKw),
#' \code{\link{dbkw}} (density for BKw),
#' \code{\link[stats]{optim}},
#' \code{\link[numDeriv]{grad}} (for numerical gradient comparison),
#' \code{\link[base]{digamma}}.
#'
#' @examples
#' set.seed(123)
#' x <- rbkw(200, alpha = 2, beta = 3, gamma = 1.5, delta = 0.5)
#' par <- c(alpha = 2, beta = 3, gamma = 1.5, delta = 0.5)
#'
#' ## Gradient of the negative log-likelihood llbkw(), not of the log-likelihood
#' g <- grbkw(par, x)
#' g
#'
#' ## A small step against the gradient lowers llbkw()
#' llbkw(par - 1e-4 * g, x) < llbkw(par, x)
#'
#' ## Agrees with a numerical derivative of llbkw()
#' if (requireNamespace("numDeriv", quietly = TRUE))
#'   all.equal(g, numDeriv::grad(llbkw, par, data = x), tolerance = 1e-6)
#'
#' @export
grbkw <- function(par, data) {
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
  .Call("_gkwdist_grbkw",
    as.numeric(par),
    as.numeric(data),
    PACKAGE = "gkwdist"
  )
}


# ----------------------------------------------------------------------------#
# 7. HESSIAN (hsbkw)
# ----------------------------------------------------------------------------#

#' @title Hessian Matrix of the Negative Log-Likelihood for the BKw Distribution
#' @author Lopes, J. E.
#' @family Hessian functions
#' @concept beta-kumaraswamy
#' @keywords distribution optimize
#'
#' @description
#' Computes the analytic 4x4 Hessian matrix (matrix of second partial derivatives)
#' of the negative log-likelihood function for the Beta-Kumaraswamy (BKw)
#' distribution with parameters \code{alpha} (\eqn{\alpha}), \code{beta}
#' (\eqn{\beta}), \code{gamma} (\eqn{\gamma}), and \code{delta} (\eqn{\delta}).
#' This distribution is the special case of the Generalized Kumaraswamy (GKw)
#' distribution where \eqn{\lambda = 1}. The Hessian is useful for estimating
#' standard errors and in optimization algorithms.
#'
#' @param par A numeric vector of length 4 containing the distribution parameters
#'   in the order: \code{alpha} (\eqn{\alpha > 0}), \code{beta} (\eqn{\beta > 0}),
#'   \code{gamma} (\eqn{\gamma > 0}), \code{delta} (\eqn{\delta \ge 0}).
#' @param data A numeric vector of observations. All values must be strictly
#'   between 0 and 1 (exclusive).
#'
#' @return Returns a 4x4 numeric matrix representing the Hessian matrix of the
#'   negative log-likelihood function, \eqn{-\partial^2 \ell / (\partial \theta_i \partial \theta_j)},
#'   where \eqn{\theta = (\alpha, \beta, \gamma, \delta)}.
#'   Returns a 4x4 matrix populated with \code{NaN} if any parameter values are
#'   invalid according to their constraints, or if any value in \code{data} is
#'   not in the interval (0, 1).
#'
#' @details
#' This function calculates the analytic second partial derivatives of the
#' negative log-likelihood function based on the BKw log-likelihood
#' (\eqn{\lambda=1} case of GKw, see \code{\link{llbkw}}):
#' \deqn{
#' \ell(\theta | \mathbf{x}) = n[\ln(\alpha) + \ln(\beta) - \ln B(\gamma, \delta+1)]
#' + \sum_{i=1}^{n} [(\alpha-1)\ln(x_i) + (\beta(\delta+1)-1)\ln(v_i) + (\gamma-1)\ln(w_i)]
#' }
#' where \eqn{\theta = (\alpha, \beta, \gamma, \delta)}, \eqn{B(a,b)}
#' is the Beta function (\code{\link[base]{beta}}), and intermediate terms are:
#' \itemize{
#'   \item \eqn{v_i = 1 - x_i^{\alpha}}
#'   \item \eqn{w_i = 1 - v_i^{\beta} = 1 - (1-x_i^{\alpha})^{\beta}}
#' }
#' The Hessian matrix returned contains the elements \eqn{- \frac{\partial^2 \ell(\theta | \mathbf{x})}{\partial \theta_i \partial \theta_j}}
#' for \eqn{\theta_i, \theta_j \in \{\alpha, \beta, \gamma, \delta\}}.
#'
#' Key properties of the returned matrix:
#' \itemize{
#'   \item Dimensions: 4x4.
#'   \item Symmetry: The matrix is symmetric.
#'   \item Ordering: Rows and columns correspond to the parameters in the order
#'     \eqn{\alpha, \beta, \gamma, \delta}.
#'   \item Content: Analytic second derivatives of the *negative* log-likelihood.
#' }
#' This corresponds to the relevant 4x4 submatrix of the 5x5 GKw Hessian (\code{\link{hsgkw}})
#' evaluated at \eqn{\lambda=1}. The exact analytical formulas are implemented directly.
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
#' (Note: Specific Hessian formulas might be derived or sourced from additional references).
#'
#' @seealso
#' \code{\link{hsgkw}} (parent distribution Hessian),
#' \code{\link{llbkw}} (negative log-likelihood for BKw),
#' \code{\link{grbkw}} (gradient for BKw),
#' \code{\link{dbkw}} (density for BKw),
#' \code{\link[stats]{optim}},
#' \code{\link[numDeriv]{hessian}} (for numerical Hessian comparison).
#'
#' @examples
#' set.seed(123)
#' x <- rbkw(1000, alpha = 2, beta = 3, gamma = 1.5, delta = 0.5)
#' par <- c(alpha = 2, beta = 3, gamma = 1.5, delta = 0.5)
#'
#' ## Hessian of the negative log-likelihood llbkw()
#' H <- hsbkw(par, x)
#' isSymmetric(H)
#'
#' ## Agrees with a numerical Hessian of llbkw()
#' if (requireNamespace("numDeriv", quietly = TRUE))
#'   all.equal(H, numDeriv::hessian(llbkw, par, data = x), tolerance = 1e-8)
#'
#' @export
hsbkw <- function(par, data) {
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
  .Call("_gkwdist_hsbkw",
    as.numeric(par),
    as.numeric(data),
    PACKAGE = "gkwdist"
  )
}
