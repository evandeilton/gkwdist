# ============================================================================#
# GENERALIZED KUMARASWAMY (GKw) DISTRIBUTION
# ============================================================================#
#
# Wrapper functions for the Generalized Kumaraswamy distribution.
# C++ implementations are in src/gkw.cpp
#
# Functions:
#   - dgkw: Probability density function (PDF)
#   - pgkw: Cumulative distribution function (CDF)
#   - qgkw: Quantile function (inverse CDF)
#   - rgkw: Random number generation
#   - llgkw: Negative log-likelihood
#   - grgkw: Gradient of negative log-likelihood
#   - hsgkw: Hessian of negative log-likelihood
# ============================================================================#


# ----------------------------------------------------------------------------#
# 1. DENSITY FUNCTION (dgkw)
# ----------------------------------------------------------------------------#

#' @title Density of the Generalized Kumaraswamy Distribution
#' @author Lopes, J. E.
#' @family density functions
#' @concept generalized kumaraswamy
#' @keywords distribution
#'
#' @description
#' Computes the probability density function (PDF) for the five-parameter
#' Generalized Kumaraswamy (GKw) distribution, defined on the interval (0, 1).
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
#' @param lambda Shape parameter \code{lambda} > 0. Can be a scalar or a vector.
#'   Default: 1.0.
#' @param log Logical; if \code{TRUE}, the logarithm of the density is
#'   returned. Default: \code{FALSE}.
#'
#' @return A vector of density values (\eqn{f(x)}) or log-density values
#'   (\eqn{\log(f(x))}). The length of the result is determined by the recycling
#'   rule applied to the arguments (\code{x}, \code{alpha}, \code{beta},
#'   \code{gamma}, \code{delta}, \code{lambda}). Returns \code{0} (or \code{-Inf}
#'   if \code{log = TRUE}) for \code{x} strictly outside the interval \[0, 1\]. At
#'   the closed boundaries \code{x = 0} and \code{x = 1} the limiting density is
#'   returned rather than \code{0}, following the convention of base R's density
#'   functions (compare \code{\link[stats]{dbeta}}); depending on the parameters
#'   that limit is \code{0}, a finite positive value, or \code{Inf}.
#'   An out-of-bound or missing parameter is an error, not a return value: the
#'   wrapper stops with a message naming the parameter. An infinite parameter
#'   is not currently intercepted there and reaches the C++ layer, which
#'   treats it as invalid.
#'
#' @details
#' The probability density function of the Generalized Kumaraswamy (GKw)
#' distribution with parameters \code{alpha} (\eqn{\alpha}), \code{beta}
#' (\eqn{\beta}), \code{gamma} (\eqn{\gamma}), \code{delta} (\eqn{\delta}), and
#' \code{lambda} (\eqn{\lambda}) is given by:
#' \deqn{
#' f(x; \alpha, \beta, \gamma, \delta, \lambda) =
#'   \frac{\lambda \alpha \beta x^{\alpha-1}(1-x^{\alpha})^{\beta-1}}
#'        {B(\gamma, \delta+1)}
#'   [1-(1-x^{\alpha})^{\beta}]^{\gamma\lambda-1}
#'   [1-[1-(1-x^{\alpha})^{\beta}]^{\lambda}]^{\delta}
#' }
#' for \eqn{x \in (0,1)}, where \eqn{B(a, b)} is the Beta function
#' \code{\link[base]{beta}}.
#'
#' This distribution was proposed by Carrasco, Ferrari & Cordeiro (2010) and includes
#' several other distributions as special cases:
#' \itemize{
#'   \item Kumaraswamy (Kw): \code{gamma = 1}, \code{delta = 0}, \code{lambda = 1}
#'   \item Exponentiated Kumaraswamy (EKw): \code{gamma = 1}, \code{delta = 0}
#'   \item Beta-Kumaraswamy (BKw): \code{lambda = 1}
#'   \item Generalized Beta type 1 (GB1 - implies McDonald): \code{alpha = 1}, \code{beta = 1}
#'   \item Beta distribution: \code{alpha = 1}, \code{beta = 1}, \code{lambda = 1}
#' }
#' The function includes checks for valid parameters and input values \code{x}.
#' It uses numerical stabilization for \code{x} close to 0 or 1.
#'
#' @references
#' Carrasco, J. M. F., Ferrari, S. L. P., & Cordeiro, G. M. (2010). A new
#' generalized Kumaraswamy distribution. *arXiv preprint arXiv:1004.0911*.
#' \doi{10.48550/arXiv.1004.0911}
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
#' \code{\link{pgkw}}, \code{\link{qgkw}}, \code{\link{rgkw}},
#' \code{\link[stats]{dbeta}}, \code{\link[stats]{integrate}}
#'
#' @examples
#' x <- c(0.1, 0.3, 0.5, 0.7, 0.9)
#' dgkw(x, alpha = 2, beta = 3, gamma = 1.5, delta = 0.5, lambda = 1.2)
#' dgkw(x, alpha = 2, beta = 3, gamma = 1.5, delta = 0.5, lambda = 1.2, log = TRUE)
#'
#' ## Kumaraswamy is GKw with gamma = 1, delta = 0, lambda = 1 (the defaults)
#' all.equal(dgkw(x, alpha = 2, beta = 3), dkw(x, alpha = 2, beta = 3))
#'
#' ## Beta(gamma, delta + 1) is GKw with alpha = beta = lambda = 1
#' all.equal(dgkw(x, gamma = 2, delta = 3), stats::dbeta(x, 2, 4))
#'
#' ## The density integrates to one
#' integrate(dgkw, 0, 1, alpha = 2, beta = 3, gamma = 1.5, delta = 0.5,
#'     lambda = 1.2, rel.tol = 1e-10)
#'
#' curve(dgkw(x, alpha = 2, beta = 3, gamma = 1.5, delta = 0.5, lambda = 1.2),
#'     from = 0, to = 1, ylab = "density")
#'
#' @export
dgkw <- function(x, alpha = 1, beta = 1, gamma = 1, delta = 0, lambda = 1, log = FALSE) {
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
  if (!is.numeric(lambda) || anyNA(lambda) || any(lambda <= 0)) {
    stop("'lambda' must be positive")
  }
  if (!is.logical(log) || length(log) != 1 || is.na(log)) {
    stop("'log' must be a single logical value")
  }

  # Call C++ implementation
  .shape_like(.Call("_gkwdist_dgkw",
    as.numeric(x),
    as.numeric(alpha),
    as.numeric(beta),
    as.numeric(gamma),
    as.numeric(delta),
    as.numeric(lambda),
    as.logical(log),
    PACKAGE = "gkwdist"
  ), x)
}


# ----------------------------------------------------------------------------#
# 2. DISTRIBUTION FUNCTION (pgkw)
# ----------------------------------------------------------------------------#

#' @title Cumulative Distribution Function (CDF) of the Generalized Kumaraswamy Distribution
#' @author Lopes, J. E.
#' @family cumulative distribution functions
#' @concept generalized kumaraswamy
#' @keywords distribution
#'
#' @description
#' Computes the cumulative distribution function (CDF) for the five-parameter
#' Generalized Kumaraswamy (GKw) distribution, defined on the interval (0, 1).
#' Calculates \eqn{P(X \le q)}.
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
#'   \code{alpha}, \code{beta}, \code{gamma}, \code{delta}, \code{lambda}). When
#'   \code{lower.tail = TRUE}, returns \code{0} (or \code{-Inf} if
#'   \code{log.p = TRUE}) for \code{q <= 0} and \code{1} (or \code{0} if
#'   \code{log.p = TRUE}) for \code{q >= 1}. An out-of-bound or missing
#'   parameter is an error, not a return value: the wrapper stops with a
#'   message naming the parameter. An infinite parameter is not currently
#'   intercepted there and reaches the C++ layer, which treats it as invalid.
#'   Boundary return values are adjusted accordingly for \code{lower.tail = FALSE}.
#'
#' @details
#' The cumulative distribution function (CDF) of the Generalized Kumaraswamy (GKw)
#' distribution with parameters \code{alpha} (\eqn{\alpha}), \code{beta}
#' (\eqn{\beta}), \code{gamma} (\eqn{\gamma}), \code{delta} (\eqn{\delta}), and
#' \code{lambda} (\eqn{\lambda}) is given by:
#' \deqn{
#' F(q; \alpha, \beta, \gamma, \delta, \lambda) =
#'   I_{x(q)}(\gamma, \delta+1)
#' }
#' where \eqn{x(q) = [1-(1-q^{\alpha})^{\beta}]^{\lambda}} and \eqn{I_x(a, b)}
#' is the regularized incomplete beta function, defined as:
#' \deqn{
#' I_x(a, b) = \frac{B_x(a, b)}{B(a, b)} = \frac{\int_0^x t^{a-1}(1-t)^{b-1} dt}{\int_0^1 t^{a-1}(1-t)^{b-1} dt}
#' }
#' This corresponds to the \code{\link[stats]{pbeta}} function in R, such that
#' \eqn{F(q; \alpha, \beta, \gamma, \delta, \lambda) = \code{pbeta}(x(q), \code{shape1} = \gamma, \code{shape2} = \delta+1)}.
#'
#' The GKw distribution includes several special cases, such as the Kumaraswamy,
#' Beta, and Exponentiated Kumaraswamy distributions (see \code{\link{dgkw}} for details).
#' The function utilizes numerical algorithms for computing the regularized
#' incomplete beta function accurately, especially near the boundaries.
#'
#' @references
#' Carrasco, J. M. F., Ferrari, S. L. P., & Cordeiro, G. M. (2010). A new
#' generalized Kumaraswamy distribution. *arXiv preprint arXiv:1004.0911*.
#' \doi{10.48550/arXiv.1004.0911}
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
#' \code{\link{dgkw}}, \code{\link{qgkw}}, \code{\link{rgkw}},
#' \code{\link[stats]{pbeta}}
#'
#' @examples
#' q <- c(0.2, 0.5, 0.8)
#' pgkw(q, alpha = 2, beta = 3, gamma = 1.5, delta = 0.5, lambda = 1.2)
#' pgkw(q, alpha = 2, beta = 3, gamma = 1.5, delta = 0.5, lambda = 1.2,
#'     lower.tail = FALSE)  # P(X > q)
#' pgkw(q, alpha = 2, beta = 3, gamma = 1.5, delta = 0.5, lambda = 1.2,
#'     log.p = TRUE)
#'
#' ## pgkw() is the integral of dgkw()
#' Fq <- pgkw(0.5, alpha = 2, beta = 3, gamma = 1.5, delta = 0.5, lambda = 1.2)
#' all.equal(Fq, integrate(dgkw, 0, 0.5, alpha = 2, beta = 3, gamma = 1.5,
#'     delta = 0.5, lambda = 1.2, rel.tol = 1e-10)$value)
#'
#' @export
pgkw <- function(q, alpha = 1, beta = 1, gamma = 1, delta = 0, lambda = 1,
                 lower.tail = TRUE, log.p = FALSE) {
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
  .shape_like(.Call("_gkwdist_pgkw",
    as.numeric(q),
    as.numeric(alpha),
    as.numeric(beta),
    as.numeric(gamma),
    as.numeric(delta),
    as.numeric(lambda),
    as.logical(lower.tail),
    as.logical(log.p),
    PACKAGE = "gkwdist"
  ), q)
}


# ----------------------------------------------------------------------------#
# 3. QUANTILE FUNCTION (qgkw)
# ----------------------------------------------------------------------------#

#' @title Quantile Function of the Generalized Kumaraswamy Distribution
#' @author Lopes, J. E.
#' @family quantile functions
#' @concept generalized kumaraswamy
#' @keywords distribution
#'
#' @description
#' Computes the quantile function (inverse CDF) for the five-parameter
#' Generalized Kumaraswamy (GKw) distribution. Finds the value \code{x} such
#' that \eqn{P(X \le x) = p}, where \code{X} follows the GKw distribution.
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
#' @param lambda Shape parameter \code{lambda} > 0. Can be a scalar or a vector.
#'   Default: 1.0.
#' @param lower.tail Logical; if \code{TRUE} (default), probabilities are
#'   \eqn{P(X \le x)}, otherwise, \eqn{P(X > x)}.
#' @param log.p Logical; if \code{TRUE}, probabilities \code{p} are given as
#'   \eqn{\log(p)}. Default: \code{FALSE}.
#'
#' @return A vector of quantiles corresponding to the given probabilities \code{p}.
#'   The length of the result is determined by the recycling rule applied to
#'   the arguments (\code{p}, \code{alpha}, \code{beta}, \code{gamma},
#'   \code{delta}, \code{lambda}). Returns:
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
#' The quantile function \eqn{Q(p)} is the inverse of the CDF \eqn{F(x)}.
#' Given \eqn{F(x) = I_{y(x)}(\gamma, \delta+1)} where
#' \eqn{y(x) = [1-(1-x^{\alpha})^{\beta}]^{\lambda}}, the quantile function is:
#' \deqn{
#' Q(p) = x = \left\{ 1 - \left[ 1 - \left( I^{-1}_{p}(\gamma, \delta+1) \right)^{1/\lambda} \right]^{1/\beta} \right\}^{1/\alpha}
#' }
#' where \eqn{I^{-1}_{p}(a, b)} is the inverse of the regularized incomplete beta
#' function, which corresponds to the quantile function of the Beta distribution,
#' \code{\link[stats]{qbeta}}.
#'
#' The computation proceeds as follows:
#' \enumerate{
#'   \item Calculate \code{y = stats::qbeta(p, shape1 = gamma, shape2 = delta + 1, lower.tail = lower.tail, log.p = log.p)}.
#'   \item Calculate \eqn{v = y^{1/\lambda}}.
#'   \item Calculate \eqn{w = (1 - v)^{1/\beta}}. Note: Requires \eqn{v \le 1}.
#'   \item Calculate \eqn{q = (1 - w)^{1/\alpha}}. Note: Requires \eqn{w \le 1}.
#' }
#' Numerical stability is maintained by handling boundary cases (\code{p = 0},
#' \code{p = 1}) directly and checking intermediate results (e.g., ensuring
#' arguments to powers are non-negative).
#'
#' @references
#' Carrasco, J. M. F., Ferrari, S. L. P., & Cordeiro, G. M. (2010). A new
#' generalized Kumaraswamy distribution. *arXiv preprint arXiv:1004.0911*.
#' \doi{10.48550/arXiv.1004.0911}
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
#' \code{\link{dgkw}}, \code{\link{pgkw}}, \code{\link{rgkw}},
#' \code{\link[stats]{qbeta}}
#'
#' @examples
#' p <- c(0.1, 0.5, 0.9)
#' qgkw(p, alpha = 2, beta = 3, gamma = 1.5, delta = 0.5, lambda = 1.2)
#' qgkw(p, alpha = 2, beta = 3, gamma = 1.5, delta = 0.5, lambda = 1.2,
#'     lower.tail = FALSE)  # upper-tail quantiles
#'
#' ## qgkw() inverts pgkw()
#' all.equal(pgkw(qgkw(p, alpha = 2, beta = 3, gamma = 1.5, delta = 0.5,
#'     lambda = 1.2), alpha = 2, beta = 3, gamma = 1.5, delta = 0.5, lambda = 1.2),
#'     p)
#'
#' @export
qgkw <- function(p, alpha = 1, beta = 1, gamma = 1, delta = 0, lambda = 1,
                 lower.tail = TRUE, log.p = FALSE) {
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
  .shape_like(.Call("_gkwdist_qgkw",
    as.numeric(p),
    as.numeric(alpha),
    as.numeric(beta),
    as.numeric(gamma),
    as.numeric(delta),
    as.numeric(lambda),
    as.logical(lower.tail),
    as.logical(log.p),
    PACKAGE = "gkwdist"
  ), p)
}


# ----------------------------------------------------------------------------#
# 4. RANDOM GENERATION (rgkw)
# ----------------------------------------------------------------------------#

#' @title Random Number Generation for the Generalized Kumaraswamy Distribution
#' @author Lopes, J. E.
#' @family random generation functions
#' @concept generalized kumaraswamy
#' @keywords distribution
#'
#' @description
#' Generates random deviates from the five-parameter Generalized Kumaraswamy (GKw)
#' distribution defined on the interval (0, 1).
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
#' @param lambda Shape parameter \code{lambda} > 0. Can be a scalar or a vector.
#'   Default: 1.0.
#'
#' @return A vector of length \code{n} containing random deviates from the GKw
#'   distribution. The length of the result is determined by \code{n} and the
#'   recycling rule applied to the parameters (\code{alpha}, \code{beta},
#'   \code{gamma}, \code{delta}, \code{lambda}). An out-of-bound or missing parameter is an
#'   error, not a return value: the wrapper stops with a message naming the
#'   parameter. An infinite parameter is not currently intercepted there and
#'   reaches the C++ layer, which treats it as invalid.
#'
#' @details
#' The generation method relies on the transformation property: if
#' \eqn{V \sim \mathrm{Beta}(\gamma, \delta+1)}, then the random variable \code{X}
#' defined as
#' \deqn{
#' X = \left\{ 1 - \left[ 1 - V^{1/\lambda} \right]^{1/\beta} \right\}^{1/\alpha}
#' }
#' follows the GKw(\eqn{\alpha, \beta, \gamma, \delta, \lambda}) distribution.
#'
#' The algorithm proceeds as follows:
#' \enumerate{
#'   \item Generate \code{V} from \code{stats::rbeta(n, shape1 = gamma, shape2 = delta + 1)}.
#'   \item Calculate \eqn{v = V^{1/\lambda}}.
#'   \item Calculate \eqn{w = (1 - v)^{1/\beta}}.
#'   \item Calculate \eqn{x = (1 - w)^{1/\alpha}}.
#' }
#' Parameters (\code{alpha}, \code{beta}, \code{gamma}, \code{delta}, \code{lambda})
#' are recycled to match the length required by \code{n}. Numerical stability is
#' maintained by handling potential edge cases during the transformations.
#'
#' @references
#' Carrasco, J. M. F., Ferrari, S. L. P., & Cordeiro, G. M. (2010). A new
#' generalized Kumaraswamy distribution. *arXiv preprint arXiv:1004.0911*.
#' \doi{10.48550/arXiv.1004.0911}
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
#' \code{\link{dgkw}}, \code{\link{pgkw}}, \code{\link{qgkw}},
#' \code{\link[stats]{rbeta}}, \code{\link[base]{set.seed}}
#'
#' @examples
#' set.seed(123)
#' x <- rgkw(1000, alpha = 2, beta = 3, gamma = 1.5, delta = 0.5, lambda = 1.2)
#' summary(x)
#'
#' ## The sample follows the distribution
#' hist(x, breaks = 30, freq = FALSE, main = "")
#' curve(dgkw(x, alpha = 2, beta = 3, gamma = 1.5, delta = 0.5, lambda = 1.2),
#'     add = TRUE)
#' ks.test(x, pgkw, alpha = 2, beta = 3, gamma = 1.5, delta = 0.5, lambda = 1.2)
#'
#' @export
rgkw <- function(n, alpha = 1, beta = 1, gamma = 1, delta = 0, lambda = 1) {
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
  if (!is.numeric(lambda) || anyNA(lambda) || any(lambda <= 0)) {
    stop("'lambda' must be positive")
  }

  # Call C++ implementation
  .Call("_gkwdist_rgkw",
    as.integer(n),
    as.numeric(alpha),
    as.numeric(beta),
    as.numeric(gamma),
    as.numeric(delta),
    as.numeric(lambda),
    PACKAGE = "gkwdist"
  )
}


# ============================================================================#
# MAXIMUM LIKELIHOOD ESTIMATION FUNCTIONS
# ============================================================================#

# ----------------------------------------------------------------------------#
# 5. NEGATIVE LOG-LIKELIHOOD (llgkw)
# ----------------------------------------------------------------------------#

#' @title Negative Log-Likelihood for the Generalized Kumaraswamy Distribution
#' @author Lopes, J. E.
#' @family log-likelihood functions
#' @concept generalized kumaraswamy
#' @keywords distribution optimize
#'
#' @description
#' Computes the negative log-likelihood function for the five-parameter
#' Generalized Kumaraswamy (GKw) distribution given a vector of observations.
#' This function is designed for use in optimization routines (e.g., maximum
#' likelihood estimation).
#'
#' @param par A numeric vector of length 5 containing the distribution parameters
#'   in the order: \code{alpha} (\eqn{\alpha > 0}), \code{beta} (\eqn{\beta > 0}),
#'   \code{gamma} (\eqn{\gamma > 0}), \code{delta} (\eqn{\delta \ge 0}),
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
#' The probability density function (PDF) of the GKw distribution is given in
#' \code{\link{dgkw}}. The log-likelihood function \eqn{\ell(\theta)} for a sample
#' \eqn{\mathbf{x} = (x_1, \dots, x_n)} is:
#' \deqn{
#' \ell(\theta | \mathbf{x}) = n\ln(\lambda\alpha\beta) - n\ln B(\gamma,\delta+1) +
#'   \sum_{i=1}^{n} [(\alpha-1)\ln(x_i) + (\beta-1)\ln(v_i) + (\gamma\lambda-1)\ln(w_i) + \delta\ln(z_i)]
#' }
#' where \eqn{\theta = (\alpha, \beta, \gamma, \delta, \lambda)}, \eqn{B(a,b)}
#' is the Beta function (\code{\link[base]{beta}}), and:
#' \itemize{
#'   \item \eqn{v_i = 1 - x_i^{\alpha}}
#'   \item \eqn{w_i = 1 - v_i^{\beta} = 1 - (1-x_i^{\alpha})^{\beta}}
#'   \item \eqn{z_i = 1 - w_i^{\lambda} = 1 - [1-(1-x_i^{\alpha})^{\beta}]^{\lambda}}
#' }
#' This function computes \eqn{-\ell(\theta|\mathbf{x})}.
#'
#' Numerical stability is prioritized using:
#' \itemize{
#'   \item \code{\link[base]{lbeta}} function for the log-Beta term.
#'   \item Log-transformations of intermediate terms (\eqn{v_i, w_i, z_i}) and
#'         use of \code{\link[base]{log1p}} where appropriate to handle values
#'         close to 0 or 1 accurately.
#'   \item Checks for invalid parameters and data.
#' }
#'
#' @references
#' Carrasco, J. M. F., Ferrari, S. L. P., & Cordeiro, G. M. (2010). A new
#' generalized Kumaraswamy distribution. *arXiv preprint arXiv:1004.0911*.
#' \doi{10.48550/arXiv.1004.0911}
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
#' \code{\link{dgkw}}, \code{\link{pgkw}}, \code{\link{qgkw}}, \code{\link{rgkw}},
#' \code{\link{grgkw}}, \code{\link{hsgkw}} (gradient and Hessian),
#' \code{\link[stats]{optim}}, \code{\link[base]{lbeta}}, \code{\link[base]{log1p}}
#'
#' @examples
#' set.seed(123)
#' x <- rgkw(1000, alpha = 2, beta = 3, gamma = 1.5, delta = 0.5, lambda = 1.2)
#' par <- c(alpha = 2, beta = 3, gamma = 1.5, delta = 0.5, lambda = 1.2)
#'
#' ## llgkw() is the negative log-likelihood, -sum(log f(x))
#' llgkw(par, x)
#' -sum(dgkw(x, alpha = 2, beta = 3, gamma = 1.5, delta = 0.5, lambda = 1.2,
#'     log = TRUE))
#'
#' ## Maximum likelihood: minimize llgkw(), with grgkw() as its gradient
#' start <- gkwgetstartvalues(x, family = "gkw")
#' fit <- optim(start, llgkw, grgkw, data = x, method = "L-BFGS-B", lower = 1e-4)
#' fit$convergence  # 0: converged
#' ## The parameters of this family are weakly identified: compare likelihoods
#' fit$value <= llgkw(par, x)  # at least as good as the true values
#'
#' @export
llgkw <- function(par, data) {
  # Input validation
  if (!is.numeric(par) || length(par) != 5) {
    stop("'par' must be a numeric vector of length 5")
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
  .Call("_gkwdist_llgkw",
    as.numeric(par),
    as.numeric(data),
    PACKAGE = "gkwdist"
  )
}


# ----------------------------------------------------------------------------#
# 6. GRADIENT (grgkw)
# ----------------------------------------------------------------------------#

#' @title Gradient of the Negative Log-Likelihood for the GKw Distribution
#' @author Lopes, J. E.
#' @family gradient functions
#' @concept generalized kumaraswamy
#' @keywords distribution optimize
#'
#' @description
#' Computes the gradient vector (vector of partial derivatives) of the negative
#' log-likelihood function for the five-parameter Generalized Kumaraswamy (GKw)
#' distribution. This provides the analytical gradient, often used for efficient
#' optimization via maximum likelihood estimation.
#'
#' @param par A numeric vector of length 5 containing the distribution parameters
#'   in the order: \code{alpha} (\eqn{\alpha > 0}), \code{beta} (\eqn{\beta > 0}),
#'   \code{gamma} (\eqn{\gamma > 0}), \code{delta} (\eqn{\delta \ge 0}),
#'   \code{lambda} (\eqn{\lambda > 0}).
#' @param data A numeric vector of observations. All values must be strictly
#'   between 0 and 1 (exclusive).
#'
#' @return Returns a numeric vector of length 5 containing the partial derivatives
#'   of the negative log-likelihood function \eqn{-\ell(\theta | \mathbf{x})} with
#'   respect to each parameter:
#'   \eqn{(-\partial \ell/\partial \alpha, -\partial \ell/\partial \beta, -\partial \ell/\partial \gamma, -\partial \ell/\partial \delta, -\partial \ell/\partial \lambda)}.
#'   Returns a vector of \code{NaN} if any parameter values are invalid according
#'   to their constraints, or if any value in \code{data} is not in the
#'   interval (0, 1).
#'
#' @details
#' The components of the gradient vector of the negative log-likelihood
#' (\eqn{-\nabla \ell(\theta | \mathbf{x})}) are:
#'
#' \deqn{
#' -\frac{\partial \ell}{\partial \alpha} = -\frac{n}{\alpha} - \sum_{i=1}^{n}\ln(x_i) +
#' \sum_{i=1}^{n}\left[x_i^{\alpha} \ln(x_i) \left(\frac{\beta-1}{v_i} -
#' \frac{(\gamma\lambda-1) \beta v_i^{\beta-1}}{w_i} +
#' \frac{\delta \lambda \beta v_i^{\beta-1} w_i^{\lambda-1}}{z_i}\right)\right]
#' }
#' \deqn{
#' -\frac{\partial \ell}{\partial \beta} = -\frac{n}{\beta} - \sum_{i=1}^{n}\ln(v_i) +
#' \sum_{i=1}^{n}\left[v_i^{\beta} \ln(v_i) \left(\frac{\gamma\lambda-1}{w_i} -
#' \frac{\delta \lambda w_i^{\lambda-1}}{z_i}\right)\right]
#' }
#' \deqn{
#' -\frac{\partial \ell}{\partial \gamma} = n[\psi(\gamma) - \psi(\gamma+\delta+1)] -
#' \lambda\sum_{i=1}^{n}\ln(w_i)
#' }
#' \deqn{
#' -\frac{\partial \ell}{\partial \delta} = n[\psi(\delta+1) - \psi(\gamma+\delta+1)] -
#' \sum_{i=1}^{n}\ln(z_i)
#' }
#' \deqn{
#' -\frac{\partial \ell}{\partial \lambda} = -\frac{n}{\lambda} -
#' \gamma\sum_{i=1}^{n}\ln(w_i) + \delta\sum_{i=1}^{n}\frac{w_i^{\lambda}\ln(w_i)}{z_i}
#' }
#'
#' where:
#' \itemize{
#'   \item \eqn{v_i = 1 - x_i^{\alpha}}
#'   \item \eqn{w_i = 1 - v_i^{\beta} = 1 - (1-x_i^{\alpha})^{\beta}}
#'   \item \eqn{z_i = 1 - w_i^{\lambda} = 1 - [1-(1-x_i^{\alpha})^{\beta}]^{\lambda}}
#'   \item \eqn{\psi(\cdot)} is the digamma function (\code{\link[base]{digamma}}).
#' }
#'
#' Numerical stability is ensured through careful implementation, including checks
#' for valid inputs and handling of intermediate calculations involving potentially
#' small or large numbers, often leveraging the Armadillo C++ library for efficiency.
#'
#' @references
#' Carrasco, J. M. F., Ferrari, S. L. P., & Cordeiro, G. M. (2010). A new
#' generalized Kumaraswamy distribution. *arXiv preprint arXiv:1004.0911*.
#' \doi{10.48550/arXiv.1004.0911}
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
#' \code{\link{llgkw}} (negative log-likelihood),
#' \code{\link{hsgkw}} (Hessian matrix),
#' \code{\link{dgkw}} (density),
#' \code{\link[stats]{optim}},
#' \code{\link[numDeriv]{grad}} (for numerical gradient comparison),
#' \code{\link[base]{digamma}}
#'
#' @examples
#' set.seed(123)
#' x <- rgkw(200, alpha = 2, beta = 3, gamma = 1.5, delta = 0.5, lambda = 1.2)
#' par <- c(alpha = 2, beta = 3, gamma = 1.5, delta = 0.5, lambda = 1.2)
#'
#' ## Gradient of the negative log-likelihood llgkw(), not of the log-likelihood
#' g <- grgkw(par, x)
#' g
#'
#' ## A small step against the gradient lowers llgkw()
#' llgkw(par - 1e-4 * g, x) < llgkw(par, x)
#'
#' ## Agrees with a numerical derivative of llgkw()
#' if (requireNamespace("numDeriv", quietly = TRUE))
#'   all.equal(g, numDeriv::grad(llgkw, par, data = x), tolerance = 1e-6)
#'
#' @export
grgkw <- function(par, data) {
  # Input validation
  if (!is.numeric(par) || length(par) != 5) {
    stop("'par' must be a numeric vector of length 5")
  }
  if (!is.numeric(data)) {
    stop("'data' must be numeric")
  }
  if (length(data) < 1) {
    stop("'data' must have at least one observation")
  }

  # Call C++ implementation
  .Call("_gkwdist_grgkw",
    as.numeric(par),
    as.numeric(data),
    PACKAGE = "gkwdist"
  )
}


# ----------------------------------------------------------------------------#
# 7. HESSIAN (hsgkw)
# ----------------------------------------------------------------------------#

#' @title Hessian Matrix of the Negative Log-Likelihood for the GKw Distribution
#' @author Lopes, J. E.
#' @family Hessian functions
#' @concept generalized kumaraswamy
#' @keywords distribution optimize
#'
#' @description
#' Computes the analytic Hessian matrix (matrix of second partial derivatives)
#' of the negative log-likelihood function for the five-parameter Generalized
#' Kumaraswamy (GKw) distribution. This is typically used to estimate standard
#' errors of maximum likelihood estimates or in optimization algorithms.
#'
#' @param par A numeric vector of length 5 containing the distribution parameters
#'   in the order: \code{alpha} (\eqn{\alpha > 0}), \code{beta} (\eqn{\beta > 0}),
#'   \code{gamma} (\eqn{\gamma > 0}), \code{delta} (\eqn{\delta \ge 0}),
#'   \code{lambda} (\eqn{\lambda > 0}).
#' @param data A numeric vector of observations. All values must be strictly
#'   between 0 and 1 (exclusive).
#'
#' @return Returns a 5x5 numeric matrix representing the Hessian matrix of the
#'   negative log-likelihood function, i.e., the matrix of second partial
#'   derivatives \eqn{-\partial^2 \ell / (\partial \theta_i \partial \theta_j)}.
#'   Returns a 5x5 matrix populated with \code{NaN} if any parameter values are
#'   invalid according to their constraints, or if any value in \code{data} is
#'   not in the interval (0, 1).
#'
#' @details
#' This function calculates the analytic second partial derivatives of the
#' negative log-likelihood function based on the GKw PDF (see \code{\link{dgkw}}).
#' The log-likelihood function \eqn{\ell(\theta | \mathbf{x})} is given by:
#' \deqn{
#' \ell(\theta) = n \ln(\lambda\alpha\beta) - n \ln B(\gamma, \delta+1)
#' + \sum_{i=1}^{n} [(\alpha-1) \ln(x_i)
#' + (\beta-1) \ln(v_i)
#' + (\gamma\lambda - 1) \ln(w_i)
#' + \delta \ln(z_i)]
#' }
#' where \eqn{\theta = (\alpha, \beta, \gamma, \delta, \lambda)}, \eqn{B(a,b)}
#' is the Beta function (\code{\link[base]{beta}}), and intermediate terms are:
#' \itemize{
#'   \item \eqn{v_i = 1 - x_i^{\alpha}}
#'   \item \eqn{w_i = 1 - v_i^{\beta} = 1 - (1-x_i^{\alpha})^{\beta}}
#'   \item \eqn{z_i = 1 - w_i^{\lambda} = 1 - [1-(1-x_i^{\alpha})^{\beta}]^{\lambda}}
#' }
#' The Hessian matrix returned contains the elements \eqn{- \frac{\partial^2 \ell(\theta | \mathbf{x})}{\partial \theta_i \partial \theta_j}}.
#'
#' Key properties of the returned matrix:
#' \itemize{
#'   \item Dimensions: 5x5.
#'   \item Symmetry: The matrix is symmetric.
#'   \item Ordering: Rows and columns correspond to the parameters in the order
#'     \eqn{\alpha, \beta, \gamma, \delta, \lambda}.
#'   \item Content: Analytic second derivatives of the *negative* log-likelihood.
#' }
#' The exact analytical formulas for the second derivatives are implemented
#' directly (often derived using symbolic differentiation) for accuracy and
#' efficiency, typically using C++.
#'
#' @references
#' Carrasco, J. M. F., Ferrari, S. L. P., & Cordeiro, G. M. (2010). A new
#' generalized Kumaraswamy distribution. *arXiv preprint arXiv:1004.0911*.
#' \doi{10.48550/arXiv.1004.0911}
#'
#' Cordeiro, G. M., & de Castro, M. (2011). A new family of generalized
#' distributions. *Journal of Statistical Computation and Simulation*,
#' *81*(7), 883-898.
#' \doi{10.1080/00949650903530745}
#'
#' Kumaraswamy, P. (1980). A generalized probability density function for
#' double-bounded random processes. *Journal of Hydrology*, *46*(1-2), 79-88.
#' \doi{10.1016/0022-1694(80)90036-0}
#' @seealso
#' \code{\link{llgkw}} (negative log-likelihood function),
#' \code{\link{grgkw}} (gradient vector),
#' \code{\link{dgkw}} (density function),
#' \code{\link[stats]{optim}},
#' \code{\link[numDeriv]{hessian}} (for numerical Hessian comparison).
#'
#' @examples
#' set.seed(123)
#' x <- rgkw(1000, alpha = 2, beta = 3, gamma = 1.5, delta = 0.5, lambda = 1.2)
#' par <- c(alpha = 2, beta = 3, gamma = 1.5, delta = 0.5, lambda = 1.2)
#'
#' ## Hessian of the negative log-likelihood llgkw()
#' H <- hsgkw(par, x)
#' isSymmetric(H)
#'
#' ## Agrees with a numerical Hessian of llgkw()
#' if (requireNamespace("numDeriv", quietly = TRUE))
#'   all.equal(H, numDeriv::hessian(llgkw, par, data = x), tolerance = 1e-8)
#'
#' @export
hsgkw <- function(par, data) {
  # Input validation
  if (!is.numeric(par) || length(par) != 5) {
    stop("'par' must be a numeric vector of length 5")
  }
  if (!is.numeric(data)) {
    stop("'data' must be numeric")
  }
  if (length(data) < 1) {
    stop("'data' must have at least one observation")
  }

  # Call C++ implementation
  .Call("_gkwdist_hsgkw",
    as.numeric(par),
    as.numeric(data),
    PACKAGE = "gkwdist"
  )
}
