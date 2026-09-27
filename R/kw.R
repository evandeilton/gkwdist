# ============================================================================#
# KUMARASWAMY (Kw) DISTRIBUTION
# ============================================================================#
#
# Wrapper functions for the two-parameter Kumaraswamy distribution.
# C++ implementations are in src/kw.cpp
#
# Functions:
#   - dkw: Probability density function (PDF)
#   - pkw: Cumulative distribution function (CDF)
#   - qkw: Quantile function (inverse CDF)
#   - rkw: Random number generation
#   - llkw: Negative log-likelihood
#   - grkw: Gradient of negative log-likelihood
#   - hskw: Hessian of negative log-likelihood
# ============================================================================#


# ----------------------------------------------------------------------------#
# 1. DENSITY FUNCTION (dkw)
# ----------------------------------------------------------------------------#


#' @title Density of the Kumaraswamy (Kw) Distribution
#' @author Lopes, J. E.
#' @family density functions
#' @concept kumaraswamy
#' @keywords distribution
#'
#' @description
#' Computes the probability density function (PDF) for the two-parameter
#' Kumaraswamy (Kw) distribution with shape parameters \code{alpha} (\eqn{\alpha})
#' and \code{beta} (\eqn{\beta}). This distribution is defined on the interval (0, 1).
#'
#' @param x Vector of quantiles (values between 0 and 1).
#' @param alpha Shape parameter \code{alpha} > 0. Can be a scalar or a vector.
#'   Default: 1.0.
#' @param beta Shape parameter \code{beta} > 0. Can be a scalar or a vector.
#'   Default: 1.0.
#' @param log Logical; if \code{TRUE}, the logarithm of the density is
#'   returned (\eqn{\log(f(x))}). Default: \code{FALSE}.
#'
#' @return A vector of density values (\eqn{f(x)}) or log-density values
#'   (\eqn{\log(f(x))}). The length of the result is determined by the recycling
#'   rule applied to the arguments (\code{x}, \code{alpha}, \code{beta}).
#'   Returns \code{0} (or \code{-Inf} if \code{log = TRUE}) for \code{x}
#'   strictly outside the interval \[0, 1\]. At the closed boundaries
#'   \code{x = 0} and \code{x = 1} the limiting density is returned rather than
#'   \code{0}, following the convention of base R's density functions (compare
#'   \code{\link[stats]{dbeta}}); depending on the parameters that limit is
#'   \code{0}, a finite positive value, or \code{Inf}.
#'   An out-of-bound or missing parameter is an error, not a return value: the
#'   wrapper stops with a message naming the parameter. An infinite parameter is
#'   not currently intercepted there and reaches the C++ layer, which treats it
#'   as invalid.
#'
#' @details
#' The probability density function (PDF) of the Kumaraswamy (Kw) distribution
#' is given by:
#' \deqn{
#' f(x; \alpha, \beta) = \alpha \beta x^{\alpha-1} (1 - x^\alpha)^{\beta-1}
#' }
#' for \eqn{0 < x < 1}, \eqn{\alpha > 0}, and \eqn{\beta > 0}.
#'
#' The Kumaraswamy distribution is identical to the Generalized Kumaraswamy (GKw)
#' distribution (\code{\link{dgkw}}) with parameters \eqn{\gamma = 1},
#' \eqn{\delta = 0}, and \eqn{\lambda = 1}. It is also a special case of the
#' Exponentiated Kumaraswamy (\code{\link{dekw}}) with \eqn{\lambda = 1}, and
#' the Kumaraswamy-Kumaraswamy (\code{\link{dkkw}}) with \eqn{\delta = 0}
#' and \eqn{\lambda = 1}.
#'
#' @references
#' Kumaraswamy, P. (1980). A generalized probability density function for
#' double-bounded random processes. *Journal of Hydrology*, *46*(1-2), 79-88.
#' \doi{10.1016/0022-1694(80)90036-0}
#'
#'
#' Jones, M. C. (2009). Kumaraswamy's distribution: A beta-type distribution
#' with some tractability advantages. *Statistical Methodology*, *6*(1), 70-81.
#' \doi{10.1016/j.stamet.2008.04.001}
#'
#'
#' @seealso
#' \code{\link{dgkw}} (parent distribution density),
#' \code{\link{dekw}}, \code{\link{dkkw}},
#' \code{\link{pkw}}, \code{\link{qkw}}, \code{\link{rkw}} (other Kw functions),
#' \code{\link[stats]{dbeta}}
#'
#' @examples
#' x <- c(0.1, 0.3, 0.5, 0.7, 0.9)
#' dkw(x, alpha = 2, beta = 3)
#' dkw(x, alpha = 2, beta = 3, log = TRUE)
#'
#' ## Closed form: alpha * beta * x^(alpha - 1) * (1 - x^alpha)^(beta - 1)
#' all.equal(dkw(x, 2, 3), 2 * 3 * x * (1 - x^2)^2)
#'
#' ## The density integrates to one
#' integrate(dkw, 0, 1, alpha = 2, beta = 3, rel.tol = 1e-10)
#'
#' curve(dkw(x, alpha = 2, beta = 3), from = 0, to = 1, ylab = "density")
#'
#' @export
dkw <- function(x, alpha = 1, beta = 1, log = FALSE) {
  # Input validation
  if (!is.numeric(x)) stop("'x' must be numeric")
  if (!is.numeric(alpha) || anyNA(alpha) || any(alpha <= 0)) {
    stop("'alpha' must be positive")
  }
  if (!is.numeric(beta) || anyNA(beta) || any(beta <= 0)) {
    stop("'beta' must be positive")
  }
  if (!is.logical(log) || length(log) != 1 || is.na(log)) {
    stop("'log' must be a single logical value")
  }

  # Call C++ implementation
  .shape_like(.Call("_gkwdist_dkw",
    as.numeric(x),
    as.numeric(alpha),
    as.numeric(beta),
    as.logical(log),
    PACKAGE = "gkwdist"
  ), x)
}


# ----------------------------------------------------------------------------#
# 2. DISTRIBUTION FUNCTION (pkw)
# ----------------------------------------------------------------------------#

#' @title Cumulative Distribution Function (CDF) of the Kumaraswamy (Kw) Distribution
#' @author Lopes, J. E.
#' @family cumulative distribution functions
#' @concept kumaraswamy
#' @keywords distribution
#'
#' @description
#' Computes the cumulative distribution function (CDF), \eqn{P(X \le q)}, for the
#' two-parameter Kumaraswamy (Kw) distribution with shape parameters \code{alpha}
#' (\eqn{\alpha}) and \code{beta} (\eqn{\beta}). This distribution is defined
#' on the interval (0, 1).
#'
#' @param q Vector of quantiles (values generally between 0 and 1).
#' @param alpha Shape parameter \code{alpha} > 0. Can be a scalar or a vector.
#'   Default: 1.0.
#' @param beta Shape parameter \code{beta} > 0. Can be a scalar or a vector.
#'   Default: 1.0.
#' @param lower.tail Logical; if \code{TRUE} (default), probabilities are
#'   \eqn{P(X \le q)}, otherwise, \eqn{P(X > q)}.
#' @param log.p Logical; if \code{TRUE}, probabilities \eqn{p} are given as
#'   \eqn{\log(p)}. Default: \code{FALSE}.
#'
#' @return A vector of probabilities, \eqn{F(q)}, or their logarithms/complements
#'   depending on \code{lower.tail} and \code{log.p}. The length of the result
#'   is determined by the recycling rule applied to the arguments (\code{q},
#'   \code{alpha}, \code{beta}). When \code{lower.tail = TRUE}, returns
#'   \code{0} (or \code{-Inf} if \code{log.p = TRUE}) for \code{q <= 0} and
#'   \code{1} (or \code{0} if \code{log.p = TRUE}) for \code{q >= 1}. An
#'   out-of-bound or missing parameter is an error, not a return value: the
#'   wrapper stops with a message naming the parameter. An infinite parameter
#'   is not currently intercepted there and reaches the C++ layer, which
#'   treats it as invalid.
#'   Boundary return values are adjusted accordingly for \code{lower.tail = FALSE}.
#'
#' @details
#' The cumulative distribution function (CDF) of the Kumaraswamy (Kw)
#' distribution is given by:
#' \deqn{
#' F(x; \alpha, \beta) = 1 - (1 - x^\alpha)^\beta
#' }
#' for \eqn{0 < x < 1}, \eqn{\alpha > 0}, and \eqn{\beta > 0}.
#'
#' The Kw distribution is a special case of several generalized distributions:
#' \itemize{
#'  \item Generalized Kumaraswamy (\code{\link{pgkw}}) with \eqn{\gamma=1, \delta=0, \lambda=1}.
#'  \item Exponentiated Kumaraswamy (\code{\link{pekw}}) with \eqn{\lambda=1}.
#'  \item Kumaraswamy-Kumaraswamy (\code{\link{pkkw}}) with \eqn{\delta=0, \lambda=1}.
#' }
#' The implementation uses the closed-form expression for efficiency.
#'
#' @references
#' Kumaraswamy, P. (1980). A generalized probability density function for
#' double-bounded random processes. *Journal of Hydrology*, *46*(1-2), 79-88.
#' \doi{10.1016/0022-1694(80)90036-0}
#'
#'
#' Jones, M. C. (2009). Kumaraswamy's distribution: A beta-type distribution
#' with some tractability advantages. *Statistical Methodology*, *6*(1), 70-81.
#' \doi{10.1016/j.stamet.2008.04.001}
#'
#'
#' @seealso
#' \code{\link{pgkw}}, \code{\link{pekw}}, \code{\link{pkkw}} (related generalized CDFs),
#' \code{\link{dkw}}, \code{\link{qkw}}, \code{\link{rkw}} (other Kw functions),
#' \code{\link[stats]{pbeta}}
#'
#' @examples
#' q <- c(0.2, 0.5, 0.8)
#' pkw(q, alpha = 2, beta = 3)
#' pkw(q, alpha = 2, beta = 3, lower.tail = FALSE)  # P(X > q)
#' pkw(q, alpha = 2, beta = 3, log.p = TRUE)
#'
#' ## pkw() is the integral of dkw()
#' Fq <- pkw(0.5, alpha = 2, beta = 3)
#' all.equal(Fq, integrate(dkw, 0, 0.5, alpha = 2, beta = 3, rel.tol = 1e-10)$value)
#'
#' @export
pkw <- function(q, alpha = 1, beta = 1, lower.tail = TRUE, log.p = FALSE) {
  # Input validation
  if (!is.numeric(q)) stop("'q' must be numeric")
  if (!is.numeric(alpha) || anyNA(alpha) || any(alpha <= 0)) {
    stop("'alpha' must be positive")
  }
  if (!is.numeric(beta) || anyNA(beta) || any(beta <= 0)) {
    stop("'beta' must be positive")
  }
  if (!is.logical(lower.tail) || length(lower.tail) != 1 || is.na(lower.tail)) {
    stop("'lower.tail' must be a single logical value")
  }
  if (!is.logical(log.p) || length(log.p) != 1 || is.na(log.p)) {
    stop("'log.p' must be a single logical value")
  }

  # Call C++ implementation
  .shape_like(.Call("_gkwdist_pkw",
    as.numeric(q),
    as.numeric(alpha),
    as.numeric(beta),
    as.logical(lower.tail),
    as.logical(log.p),
    PACKAGE = "gkwdist"
  ), q)
}


# ----------------------------------------------------------------------------#
# 3. QUANTILE FUNCTION (qkw)
# ----------------------------------------------------------------------------#

#' @title Quantile Function of the Kumaraswamy (Kw) Distribution
#' @author Lopes, J. E.
#' @family quantile functions
#' @concept kumaraswamy
#' @keywords distribution
#'
#' @description
#' Computes the quantile function (inverse CDF) for the two-parameter
#' Kumaraswamy (Kw) distribution with shape parameters \code{alpha} (\eqn{\alpha})
#' and \code{beta} (\eqn{\beta}). It finds the value \code{q} such that
#' \eqn{P(X \le q) = p}.
#'
#' @param p Vector of probabilities (values between 0 and 1).
#' @param alpha Shape parameter \code{alpha} > 0. Can be a scalar or a vector.
#'   Default: 1.0.
#' @param beta Shape parameter \code{beta} > 0. Can be a scalar or a vector.
#'   Default: 1.0.
#' @param lower.tail Logical; if \code{TRUE} (default), probabilities are \eqn{p = P(X \le q)},
#'   otherwise, probabilities are \eqn{p = P(X > q)}.
#' @param log.p Logical; if \code{TRUE}, probabilities \code{p} are given as
#'   \eqn{\log(p)}. Default: \code{FALSE}.
#'
#' @return A vector of quantiles corresponding to the given probabilities \code{p}.
#'   The length of the result is determined by the recycling rule applied to
#'   the arguments (\code{p}, \code{alpha}, \code{beta}).
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
#' for the Kumaraswamy distribution is \eqn{F(q) = 1 - (1 - q^\alpha)^\beta}
#' (see \code{\link{pkw}}). Inverting this equation for \eqn{q} yields the
#' quantile function:
#' \deqn{
#' Q(p) = \left\{ 1 - (1 - p)^{1/\beta} \right\}^{1/\alpha}
#' }
#' The function uses this closed-form expression and correctly handles the
#' \code{lower.tail} and \code{log.p} arguments by transforming \code{p}
#' appropriately before applying the formula. This is equivalent to the general
#' GKw quantile function (\code{\link{qgkw}}) evaluated with \eqn{\gamma=1, \delta=0, \lambda=1}.
#'
#' @references
#' Kumaraswamy, P. (1980). A generalized probability density function for
#' double-bounded random processes. *Journal of Hydrology*, *46*(1-2), 79-88.
#' \doi{10.1016/0022-1694(80)90036-0}
#'
#'
#' Jones, M. C. (2009). Kumaraswamy's distribution: A beta-type distribution
#' with some tractability advantages. *Statistical Methodology*, *6*(1), 70-81.
#' \doi{10.1016/j.stamet.2008.04.001}
#'
#'
#' @seealso
#' \code{\link{qgkw}} (parent distribution quantile function),
#' \code{\link{dkw}}, \code{\link{pkw}}, \code{\link{rkw}} (other Kw functions),
#' \code{\link[stats]{qbeta}}, \code{\link[stats]{qunif}}
#'
#' @examples
#' p <- c(0.1, 0.5, 0.9)
#' qkw(p, alpha = 2, beta = 3)
#' qkw(p, alpha = 2, beta = 3, lower.tail = FALSE)  # upper-tail quantiles
#'
#' ## qkw() inverts pkw()
#' all.equal(pkw(qkw(p, alpha = 2, beta = 3), alpha = 2, beta = 3), p)
#'
#' @export
qkw <- function(p, alpha = 1, beta = 1, lower.tail = TRUE, log.p = FALSE) {
  # Input validation
  if (!is.numeric(p)) stop("'p' must be numeric")
  if (!is.numeric(alpha) || anyNA(alpha) || any(alpha <= 0)) {
    stop("'alpha' must be positive")
  }
  if (!is.numeric(beta) || anyNA(beta) || any(beta <= 0)) {
    stop("'beta' must be positive")
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
  .shape_like(.Call("_gkwdist_qkw",
    as.numeric(p),
    as.numeric(alpha),
    as.numeric(beta),
    as.logical(lower.tail),
    as.logical(log.p),
    PACKAGE = "gkwdist"
  ), p)
}


# ----------------------------------------------------------------------------#
# 4. RANDOM GENERATION (rkw)
# ----------------------------------------------------------------------------#

#' @title Random Number Generation for the Kumaraswamy (Kw) Distribution
#' @author Lopes, J. E.
#' @family random generation functions
#' @concept kumaraswamy
#' @keywords distribution
#'
#' @description
#' Generates random deviates from the two-parameter Kumaraswamy (Kw)
#' distribution with shape parameters \code{alpha} (\eqn{\alpha}) and
#' \code{beta} (\eqn{\beta}).
#'
#' @param n Number of observations. If \code{length(n) > 1}, the length is
#'   taken to be the number required. Must be a non-negative integer.
#' @param alpha Shape parameter \code{alpha} > 0. Can be a scalar or a vector.
#'   Default: 1.0.
#' @param beta Shape parameter \code{beta} > 0. Can be a scalar or a vector.
#'   Default: 1.0.
#'
#' @return A vector of length \code{n} containing random deviates from the Kw
#'   distribution, with values in (0, 1). The length of the result is determined
#'   by \code{n} and the recycling rule applied to the parameters (\code{alpha},
#'   \code{beta}). An out-of-bound or missing parameter is an
#'   error, not a return value: the wrapper stops with a message naming the
#'   parameter. An infinite parameter is not currently intercepted there and
#'   reaches the C++ layer, which treats it as invalid.
#'
#' @details
#' The generation method uses the inverse transform (quantile) method.
#' That is, if \eqn{U} is a random variable following a standard Uniform
#' distribution on (0, 1), then \eqn{X = Q(U)} follows the Kw distribution,
#' where \eqn{Q(p)} is the Kw quantile function (\code{\link{qkw}}):
#' \deqn{
#' Q(p) = \left\{ 1 - (1 - p)^{1/\beta} \right\}^{1/\alpha}
#' }
#' The implementation generates \eqn{U} using \code{\link[stats]{runif}}
#' and applies this transformation. This is equivalent to the general GKw
#' generation method (\code{\link{rgkw}}) evaluated at \eqn{\gamma=1, \delta=0, \lambda=1}.
#'
#' @references
#' Kumaraswamy, P. (1980). A generalized probability density function for
#' double-bounded random processes. *Journal of Hydrology*, *46*(1-2), 79-88.
#' \doi{10.1016/0022-1694(80)90036-0}
#'
#'
#' Jones, M. C. (2009). Kumaraswamy's distribution: A beta-type distribution
#' with some tractability advantages. *Statistical Methodology*, *6*(1), 70-81.
#' \doi{10.1016/j.stamet.2008.04.001}
#'
#'
#' Devroye, L. (1986). *Non-Uniform Random Variate Generation*. Springer-Verlag.
#' (General methods for random variate generation).
#'
#' @seealso
#' \code{\link{rgkw}} (parent distribution random generation),
#' \code{\link{dkw}}, \code{\link{pkw}}, \code{\link{qkw}} (other Kw functions),
#' \code{\link[stats]{runif}}
#'
#' @examples
#' set.seed(123)
#' x <- rkw(1000, alpha = 2, beta = 3)
#' summary(x)
#'
#' ## The sample follows the distribution
#' hist(x, breaks = 30, freq = FALSE, main = "")
#' curve(dkw(x, alpha = 2, beta = 3), add = TRUE)
#' ks.test(x, pkw, alpha = 2, beta = 3)
#'
#' @export
rkw <- function(n, alpha = 1, beta = 1) {
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

  # Call C++ implementation
  .Call("_gkwdist_rkw",
    as.integer(n),
    as.numeric(alpha),
    as.numeric(beta),
    PACKAGE = "gkwdist"
  )
}


# ============================================================================#
# MAXIMUM LIKELIHOOD ESTIMATION FUNCTIONS
# ============================================================================#

# ----------------------------------------------------------------------------#
# 5. NEGATIVE LOG-LIKELIHOOD (llkw)
# ----------------------------------------------------------------------------#

#' @title Negative Log-Likelihood for the Kumaraswamy (Kw) Distribution
#' @author Lopes, J. E.
#' @family log-likelihood functions
#' @concept kumaraswamy
#' @keywords distribution optimize
#'
#' @description
#' Computes the negative log-likelihood function for the two-parameter
#' Kumaraswamy (Kw) distribution with parameters \code{alpha} (\eqn{\alpha})
#' and \code{beta} (\eqn{\beta}), given a vector of observations. This function
#' is suitable for maximum likelihood estimation.
#'
#' @param par A numeric vector of length 2 containing the distribution parameters
#'   in the order: \code{alpha} (\eqn{\alpha > 0}), \code{beta} (\eqn{\beta > 0}).
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
#' The Kumaraswamy (Kw) distribution's probability density function (PDF) is
#' (see \code{\link{dkw}}):
#' \deqn{
#' f(x | \theta) = \alpha \beta x^{\alpha-1} (1 - x^\alpha)^{\beta-1}
#' }
#' for \eqn{0 < x < 1} and \eqn{\theta = (\alpha, \beta)}.
#' The log-likelihood function \eqn{\ell(\theta | \mathbf{x})} for a sample
#' \eqn{\mathbf{x} = (x_1, \dots, x_n)} is \eqn{\sum_{i=1}^n \ln f(x_i | \theta)}:
#' \deqn{
#' \ell(\theta | \mathbf{x}) = n[\ln(\alpha) + \ln(\beta)]
#' + \sum_{i=1}^{n} [(\alpha-1)\ln(x_i) + (\beta-1)\ln(v_i)]
#' }
#' where \eqn{v_i = 1 - x_i^{\alpha}}.
#' This function computes and returns the *negative* log-likelihood, \eqn{-\ell(\theta|\mathbf{x})},
#' suitable for minimization using optimization routines like \code{\link[stats]{optim}}.
#' It is equivalent to the negative log-likelihood of the GKw distribution
#' (\code{\link{llgkw}}) evaluated at \eqn{\gamma=1, \delta=0, \lambda=1}.
#'
#' @references
#' Kumaraswamy, P. (1980). A generalized probability density function for
#' double-bounded random processes. *Journal of Hydrology*, *46*(1-2), 79-88.
#' \doi{10.1016/0022-1694(80)90036-0}
#'
#'
#' Jones, M. C. (2009). Kumaraswamy's distribution: A beta-type distribution
#' with some tractability advantages. *Statistical Methodology*, *6*(1), 70-81.
#' \doi{10.1016/j.stamet.2008.04.001}
#'
#'
#' @seealso
#' \code{\link{llgkw}} (parent distribution negative log-likelihood),
#' \code{\link{dkw}}, \code{\link{pkw}}, \code{\link{qkw}}, \code{\link{rkw}},
#' \code{\link{grkw}} (gradient),
#' \code{\link{hskw}} (Hessian),
#' \code{\link[stats]{optim}}
#'
#' @examples
#' set.seed(123)
#' x <- rkw(1000, alpha = 2, beta = 3)
#' par <- c(alpha = 2, beta = 3)
#'
#' ## llkw() is the negative log-likelihood, -sum(log f(x))
#' llkw(par, x)
#' -sum(dkw(x, alpha = 2, beta = 3, log = TRUE))
#'
#' ## Maximum likelihood: minimize llkw(), with grkw() as its gradient
#' start <- gkwgetstartvalues(x, family = "kw")
#' fit <- optim(start, llkw, grkw, data = x, method = "L-BFGS-B", lower = 1e-4)
#' fit$convergence  # 0: converged
#' fit$par
#' fit$value <= llkw(par, x)  # at least as good as the true values
#'
#' @export
llkw <- function(par, data) {
  # Input validation
  if (!is.numeric(par) || length(par) != 2) {
    stop("'par' must be a numeric vector of length 2")
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
  .Call("_gkwdist_llkw",
    as.numeric(par),
    as.numeric(data),
    PACKAGE = "gkwdist"
  )
}


# ----------------------------------------------------------------------------#
# 6. GRADIENT (grkw)
# ----------------------------------------------------------------------------#

#' @title Gradient of the Negative Log-Likelihood for the Kumaraswamy (Kw) Distribution
#' @author Lopes, J. E.
#' @family gradient functions
#' @concept kumaraswamy
#' @keywords distribution optimize
#'
#' @description
#' Computes the gradient vector (vector of first partial derivatives) of the
#' negative log-likelihood function for the two-parameter Kumaraswamy (Kw)
#' distribution with parameters \code{alpha} (\eqn{\alpha}) and \code{beta}
#' (\eqn{\beta}). This provides the analytical gradient often used for efficient
#' optimization via maximum likelihood estimation.
#'
#' @param par A numeric vector of length 2 containing the distribution parameters
#'   in the order: \code{alpha} (\eqn{\alpha > 0}), \code{beta} (\eqn{\beta > 0}).
#' @param data A numeric vector of observations. All values must be strictly
#'   between 0 and 1 (exclusive).
#'
#' @return Returns a numeric vector of length 2 containing the partial derivatives
#'   of the negative log-likelihood function \eqn{-\ell(\theta | \mathbf{x})} with
#'   respect to each parameter: \eqn{(-\partial \ell/\partial \alpha, -\partial \ell/\partial \beta)}.
#'   Returns a vector of \code{NaN} if any parameter values are invalid according
#'   to their constraints, or if any value in \code{data} is not in the
#'   interval (0, 1).
#'
#' @details
#' The components of the gradient vector of the negative log-likelihood
#' (\eqn{-\nabla \ell(\theta | \mathbf{x})}) for the Kw model are:
#'
#' \deqn{
#' -\frac{\partial \ell}{\partial \alpha} = -\frac{n}{\alpha} - \sum_{i=1}^{n}\ln(x_i)
#' + (\beta-1)\sum_{i=1}^{n}\frac{x_i^{\alpha}\ln(x_i)}{v_i}
#' }
#' \deqn{
#' -\frac{\partial \ell}{\partial \beta} = -\frac{n}{\beta} - \sum_{i=1}^{n}\ln(v_i)
#' }
#'
#' where \eqn{v_i = 1 - x_i^{\alpha}}.
#' These formulas represent the derivatives of \eqn{-\ell(\theta)}, consistent with
#' minimizing the negative log-likelihood. They correspond to the relevant components
#' of the general GKw gradient (\code{\link{grgkw}}) evaluated at \eqn{\gamma=1, \delta=0, \lambda=1}.
#'
#' @references
#' Kumaraswamy, P. (1980). A generalized probability density function for
#' double-bounded random processes. *Journal of Hydrology*, *46*(1-2), 79-88.
#' \doi{10.1016/0022-1694(80)90036-0}
#'
#'
#' Jones, M. C. (2009). Kumaraswamy's distribution: A beta-type distribution
#' with some tractability advantages. *Statistical Methodology*, *6*(1), 70-81.
#' \doi{10.1016/j.stamet.2008.04.001}
#'
#' (Note: Specific gradient formulas might be derived or sourced from additional references).
#'
#' @seealso
#' \code{\link{grgkw}} (parent distribution gradient),
#' \code{\link{llkw}} (negative log-likelihood for Kw),
#' \code{\link{hskw}} (Hessian for Kw),
#' \code{\link{dkw}} (density for Kw),
#' \code{\link[stats]{optim}},
#' \code{\link[numDeriv]{grad}} (for numerical gradient comparison).
#'
#' @examples
#' set.seed(123)
#' x <- rkw(200, alpha = 2, beta = 3)
#' par <- c(alpha = 2, beta = 3)
#'
#' ## Gradient of the negative log-likelihood llkw(), not of the log-likelihood
#' g <- grkw(par, x)
#' g
#'
#' ## A small step against the gradient lowers llkw()
#' llkw(par - 1e-4 * g, x) < llkw(par, x)
#'
#' ## Agrees with a numerical derivative of llkw()
#' if (requireNamespace("numDeriv", quietly = TRUE))
#'   all.equal(g, numDeriv::grad(llkw, par, data = x), tolerance = 1e-6)
#'
#' @export
grkw <- function(par, data) {
  # Input validation
  if (!is.numeric(par) || length(par) != 2) {
    stop("'par' must be a numeric vector of length 2")
  }
  if (!is.numeric(data)) {
    stop("'data' must be numeric")
  }
  if (length(data) < 1) {
    stop("'data' must have at least one observation")
  }

  # Call C++ implementation
  .Call("_gkwdist_grkw",
    as.numeric(par),
    as.numeric(data),
    PACKAGE = "gkwdist"
  )
}


# ----------------------------------------------------------------------------#
# 7. HESSIAN (hskw)
# ----------------------------------------------------------------------------#

#' @title Hessian Matrix of the Negative Log-Likelihood for the Kw Distribution
#' @author Lopes, J. E.
#' @family Hessian functions
#' @concept kumaraswamy
#' @keywords distribution optimize
#'
#' @description
#' Computes the analytic 2x2 Hessian matrix (matrix of second partial derivatives)
#' of the negative log-likelihood function for the two-parameter Kumaraswamy (Kw)
#' distribution with parameters \code{alpha} (\eqn{\alpha}) and \code{beta}
#' (\eqn{\beta}). The Hessian is useful for estimating standard errors and in
#' optimization algorithms.
#'
#' @param par A numeric vector of length 2 containing the distribution parameters
#'   in the order: \code{alpha} (\eqn{\alpha > 0}), \code{beta} (\eqn{\beta > 0}).
#' @param data A numeric vector of observations. All values must be strictly
#'   between 0 and 1 (exclusive).
#'
#' @return Returns a 2x2 numeric matrix representing the Hessian matrix of the
#'   negative log-likelihood function, \eqn{-\partial^2 \ell / (\partial \theta_i \partial \theta_j)},
#'   where \eqn{\theta = (\alpha, \beta)}.
#'   Returns a 2x2 matrix populated with \code{NaN} if any parameter values are
#'   invalid according to their constraints, or if any value in \code{data} is
#'   not in the interval (0, 1).
#'
#' @details
#' This function calculates the analytic second partial derivatives of the
#' negative log-likelihood function (\eqn{-\ell(\theta|\mathbf{x})}). The components
#' are the negative of the second derivatives of the log-likelihood \eqn{\ell}
#' (derived from the PDF in \code{\link{dkw}}).
#'
#' Let \eqn{v_i = 1 - x_i^{\alpha}}. The second derivatives of the positive log-likelihood (\eqn{\ell}) are:
#' \deqn{
#' \frac{\partial^2 \ell}{\partial \alpha^2} = -\frac{n}{\alpha^2} -
#' (\beta-1)\sum_{i=1}^{n}\frac{x_i^{\alpha}(\ln(x_i))^2}{v_i^2}
#' }
#' \deqn{
#' \frac{\partial^2 \ell}{\partial \alpha \partial \beta} = -
#' \sum_{i=1}^{n}\frac{x_i^{\alpha}\ln(x_i)}{v_i}
#' }
#' \deqn{
#' \frac{\partial^2 \ell}{\partial \beta^2} = -\frac{n}{\beta^2}
#' }
#' The function returns the Hessian matrix containing the negative of these values.
#'
#' Key properties of the returned matrix:
#' \itemize{
#'   \item Dimensions: 2x2.
#'   \item Symmetry: The matrix is symmetric.
#'   \item Ordering: Rows and columns correspond to the parameters in the order
#'     \eqn{\alpha, \beta}.
#'   \item Content: Analytic second derivatives of the *negative* log-likelihood.
#' }
#' This corresponds to the relevant 2x2 submatrix of the 5x5 GKw Hessian (\code{\link{hsgkw}})
#' evaluated at \eqn{\gamma=1, \delta=0, \lambda=1}.
#'
#' @references
#' Kumaraswamy, P. (1980). A generalized probability density function for
#' double-bounded random processes. *Journal of Hydrology*, *46*(1-2), 79-88.
#' \doi{10.1016/0022-1694(80)90036-0}
#'
#'
#' Jones, M. C. (2009). Kumaraswamy's distribution: A beta-type distribution
#' with some tractability advantages. *Statistical Methodology*, *6*(1), 70-81.
#' \doi{10.1016/j.stamet.2008.04.001}
#'
#' (Note: Specific Hessian formulas might be derived or sourced from additional references).
#'
#' @seealso
#' \code{\link{hsgkw}} (parent distribution Hessian),
#' \code{\link{llkw}} (negative log-likelihood for Kw),
#' \code{\link{grkw}} (gradient for Kw),
#' \code{\link{dkw}} (density for Kw),
#' \code{\link[stats]{optim}},
#' \code{\link[numDeriv]{hessian}} (for numerical Hessian comparison).
#'
#' @examples
#' set.seed(123)
#' x <- rkw(1000, alpha = 2, beta = 3)
#' par <- c(alpha = 2, beta = 3)
#'
#' ## Hessian of the negative log-likelihood llkw()
#' H <- hskw(par, x)
#' isSymmetric(H)
#'
#' ## Agrees with a numerical Hessian of llkw()
#' if (requireNamespace("numDeriv", quietly = TRUE))
#'   all.equal(H, numDeriv::hessian(llkw, par, data = x), tolerance = 1e-8)
#'
#' ## At the MLE it is the observed information; its inverse estimates the
#' ## covariance of the estimates
#' fit <- optim(par, llkw, grkw, data = x, method = "L-BFGS-B", lower = 1e-4)
#' sqrt(diag(solve(hskw(fit$par, x))))  # standard errors
#'
#' @export
hskw <- function(par, data) {
  # Input validation
  if (!is.numeric(par) || length(par) != 2) {
    stop("'par' must be a numeric vector of length 2")
  }
  if (!is.numeric(data)) {
    stop("'data' must be numeric")
  }
  if (length(data) < 1) {
    stop("'data' must have at least one observation")
  }

  # Call C++ implementation
  .Call("_gkwdist_hskw",
    as.numeric(par),
    as.numeric(data),
    PACKAGE = "gkwdist"
  )
}

# ============================================================================#
# END
# ============================================================================#
