# ============================================================================#
# MCDONALD (Mc) / BETA POWER DISTRIBUTION
# ============================================================================#
#
# Wrapper functions for the McDonald (Beta Power) distribution.
# C++ implementations are in src/mc.cpp
#
# Functions:
#   - dmc: Probability density function (PDF)
#   - pmc: Cumulative distribution function (CDF)
#   - qmc: Quantile function (inverse CDF)
#   - rmc: Random number generation
#   - llmc: Negative log-likelihood
#   - grmc: Gradient of negative log-likelihood
#   - hsmc: Hessian of negative log-likelihood
# ============================================================================#


# ----------------------------------------------------------------------------#
# 1. DENSITY FUNCTION (dmc)
# ----------------------------------------------------------------------------#

#' @title Density of the McDonald (Mc)/Beta Power Distribution
#' @author Lopes, J. E.
#' @family density functions
#' @concept mcdonald
#' @keywords distribution
#'
#' @description
#' Computes the probability density function (PDF) for the McDonald (Mc)
#' distribution (also previously referred to as Beta Power) with parameters
#' \code{gamma} (\eqn{\gamma}), \code{delta} (\eqn{\delta}), and \code{lambda}
#' (\eqn{\lambda}). This distribution is defined on the interval (0, 1).
#'
#' @param x Vector of quantiles (values between 0 and 1).
#' @param gamma Shape parameter \code{gamma} > 0. Can be a scalar or a vector.
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
#'   rule applied to the arguments (\code{x}, \code{gamma}, \code{delta},
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
#' The probability density function (PDF) of the McDonald (Mc) distribution
#' is given by:
#' \deqn{
#' f(x; \gamma, \delta, \lambda) = \frac{\lambda}{B(\gamma,\delta+1)} x^{\gamma \lambda - 1} (1 - x^\lambda)^\delta
#' }
#' for \eqn{0 < x < 1}, where \eqn{B(a,b)} is the Beta function
#' (\code{\link[base]{beta}}).
#'
#' The Mc distribution is a special case of the five-parameter
#' Generalized Kumaraswamy (GKw) distribution (\code{\link{dgkw}}) obtained
#' by setting the parameters \eqn{\alpha = 1} and \eqn{\beta = 1}.
#' It was introduced by McDonald (1984) and is related to the Generalized Beta
#' distribution of the first kind (GB1). When \eqn{\lambda=1}, it simplifies
#' to the standard Beta distribution with parameters \eqn{\gamma} and
#' \eqn{\delta+1}.
#'
#' @references
#' McDonald, J. B. (1984). Some generalized functions for the size distribution
#' of income. *Econometrica*, *52*(3), 647-663.
#' \doi{10.2307/1913469}
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
#'
#' @seealso
#' \code{\link{dgkw}} (parent distribution density),
#' \code{\link{pmc}}, \code{\link{qmc}}, \code{\link{rmc}} (other Mc functions),
#' \code{\link[stats]{dbeta}}
#'
#' @examples
#' x <- c(0.1, 0.3, 0.5, 0.7, 0.9)
#' dmc(x, gamma = 0.5, delta = 5, lambda = 3)
#' dmc(x, gamma = 0.5, delta = 5, lambda = 3, log = TRUE)
#'
#' ## Mc is GKw with alpha = beta = 1
#' all.equal(dmc(x, 0.5, 5, 3), dgkw(x, 1, 1, 0.5, 5, 3))
#'
#' ## The density integrates to one
#' integrate(dmc, 0, 1, gamma = 0.5, delta = 5, lambda = 3, rel.tol = 1e-10)
#'
#' curve(dmc(x, gamma = 0.5, delta = 5, lambda = 3), from = 0, to = 1, ylab = "density")
#'
#' @export
dmc <- function(x, gamma = 1, delta = 0, lambda = 1, log = FALSE) {
  # Input validation
  if (!is.numeric(x)) stop("'x' must be numeric")
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
  .shape_like(.Call("_gkwdist_dmc",
    as.numeric(x),
    as.numeric(gamma),
    as.numeric(delta),
    as.numeric(lambda),
    as.logical(log),
    PACKAGE = "gkwdist"
  ), x)
}


# ----------------------------------------------------------------------------#
# 2. DISTRIBUTION FUNCTION (pmc)
# ----------------------------------------------------------------------------#

#' @title Cumulative Distribution Function (CDF) of the McDonald (Mc)/Beta Power Distribution
#' @author Lopes, J. E.
#' @family cumulative distribution functions
#' @concept mcdonald
#' @keywords distribution
#'
#' @description
#' Computes the cumulative distribution function (CDF), \eqn{F(q) = P(X \le q)},
#' for the McDonald (Mc) distribution (also known as Beta Power) with
#' parameters \code{gamma} (\eqn{\gamma}), \code{delta} (\eqn{\delta}), and
#' \code{lambda} (\eqn{\lambda}). This distribution is defined on the interval
#' (0, 1) and is a special case of the Generalized Kumaraswamy (GKw)
#' distribution where \eqn{\alpha = 1} and \eqn{\beta = 1}.
#'
#' @param q Vector of quantiles (values generally between 0 and 1).
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
#'   \code{gamma}, \code{delta}, \code{lambda}). When \code{lower.tail = TRUE},
#'   returns \code{0} (or \code{-Inf} if \code{log.p = TRUE}) for \code{q <= 0}
#'   and \code{1} (or \code{0} if \code{log.p = TRUE}) for \code{q >= 1}. An
#'   out-of-bound or missing parameter is an error, not a return value: the
#'   wrapper stops with a message naming the parameter. An infinite parameter
#'   is not currently intercepted there and reaches the C++ layer, which
#'   treats it as invalid.
#'   Boundary return values are adjusted accordingly for \code{lower.tail = FALSE}.
#'
#' @details
#' The McDonald (Mc) distribution is a special case of the five-parameter
#' Generalized Kumaraswamy (GKw) distribution (\code{\link{pgkw}}) obtained
#' by setting parameters \eqn{\alpha = 1} and \eqn{\beta = 1}.
#'
#' The CDF of the GKw distribution is \eqn{F_{GKw}(q) = I_{y(q)}(\gamma, \delta+1)},
#' where \eqn{y(q) = [1-(1-q^{\alpha})^{\beta}]^{\lambda}} and \eqn{I_x(a,b)}
#' is the regularized incomplete beta function (\code{\link[stats]{pbeta}}).
#' Setting \eqn{\alpha=1} and \eqn{\beta=1} simplifies \eqn{y(q)} to \eqn{q^\lambda},
#' yielding the Mc CDF:
#' \deqn{
#' F(q; \gamma, \delta, \lambda) = I_{q^\lambda}(\gamma, \delta+1)
#' }
#' This is evaluated using the \code{\link[stats]{pbeta}} function as
#' \code{pbeta(q^lambda, shape1 = gamma, shape2 = delta + 1)}.
#'
#' @references
#' McDonald, J. B. (1984). Some generalized functions for the size distribution
#' of income. *Econometrica*, *52*(3), 647-663.
#' \doi{10.2307/1913469}
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
#'
#' @seealso
#' \code{\link{pgkw}} (parent distribution CDF),
#' \code{\link{dmc}}, \code{\link{qmc}}, \code{\link{rmc}} (other Mc functions),
#' \code{\link[stats]{pbeta}}
#'
#' @examples
#' q <- c(0.2, 0.5, 0.8)
#' pmc(q, gamma = 0.5, delta = 5, lambda = 3)
#' pmc(q, gamma = 0.5, delta = 5, lambda = 3, lower.tail = FALSE)  # P(X > q)
#' pmc(q, gamma = 0.5, delta = 5, lambda = 3, log.p = TRUE)
#'
#' ## pmc() is the integral of dmc()
#' Fq <- pmc(0.5, gamma = 0.5, delta = 5, lambda = 3)
#' all.equal(Fq, integrate(dmc, 0, 0.5, gamma = 0.5, delta = 5, lambda = 3,
#'     rel.tol = 1e-10)$value)
#'
#' @export
pmc <- function(q, gamma = 1, delta = 0, lambda = 1, lower.tail = TRUE, log.p = FALSE) {
  # Input validation
  if (!is.numeric(q)) stop("'q' must be numeric")
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
  .shape_like(.Call("_gkwdist_pmc",
    as.numeric(q),
    as.numeric(gamma),
    as.numeric(delta),
    as.numeric(lambda),
    as.logical(lower.tail),
    as.logical(log.p),
    PACKAGE = "gkwdist"
  ), q)
}


# ----------------------------------------------------------------------------#
# 3. QUANTILE FUNCTION (qmc)
# ----------------------------------------------------------------------------#

#' @title Quantile Function of the McDonald (Mc)/Beta Power Distribution
#' @author Lopes, J. E.
#' @family quantile functions
#' @concept mcdonald
#' @keywords distribution
#'
#' @description
#' Computes the quantile function (inverse CDF) for the McDonald (Mc) distribution
#' (also known as Beta Power) with parameters \code{gamma} (\eqn{\gamma}),
#' \code{delta} (\eqn{\delta}), and \code{lambda} (\eqn{\lambda}). It finds the
#' value \code{q} such that \eqn{P(X \le q) = p}. This distribution is a special
#' case of the Generalized Kumaraswamy (GKw) distribution where \eqn{\alpha = 1}
#' and \eqn{\beta = 1}.
#'
#' @param p Vector of probabilities (values between 0 and 1).
#' @param gamma Shape parameter \code{gamma} > 0. Can be a scalar or a vector.
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
#'   the arguments (\code{p}, \code{gamma}, \code{delta}, \code{lambda}).
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
#' for the Mc (\eqn{\alpha=1, \beta=1}) distribution is \eqn{F(q) = I_{q^\lambda}(\gamma, \delta+1)},
#' where \eqn{I_z(a,b)} is the regularized incomplete beta function (see \code{\link{pmc}}).
#'
#' To find the quantile \eqn{q}, we first invert the Beta function part: let
#' \eqn{y = I^{-1}_{p}(\gamma, \delta+1)}, where \eqn{I^{-1}_p(a,b)} is the
#' inverse computed via \code{\link[stats]{qbeta}}. We then solve \eqn{q^\lambda = y}
#' for \eqn{q}, yielding the quantile function:
#' \deqn{
#' Q(p) = \left[ I^{-1}_{p}(\gamma, \delta+1) \right]^{1/\lambda}
#' }
#' The function uses this formula, calculating \eqn{I^{-1}_{p}(\gamma, \delta+1)}
#' via \code{qbeta(p, gamma, delta + 1, ...)} while respecting the
#' \code{lower.tail} and \code{log.p} arguments. This is equivalent to the general
#' GKw quantile function (\code{\link{qgkw}}) evaluated with \eqn{\alpha=1, \beta=1}.
#'
#' @references
#' McDonald, J. B. (1984). Some generalized functions for the size distribution
#' of income. *Econometrica*, *52*(3), 647-663.
#' \doi{10.2307/1913469}
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
#'
#' @seealso
#' \code{\link{qgkw}} (parent distribution quantile function),
#' \code{\link{dmc}}, \code{\link{pmc}}, \code{\link{rmc}} (other Mc functions),
#' \code{\link[stats]{qbeta}}
#'
#' @examples
#' p <- c(0.1, 0.5, 0.9)
#' qmc(p, gamma = 0.5, delta = 5, lambda = 3)
#' # upper-tail quantiles
#' qmc(p, gamma = 0.5, delta = 5, lambda = 3, lower.tail = FALSE)
#'
#' ## qmc() inverts pmc()
#' all.equal(pmc(qmc(p, gamma = 0.5, delta = 5, lambda = 3), gamma = 0.5,
#'     delta = 5, lambda = 3), p)
#'
#' @export
qmc <- function(p, gamma = 1, delta = 0, lambda = 1, lower.tail = TRUE, log.p = FALSE) {
  # Input validation
  if (!is.numeric(p)) stop("'p' must be numeric")
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
  .shape_like(.Call("_gkwdist_qmc",
    as.numeric(p),
    as.numeric(gamma),
    as.numeric(delta),
    as.numeric(lambda),
    as.logical(lower.tail),
    as.logical(log.p),
    PACKAGE = "gkwdist"
  ), p)
}


# ----------------------------------------------------------------------------#
# 4. RANDOM GENERATION (rmc)
# ----------------------------------------------------------------------------#

#' @title Random Number Generation for the McDonald (Mc)/Beta Power Distribution
#' @author Lopes, J. E.
#' @family random generation functions
#' @concept mcdonald
#' @keywords distribution
#'
#' @description
#' Generates random deviates from the McDonald (Mc) distribution (also known as
#' Beta Power) with parameters \code{gamma} (\eqn{\gamma}), \code{delta}
#' (\eqn{\delta}), and \code{lambda} (\eqn{\lambda}). This distribution is a
#' special case of the Generalized Kumaraswamy (GKw) distribution where
#' \eqn{\alpha = 1} and \eqn{\beta = 1}.
#'
#' @param n Number of observations. If \code{length(n) > 1}, the length is
#'   taken to be the number required. Must be a non-negative integer.
#' @param gamma Shape parameter \code{gamma} > 0. Can be a scalar or a vector.
#'   Default: 1.0.
#' @param delta Shape parameter \code{delta} >= 0. Can be a scalar or a vector.
#'   Default: 0.0.
#' @param lambda Shape parameter \code{lambda} > 0. Can be a scalar or a vector.
#'   Default: 1.0.
#'
#' @return A vector of length \code{n} containing random deviates from the Mc
#'   distribution, with values in (0, 1). The length of the result is determined
#'   by \code{n} and the recycling rule applied to the parameters (\code{gamma},
#'   \code{delta}, \code{lambda}). An out-of-bound or missing parameter is an
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
#' For the Mc distribution, \eqn{\alpha=1} and \eqn{\beta=1}. Therefore, the
#' algorithm simplifies significantly:
#' \enumerate{
#'   \item Generate \eqn{U \sim \mathrm{Beta}(\gamma, \delta+1)} using
#'         \code{\link[stats]{rbeta}}.
#'   \item Compute the Mc variate \eqn{X = U^{1/\lambda}}.
#' }
#' This procedure is implemented efficiently, handling parameter recycling as needed.
#'
#' @references
#' McDonald, J. B. (1984). Some generalized functions for the size distribution
#' of income. *Econometrica*, *52*(3), 647-663.
#' \doi{10.2307/1913469}
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
#'
#' Devroye, L. (1986). *Non-Uniform Random Variate Generation*. Springer-Verlag.
#' (General methods for random variate generation).
#'
#' @seealso
#' \code{\link{rgkw}} (parent distribution random generation),
#' \code{\link{dmc}}, \code{\link{pmc}}, \code{\link{qmc}} (other Mc functions),
#' \code{\link[stats]{rbeta}}
#'
#' @examples
#' set.seed(123)
#' x <- rmc(1000, gamma = 0.5, delta = 5, lambda = 3)
#' summary(x)
#'
#' ## The sample follows the distribution
#' hist(x, breaks = 30, freq = FALSE, main = "")
#' curve(dmc(x, gamma = 0.5, delta = 5, lambda = 3), add = TRUE)
#' ks.test(x, pmc, gamma = 0.5, delta = 5, lambda = 3)
#'
#' @export
rmc <- function(n, gamma = 1, delta = 0, lambda = 1) {
  # Input validation
  if (length(n) > 1) n <- length(n)
  # n = 0 is legal and yields numeric(0), as stats::rbeta(0, 2, 3) does.
  # Negative and missing n stay errors, which is also what base R does.
  if (!is.numeric(n) || length(n) != 1 || is.na(n) || n < 0) {
    stop("'n' must be a single non-negative integer")
  }
  n <- as.integer(n)

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
  .Call("_gkwdist_rmc",
    as.integer(n),
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
# 5. NEGATIVE LOG-LIKELIHOOD (llmc)
# ----------------------------------------------------------------------------#

#' @title Negative Log-Likelihood for the McDonald (Mc)/Beta Power Distribution
#' @author Lopes, J. E.
#' @family log-likelihood functions
#' @concept mcdonald
#' @keywords distribution optimize
#'
#' @description
#' Computes the negative log-likelihood function for the McDonald (Mc)
#' distribution (also known as Beta Power) with parameters \code{gamma}
#' (\eqn{\gamma}), \code{delta} (\eqn{\delta}), and \code{lambda} (\eqn{\lambda}),
#' given a vector of observations. This distribution is the special case of the
#' Generalized Kumaraswamy (GKw) distribution where \eqn{\alpha = 1} and
#' \eqn{\beta = 1}. This function is suitable for maximum likelihood estimation.
#'
#' @param par A numeric vector of length 3 containing the distribution parameters
#'   in the order: \code{gamma} (\eqn{\gamma > 0}), \code{delta} (\eqn{\delta \ge 0}),
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
#' The McDonald (Mc) distribution is the GKw distribution (\code{\link{dmc}})
#' with \eqn{\alpha=1} and \eqn{\beta=1}. Its probability density function (PDF) is:
#' \deqn{
#' f(x | \theta) = \frac{\lambda}{B(\gamma,\delta+1)} x^{\gamma \lambda - 1} (1 - x^\lambda)^\delta
#' }
#' for \eqn{0 < x < 1}, \eqn{\theta = (\gamma, \delta, \lambda)}, and \eqn{B(a,b)}
#' is the Beta function (\code{\link[base]{beta}}).
#' The log-likelihood function \eqn{\ell(\theta | \mathbf{x})} for a sample
#' \eqn{\mathbf{x} = (x_1, \dots, x_n)} is \eqn{\sum_{i=1}^n \ln f(x_i | \theta)}:
#' \deqn{
#' \ell(\theta | \mathbf{x}) = n[\ln(\lambda) - \ln B(\gamma, \delta+1)]
#' + \sum_{i=1}^{n} [(\gamma\lambda - 1)\ln(x_i) + \delta\ln(1 - x_i^\lambda)]
#' }
#' This function computes and returns the *negative* log-likelihood, \eqn{-\ell(\theta|\mathbf{x})},
#' suitable for minimization using optimization routines like \code{\link[stats]{optim}}.
#' Numerical stability is maintained, including using the log-gamma function
#' (\code{\link[base]{lgamma}}) for the Beta function term.
#'
#' @references
#' McDonald, J. B. (1984). Some generalized functions for the size distribution
#' of income. *Econometrica*, *52*(3), 647-663.
#' \doi{10.2307/1913469}
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
#'
#' @seealso
#' \code{\link{llgkw}} (parent distribution negative log-likelihood),
#' \code{\link{dmc}}, \code{\link{pmc}}, \code{\link{qmc}}, \code{\link{rmc}},
#' \code{\link{grmc}} (gradient),
#' \code{\link{hsmc}} (Hessian),
#' \code{\link[stats]{optim}}, \code{\link[base]{lbeta}}
#'
#' @examples
#' set.seed(123)
#' x <- rmc(1000, gamma = 0.5, delta = 5, lambda = 3)
#' par <- c(gamma = 0.5, delta = 5, lambda = 3)
#'
#' ## llmc() is the negative log-likelihood, -sum(log f(x))
#' llmc(par, x)
#' -sum(dmc(x, gamma = 0.5, delta = 5, lambda = 3, log = TRUE))
#'
#' ## Maximum likelihood: minimize llmc(), with grmc() as its gradient
#' start <- gkwgetstartvalues(x, family = "mc")
#' fit <- optim(start, llmc, grmc, data = x, method = "L-BFGS-B", lower = 1e-4)
#' fit$convergence  # 0: converged
#' fit$par
#' fit$value <= llmc(par, x)  # at least as good as the true values
#'
#' @export
llmc <- function(par, data) {
  # Input validation
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

  # Call C++ implementation
  .Call("_gkwdist_llmc",
    as.numeric(par),
    as.numeric(data),
    PACKAGE = "gkwdist"
  )
}


# ----------------------------------------------------------------------------#
# 6. GRADIENT (grmc)
# ----------------------------------------------------------------------------#

#' @title Gradient of the Negative Log-Likelihood for the McDonald (Mc)/Beta Power Distribution
#' @author Lopes, J. E.
#' @family gradient functions
#' @concept mcdonald
#' @keywords distribution optimize
#'
#' @description
#' Computes the gradient vector (vector of first partial derivatives) of the
#' negative log-likelihood function for the McDonald (Mc) distribution (also
#' known as Beta Power) with parameters \code{gamma} (\eqn{\gamma}), \code{delta}
#' (\eqn{\delta}), and \code{lambda} (\eqn{\lambda}). This distribution is the
#' special case of the Generalized Kumaraswamy (GKw) distribution where
#' \eqn{\alpha = 1} and \eqn{\beta = 1}. The gradient is useful for optimization.
#'
#' @param par A numeric vector of length 3 containing the distribution parameters
#'   in the order: \code{gamma} (\eqn{\gamma > 0}), \code{delta} (\eqn{\delta \ge 0}),
#'   \code{lambda} (\eqn{\lambda > 0}).
#' @param data A numeric vector of observations. All values must be strictly
#'   between 0 and 1 (exclusive).
#'
#' @return Returns a numeric vector of length 3 containing the partial derivatives
#'   of the negative log-likelihood function \eqn{-\ell(\theta | \mathbf{x})} with
#'   respect to each parameter:
#'   \eqn{(-\partial \ell/\partial \gamma, -\partial \ell/\partial \delta, -\partial \ell/\partial \lambda)}.
#'   Returns a vector of \code{NaN} if any parameter values are invalid according
#'   to their constraints, or if any value in \code{data} is not in the
#'   interval (0, 1).
#'
#' @details
#' The components of the gradient vector of the negative log-likelihood
#' (\eqn{-\nabla \ell(\theta | \mathbf{x})}) for the Mc (\eqn{\alpha=1, \beta=1})
#' model are:
#'
#' \deqn{
#' -\frac{\partial \ell}{\partial \gamma} = n[\psi(\gamma) - \psi(\gamma+\delta+1)] -
#' \lambda\sum_{i=1}^{n}\ln(x_i)
#' }
#' \deqn{
#' -\frac{\partial \ell}{\partial \delta} = n[\psi(\delta+1) - \psi(\gamma+\delta+1)] -
#' \sum_{i=1}^{n}\ln(1-x_i^{\lambda})
#' }
#' \deqn{
#' -\frac{\partial \ell}{\partial \lambda} = -\frac{n}{\lambda} - \gamma\sum_{i=1}^{n}\ln(x_i) +
#' \delta\sum_{i=1}^{n}\frac{x_i^{\lambda}\ln(x_i)}{1-x_i^{\lambda}}
#' }
#'
#' where \eqn{\psi(\cdot)} is the digamma function (\code{\link[base]{digamma}}).
#' These formulas represent the derivatives of \eqn{-\ell(\theta)}, consistent with
#' minimizing the negative log-likelihood. They correspond to the relevant components
#' of the general GKw gradient (\code{\link{grgkw}}) evaluated at \eqn{\alpha=1, \beta=1}.
#'
#' @references
#' McDonald, J. B. (1984). Some generalized functions for the size distribution
#' of income. *Econometrica*, *52*(3), 647-663.
#' \doi{10.2307/1913469}
#'
#' Cordeiro, G. M., & de Castro, M. (2011). A new family of generalized
#' distributions. *Journal of Statistical Computation and Simulation*,
#' *81*(7), 883-898.
#' \doi{10.1080/00949650903530745}
#'
#' (Note: Specific gradient formulas might be derived or sourced from additional references).
#'
#' @seealso
#' \code{\link{grgkw}} (parent distribution gradient),
#' \code{\link{llmc}} (negative log-likelihood for Mc),
#' \code{\link{hsmc}} (Hessian for Mc),
#' \code{\link{dmc}} (density for Mc),
#' \code{\link[stats]{optim}},
#' \code{\link[numDeriv]{grad}} (for numerical gradient comparison),
#' \code{\link[base]{digamma}}.
#'
#' @examples
#' set.seed(123)
#' x <- rmc(200, gamma = 0.5, delta = 5, lambda = 3)
#' par <- c(gamma = 0.5, delta = 5, lambda = 3)
#'
#' ## Gradient of the negative log-likelihood llmc(), not of the log-likelihood
#' g <- grmc(par, x)
#' g
#'
#' ## A small step against the gradient lowers llmc()
#' llmc(par - 1e-4 * g, x) < llmc(par, x)
#'
#' ## Agrees with a numerical derivative of llmc()
#' if (requireNamespace("numDeriv", quietly = TRUE))
#'   all.equal(g, numDeriv::grad(llmc, par, data = x), tolerance = 1e-6)
#'
#' @export
grmc <- function(par, data) {
  # Input validation
  if (!is.numeric(par) || length(par) != 3) {
    stop("'par' must be a numeric vector of length 3")
  }
  if (!is.numeric(data)) {
    stop("'data' must be numeric")
  }
  if (length(data) < 1) {
    stop("'data' must have at least one observation")
  }

  # Call C++ implementation
  .Call("_gkwdist_grmc",
    as.numeric(par),
    as.numeric(data),
    PACKAGE = "gkwdist"
  )
}


# ----------------------------------------------------------------------------#
# 7. HESSIAN (hsmc)
# ----------------------------------------------------------------------------#

#' @title Hessian Matrix of the Negative Log-Likelihood for the McDonald (Mc)/Beta Power Distribution
#' @author Lopes, J. E.
#' @family Hessian functions
#' @concept mcdonald
#' @keywords distribution optimize
#'
#' @description
#' Computes the analytic 3x3 Hessian matrix (matrix of second partial derivatives)
#' of the negative log-likelihood function for the McDonald (Mc) distribution
#' (also known as Beta Power) with parameters \code{gamma} (\eqn{\gamma}),
#' \code{delta} (\eqn{\delta}), and \code{lambda} (\eqn{\lambda}). This distribution
#' is the special case of the Generalized Kumaraswamy (GKw) distribution where
#' \eqn{\alpha = 1} and \eqn{\beta = 1}. The Hessian is useful for estimating
#' standard errors and in optimization algorithms.
#'
#' @param par A numeric vector of length 3 containing the distribution parameters
#'   in the order: \code{gamma} (\eqn{\gamma > 0}), \code{delta} (\eqn{\delta \ge 0}),
#'   \code{lambda} (\eqn{\lambda > 0}).
#' @param data A numeric vector of observations. All values must be strictly
#'   between 0 and 1 (exclusive).
#'
#' @return Returns a 3x3 numeric matrix representing the Hessian matrix of the
#'   negative log-likelihood function, \eqn{-\partial^2 \ell / (\partial \theta_i \partial \theta_j)},
#'   where \eqn{\theta = (\gamma, \delta, \lambda)}.
#'   Returns a 3x3 matrix populated with \code{NaN} if any parameter values are
#'   invalid according to their constraints, or if any value in \code{data} is
#'   not in the interval (0, 1).
#'
#' @details
#' This function calculates the analytic second partial derivatives of the
#' negative log-likelihood function (\eqn{-\ell(\theta|\mathbf{x})}).
#' The components are based on the second derivatives of the log-likelihood \eqn{\ell}
#' (derived from the PDF in \code{\link{dmc}}).
#'
#' **Note:** The formulas below represent the second derivatives of the positive
#' log-likelihood (\eqn{\ell}). The function returns the **negative** of these values.
#'
#' \deqn{
#' \frac{\partial^2 \ell}{\partial \gamma^2} = -n[\psi'(\gamma) - \psi'(\gamma+\delta+1)]
#' }
#' \deqn{
#' \frac{\partial^2 \ell}{\partial \gamma \partial \delta} = n\psi'(\gamma+\delta+1)
#' }
#' \deqn{
#' \frac{\partial^2 \ell}{\partial \gamma \partial \lambda} = \sum_{i=1}^{n}\ln(x_i)
#' }
#' \deqn{
#' \frac{\partial^2 \ell}{\partial \delta^2} = -n[\psi'(\delta+1) - \psi'(\gamma+\delta+1)]
#' }
#' \deqn{
#' \frac{\partial^2 \ell}{\partial \delta \partial \lambda} = -\sum_{i=1}^{n}\frac{x_i^{\lambda}\ln(x_i)}{1-x_i^{\lambda}}
#' }
#' \deqn{
#' \frac{\partial^2 \ell}{\partial \lambda^2} = -\frac{n}{\lambda^2} -
#' \delta\sum_{i=1}^{n}\frac{x_i^{\lambda}[\ln(x_i)]^2}{(1-x_i^{\lambda})^2}
#' }
#'
#' where \eqn{\psi'(\cdot)} is the trigamma function (\code{\link[base]{trigamma}}).
#' The \eqn{\partial^2 \ell / \partial \lambda^2} term matches the C++ implementation
#' (\code{src/bpmc.cpp}) and is covered by the numerical Hessian checks in
#' \code{tests/testthat/test-derivatives-validation.R}.
#'
#' The returned matrix is symmetric, with rows/columns corresponding to
#' \eqn{\gamma, \delta, \lambda}.
#'
#' @references
#' McDonald, J. B. (1984). Some generalized functions for the size distribution
#' of income. *Econometrica*, *52*(3), 647-663.
#' \doi{10.2307/1913469}
#'
#' Cordeiro, G. M., & de Castro, M. (2011). A new family of generalized
#' distributions. *Journal of Statistical Computation and Simulation*,
#' *81*(7), 883-898.
#' \doi{10.1080/00949650903530745}
#'
#' (Note: Specific Hessian formulas might be derived or sourced from additional references).
#'
#' @seealso
#' \code{\link{hsgkw}} (parent distribution Hessian),
#' \code{\link{llmc}} (negative log-likelihood for Mc),
#' \code{\link{grmc}} (gradient for Mc),
#' \code{\link{dmc}} (density for Mc),
#' \code{\link[stats]{optim}},
#' \code{\link[numDeriv]{hessian}} (for numerical Hessian comparison),
#' \code{\link[base]{trigamma}}.
#'
#' @examples
#' set.seed(123)
#' x <- rmc(1000, gamma = 0.5, delta = 5, lambda = 3)
#' par <- c(gamma = 0.5, delta = 5, lambda = 3)
#'
#' ## Hessian of the negative log-likelihood llmc()
#' H <- hsmc(par, x)
#' isSymmetric(H)
#'
#' ## Agrees with a numerical Hessian of llmc()
#' if (requireNamespace("numDeriv", quietly = TRUE))
#'   all.equal(H, numDeriv::hessian(llmc, par, data = x), tolerance = 1e-8)
#'
#' ## At the MLE it is the observed information; its inverse estimates the
#' ## covariance of the estimates
#' fit <- optim(par, llmc, grmc, data = x, method = "L-BFGS-B", lower = 1e-4)
#' sqrt(diag(solve(hsmc(fit$par, x))))  # standard errors
#'
#' @export
hsmc <- function(par, data) {
  # Input validation
  if (!is.numeric(par) || length(par) != 3) {
    stop("'par' must be a numeric vector of length 3")
  }
  if (!is.numeric(data)) {
    stop("'data' must be numeric")
  }
  if (length(data) < 1) {
    stop("'data' must have at least one observation")
  }

  # Call C++ implementation
  .Call("_gkwdist_hsmc",
    as.numeric(par),
    as.numeric(data),
    PACKAGE = "gkwdist"
  )
}
