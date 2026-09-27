# ============================================================================#
# BETA DISTRIBUTION (gamma, delta+1 Parameterization)
# ============================================================================#
#
# Wrapper functions for the Beta distribution using the parameterization
# common in the Generalized Kumaraswamy family.
#
# IMPORTANT PARAMETERIZATION NOTE:
# This implementation uses gamma (γ) and delta (δ) parameters, where:
#   shape1 = gamma
#   shape2 = delta + 1
#
# This differs from stats::dbeta(x, shape1, shape2) where shape2 is used directly.
#
# Relationship to stats::Beta:
#   dbeta_(x, gamma, delta) ≡ dbeta(x, shape1 = gamma, shape2 = delta + 1)
#
# C++ implementations are in src/beta.cpp
#
# Functions:
#   - dbeta_: Probability density function (PDF)
#   - pbeta_: Cumulative distribution function (CDF)
#   - qbeta_: Quantile function (inverse CDF)
#   - rbeta_: Random number generation
#   - llbeta: Negative log-likelihood
#   - grbeta: Gradient of negative log-likelihood
#   - hsbeta: Hessian of negative log-likelihood
# ============================================================================#


# ----------------------------------------------------------------------------#
# 1. DENSITY FUNCTION (dbeta_)
# ----------------------------------------------------------------------------#

#' @title Density of the Beta Distribution (gamma, delta+1 Parameterization)
#' @author Lopes, J. E.
#' @family density functions
#' @concept beta
#' @keywords distribution
#'
#' @description
#' Computes the probability density function (PDF) for the standard Beta
#' distribution, using a parameterization common in generalized distribution
#' families. The distribution is parameterized by \code{gamma} (\eqn{\gamma}) and
#' \code{delta} (\eqn{\delta}), corresponding to the standard Beta distribution
#' with shape parameters \code{shape1 = gamma} and \code{shape2 = delta + 1}.
#' The distribution is defined on the interval (0, 1).
#'
#' @param x Vector of quantiles (values between 0 and 1).
#' @param gamma First shape parameter (\code{shape1}), \eqn{\gamma > 0}. Can be a
#'   scalar or a vector. Default: 1.0.
#' @param delta Second shape parameter is \code{delta + 1} (\code{shape2}), requires
#'   \eqn{\delta \ge 0} so that \code{shape2 >= 1}. Can be a scalar or a vector.
#'   Default: 0.0 (leading to \code{shape2 = 1}).
#' @param log Logical; if \code{TRUE}, the logarithm of the density is
#'   returned (\eqn{\log(f(x))}). Default: \code{FALSE}.
#'
#' @return A vector of density values (\eqn{f(x)}) or log-density values
#'   (\eqn{\log(f(x))}). The length of the result is determined by the recycling
#'   rule applied to the arguments (\code{x}, \code{gamma}, \code{delta}).
#'   Returns \code{0} (or \code{-Inf} if \code{log = TRUE}) for \code{x}
#'   strictly outside the interval \[0, 1\]. At the closed boundaries
#'   \code{x = 0} and \code{x = 1} the limiting density is returned rather than
#'   \code{0}, matching \code{\link[stats]{dbeta}} with
#'   \code{shape1 = gamma} and \code{shape2 = delta + 1}; depending on the
#'   parameters that limit is \code{0}, a finite positive value, or \code{Inf}.
#'   An out-of-bound or missing parameter is an error, not a return value: the
#'   wrapper stops with a message naming the parameter. An infinite parameter is
#'   not currently intercepted there and reaches the C++ layer, which treats it
#'   as invalid.
#'
#' @details
#' The probability density function (PDF) calculated by this function corresponds
#' to a standard Beta distribution \eqn{Beta(\gamma, \delta+1)}:
#' \deqn{
#' f(x; \gamma, \delta) = \frac{x^{\gamma-1} (1-x)^{(\delta+1)-1}}{B(\gamma, \delta+1)} = \frac{x^{\gamma-1} (1-x)^{\delta}}{B(\gamma, \delta+1)}
#' }
#' for \eqn{0 < x < 1}, where \eqn{B(a,b)} is the Beta function
#' (\code{\link[base]{beta}}).
#'
#' This specific parameterization arises as a special case of the five-parameter
#' Generalized Kumaraswamy (GKw) distribution (\code{\link{dgkw}}) obtained
#' by setting the parameters \eqn{\alpha = 1}, \eqn{\beta = 1}, and \eqn{\lambda = 1}.
#' It is therefore equivalent to the McDonald (Mc)/Beta Power distribution
#' (\code{\link{dmc}}) with \eqn{\lambda = 1}.
#'
#' Note the difference in the second parameter compared to \code{\link[stats]{dbeta}},
#' where \code{dbeta(x, shape1, shape2)} uses \code{shape2} directly. Here,
#' \code{shape1 = gamma} and \code{shape2 = delta + 1}.
#'
#' @references
#' Johnson, N. L., Kotz, S., & Balakrishnan, N. (1995). *Continuous Univariate
#' Distributions, Volume 2* (2nd ed.). Wiley.
#'
#' Cordeiro, G. M., & de Castro, M. (2011). A new family of generalized
#' distributions. *Journal of Statistical Computation and Simulation*,
#' *81*(7), 883-898.
#' \doi{10.1080/00949650903530745}
#'
#'
#' @seealso
#' \code{\link[stats]{dbeta}} (standard R implementation),
#' \code{\link{dgkw}} (parent distribution density),
#' \code{\link{dmc}} (McDonald/Beta Power density),
#' \code{\link{pbeta_}}, \code{\link{qbeta_}}, \code{\link{rbeta_}}.
#'
#' @examples
#' x <- c(0.1, 0.3, 0.5, 0.7, 0.9)
#' dbeta_(x, gamma = 2, delta = 3)
#' dbeta_(x, gamma = 2, delta = 3, log = TRUE)
#'
#' ## The package's Beta(gamma, delta) is stats::dbeta with shapes gamma, delta + 1
#' all.equal(dbeta_(x, 2, 3), stats::dbeta(x, 2, 4))
#'
#' ## The density integrates to one
#' integrate(dbeta_, 0, 1, gamma = 2, delta = 3, rel.tol = 1e-10)
#'
#' curve(dbeta_(x, gamma = 2, delta = 3), from = 0, to = 1, ylab = "density")
#'
#' @export
dbeta_ <- function(x, gamma = 1, delta = 0, log = FALSE) {
  # Input validation
  if (!is.numeric(x)) {
    stop("'x' must be numeric")
  }
  if (!is.numeric(gamma) || anyNA(gamma) || any(gamma <= 0)) {
    stop("'gamma' must be positive (gamma > 0)")
  }
  if (!is.numeric(delta) || anyNA(delta) || any(delta < 0)) {
    stop("'delta' must be non-negative (delta >= 0)")
  }
  if (!is.logical(log) || length(log) != 1 || is.na(log)) {
    stop("'log' must be a single logical value")
  }

  # Informative warning about parameterization
  if (getOption("gkwdist.warn.beta.param", default = FALSE)) {
    message(
      "Note: dbeta_(x, gamma, delta) uses shape1=gamma, shape2=delta+1\n",
      "Equivalent to: dbeta(x, shape1=", gamma[1], ", shape2=", delta[1] + 1, ")\n",
      "Set options(gkwdist.warn.beta.param=FALSE) to suppress this message."
    )
  }

  # Call C++ implementation
  .shape_like(.Call("_gkwdist_dbeta_",
    as.numeric(x),
    as.numeric(gamma),
    as.numeric(delta),
    as.logical(log),
    PACKAGE = "gkwdist"
  ), x)
}


# ----------------------------------------------------------------------------#
# 2. DISTRIBUTION FUNCTION (pbeta_)
# ----------------------------------------------------------------------------#

#' @title Cumulative Distribution Function (CDF) of the Beta Distribution (gamma, delta+1 Parameterization)
#' @author Lopes, J. E.
#' @family cumulative distribution functions
#' @concept beta
#' @keywords distribution
#'
#' @description
#' Computes the cumulative distribution function (CDF), \eqn{F(q) = P(X \le q)},
#' for the standard Beta distribution, using a parameterization common in
#' generalized distribution families. The distribution is parameterized by
#' \code{gamma} (\eqn{\gamma}) and \code{delta} (\eqn{\delta}), corresponding to
#' the standard Beta distribution with shape parameters \code{shape1 = gamma}
#' and \code{shape2 = delta + 1}.
#'
#' @param q Vector of quantiles (values generally between 0 and 1).
#' @param gamma First shape parameter (\code{shape1}), \eqn{\gamma > 0}. Can be a
#'   scalar or a vector. Default: 1.0.
#' @param delta Second shape parameter is \code{delta + 1} (\code{shape2}), requires
#'   \eqn{\delta \ge 0} so that \code{shape2 >= 1}. Can be a scalar or a vector.
#'   Default: 0.0 (leading to \code{shape2 = 1}).
#' @param lower.tail Logical; if \code{TRUE} (default), probabilities are
#'   \eqn{P(X \le q)}, otherwise, \eqn{P(X > q)}.
#' @param log.p Logical; if \code{TRUE}, probabilities \eqn{p} are given as
#'   \eqn{\log(p)}. Default: \code{FALSE}.
#'
#' @return A vector of probabilities, \eqn{F(q)}, or their logarithms/complements
#'   depending on \code{lower.tail} and \code{log.p}. The length of the result
#'   is determined by the recycling rule applied to the arguments (\code{q},
#'   \code{gamma}, \code{delta}). When \code{lower.tail = TRUE}, returns
#'   \code{0} (or \code{-Inf} if \code{log.p = TRUE}) for \code{q <= 0} and
#'   \code{1} (or \code{0} if \code{log.p = TRUE}) for \code{q >= 1}. An
#'   out-of-bound or missing parameter is an error, not a return value: the
#'   wrapper stops with a message naming the parameter. An infinite parameter
#'   is not currently intercepted there and reaches the C++ layer, which
#'   treats it as invalid.
#'   Boundary return values are adjusted accordingly for \code{lower.tail = FALSE}.
#'
#' @details
#' This function computes the CDF of a Beta distribution with parameters
#' \code{shape1 = gamma} and \code{shape2 = delta + 1}. It is equivalent to
#' calling \code{stats::pbeta(q, shape1 = gamma, shape2 = delta + 1,
#' lower.tail = lower.tail, log.p = log.p)}.
#'
#' This distribution arises as a special case of the five-parameter
#' Generalized Kumaraswamy (GKw) distribution (\code{\link{pgkw}}) obtained
#' by setting \eqn{\alpha = 1}, \eqn{\beta = 1}, and \eqn{\lambda = 1}.
#' It is therefore also equivalent to the McDonald (Mc)/Beta Power distribution
#' (\code{\link{pmc}}) with \eqn{\lambda = 1}.
#'
#' The function likely calls R's underlying \code{pbeta} function but ensures
#' consistent parameter recycling and handling within the C++ environment,
#' matching the style of other functions in the related families.
#'
#' @references
#' Johnson, N. L., Kotz, S., & Balakrishnan, N. (1995). *Continuous Univariate
#' Distributions, Volume 2* (2nd ed.). Wiley.
#'
#' Cordeiro, G. M., & de Castro, M. (2011). A new family of generalized
#' distributions. *Journal of Statistical Computation and Simulation*,
#' *81*(7), 883-898.
#' \doi{10.1080/00949650903530745}
#'
#'
#' @seealso
#' \code{\link[stats]{pbeta}} (standard R implementation),
#' \code{\link{pgkw}} (parent distribution CDF),
#' \code{\link{pmc}} (McDonald/Beta Power CDF),
#' \code{\link{dbeta_}}, \code{\link{qbeta_}}, \code{\link{rbeta_}}.
#'
#' @examples
#' q <- c(0.2, 0.5, 0.8)
#' pbeta_(q, gamma = 2, delta = 3)
#' pbeta_(q, gamma = 2, delta = 3, lower.tail = FALSE)  # P(X > q)
#' pbeta_(q, gamma = 2, delta = 3, log.p = TRUE)
#'
#' ## pbeta_() is the integral of dbeta_()
#' Fq <- pbeta_(0.5, gamma = 2, delta = 3)
#' all.equal(Fq, integrate(dbeta_, 0, 0.5, gamma = 2, delta = 3, rel.tol = 1e-10)$value)
#'
#' @export
pbeta_ <- function(q, gamma = 1, delta = 0, lower.tail = TRUE, log.p = FALSE) {
  # Input validation
  if (!is.numeric(q)) {
    stop("'q' must be numeric")
  }
  if (!is.numeric(gamma) || anyNA(gamma) || any(gamma <= 0)) {
    stop("'gamma' must be positive (gamma > 0)")
  }
  if (!is.numeric(delta) || anyNA(delta) || any(delta < 0)) {
    stop("'delta' must be non-negative (delta >= 0)")
  }
  if (!is.logical(lower.tail) || length(lower.tail) != 1 || is.na(lower.tail)) {
    stop("'lower.tail' must be a single logical value")
  }
  if (!is.logical(log.p) || length(log.p) != 1 || is.na(log.p)) {
    stop("'log.p' must be a single logical value")
  }

  # Call C++ implementation
  .shape_like(.Call("_gkwdist_pbeta_",
    as.numeric(q),
    as.numeric(gamma),
    as.numeric(delta),
    as.logical(lower.tail),
    as.logical(log.p),
    PACKAGE = "gkwdist"
  ), q)
}


# ----------------------------------------------------------------------------#
# 3. QUANTILE FUNCTION (qbeta_)
# ----------------------------------------------------------------------------#

#' @title Quantile Function of the Beta Distribution (gamma, delta+1 Parameterization)
#' @author Lopes, J. E.
#' @family quantile functions
#' @concept beta
#' @keywords distribution
#'
#' @description
#' Computes the quantile function (inverse CDF) for the standard Beta
#' distribution, using a parameterization common in generalized distribution
#' families. It finds the value \code{q} such that \eqn{P(X \le q) = p}. The
#' distribution is parameterized by \code{gamma} (\eqn{\gamma}) and \code{delta}
#' (\eqn{\delta}), corresponding to the standard Beta distribution with shape
#' parameters \code{shape1 = gamma} and \code{shape2 = delta + 1}.
#'
#' @param p Vector of probabilities (values between 0 and 1).
#' @param gamma First shape parameter (\code{shape1}), \eqn{\gamma > 0}. Can be a
#'   scalar or a vector. Default: 1.0.
#' @param delta Second shape parameter is \code{delta + 1} (\code{shape2}), requires
#'   \eqn{\delta \ge 0} so that \code{shape2 >= 1}. Can be a scalar or a vector.
#'   Default: 0.0 (leading to \code{shape2 = 1}).
#' @param lower.tail Logical; if \code{TRUE} (default), probabilities are \eqn{p = P(X \le q)},
#'   otherwise, probabilities are \eqn{p = P(X > q)}.
#' @param log.p Logical; if \code{TRUE}, probabilities \code{p} are given as
#'   \eqn{\log(p)}. Default: \code{FALSE}.
#'
#' @return A vector of quantiles corresponding to the given probabilities \code{p}.
#'   The length of the result is determined by the recycling rule applied to
#'   the arguments (\code{p}, \code{gamma}, \code{delta}).
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
#' This function computes the quantiles of a Beta distribution with parameters
#' \code{shape1 = gamma} and \code{shape2 = delta + 1}. It is equivalent to
#' calling \code{stats::qbeta(p, shape1 = gamma, shape2 = delta + 1,
#' lower.tail = lower.tail, log.p = log.p)}.
#'
#' This distribution arises as a special case of the five-parameter
#' Generalized Kumaraswamy (GKw) distribution (\code{\link{qgkw}}) obtained
#' by setting \eqn{\alpha = 1}, \eqn{\beta = 1}, and \eqn{\lambda = 1}.
#' It is therefore also equivalent to the McDonald (Mc)/Beta Power distribution
#' (\code{\link{qmc}}) with \eqn{\lambda = 1}.
#'
#' The function likely calls R's underlying \code{qbeta} function but ensures
#' consistent parameter recycling and handling within the C++ environment,
#' matching the style of other functions in the related families. Boundary
#' conditions (p=0, p=1) are handled explicitly.
#'
#' @references
#' Johnson, N. L., Kotz, S., & Balakrishnan, N. (1995). *Continuous Univariate
#' Distributions, Volume 2* (2nd ed.). Wiley.
#'
#' Cordeiro, G. M., & de Castro, M. (2011). A new family of generalized
#' distributions. *Journal of Statistical Computation and Simulation*,
#' *81*(7), 883-898.
#' \doi{10.1080/00949650903530745}
#'
#'
#' @seealso
#' \code{\link[stats]{qbeta}} (standard R implementation),
#' \code{\link{qgkw}} (parent distribution quantile function),
#' \code{\link{qmc}} (McDonald/Beta Power quantile function),
#' \code{\link{dbeta_}}, \code{\link{pbeta_}}, \code{\link{rbeta_}}.
#'
#' @examples
#' p <- c(0.1, 0.5, 0.9)
#' qbeta_(p, gamma = 2, delta = 3)
#' qbeta_(p, gamma = 2, delta = 3, lower.tail = FALSE)  # upper-tail quantiles
#'
#' ## qbeta_() inverts pbeta_()
#' all.equal(pbeta_(qbeta_(p, gamma = 2, delta = 3), gamma = 2, delta = 3), p)
#'
#' @export
qbeta_ <- function(p, gamma = 1, delta = 0, lower.tail = TRUE, log.p = FALSE) {
  # Input validation
  if (!is.numeric(p)) {
    stop("'p' must be numeric")
  }
  if (!is.numeric(gamma) || anyNA(gamma) || any(gamma <= 0)) {
    stop("'gamma' must be positive (gamma > 0)")
  }
  if (!is.numeric(delta) || anyNA(delta) || any(delta < 0)) {
    stop("'delta' must be non-negative (delta >= 0)")
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
  .shape_like(.Call("_gkwdist_qbeta_",
    as.numeric(p),
    as.numeric(gamma),
    as.numeric(delta),
    as.logical(lower.tail),
    as.logical(log.p),
    PACKAGE = "gkwdist"
  ), p)
}


# ----------------------------------------------------------------------------#
# 4. RANDOM GENERATION (rbeta_)
# ----------------------------------------------------------------------------#

#' @title Random Number Generation for the Beta Distribution (gamma, delta+1 Parameterization)
#' @author Lopes, J. E.
#' @family random generation functions
#' @concept beta
#' @keywords distribution
#'
#' @description
#' Generates random deviates from the standard Beta distribution, using a
#' parameterization common in generalized distribution families. The distribution
#' is parameterized by \code{gamma} (\eqn{\gamma}) and \code{delta} (\eqn{\delta}),
#' corresponding to the standard Beta distribution with shape parameters
#' \code{shape1 = gamma} and \code{shape2 = delta + 1}. This is a special case
#' of the Generalized Kumaraswamy (GKw) distribution where \eqn{\alpha = 1},
#' \eqn{\beta = 1}, and \eqn{\lambda = 1}.
#'
#' @param n Number of observations. If \code{length(n) > 1}, the length is
#'   taken to be the number required. Must be a non-negative integer.
#' @param gamma First shape parameter (\code{shape1}), \eqn{\gamma > 0}. Can be a
#'   scalar or a vector. Default: 1.0.
#' @param delta Second shape parameter is \code{delta + 1} (\code{shape2}), requires
#'   \eqn{\delta \ge 0} so that \code{shape2 >= 1}. Can be a scalar or a vector.
#'   Default: 0.0 (leading to \code{shape2 = 1}, i.e., Uniform).
#'
#' @return A numeric vector of length \code{n} containing random deviates from the
#'   Beta(\eqn{\gamma, \delta+1}) distribution, with values in (0, 1). The length
#'   of the result is determined by \code{n} and the recycling rule applied to
#'   the parameters (\code{gamma}, \code{delta}). An out-of-bound or missing parameter is an
#'   error, not a return value: the wrapper stops with a message naming the
#'   parameter. An infinite parameter is not currently intercepted there and
#'   reaches the C++ layer, which treats it as invalid.
#'
#' @details
#' This function generates samples from a Beta distribution with parameters
#' \code{shape1 = gamma} and \code{shape2 = delta + 1}. It is equivalent to
#' calling \code{stats::rbeta(n, shape1 = gamma, shape2 = delta + 1)}.
#'
#' This distribution arises as a special case of the five-parameter
#' Generalized Kumaraswamy (GKw) distribution (\code{\link{rgkw}}) obtained
#' by setting \eqn{\alpha = 1}, \eqn{\beta = 1}, and \eqn{\lambda = 1}.
#' It is therefore also equivalent to the McDonald (Mc)/Beta Power distribution
#' (\code{\link{rmc}}) with \eqn{\lambda = 1}.
#'
#' The function likely calls R's underlying \code{rbeta} function but ensures
#' consistent parameter recycling and handling within the C++ environment,
#' matching the style of other functions in the related families.
#'
#' @references
#' Johnson, N. L., Kotz, S., & Balakrishnan, N. (1995). *Continuous Univariate
#' Distributions, Volume 2* (2nd ed.). Wiley.
#'
#' Cordeiro, G. M., & de Castro, M. (2011). A new family of generalized
#' distributions. *Journal of Statistical Computation and Simulation*,
#' *81*(7), 883-898.
#' \doi{10.1080/00949650903530745}
#'
#'
#' Devroye, L. (1986). *Non-Uniform Random Variate Generation*. Springer-Verlag.
#'
#' @seealso
#' \code{\link[stats]{rbeta}} (standard R implementation),
#' \code{\link{rgkw}} (parent distribution random generation),
#' \code{\link{rmc}} (McDonald/Beta Power random generation),
#' \code{\link{dbeta_}}, \code{\link{pbeta_}}, \code{\link{qbeta_}}.
#'
#' @examples
#' set.seed(123)
#' x <- rbeta_(1000, gamma = 2, delta = 3)
#' summary(x)
#'
#' ## The sample follows the distribution
#' hist(x, breaks = 30, freq = FALSE, main = "")
#' curve(dbeta_(x, gamma = 2, delta = 3), add = TRUE)
#' ks.test(x, pbeta_, gamma = 2, delta = 3)
#'
#' @export
rbeta_ <- function(n, gamma = 1, delta = 0) {
  # Input validation
  if (length(n) > 1) n <- length(n)
  # n = 0 is legal and yields numeric(0), as stats::rbeta(0, 2, 3) does.
  # Negative and missing n stay errors, which is also what base R does.
  if (!is.numeric(n) || length(n) != 1 || is.na(n) || n < 0) {
    stop("'n' must be a single non-negative integer")
  }
  n <- as.integer(n)

  if (!is.numeric(gamma) || anyNA(gamma) || any(gamma <= 0)) {
    stop("'gamma' must be positive (gamma > 0)")
  }
  if (!is.numeric(delta) || anyNA(delta) || any(delta < 0)) {
    stop("'delta' must be non-negative (delta >= 0)")
  }

  # Call C++ implementation
  .Call("_gkwdist_rbeta_",
    as.integer(n),
    as.numeric(gamma),
    as.numeric(delta),
    PACKAGE = "gkwdist"
  )
}


# ============================================================================#
# MAXIMUM LIKELIHOOD ESTIMATION FUNCTIONS
# ============================================================================#

# ----------------------------------------------------------------------------#
# 5. NEGATIVE LOG-LIKELIHOOD (llbeta)
# ----------------------------------------------------------------------------#

#' @title Negative Log-Likelihood for the Beta Distribution (gamma, delta+1 Parameterization)
#' @author Lopes, J. E.
#' @family log-likelihood functions
#' @concept beta
#' @keywords distribution optimize
#'
#' @description
#' Computes the negative log-likelihood function for the standard Beta
#' distribution, using a parameterization common in generalized distribution
#' families. The distribution is parameterized by \code{gamma} (\eqn{\gamma}) and
#' \code{delta} (\eqn{\delta}), corresponding to the standard Beta distribution
#' with shape parameters \code{shape1 = gamma} and \code{shape2 = delta + 1}.
#' This function is suitable for maximum likelihood estimation.
#'
#' @param par A numeric vector of length 2 containing the distribution parameters
#'   in the order: \code{gamma} (\eqn{\gamma > 0}), \code{delta} (\eqn{\delta \ge 0}).
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
#' This function calculates the negative log-likelihood for a Beta distribution
#' with parameters \code{shape1 = gamma} (\eqn{\gamma}) and \code{shape2 = delta + 1} (\eqn{\delta+1}).
#' The probability density function (PDF) is:
#' \deqn{
#' f(x | \gamma, \delta) = \frac{x^{\gamma-1} (1-x)^{\delta}}{B(\gamma, \delta+1)}
#' }
#' for \eqn{0 < x < 1}, where \eqn{B(a,b)} is the Beta function (\code{\link[base]{beta}}).
#' The log-likelihood function \eqn{\ell(\theta | \mathbf{x})} for a sample
#' \eqn{\mathbf{x} = (x_1, \dots, x_n)} is \eqn{\sum_{i=1}^n \ln f(x_i | \theta)}:
#' \deqn{
#' \ell(\theta | \mathbf{x}) = \sum_{i=1}^{n} [(\gamma-1)\ln(x_i) + \delta\ln(1-x_i)] - n \ln B(\gamma, \delta+1)
#' }
#' where \eqn{\theta = (\gamma, \delta)}.
#' This function computes and returns the *negative* log-likelihood, \eqn{-\ell(\theta|\mathbf{x})},
#' suitable for minimization using optimization routines like \code{\link[stats]{optim}}.
#' It is equivalent to the negative log-likelihood of the GKw distribution
#' (\code{\link{llgkw}}) evaluated at \eqn{\alpha=1, \beta=1, \lambda=1}, and also
#' to the negative log-likelihood of the McDonald distribution (\code{\link{llmc}})
#' evaluated at \eqn{\lambda=1}. The term \eqn{\ln B(\gamma, \delta+1)} is typically
#' computed using log-gamma functions (\code{\link[base]{lgamma}}) for numerical stability.
#'
#' @references
#' Johnson, N. L., Kotz, S., & Balakrishnan, N. (1995). *Continuous Univariate
#' Distributions, Volume 2* (2nd ed.). Wiley.
#'
#' Cordeiro, G. M., & de Castro, M. (2011). A new family of generalized
#' distributions. *Journal of Statistical Computation and Simulation*,
#' *81*(7), 883-898.
#' \doi{10.1080/00949650903530745}
#'
#'
#' @seealso
#' \code{\link{llgkw}}, \code{\link{llmc}} (related negative log-likelihoods),
#' \code{\link{dbeta_}}, \code{\link{pbeta_}}, \code{\link{qbeta_}}, \code{\link{rbeta_}},
#' \code{\link{grbeta}} (gradient),
#' \code{\link{hsbeta}} (Hessian),
#' \code{\link[stats]{optim}}, \code{\link[base]{lbeta}}.
#'
#' @examples
#' set.seed(123)
#' x <- rbeta_(1000, gamma = 2, delta = 3)
#' par <- c(gamma = 2, delta = 3)
#'
#' ## llbeta() is the negative log-likelihood, -sum(log f(x))
#' llbeta(par, x)
#' -sum(dbeta_(x, gamma = 2, delta = 3, log = TRUE))
#'
#' ## Maximum likelihood: minimize llbeta(), with grbeta() as its gradient
#' start <- gkwgetstartvalues(x, family = "beta")
#' fit <- optim(start, llbeta, grbeta, data = x, method = "L-BFGS-B", lower = 1e-4)
#' fit$convergence  # 0: converged
#' fit$par
#' fit$value <= llbeta(par, x)  # at least as good as the true values
#'
#' @export
llbeta <- function(par, data) {
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
  .Call("_gkwdist_llbeta",
    as.numeric(par),
    as.numeric(data),
    PACKAGE = "gkwdist"
  )
}


# ----------------------------------------------------------------------------#
# 6. GRADIENT (grbeta)
# ----------------------------------------------------------------------------#

#' @title Gradient of the Negative Log-Likelihood for the Beta Distribution (gamma, delta+1 Parameterization)
#' @author Lopes, J. E.
#' @family gradient functions
#' @concept beta
#' @keywords distribution optimize
#'
#' @description
#' Computes the gradient vector (vector of first partial derivatives) of the
#' negative log-likelihood function for the standard Beta distribution, using
#' a parameterization common in generalized distribution families. The
#' distribution is parameterized by \code{gamma} (\eqn{\gamma}) and \code{delta}
#' (\eqn{\delta}), corresponding to the standard Beta distribution with shape
#' parameters \code{shape1 = gamma} and \code{shape2 = delta + 1}.
#' The gradient is useful for optimization algorithms.
#'
#' @param par A numeric vector of length 2 containing the distribution parameters
#'   in the order: \code{gamma} (\eqn{\gamma > 0}), \code{delta} (\eqn{\delta \ge 0}).
#' @param data A numeric vector of observations. All values must be strictly
#'   between 0 and 1 (exclusive).
#'
#' @return Returns a numeric vector of length 2 containing the partial derivatives
#'   of the negative log-likelihood function \eqn{-\ell(\theta | \mathbf{x})} with
#'   respect to each parameter: \eqn{(-\partial \ell/\partial \gamma, -\partial \ell/\partial \delta)}.
#'   Returns a vector of \code{NaN} if any parameter values are invalid according
#'   to their constraints, or if any value in \code{data} is not in the
#'   interval (0, 1).
#'
#' @details
#' This function calculates the gradient of the negative log-likelihood for a
#' Beta distribution with parameters \code{shape1 = gamma} (\eqn{\gamma}) and
#' \code{shape2 = delta + 1} (\eqn{\delta+1}). The components of the gradient
#' vector (\eqn{-\nabla \ell(\theta | \mathbf{x})}) are:
#'
#' \deqn{
#' -\frac{\partial \ell}{\partial \gamma} = n[\psi(\gamma) - \psi(\gamma+\delta+1)] -
#' \sum_{i=1}^{n}\ln(x_i)
#' }
#' \deqn{
#' -\frac{\partial \ell}{\partial \delta} = n[\psi(\delta+1) - \psi(\gamma+\delta+1)] -
#' \sum_{i=1}^{n}\ln(1-x_i)
#' }
#'
#' where \eqn{\psi(\cdot)} is the digamma function (\code{\link[base]{digamma}}).
#' These formulas represent the derivatives of \eqn{-\ell(\theta)}, consistent with
#' minimizing the negative log-likelihood. They correspond to the relevant components
#' of the general GKw gradient (\code{\link{grgkw}}) evaluated at \eqn{\alpha=1, \beta=1, \lambda=1}.
#' Note the parameterization: the standard Beta shape parameters are \eqn{\gamma} and \eqn{\delta+1}.
#'
#' @references
#' Johnson, N. L., Kotz, S., & Balakrishnan, N. (1995). *Continuous Univariate
#' Distributions, Volume 2* (2nd ed.). Wiley.
#'
#' Cordeiro, G. M., & de Castro, M. (2011). A new family of generalized
#' distributions. *Journal of Statistical Computation and Simulation*,
#' *81*(7), 883-898.
#' \doi{10.1080/00949650903530745}
#'
#' (Note: Specific gradient formulas might be derived or sourced from additional references).
#'
#' @seealso
#' \code{\link{grgkw}}, \code{\link{grmc}} (related gradients),
#' \code{\link{llbeta}} (negative log-likelihood function),
#' \code{\link{hsbeta}} (Hessian),
#' \code{\link{dbeta_}}, \code{\link{pbeta_}}, \code{\link{qbeta_}}, \code{\link{rbeta_}},
#' \code{\link[stats]{optim}},
#' \code{\link[numDeriv]{grad}} (for numerical gradient comparison),
#' \code{\link[base]{digamma}}.
#'
#' @examples
#' set.seed(123)
#' x <- rbeta_(200, gamma = 2, delta = 3)
#' par <- c(gamma = 2, delta = 3)
#'
#' ## Gradient of the negative log-likelihood llbeta(), not of the log-likelihood
#' g <- grbeta(par, x)
#' g
#'
#' ## A small step against the gradient lowers llbeta()
#' llbeta(par - 1e-4 * g, x) < llbeta(par, x)
#'
#' ## Agrees with a numerical derivative of llbeta()
#' if (requireNamespace("numDeriv", quietly = TRUE))
#'   all.equal(g, numDeriv::grad(llbeta, par, data = x), tolerance = 1e-6)
#'
#' @export
grbeta <- function(par, data) {
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
  .Call("_gkwdist_grbeta",
    as.numeric(par),
    as.numeric(data),
    PACKAGE = "gkwdist"
  )
}


# ----------------------------------------------------------------------------#
# 7. HESSIAN (hsbeta)
# ----------------------------------------------------------------------------#

#' @title Hessian Matrix of the Negative Log-Likelihood for the Beta Distribution (gamma, delta+1 Parameterization)
#' @author Lopes, J. E.
#' @family Hessian functions
#' @concept beta
#' @keywords distribution optimize
#'
#' @description
#' Computes the analytic 2x2 Hessian matrix (matrix of second partial derivatives)
#' of the negative log-likelihood function for the standard Beta distribution,
#' using a parameterization common in generalized distribution families. The
#' distribution is parameterized by \code{gamma} (\eqn{\gamma}) and \code{delta}
#' (\eqn{\delta}), corresponding to the standard Beta distribution with shape
#' parameters \code{shape1 = gamma} and \code{shape2 = delta + 1}. The Hessian
#' is useful for estimating standard errors and in optimization algorithms.
#'
#' @param par A numeric vector of length 2 containing the distribution parameters
#'   in the order: \code{gamma} (\eqn{\gamma > 0}), \code{delta} (\eqn{\delta \ge 0}).
#' @param data A numeric vector of observations. All values must be strictly
#'   between 0 and 1 (exclusive).
#'
#' @return Returns a 2x2 numeric matrix representing the Hessian matrix of the
#'   negative log-likelihood function, \eqn{-\partial^2 \ell / (\partial \theta_i \partial \theta_j)},
#'   where \eqn{\theta = (\gamma, \delta)}.
#'   Returns a 2x2 matrix populated with \code{NaN} if any parameter values are
#'   invalid according to their constraints, or if any value in \code{data} is
#'   not in the interval (0, 1).
#'
#' @details
#' This function calculates the analytic second partial derivatives of the
#' negative log-likelihood function (\eqn{-\ell(\theta|\mathbf{x})}) for a Beta
#' distribution with parameters \code{shape1 = gamma} (\eqn{\gamma}) and
#' \code{shape2 = delta + 1} (\eqn{\delta+1}). The components of the Hessian
#' matrix (\eqn{-\mathbf{H}(\theta)}) are:
#'
#' \deqn{
#' -\frac{\partial^2 \ell}{\partial \gamma^2} = n[\psi'(\gamma) - \psi'(\gamma+\delta+1)]
#' }
#' \deqn{
#' -\frac{\partial^2 \ell}{\partial \gamma \partial \delta} = -n\psi'(\gamma+\delta+1)
#' }
#' \deqn{
#' -\frac{\partial^2 \ell}{\partial \delta^2} = n[\psi'(\delta+1) - \psi'(\gamma+\delta+1)]
#' }
#'
#' where \eqn{\psi'(\cdot)} is the trigamma function (\code{\link[base]{trigamma}}).
#' These formulas represent the second derivatives of \eqn{-\ell(\theta)},
#' consistent with minimizing the negative log-likelihood. They correspond to
#' the relevant 2x2 submatrix of the general GKw Hessian (\code{\link{hsgkw}})
#' evaluated at \eqn{\alpha=1, \beta=1, \lambda=1}. Note the parameterization
#' difference from the standard Beta distribution (\code{shape2 = delta + 1}).
#'
#' The returned matrix is symmetric.
#'
#' @references
#' Johnson, N. L., Kotz, S., & Balakrishnan, N. (1995). *Continuous Univariate
#' Distributions, Volume 2* (2nd ed.). Wiley.
#'
#' Cordeiro, G. M., & de Castro, M. (2011). A new family of generalized
#' distributions. *Journal of Statistical Computation and Simulation*,
#' *81*(7), 883-898.
#' \doi{10.1080/00949650903530745}
#'
#' (Note: Specific Hessian formulas might be derived or sourced from additional references).
#'
#' @seealso
#' \code{\link{hsgkw}}, \code{\link{hsmc}} (related Hessians),
#' \code{\link{llbeta}} (negative log-likelihood function),
#' \code{\link{grbeta}} (gradient),
#' \code{\link{dbeta_}}, \code{\link{pbeta_}}, \code{\link{qbeta_}}, \code{\link{rbeta_}},
#' \code{\link[stats]{optim}},
#' \code{\link[numDeriv]{hessian}} (for numerical Hessian comparison),
#' \code{\link[base]{trigamma}}.
#'
#' @examples
#' set.seed(123)
#' x <- rbeta_(1000, gamma = 2, delta = 3)
#' par <- c(gamma = 2, delta = 3)
#'
#' ## Hessian of the negative log-likelihood llbeta()
#' H <- hsbeta(par, x)
#' isSymmetric(H)
#'
#' ## Agrees with a numerical Hessian of llbeta()
#' if (requireNamespace("numDeriv", quietly = TRUE))
#'   all.equal(H, numDeriv::hessian(llbeta, par, data = x), tolerance = 1e-8)
#'
#' ## At the MLE it is the observed information; its inverse estimates the
#' ## covariance of the estimates
#' fit <- optim(par, llbeta, grbeta, data = x, method = "L-BFGS-B", lower = 1e-4)
#' sqrt(diag(solve(hsbeta(fit$par, x))))  # standard errors
#'
#' @export
hsbeta <- function(par, data) {
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
  .Call("_gkwdist_hsbeta",
    as.numeric(par),
    as.numeric(data),
    PACKAGE = "gkwdist"
  )
}
