# There is deliberately no utils::globalVariables() call here. This package has
# no non-standard evaluation, so codetools finds no undefined globals:
# `R CMD check` reports "checking R code for possible problems ... OK" with no
# registration at all. Do not add one without a check finding that names it.

#' Pipe operator
#'
#' See \code{magrittr::\link[magrittr:pipe]{\%>\%}} for details.
#'
#' @name %>%
#' @rdname pipe
#' @keywords internal
#' @export
#' @importFrom magrittr %>%
#' @usage lhs \%>\% rhs
#' @param lhs A value or the magrittr placeholder.
#' @param rhs A function call using the magrittr semantics.
#' @return The result of calling `rhs(lhs)`.
NULL


## usethis namespace: start
#' @importFrom Rcpp sourceCpp evalCpp
#' @import graphics
## usethis namespace: end
NULL

#' @useDynLib gkwdist, .registration = TRUE
NULL
