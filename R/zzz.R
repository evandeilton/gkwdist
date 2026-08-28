# There is deliberately no utils::globalVariables() call here. This package has
# no non-standard evaluation, so codetools finds no undefined globals:
# `R CMD check` reports "checking R code for possible problems ... OK" with no
# registration at all. Do not add one without a check finding that names it.

#' Pipe operator
#'
#' See \code{magrittr::\link[magrittr:pipe]{\%>\%}} for details.
#'
#' @section Deprecation notice:
#' This re-export is deprecated and is scheduled for removal in a future
#' release. gkwdist does not use the pipe anywhere in its own code; it
#' re-exports `magrittr`'s operator only, which makes `magrittr` a hard
#' dependency of a package that has no need of it. Nothing changes in this
#' release: `%>%` remains exported and behaves exactly as before.
#'
#' If your code relies on gkwdist supplying `%>%`, take it from its own source
#' instead -- `library(magrittr)`, or any of the tidyverse packages that
#' re-export it -- or use R's native pipe, `|>`, available since R 4.1.0.
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
