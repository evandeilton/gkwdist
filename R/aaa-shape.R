# Internal: give a d/p/q result the shape of its first argument.
#
# base R's density, distribution and quantile functions carry the input's
# attributes through to the output --- dbeta(matrix(x, 2, 2), 2, 3) keeps its
# dim, and dbeta(c(a = .2, b = .5), 2, 3) keeps its names --- so code that
# indexes or plots the result by shape keeps working. This package dropped both.
#
# The copy is conditional on the lengths agreeing, which is also what base R
# does: once a recycled parameter makes the output longer than the first
# argument, dim(dbeta(matrix(x, 2, 2), c(2, 3, 4, 5, 6), 3)) is NULL.
#
# @param out The numeric vector returned by the C++ routine.
# @param x   The first argument of the calling function (x, q or p).
# @return `out`, carrying `x`'s dim and dimnames, or its names.
# @noRd
.shape_like <- function(out, x) {
  if (length(out) != length(x)) {
    return(out)
  }
  d <- attr(x, "dim")
  if (!is.null(d)) {
    dim(out) <- d
    dn <- attr(x, "dimnames")
    if (!is.null(dn)) dimnames(out) <- dn
  } else {
    nm <- attr(x, "names")
    if (!is.null(nm)) names(out) <- nm
  }
  out
}
