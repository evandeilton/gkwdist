# gkwdist 0.1.0

## New Features

* Initial CRAN release
* Generalized Kumaraswamy distribution (5 parameters)
* Six nested sub-families: Beta, Kumaraswamy, Exponentiated-Kumaraswamy, 
  Kumaraswamy-Kumaraswamy, Beta-Kumaraswamy, and McDonald distributions
* Fast MLE estimation via TMB
* Complete set of distribution functions (d/p/q/r)
* Log-likelihood, gradient, and Hessian functions for all families

## Performance

* Optimized C++ implementation via Rcpp
* Template Model Builder integration for efficient estimation
* Vectorized operations for speed
