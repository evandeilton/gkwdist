# Estimate Distribution Parameters Using Method of Moments

Estimates parameters for various distribution families from the
Generalized Kumaraswamy family using the method of moments. The
implementation is optimized for numerical stability and computational
efficiency through Nelder-Mead optimization and adaptive numerical
integration.

## Usage

``` r
gkwgetstartvalues(x, family = "gkw", n_starts = 5L)
```

## Arguments

- x:

  Numeric vector of observations. All values must be in the open
  interval (0,1). Values outside it – including exact 0 and exact 1 –
  are truncated to the interval so that a single boundary observation
  does not abort a fit, and a warning naming the number of truncated
  observations and their observed range is issued. Truncation shifts
  every sample moment and therefore every estimate returned, so treat
  the warning as a signal to check the scale of the data rather than as
  noise: values such as 5 or -3 usually mean a percentage or 0-100 scale
  that has to be rescaled first. `NA` and non-finite values are dropped
  without a warning.

- family:

  Character string specifying the distribution family. Valid options
  are: `"gkw"` (Generalized Kumaraswamy - 5 parameters), `"bkw"`
  (Beta-Kumaraswamy - 4 parameters), `"kkw"` (Kumaraswamy-Kumaraswamy -
  4 parameters), `"ekw"` (Exponentiated Kumaraswamy - 3 parameters),
  `"mc"` (McDonald - 3 parameters), `"kw"` (Kumaraswamy - 2 parameters),
  `"beta"` (Beta - 2 parameters). The string is case-insensitive.
  Default is `"gkw"`.

- n_starts:

  Integer specifying the number of different initial parameter values to
  try during optimization. Every starting point is optimized and the
  candidate with the smallest objective is returned, so more starting
  points can only improve the fit, at a cost that grows linearly in
  `n_starts`. Default is 5. Four fixed, family-specific starting points
  are always used, so values below 4 behave as 4.

## Value

Named numeric vector containing the estimated parameters for the
specified distribution family. Parameter names correspond to the
distribution specification. If estimation fails, returns a vector of NA
values with appropriate parameter names.

## Details

The function uses the method of moments to estimate distribution
parameters by minimizing the weighted sum of squared relative errors
between theoretical and sample moments (orders 1 through 5). The
optimization employs the Nelder-Mead simplex algorithm, which is
derivative-free and particularly robust for this problem.

Key implementation features: logarithmic calculations for numerical
stability, adaptive numerical integration using Simpson's rule with
fallback to trapezoidal rule, multiple starting points to avoid local
minima, decreasing weights for higher-order moments (1.0, 0.8, 0.6, 0.4,
0.2), and automatic parameter constraint enforcement.

Multiple starts: the grid of starting points is built from four fixed,
family-specific points – one of them derived from the sample moments –
plus `n_starts - 4` further points drawn over the family's parameter
box. Nelder-Mead is run from every point in the grid, each result is
clipped to the parameter box, and the clipped candidate with the
smallest objective is returned. Because the candidate set grows with
`n_starts`, the returned objective is non-increasing in `n_starts`.

Determinism: the extra starting points are drawn from a generator with a
fixed internal seed, deliberately not from R's random number stream. The
function is therefore deterministic – two calls with the same `x`,
`family` and `n_starts` return the same vector, so a fit seeded by these
values is reproducible without any further precaution. Two consequences
are worth stating explicitly:
[`set.seed`](https://rdrr.io/r/base/Random.html) has no effect on
`gkwgetstartvalues`, and the function neither reads nor advances
`.Random.seed`, so it cannot perturb simulations running alongside it.
To widen the search, raise `n_starts` rather than looking for a seed
argument.

Parameter Constraints: All parameters are constrained to positive
values. Additionally, family-specific constraints are enforced: alpha
and beta in (0.1, 50.0), gamma in (0.1, 10.0) for GKw-related families
or (0.1, 50.0) for Beta, delta in (0.01, 10.0), and lambda in (0.1,
20.0).

The function will issue warnings for empty input vectors, observations
outside the open interval (0,1) (truncated to it), sample sizes less
than 10 (unreliable estimation), or failure to find valid parameter
estimates (returns defaults).

## References

Jones, M. C. (2009). Kumaraswamy's distribution: A beta-type
distribution with some tractability advantages. *Statistical
Methodology*, *6*(1), 70-81.
[doi:10.1016/j.stamet.2008.04.001](https://doi.org/10.1016/j.stamet.2008.04.001)

## See also

[`llgkw`](https://evandeilton.github.io/gkwdist/reference/llgkw.md),
[`llbkw`](https://evandeilton.github.io/gkwdist/reference/llbkw.md),
[`llkkw`](https://evandeilton.github.io/gkwdist/reference/llkkw.md),
[`llekw`](https://evandeilton.github.io/gkwdist/reference/llekw.md),
[`llmc`](https://evandeilton.github.io/gkwdist/reference/llmc.md),
[`llkw`](https://evandeilton.github.io/gkwdist/reference/llkw.md),
[`llbeta`](https://evandeilton.github.io/gkwdist/reference/llbeta.md)
(the objectives these values are meant to seed),
[`optim`](https://rdrr.io/r/stats/optim.html)

## Examples

``` r
# \donttest{
# Generate sample data from a Beta distribution. set.seed() here makes the
# SAMPLE reproducible; gkwgetstartvalues() itself is deterministic and is
# unaffected by the seed (see the Determinism note in Details).
set.seed(123)
x <- rbeta(100, shape1 = 2, shape2 = 3)

# Estimate Beta parameters
params_beta <- gkwgetstartvalues(x, family = "beta")
print(params_beta)
#>    gamma    delta 
#> 2.421561 2.455624 

# Estimate Kumaraswamy parameters
params_kw <- gkwgetstartvalues(x, family = "kw")
print(params_kw)
#>    alpha     beta 
#> 1.972954 3.791382 

# Estimate GKw parameters with more starting points
params_gkw <- gkwgetstartvalues(x, family = "gkw", n_starts = 10)
print(params_gkw)
#>     alpha      beta     gamma     delta    lambda 
#> 0.5562367 1.4240218 1.9332680 1.2036964 2.4553624 

# Deterministic: the seed in force does not change the answer, and the
# function does not consume draws from R's stream.
set.seed(1)
identical(gkwgetstartvalues(x, family = "kw"), params_kw)
#> [1] TRUE
set.seed(9999)
identical(gkwgetstartvalues(x, family = "kw"), params_kw)
#> [1] TRUE

before <- .Random.seed
invisible(gkwgetstartvalues(x, family = "kw"))
identical(before, .Random.seed)
#> [1] TRUE
# }
```
