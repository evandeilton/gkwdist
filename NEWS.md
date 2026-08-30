# gkwdist 1.1.6

## Numerical Utilities

* **`gkw_log1mexp()` had the sign of its second-order correction inverted**
  (`utils.h`): the Taylor branch, used for `-1e-14 < u <= 0`, returned
  `log(-u) - u/2` where the expansion gives `log(-u) + u/2`. Since
  `1 - exp(u) = -u (1 + u/2 + u^2/6 + ...)`, the correction is
  `log(1 + u/2) ~ +u/2`, and the derivation in the comment above the line
  carried the same slip. With `u < 0` the two forms differ by `|u|`. Against a
  400-digit reference:

  ```
  u          reference                err before   err after
  -9.9e-15   -32.246241637770147      1.42e-14     0
  -5.0e-15   -32.929338482476588      0            0
  ```

  About 1.4 ulp at this magnitude, but in the wrong direction, and it left a
  step where the function crosses into the `expm1` branch -- breaking the
  "relative error < 2*EPSILON" the header promises.

  Downstream the effect is at the noise floor, and is reported as such: 56 of
  238,140 grid values move, by at most 2.3e-13 relative, and adjudicating the
  changed densities against a 400-digit reference gives 4 closer and 2 farther,
  all between 1e-15 and 2e-14. The defensible claim is the direct one, on the
  function itself.

* **`safe_pow()` cast a `double` exponent to `int` without a range check**
  (`utils.h`): undefined behaviour once `|y| > INT_MAX`, which UBSan flags.
  `vec_safe_pow()` had a guard for it, but that guard answered the parity
  question wrongly in the process, reporting "even" for every `|y|` above
  `INT_MAX` -- and an odd integer between `INT_MAX` and `2^53` is exactly
  representable as a `double`. Both now take the parity from
  `fmod(|y|, 2) == 1`, which is correct at every magnitude: above `2^53` every
  `double` is even, and `fmod` says so.

* **`safe_pow()`'s documentation claimed an accuracy it does not have**
  (`utils.h`): it said the `exp(y * log(x))` form "provides better numerical
  stability than direct `pow()`". The reverse is true. That form carries a
  relative error of roughly `|y log x| * EPSILON`, while `std::pow` on a
  conforming libm is very nearly correctly rounded. Measured against a 60-digit
  reference:

  ```
  x     y      exp(y*log x)   std::pow
  10    100    1.11e-14       0
  10    300    9.00e-14       0
  2     1000   6.85e-14       0
  ```

  The note now says what the form is actually for -- intercepting overflow and
  underflow before they happen -- and points callers who do not need that to
  `std::pow`.

## Base R Contract

* **The `@return` contract said `NaN`, the wrappers said `stop()`** (all 28
  `d*`, `p*`, `q*` and `r*` functions): the audit asked for one contract chosen
  and propagated. `stop()` is the one kept -- it is what the package has shipped
  for its whole CRAN life, a named error is a better diagnostic than a `NaN` for
  the case that actually occurs, and switching to `NaN` would break every
  `tryCatch(..., error = )` guard written against it. All 28 `@return` blocks now
  describe the behaviour the functions have. `gr*()` and `hs*()` genuinely do
  return `NaN`, and their 14 blocks are unchanged.

  Documentation only. No guard, message or value changes.

  **Known and deliberately left open:** an infinite parameter is not intercepted
  by the wrapper -- `any(alpha <= 0)` is `FALSE` for `Inf` and `anyNA()` does not
  catch it -- so it reaches C++, where `check_*_pars()` has always rejected it,
  and the routine leaves its fill value. Tightening the R guard is correct in
  principle but is a behaviour change with a reverse-dependency cost:
  `gkwreg::predict()` clamps only the lower bound of its linear predictor, so
  `exp(eta)` can overflow to `Inf` on extrapolated `newdata`, and one such row
  would abort the whole prediction vector. The value it gets today is 0, which is
  the correct limit -- `Kw(alpha, beta)` concentrates at 1 as `alpha` grows -- so
  nothing is currently wrong, only silent. An upper clamp in `gkwreg` is the
  prerequisite. The current behaviour is pinned in
  `tests/testthat/test-return-contract.R` so the eventual change is visible.

* **Only two of seven `ll*()` warned about data outside the open support**
  (`llekw` and `llbeta`): `ll*()` returns `+Inf` there -- an infinite objective
  with no gradient direction, which an optimiser can only sit on -- and five of
  the seven announced a corrupted sample not at all. All seven warn now.

* **`Rcpp::warning()` fired once per element inside the vectorised loops**
  (`dgkw`, `pgkw`, `qgkw`, `rgkw`, and the six nested `r*`): on 50,000 values
  with half the parameters invalid that cost 51x, and under
  `options(warn = 2)` each call longjmps out of the loop through C++ frames
  holding live Armadillo objects. R's own convention is a single warning per
  call. Each routine now sets a flag and warns once after the loop.

  ```
  50,000 values, half invalid      before      after
  elapsed                          0.514 s     0.005 s
  warnings raised                  25,000      1
  ```

  The message now also names what the routine actually returns, which base R
  pairs up: `dbeta(0.5, -1, 1)` is `NaN` and warns *"NaNs produced"*, while
  `rbeta(2, -1, 1)` is `NA` and warns *"NAs produced"*. `p*`, `q*` and `r*` fill
  `NA_REAL` and say so. `dgkw()` leaves its fill value, 0, so it says
  *"invalid parameters"* and claims no return value at all -- promising a `NaN`
  that is not there is the defect this release fixed for `q*()`.

  Values are bit-identical over the 238,140-value regression grid. The path is
  reachable from the exported API only through an `Inf` parameter, since every
  wrapper `stop()`s on `<= 0`, `NA` and `NaN`.

  **Known inconsistency:** `man/dgkw.Rd` said the function returns `NaN` for
  invalid parameters; it returns 0, and the other six families document and
  return 0 too. The `@return` contract entry above corrects the documentation.

* **`d*()` returned 0 at `x = 0` and `x = 1`, where base R returns the limit**
  (all seven families): the GKw support is the open interval, but base R's
  density functions carry the limiting value at the closed boundary --
  `dbeta(0, 0.5, 1)` is `Inf` and `dbeta(1, 2, 1)` is 2 -- and any code that
  plots a density across `[0, 1]` depends on it. Curves fell to zero exactly
  where they should have diverged.

  Substituting the first-order forms the log chain already uses, the
  log-density at each boundary collapses to a constant plus one power of the
  vanishing quantity, so a single exponent decides the answer:

  ```
  at x = 0:  alpha*gamma*lambda - 1        at x = 1:  beta*(delta + 1) - 1
      > 0  ->  0        = 0  ->  the constant        < 0  ->  Inf
  ```

  Each nested family reaches this through its own fixed parameters. The rule was
  checked against `stats::dbeta` at all ten combinations the Beta
  parameterisation can express, and against the nesting identities at both
  boundaries for the other six.

  ```
  dkw(0, 0.5, 1)     0  ->  Inf        dbeta_(1, 2, 0)     0  ->  2
  ```

  Anything strictly outside `[0, 1]` is still 0, as in base R, and the interior
  is bit-identical over the 238,140-value regression grid.

* **`d*()`, `p*()` and `q*()` dropped `dim`, `dimnames` and `names`** (all seven
  families): base R carries the first argument's attributes through to the
  output, so code that indexes or plots the result by shape keeps working.

  ```
                                    before   after   stats::dbeta
  dim(d(matrix(x, 2, 2), 2, 3))      NULL     2 2     2 2
  names(d(c(a = .2, b = .5), 2, 3))  NULL     a b     a b
  ```

  The copy is conditional on the lengths agreeing, which is also what base R
  does: once a recycled parameter makes the output longer than the first
  argument, `dim()` is `NULL` in both. Values are bit-identical over the
  238,140-value regression grid.

* **`r*(0)` raised an error instead of returning `numeric(0)`** (all seven
  families): `stats::rbeta(0, 2, 3)` is `numeric(0)`, and a generator that
  errors instead breaks any loop or `replicate()` that reaches an empty case.
  All seven now return `numeric(0)`. A negative or missing `n` is still an
  error, as in base R.

## Critical Bug Fixes

* **`pmc()` reached `R::pbeta` through `exp(lambda * log(x))`, which is not a
  round trip** (`bpmc.cpp`): for `lambda = 1` the Mc family is the Beta family,
  and `pmc(x, gamma, delta, 1)` should be `R::pbeta(x, gamma, delta+1)` exactly.
  It was not: `exp(1 * log(x))` fails to return `x` for 2 of 9 ordinary values
  under glibc, and for a different pair under the macOS ARM64 libm, which is
  where CI caught it.

  `std::pow` is used instead. C99 requires `pow(x, 1.0) == x`, so the identity
  now holds bit-for-bit on every platform, and the form is also the more
  accurate one for every other `lambda`. Against a 60-digit reference over 1,030
  grid points:

  ```
  lambda    exp(l*log x)    std::pow     improved   worse
  1.5       1.19e-14        1.67e-16     403        3
  0.1       6.57e-16        1.64e-16      56        0
  2.5       1.57e-14        1.11e-16     527        0
  ```

  Only `pmc` moves; 1,093 of 238,140 grid values, all in `p mc`.

* **The upper tail of `pgkw()` and `pbkw()` collapsed to exactly zero**
  (`gkw.cpp`, `bkw.cpp`): the CDF batch moved `lower.tail` and `log.p` onto
  `R::pbeta` instead of applying them afterwards, which fixed the lower tail. A
  second defect remained, in the *argument* rather than the tail flag.

  `pgkw()` forms `y = [1 - (1 - x^alpha)^beta]^lambda` and evaluates
  `I_y(gamma, delta+1)`. As `x` approaches 1 the exponent `lambda*log_w` falls
  below 1.1e-16, `exp()` returns exactly 1, and
  `R::pbeta(1, ., ., lower = FALSE)` returns exactly 0. `pbkw()` reaches the
  same place through `-expm1` of an exponent running off to `-Inf`. Against a
  300-digit incomplete beta, for `pgkw(x, 2, 3, 1.5, 2, 0.8, lower.tail = FALSE)`:

  ```
  1-x     returned        exact           rel err
  1e-4    5.731692e-34    5.731820e-34    2.2e-05
  1e-5    5.840671e-43    5.734142e-43    1.9e-02
  1e-6    0               5.734374e-52    1.00     <- collapse
  1e-16   0               1.469535e-141   1.00
  ```

  The true tail is still representable sixteen decades past the point where the
  routine gave up, and `lower.tail = FALSE` is ordinary documented API, so
  survival probabilities, p-values and quantile residuals were silently zero.

  `I_y(a,b) = 1 - I_{1-y}(b,a)` is exact, and `1 - y` is `-expm1` of the same
  exponent that produces `y`, at full relative accuracy. Reflecting above
  `y = 1/2` sends the small quantity into `pbeta` and the large one out of it;
  below that crossover the direct form already holds the small quantity and is
  left alone. This is the same correction `pmc()` received in this release.

  After the fix the maximum relative error over those fifteen decades is
  6.1e-14 for `pgkw()` and 5.3e-14 for `pbkw()`. Confinement over the
  238,140-value grid: only `pgkw` and `pbkw` move, only with
  `lower.tail = FALSE`, and no `d*` or `q*` value changes at all. The nesting
  identity against `pmc()` -- corrected independently, in another translation
  unit -- goes from 4.88e-01 to 2.84e-14.

* **`NA`, `NaN` and infinite input did not propagate** (all seven families, 21
  routines): `NA_REAL` is a `NaN`, so every comparison against it is false and
  `!R_finite()` is true. Missing input therefore fell into the "outside the
  support" branch and was silently replaced by the fill value:

  ```
                    before            after      base R
  d*(NA)              0              NA          NA
  d*(NaN)             0             NaN         NaN
  p*(NA)              0              NA          NA
  p*(NaN)             0             NaN         NaN
  p*(+Inf)            0               1           1
  p*(-Inf)            0               0           0
  ```

  `p*(Inf) = 0` also violated monotonicity outright: `pkw(2, 2, 3)` was 1 while
  `pkw(Inf, 2, 3)` was 0. The fix needed no new branch -- dropping `!R_finite()`
  from the lower boundary test lets `+Inf` fall through to the `q >= 1` case
  that was already there.

  `NA` and `NaN` are distinguished, as base R distinguishes them: `R_IsNA()` is
  asked before `R_IsNaN()`, so `is.na()` and `is.nan()` both answer correctly.

* **`q*(p)` saturated at 0 or 1 for probabilities outside `[0, 1]`** (all seven
  families): the R wrappers have always warned that such a value *"will produce
  NaN"*, while the C++ returned a bound. Defensive code testing `is.nan()` --
  exactly what the warning tells the caller to expect -- saw nothing, and 0 and
  1 are outside the open support, so the value flowed on into `d*()` and `ll*()`
  as valid data.

  ```
  qkw(-0.5, 2, 3)   0  ->  NaN        qkw(1.5, 2, 3)   1  ->  NaN
  qkw(-Inf, 2, 3)   0  ->  NaN        qkw(Inf, 2, 3)   1  ->  NaN
  ```

  The closed boundary is unchanged: `q*(0)` is still 0 and `q*(1)` still 1,
  in both tails and on both scales.

  Confinement was checked by bit-comparison rather than adjudication, since no
  in-range value should move: 238,140 values over the seven families, both
  tails, both scales, and x from 1e-320 to 1 - 1e-16 are **bit-identical**.

* **`log`, `lower.tail` and `log.p` silently accepted `NA` and read it as
  `TRUE`** (all seven families, 35 guards): the check was
  `!is.logical(flag) || length(flag) != 1`, and `NA` passes both halves --
  `is.logical(NA)` is `TRUE` and `length(NA)` is 1. The value reached C++ as
  `NA_LOGICAL`, which is a non-zero integer, so it was read as `TRUE`:

  ```
  dgkw(0.5, 2, 3, 1.5, 0.5, 2, log = NA)   0.8517722   <- the LOG density
  dgkw(0.5, 2, 3, 1.5, 0.5, 2)             2.343797    <- the density
  ```

  The documented error is now raised, with the message the help pages already
  promised.

* **An `NA` shape parameter behaved three different ways** (all seven families,
  92 guards): `gkw`, `bkw`, `kkw`, `mc` and `kw` wrote `any(alpha <= 0)`, so `if`
  received `NA` and R raised its own opaque *"missing value where TRUE/FALSE
  needed"* instead of the documented message; `ekw` and `beta_` wrote
  `any(alpha <= 0, na.rm = TRUE)`, which dropped the `NA` and returned 0 as
  though the call had been valid. All seven now raise the package's own error:

  ```
  dkw(0.5, NA, 3)        before  Error: missing value where TRUE/FALSE needed
  dekw(0.5, NA, 3, 1)    before  0
  ```

  Both now raise the package's own error.

  Valid calls are unaffected: `dgkw(0.5, 2, 3, 1.5, 0.5, 2)` is still 2.343797,
  and `delta = 0` is still accepted.

* **`grkkw()` and `hskkw()` failed silently and only partly** (`kkw.cpp`):
  alone among the seven families, they carried none of the guards their BKw
  counterparts have. A refused input -- a short parameter vector, an invalid
  parameter, data outside `(0,1)` -- returned a `NaN` result with nothing said,
  so the caller could not tell a rejected call from a genuine boundary. Neither
  function checked its result at all, so where the chain did reach a boundary
  they returned whatever survived:

  ```
  par = (1e-8, 1e-8, 0, 1e300), x = c(0.01, 0.3, 0.6, 0.9)
    grkkw   1.378417e+307   -Inf   -2.000000   31.43853     (no warning)
    hskkw   2 of 16 entries NaN, 14 finite                  (no warning)
    grbkw at the corresponding BKw point: warns, and every component is NaN
  ```

  A gradient with a finite component next to an `-Inf`, or a Hessian with two
  `NaN` entries among fourteen, is worse than no answer: it looks partly usable,
  and the Hessian would have been inverted for a standard error. Both now warn
  and return a uniformly `NaN` result, matching `grbkw()` and `hsbkw()`.

  The support test also gained `has_nan()`. A `NaN` compares false against both
  bounds, so `NaN` data passed straight through; `hskkw()` returned a matrix
  with fifteen `NaN` entries and one finite one.

  This is a reporting change only. Over the same 43,792-value grid used above,
  no value differs from the previous fix -- every case the guards catch was
  already `NaN` or `Inf`.

* **The BKw and KKw likelihoods collapsed to `+Inf` on ordinary data**
  (`bkw.cpp`, `kkw.cpp`): both families walk the same log-space chain as their
  GKw parent -- `v = 1 - x^alpha`, `w = 1 - v^beta`, `z = 1 - w^lambda`, with
  `lambda = 1` for BKw and `gamma = 1` for KKw -- and both stopped at the point
  where `v` underflows to exactly 1. `log1mexp()` then receives an argument of
  exactly 0 and can only answer `-Inf`. The threshold is `alpha*log(min(x)) <
  -745`, which perfectly ordinary data crosses:

  ```
  x = c(0.01, 0.3, 0.6, 0.9)
    alpha = 161   llbkw(c(a,2,1.5,1), x) = 1515.5261017902   llkkw(c(a,2,1,1.5), x) = 1516.4186759096
    alpha = 162   llbkw                  = Inf               llkkw                  = Inf
    alpha = 200   llbkw                  = Inf               llkkw                  = Inf
  ```

  Every larger `alpha` stayed at `Inf`, so the likelihood surface carried an
  infinite plateau that an optimiser cannot leave. The true values, 1525.13 and
  1526.03 at `alpha = 162`, are now returned.

  The same boundary reached the density, the gradient and the Hessian.
  `dbkw()` and `dkkw()` dropped such observations and returned the fill value
  (`-Inf` in log, 0 otherwise). The derivatives built the ratios `v^beta/w` and
  `w^lambda/z` as separate factors; each overflowed to `+Inf` on its own while
  the quantity it multiplied had underflowed to 0, and `0 * Inf` is `NaN`:

  ```
  par = (200, 2, 1.5, 1), x = c(.01, .30, .60, .90)
    grbkw            NaN  NaN  NaN  NaN
    grgkw(lambda=1)  9.617994  -3.000000  1278.026571  -2.721489
  par = (200, 2, 1, 1.5), x = c(.01, .30, .60, .90)
    grkkw            NaN  NaN  -2  NaN          (partly NaN, and silently so)
    grgkw(gamma=1)   9.617994  -3.000000  -2.000000  1279.626571
  ```

  Both files now use the first-order limits that are exact to the last
  representable bit -- as `x -> 0`, `log_w = log(beta) + alpha*log(x)`; as
  `x -> 1`, `log_z = log(lambda) + beta*log_v` -- keep every ratio inside a
  single `exp()` of a sum of logs, and never let a coefficient of exactly zero
  multiply a logarithm. This is the same repair `gkw.cpp` received, so the
  nesting identities `BKw(a,b,g,d) == GKw(a,b,g,d,1)` and
  `KKw(a,b,d,l) == GKw(a,b,1,d,l)` now hold where they previously did not.

  Over a grid of 19,488 values spanning `x` from 1e-300 to `1 - 1e-16` and
  `alpha` from 0.01 to 5000, 1,966 results changed from `NaN`/`Inf` to a finite
  value and none went the other way. 1,089 `NaN` Hessian entries -- 71 whole
  unusable matrices -- became finite. Adjudicated against a 400-digit reference
  and against the GKw parent: the maximum relative error fell from infinite to
  1.8e-07 for the log-likelihoods and 1.4e-04 for the gradients, both attained
  only at `alpha >= 1000` where double precision has no digits left to lose.
  No value that was already exactly correct changed; values correct to within
  one ulp rose from 100 to 152 (log-likelihood), 348 to 479 (gradient) and 1,337
  to 1,716 (Hessian). `pbkw()`, `qbkw()`, `pkkw()` and `qkkw()` are untouched
  and bit-identical.

* **`pmc()`'s upper tail was quantised by the argument it handed to
  `R::pbeta`** (`bpmc.cpp`): `F(x) = I_{x^lambda}(gamma, delta+1)`, and `pmc()`
  formed `x^lambda` in linear arithmetic. Once `x^lambda` passes 1/2 a double
  holds it no more finely than 1.1e-16, and the upper tail is a function of
  `1 - x^lambda` alone, so it was quantised to whatever that left -- and to
  exactly 0 once `x^lambda` reached 1:

  ```
  pmc(x, 4, 2, 0.25, lower.tail = FALSE)      before             exact
    x = 1 - 1e-15                       2.1895288505e-46   3.1175127579e-46
    x = 1 - 1.1e-16                     0                  4.2764235361e-49
  ```

  a relative error of 700%, then of 100%. `I_y(a,b) = 1 - I_{1-y}(b,a)` is
  exact, and `1 - x^lambda` comes from `-expm1` of the same exponent at full
  relative accuracy, so reflecting sends the small quantity into `pbeta`. The
  reflection is applied only where the direct form is the one holding the large
  quantity: the lower tail never changes, and neither does any upper tail with
  `x^lambda <= 1/2`.

  Over 8942 grid cells whose exact tail is representable at all, 8494 are
  bit-identical, 353 improved and 76 moved the other way. The maximum relative
  error fell from 7.00 to 1.36e-13, and that residual sits on a cell this change
  did not touch. The 76 that moved the other way went from at most 2.67e-14 to
  at most 4.10e-14 relative, which is `R::pbeta`'s own accuracy at tail
  probabilities of 1e-71: both routes hand it an argument good to one ulp there,
  and they round differently. `dmc()`, `qmc()`, `rmc()`, `llmc()`, `grmc()` and
  `hsmc()` are bit-identical, as is every lower tail and every `lambda = 1`
  result, which stays identical to `stats::pbeta` to the bit.

* **`grmc()` and `hsmc()` swapped `R::digamma` and `R::trigamma` for two-term
  asymptotic expansions above three separate thresholds** (`bpmc.cpp`):
  `gamma > 100`, `delta > 100` and `gamma + delta > 100`. `log(z) - 1/(2z)`
  truncates psi's expansion before the `1/(12z^2)` term and is wrong by 8.33e-06
  at `z = 100` and by 1.30e-03 at `z = 8`, which the `gamma + delta` threshold
  can reach with `gamma` that small; `1/z + 1/(2z^2)` drops psi'-s `1/(6z^3)`
  term and is wrong by 1.67e-07 at `z = 100`.

  Each threshold put a step of `n` times that error into a different component,
  at a different place. On the seven observations `c(.1,.25,.4,.5,.6,.75,.9)`:

  ```
  grmc(c(gamma, 3, 1), x)[1]   gamma = 99.999    5.9262333798
                               gamma = 100.001   5.9262971512
  ```

  a jump of 6.38e-05, of which 5.83e-05 is discontinuity rather than slope; the
  step scales with `n` and reaches 0.018 at `n = 2160`. The largest step between
  neighbouring `gamma` falls from 6.11e-05 to 2.72e-06 in the gradient and from
  1.22e-06 to 5.38e-08 in `H[gamma, gamma]`. `R::digamma` and `R::trigamma` are
  accurate to 1e-16 over the whole range, so the substitution bought nothing.

  Over 330 gradient cells the maximum relative error fell from 3.85e-04 to
  1.07e-12 and no cell got worse; over 660 Hessian cells 105 improved and 20
  moved the other way. Ten of those twenty are `H[gamma, delta]` moving by one
  ulp. The other ten are `H[gamma, gamma]` and `H[delta, delta]` at
  `gamma` or `delta = 1e12`, where the entry is a difference of two psi' values
  that agree to eleven digits: the result, around 1.8e-23, can carry no better
  than 1e-04 relative in double precision whatever psi' returns, and the
  measured relative error moves from 1.9e-05 to 7.2e-04, both inside that floor.
  The smooth asymptotic form landed inside it by luck at that one point while
  being 5.1e-04 wrong and discontinuous at the far more plausible `gamma = 100`.
  `dmc()`, `pmc()`, `qmc()`, `rmc()` and `llmc()` are bit-identical.

* **`llmc()` swapped `R::lbeta` for a difference of `lgamma` above
  `gamma = 100` or `delta = 100`** (`bpmc.cpp`): that difference is the
  cancellation `R::lbeta` exists to avoid. At `gamma = 1e12`, `delta = 2` the
  two outer `lgamma` values are 2.66e13, where one ulp is 3.9e-03, so their
  difference cannot resolve an answer of -82.2 any better than that:

  ```
  R::lbeta(1e12, 3)                          -82.1999161672287   (exact)
  lgamma(1e12) + lgamma(3) - lgamma(1e12+3)  -82.203125          (off by 3.2e-03)
  ```

  `dmc()` and `llbeta()` always called `R::lbeta`, so `llmc()` also disagreed
  with `-sum(dmc(..., log = TRUE))`, the objective it is supposed to be, and
  with `llbeta()` at `lambda = 1`, where the two are the same model:

  ```
  llmc(c(1e12, 2, 1), 1 - 1e-12)   before -25.9371543959   exact -25.9378518132
  llmc(c(2, 1e12, 1), 1e-12)       before -26.6306976341   exact -26.6310211159
  ```

  `R::lbeta` is now called at every `gamma` and `delta`. Over 110
  likelihood cells the maximum error fell from 16283 ulps of the working
  magnitude to 3.00, which is where the branch-free regimes already sat; 91
  cells are bit-identical, 16 improved and 2 moved by at most 2 ulps.
  `dmc()`, `pmc()`, `qmc()`, `rmc()`, `grmc()` and `hsmc()` are bit-identical.

* **`dmc()` lost the density as `x` approached 1** (`bpmc.cpp`): it formed
  `x^lambda` in linear arithmetic and then took `log(1 - x^lambda)`. Doubles are
  spaced 2.2e-16 apart just below 1, so `1 - x^lambda` carries an absolute error
  of one ulp of 1 however small it truly is, and `x^lambda` rounds to exactly 1
  once `1 - x` drops under about 1e-16, at which point a guard returned a density
  of zero.

  `Mc(gamma, delta, lambda)` is `GKw(1, 1, gamma, delta, lambda)`, so `dgkw()`
  with `alpha = beta = 1` is the same density computed a different way, and it
  was already right:

  ```
  gamma = 1.5, delta = 2, lambda = 0.8
                       exact (400 digits)   dgkw(x,1,1,..)          dmc
    x = 1 - 1e-13         -58.65464965      -58.65464965     -58.65409479
    x = 1 - 1e-15         -67.86721101      -67.86721101     -67.92355276
    x = 1 - 1e-16         -72.26166017      -72.26166017     -71.81537306
  ```

  `log(1 - x^lambda)` now goes through `gkw_log1mexp(lambda * log(x))`, the
  helper `dgkw()` already uses, and the guard is gone.

  `llmc()` and `grmc()` carried the mirror image of the same defect at the other
  end of the support: `log(-expm1(u))` has to represent a number just below 1
  and so reported `log(1 - x^lambda)` as a multiple of 1.11e-16 -- usually as
  exactly 0 -- for every `x^lambda` under one ulp. At `delta = 1e12` the missing
  term is worth 2e-05 nats an observation. All four functions now share
  `gkw_log1mexp()`.

  Adjudicated against a 120-digit `decimal` reference over 9130 density cells
  (22 parameter settings x 415 quantiles from 5e-324 to 1 - 1.1e-16): 8424 cells
  are bit-identical, 512 improved, 143 moved by at most 4 ulps of the working
  magnitude, and the maximum error fell from infinite -- six cells returned
  `-Inf` for a finite density -- to 10 ulps. The nesting identity
  `dmc(x, g, d, l) == dgkw(x, 1, 1, g, d, l)` closed from 9.8e13 ulps to 10.
  `pmc()`, `qmc()`, `rmc()`, `hsmc()`, `dgkw()`, `llgkw()` and `llbeta()` are
  bit-identical across the whole grid.

* **`dgkw()`, `llgkw()` and `grgkw()` broke down along the same log-space
  chain** (`gkw.cpp`): all three walk `v = 1 - x^alpha`, `w = 1 - v^beta`,
  `z = 1 - w^lambda`, and each lost the chain in its own way.

  `llgkw()` computed `log(x^alpha)` as `vec_safe_log(vec_safe_pow(x, alpha))`, a
  round trip that both lost digits and made it disagree with `dgkw()`, which
  already used `alpha * log(x)`.

  Two of the three transformations underflow to a boundary that `log1mexp()`
  cannot recover from: its argument arrives as exactly 0 and `log(1 - exp(0))`
  is `-Inf`. With a zero coefficient in front -- `delta = 0`, or
  `gamma*lambda = 1` -- `0 * -Inf` is `NaN`:

  ```
  llgkw(c(1, 300, 1, 0, 1), c(.8, .85, .9, .95))   NaN, for an exact 2609.84
  ```

  Both regimes have a first-order limit that is exact to the last representable
  bit: as `x -> 0`, `log_w = log(beta) + alpha*log(x)`; as `x -> 1`,
  `log_z = log(lambda) + beta*log_v`. These are now used where the direct form
  underflows, and a coefficient of exactly zero never multiplies a logarithm.

  `grgkw()` built `1/v`, `1/w` and `1/z` as separate reciprocals, each of which
  overflowed on its own long before the product it belonged to was large:

  ```
  par = (1, 70, 1.5, 2, 1), x = c(.10, .25, .40, .72, .99)
    llgkw           1386.78983044        (finite, correct)
    grgkw           NaN NaN NaN NaN NaN
    numDeriv::grad  -662.27 20.27 -6.76 472.41 -15.00
  ```

  Correcting `LOG_DBL_MAX` in 1.1.6 moved that boundary out by 2.3x but did not
  remove it; `beta = 70` recovered while `beta = 200` still failed. Every ratio
  is now a single `exp()` of a difference of logs, so only the difference has to
  be representable, not the reciprocal.

  Over 96 parameter/data blocks: `llgkw()` was non-finite in 30 and is now
  non-finite in none; `grgkw()` returned `NaN` in 30 and now in none. Adjudicated
  against a 900-digit reference and against `grbkw()`, an independent
  implementation of the same gradient at `lambda = 1`: the maximum error in
  `llgkw()` fell from 8.7e-05 to 7.5e-16, and `grgkw()` agrees with `grbkw()` to
  5.4e-08. `dgkw()` dropped 812 spurious `-Inf` log-densities across eight
  parameter settings while leaving every already-finite value bit-identical.

  Note on references: `numDeriv` is not usable as an arbiter in part of this
  region. For `beta = 1000` the intermediate `log_w` becomes subnormal, with 15
  significant bits left, and `lambda * log_w` is then bit-identical for
  `lambda = 1 +/- 1e-6` -- the finite difference sees no dependence at all and
  reports a `lambda` component short by exactly `delta/lambda`. The analytic
  value is correct there; `grbkw()` and the 900-digit reference confirm it.

* **`grkw()` and `hskw()` were not the gradient and Hessian of `llkw()`**
  (`kw.cpp`): `kw.cpp` was the last family file whose derivatives were still
  evaluated in linear arithmetic. It formed `v = 1 - x^alpha` and then applied
  `arma::clamp(v, eps, 1 - eps)` with `eps = 2.22e-14`, freezing `log(v)` at
  `-31.4384832` for every observation near 1 regardless of the data:

  ```
  par = (0.5, 2), x = 1 - 1e-14        d/dalpha            d/dbeta
    grkw                            -2.44999999999999   30.938483203129
    numDeriv::grad(llkw)            -3.99999999995549   32.430138079912
    grekw(lambda = 1), same dist.   -3.99999999999997   32.430138079907
  ```

  The relative error reached 38.75% in the gradient, 47% in `H[alpha,alpha]` and
  78% in `H[alpha,beta]`, and ordinary data was enough to show it:
  `c(1-1e-9, 1-1e-11, 0.5)` already diverged by 3.4e-05. Standard errors and
  confidence intervals were wrong whenever the sample held observations close
  to 1.

  Both routines are now `grekw()` / `hsekw()` with `lambda` fixed at 1, evaluated
  from logarithms. Adjudicated over 80 parameter/data blocks against two
  independent references: the maximum relative disagreement with the EKw path
  fell from 3.93 to exactly 0 in the gradient and from 4.00 to 1.4e-14 in the
  Hessian, and against `numDeriv` from 3.93 to 1.6e-09, which is numDeriv's own
  noise. No block moved in the wrong direction and no sign changed.

* **`llgkw()` returned `-Inf` for data outside the open support** (`gkw.cpp`):
  `ll*()` is the negative log-likelihood, so an invalid point must be `+Inf` --
  the value `optim()` moves away from. `llgkw()` returned `-Inf`, making data
  outside `(0, 1)` the global minimum of the function being minimised, and it
  was the only one of the seven families with that sign; `llbkw()`, `llkkw()`,
  `llekw()`, `llkw()`, `llmc()` and `llbeta()` all returned `+Inf`. The parameter
  path in `llgkw()` was already `+Inf` and is unchanged.

  The practical damage was in comparing likelihoods rather than in optimisation:
  `optim()` refuses to start at either infinity, so no optimiser silently
  converged on bad data. But on a sample holding a single `0` -- ordinary in
  untransformed proportions -- the GKw family won every comparison:

  ```
  gkw  nll = -Inf     <- wins        argmin = gkw
  bkw  nll =  Inf                    AIC    = -Inf
  ...  nll =  Inf
  ```

  Values for valid data are bit-identical, and the nesting identities against
  `llkw()`, `llbkw()` and `llekw()` still agree to 1e-12.

* **Zero-length arguments crashed the R process** (all seven families): the
  vectorised `d*()`, `p*()`, `q*()` and `r*()` routines size their output as the
  maximum length of their inputs and then recycle with `i % vec.n_elem`. When one
  argument had length zero while another did not, the output length stayed at one
  or more and the recycling evaluated `i % 0`. Integer division by zero is
  undefined behaviour; on x86-64 it raises `SIGFPE`, terminating the R process
  with no error, no message and nothing for `tryCatch()` to catch. The R-level
  validation did not intercept it either, because `any(numeric(0) <= 0)` is
  `FALSE`. All 28 exported routines now short-circuit before the loop:
  `d*()`, `p*()` and `q*()` return `numeric(0)`, matching
  `stats::dbeta(numeric(0), 1, 1)`, and `r*()` return `n` missing values with a
  warning, matching `stats::rbeta(3, numeric(0), 1)`. A filtered vector that
  happened to be empty, such as `dkw(x[x > 1], 2, 3)`, was enough to trigger the
  crash. Numerical output is unchanged for every non-empty input.

* **`rbkw()` and `rkkw()` generated values outside the open support**
  (`gkw.cpp`, `bkw.cpp`, `kkw.cpp`, `ekw.cpp`, `kw.cpp`): `rbkw()` drew
  `V ~ Beta(gamma, delta+1)` and then formed `1.0 - V`. For `V` below `1.1e-16`
  that rounds to exactly 1 and the generator returned 0. `R::rbeta` itself never
  returned a zero -- every one was fabricated by the subtraction:

  ```
  rbkw(1e5, 2, 3, 0.02, 0)    48,602 exact zeros    48.6% of the sample
  rbkw(1e5, 2, 3, 0.05, 0)    16,440 exact zeros
  rkkw(1e5, 0.2, 3, 0, 0.3)        6 exact zeros
  ```

  A zero is outside the `(0,1)` the likelihood accepts, so the package's own
  simulate-then-fit workflow broke on six zeros in a hundred thousand:
  `llkkw()` at the *true* parameters returned `Inf` and `optim()` stopped with
  "L-BFGS-B needs finite values of 'fn'". Kolmogorov-Smirnov against the
  package's own CDF rejected the `gamma = 0.02` sample outright, `D = 0.486`.

  The five generators that formed `1 - u` -- `rgkw()`, `rbkw()`, `rkkw()`,
  `rekw()`, `rkw()` -- now invert in log space, the same chain the quantile
  functions use. `rmc()` and `rbeta_()` never had the defect and are untouched.

  The draws themselves are unchanged, so `set.seed()` reproduces exactly the
  stream it did before: `.Random.seed` after 100,000 variates is bit-identical
  for all seven generators. Only the inversion that follows differs. Replaying
  the same Beta draws, 97,559 of 100,000 `rbkw()` values changed and every one
  moved closer to the closed-form inversion, none away; the largest relative
  error falls from `1.0` to `7.2e-15`, and for `rkkw()` from `1.0` to `8.5e-14`.
  The Kolmogorov-Smirnov statistic for `gamma = 0.02` goes from `D = 0.486`
  (`p < 1e-16`, 24,277 variates outside the support) to `D = 0.0042`,
  `p = 0.336`, none outside. The fit now converges and recovers the parameters.

* **All seven quantile functions returned values outside the open support**
  (`gkw.cpp`, `bkw.cpp`, `kkw.cpp`, `ekw.cpp`, `bpmc.cpp`, `kw.cpp`,
  `beta_.cpp`): each `q*()` undid `log.p` with `exp()`, folded the upper tail
  with `1 - p`, and then inverted using `1 - u` in linear space at every step.
  The result was not merely imprecise -- it left `(0,1)` altogether:

  ```
  qekw(0.02, 20, 0.1, 0.1)       returned 0   true value 0.1587
  qekw(1e-08, 5, 2, 0.5)         returned 0   true value 5.49e-04
  qkw(1e-16, 0.2, 2)             returned 0   true value 3.125e-82
  qbeta_(-1000, 2, 3, log.p=T)   returned 0   true value 2.25e-218
  ```

  A quantile of exactly 0 or 1 then feeds `d*()` and `ll*()` a value outside the
  support they accept, so the damage propagates into simulation by inversion and
  into any likelihood built on it.

  The inversion now carries `log(u)` and `log(1-u)` from whatever scale and tail
  the caller used, so neither is recovered by subtraction, and the four families
  that route through the incomplete beta hand `lower_tail`/`log_p` to
  `R::qbeta`. `qbkw()` needs `log(1-z)`: it takes `log1p(-z)` while `z <= 1/2`
  and otherwise gets `1-z` directly from `R::qbeta` through the symmetry
  `I_z(a,b) = 1 - I_{1-z}(b,a)`.

  Judged against closed-form inversions written independently in R: of 22,236
  values, 8,311 changed and every one improved. The largest relative error falls
  from `2.06` to `5.7e-14`. The boundary conventions of 1.1.5 are preserved
  exactly, including the saturating result for out-of-range `p`.

  One limit remains: below about `log(p) = -745`, `exp(log u)` underflows, `1-u`
  rounds to exactly 1 and the inversion has nothing left to invert, so the
  quantile is still 0. Recovering it needs each step to carry both `log(q)` and
  `log(1-q)`. 1.1.5 already returned 0 from `log(p) = -40` downward.

* **All seven cumulative distribution functions collapsed to 0 or 1**
  (`gkw.cpp`, `bkw.cpp`, `kkw.cpp`, `ekw.cpp`, `bpmc.cpp`, `kw.cpp`,
  `beta_.cpp`): each `p*()` formed `1 - x^alpha` and `1 - (1 - x^alpha)^beta`
  in linear space. Once `x^alpha` fell below `1.1e-16` the first rounded to
  exactly 1 and the second to exactly 0, and the CDF returned 0 or 1. The error
  was absolute, not a lost digit:

  ```
  pekw(5.62e-09, 2, 5, 0.02)   returned 0     true value 0.483
  pekw(0.14, 20, 20, 0.1)      returned 0     true value 0.0264
  pkw(1e-09, 2, 5, log.p=TRUE) returned -Inf  true value -39.84
  ```

  The second sits at `x = 0.14`, nowhere near a tail: a small `lambda`
  compresses the result toward 1 and pulls the collapse into the body of the
  distribution. A systematic sweep found 8,917 affected points for `pgkw()`
  alone, with a maximum absolute error of 1.0 -- the largest a probability can
  be wrong by.

  `lower.tail` and `log.p` were also applied afterwards, as `1 - p` and
  `log(p)`, instead of being passed to `R::pbeta`, which implements both without
  ever forming those quantities. That cost the opposite tail:
  `pmc(1 - 1e-06, 2, 3, 2.5, lower.tail = FALSE)` returned exactly 0 against a
  true `1.95e-22`, and `pbeta_(1e-200, 2, 3, log.p = TRUE)` returned `-Inf`
  against a true `-918.73`.

  Every chain now runs in log space through `gkw_log1mexp()`, the survival
  function is computed directly rather than as `1 - F`, and `lower_tail`/`log_p`
  go straight to `R::pbeta` where the family routes through the incomplete beta.

  Judged against log-space references written independently in R: of 61,100
  values over a grid spanning `1e-300` to `1 - 1e-16`, 23,720 changed, 23,719
  improved and one moved by a single ulp. The largest relative error falls from
  `5.9e+305` to `1.6e-01`, and only 14 values of the 61,100 still exceed `1e-9`.
  Those 14 sit at `x >= 0.999`, where `x^lambda` is within an ulp of 1 and the
  argument handed to `R::pbeta` cannot carry more precision. All seven CDFs are
  monotone, stay within `[0, 1]`, satisfy `F + S = 1` to `1.1e-16`, and
  reproduce their nesting identities exactly.

* **The McDonald log-likelihood was unbounded below, and silently rewrote the
  data** (`src/bpmc.cpp`): `llmc()`, `grmc()` and `hsmc()` clamped every
  observation to `[1e-10, 1-1e-10]` before use. For a legitimate observation at
  `1e-20` that moved the likelihood by 23 nats, and it broke the identity
  `llmc(gamma, delta, 1) == llbeta(gamma, delta)`, where the two are the same
  model: the disagreement reached 1140 nats.

  Separately, for `delta > 1000` the term `delta * log(1 - x^lambda)` was
  floored at `-700` per observation, so it stopped growing with `delta` while
  the constant term `n(log lambda - log B(gamma, delta+1))` kept growing. The
  negative log-likelihood became **unbounded below**:
  `llmc(c(1e300, 1e300, 1e-6), x)` returned `-2.77e+302`, a global minimum at
  absurd parameters, with a visible step at `delta = 1000`. It now returns
  `+2.58e+303`.

  `grmc()` and `hsmc()` additionally floored `v = 1 - x^lambda` at `1e-10` and
  capped their lambda terms at `±1e6`, so the gradient plateaued where the
  objective kept moving. `llmc()` also computed `log(1 - x^lambda)` as
  `log1p(-x^lambda)`, which cannot recover digits `x^lambda` has already lost,
  while `grmc()` and `hsmc()` already used `-expm1(lambda * log(x))`; the
  objective and its gradient therefore disagreed as `x` approached 1. All three
  now use `-expm1` of the same exponent.

  These clamps were removed together rather than one at a time: `llmc()` shared
  them with `grmc()` and `hsmc()`, so removing only the documented subset would
  have introduced an objective/gradient mismatch that did not previously exist.

  Because the three functions shared the same clamps, checking the analytic
  gradient against `numDeriv::grad(llmc)` passed even with the defect present.
  Validation therefore used a closed-form reference written independently in R.
  Against it, the largest relative error falls from `Inf` to `1.0e-14` for
  `llmc()`, from `1.12` to `3.1e-05` for `grmc()`, and `hsmc()` now agrees with
  the jacobian of the analytic gradient to `6.0e-08`. The nesting identity with
  `stats::dbeta()` goes from 1140 nats of error to `9.1e-13`. Maximum-likelihood
  fits on well-behaved data are bit-identical, under both BFGS with the analytic
  gradient and Nelder-Mead.

  The residual `3.1e-05` in `grmc()` appears only for `gamma + delta > 100` and
  is unchanged by this commit: it comes from the asymptotic digamma expansions,
  a separate defect.

* **`hsgkw()` silently returned the Hessian of a smaller sample**
  (`src/gkw.cpp`): the observation loop skipped past any point whose
  `log(1-x^alpha)`, `log(1-v^beta)` or `log(1-w^lambda)` came out non-finite,
  leaving the remaining terms to be returned as a finite, symmetric matrix with
  no `NaN` and no warning. For a quantity whose purpose is to produce standard
  errors, that is the worst available failure mode. With `beta = 500` and four
  observations every point was dropped and only the parameter-only terms
  survived, so `H(alpha, alpha)` came back as `n / alpha^2 = 4` against a true
  `1996.3` -- wrong by a factor of 499, and indistinguishable from a valid
  result. With five observations one survived and the function returned the
  Hessian of a single point as if it described all five.

  The loop now stops on the first such observation and returns a `NaN` matrix
  with a warning, matching what the function's own intermediate-value check
  already did. Matrices are bit-identical wherever they were finite before.
  Computing those terms correctly requires the log-space rework and is not
  attempted here. `grgkw()` is unaffected: it is fully vectorised and propagates
  `NaN` rather than dropping observations.

* **`dgkw()` returned a density of zero as `x` approached 1** (`src/gkw.cpp`):
  the density formed `x^alpha` in linear space and bailed out whenever
  `x^alpha >= 1 - sqrt(.Machine$double.eps)`. The guard was there because
  `log(x^alpha)` loses its significant digits in that band -- doubles are spaced
  `2.2e-16` apart near 1, so the relative error reaches `4e-6` by
  `1 - x = 1e-12` -- but returning zero is a far worse answer than an imprecise
  one. It also broke the nesting identity: `dgkw(1 - 1e-9, 1, 0.1, 1, 0, 1)`
  returned 0 while `dkw(1 - 1e-9, 1, 0.1)`, the same density, returned
  `1.26e+07`. For `GKw(0.1, 0.1, 10, 0.1, 0.1)` the discarded band held 13% of
  the probability mass, and for `beta < 1`, where the density diverges at 1, the
  rising tail was replaced by a cliff to zero.

  `log(x^alpha)` is now taken as `alpha * log(x)`, which is exact and removes the
  need for the guard; `gkw_log1mexp()` already covers the resulting regime. Over
  the regression grid, 6,780 of 47,104 `dgkw()` values changed, every one closer
  to an independent log-space reference and none further away; no other family
  moved. Recovered mass shows up in the integral: `GKw(0.1, 0.1, 10, 0.1, 0.1)`
  goes from 0.8671 to 0.9998.

  Two `continue` guards further down `dgkw()` still discard a point when
  `log(1 - w^lambda)` underflows, even where `delta = 0` makes that term vanish
  from the density. That is unchanged here and belongs with the log-space rework.

* **Two logarithmic bound constants held the wrong quantity** (`src/utils.h`):
  `LOG_DBL_MAX` was documented as `log(DBL_MAX_SAFE)` but held
  `log10(DBL_MAX) = 308.2547`, while the correct natural logarithm is `707.4801`.
  Since it is used as the overflow threshold of `safe_exp()` and `safe_pow()`,
  every result above `exp(308.25)` was returned as `+Inf`, discarding roughly 174
  orders of magnitude of representable double range. Reachable from the public
  API: `dkw(1e-300, 0.5, 2)` returned `Inf` instead of `1e+150`.

  Separately, `safe_log()` scaled its underflow branch by `LOG_DBL_MIN`, which is
  `log(DBL_MIN)`, while dividing by `DBL_MIN_SAFE`, which is `10 * DBL_MIN`. Every
  result below `2.225e-307` was therefore off by exactly `log(10) = 2.302585` --
  a finite, plausible, wrong number rather than a visible failure. It propagated
  into `dkw(x, log = TRUE)`, `llkw()`, `llgkw()` and `pmc(log.p = TRUE)`, and made
  `llgkw()` disagree with `dgkw()`, which takes `log(x)` directly.

  The constants are now named for what they are -- `LOG_DBL_MIN`,
  `LOG_DBL_MIN_SAFE` and `LOG_DBL_MAX` -- and `safe_log()` scales by the logarithm
  of the divisor it actually uses. Over a regression grid of 401,373 values, 328
  density values, 16 log-likelihoods and 3 tail probabilities changed; every one
  moved closer to an independent log-space reference, and none moved away. The
  largest relative error against that reference fell from `Inf` to `1.6e-16` for
  densities and from 4.5% to `1.2e-10` for `llgkw()`. As a side effect the
  all-`NaN` region of `grgkw()` recedes: with `x_max = 0.99` it began at
  `beta = 80` and now extends past `beta = 130`, with the newly finite values
  agreeing with `numDeriv::grad()` to 1.3e-9 or better.

  `safe_exp()` still saturates above `log(DBL_MAX_SAFE)`, i.e. one order of
  magnitude below the true double maximum. That headroom is the documented intent
  of the `DBL_MAX_SAFE` constant and is left in place.

* **`gkwgetstartvalues()` never ran the multi-start it documents; `n_starts` was
  inert** (`gkwinit.cpp`): the selection loop decided whether to optimize a
  starting point by comparing the *raw* objective at that point against
  `best_obj`, which after the first iteration already held an *optimized* value.
  A start that was merely poor -- exactly the case multi-start exists to rescue --
  was therefore discarded before Nelder-Mead ever saw it, and in practice only
  the first of the candidates was optimized at all. Every value of `n_starts`
  returned the same answer, bit for bit:

  ```
  set.seed(202); x <- rgkw(500, 5, 1.2, 3, 0.5, 2)
  gkwgetstartvalues(x, "gkw", n_starts = k)   moment objective
    k =   1                                     6.262579e-04
    k =  10                                     6.262579e-04   (identical)
    k = 200                                     6.262579e-04   (identical)
  after
    k =   1                                     1.008584e-07
    k =  10                                     1.369536e-08
    k = 200                                     1.037278e-10
  ```

  Every candidate is now optimized and only the optimized objectives are
  compared. A second defect surfaced once the multi-start was live: Nelder-Mead
  is unconstrained and can leave the family's parameter box, so a winner chosen
  on its pre-clamp objective could be handed back worse than a rival that stayed
  inside, making the returned error non-monotone in `n_starts`. Candidates are
  now clipped to the box *before* being scored, so the objective that is compared
  is the objective of the vector that is returned.

  The practical failure was worse than a suboptimal start. The single optimized
  path ran into a degenerate corner in which the numerical integral of the
  density underflows, `moment_theoretical()` falls back to its closed-form
  Kumaraswamy moment, and the optimizer is rewarded for parameters whose real
  moments are nothing like the sample's. In 15 of 168 sweep cases the returned
  vector scored the maximum possible objective of exactly 3.0 -- all five
  relative moment errors equal to 1:

  ```
  set.seed(3050); x <- rgkw(50, 2, 3, 1.5, 2, 0.8)      sample mean 0.296299
  before  alpha = 0.100000 (pinned at the lower bound), beta = 9.874503,
          gamma = 0.574504, delta = 0.131984, lambda = 1.276909
          theoretical mean 1.401418e-06,  objective 3.000000
  after   alpha = 0.773802, beta = 3.322900, gamma = 3.662037,
          delta = 0.670337, lambda = 1.437505,  objective 4.830e-07
  ```

  Downstream the damage reached the fits themselves. Starting `optim()` from that
  corner, 8 of 10 GKw samples of size 500 converged to a *positive* negative
  log-likelihood -- around +4100 to +4900 where the correct region is near -300:

  ```
                          nll from old start   nll from new start
    gkw seed 1                    4117.357            -295.479
    gkw seed 4                    4829.810            -297.731
    gkw seed 10                   4855.630            -284.388
  ```

  Verified over 168 cases (seven families x n in {50, 200, 1000} x 8 seeds) at
  the default `n_starts = 5`: 77 improved, 91 unchanged, none worse, worst ratio
  exactly 1.000000. The 15 cases at the maximum objective of 3.0 fell to none,
  and the largest objective over the sweep fell from 3.0 to 5.876e-04. Over 70
  MLE fits driven from the two sets of starting values, the largest improvement
  in the attained negative log-likelihood was 5230.4 nats and the largest
  regression 0.09 nats, on a `kkw` sample that settled in a neighbouring local
  optimum; mean relative parameter error fell from 0.507 to 0.409. The full test
  suite is unchanged at 0 failures.

  Cost: `n_starts` now buys what it claims, so it also costs what it claims. At
  the default `n_starts = 5` a GKw call goes from 0.058s to 0.27s (n = 300); at
  `n_starts = 1000`, from 0.09s to 50s. The default is unchanged. Four fixed,
  family-specific starting points are always used, so `n_starts` below 4 still
  behaves as 4; this is now documented rather than silently true.

* **`gkwgetstartvalues()` truncated out-of-support data without saying so**
  (`gkwinit.cpp`): every observation was clamped into `[1e-10, 1 - 1e-10]` in
  silence. Truncation moves every sample moment and therefore every estimate the
  function returns, and its commonest cause -- data on a percentage or 0-100
  scale -- is exactly the case a caller needs to be told about. On that input the
  function did not fail; it answered, and the answer was both wrong and
  unremarkable-looking:

  ```
  set.seed(1); y <- rkw(300, 2, 3)
  gkwgetstartvalues(y, "kw")                alpha 2.137549   beta 3.506511
  gkwgetstartvalues(c(y, 5, -3), "kw")      alpha 2.047567   beta 3.251306
  gkwgetstartvalues(y * 100, "kw")          alpha 50.000000  beta 50.000000
  ```

  The last line is the whole problem in one row: a sample handed over on a 0-100
  scale came back with both parameters pinned at the upper edge of the parameter
  box, with nothing to distinguish it from a fit. `tryCatch(..., warning = )`
  caught no condition in any of the three calls.

  Observations outside the open interval `(0,1)` -- exact 0 and exact 1 included,
  matching the support `ll*()` enforces since the fix earlier in this release --
  now raise a warning naming how many were truncated and the range they spanned:

  ```
  gkwgetstartvalues: 300 of 300 observations lie outside the open interval (0,1)
  (observed range [6.6169, 89.7703]) and were clamped to it; the estimates below
  are those of the clamped sample. Data on a percentage or 0-100 scale must be
  rescaled before use.
  ```

  The clamp itself is kept, so a single boundary observation still does not abort
  a fit and no existing call changes its return value; only the silence is
  removed. `NA` and non-finite values continue to be dropped without a warning,
  which the `@param x` entry now states.

  Verified: the warning fires on `c(y, 5, -3)`, on a lone exact 0, on a lone
  exact 1 and on `y * 100`, reporting 2, 1, 1 and 300 offenders respectively with
  the correct observed range in each; it does not fire on the clean sample, nor
  on samples carrying `NA` or `Inf`. Every returned vector is unchanged. The full
  test suite, whose data are all strictly inside the support, still reports 0
  warnings.

## Documentation Fixes

* **Two sub-family constraints in the package overview were mathematically
  wrong** (`R/gkwdist-package.R`): the "Distribution Family Hierarchy" block
  gave Kumaraswamy as "GKw with \eqn{\gamma = \delta = \lambda = 1}" and the
  uniform as "set all shape parameters to 1". In this parameterization
  \eqn{\delta} is neutral at **0**, not 1 -- it enters through
  \eqn{B(\gamma, \delta+1)} and \eqn{z^{\delta}} -- so both statements name a
  different distribution than the one they claim:

  ```
  claim                                        max |difference|
  Kw(2.3, 3.1)  vs  GKw(2.3, 3.1, 1, 1, 1)              0.887
  Kw(2.3, 3.1)  vs  GKw(2.3, 3.1, 1, 0, 1)              0        <- correct
  Uniform       vs  GKw(1, 1, 1, 1, 1)                  0.960
  Uniform       vs  GKw(1, 1, 1, 0, 1)                  0        <- correct
  ```

  This is the same off-by-one that the deleted `_pkgdown.yml` block carried,
  and it survived there because the overview was the one place where the
  constraints were never checked against the implementation. Every other
  statement of a sub-family constraint in the package -- the `\itemize` list in
  `dgkw()`'s details, the `@details` of each family, and the `desc` lines in
  `_pkgdown.yml` -- was verified numerically in this pass and is correct. The
  uniform entry now says explicitly why \eqn{\delta = 0} rather than
  \eqn{\delta = 1} is the neutral value, since that is what the two wrong
  statements had in common.

* **The `@return` of all seven densities denied the boundary contract**
  (`R/gkw.R`, `R/bkw.R`, `R/kkw.R`, `R/ekw.R`, `R/bpmc.R`, `R/kw.R`,
  `R/beta.R`): once the closed boundaries began carrying the limiting density,
  the seven blocks still promised `0` -- or `-Inf` on the log scale -- for `x`
  "outside the interval (0, 1)". No rendered page mentioned the limit at all,
  and 0 and 1 lie outside that open interval, so the sentence denied precisely
  the two points the change had added:

  ```
  call                              documented   returned
  dgkw(c(0, 1), 0.5, 0.5, 1, 0, 1)       0 0      Inf Inf
  dkw(c(0, 1), 2, 1)                     0 0          0 2
  dbeta_(c(0, 1), 0.5, 0)                0 0      Inf 0.5
  dgkw(c(-0.1, 1.1), 2, 3, 1, 0, 1)      0 0          0 0
  ```

  The last row is the case that really does return `0`, and the wording is
  narrowed to it: "strictly outside the interval [0, 1]". Each block now states
  that `x = 0` and `x = 1` carry the limiting density, names `stats::dbeta` as
  the base R convention being followed -- with the `shape2 = delta + 1` shift
  spelled out in `dbeta_()`, whose parameterization differs -- and records that
  the limit is `0`, a finite positive value, or `Inf` according to the
  parameters.

  Documentation only; no executable code changes and no numerical result is
  affected.

* **The out-of-support warning was undocumented in all seven `ll*()`**
  (`R/gkw.R`, `R/bkw.R`, `R/kkw.R`, `R/ekw.R`, `R/bpmc.R`, `R/kw.R`,
  `R/beta.R`): the guard that warns `'data' contains values outside (0, 1)`
  appeared in no help page, so the one signal separating a mis-scaled sample
  from a genuine fit failure was invisible to a reader of the documentation.
  All seven `@return` blocks now record it, with the reason: an infinite
  objective offers an optimiser no gradient direction to follow.

  In the same pass `llgkw()` loses "returns a large positive value (e.g.,
  `Inf`)", the only one of the seven that did not name `Inf` exactly. Measured,
  all seven return `Inf` exactly, for an invalid `par` and for out-of-support
  `data` alike.

  Documentation only; no executable code changes and no numerical result is
  affected.

* **The Cordeiro & de Castro (2011) citation was truncated in 41 of its 42
  appearances** (the six family files that cite it -- `R/kw.R` does not): the
  entry existed in two broken
  shapes -- 14 ended at the journal name with no volume, pages or full stop,
  and 27 ended at a dangling comma. Only `dgkw()` carried it complete. All 42
  now read `*Journal of Statistical Computation and Simulation*, *81*(7),
  883-898.`

  In the same pass, Carrasco, Ferrari & Cordeiro (2010) -- the primary source
  for the five-parameter distribution this package implements -- is added to
  the `@references` of the seven GKw topics. Their `@details` credited it in
  prose while `@references` listed only the two secondary sources; the earlier
  attribution fix reached the prose and not the list. The six sub-families are
  left alone: each defines itself as a special case of GKw and links to
  `dgkw()`, so the attribution reaches them through that link.

* **References were not linkable outside the package overview** (all seven
  family files, `src/gkwinit.cpp`): `\doi{}` appeared on 7 citations in
  `R/gkwdist-package.R` and nowhere else, so 103 citations across the 50
  function topics rendered as plain text. Every citation whose DOI was already
  recorded in the package -- Carrasco (2010), Cordeiro & de Castro (2011),
  Jones (2009), Kumaraswamy (1980) and McDonald (1984) -- now carries it, taken
  verbatim from `R/gkwdist-package.R` and `DESCRIPTION`. Every topic that has a
  `\references` section now has at least one DOI.

  Nadarajah, Johnson/Kotz/Balakrishnan and Devroye are deliberately left
  without one: no DOI for those works is recorded anywhere in this package, and
  a DOI is not something to reconstruct from memory. `gkwgetstartvalues()`'s
  Jones citation, the only one still in plain text with no `\emph`, is brought
  into the same form as the other seven, and McDonald's volume number is
  italicised to match every other volume number in the package.

* **The package overview page was generated but unreachable**
  (`R/gkwdist-package.R`): `_PACKAGE` carried `@keywords internal`, which
  removes a topic from the help index and from the pkgdown reference.
  `docs/reference/gkwdist-package.html` was being built and nothing linked to
  it, so the whole overview -- family hierarchy, performance notes,
  model-selection workflow, four worked estimation examples -- could be reached
  only by typing the URL. The keyword is dropped and the topic is added to
  `_pkgdown.yml` under a `Package Overview` heading, which is also what keeps
  `pkgdown::check_pkgdown()` clean once the topic is no longer internal.

* **Ten of the thirteen `\keyword` entries were outside R's controlled
  vocabulary** (all seven family files): `density`, `cumulative`, `quantile`,
  `random`, `likelihood`, `gradient`, `hessian`, `beta`, `kumaraswamy` and
  `mcdonald` are absent from `R/doc/KEYWORDS`, so they indexed nothing. The
  seven that name a role become `@family` groupings -- density, cumulative
  distribution, quantile, random generation, log-likelihood, gradient and
  Hessian functions, seven members each -- which roxygen2 renders as
  bidirectional "Other ...:" links and pkgdown exposes as concepts. The three
  that name a distribution become `@concept`. `distribution` and `optimize`,
  which are standard, are kept.

  This is the axis the hand-written `\seealso` blocks did not cover: they link
  each function to the siblings of its own distribution, never across
  distributions, so nothing led from `dgkw()` to the other six densities. The
  curated blocks are untouched and the generated lists are appended below them.

  Only `beta`, `kumaraswamy` and `mcdonald` existed as distribution keywords.
  The four families that had none -- `generalized kumaraswamy`,
  `beta-kumaraswamy`, `kumaraswamy-kumaraswamy` and `exponentiated
  kumaraswamy` -- are given the matching concept, so all 49 topics now carry
  one family and one distribution concept.

* **Every example in the package sat inside `\donttest{}`** (all seven family
  files): 51 of 52 topics wrapped their entire `@examples` block, so
  `R CMD check` without `--run-donttest` -- the form run locally and in most
  CI configurations -- executed no example at all, while `--as-cran` ran all
  9,217 lines regardless. The wrapper bought nothing and hid everything.

  Measured on one machine, the 28 `d`/`p`/`q`/`r` blocks take 0.05 s in total
  and the 21 `ll`/`gr`/`hs` blocks take 41 s. The wrapper is removed from the
  first group and kept on the second, where the cost is real and where the
  weakly-identified fits behind the confidence-region entry below make the
  examples platform-sensitive. The default check now exercises 1,515 lines of
  examples instead of none.

* **`gkwgetstartvalues()` had no `@seealso`** (`src/gkwinit.cpp`): the one
  topic whose output is meant to be fed straight into other functions of this
  package linked to none of them. It now points at the seven `ll*()` objectives
  it seeds and at `stats::optim()`.

* **Help page titles followed five competing patterns** (all seven family
  files): `dmc()` read "Beta Power Distribution Distribution"; the GKw CDF,
  quantile and RNG topics put the role after the distribution ("Generalized
  Kumaraswamy Distribution CDF") where the other six put it first; "CDF of the"
  and "Cumulative Distribution Function (CDF) of the" both appeared, as did
  "Random Generation for" and "Random Number Generation for", and
  "Negative Log-Likelihood" took "for the", "for" and "of the" in different
  topics. The KKw family was written `kkw` in all seven of its titles while the
  README, `_pkgdown.yml` and the package overview write `KKw`.

  Sixteen titles are repaired -- the duplicated word, nine against the
  majority form, and the seven KKw case fixes below -- leaving all 49 on
  exactly seven patterns, one per role. Titles that abbreviate the distribution
  in the gradient and Hessian topics are left abbreviated: those are the
  longest titles in the package and spelling the family out would push them
  past 120 characters.

  The lower-case `kkw` is corrected in the prose of `R/kkw.R` as well, 27
  occurrences across descriptions, details, `@return` blocks, `@seealso`
  labels and example plot titles. Three kinds of `kkw` are deliberately left
  alone, because they are identifiers rather than the name of the
  distribution: the file references `src/kkw.cpp`, the exported function names
  (`dkkw`, `pkkw`, `qkkw`, `rkkw`, `llkkw`, `grkkw`, `hskkw`), and the
  `family = "kkw"` argument value that users pass to
  `gkwgetstartvalues()`.

* **Eight verifications inside `@examples` were commented out** (all seven
  family files): six `q*()` topics computed a round trip
  `p -> q*() -> p*()`, printed both numbers, and left the assertion
  `abs(p_check - p_recalc) < 1e-9` commented. `dgkw()` and `pgkw()` were worse:
  each built `pdf_beta_check` / `cdf_beta_check` against `stats::dbeta()` /
  `stats::pbeta()` and then commented out the only line that used it, leaving a
  variable computed for nothing. All eight now run and print. Checked before
  enabling: the two `stats` comparisons agree to 4.4e-16 and 3.3e-16, and every
  round trip is exact to at worst 1.1e-16.

* **The `intro` navbar entry pointed at nothing** (`_pkgdown.yml`,
  `vignettes/`): pkgdown fills "Get started" from `vignettes/<package>.Rmd`,
  and the introductory vignette was `into-gkwdist.Rmd` -- "into" for "intro" --
  so the entry was silently dropped and the published site had no "Get started"
  link. The file is renamed `gkwdist.Rmd`, which fixes the spelling and
  activates the entry in one move. Its title and `VignetteIndexEntry` are
  unchanged. **Links to `articles/into-gkwdist.html` and calls to
  `vignette("into-gkwdist")` will no longer resolve.**

* **Half of `_pkgdown.yml` was a superseded copy of itself** (`_pkgdown.yml`):
  131 of its 284 lines were a commented-out earlier reference layout. It was
  not a
  duplicate -- it carried the nesting relation for each sub-family, which the
  active version had dropped -- and two of those relations were wrong, giving
  EKw as "GKw with γ = δ = 1" and Kw as "GKw with γ = δ = λ = 1" where δ = 0 is
  the neutral value in this parameterization and is what every `@details` block
  and every function default states. The nesting is carried over to the active
  `desc` lines with δ = 0, and the dead block is removed.

* **No `inst/WORDLIST`** (new file): `spelling::spell_check_package()` reported
  191 words and was therefore unusable as a check. A curated list of 45 -- the
  family abbreviations, the cited authors, the institutions, terms such as
  `digamma`, `trigamma`, `unimodality` and `bimodality`, and the three
  fragments the `<doi:...>` markup and the quoted package name leave in
  `DESCRIPTION` -- now leaves the `.Rd` files and `DESCRIPTION` reporting
  nothing, so a real typo will show up. Nothing consumes the list
  automatically: there is no `tests/spelling.R` and `spelling` is not in
  `Suggests`, so it serves whoever runs the check by hand.

* **The author name was rendered two ways** (`R/gkwdist-package.R`): 49 topics
  say "Lopes, J. E." and the package overview said "J. E. Lopes". The overview
  now matches the rest.

* **Editorial hedging in the rendered help pages** (all seven family files):
  29 `\seealso` entries qualified functions that exist and are exported --
  "(if these exist)", "(gradient, if available)", "(other functions for this
  parameterization, if they exist)". A CRAN help page should not speculate about
  its own package's contents. The qualifiers are removed and the informative
  half of each label is kept, so `\code{grbkw} (gradient, if available)` reads
  `\code{\link{grbkw}} (gradient)`.

  41 cross-references in those same blocks used `\code{}` where `\link{}` was
  meant, so they rendered as plain text and the help pages could not be
  navigated between. All 50 distinct `\link{}` targets now resolve to an alias
  in the package.

  Documentation only; no executable code changes and no numerical result is
  affected.

* **`gkwgetstartvalues()` is deterministic, and now says so** (`gkwinit.cpp`):
  the help page described "multiple random starting points" without stating that
  the randomness is internal and fixed. The extra starting points come from a
  generator seeded with a constant, so `set.seed()` has no effect on the returned
  value and `.Random.seed` is neither read nor advanced:

  ```
  set.seed(1);    a <- gkwgetstartvalues(x, "gkw", 20)
  set.seed(9999); b <- gkwgetstartvalues(x, "gkw", 20)
  identical(a, b)                                        TRUE
  .Random.seed unchanged across the call                 TRUE
  the caller's next runif(1) unchanged                   TRUE
  ```

  This is deliberate and is being documented, not changed. The function is a
  method-of-moments estimator whose output seeds optimisers elsewhere in a fit;
  two calls on the same data must agree, or every downstream fit would inherit a
  dependence on the ambient seed and would silently consume draws the caller did
  not ask to spend. The lever for a wider search is `n_starts`, which since the
  fix above is both effective and monotone -- more starts can only lower the
  objective -- so a seed argument would add a lottery where a monotone control
  already exists. A `Determinism` paragraph in `Details` now states all three
  facts, the `@examples` block demonstrates them, and a comment at the generator
  in `gkwinit.cpp` records the intent so the constant seed is not mistaken for an
  oversight.

  The `set.seed(123)` opening the example is correct and is kept: it makes the
  `rbeta()` sample on the next line reproducible. Its comment now says which of
  the two calls it governs.

  Documentation only; no numerical result changes.

* **Confidence-region examples were undrawable where the observed information
  was not positive definite** (29 `@examples` blocks across the seven families):
  every one built a confidence region from
  `eigen(solve(hs*(mle, data))[1:2, 1:2])` and then took
  `diag(sqrt(eig_decomp$values))`, with nothing to guarantee the eigenvalues were
  non-negative. `solve()` of an observed information matrix is a covariance
  matrix only where that information is positive definite; `optim()` reports
  `convergence = 0` on a flat likelihood ridge without establishing it. When an
  eigenvalue came back negative, `sqrt()` produced `NaN`, the whole region became
  `NaN`, and `plot()` aborted with `need finite 'xlim' values`.

  The fit these examples rest on is weakly identified -- the observed information
  has a condition number between `4.4e+06` and `1.3e+07` -- so which side of the
  boundary it lands on depends on the BLAS and the optimiser's path. The examples
  passed on Linux and macOS and failed on Windows under `--run-donttest`.

  `eigen()` is now called with `symmetric = TRUE`, which is what a covariance
  matrix warrants and which keeps the eigenvalues real and ordered, and the
  eigenvalues are clamped at zero, so the region degenerates rather than
  vanishing. Reproduced against the indefinite block directly: 500 of 500 ellipse
  coordinates were `NaN` before and none are after, and `plot()` raises the same
  `need finite 'xlim' values` before and succeeds after.

  Documentation only; no executable code in `R/` or `src/` is changed, and every
  numerical result is unaffected.

* **`grmc()` gradient formula had inverted digamma signs** (`R/bpmc.R`): the
  `@details` block documented `psi(gamma + delta + 1) - psi(gamma)` for the gamma
  component and `psi(gamma + delta + 1) - psi(delta + 1)` for delta. Since
  `d log B(gamma, delta+1) / d gamma = psi(gamma) - psi(gamma + delta + 1)`, both
  signs were reversed, and the documented formula disagreed with the returned
  value by two orders of magnitude. `R/beta.R` documented the opposite sign for
  the same quantity. The implementation was correct throughout; only the
  documentation is changed. The same block's Hessian entry for
  `d2l/dgamma ddelta` in `hsmc()` had the same sign reversal.

* **README example 6 inverted the sign of the observed information matrix**
  (`README.Rmd`): `hsekw()` already returns the Hessian of the negative
  log-likelihood, so negating it again produced a negative definite matrix and
  printed `NaN` for every asymptotic standard error. Example 3 of the same README
  and the vignettes were already correct.

* **GKw attribution corrected** (`R/gkw.R`): the main help page credited Cordeiro
  & de Castro (2011), which introduces the Kw-G family, rather than Carrasco,
  Ferrari & Cordeiro (2010), which introduces the five-parameter generalized
  Kumaraswamy distribution implemented here. `DESCRIPTION`, the README and the
  vignettes already cited the latter.

## Packaging

* **`RcppArmadillo` moved out of `Imports`.** It was listed there only because
  `R/zzz.R` carried `@import RcppArmadillo`, which put `import(RcppArmadillo)`
  in `NAMESPACE`. Armadillo is header-only for a client package: everything
  gkwdist uses from it is compiled into `gkwdist.so` at install time through
  `LinkingTo`, where `RcppArmadillo` already appeared. Loading its R namespace
  at run time bought nothing, and `R CMD check --as-cran` reported
  `Package in Depends/Imports which should probably only be in LinkingTo:
  'RcppArmadillo'`. The tag, the `NAMESPACE` entry and the `Imports` line are
  gone; `LinkingTo` is untouched, so the build is unchanged.

* **`numDeriv` moved from `Imports` to `Suggests`.** No function in `R/` calls
  `grad()` or `hessian()`; the package's own derivatives are analytic and live
  in `src/`. `numDeriv` is used only by the test suite, as the independent
  reference the analytic `gr*()` and `hs*()` routines are checked against, and
  by `\seealso` cross-references in the help pages, which `Suggests` keeps
  valid. Every test that calls it is guarded by `skip_if_not_installed()`, so
  the suite runs to completion without it. Installing gkwdist no longer pulls
  `numDeriv` in.

* **The `utils::globalVariables()` registration in `R/zzz.R` is gone.** It
  listed 39 names, all 39 of which also appear in the sibling package
  `gkwreg`'s own registration -- regression-diagnostic artefacts such as
  `cook_dist`, `leverage`, `linpred` and `model_label` that no distributions
  package produces, plus `"::"`, `":::"` and `"log"`, which are functions, not
  variables. The list had been copied across and never pruned. With the whole
  call removed, `R CMD check` still reports
  `checking R code for possible problems ... OK`, so not one of the 39 was
  suppressing a real finding. Removing it restores the check's ability to
  notice a genuine undefined global in future.

* **New `.github/workflows/sanitizers.yaml` runs the compiled code under
  runtime sanitizers.** CI had none: no ASan, UBSan, valgrind or rhub job
  anywhere. That gap is what let the worst defect of this release through. The
  `SIGFPE` from `i % 0` on zero-length input, which killed the R process
  outright, produced no diagnostic under `-Wall -Wextra -Wformat=2`, passed
  `R CMD check` cleanly, and kept the whole five-platform `R-CMD-check` matrix
  green. UBSan names it on the first call, with a stack trace:

  ```
  gkw.cpp:134:21: runtime error: division by zero
      #0 ... in dgkw(...) src/gkw.cpp:134
      #1 ... in _gkwdist_dgkw   src/RcppExports.cpp:415
  ```

  The workflow has two jobs: UBSan on stock R with GCC, which links `libubsan`
  into `gkwdist.so` and needs no instrumented R, and R-hub's `clang-asan`
  container, which adds AddressSanitizer. Both are `continue-on-error: true`
  for now, so a finding reports without blocking a pull request.

## Deprecated

* **The re-exported pipe, `%>%`, is deprecated and will be removed in a future
  release. Nothing changes in this one:** it is still exported and still
  behaves exactly as before.

  gkwdist does not use the pipe anywhere -- not in `R/`, not in the tests, not
  in the vignettes. It re-exports `magrittr`'s operator and nothing else, which
  is the sole reason `magrittr` is a hard dependency, so every installation of
  a distributions package pulls in a package it never calls. R has had a native
  pipe, `|>`, since 4.1.0, and users who want `%>%` can attach it from its own
  source.

  This is announced a release ahead rather than done now because
  `export("%>%")` is public API: code that reads `library(gkwdist)` and then
  uses `%>%` without attaching magrittr or a tidyverse package would stop
  working the moment the export went away, with `could not find function
  "%>%"`. If your code depends on gkwdist supplying it, switch now to
  `library(magrittr)` (or any tidyverse package that re-exports it), or to
  `|>`. The change is invisible if you already attach magrittr yourself.

## Testing

* `test-return-contract.R` -- sweeps every parameter of every family through
  `d`, `p`, `q` and `r` with `-1`, `0`, `NA` and `NaN`,
  asserting they take the identical route, and asserts separately
  that `delta = 0` is still accepted -- `delta` is the one parameter whose
  bound is `>= 0`, and the only place the change could have gone off by one.
  Also covers the out-of-support warning in all seven `ll*()`. Fails 60
  assertions and errors on 1 against the preceding commit.

* New `tests/testthat/test-zero-length-input.R` covers all 28 routines with
  zero-length data and zero-length parameters, the empty-subset idiom, and the
  correspondence with the `stats` package's convention.

* New `tests/testthat/test-deep-tail-precision.R` pins the subnormal and
  large-density regimes against a log-space reference. It fails 37 assertions
  against 1.1.5.

* New `tests/testthat/test-density-near-upper-bound.R` pins `dgkw()` in the band
  the old guard rejected, together with the nesting identities and the total
  mass. It fails 17 assertions against 1.1.5.

* New `tests/testthat/test-hessian-degenerate.R` pins the degenerate cases of
  `hsgkw()` and checks that healthy parameters keep their finite, symmetric
  matrices. It fails 8 assertions against 1.1.5.

* New `tests/testthat/test-mcdonald-no-clamping.R` pins `llmc()`, `grmc()` and
  `hsmc()` against a closed-form reference, checks the Beta nesting identity and
  the boundedness of the objective, and fits a model end to end. It fails 45
  assertions against 1.1.5.

* New `tests/testthat/test-cdf-log-space.R` pins the seven CDFs against
  log-space references, checks monotonicity, range, `F + S = 1`, the nesting
  identities and agreement with the integral of the density. It fails 9
  assertions against 1.1.5.

* New `tests/testthat/test-quantile-log-space.R` pins the seven quantiles
  against closed-form inversions, checks that they stay inside `(0,1)`, that
  `p(q(u))` recovers `u`, that the boundary conventions are unchanged and that
  the nesting identities hold. It fails 28 assertions against 1.1.5.

* New `tests/testthat/test-startvalues-contract.R` pins the three contracts
  `gkwgetstartvalues()` advertises: that `n_starts` widens the search and never
  worsens the fit, that the estimate reproduces the first sample moment, that
  the answer is deterministic and leaves `.Random.seed` alone, and that
  truncating out-of-support data raises a warning naming how many observations
  were moved. It fails 25 assertions against 1.1.5 and passes 54 on this branch.

* `test-mle-performance.R` compared each scenario's mean parameter error over
  its own converged subset, which penalises the more robust scenario: the
  analytical gradient fits datasets the numerical baseline gives up on, and
  those are the hard ones. After the generators changed, a `gkw` run converged
  on 4 reps the baseline could not touch, whose mean error was 16.1 against 0.50
  on the 23 shared reps, and the reported ratio went from 0.82 to 4.66. The
  gradient was in fact the better of the two throughout -- it also reached a
  lower negative log-likelihood in 21 of those 23 reps. The comparison is now
  paired over the reps where both converged, which is what the data generation
  in that file was already written for. The corrected check passes against
  1.1.5, against the previous commit and against this one.

* New `tests/testthat/test-rng-log-space.R` checks that every generator stays
  inside `(0,1)`, that `rbkw()` reproduces the closed-form inversion of its own
  replayed draw, that `set.seed()` still reproduces, that the sample passes a
  Kolmogorov-Smirnov test against its own CDF, and that simulate-then-fit runs
  end to end. It fails 8 assertions against 1.1.5.

* New `tests/testthat/test-argument-contract.R` covers the two surfaces the
  suite had never touched: the roughly 218 documented `stop()` conditions in
  the R wrappers, and non-finite input. It asserts every bound on every shape
  parameter of every `d*`/`p*`/`q*`/`r*`, the `n` guard, the `log`,
  `lower.tail` and `log.p` guards, and the `par`-length and `data` guards of
  every `ll*`/`gr*`/`hs*`; then that `NA_real_`, `NaN`, `+Inf` and `-Inf` are
  accepted without error, return one double per input element and leave their
  finite neighbours untouched. 526 assertions, and it fails 11 of them against
  1.1.5: `qkw(NA)` returned 1 there, and `q*(NaN)` collapsed to `NA` in six of
  the seven families. The log-space quantile inversion in this release fixed
  both, and this file pins them.

  Six of its tests are marked `skip()`. They assert the behaviour the
  functions *should* have for non-finite input -- `d*(NA)` giving `NA` rather
  than `0`, `p*(+Inf)` giving `1` rather than `0`, `q*` outside `[0, 1]`
  giving the `NaN` its own warning promises, and the `log = NA` and NA-shape
  -parameter holes -- and will fail 86 assertions until that defect is fixed.
  They are the specification, written down and executable; the fix removes
  the `skip()` line and nothing else.

* `tests/testthat/test-derivatives-validation.R` had 69 tests where its own
  header promises 70: BKw was missing Hessian config 3. Restored.

* `tests/testthat/test-loglikelihood-functions.R` asserted
  `expect_true(result < 0)` on all seven families, commented "Log-likelihood
  should be negative". The `ll*()` functions return the *negative* log
  likelihood, whose sign is not a property of anything -- `llkw(c(1,1), .)` is
  0 on a uniform sample and `llkw(c(2,2), .)` is +2.1. Each test now asserts
  the defining identity, `ll*(par, data) == -sum(d*(data, ..., log = TRUE))`.

# gkwdist 1.1.5

## Critical Bug Fixes

* **`dgkw()` returned zero for every input** (`gkw.cpp`, `utils.h`): the package's
  numerical helpers `log1mexp()` and `log1pexp()` collide with functions of the same
  name in R's public `Rmath.h` API, which use the opposite convention
  (`log(1 - exp(-x))` for `x >= 0`). In translation units where `Rmath.h`'s macro was
  active, calls bound to R's version, which returns `NaN` for the negative arguments
  used here; every density evaluation then failed its finiteness guard and returned 0.
  The helpers are now named `gkw_log1mexp()` and `gkw_log1pexp()`. The sub-family
  densities were unaffected, as was `llgkw()`, which routes through `vec_log1mexp()`.

* **Log-likelihoods of EKw, KKw and BKw were wrong for data near zero**
  (`ekw.cpp`, `kkw.cpp`, `bkw.cpp`): these routines clamped `v = 1 - x^alpha` and
  `w = 1 - v^beta` at `1e-10` instead of working in log space. For small `x` and
  moderate `alpha`, `x^alpha` rounds to 1 in double precision and `w` collapses to
  zero, so the clamp replaced `log(w) = -53` by `log(1e-10) = -23`. Deviations
  reached 6,100 log-units, which silently corrupts AIC, BIC and likelihood ratio
  tests. All three families now use the same `gkw_log1mexp()` formulation as
  `gkw.cpp`, and their scores and Hessians are expressed as ratios of logarithms.

* **Mixed second derivatives were zeroed at degenerate parameter values**
  (`bkw.cpp`, `ekw.cpp`, `kkw.cpp`): guards of the form `if (abs(p - 1) > eps)`
  gated mixed partial derivatives that do not carry the vanishing factor. Because
  `d2l/dalpha dgamma` is obtained by differentiating `(gamma-1)*log(w)` once in
  `gamma`, the `(gamma-1)` factor is consumed and the term survives at `gamma = 1`.
  `hsbkw()` returned 0 where the correct value was 271.12; `hsekw()` and `hskkw()`
  had the same defect at `beta = 1`.

* **`grkkw()` clamped gradient terms at 1000** (`kkw.cpp`): arbitrary
  `std::min(..., 1000.0)` caps distorted the score in `beta` by up to 5%. The same
  clamp appeared as `effective_delta` inside `llkkw()`, capping the likelihood for
  `delta > 1000`. This is the defect removed from `ekw.cpp` in 1.1.3, which had
  survived here.

* **`grkkw()` and `hskkw()` skipped the `z` block at `delta = 0`** (`kkw.cpp`):
  the shortcut omitted `sum(log(z))` from `dl/ddelta` and zeroed
  `d2l/dalpha ddelta`, `d2l/dbeta ddelta` and `d2l/ddelta dlambda`, none of which
  carry a `delta` factor. `delta = 0` is a valid interior value of the likelihood.

* **The Beta sub-family rejected `delta = 0`** (`utils.h`): `check_beta_pars()`
  required `delta > 0`, unlike the other five validators. Since the sub-family is
  parameterised as `Beta(gamma, delta + 1)`, `delta = 0` is the legitimate
  `Beta(gamma, 1)` boundary; `dbeta_()`, `pbeta_()`, `qbeta_()`, `rbeta_()`,
  `llbeta()`, `grbeta()` and `hsbeta()` all returned `NA`/`Inf` there.

## Validation

* **`test-boundary-derivatives.R`** (new): every gradient component and every
  Hessian entry of all seven sub-families is now compared individually against two
  independent references, the general GKw routines restricted to the constrained
  parameter point and `numDeriv` Richardson extrapolation, over grids that include
  the degenerate values `gamma = 1`, `beta = 1`, `lambda = 1` and `delta = 0` and
  samples containing observations near zero.

* **`test-density-correctness.R`** (new): densities are checked to integrate to one,
  to agree with the general GKw density at the constrained parameter point, to match
  base R for the Beta and closed-form Kumaraswamy cases, and to be the derivative of
  the corresponding distribution function. The previous PDF tests asserted only type,
  length, non-negativity and finiteness, all of which a vector of zeros satisfies.

## Accuracy after the fixes

Componentwise maximum relative error over 720 parameter configurations per family,
against the general GKw routines and against `numDeriv`:

| Family | log-likelihood | gradient | Hessian |
|:-------|---------------:|---------:|--------:|
| GKw    | 1.5e-11 | 1.5e-08 | 1.5e-07 |
| BKw    | 1.6e-12 | 3.7e-08 | 3.6e-08 |
| KKw    | 3.1e-13 | 3.5e-08 | 4.9e-08 |
| EKw    | 5.4e-14 | 7.8e-09 | 2.7e-08 |
| Mc     | 3.5e-13 | 2.5e-09 | 9.5e-10 |
| Kw     | 4.1e-15 | 7.2e-10 | 1.2e-09 |
| Beta   | 7.2e-15 | 4.6e-10 | 8.9e-11 |

The residual gradient and Hessian errors are at the accuracy limit of Richardson
extrapolation itself; against the GKw reference all seven families agree to 1e-13.

## Documentation and Project Infrastructure

* **`inst/paper/`**: the JOSS manuscript was rewritten. It now positions the
  package explicitly as the distribution layer of the GKw ecosystem, records that
  the split from `gkwreg` was made at the request of JOSS reviewers during that
  package's review, reports the measured validation and timing results in place of
  the previous unverified figures, and follows the current JOSS AI disclosure
  policy. The bibliography was expanded to 19 entries with DOIs verified against
  Crossref.

* **`CONTRIBUTING.md`** and **`CODE_OF_CONDUCT.md`** (new): contribution workflow,
  support expectations, governance, and the testing standard numerical
  contributions are held to. The contributing guide documents the cross-check that
  makes bug reports actionable: every sub-family routine must agree with the
  general GKw routine evaluated at the corresponding constrained parameter point.

* **`README`**: the claim that the C++ routines are "10-50x faster than equivalent
  R implementations" was replaced with measured figures. The original benchmark
  compared `-sum(log(dkw(x, 2, 3)))` against `llkw()`, which is C++ against C++
  plus R loop overhead, and gives roughly 3x. The genuine gain is in the
  derivatives: the analytical score is about 9x faster than Richardson
  extrapolation and the analytical Hessian about 38x faster at n = 20,000.

* **`inst/CITATION`**: updated to version 1.1.5 and pointed at the CRAN canonical
  URL; it had been stale at version 1.0.8.

* Test coverage rose from 70.8% to 74.2% of combined R and C++ lines. The largest
  single gain is in `src/gkw.cpp` (42.9% to 71.5%), which reflects that `dgkw()`
  now executes its density computation instead of falling through to its
  finiteness guard.

# gkwdist 1.1.4

## CRAN Fix

* **`test-mle-performance.R`**: Added `skip_on_cran()` to all timing-based benchmark
  tests. These tests compare wall-clock times of analytical vs. numerical gradients and
  are inherently unreliable on shared/loaded CRAN check machines, causing spurious
  `ERROR` results. The tests remain available for local development.

# gkwdist 1.1.3

## Bug Fixes

* **`llgkw()` invalid parameter return** (`gkw.cpp`): Fixed critical error where the
  negative log-likelihood returned `R_NegInf` (−∞) for invalid parameters instead of
  `R_PosInf` (+∞). Gradient-based MLE optimizers interpret −∞ as a global minimum,
  causing them to converge to the invalid boundary rather than the true MLE.

* **`gkwinit.cpp` — delta validation** (`gkwinit.cpp`): Fixed internal `gkw_pdf()`
  rejecting `delta = 0` (a valid GKw parameter value) due to a strict `delta <= 0`
  check that should have been `delta < 0`.

* **`gkwinit.cpp` — EKw/Kw sub-family PDF mapping** (`gkwinit.cpp`): Fixed
  `ekw_pdf()` and `kw_pdf()` passing `delta = 1` instead of the correct `delta = 0`
  when delegating to `gkw_pdf()`. EKw and Kw are GKw sub-families with `delta = 0`,
  not `delta = 1`. This produced wrong starting values for MLE of these families.

* **`hsbkw()` — v^(β−1) computation** (`bkw.cpp`): Fixed the Hessian of the BKw
  negative log-likelihood returning a wrong value for β < 1. The ternary expression
  `(beta > 1.0) ? v_beta/v : 1.0` coincidentally produces the correct result for
  β = 1 but is wrong for all 0 < β < 1. Replaced with the exact formula
  `safe_exp((beta - 1.0) * ln_v)`.

## Numerical Stability

* **`safe_exp()` underflow scaling** (`utils.h`): Fixed a systematic 10× error in
  the moderate-underflow branch. The previous implementation used
  `DBL_MIN_SAFE * exp(x − log(DBL_MIN))` where `DBL_MIN_SAFE = 10 * DBL_MIN`,
  yielding `10 * exp(x)` instead of `exp(x)`. The fix uses
  `DBL_MIN * exp(x − log(DBL_MIN)) = exp(x)` exactly.

* **`dgkw()` silent boundary truncation removed** (`gkw.cpp`): Removed a block that
  silently skipped data points within `SQRT_EPSILON^(1/α)` of 0 or 1, returning
  density 0 for those points without warning. The log-space computation handles
  near-boundary values correctly without this truncation.

* **`llekw()` / `grekw()` — lambda clamping removed** (`ekw.cpp`): Removed the
  arbitrary cap `lambda_factor = min(lambda_factor, 1000)` applied to gradient and
  Hessian terms when λ > 1000. This distorted optimization for large-λ scenarios and
  produced incorrect standard errors.

## Code Quality

* **`gkwinit.cpp`**: Removed `using namespace Rcpp;` at file scope; replaced with
  explicit `Rcpp::` qualifications. Added NA/NaN filtering before moment computation
  to prevent silent corruption when input data contains missing values.

* **`bkw.cpp`**: Removed spurious `try/catch` blocks wrapping `Rcpp::as<arma::vec>()`
  conversions in `grbkw()` and `hsbkw()`. These conversions cannot throw in this
  context and the silent fallback masked type errors.

* **`gkw.cpp` / `ekw.cpp`**: Refactored Hessian accumulation to build only the upper
  triangle inside the observation loop and symmetrize once afterwards with
  `arma::symmatu()`, eliminating O(n × p²) redundant assignments.

* **`utils.h` — `vec_safe_pow()` UB guard**: Added guard preventing undefined
  behaviour when casting large `y_rounded` values (> `INT_MAX`) to `int` for
  odd-exponent sign detection.

* **`utils.h` — `vec_safe_pow()` SIMD fast path**: Added an early-return path
  `arma::exp(y * arma::log(x))` for the common case (y > 0, all x > 0) that
  is fully auto-vectorizable, improving throughput in gradient/Hessian evaluation.

# gkwdist 1.1.2

## Code Cleanup and Testing Enhancement

### C++ Code Cleanup

* **Removed legacy commented code**: Cleaned up all C++ source files (`gkw.cpp`, `bkw.cpp`, `kkw.cpp`, `ekw.cpp`, `kw.cpp`, `bpmc.cpp`, `beta_.cpp`) by removing old commented-out implementations that were kept for reference.
* **Code formatting**: Improved R wrapper formatting with consistent indentation and alignment in `.Call()` invocations and roxygen examples.

### New Test Suites

* **Analytical derivatives validation** (`test-derivatives-validation.R`):
  - 70 comprehensive tests validating gradient (`gr*`) and Hessian (`hs*`) functions
  - Compares analytical derivatives against numerical differentiation via `numDeriv`
  - Covers all 7 subfamilies: GKw, BKw, KKw, EKw, Mc, Kw, Beta
  - Multiple parameter configurations per subfamily for robustness

* **MLE performance benchmarks** (`test-mle-performance.R`):
  - Compares optimization efficiency across three scenarios: numerical-only, analytical gradient, and analytical gradient + Hessian
  - Validates that analytical derivatives provide equivalent or better accuracy
  - Tests convergence rates and computational time across all distribution families

### JOSS Paper

* **Added paper for JOSS submission** (`inst/paper/`):
  - Complete manuscript describing the package's statistical framework
  - Comprehensive bibliography with foundational references
  - Compiled PDF ready for submission

---

# gkwdist 1.1.1

## Major Refactoring Release

This release represents a comprehensive refactoring of the entire package codebase, focusing on numerical stability, code consistency, and maintainability.

### C++ Backend Overhaul

* **Unified utility functions**: Introduced `utils.h` header providing numerically stable implementations of critical functions:

  - `log1mexp()`: Stable computation of log(1 - exp(x)) using Mächler (2012) methodology
  - `log1pexp()`: Overflow-protected computation of log(1 + exp(x))
  - `safe_log()`, `safe_exp()`, `safe_pow()`: Protected arithmetic operations with graceful handling of edge cases
  - Vectorized versions (`vec_safe_log`, `vec_log1mexp`, etc.) for efficient array operations

* **Consistent parameter validation**: All distribution families now use dedicated parameter checkers (`check_pars()`, `check_kw_pars()`, `check_ekw_pars()`, etc.) that properly handle NaN, Inf, and boundary conditions.

* **Complete documentation**: All C++ source files now include comprehensive Doxygen-style documentation headers describing:
  - Mathematical formulas for PDF, CDF, quantile, and random generation
  - Parameter constraints and special cases
  - Numerical stability considerations
  - Relationship to parent GKw distribution

### Bug Fixes

* **Fixed critical bug in `qgkw()`**: Corrected logic error where `lower_tail` transformation was incorrectly applied when `log_p = TRUE`. The probability is now properly converted to linear scale before tail adjustment.

* **Fixed gradient calculation in `grkkw()`**: Resolved issue where `log_z` was not recomputed after clamping `z` to minimum threshold, causing corrupted gradient values near boundaries.

* **Fixed Hessian calculation in `hsmc()`**: Corrected sign errors and formula for the lambda component of the Hessian matrix for the Beta-Power/McDonald distribution.

* **Fixed gradient signs in `grmc()`**: Ensured consistent computation of log-likelihood gradient before negation for optimization.

### Code Quality Improvements

* **Eliminated unused variables**: Removed declared but unused constants (`exp_threshold`) and intermediate variables across all distribution files.

* **Removed redundant calculations**: Streamlined computations, notably in `pgkw()` where logarithm was computed twice for the same quantity.

* **Simplified parameter recycling**: Replaced double-modulo indexing pattern (`idx = i % k; vec[idx % vec.n_elem]`) with direct single-modulo access (`vec[i % vec.n_elem]`) in random generation functions.

* **Standardized function signatures**: All distribution functions now follow consistent patterns for parameter order, validation, and return value handling.

### R Wrapper Layer

* **Complete separation of R and C++ interfaces**: All exported R functions now serve as wrappers around internal C++ implementations (`.dgkw_cpp`, `.pgkw_cpp`, etc.), providing:
  - Enhanced input validation with informative error messages
  - Consistent argument checking across all distribution families
  - Proper NA/NaN propagation
  - Documentation accessible via standard R help system

### Distribution Families

All seven distribution families have been refactored with identical improvements:

| Distribution | Parameters | File |
|--------------|------------|------|
| Generalized Kumaraswamy (GKw) | α, β, γ, δ, λ | `gkw.cpp` |
| Kumaraswamy-Kumaraswamy (KKw) | α, β, δ, λ | `kkw.cpp` |
| Beta-Kumaraswamy (BKw) | α, β, γ, δ | `bkw.cpp` |
| Exponentiated Kumaraswamy (EKw) | α, β, λ | `ekw.cpp` |
| Beta-Power/McDonald (BP/Mc) | γ, δ, λ | `bpmc.cpp` |
| Kumaraswamy (Kw) | α, β | `kw.cpp` |
| Beta (GKw-style) | γ, δ | `beta.cpp` |

Each family includes: density (`d*`), distribution (`p*`), quantile (`q*`), random generation (`r*`), negative log-likelihood (`ll*`), gradient (`gr*`), and Hessian (`hs*`) functions.

### Technical Notes

* Minimum supported R version remains 3.5.0
* C++11 standard required (enabled via Rcpp plugin)
* Depends on RcppArmadillo for efficient linear algebra operations

### Acknowledgments

Special thanks to the thorough code review process that identified subtle numerical issues in edge cases, particularly for extreme parameter values and observations near distribution boundaries.

# gkwdist 1.0.7

# gkwdist 1.0.5

## Documentation Improvements

* **Enhanced Examples for Likelihood Functions**: All `ll*`, `gr*`, and `hs*` functions now include comprehensive examples demonstrating:
  - Maximum likelihood estimation with analytical gradients
  - Univariate profile likelihoods with confidence thresholds
  - 2D likelihood surfaces with confidence regions (90%, 95%, 99%)
  - Confidence ellipses with marginal intervals for parameter pairs
  - Numerical vs analytical derivative verification
  - Likelihood ratio tests and score tests

* **Professional Visualization Standards**: 
  - Consistent color scheme across all examples
  - Grid-adaptive algorithms for computational efficiency
  - Base R only - no external dependencies required

* **Complete Coverage**: Enhanced documentation for all distribution families (Kw, EKw, KKw, GKw) covering 2 to 5 parameters

* **Theoretical References**: Documentation cites foundational work by Carrasco et al. (2010), Jones (2009), Kumaraswamy (1980), and standard inference theory from Casella & Berger (2002)


# gkwdist 1.0.3
* **README.md**: Fix typos and faill link
  - Fix zzz.R file by removing useless texts

# gkwdist 1.0.2

# gkwdist 1.0.1

## Major Improvements

### Enhanced `gkwgetstartvalues()` Function
* **NEW**: Added `family` parameter to support all distribution families
  - Automatically returns correct number of parameters for each family
  - Family-specific initial value strategies for better convergence
  - Supported families: `"gkw"`, `"bkw"`, `"kkw"`, `"ekw"`, `"mc"`, `"kw"`, `"beta"`
  - Case-insensitive family names for user convenience

### Documentation Enhancements
* **README.md**: Complete rewrite with mathematical rigor
  - All LaTeX formulas corrected and verified for proper rendering
  - Eight comprehensive examples using `optim()` with analytical gradients
  - Corrected function signatures: all `ll*()`, `gr*()`, and `hs*()` functions use `(par, data)` signature
  - Added performance benchmarks demonstrating 10-50× speedup with C++ implementation
  - Hierarchical structure diagram for all distribution families
  - Model selection workflow and practical guidelines
  - Removed all references to deprecated `gkwfit()` function

### CRAN Submission Readiness
* **DESCRIPTION**: Fixed to meet CRAN requirements
  - Proper `Authors@R` field formatting
  - Removed unused dependencies (`numDeriv`)
  - Corrected package dependencies (`RcppArmadillo` only in `LinkingTo`)
  - Enhanced description with DOI references
  - Fixed maintainer email formatting

## Bug Fixes

* Fixed function call signatures in all README examples to match actual implementation
* Corrected parameter passing in optimization examples (now consistently use `(par, data)`)
* Fixed LaTeX rendering issues with `\left`/`\right` delimiters in GitHub Markdown

## Testing

* **NEW**: Comprehensive test suite using `testthat`
  - 100+ tests covering all exported functions
  - Tests for all 7 distribution families (GKw, BKw, KKw, EKw, MC, Kw, Beta)
  - PDF, CDF, quantile, and random generation tests
  - Log-likelihood, gradient, and Hessian validation
  - Parameter recovery tests with MLE
  - Edge cases and boundary condition handling
  - Integration tests for PDF-CDF consistency

## Performance

* All functions implemented in C++ for maximum computational efficiency
* Analytical derivatives (gradient and Hessian) provide exact computations
* Optimized numerical stability for extreme parameter values

## Notes

* This is the initial CRAN submission
* Package focuses exclusively on distribution functions (no high-level fitting interface)
* Companion package `gkwreg` provides regression modeling capabilities
* All user-facing functions maintain backward compatibility
* C++ implementation uses RcppArmadillo for linear algebra operations
* Analytical functions use robust log-scale computations to prevent overflow/underflow
* Random generation uses inverse CDF method where closed-form solutions exist

# gkwdist 0.1.0

## New Features

* Initial CRAN release
* Generalized Kumaraswamy distribution (5 parameters)
* Six nested sub-families: Beta, Kumaraswamy, Exponentiated-Kumaraswamy, 
  Kumaraswamy-Kumaraswamy, Beta-Kumaraswamy, and McDonald distributions
* Complete set of distribution functions (d/p/q/r)
* Log-likelihood, gradient, and Hessian functions for all families

## Performance

* Optimized C++ implementation via Rcpp
* Vectorized operations for speed
