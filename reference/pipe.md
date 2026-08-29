# Pipe operator

See `magrittr::%>%` for details.

## Usage

``` r
lhs %>% rhs
```

## Arguments

- lhs:

  A value or the magrittr placeholder.

- rhs:

  A function call using the magrittr semantics.

## Value

The result of calling `rhs(lhs)`.

## Deprecation notice

This re-export is deprecated and is scheduled for removal in a future
release. gkwdist does not use the pipe anywhere in its own code; it
re-exports `magrittr`'s operator only, which makes `magrittr` a hard
dependency of a package that has no need of it. Nothing changes in this
release: `%>%` remains exported and behaves exactly as before.

If your code relies on gkwdist supplying `%>%`, take it from its own
source instead – [`library(magrittr)`](https://magrittr.tidyverse.org),
or any of the tidyverse packages that re-export it – or use R's native
pipe, `|>`, available since R 4.1.0.
