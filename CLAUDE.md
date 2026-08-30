# gkwdist

R package providing the Generalized Kumaraswamy (GKw) family of distributions and its subfamilies,
with `d`/`p`/`q`/`r` functions, log-likelihood/gradient/Hessian routines, and C++ (Rcpp) internals.

## Agent skills

### Issue tracker

Issues live as GitHub issues in `evandeilton/gkwdist`, managed with the `gh` CLI.
See `.agents/issue-tracker.md`.

### Triage labels

The five canonical triage roles, each label string equal to its name.
See `.agents/triage-labels.md`.

### Domain docs

Single-context: `CONTEXT.md` at the repo root, ADRs in `.agents/adr/` — **not** under `docs/`, which
is pkgdown output and is wiped by `build_site_github_pages(clean = TRUE)`.
See `.agents/domain.md`.

### R documentation

For any work whose unit is documentation — roxygen2 blocks, `.Rd`/man pages, `NEWS.md`, README,
vignettes, the pkgdown reference index, Rd/LaTeX math (`\eqn`, `\deqn`), documenting the Rcpp
sources under `src/`, or clearing an `R CMD check` documentation NOTE/WARNING — use the
`dev-r-package-documentation` skill rather than writing the docs ad hoc.
