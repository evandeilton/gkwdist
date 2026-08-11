---
title: 'gkwdist: Analytical Likelihood Derivatives for the Generalized Kumaraswamy Distribution Family in R'
tags:
  - R
  - Kumaraswamy distribution
  - bounded data
  - unit interval
  - maximum likelihood
  - analytical derivatives
  - RcppArmadillo
authors:
  - name: José Evandeilton Lopes
    orcid: 0009-0007-5887-4084
    affiliation: "1"
affiliations:
 - name: Federal University of Paraná (UFPR), Brazil
   index: 1
citation_author: Lopes, JE
date: 11 August 2026
year: 2026
bibliography: paper.bib
output: rticles::joss_article
csl: apa.csl
journal: JOSS
---

# Summary

`gkwdist` provides the distribution layer of the Generalized Kumaraswamy (GKw)
family [@carrasco2010] in R: densities, distribution and quantile functions,
random generation, and — its distinguishing contribution — exact analytical
log-likelihoods, score vectors and Hessian matrices for the five-parameter GKw
distribution and each of its seven nested sub-families. All routines are
implemented in C++ through `RcppArmadillo` [@Eddelbuettel2014; @eddelbuettel2011]
and evaluated entirely in log space, which keeps them accurate for data
concentrated near the boundaries of $(0,1)$, where the naive algebra loses every
significant digit. The analytical derivatives make gradient-based maximum
likelihood estimation, observed-information standard errors and likelihood ratio
tests available across the hierarchy without numerical differentiation.

# Statement of Need

Bounded continuous data on $(0,1)$ — proportions, rates, indices — are
ubiquitous, and the Beta [@ferrari2004; @smithson2006] and Kumaraswamy
[@kumaraswamy1980; @jones2009] distributions are the standard tools for them.
Both struggle with samples that are simultaneously asymmetric, heavy-tailed and
concentrated near a boundary. The GKw family embeds both as special cases within
a five-parameter hierarchy, so an analyst can start from a two-parameter model
and adopt extra parameters only when a likelihood ratio test justifies them.

That workflow needs three things the existing ecosystem does not supply
together: every member of the hierarchy under one interface, exact derivatives of
each log-likelihood, and numerics that survive the boundary. The third is not a
detail. For $x = 10^{-7}$ and $\alpha = 3.5$, the quantity $1 - x^{\alpha}$
rounds to exactly $1$ in double precision, so $w = 1 - (1-x^{\alpha})^{\beta}$
collapses to zero and $\log w$ is lost; clamping $w$ at a small constant instead
of using a $\log(1-e^{u})$ formulation [@machler2012] yields log-likelihoods
wrong by thousands of units, silently corrupting AIC, BIC and every likelihood
ratio test built on them. `gkwdist` computes $\log v$, $\log w$ and $\log z$
through one stable kernel and expresses every derivative as a ratio of
logarithms, so the score and Hessian inherit that accuracy.

# State of the Field

`extraDistr` [@extraDistr] supplies basic Kumaraswamy density, distribution and
quantile functions but no likelihood derivatives. `VGAM` [@yee2015] offers
Kumaraswamy regression via its `kumar()` family and `betareg` [@betareg] covers
Beta regression; both are regression packages exposing no reusable
distribution-level derivative interface. `gamlss.dist` [@rigby2019] includes Beta
variants for distributional regression, and `unitquantreg` [@unitquantreg]
implements quantile regression for several unit-interval distributions. None
covers the GKw hierarchy, and none exports analytical score and Hessian functions
that a user can pass directly to an optimiser.

`gkwdist` is deliberately the distribution layer only, and that split is not
incidental. It was made during the review of `gkwreg` [@gkwreg] at the reviewers'
request, to reduce that package's complexity and improve maintainability;
`gkwreg` 2.0.0 moved its `d`/`p`/`q`/`r` functions here and now imports them,
retaining regression with covariates and estimation through `TMB` [@tmb]. The
separation leaves the densities and derivatives usable — and independently
testable — by anyone fitting a GKw model with `optim()`, `nlminb()`, or their own
machinery, which is what made the componentwise audit described below possible.

The family arises from the Kumaraswamy-G construction [@cordeiro2011;
@nadarajah2012]. Seven sub-families follow from parameter constraints
(\autoref{tab:subfamilies}); the McDonald case recovers the generalized beta of
the first kind [@mcdonald1984], and the Beta distribution appears at
$\alpha=\beta=\lambda=1$ as $\mathrm{Beta}(\gamma,\delta+1)$.

: GKw sub-family hierarchy. \label{tab:subfamilies}

| Sub-family | Par. | Constraint | Closed quantile |
|:-----------|:----:|:-----------|:---------------:|
| GKw (`gkw`) | 5 | — | No |
| BKw (`bkw`) | 4 | $\lambda=1$ | No |
| KKw (`kkw`) | 4 | $\gamma=1$ | Yes |
| EKw (`ekw`) | 3 | $\gamma=1, \delta=0$ | Yes |
| Mc (`mc`) | 3 | $\alpha=\beta=1$ | No |
| Kw (`kw`) | 2 | $\gamma=\delta=0, \lambda=1$ | Yes |
| Beta (`beta_`) | 2 | $\alpha=\beta=\lambda=1$ | No |

# Implementation and Validation

The package exports 49 functions following R's distribution conventions:
`d`/`p`/`q`/`r` for each sub-family, plus `ll`, `gr` and `hs` for the negative
log-likelihood, its gradient and its Hessian, and `gkwgetstartvalues()` for
moment-based starting values. The theory vignette develops the score and observed
information for each sub-family in full.

Because every sub-family is the GKw distribution with parameters fixed, each
specialised routine has an exact reference: the general GKw routine evaluated at
the constrained point. The test suite exploits this. Every gradient component and
every Hessian entry is compared individually — not through an aggregate norm —
against both that reference and Richardson extrapolation [@numDeriv], over grids
including the degenerate values $\gamma=1$, $\beta=1$, $\lambda=1$, $\delta=0$
and samples with observations near zero. Those points matter: a mixed derivative
such as $\partial^{2}\ell/\partial\alpha\,\partial\gamma$ loses its $(\gamma-1)$
factor under differentiation and so does not vanish at $\gamma=1$, precisely
where a guarded implementation is most likely to be wrong. Across 720
configurations per family the maximum componentwise relative error is
$4\times10^{-8}$ against Richardson extrapolation and $10^{-13}$ against the GKw
reference; log-likelihoods match an independent log-space reference to
$2\times10^{-12}$. Densities are additionally checked to integrate to one and to
reproduce base R for the Beta and closed-form Kumaraswamy cases.

At $n = 2\times10^{4}$ the analytical score is about 9 times faster than
Richardson extrapolation and the analytical Hessian about 38 times faster, and
supplying the analytical gradient roughly halves the time to convergence in full
maximum likelihood fits. In a recovery study of 1,400 fits the optimiser reached
a log-likelihood at least as high as the one at the data-generating parameter in
essentially every replicate, while individual estimates of the four- and
five-parameter members were often far from their true values — a direct
illustration of the weak identifiability of the larger models, and the reason the
package is built to make the parsimonious members easy to fit and compare.

# Research Impact

`gkwdist` is the computational foundation of `gkwreg` [@gkwreg], published in
this journal, which declares it among its imports and relies on it for every
density, distribution and quantile evaluation. The package has been on CRAN since
November 2025 and downloaded more than 4,400 times, currently around 160 times a
month. Development has proceeded publicly since October 2025 across five tagged
releases, with continuous integration, a changelog, contribution guidelines and
an issue tracker.

# AI Usage Disclosure

Claude (Anthropic) assisted with prose editing of the documentation and this
manuscript, and with the numerical audit that compared each analytical derivative
component against independent references. The distribution theory, the derivations
of the score and Hessian, the C++ implementation and all design decisions are the
author's own. Every AI-assisted output was reviewed and validated by the author,
and all numerical claims reported here are reproducible from the scripts and test
suite in the repository.

# Acknowledgements

The author thanks Prof. Wagner Hugo Bonat (UFPR) for guidance on statistical
methodology and the R community for feedback during development.

# References
