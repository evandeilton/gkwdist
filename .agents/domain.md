# Domain Docs

How the engineering skills should consume this repo's domain documentation when exploring the
codebase.

## Where these files live in gkwdist

This is an R package whose `docs/` directory is the **pkgdown-generated website**, and
`.github/workflows/pkgdown.yaml` runs `pkgdown::build_site_github_pages()`, whose default
`clean = TRUE` deletes everything under `docs/` before rebuilding. So the usual `docs/agents/` and
`docs/adr/` locations are unsafe here: anything written there is wiped on the next site build.

Agent configuration and ADRs therefore live under **`.agents/`** at the repo root, which is listed
in `.Rbuildignore` (`^\.agents$`) so `R CMD check` doesn't flag it as a non-standard top-level
directory. Read `docs/adr/` in this file as `.agents/adr/`.

## Before exploring, read these

- **`CONTEXT.md`** at the repo root, if it exists.
- **`.agents/adr/`**: read ADRs that touch the area you're about to work in.

If any of these files don't exist, **proceed silently**. Don't flag their absence; don't suggest
creating them upfront. The `/domain-modeling` skill (reached via `/grill-with-docs` and
`/improve-codebase-architecture`) creates them lazily when terms or decisions actually get resolved.

## File structure

Single-context repo (this one):

```
/
├── CONTEXT.md          ← created lazily
├── .agents/
│   ├── issue-tracker.md
│   ├── triage-labels.md
│   ├── domain.md
│   └── adr/
│       ├── 0001-....md
│       └── 0002-....md
├── R/  src/  man/  tests/  vignettes/
└── docs/               ← pkgdown output; never write agent docs here
```

If gkwdist ever grows into a multi-context layout, the root marker is `CONTEXT-MAP.md` pointing at
one `CONTEXT.md` per context, with context-scoped ADRs beside each. That isn't the case today.

## Use the glossary's vocabulary

When your output names a domain concept (in an issue title, a refactor proposal, a hypothesis, a
test name), use the term as defined in `CONTEXT.md`. Don't drift to synonyms the glossary explicitly
avoids.

For this package that especially means the distribution vocabulary: the GKw family and its
subfamilies, the parameter names as they appear in the R signatures, and the `d`/`p`/`q`/`r`
prefixes. Match the names the exported functions actually use rather than inventing near-synonyms.

If the concept you need isn't in the glossary yet, that's a signal: either you're inventing language
the project doesn't use (reconsider) or there's a real gap (note it for `/domain-modeling`).

## Flag ADR conflicts

If your output contradicts an existing ADR, surface it explicitly rather than silently overriding:

> _Contradicts ADR-0007 (limiting densities at the support boundary), but worth reopening because…_
