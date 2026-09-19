# gkwdist cheat sheet

Two pages, A4 landscape, in the style of the Posit cheat sheets.

## Build it

```sh
Rscript cheatsheet/build.R
```

That writes both files into `pkgdown/assets/cheatsheet/`, which pkgdown copies
to the site root — so the sheet is published at
[/cheatsheet/gkwdist-cheatsheet.html](https://evandeilton.github.io/gkwdist/cheatsheet/gkwdist-cheatsheet.html)
and linked from the navbar and the home sidebar.

| Path | Role |
|:--|:--|
| `cheatsheet/template.html` | **The source.** Self-contained page with `{{...}}` placeholders. Edit this. |
| `cheatsheet/build.R` | The generator. |
| `pkgdown/assets/cheatsheet/gkwdist-cheatsheet.html` | Generated. Do not edit. |
| `pkgdown/assets/cheatsheet/gkwdist-cheatsheet.pdf` | Generated. Do not edit. |

Neither directory reaches the CRAN tarball; both are in `.Rbuildignore`.

## Nothing on the sheet is typed twice

The sheet prints the package version, so a hand-maintained file would go stale
at the first release. It does not, because `build.R` fills everything in from
the package:

- **version** — read from `DESCRIPTION`;
- **logo** — inlined from `man/figures/gkwdist.png` as a data URI;
- **the six density curves** — evaluated with the package's own `dkw()` and
  `dgkw()`, through `pkgload::load_all()` when it is available, so the shapes
  track the sources rather than a stale drawing. The call that produced each
  curve is printed in its caption, and the parameters live in the `PANELS` list
  at the top of `build.R`.

The family tree, the function matrix and the prose are static and live in
`template.html`.

## The staleness guard

```sh
Rscript cheatsheet/build.R --check
```

exits non-zero if the committed HTML differs from a fresh build — a bumped
version, an edited template, a gained or lost export. The pkgdown workflow runs
it before building the site, so a release that forgets to regenerate the sheet
fails CI rather than publishing the wrong version number.

The comparison ignores the thumbnails' SVG coordinates. They are hundreds of
numbers printed to one decimal, and a last-digit difference on another machine's
libm would fail the check — and with it the site build — over a curve nobody
could see move. A density that genuinely moves is the package's regression
tests' job, not this one's.

**On release: bump `DESCRIPTION`, then run `Rscript cheatsheet/build.R` and
commit both generated files.**

## Rendering the PDF

`build.R` drives headless Chromium, taking the first of `CHROME_BIN`,
`google-chrome`, `google-chrome-stable`, `chromium`, `chromium-browser` or
`chrome` that it finds. If none is present it writes the HTML, warns, and
leaves the existing PDF alone — so `--check`, which only looks at the HTML,
still works on a machine without a browser.

Page geometry comes from `@page { size: 297mm 210mm }` in the template; no
print options beyond `--no-pdf-header-footer` are needed.

## What is on it

**Page 1 — the family.** The nesting tree (which parameter you fix to drop from
one model to the next), what each of the five parameters does, the boundary rule
for `f(0)` and `f(1)`, the 7 × 7 matrix of all 49 exported distribution and
likelihood functions, the `d`/`p`/`q`/`r` signatures and the base-R contract
they keep, a gallery of the shapes the family reaches, how this Beta differs
from `stats::dbeta`, and which family to start from.

**Page 2 — estimation.** The `optim` recipe with analytical gradient and
Hessian, `gkwgetstartvalues()`, what `ll*`/`gr*`/`hs*` return when a fit goes
wrong, the per-family `par` ordering, fitting all seven families at once, the
nested-model map with likelihood-ratio degrees of freedom, information criteria,
and fit checks.

## Editing the layout

Each page is a three-column CSS grid of fixed height, so a block that grows past
the bottom of its column is clipped rather than reflowed. After editing, check
the slack in every column — open the built HTML in a browser and run:

```js
[...document.querySelectorAll('.page')].map((p, i) => ({
  page: i + 1,
  cols: [...p.querySelectorAll('.col')].map((c, j) => {
    const col = c.getBoundingClientRect(),
          last = c.lastElementChild.getBoundingClientRect();
    return { col: j + 1, slack: Math.round(col.bottom - last.bottom) };
  })
}))
```

Every `slack` must be positive; aim for 20 px or more so print margins have room.
