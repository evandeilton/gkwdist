#!/usr/bin/env Rscript

# Build the gkwdist cheat sheet.
#
#   Rscript cheatsheet/build.R           # regenerate HTML, and the PDF if a
#                                        # Chromium binary is available
#   Rscript cheatsheet/build.R --check   # fail if the committed HTML is stale
#
# Nothing on the sheet is typed twice. The version comes from DESCRIPTION, the
# logo from man/figures/gkwdist.png, and every density curve is evaluated with
# the package's own d* functions, so a release that changes the version -- or a
# change that moves a density -- is picked up by re-running this script rather
# than by editing HTML. Edit cheatsheet/template.html, never the output.

args <- commandArgs(trailingOnly = TRUE)
check_only <- "--check" %in% args

root <- normalizePath(file.path(dirname(sub("^--file=", "", grep("^--file=",
  commandArgs(FALSE), value = TRUE)[1])), ".."), mustWork = FALSE)
if (!file.exists(file.path(root, "DESCRIPTION"))) root <- normalizePath(".")
stopifnot("run from the package root" = file.exists(file.path(root, "DESCRIPTION")))

template <- file.path(root, "cheatsheet", "template.html")
out_dir <- file.path(root, "pkgdown", "assets", "cheatsheet")
out_html <- file.path(out_dir, "gkwdist-cheatsheet.html")
out_pdf <- file.path(out_dir, "gkwdist-cheatsheet.pdf")

version <- as.character(read.dcf(file.path(root, "DESCRIPTION"), "Version")[1, 1])
message("gkwdist ", version)

# The package supplies the curves, so load the sources under development rather
# than whatever happens to be installed.
if (requireNamespace("pkgload", quietly = TRUE)) {
  suppressMessages(pkgload::load_all(root, quiet = TRUE, export_all = FALSE))
} else {
  library(gkwdist)
  if (as.character(utils::packageVersion("gkwdist")) != version) {
    warning("installed gkwdist ", utils::packageVersion("gkwdist"),
            " does not match DESCRIPTION ", version,
            "; install pkgload to build against the sources")
  }
}

# ---- density thumbnails ----------------------------------------------------
# Each panel is drawn from the package itself. `cap` clips the y axis so that a
# divergence at a boundary runs off the top of the frame -- which is what an
# infinite limit should look like -- while the interior shape stays readable.
PANELS <- list(
  unimodal = list(quote(dkw(x, 2, 5)), 2.2),
  ushape   = list(quote(dkw(x, 0.5, 0.5)), 6.0),
  jup      = list(quote(dkw(x, 3, 0.6)), 4.0),
  jdown    = list(quote(dkw(x, 0.6, 3)), 4.0),
  spike0   = list(quote(dgkw(x, 0.2, 0.4, 0.6, 5, 6)), 2.0),
  spike1   = list(quote(dgkw(x, 6, 0.2, 1.2, 2, 0.2)), 2.0)
)

W <- 154
H <- 58
PAD <- 3

thumbnail <- function(expr, cap) {
  x <- seq(1e-5, 1 - 1e-5, length.out = 200)
  d <- eval(expr, list(x = x))
  d[!is.finite(d)] <- cap * 10
  y <- pmin(d, cap)
  px <- PAD + x * (W - 2 * PAD)
  py <- (H - PAD) - (y / cap) * (H - 2 * PAD)
  pts <- paste0(sprintf("%.1f %.1f", px, py), collapse = " L")
  sprintf(paste0(
    '<svg class="thumb" viewBox="0 0 %d %d" role="img">',
    '<path class="a" d="M%.1f %.1f L%s L%.1f %.1f Z"/>',
    '<path class="l" d="M%s"/>',
    '<path class="ax" d="M%d %.1f H%d"/></svg>'),
    W, H,
    px[1], H - PAD, pts, px[length(px)], H - PAD,
    pts,
    PAD, H - PAD + 0.5, W - PAD)
}

# ---- assemble --------------------------------------------------------------
html <- readLines(template, warn = FALSE, encoding = "UTF-8")
html <- paste(html, collapse = "\n")

logo <- file.path(root, "man", "figures", "gkwdist.png")
stopifnot("logo not found" = file.exists(logo))
if (!requireNamespace("xfun", quietly = TRUE)) stop("package 'xfun' is required")

# The matrix on page 1 names every exported distribution and likelihood
# function, and its heading counts them. Take both from the namespace, so a
# gained or lost export is a build error rather than a quietly wrong sheet.
api <- grep("^(d|p|q|r|ll|gr|hs)[a-z_]+$", getNamespaceExports("gkwdist"), value = TRUE)
absent <- api[!vapply(api, function(f)
  grepl(paste0(">", f, "<"), html, fixed = TRUE), logical(1))]
if (length(absent)) {
  stop("exported but missing from the function matrix: ", paste(absent, collapse = ", "))
}
listed <- unique(unlist(regmatches(html,
  gregexpr("(?<=<td>)(d|p|q|r|ll|gr|hs)[a-z_]+(?=</td>)", html, perl = TRUE))))
if (length(setdiff(listed, api))) {
  stop("in the function matrix but not exported: ", paste(setdiff(listed, api), collapse = ", "))
}

fill <- c(
  VERSION = version,
  NFUN = length(api),
  LOGO = xfun::base64_uri(logo),
  vapply(PANELS, function(p) thumbnail(p[[1]], p[[2]]), character(1))
)
for (nm in names(fill)) {
  token <- paste0("{{", nm, "}}")
  if (!grepl(token, html, fixed = TRUE)) stop("placeholder ", token, " not in template")
  html <- gsub(token, fill[[nm]], html, fixed = TRUE)
}
left <- regmatches(html, gregexpr("\\{\\{[A-Za-z0-9_]+\\}\\}", html))[[1]]
if (length(left)) stop("unfilled placeholders: ", paste(unique(left), collapse = ", "))

if (check_only) {
  if (!file.exists(out_html)) stop("no built cheat sheet at ", out_html)
  current <- paste(readLines(out_html, warn = FALSE, encoding = "UTF-8"), collapse = "\n")

  # The comparison drops the thumbnails' SVG path data. Those are hundreds of
  # coordinates printed to one decimal, and a last-digit difference on another
  # machine's libm would fail this check -- and with it the site build -- for a
  # curve nobody could see move. What the check is for is the staleness a reader
  # would notice: the version string, an edited template, a gained or lost
  # export. Those all survive the substitution. A density that genuinely moves
  # is caught by the package's own regression tests, not here.
  curves <- function(x) gsub('(<path class="[al]" d=")[^"]*"', '\\1...."', x)
  if (!identical(curves(current), curves(html))) {
    stop("cheatsheet/ is stale: ", basename(out_html), " does not match a fresh ",
         "build for gkwdist ", version, ". Run: Rscript cheatsheet/build.R")
  }
  message("up to date")
  quit(save = "no", status = 0)
}

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
writeLines(html, out_html, useBytes = TRUE)
message("wrote ", out_html, " (", round(file.size(out_html) / 1024), " KB)")

# ---- PDF -------------------------------------------------------------------
# Page geometry lives in the template's @page rule, so the only thing asked of
# the browser is to print without adding headers of its own.
chrome <- Sys.getenv("CHROME_BIN", "")
if (!nzchar(chrome)) {
  for (cmd in c("google-chrome", "google-chrome-stable", "chromium",
                "chromium-browser", "chrome")) {
    found <- Sys.which(cmd)
    if (nzchar(found)) { chrome <- unname(found); break }
  }
}
if (!nzchar(chrome)) {
  warning("no Chromium binary found; the HTML is current but ", basename(out_pdf),
          " was not re-rendered. Set CHROME_BIN to build it.", call. = FALSE)
} else {
  status <- system2(chrome, c(
    "--headless=new", "--disable-gpu", "--no-sandbox", "--no-pdf-header-footer",
    "--run-all-compositor-stages-before-draw", "--virtual-time-budget=6000",
    shQuote(paste0("--print-to-pdf=", out_pdf)),
    shQuote(paste0("file://", normalizePath(out_html)))
  ), stdout = FALSE, stderr = FALSE)
  if (status != 0 || !file.exists(out_pdf)) stop("PDF render failed (status ", status, ")")
  message("wrote ", out_pdf, " (", round(file.size(out_pdf) / 1024), " KB)")
}
