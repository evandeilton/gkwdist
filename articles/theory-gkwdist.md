# On the Statistical Properties and Computational Inference of the Generalized Kumaraswamy Distribution Family

**Abstract.** We present a comprehensive mathematical treatment of the
Generalized Kumaraswamy (GKw) distribution, a five-parameter family for
modeling continuous random variables on the unit interval by Carrasco et
all (2010). We establish the hierarchical structure connecting GKw to
several nested sub-models including the Beta and Kumaraswamy
distributions, derive closed-form expressions for the log-likelihood
function, score vector, and observed information matrix, and prove
asymptotic properties of maximum likelihood estimators. All analytical
derivatives are derived from the compositional structure of the
distribution and written in a form suitable for stable numerical
implementation. The theoretical results provide the foundation for
efficient numerical routines in the R package `gkwdist`.

**Keywords:** Bounded distributions, Beta distribution, Kumaraswamy
distribution, Maximum likelihood estimation, Fisher information,
Numerical stability

------------------------------------------------------------------------

## 1. Introduction and Preliminaries

### 1.1 Motivation and Background

The analysis of continuous random variables constrained to the unit
interval $(0,1)$ arises naturally in numerous statistical applications,
including proportions, rates, percentages, and index measurements. The
classical Beta distribution (Johnson et al., 1995) has long served as
the canonical model for such data, offering analytical tractability and
well-understood properties. However, its cumulative distribution
function (CDF) involves the incomplete beta function, requiring
numerical evaluation of special functions for quantile computation and
simulation.

Kumaraswamy (1980) introduced an alternative two-parameter family with
closed-form CDF and quantile function, facilitating computational
efficiency while maintaining comparable flexibility to the Beta
distribution. Jones (2009) demonstrated that the Kumaraswamy
distribution exhibits similar shape characteristics to the Beta family
while offering superior computational advantages.

Building upon these foundations, Cordeiro and de Castro (2011) developed
the Generalized Kumaraswamy (GKw) distribution, a five-parameter
extension incorporating both Beta and Kumaraswamy structures through
nested transformations. This distribution encompasses a rich hierarchy
of submodels, providing substantial flexibility for modeling diverse
patterns in bounded data.

Despite its theoretical appeal, a fully explicit and internally
consistent analytical treatment of the GKw family—particularly for
likelihood-based inference—has remained incomplete in the literature.
This vignette fills this gap by providing a rigorous development,
including validated expressions for all first and second derivatives of
the log-likelihood function, written in a form convenient for
implementation in the `gkwdist` R package.

### 1.2 Mathematical Preliminaries

We establish notation and fundamental results required for subsequent
development.

**Notation 1.1.** Throughout, we denote

- $\Gamma( \cdot )$: the gamma function
- $B(a,b) = \frac{\Gamma(a)\Gamma(b)}{\Gamma(a + b)}$: the beta function
- $I_{z}(a,b) = \frac{B_{z}(a,b)}{B(a,b)}$: the regularized incomplete
  beta function
- $\psi(x) = \Gamma\prime(x)/\Gamma(x) = \left( \ln\Gamma(x) \right)\prime$:
  the digamma function
- $\psi_{1}(x) = \psi\prime(x) = \left( \ln\Gamma(x) \right)''$: the
  trigamma function
- $\mathbf{1}_{A}$: the indicator function of a set $A$

We recall basic derivatives of the beta function.

**Lemma 1.1 (Derivatives of the beta function).** For $a,b > 0$,

$$\begin{aligned}
{\frac{\partial}{\partial a}\ln B(a,b)} & {= \psi(a) - \psi(a + b),} \\
{\frac{\partial^{2}}{\partial a^{2}}\ln B(a,b)} & {= \psi_{1}(a) - \psi_{1}(a + b),} \\
{\frac{\partial^{2}}{\partial a\,\partial b}\ln B(a,b)} & {= - \psi_{1}(a + b).}
\end{aligned}$$

*Proof.* Since
$$\ln B(a,b) = \ln\Gamma(a) + \ln\Gamma(b) - \ln\Gamma(a + b),$$ the
identities follow immediately from the definitions of $\psi$ and
$\psi_{1}$ and the chain rule. $▫$

We will also repeatedly use the following cascade of transformations.

**Lemma 1.2 (Cascade transformations).** Define, for $x \in (0,1)$,

$$\begin{aligned}
{v(x;\alpha)} & {= 1 - x^{\alpha},} \\
{w(x;\alpha,\beta)} & {= 1 - v(x;\alpha)^{\beta} = 1 - \left( 1 - x^{\alpha} \right)^{\beta},} \\
{z(x;\alpha,\beta,\lambda)} & {= 1 - w(x;\alpha,\beta)^{\lambda} = 1 - \left\lbrack 1 - \left( 1 - x^{\alpha} \right)^{\beta} \right\rbrack^{\lambda}.}
\end{aligned}$$

Then, for $\alpha,\beta,\lambda > 0$,

$$\begin{aligned}
\frac{\partial v}{\partial x} & {= - \alpha x^{\alpha - 1},} \\
\frac{\partial w}{\partial x} & {= \alpha\beta x^{\alpha - 1}\left( 1 - x^{\alpha} \right)^{\beta - 1},} \\
\frac{\partial z}{\partial x} & {= \alpha\beta\lambda\, x^{\alpha - 1}\left( 1 - x^{\alpha} \right)^{\beta - 1}\lbrack 1 - \left( 1 - x^{\alpha} \right)^{\beta}\rbrack^{\lambda - 1}.}
\end{aligned}$$

*Proof.* Direct differentiation and repeated application of the chain
rule. $▫$

For brevity we will often write $v(x)$, $w(x)$ and $z(x)$ when the
dependence on $(\alpha,\beta,\lambda)$ is clear from the context.

------------------------------------------------------------------------

## 2. The Generalized Kumaraswamy Distribution and Its Subfamily

### 2.1 Definition and Fundamental Properties

We start from the five-parameter Generalized Kumaraswamy family.

**Definition 2.1 (Generalized Kumaraswamy distribution).**  
A random variable $X$ has a Generalized Kumaraswamy distribution with
parameter vector
$${\mathbf{θ}} = (\alpha,\beta,\gamma,\delta,\lambda)^{\top},$$ denoted
$X \sim {GKw}(\alpha,\beta,\gamma,\delta,\lambda)$, if its probability
density function (pdf) is
$$\boxed{f(x;{\mathbf{θ}}) = \frac{\lambda\alpha\beta}{B(\gamma,\delta + 1)}\; x^{\alpha - 1}v(x)^{\beta - 1}w(x)^{\gamma\lambda - 1}z(x)^{\delta}\,\mathbf{1}_{(0,1)}(x),}$$
where
$$v(x) = 1 - x^{\alpha},\qquad w(x) = 1 - \left( 1 - x^{\alpha} \right)^{\beta},\qquad z(x) = 1 - w(x)^{\lambda},$$
and the parameter space is \$\$ \Theta =
\Bigl\\(\alpha,\beta,\gamma,\delta,\lambda)^\top :
\alpha,\beta,\gamma,\lambda\>0,\\ \delta\ge 0\Bigr\\. \tag{2.2} \$\$

Note that $B(\gamma,\delta + 1)$ is well-defined for all $\gamma > 0$
and $\delta > - 1$; we restrict to $\delta \geq 0$ for convenience and
consistency with the literature.

We now verify that (2.1) defines a proper density.

**Theorem 2.1 (Validity of the pdf).**  
For any ${\mathbf{θ}} \in \Theta$, the function
$f( \cdot ;{\mathbf{θ}})$ in (2.1) is a valid probability density on
$(0,1)$.

*Proof.* Non-negativity is immediate from the definition. To prove
normalization, consider the change of variable
$$u = w(x)^{\lambda},\qquad 0 < u < 1.$$ From Lemma 1.2 and the chain
rule,
$$\frac{du}{dx} = \lambda w(x)^{\lambda - 1}\frac{\partial w(x)}{\partial x} = \lambda\alpha\beta\, x^{\alpha - 1}v(x)^{\beta - 1}w(x)^{\lambda - 1}.$$
Hence
$$dx = \frac{du}{\lambda\alpha\beta\, x^{\alpha - 1}v(x)^{\beta - 1}w(x)^{\lambda - 1}}.$$

Substituting into the integral of $f$, $$\begin{aligned}
{\int_{0}^{1}f(x;{\mathbf{θ}})\, dx} & {= \frac{\lambda\alpha\beta}{B(\gamma,\delta + 1)}\int_{0}^{1}x^{\alpha - 1}v(x)^{\beta - 1}w(x)^{\gamma\lambda - 1}z(x)^{\delta}\, dx} \\
 & {= \frac{\lambda\alpha\beta}{B(\gamma,\delta + 1)}\int_{0}^{1}x^{\alpha - 1}v(x)^{\beta - 1}w(x)^{\gamma\lambda - \lambda}w(x)^{\lambda - 1}z(x)^{\delta}\, dx} \\
 & {= \frac{\lambda\alpha\beta}{B(\gamma,\delta + 1)}\int_{0}^{1}w(x)^{\lambda{(\gamma - 1)}}z(x)^{\delta}\,\underset{du/{(\lambda\alpha\beta)}}{\underbrace{x^{\alpha - 1}v(x)^{\beta - 1}w(x)^{\lambda - 1}dx}}} \\
 & {= \frac{1}{B(\gamma,\delta + 1)}\int_{0}^{1}u^{\gamma - 1}(1 - u)^{\delta}\, du} \\
 & {= \frac{B(\gamma,\delta + 1)}{B(\gamma,\delta + 1)} = 1,}
\end{aligned}$$ because
$w(x)^{\lambda{(\gamma - 1)}} = \left( w(x)^{\lambda} \right)^{\gamma - 1} = u^{\gamma - 1}$
and $z(x) = 1 - w(x)^{\lambda} = 1 - u$. $▫$

The same change of variable yields the CDF.

**Theorem 2.2 (Cumulative distribution function).**  
If $X \sim {GKw}({\mathbf{θ}})$, then for $x \in (0,1)$,
$$\boxed{F(x;{\mathbf{θ}}) = I_{w{(x)}^{\lambda}}(\gamma,\delta + 1) = I_{{\lbrack 1 - {(1 - x^{\alpha})}^{\beta}\rbrack}^{\lambda}}(\gamma,\delta + 1).}$$

*Proof.* From the same substitution as in (2.3), $$\begin{aligned}
{F(x)} & {= \int_{0}^{x}f(t;{\mathbf{θ}})\, dt} \\
 & {= \frac{1}{B(\gamma,\delta + 1)}\int_{0}^{w{(x)}^{\lambda}}u^{\gamma - 1}(1 - u)^{\delta}\, du} \\
 & {= I_{w{(x)}^{\lambda}}(\gamma,\delta + 1).}
\end{aligned}$$ At the endpoints, $w\left( 0^{+} \right)^{\lambda} = 0$
and $w\left( 1^{-} \right)^{\lambda} = 1$, so $F(0) = 0$ and $F(1) = 1$.
$▫$

### 2.2 The Hierarchical Structure

The GKw family exhibits a rich nested structure. Several well-known
bounded distributions arise as particular choices (and mild
reparameterizations) of $\mathbf{θ}$.

For a data point $x \in (0,1)$ we will often write
$$v = 1 - x^{\alpha},\quad w = 1 - \left( 1 - x^{\alpha} \right)^{\beta},\quad z = 1 - w^{\lambda}.$$

#### 2.2.1 Beta–Kumaraswamy distribution

**Theorem 2.3 (Beta–Kumaraswamy distribution).**  
Setting $\lambda = 1$ in (2.1) yields the four-parameter
Beta–Kumaraswamy (BKw) distribution with pdf
$$\boxed{f_{BKw}(x;\alpha,\beta,\gamma,\delta) = \frac{\alpha\beta}{B(\gamma,\delta + 1)}\; x^{\alpha - 1}\left( 1 - x^{\alpha} \right)^{\beta{(\delta + 1)} - 1}\lbrack 1 - \left( 1 - x^{\alpha} \right)^{\beta}\rbrack^{\gamma - 1},}$$
and CDF
$$\boxed{F_{BKw}(x;\alpha,\beta,\gamma,\delta) = I_{1 - {(1 - x^{\alpha})}^{\beta}}(\gamma,\delta + 1).}$$

*Proof.* For $\lambda = 1$, we have
$z(x) = 1 - w(x) = \left( 1 - x^{\alpha} \right)^{\beta}$, so from
(2.1), $$\begin{aligned}
{f(x)} & {= \frac{\alpha\beta}{B(\gamma,\delta + 1)}\; x^{\alpha - 1}v^{\beta - 1}w^{\gamma - 1}z^{\delta}} \\
 & {= \frac{\alpha\beta}{B(\gamma,\delta + 1)}\; x^{\alpha - 1}\left( 1 - x^{\alpha} \right)^{\beta - 1}\lbrack 1 - \left( 1 - x^{\alpha} \right)^{\beta}\rbrack^{\gamma - 1}\lbrack\left( 1 - x^{\alpha} \right)^{\beta}\rbrack^{\delta}} \\
 & {= \frac{\alpha\beta}{B(\gamma,\delta + 1)}\; x^{\alpha - 1}\left( 1 - x^{\alpha} \right)^{\beta{(\delta + 1)} - 1}\lbrack 1 - \left( 1 - x^{\alpha} \right)^{\beta}\rbrack^{\gamma - 1},}
\end{aligned}$$ which is (2.5). The CDF follows from Theorem 2.2 with
$\lambda = 1$. $▫$

#### 2.2.2 Kumaraswamy–Kumaraswamy distribution

The KKw submodel is most naturally obtained via a mild
reparameterization of $\delta$.

**Theorem 2.4 (Kumaraswamy–Kumaraswamy distribution).**  
Fix $\alpha,\beta,\lambda > 0$ and $\widetilde{\delta} > 0$. Consider
the GKw submodel
$$X \sim {GKw}\left( \alpha,\beta,\gamma = 1,\delta = \widetilde{\delta} - 1,\lambda \right),\quad{\text{with}\mspace{6mu}}\widetilde{\delta} \geq 1.$$
Then $X$ has pdf
$$\boxed{f_{KKw}\left( x;\alpha,\beta,\widetilde{\delta},\lambda \right) = \widetilde{\delta}\,\lambda\alpha\beta\; x^{\alpha - 1}v^{\beta - 1}w^{\lambda - 1}z^{\widetilde{\delta} - 1},}$$
CDF
$$\boxed{F_{KKw}\left( x;\alpha,\beta,\widetilde{\delta},\lambda \right) = 1 - z(x)^{\widetilde{\delta}} = 1 - \lbrack 1 - w(x)^{\lambda}\rbrack^{\widetilde{\delta}},}$$
and quantile function \$\$ \boxed{ Q\_{\mathrm{KKw}}(p;
\alpha,\beta,\tilde\delta,\lambda) = \left\\ 1-\left\[
1-\Bigl(1-(1-p)^{1/\tilde\delta}\Bigr)^{1/\lambda} \right\]^{1/\beta}
\right\\^{1/\alpha},\quad 0\<p\<1. } \tag{2.9} \$\$

*Proof.* Take $\gamma = 1$ and $\delta = \widetilde{\delta} - 1$ in
(2.1):
$$f(x) = \frac{\lambda\alpha\beta}{B\left( 1,\widetilde{\delta} \right)}x^{\alpha - 1}v^{\beta - 1}w^{\lambda - 1}z^{\widetilde{\delta} - 1}.$$
Since $B\left( 1,\widetilde{\delta} \right) = 1/\widetilde{\delta}$, we
obtain (2.7). From (2.4),
$$F(x) = I_{w{(x)}^{\lambda}}\left( 1,\widetilde{\delta} \right) = 1 - (1 - w(x)^{\lambda})^{\widetilde{\delta}} = 1 - z(x)^{\widetilde{\delta}},$$
which is (2.8). Inverting $F(x) = p$ yields
$$z(x) = (1 - p)^{1/\widetilde{\delta}},\quad w(x)^{\lambda} = 1 - (1 - p)^{1/\widetilde{\delta}},\quad w(x) = \lbrack 1 - (1 - p)^{1/\widetilde{\delta}}\rbrack^{1/\lambda},$$
and
$$v(x)^{\beta} = 1 - w(x) = 1 - \lbrack 1 - (1 - p)^{1/\widetilde{\delta}}\rbrack^{1/\lambda},$$
leading to (2.9). $▫$

For notational simplicity, in the remainder we drop the tilde and write
$\delta$ for the KKw shape parameter; the mapping to the GKw parameters
is $\delta_{GKw} = \delta - 1$.

#### 2.2.3 Exponentiated Kumaraswamy distribution

**Theorem 2.5 (Exponentiated Kumaraswamy distribution).**  
Setting $\gamma = 1$ and $\delta = 0$ in (2.1) yields the
three-parameter exponentiated Kumaraswamy (EKw) distribution
$$\boxed{f_{EKw}(x;\alpha,\beta,\lambda) = \lambda\alpha\beta\; x^{\alpha - 1}\left( 1 - x^{\alpha} \right)^{\beta - 1}\lbrack 1 - \left( 1 - x^{\alpha} \right)^{\beta}\rbrack^{\lambda - 1},}$$
with CDF
$$\boxed{F_{EKw}(x;\alpha,\beta,\lambda) = \lbrack 1 - \left( 1 - x^{\alpha} \right)^{\beta}\rbrack^{\lambda},}$$
and quantile function
$$\boxed{Q_{EKw}(p;\alpha,\beta,\lambda) = \lbrack 1 - (1 - p^{1/\lambda})^{1/\beta}\rbrack^{1/\alpha},\quad 0 < p < 1.}$$

*Proof.* With $\gamma = 1$ and $\delta = 0$, we have $B(1,1) = 1$,
$z(x)^{0} = 1$, and $w(x)^{\gamma\lambda - 1} = w(x)^{\lambda - 1}$.
Thus (2.1) reduces to (2.10). From (2.4),
$$F(x) = I_{w{(x)}^{\lambda}}(1,1) = w(x)^{\lambda} = \lbrack 1 - \left( 1 - x^{\alpha} \right)^{\beta}\rbrack^{\lambda},$$
yielding (2.11). Inverting $F(x) = p$ gives $w(x) = p^{1/\lambda}$ and
then $x$ as in (2.12). $▫$

Note that the standard Kumaraswamy distribution appears as the special
case $\lambda = 1$ of EKw.

#### 2.2.4 McDonald distribution

**Theorem 2.6 (McDonald distribution).**  
Setting $\alpha = \beta = 1$ in (2.1) yields the three-parameter
McDonald distribution
$$\boxed{f_{MC}(x;\gamma,\delta,\lambda) = \frac{\lambda}{B(\gamma,\delta + 1)}\; x^{\gamma\lambda - 1}\left( 1 - x^{\lambda} \right)^{\delta},}$$
with CDF
$$\boxed{F_{MC}(x;\gamma,\delta,\lambda) = I_{x^{\lambda}}(\gamma,\delta + 1).}$$

*Proof.* For $\alpha = \beta = 1$ we have $v(x) = 1 - x$, $w(x) = x$,
and $z(x) = 1 - x^{\lambda}$. Substituting into (2.1) yields (2.13); the
CDF follows from (2.4). $▫$

#### 2.2.5 Kumaraswamy distribution

**Theorem 2.7 (Kumaraswamy distribution).**  
The standard two-parameter Kumaraswamy distribution is obtained from GKw
by taking
$$X \sim {GKw}(\alpha,\beta,\gamma = 1,\delta = 0,\lambda = 1),$$
equivalently as the submodel EKw($\alpha,\beta,\lambda$) with
$\lambda = 1$. Its pdf is
$$\boxed{f_{Kw}(x;\alpha,\beta) = \alpha\beta\; x^{\alpha - 1}\left( 1 - x^{\alpha} \right)^{\beta - 1},}$$
with CDF
$$\boxed{F_{Kw}(x;\alpha,\beta) = 1 - \left( 1 - x^{\alpha} \right)^{\beta},}$$
quantile function
$$\boxed{Q_{Kw}(p;\alpha,\beta) = \lbrack 1 - (1 - p)^{1/\beta}\rbrack^{1/\alpha},}$$
and $r$-th moment
$$\boxed{{\mathbb{E}}\left( X^{r} \right) = \beta\, B\!\left( 1 + \frac{r}{\alpha},\beta \right) = \frac{\beta\,\Gamma(1 + r/\alpha)\Gamma(\beta)}{\Gamma(1 + r/\alpha + \beta)}.}$$

*Proof.* With $\gamma = 1$, $\delta = 0$ and $\lambda = 1$, (2.1)
reduces to (2.15) because $B(1,1) = 1$,
$w^{\gamma\lambda - 1} = w^{0} = 1$ and $z^{0} = 1$. Equations (2.16)
and (2.17) follow from (2.11) with $\lambda = 1$. For the moment,
$$\begin{aligned}
{{\mathbb{E}}\left( X^{r} \right)} & {= \alpha\beta\int_{0}^{1}x^{r + \alpha - 1}\left( 1 - x^{\alpha} \right)^{\beta - 1}\, dx} \\
 & {{\text{(let}\mspace{6mu}}u = x^{\alpha},\ du = \alpha x^{\alpha - 1}dx)} \\
 & {= \beta\int_{0}^{1}u^{r/\alpha}(1 - u)^{\beta - 1}\, du = \beta B(1 + r/\alpha,\beta),}
\end{aligned}$$ which yields (2.18). $▫$

#### 2.2.6 Beta distribution

**Theorem 2.8 (Beta distribution).**  
Setting $\alpha = \beta = \lambda = 1$ in (2.1) yields
$$\boxed{f_{Beta}(x;\gamma,\delta) = \frac{x^{\gamma - 1}(1 - x)^{\delta}}{B(\gamma,\delta + 1)},}$$
with CDF
$$\boxed{F_{Beta}(x;\gamma,\delta) = I_{x}(\gamma,\delta + 1),}$$ and
$r$-th moment
$$\boxed{{\mathbb{E}}\left( X^{r} \right) = \frac{B(\gamma + r,\delta + 1)}{B(\gamma,\delta + 1)} = \frac{\Gamma(\gamma + r)\Gamma(\gamma + \delta + 1)}{\Gamma(\gamma)\Gamma(\gamma + \delta + r + 1)}.}$$

*Proof.* For $\alpha = \beta = \lambda = 1$, we have $v(x) = 1 - x$,
$w(x) = x$, and $z(x) = 1 - x$. Substituting into (2.1) gives (2.19);
the CDF and moment follow from standard Beta distribution theory with
shape parameters $(\gamma,\delta + 1)$. $▫$

------------------------------------------------------------------------

## 3. Likelihood-Based Inference

### 3.1 The Log-Likelihood Function

Let $\mathbf{X} = \left( X_{1},\ldots,X_{n} \right)^{\top}$ be an i.i.d.
sample from ${GKw}({\mathbf{θ}})$, with observed values
$\mathbf{x} = \left( x_{1},\ldots,x_{n} \right)^{\top}$. For each $i$,
define
$$v_{i} = v\left( x_{i} \right),\quad w_{i} = w\left( x_{i} \right),\quad z_{i} = z\left( x_{i} \right).$$

**Definition 3.1 (Log-likelihood function).**  
The log-likelihood is
$$\ell({\mathbf{θ}};\mathbf{x}) = \sum\limits_{i = 1}^{n}\ln f\left( x_{i};{\mathbf{θ}} \right).$$

**Theorem 3.1 (Decomposition of the log-likelihood).**  
The log-likelihood can be written as
$$\boxed{\ell({\mathbf{θ}}) = n\ln(\lambda\alpha\beta) - n\ln B(\gamma,\delta + 1) + \sum\limits_{i = 1}^{n}S_{i}({\mathbf{θ}}),}$$
where
$$S_{i}({\mathbf{θ}}) = (\alpha - 1)\ln x_{i} + (\beta - 1)\ln v_{i} + (\gamma\lambda - 1)\ln w_{i} + \delta\ln z_{i}.$$

Equivalently,
$$\ell({\mathbf{θ}}) = L_{1} + L_{2} + L_{3} + L_{4} + L_{5} + L_{6} + L_{7} + L_{8},$$
where $$\begin{aligned}
L_{1} & {= n\ln\lambda,} \\
L_{2} & {= n\ln\alpha,} \\
L_{3} & {= n\ln\beta,} \\
L_{4} & {= - n\ln B(\gamma,\delta + 1),} \\
L_{5} & {= (\alpha - 1)\sum\limits_{i = 1}^{n}\ln x_{i},} \\
L_{6} & {= (\beta - 1)\sum\limits_{i = 1}^{n}\ln v_{i},} \\
L_{7} & {= (\gamma\lambda - 1)\sum\limits_{i = 1}^{n}\ln w_{i},} \\
L_{8} & {= \delta\sum\limits_{i = 1}^{n}\ln z_{i}.}
\end{aligned}$$

*Proof.* Take logarithms of (2.1) and sum over $i$. $▫$

### 3.2 Maximum Likelihood Estimation

**Definition 3.2 (Maximum likelihood estimator).**  
The maximum likelihood estimator (MLE) is defined by
$${\widehat{\mathbf{θ}}}_{n} = \arg\max\limits_{{\mathbf{θ}} \in \Theta}\ell({\mathbf{θ}};\mathbf{x}).$$

We will refer to the true parameter value as
${\mathbf{θ}}_{0} \in \Theta$.

**Theorem 3.2 (Consistency and asymptotic normality).**  
Assume the usual regularity conditions for likelihood inference hold
(e.g. Lehmann and Casella, 1998; van der Vaart, 1998), in particular
that ${\mathbf{θ}}_{0} \in {int}(\Theta)$ and \$\$
\mathcal{I}(\boldsymbol{\theta}\_0) =
\mathbb{E}\_{\boldsymbol{\theta}\_0} \Bigl\[-\nabla^2 \ln
f(X;\boldsymbol{\theta})\Bigr\] \$\$ is positive definite. Then
$$\left. {\widehat{\mathbf{θ}}}_{n}\overset{p}{\rightarrow}{\mathbf{θ}}_{0},\qquad n\rightarrow\infty, \right.$$
and
$$\left. \sqrt{n}\left( {\widehat{\mathbf{θ}}}_{n} - {\mathbf{θ}}_{0} \right)\overset{d}{\rightarrow}\mathcal{N}_{5}(\mathbf{0},\mathcal{I}\left( {\mathbf{θ}}_{0} \right)^{- 1}),\qquad n\rightarrow\infty. \right.$$

*Proof.* The regularity conditions are satisfied for all $x \in (0,1)$
and ${\mathbf{θ}} \in {int}(\Theta)$: the support does not depend on
$\mathbf{θ}$, $f(x;{\mathbf{θ}})$ is continuously differentiable in a
neighborhood of ${\mathbf{θ}}_{0}$, the score has mean zero and finite
second moment, and the Fisher information is non-singular. The result
then follows from standard MLE asymptotic theory (see, e.g., van der
Vaart, 1998). $▫$

### 3.3 Likelihood Ratio Tests

For nested models, likelihood ratio tests follow Wilks’ theorem.

**Theorem 3.3 (Wilks’ theorem).**  
Consider testing nested hypotheses
$$H_{0}:\ {\mathbf{θ}} \in \Theta_{0}\quad\text{vs.}\quad H_{1}:\ {\mathbf{θ}} \in \Theta,$$
where $\Theta_{0} \subset \Theta$ is defined by $r$ independent
constraints. Let \$\$ \Lambda_n = 2\Bigl\\
\ell(\hat{\boldsymbol{\theta}}\_n)
-\ell(\hat{\boldsymbol{\theta}}\_{0,n}) \Bigr\\, \tag{3.16} \$\$ where
${\widehat{\mathbf{θ}}}_{n}$ and ${\widehat{\mathbf{θ}}}_{0,n}$ are the
unconstrained and constrained MLEs, respectively. Under $H_{0}$,
$$\left. \Lambda_{n}\overset{d}{\rightarrow}\chi_{r}^{2},\qquad n\rightarrow\infty. \right.$$

*Proof.* Standard; see Casella and Berger (2002), Chapter 10. $▫$

Within the GKw hierarchy we obtain the following tests.

**Corollary 3.1 (LR tests within the GKw hierarchy).**  
Under the usual regularity conditions and away from parameter
boundaries, the following likelihood ratio tests are asymptotically
$\chi^{2}$:

$$\begin{aligned}
{\text{GKw vs. BKw:}\quad} & {H_{0}:\ \lambda = 1} & & \left. \Rightarrow\ \Lambda_{n}\overset{d}{\rightarrow}\chi_{1}^{2}, \right. \\
{\text{GKw vs. KKw:}\quad} & {H_{0}:\ \gamma = 1} & & \left. \Rightarrow\ \Lambda_{n}\overset{d}{\rightarrow}\chi_{1}^{2}, \right. \\
{\text{BKw vs. Beta:}\quad} & {H_{0}:\ \alpha = \beta = 1} & & \left. \Rightarrow\ \Lambda_{n}\overset{d}{\rightarrow}\chi_{2}^{2}, \right. \\
{\text{KKw vs. Kw:}\quad} & {H_{0}:\ \delta = \lambda = 1} & & \left. \Rightarrow\ \Lambda_{n}\overset{d}{\rightarrow}\chi_{2}^{2}. \right.
\end{aligned}$$

The precise mapping between $\delta$ in KKw and $\delta$ in GKw is as
described in Theorem 2.4; the dimension reduction in each hypothesis is
nevertheless $r$ as indicated.

------------------------------------------------------------------------

## 4. Analytical Derivatives and Information Matrix

We now derive explicit expressions for the score vector and the observed
information matrix in terms of the cascade transformations $v,w,z$ and
their derivatives.

### 4.1 The Score Vector

**Definition 4.1 (Score function).**  
The score vector is
$$U({\mathbf{θ}}) = \nabla_{\mathbf{θ}}\ell({\mathbf{θ}}) = \left( \frac{\partial\ell}{\partial\alpha},\frac{\partial\ell}{\partial\beta},\frac{\partial\ell}{\partial\gamma},\frac{\partial\ell}{\partial\delta},\frac{\partial\ell}{\partial\lambda} \right)^{\top}.$$

For the derivatives of $v,w,z$ with respect to the parameters we will
use $$\begin{aligned}
\frac{\partial v_{i}}{\partial\alpha} & {= - x_{i}^{\alpha}\ln x_{i},} \\
\frac{\partial w_{i}}{\partial\alpha} & {= \beta v_{i}^{\beta - 1}x_{i}^{\alpha}\ln x_{i},} \\
\frac{\partial w_{i}}{\partial\beta} & {= - v_{i}^{\beta}\ln v_{i},} \\
\frac{\partial z_{i}}{\partial\alpha} & {= - \lambda w_{i}^{\lambda - 1}\frac{\partial w_{i}}{\partial\alpha},} \\
\frac{\partial z_{i}}{\partial\beta} & {= - \lambda w_{i}^{\lambda - 1}\frac{\partial w_{i}}{\partial\beta},} \\
\frac{\partial z_{i}}{\partial\lambda} & {= - w_{i}^{\lambda}\ln w_{i}.}
\end{aligned}$$

**Theorem 4.1 (Score components).**  
The components of $U({\mathbf{θ}})$ are

$$\begin{aligned}
\frac{\partial\ell}{\partial\alpha} & {= \frac{n}{\alpha} + \sum\limits_{i = 1}^{n}\ln x_{i} - \sum\limits_{i = 1}^{n}x_{i}^{\alpha}\ln x_{i}\left\lbrack \frac{\beta - 1}{v_{i}} - \frac{(\gamma\lambda - 1)\beta v_{i}^{\beta - 1}}{w_{i}} + \frac{\delta\lambda\beta v_{i}^{\beta - 1}w_{i}^{\lambda - 1}}{z_{i}} \right\rbrack,} \\
\frac{\partial\ell}{\partial\beta} & {= \frac{n}{\beta} + \sum\limits_{i = 1}^{n}\ln v_{i} - \sum\limits_{i = 1}^{n}v_{i}^{\beta}\ln v_{i}\left\lbrack \frac{\gamma\lambda - 1}{w_{i}} - \frac{\delta\lambda w_{i}^{\lambda - 1}}{z_{i}} \right\rbrack,} \\
\frac{\partial\ell}{\partial\gamma} & {= - n\lbrack\psi(\gamma) - \psi(\gamma + \delta + 1)\rbrack + \lambda\sum\limits_{i = 1}^{n}\ln w_{i},} \\
\frac{\partial\ell}{\partial\delta} & {= - n\lbrack\psi(\delta + 1) - \psi(\gamma + \delta + 1)\rbrack + \sum\limits_{i = 1}^{n}\ln z_{i},} \\
\frac{\partial\ell}{\partial\lambda} & {= \frac{n}{\lambda} + \gamma\sum\limits_{i = 1}^{n}\ln w_{i} - \delta\sum\limits_{i = 1}^{n}\frac{w_{i}^{\lambda}\ln w_{i}}{z_{i}}.}
\end{aligned}$$

*Proof.* We differentiate the decomposition (3.4)–(3.12) term by term.

**(i) Derivative with respect to $\alpha$.**

From (3.6) and (3.9),
$$\frac{\partial L_{2}}{\partial\alpha} = \frac{n}{\alpha},\quad\frac{\partial L_{5}}{\partial\alpha} = \sum\limits_{i = 1}^{n}\ln x_{i}.$$
Using $\partial v_{i}/\partial\alpha = - x_{i}^{\alpha}\ln x_{i}$,
$$\frac{\partial L_{6}}{\partial\alpha} = (\beta - 1)\sum\limits_{i = 1}^{n}\frac{1}{v_{i}}\frac{\partial v_{i}}{\partial\alpha} = - (\beta - 1)\sum\limits_{i = 1}^{n}\frac{x_{i}^{\alpha}\ln x_{i}}{v_{i}}.$$
Next,
$\partial w_{i}/\partial\alpha = \beta v_{i}^{\beta - 1}x_{i}^{\alpha}\ln x_{i}$,
so
$$\frac{\partial L_{7}}{\partial\alpha} = (\gamma\lambda - 1)\sum\limits_{i = 1}^{n}\frac{1}{w_{i}}\frac{\partial w_{i}}{\partial\alpha} = (\gamma\lambda - 1)\beta\sum\limits_{i = 1}^{n}\frac{v_{i}^{\beta - 1}x_{i}^{\alpha}\ln x_{i}}{w_{i}}.$$
Similarly,
$$\frac{\partial z_{i}}{\partial\alpha} = - \lambda w_{i}^{\lambda - 1}\frac{\partial w_{i}}{\partial\alpha} = - \lambda\beta v_{i}^{\beta - 1}w_{i}^{\lambda - 1}x_{i}^{\alpha}\ln x_{i},$$
so
$$\frac{\partial L_{8}}{\partial\alpha} = \delta\sum\limits_{i = 1}^{n}\frac{1}{z_{i}}\frac{\partial z_{i}}{\partial\alpha} = - \delta\lambda\beta\sum\limits_{i = 1}^{n}\frac{v_{i}^{\beta - 1}w_{i}^{\lambda - 1}x_{i}^{\alpha}\ln x_{i}}{z_{i}}.$$
Collecting terms gives (4.2).

**(ii) Derivative with respect to $\beta$.**

From (3.7) and (3.10),
$$\frac{\partial L_{3}}{\partial\beta} = \frac{n}{\beta},\quad\frac{\partial L_{6}}{\partial\beta} = \sum\limits_{i = 1}^{n}\ln v_{i},$$
since $v_{i}$ does not depend on $\beta$. Using
$\partial w_{i}/\partial\beta = - v_{i}^{\beta}\ln v_{i}$,
$$\frac{\partial L_{7}}{\partial\beta} = (\gamma\lambda - 1)\sum\limits_{i = 1}^{n}\frac{1}{w_{i}}\frac{\partial w_{i}}{\partial\beta} = - (\gamma\lambda - 1)\sum\limits_{i = 1}^{n}\frac{v_{i}^{\beta}\ln v_{i}}{w_{i}}.$$
Furthermore,
$$\frac{\partial z_{i}}{\partial\beta} = - \lambda w_{i}^{\lambda - 1}\frac{\partial w_{i}}{\partial\beta} = \lambda w_{i}^{\lambda - 1}v_{i}^{\beta}\ln v_{i},$$
so
$$\frac{\partial L_{8}}{\partial\beta} = \delta\sum\limits_{i = 1}^{n}\frac{1}{z_{i}}\frac{\partial z_{i}}{\partial\beta} = \delta\lambda\sum\limits_{i = 1}^{n}\frac{w_{i}^{\lambda - 1}v_{i}^{\beta}\ln v_{i}}{z_{i}}.$$
Combining terms yields (4.3).

**(iii) Derivative with respect to $\gamma$.**

Only $L_{4}$ and $L_{7}$ depend on $\gamma$. From Lemma 1.1,
$$\frac{\partial L_{4}}{\partial\gamma} = - n\lbrack\psi(\gamma) - \psi(\gamma + \delta + 1)\rbrack,\qquad\frac{\partial L_{7}}{\partial\gamma} = \lambda\sum\limits_{i = 1}^{n}\ln w_{i},$$
giving (4.4).

**(iv) Derivative with respect to $\delta$.**

Similarly,
$$\frac{\partial L_{4}}{\partial\delta} = - n\lbrack\psi(\delta + 1) - \psi(\gamma + \delta + 1)\rbrack,\qquad\frac{\partial L_{8}}{\partial\delta} = \sum\limits_{i = 1}^{n}\ln z_{i},$$
giving (4.5).

**(v) Derivative with respect to $\lambda$.**

We have
$$\frac{\partial L_{1}}{\partial\lambda} = \frac{n}{\lambda},\qquad\frac{\partial L_{7}}{\partial\lambda} = \gamma\sum\limits_{i = 1}^{n}\ln w_{i},$$
and
$$\left. \frac{\partial z_{i}}{\partial\lambda} = - w_{i}^{\lambda}\ln w_{i}\quad\Rightarrow\quad\frac{\partial L_{8}}{\partial\lambda} = \delta\sum\limits_{i = 1}^{n}\frac{1}{z_{i}}\frac{\partial z_{i}}{\partial\lambda} = - \delta\sum\limits_{i = 1}^{n}\frac{w_{i}^{\lambda}\ln w_{i}}{z_{i}}. \right.$$
Together these yield (4.6). $▫$

### 4.2 The Hessian and Observed Information Matrix

We now consider second-order derivatives. Let
$$H({\mathbf{θ}}) = \nabla^{2}\ell({\mathbf{θ}}) = \lbrack\frac{\partial^{2}\ell}{\partial\theta_{j}\partial\theta_{k}}\rbrack_{j,k = 1}^{5}$$
denote the Hessian matrix of the log-likelihood, where
${\mathbf{θ}} = (\alpha,\beta,\gamma,\delta,\lambda)^{\top}$.

**Definition 4.2 (Observed information).**  
The observed information matrix is defined as
$$\mathcal{J}({\mathbf{θ}}) = - H({\mathbf{θ}}) = - \nabla^{2}\ell({\mathbf{θ}}).$$

To keep the formulas compact, for each observation $i$ and each
transformation $u_{i} \in \{ v_{i},w_{i},z_{i}\}$ we define, for
parameters $\theta_{j},\theta_{k}$,
$$D_{jk}^{u}(i) = \frac{\frac{\partial^{2}u_{i}}{\partial\theta_{j}\partial\theta_{k}}u_{i} - \frac{\partial u_{i}}{\partial\theta_{j}}\frac{\partial u_{i}}{\partial\theta_{k}}}{u_{i}^{2}} = \frac{\partial^{2}}{\partial\theta_{j}\partial\theta_{k}}\ln u_{i}.$$

In particular,
$$\frac{\partial^{2}}{\partial\theta_{j}^{2}}\ln u_{i} = D_{jj}^{u}(i).$$

#### 4.2.1 Diagonal elements

**Theorem 4.2 (Diagonal elements of the Hessian).**  
The second derivatives of $\ell$ with respect to each parameter are
$$\begin{aligned}
\frac{\partial^{2}\ell}{\partial\alpha^{2}} & {= - \frac{n}{\alpha^{2}} + (\beta - 1)\sum\limits_{i = 1}^{n}D_{\alpha\alpha}^{v}(i) + (\gamma\lambda - 1)\sum\limits_{i = 1}^{n}D_{\alpha\alpha}^{w}(i) + \delta\sum\limits_{i = 1}^{n}D_{\alpha\alpha}^{z}(i),} \\
\frac{\partial^{2}\ell}{\partial\beta^{2}} & {= - \frac{n}{\beta^{2}} + (\gamma\lambda - 1)\sum\limits_{i = 1}^{n}D_{\beta\beta}^{w}(i) + \delta\sum\limits_{i = 1}^{n}D_{\beta\beta}^{z}(i),} \\
\frac{\partial^{2}\ell}{\partial\gamma^{2}} & {= - n\lbrack\psi_{1}(\gamma) - \psi_{1}(\gamma + \delta + 1)\rbrack,} \\
\frac{\partial^{2}\ell}{\partial\delta^{2}} & {= - n\lbrack\psi_{1}(\delta + 1) - \psi_{1}(\gamma + \delta + 1)\rbrack,} \\
\frac{\partial^{2}\ell}{\partial\lambda^{2}} & {= - \frac{n}{\lambda^{2}} + \delta\sum\limits_{i = 1}^{n}D_{\lambda\lambda}^{z}(i).}
\end{aligned}$$

Equivalently, one may write $$\begin{aligned}
T_{\alpha\alpha}^{(7)} & {= (\gamma\lambda - 1)\sum\limits_{i = 1}^{n}D_{\alpha\alpha}^{w}(i),} \\
T_{\alpha\alpha}^{(8)} & {= \delta\sum\limits_{i = 1}^{n}D_{\alpha\alpha}^{z}(i),} \\
T_{\beta\beta}^{(7)} & {= (\gamma\lambda - 1)\sum\limits_{i = 1}^{n}D_{\beta\beta}^{w}(i),} \\
T_{\beta\beta}^{(8)} & {= \delta\sum\limits_{i = 1}^{n}D_{\beta\beta}^{z}(i),}
\end{aligned}$$ so that (4.8)–(4.9) can be expressed in the same
notation as in the original decomposition.

*Proof.* Differentiate the score components (4.2)–(4.6) with respect to
the same parameter. The term $n\ln\alpha$ contributes $- n/\alpha^{2}$
to $\partial^{2}\ell/\partial\alpha^{2}$, and similarly for $\beta$ and
$\lambda$. The contributions from $L_{6},L_{7},L_{8}$ are precisely the
second derivatives of $(\beta - 1)\ln v_{i}$,
$(\gamma\lambda - 1)\ln w_{i}$ and $\delta\ln z_{i}$, which yield the
terms in $D_{\alpha\alpha}^{u}(i)$ or $D_{\beta\beta}^{u}(i)$.

For $\gamma$ and $\delta$, only $L_{4}$ depends on these parameters
through $B(\gamma,\delta + 1)$; the formulas (4.10)–(4.11) follow from
Lemma 1.1. Finally, $L_{7}$ does not depend on $\lambda$ beyond the
linear factor $\gamma\lambda - 1$, so
$\partial^{2}L_{7}/\partial\lambda^{2} = 0$, and the only contribution
to (4.12) besides $- n/\lambda^{2}$ comes from
$L_{8} = \delta\sum\ln z_{i}$, whose second derivative w.r.t. $\lambda$
is $\delta\sum D_{\lambda\lambda}^{z}(i)$. $▫$

#### 4.2.2 Off-diagonal elements

**Theorem 4.3 (Off-diagonal elements of the Hessian).**  
For $j \neq k$, the mixed second derivatives of $\ell$ are:

$$\begin{aligned}
\frac{\partial^{2}\ell}{\partial\alpha\,\partial\beta} & {= \sum\limits_{i = 1}^{n}\left\lbrack \frac{1}{v_{i}}\frac{\partial v_{i}}{\partial\alpha} + (\gamma\lambda - 1)D_{\alpha\beta}^{w}(i) + \delta D_{\alpha\beta}^{z}(i) \right\rbrack,} \\
\frac{\partial^{2}\ell}{\partial\gamma\,\partial\delta} & {= n\,\psi_{1}(\gamma + \delta + 1),} \\
\frac{\partial^{2}\ell}{\partial\gamma\,\partial\alpha} & {= \lambda\sum\limits_{i = 1}^{n}\frac{1}{w_{i}}\frac{\partial w_{i}}{\partial\alpha},} \\
\frac{\partial^{2}\ell}{\partial\gamma\,\partial\beta} & {= \lambda\sum\limits_{i = 1}^{n}\frac{1}{w_{i}}\frac{\partial w_{i}}{\partial\beta},} \\
\frac{\partial^{2}\ell}{\partial\delta\,\partial\alpha} & {= \sum\limits_{i = 1}^{n}\frac{1}{z_{i}}\frac{\partial z_{i}}{\partial\alpha},} \\
\frac{\partial^{2}\ell}{\partial\delta\,\partial\beta} & {= \sum\limits_{i = 1}^{n}\frac{1}{z_{i}}\frac{\partial z_{i}}{\partial\beta},} \\
\frac{\partial^{2}\ell}{\partial\lambda\,\partial\alpha} & {= \gamma\sum\limits_{i = 1}^{n}\frac{1}{w_{i}}\frac{\partial w_{i}}{\partial\alpha} + \delta\sum\limits_{i = 1}^{n}\frac{\partial}{\partial\alpha}\left( \frac{1}{z_{i}}\frac{\partial z_{i}}{\partial\lambda} \right),} \\
\frac{\partial^{2}\ell}{\partial\lambda\,\partial\beta} & {= \gamma\sum\limits_{i = 1}^{n}\frac{1}{w_{i}}\frac{\partial w_{i}}{\partial\beta} + \delta\sum\limits_{i = 1}^{n}\frac{\partial}{\partial\beta}\left( \frac{1}{z_{i}}\frac{\partial z_{i}}{\partial\lambda} \right),} \\
\frac{\partial^{2}\ell}{\partial\lambda\,\partial\gamma} & {= \sum\limits_{i = 1}^{n}\ln w_{i},} \\
\frac{\partial^{2}\ell}{\partial\lambda\,\partial\delta} & {= \sum\limits_{i = 1}^{n}\frac{1}{z_{i}}\frac{\partial z_{i}}{\partial\lambda}.}
\end{aligned}$$

*Proof.*

- Equation (4.18) follows by differentiating (4.4) with respect to
  $\delta$; only the term involving $\psi(\gamma + \delta + 1)$
  contributes a non-zero derivative, giving
  $n\psi_{1}(\gamma + \delta + 1)$.

- Equations (4.19)–(4.22) follow from differentiating (4.4) and (4.5)
  with respect to $\alpha$ or $\beta$. For example,
  $$\frac{\partial^{2}\ell}{\partial\gamma\,\partial\alpha} = \lambda\sum\limits_{i = 1}^{n}\frac{\partial}{\partial\alpha}\ln w_{i} = \lambda\sum\limits_{i = 1}^{n}\frac{1}{w_{i}}\frac{\partial w_{i}}{\partial\alpha}.$$

- Equation (4.17) is obtained by differentiating (4.2) with respect to
  $\beta$; only the factor $(\beta - 1)\ln v_{i}$ contributes the term
  $\left( \partial v_{i}/\partial\alpha \right)/v_{i}$, while the
  dependence of $w_{i},z_{i}$ on $\beta$ is captured by
  $D_{\alpha\beta}^{w}(i)$ and $D_{\alpha\beta}^{z}(i)$.

- For $\lambda$, note from (4.6) that
  $$\frac{\partial\ell}{\partial\lambda} = \frac{n}{\lambda} + \gamma\sum\limits_{i = 1}^{n}\ln w_{i} + \delta\sum\limits_{i = 1}^{n}\frac{1}{z_{i}}\frac{\partial z_{i}}{\partial\lambda}.$$
  Differentiating with respect to $\alpha$ or $\beta$ yields
  (4.23)–(4.24); differentiating with respect to $\gamma$ and $\delta$
  gives (4.25)–(4.26).

Mixed derivatives commute by Schwarz’s theorem, so the Hessian is
symmetric. $▫$

In practice, the resulting expressions for
$\mathcal{J}\left( \widehat{\mathbf{θ}} \right)$ are evaluated
numerically by plugging in the analytic first and second derivatives of
$v_{i},w_{i},z_{i}$, which follow recursively from the definitions of
these transformations.

### 4.3 Asymptotic Variance–Covariance Matrix

Under the conditions of Theorem 3.2, the asymptotic variance–covariance
matrix of the MLE is governed by the Fisher information.

Let $\mathcal{I}\left( {\mathbf{θ}}_{0} \right)$ denote the
**per-observation** Fisher information,
$$\mathcal{I}\left( {\mathbf{θ}}_{0} \right) = {\mathbb{E}}_{{\mathbf{θ}}_{0}}\left\lbrack - \nabla^{2}\ln f(X;{\mathbf{θ}}) \right\rbrack.$$

For the full sample of size $n$, the expected information is
$n\mathcal{I}\left( {\mathbf{θ}}_{0} \right)$, while the observed
information is $\mathcal{J}\left( {\widehat{\mathbf{θ}}}_{n} \right)$.

**Theorem 4.4 (Variance–covariance matrix of the MLE).**  
Under the regularity assumptions of Theorem 3.2,
$$\operatorname{Var}\left( {\widehat{\mathbf{θ}}}_{n} \right) \approx \frac{1}{n}\mathcal{I}\left( {\mathbf{θ}}_{0} \right)^{- 1} \approx \mathcal{J}\left( {\widehat{\mathbf{θ}}}_{n} \right)^{- 1},$$
and the asymptotic standard error of ${\widehat{\theta}}_{j}$ is
approximated by
$$\operatorname{SE}\left( {\widehat{\theta}}_{j} \right) = \sqrt{\lbrack\mathcal{J}\left( {\widehat{\mathbf{θ}}}_{n} \right)^{- 1}\rbrack_{jj}}.$$

*Proof.* The convergence
$\mathcal{J}\left( {\widehat{\mathbf{θ}}}_{n} \right)/n\overset{p}{\rightarrow}\mathcal{I}\left( {\mathbf{θ}}_{0} \right)$
follows from a law of large numbers for the Hessian (Cox and Hinkley,
1974). Combining this with the asymptotic normality (3.15) yields
(4.27)–(4.28). $▫$

------------------------------------------------------------------------

## 5. Computational Aspects and Discussion

### 5.1 Numerical Stability

Direct evaluation of expressions such as $v_{i} = 1 - x_{i}^{\alpha}$
can suffer from catastrophic cancellation when
$x_{i}^{\alpha} \approx 1$ and from underflow when $x_{i}^{\alpha}$ is
very small. To mitigate these issues, the implementation in `gkwdist`
works primarily in log-scale.

For example, we compute $\ln v_{i}$ via a numerically stable `log1mexp`
transformation.

**Algorithm 5.1 (Stable computation of $\log\left( 1 - e^{a} \right)$
for $a < 0$).**

For $a < 0$, define $$\text{log1mexp}(a) = \begin{cases}
{\ln( - \operatorname{expm1}(a)),} & {a < - \ln 2,} \\
{\ln(1 - e^{a}),} & {\text{otherwise}.}
\end{cases}$$ Then
$$\ln v_{i} = \text{log1mexp}\left( \alpha\ln x_{i} \right).$$

This strategy ensures high relative accuracy across the full range
$a \in ( - \infty,0)$ (see Mächler, 2012), and analogous transformations
are used wherever expressions of the form
$\log\left( 1 - \text{something} \right)$ occur.

### 5.2 Optimization

The analytic score in Theorem 4.1 allows efficient use of quasi-Newton
methods such as BFGS.

**Algorithm 5.2 (Maximum likelihood estimation via BFGS).**

**Input:** data $\mathbf{x} \in (0,1)^{n}$.

**Output:** MLE ${\widehat{\mathbf{θ}}}_{n}$ and observed information
$\mathcal{J}\left( {\widehat{\mathbf{θ}}}_{n} \right)$.

1.  **Initialization.** Obtain starting values ${\mathbf{θ}}^{(0)}$
    using method-of-moments, simple submodels (e.g. Beta or Kw), or a
    coarse grid search.
2.  **Quasi-Newton iteration.** For $k = 0,1,2,\ldots$:
    - Evaluate $\ell\left( {\mathbf{θ}}^{(k)} \right)$ and
      $U\left( {\mathbf{θ}}^{(k)} \right)$ using (3.2)–(3.3) and Theorem
      4.1.
    - Update
      $${\mathbf{θ}}^{(k + 1)} = {\mathbf{θ}}^{(k)} - \rho_{k}B_{k}^{- 1}U\left( {\mathbf{θ}}^{(k)} \right),$$
      where $B_{k}$ is a positive-definite approximation of the Hessian
      and $\rho_{k}$ is a step size obtained by line search.
    - Update $B_{k}$ using the standard BFGS formula.
    - Stop when
      $\parallel U\left( {\mathbf{θ}}^{(k + 1)} \right) \parallel$ is
      below a specified tolerance.
3.  **Observed information.** At convergence, compute
    $\mathcal{J}\left( {\widehat{\mathbf{θ}}}_{n} \right) = - H\left( {\widehat{\mathbf{θ}}}_{n} \right)$
    using Theorems 4.2–4.3.
4.  **Return** ${\widehat{\mathbf{θ}}}_{n}$ and
    $\mathcal{J}\left( {\widehat{\mathbf{θ}}}_{n} \right)$.

**Theorem 5.1 (Superlinear convergence).**  
Under standard assumptions (Nocedal and Wright, 2006), if
${\mathbf{θ}}^{(0)}$ is sufficiently close to
${\widehat{\mathbf{θ}}}_{n}$, the BFGS algorithm with exact analytical
gradients achieves superlinear convergence:
$$\parallel {\mathbf{θ}}^{(k + 1)} - {\widehat{\mathbf{θ}}}_{n} \parallel = o( \parallel {\mathbf{θ}}^{(k)} - {\widehat{\mathbf{θ}}}_{n} \parallel ).$$

In practice, the availability of exact gradients greatly improves both
speed and robustness relative to numerical differentiation.

### 5.3 Gradient Accuracy

Numerical differentiation can be used to validate the analytic
derivatives but is less efficient for routine computation.

**Lemma 5.1 (Finite-difference error).**  
Consider the central finite-difference approximation to
$\partial\ell/\partial\theta_{j}$ with step size $h > 0$:
$$D_{h} = \frac{\ell\left( {\mathbf{θ}} + h\mathbf{e}_{j} \right) - \ell\left( {\mathbf{θ}} - h\mathbf{e}_{j} \right)}{2h}.$$
Then
$$\left| D_{h} - \frac{\partial\ell}{\partial\theta_{j}} \right| = O\left( h^{2} \right) + O\!\left( \frac{\epsilon}{h} \right),$$
where $\epsilon$ is machine precision (approximately
$2.22 \times 10^{- 16}$ in double precision). The optimal step size is
$h^{*} \asymp (\epsilon/M)^{1/3}$, where $M$ bounds the third derivative
of $\ell$ in a neighborhood of $\mathbf{θ}$.

*Proof.* Standard finite-difference error analysis; see Nocedal and
Wright (2006), Chapter 8. $▫$

In contrast, the analytical gradients of Theorem 4.1 can be evaluated
with accuracy limited essentially only by floating-point roundoff and
require a single evaluation of $f$ per data point, rather than $2p$
evaluations per gradient component ($p = 5$ here) for central
differences.

### 5.4 Practical Recommendations

**Guideline 5.1 (Model selection within the GKw hierarchy).**

1.  Start from the simplest two-parameter models:
    - Beta$(\gamma,\delta + 1)$,
    - Kumaraswamy$(\alpha,\beta)$.
2.  If these are inadequate, consider three-parameter extensions:
    - EKw$(\alpha,\beta,\lambda)$,
    - McDonald$(\gamma,\delta,\lambda)$.
3.  For more complex patterns, move to four-parameter models:
    - BKw$(\alpha,\beta,\gamma,\delta)$,
    - KKw$(\alpha,\beta,\delta,\lambda)$.
4.  Use the full five-parameter GKw model only when the sample size is
    sufficiently large (e.g. $n \gtrsim 500$) to avoid
    over-parameterization and numerical instability.
5.  Compare candidate models using information criteria such as
    $${AIC} = - 2\ell\left( \widehat{\mathbf{θ}} \right) + 2p,\quad{BIC} = - 2\ell\left( \widehat{\mathbf{θ}} \right) + p\ln n,$$
    where $p$ is the number of free parameters.

**Guideline 5.2 (Diagnostics).**

1.  **Q–Q plot.** Compare empirical quantiles with theoretical quantiles
    from the fitted model.
2.  **Probability integral transform.** The transformed values
    $\{ F\left( x_{i};\widehat{\mathbf{θ}} \right)\}_{i = 1}^{n}$ should
    be approximately i.i.d. ${Uniform}(0,1)$.
3.  **Conditioning of the information matrix.** Check
    $\kappa\left( \mathcal{J}\left( \widehat{\mathbf{θ}} \right) \right)$,
    the condition number of the observed information; large values
    (e.g. $> 10^{8}$) indicate potential identifiability problems.
4.  **Positive definiteness.** All eigenvalues of
    $\mathcal{J}\left( \widehat{\mathbf{θ}} \right)$ should be strictly
    positive for valid standard error estimates.

### 5.5 Discussion

We have developed a rigorous mathematical framework for the Generalized
Kumaraswamy (GKw) family, including:

1.  **Hierarchical embedding.**  
    The GKw family neatly contains Beta, McDonald, Kumaraswamy,
    exponentiated Kumaraswamy, Beta–Kumaraswamy and
    Kumaraswamy–Kumaraswamy distributions as submodels, with explicit
    parameter mappings.

2.  **Likelihood theory.**  
    We derived explicit expressions for the log-likelihood, the score
    vector and the full observed information matrix in terms of the
    cascade transformations $v,w,z$, in a form suitable for stable
    numerical implementation.

3.  **Asymptotic properties.**  
    Under standard regularity conditions, the MLEs are consistent and
    asymptotically normal, with variance–covariance matrix obtained from
    the inverse observed information.

4.  **Computational considerations.**  
    Log-scale evaluations and carefully structured derivatives provide
    numerical stability and efficiency. In our C++ implementation via
    RcppArmadillo, analytical gradients and Hessians yield substantial
    speedups over finite-difference approximations, together with better
    numerical accuracy.

Open problems and possible extensions include:

- Closed-form expressions for moments of the full GKw distribution
  (currently only some sub-families, such as Kw and Beta, admit simple
  formulas).
- Analytic inversion of the BKw CDF (solving
  $I_{y}(\gamma,\delta + 1) = p$ for $y$, followed by inversion of the
  cascade).
- Multivariate generalizations using copulas constructed from GKw
  marginals.
- Fully Bayesian treatments with suitable priors on
  $(\alpha,\beta,\gamma,\delta,\lambda)$.

The `gkwdist` R package implements all the theoretical results described
in this vignette and provides a practical toolkit for likelihood-based
inference in bounded continuous data models.

------------------------------------------------------------------------

## References

Carrasco, J. M. F., Ferrari, S. L. P., & Cordeiro, G. M. (2010). **A new
generalized Kumaraswamy distribution.** *arXiv:1004.0911*.
[arxiv.org/abs/1004.0911](https://arxiv.org/abs/1004.0911)

Casella, G. and Berger, R. L. (2002). *Statistical Inference*, 2nd
ed. Duxbury Press, Pacific Grove, CA.

Cordeiro, G. M. and de Castro, M. (2011). A new family of generalized
distributions. *J. Stat. Comput. Simul.* **81**, 883–898.

Cox, D. R. and Hinkley, D. V. (1974). *Theoretical Statistics*. Chapman
and Hall, London.

Johnson, N. L., Kotz, S. and Balakrishnan, N. (1995). *Continuous
Univariate Distributions*, Volume 2, 2nd ed. Wiley, New York.

Jones, M. C. (2009). Kumaraswamy’s distribution: A beta-type
distribution with some tractability advantages. *Statist. Methodol.*
**6**, 70–81.

Kumaraswamy, P. (1980). A generalized probability density function for
double-bounded random processes. *J. Hydrol.* **46**, 79–88.

Lehmann, E. L. and Casella, G. (1998). *Theory of Point Estimation*, 2nd
ed. Springer, New York.

Mächler, M. (2012). Accurately computing
$\log\left( 1 - \exp\left( - |a| \right) \right)$. R package vignette,
<https://CRAN.R-project.org/package=Rmpfr>.

Nocedal, J. and Wright, S. J. (2006). *Numerical Optimization*, 2nd
ed. Springer, New York.

van der Vaart, A. W. (1998). *Asymptotic Statistics*. Cambridge
University Press, Cambridge.

------------------------------------------------------------------------

**Author’s address:**

J. E. Lopes  
Laboratory of Statistics and Geoinformation (LEG)  
Graduate Program in Numerical Methods in Engineering (PPGMNE)  
Federal University of Paraná (UFPR)  
Curitiba, PR, Brazil  
E-mail: <evandeilton@gmail.com>
