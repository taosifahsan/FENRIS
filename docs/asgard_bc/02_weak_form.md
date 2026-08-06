# The weak-form account of boundary conditions in ASGarD

*Companion to `00_session_findings.md` (measurements) and `01_*` (code layer).
This document derives, from the discontinuous-Galerkin weak form alone, why
ASGarD's `boundary_type` flag behaves the way it was measured to behave: why
sealing is exact, why `div` and `grad` must be flagged as a pair, why sealing a
set of terms imposes one condition and not several, and why a Robin coefficient
is a flux per unit $f$ rather than a logarithmic derivative.*

## 0. Notation and standing assumptions

Let $\Omega=[a,b]$ be partitioned into cells $I_j=[x_{j-1/2},x_{j+1/2}]$,
$j=1,\dots,N$, of width $h$. The discrete space is

$$V_h=\{\,v:\ v|_{I_j}\in P^k(I_j)\,\},$$

with **no continuity imposed between cells**. That single fact — that $V_h$ is
broken — is the origin of everything below. A function $w\in V_h$ has two values
at every interior edge; write

$$w^-_{j+1/2}=\lim_{x\uparrow x_{j+1/2}}w(x),\qquad
  w^+_{j+1/2}=\lim_{x\downarrow x_{j+1/2}}w(x),$$

$$[\![w]\!]=w^+-w^-,\qquad \{\!\{w\}\!\}=\tfrac12(w^++w^-).$$

At the two **domain** edges $x_{1/2}=a$ and $x_{N+1/2}=b$ only one of the two
traces exists; the other would have to come from outside the domain. This
asymmetry is the entire boundary-condition problem.

Standing assumptions, stated once:

* All results below use only $v\equiv 1\in V_h$ (valid for any $k\ge 0$) and
  endpoint traces. **Nothing depends on the polynomial degree $k$**, on the
  quadrature order, or on whether the basis is nodal Legendre or the
  hierarchical wavelet basis — the wavelet transform is an orthogonal change of
  basis on the *same* $V_h$ and commutes with every statement here.
* The upwinding parameter is $s\in\{+1,0,-1\}$ (`flux_type::upwind`, `central`,
  `downwind`). Where a result is independent of $s$ this is noted; the
  conservation results are.
* Diffusivities are assumed positive, $D>0$; this is needed only for the
  dissipativity argument of §7.
* FENRIS terms often carry a Jacobian, $\partial_t f = J^{-1}\partial_x F$
  (e.g. $J=v^2$). Then the conserved density is $Jf$ and the balance law is
  $\frac{d}{dt}\int J f = [F]_a^b$ — the mass matrix multiplies rows on the left
  and leaves the boundary-flux structure untouched. We suppress $J$ throughout.

---

## 1. The DG weak form of a first-order conservation law

Take the scalar conservation law in the sign convention used by FENRIS,

$$\partial_t f=\partial_x F,\qquad F=F(f,\partial_x f,x),$$

so that $F$ is the flux **in the direction of increasing $x$** and the right-hand
side is a divergence.

Multiply by a test function $v\in P^k(I_j)$ and integrate over one cell. Because
$v$ is supported on $I_j$ only, no information about the neighbours enters except
through the endpoints:

$$\int_{I_j} v\,\partial_t f
 =\int_{I_j} v\,\partial_x F
 =-\int_{I_j} v'\,F\;+\;\Big[v\,F\Big]_{x_{j-1/2}}^{x_{j+1/2}} .$$

This is the only integration by parts in the problem, and it has already produced
the whole story: a **volume term** $-\int v' F$ and a **surface bracket**
$[vF]$ evaluated at the two cell edges.

### 1.1 Why a numerical flux is needed, and why it must be single-valued

On $V_h$ the bracket is ambiguous: at $x_{j+1/2}$ the quantity $F$ computed from
the discrete solution has two values, $F(f^-)$ and $F(f^+)$. The DG method
replaces it with a **numerical flux** $\hat F_{j+1/2}$, a single number per edge,
built from both traces:

$$\boxed{\;\int_{I_j} v\,\partial_t f_h
 =-\int_{I_j} v'\,F(f_h)
 \;+\;v^-_{j+1/2}\hat F_{j+1/2}\;-\;v^+_{j-1/2}\hat F_{j-1/2}\;}\tag{1}$$

Note carefully which trace of $v$ multiplies which flux: cell $I_j$ sees its own
$v^-$ at its right edge and its own $v^+$ at its left edge, but the *same*
$\hat F$ that its neighbour sees. Two requirements pin $\hat F$ down:

1. **Consistency.** $\hat F(w,w)=F(w)$, so that a smooth exact solution
   satisfies (1) up to truncation error.
2. **Single-valuedness (conservation).** $\hat F_{j+1/2}$ must be *one* number
   shared by cells $I_j$ and $I_{j+1}$. If the two cells were allowed to use
   different values, the flux leaving one cell would not equal the flux entering
   the next, and mass would be created at every interior edge. Single-valuedness
   is not a convenience; it *is* the discrete statement of local conservation.
   §2 makes this exact.

### 1.2 The flux ASGarD actually assembles

For a linear flux $F=c(x)f$ the ASGarD edge blocks correspond to

$$\hat F_{j+1/2}=c\,\{\!\{f\}\!\}-\frac{s\,|c|}{2}[\![f]\!]
 \;=\;\begin{cases} c\,f^- & c>0,\ s=+1\ \text{(upwind)}\\[2pt]
                    c\,f^+ & c<0,\ s=+1,\end{cases}\tag{2}$$

i.e. the standard upwind flux: the value is taken from whichever side the
characteristic comes from. Setting $s=0$ gives the central flux, $s=-1$ the
downwind flux. (In the assembled matrix the edge blocks are rank-one outer
products of endpoint basis values, $\texttt{to\_left}=\ell_L\otimes\ell_L$,
$\texttt{from\_left}=\ell_L\otimes\ell_R$, etc., each carrying $h^{-1/2}$ per
factor, hence the $1/h$ edge scale; the volume block carries $2/h$ from the
reference-cell derivative. These scalings are bookkeeping and play no role in
what follows.)

At a **domain** edge there is no exterior trace, so (2) cannot be evaluated. The
flag chooses among the only three self-consistent options, and that choice is the
boundary condition. This is developed in §2.

---

## 2. Telescoping: all non-conservation lives at the two domain edges

Set $v\equiv 1$ in (1). This is legitimate for every $k\ge0$: constants belong to
$P^k$ on each cell. The volume term dies because $v'=0$, and (1) collapses to a
statement about the cell mass $m_j=\int_{I_j}f_h$:

$$\frac{d m_j}{dt}=\hat F_{j+1/2}-\hat F_{j-1/2}.\tag{3}$$

Each cell is an exact bookkeeping ledger: what it gains is what came in the left
edge minus what left the right edge. Now sum over $j=1,\dots,N$. Every
**interior** flux $\hat F_{j+1/2}$, $1\le j\le N-1$, appears exactly twice — with
$+$ from cell $I_j$ and $-$ from cell $I_{j+1}$ — and because it is *the same
number* in both (single-valuedness, §1.1), the two occurrences cancel identically.
The sum telescopes, leaving

$$\boxed{\;\frac{d}{dt}\int_\Omega f_h\;=\;\hat F_b-\hat F_a\;}\tag{4}$$

where $\hat F_a\equiv\hat F_{1/2}$ and $\hat F_b\equiv\hat F_{N+1/2}$ are the two
domain-edge fluxes.

Equation (4) is the master statement of this document, and it deserves to be read
slowly:

* The cancellation is **algebraic, not asymptotic**. It does not improve with
  $h\to0$ or $k\to\infty$; it is exact at every resolution, for any upwinding
  parameter $s$, for any coefficient $c(x)$ however rough, and for any quadrature
  rule used on the volume term. The volume term never entered (4) at all.
* Therefore **the interior discretization cannot create or destroy mass**. It is
  incapable of it. Every unit of mass that appears or disappears in an ASGarD run
  passed through one of exactly two numbers, $\hat F_a$ and $\hat F_b$.
* Consequently, in DG a "boundary condition" is *nothing more and nothing less
  than a rule for those two numbers*. There is no other place to put one. Any
  discussion of boundary conditions that is not a discussion of $\hat F_a$ and
  $\hat F_b$ is misdirected.

### 2.1 ASGarD's three rules for a domain edge

With no exterior trace available, (2) degenerates and the flag selects:

| `boundary_type` | assembled block | resulting $\hat F_{\rm wall}$ | reading |
|---|---|---|---|
| `none` ("free") | edge block written with the **full** coefficient, no upwind blend | $c\,f^{\rm int}$ | outflow / extrapolation |
| `bothsides` ("sealed") | **no block written at all** | $0$ | zero flux |
| `left` / `right` | as above, one wall only | $0$ on that wall | one-sided seal |

Two remarks.

**Sealing is exact, not approximate.** `bothsides` does not *approximate* a
zero-flux condition by adding a small block; it *omits* the block. The
corresponding row contribution is literally absent from the matrix, so
$\hat F_{\rm wall}=0$ to machine precision, at every level and degree. This is
why the measured mass error for the sealed pure-diffusion test was $2\times10^{-6}$
— that is the time integrator's error, not the spatial operator's, which is $0$.

**"Dirichlet flux" is a trap.** The source comment at the `bothsides` branch reads
*"dirichlet flux, nothing to set"*. It means a Dirichlet condition **on the
flux**, i.e. $\hat F=0$ — which as a condition on $f$ is a **Neumann** (zero
normal derivative) condition when the term is diffusive, and a no-outflow
condition when the term is advective. It is not a Dirichlet condition on $f$.
Confusing the two is the single most common way to mis-flag a term.

**The free flux is only well-posed on outflow.** $\hat F=c f^{\rm int}$ is the
correct numerical flux at a wall precisely when the characteristic points *out*
of the domain, since then the upwind value genuinely is the interior one. When
the characteristic points *in* — as it does at the outer velocity wall for a drag
term, where drag pushes toward $v=0$, i.e. inward — the exterior state is not
determined by the interior one, and extrapolating amounts to positing a reservoir
outside the domain whose density equals the solution's own wall trace. Then (4)
reads $\dot M \ni c\,f^{\rm int}_{\rm wall}$: a source term **proportional to the
unknown**, i.e. a linear feedback on the boundary value. This is not a
discretization error that shrinks with $h$; it is a modelling error of $O(1)$,
and refinement merely converges faster to the wrong answer. That is the exact
signature recorded for ICRF_2D — $+63\%$ manufactured particles by $t=100$, and
*worse* under refinement — and for LHCD_2D's steady $0.0043\%/\tau$ leak, which
was large enough to fake a steady state by balancing the RF input against wall
loss.

---

## 3. LDG: a second-order term needs *two* numerical traces

This is the key section. It explains why `div` and `grad` each carry a
`boundary_type`, and why the two flags are meaningless individually.

### 3.1 Why one integration by parts is not enough

Consider the diffusion term

$$\partial_t f=\partial_x\!\big(D\,\partial_x f\big),\qquad D>0.$$

The naive move — test, integrate by parts once, and write
$-\int v' D f_h' + [vD\hat{f'}]$ — fails for a reason that is structural, not
technical. The quantity $D\partial_x f_h$ is a *derivative of a broken function*.
Across an edge, $f_h$ jumps, so $\partial_x f_h$ contains a Dirac mass that the
cellwise derivative silently discards; the resulting operator has no consistent
edge flux and, for $k=0$, no derivative at all. One cannot integrate by parts
twice on a space that is not $H^1$.

The **Local Discontinuous Galerkin (LDG)** fix is to refuse to differentiate
twice. Split the second-order operator into a first-order *system* by introducing
the auxiliary variable

$$q=D\,\partial_x f\;\Longrightarrow\;
\begin{cases} q=D\,\partial_x f & \text{(the \texttt{grad} equation)}\\[2pt]
              \partial_t f=\partial_x q & \text{(the \texttt{div} equation)}\end{cases}\tag{5}$$

Now every derivative that appears is first order, and §1's machinery applies —
**twice**.

### 3.2 The two coupled weak forms

Discretize $q$ in the *same* broken space, $q_h\in V_h$. Test the `grad` equation
with $w\in P^k(I_j)$ and integrate by parts (take $D$ constant for clarity; a
variable $D$ simply rides along inside the integrand and the traces):

$$\int_{I_j} w\,q_h
 =-\int_{I_j} D\,w'\,f_h
 \;+\;D\,w^-_{j+1/2}\,\widehat f_{j+1/2}
 \;-\;D\,w^+_{j-1/2}\,\widehat f_{j-1/2}.\tag{6}$$

Test the `div` equation with $v\in P^k(I_j)$:

$$\int_{I_j} v\,\partial_t f_h
 =-\int_{I_j} v'\,q_h
 \;+\;v^-_{j+1/2}\,\widehat q_{j+1/2}
 \;-\;v^+_{j-1/2}\,\widehat q_{j-1/2}.\tag{7}$$

**Stare at the pair (6)–(7).** Two integrations by parts have produced **two
independent numerical traces**:

* $\widehat f$ — a single-valued trace of $f$, appearing **only in the `grad`
  equation**;
* $\widehat q$ — a single-valued trace of $q$, appearing **only in the `div`
  equation**.

Neither equation can see the other's trace. Equation (6) has no place to put a
condition on the flux $q$; equation (7) has no place to put a condition on the
value $f$. So:

$$\boxed{\;
\begin{aligned}
&\text{the flag on the \texttt{div} factor sets }\widehat q\text{ at the wall}
   \;\;\Longleftrightarrow\;\; \text{a condition on the \emph{flux}},\\
&\text{the flag on the \texttt{grad} factor sets }\widehat f\text{ at the wall}
   \;\;\Longleftrightarrow\;\; \text{a condition on the \emph{value}}.
\end{aligned}\;}$$

That is the whole answer to "why does each of `div` and `grad` carry a boundary
flag?". They are not two chances to say the same thing. They are two *different*
degrees of freedom at each wall, and a well-posed second-order problem needs
exactly one condition per wall — so they must be chosen as a **complementary
pair**, one prescribed and one taken from the interior. §4 enumerates the four
possibilities.

Interior edges have no such freedom: both traces exist on both sides, and both
$\widehat f$ and $\widehat q$ are ordinary two-sided numerical fluxes. LDG
stability requires them to be taken from **opposite** sides (the *alternating*
flux, e.g. $\widehat f=f^-$ with $\widehat q=q^+$, or the mirror image); §7 shows
why.

### 3.3 How ASGarD encodes this

In ASGarD a diffusion term is a *chain* `{term_div(...), term_grad(...)}`, and the
assembled operator is the matrix product $\mathrm{Div}\cdot\mathrm{Grad}$: `grad`
acts first on $f$ to make $q_h$, then `div` acts on $q_h$. Two structural facts in
`gen_tri_cmat` implement the pairing:

1. **`grad` is assembled as the negative transpose of a `div` matrix.** After
   building the tri-block matrix, the routine applies
   $\mathrm{Grad}=-\,T^{\mathsf T}$. This is the discrete statement that
   $\partial_x^{\;*}=-\partial_x$. It has a free bonus: transposition swaps the
   `from_left`/`from_right` couplings (indeed $\texttt{from\_left}^{\mathsf T}
   =\texttt{from\_right}$), which flips the effective sign of the upwinding
   parameter $s$. So **the interior alternation of $\widehat f$ and $\widehat q$
   is automatic** — the user never chooses it, and cannot get it wrong.

2. **The `grad` flag is flipped before assembly**
   (`bothsides`$\leftrightarrow$`none`, `left`$\leftrightarrow$`right`). Writing
   $T[\varphi]$ for the div-form matrix assembled with flag $\varphi$, a chain
   with user flags $(\varphi_d,\varphi_g)$ produces

   $$\mathrm{Div}\cdot\mathrm{Grad}\;=\;-\,T[\varphi_d]\;T[\mathrm{flip}(\varphi_g)]^{\mathsf T}.\tag{8}$$

   This is a **Gram form** $-BB^{\mathsf T}$ — symmetric and negative
   semi-definite — precisely when the two factors coincide, i.e. when

   $$\varphi_g=\mathrm{flip}(\varphi_d).\tag{9}$$

Condition (9) *is* the consistent-pair condition of §3.2, in matrix form. Since
`term_grad`'s default flag is `none`, and $\mathrm{flip}(\texttt{none})
=\texttt{bothsides}$, the idiom

```cpp
term_1d({term_div(D, boundary_type::bothsides), term_grad(D)})   // grad default = none
```

is matched, and is the sealed (Neumann) pair. This is exactly the configuration
FENRIS converged on in all four solvers.

---

## 4. The four pairings

At a given wall, each of $\widehat f$ and $\widehat q$ is either **prescribed**
(here: set to zero, the block deleted) or **free** (taken from the interior
trace). Four combinations; two are the classical boundary conditions, two are
ill-posed. The rule to be derived in §7 is:

> **Exactly one of $\{\widehat f,\widehat q\}$ must be prescribed at each wall;
> the other must be the interior trace.**

### 4.1 Neumann — zero flux — the *sealed* pair

$$\widehat q_{\rm wall}=0,\qquad \widehat f_{\rm wall}=f^{\rm int}.$$

The `div` factor's wall block is deleted, so by (4) $\dot M=0$ **exactly**. The
`grad` factor still sees the true interior trace of $f$, so $q_h\approx D f'$ is
computed correctly right up to the wall — the interior physics is not distorted.
Any constant is a steady state.

*Flags:* `{div bothsides, grad none(default)}` — matched by (9).
*Measured* (pure diffusion $\partial_t f=\partial_x(f')$ on $[0,2]$, exact steady
state $f=\text{const}$): flat steady state with the wall values sitting on the
constant, max deviation $5.3\times10^{-4}$, mass error $2\times10^{-6}$. The mass
error is the time integrator; the spatial operator's is identically zero.

### 4.2 Dirichlet $f=0$ — the *open* pair

$$\widehat f_{\rm wall}=0,\qquad \widehat q_{\rm wall}=q^{\rm int}.$$

Now the `grad` equation is told the wall value is zero, and the `div` equation
takes whatever flux the solution generates. Mass leaves — as it must; a Dirichlet
wall is an absorbing wall.

*Flags:* `{div none, grad bothsides}` — matched by (9).
*Measured:* decay rate $2.467$. The exact ground mode of $\partial_t f=f''$ on
$[0,2]$ with $f(0)=f(2)=0$ is $\sin(\pi x/2)$ with eigenvalue
$(\pi/L)^2=(\pi/2)^2=2.4674$. Four-digit agreement. This is a sharp test: it
confirms not merely that mass leaves, but that the operator realizes the correct
Dirichlet spectrum.

### 4.3 Both prescribed — over-determined

$$\widehat f_{\rm wall}=0\quad\text{and}\quad\widehat q_{\rm wall}=0 .$$

One asserts $f=0$ *and* $\partial_x f=0$ at the same point. For a second-order
problem this is one condition too many; the continuum problem is generically
unsolvable, and the discrete operator (8) is $-T[\texttt{both}]\,
T[\texttt{none}]^{\mathsf T}$ — a product of two *different* matrices, hence
neither symmetric nor sign-definite. §7 shows the wall then contributes
$-(fq)^{\rm int}$ to the energy budget, a quantity of no fixed sign: whenever
$f$ and $q$ have opposite signs at the wall the operator *injects* energy.

*Flags:* `{div bothsides, grad bothsides}` — violates (9).
*Measured:* $f\to1.4\times10^{6}$, mass error $8\times10^{2}$. Blow-up.

**Locality.** The bracket in (1) is a sum of two *separate* endpoint terms, so the
two walls are algebraically independent. Corrupting one wall cannot corrupt the
other. Measured exactly so: `{div bothsides, grad right/left}` blows up on that
wall only, while the other wall stays exactly Neumann ($0.2228$ against the
constant $0.2215$). This locality is itself evidence for the whole bookkeeping —
there really are four independent numbers ($\widehat f,\widehat q$ at each of two
walls), not one global "boundary condition".

### 4.4 Neither prescribed — under-determined

$$\widehat f_{\rm wall}=f^{\rm int},\qquad \widehat q_{\rm wall}=q^{\rm int}.$$

Both traces extrapolated; nothing pins the solution at the wall. What one gets is
then decided by the *upwinding* convention, which for a parabolic operator has no
physical meaning at all. Worse, §7 shows the two walls acquire energy
contributions of **opposite sign** — $+(fq)^{\rm int}$ at $b$ and $-(fq)^{\rm int}$
at $a$ — so the left and right walls behave differently even though the flags
are symmetric.

*Flags:* `{div none, grad none(default)}` — violates (9).
*Measured:* precisely that pathology — "upwind-asymmetric mongrel, left and right
behave differently despite symmetric flags", decay rate $0.77$ matching no
Dirichlet mode, profile deviation $5.6\times10^{2}$, and negative mass in
three-term tests.

### 4.5 Summary table

| $\widehat q$ (div) | $\widehat f$ (grad) | user flags | (9)? | condition realized | measured |
|---|---|---|---|---|---|
| $0$ | interior | `div bothsides`, `grad none` | ✓ | Neumann, $\dot M=0$ | flat, mass err $2\!\times\!10^{-6}$ |
| interior | $0$ | `div none`, `grad bothsides` | ✓ | Dirichlet $f=0$ | decay $2.467=(\pi/2)^2$ |
| $0$ | $0$ | `div bothsides`, `grad bothsides` | ✗ | over-determined | $f\to1.4\!\times\!10^{6}$ |
| interior | interior | `div none`, `grad none` | ✗ | under-determined | asymmetric, rate $0.77$ |

### 4.6 A note on penalty terms

`term_penalty` appears to invert the convention — `bothsides` *adds* boundary
blocks instead of deleting them. This is consistent, not contradictory. A penalty
term has **no volume integral**; it is purely the jump stabilization
$\sum_e\sigma\,[\![f]\!][\![v]\!]$. At a wall there is no jump, so `bothsides`
supplies the missing partner by adding $\sigma f^{\rm int}v^{\rm int}$, which
penalizes the wall *value* toward zero — a Nitsche-style weak Dirichlet pin. So
for `div`/`grad` the flag names **a flux to delete**; for `penalty` it names
**a value to pin**. Same enum, different operator, different meaning.

---

## 5. The sum rule, and the emergent Robin condition

Real kinetic equations have several flux channels,

$$\partial_t f=\partial_x F,\qquad F=\sum_k F_k
 =\underbrace{A f}_{\text{drag}}+\underbrace{D_1\partial_x f}_{\text{coll.}}
  +\underbrace{D_2\partial_x f}_{\text{QL}}+\cdots$$

each assembled as its own term with its own flag. What does sealing a *subset*
$S$ actually enforce?

### 5.1 Statement

$$\boxed{\;\text{Sealing }S\ \text{imposes the single weak condition}\quad
 \sum_{k\in S}F_k\ \longrightarrow\ 0\quad\text{at the wall.}\;}$$

**One condition per wall, not one per term.** This is not obvious: the assembly
deletes each sealed term's wall block *individually*, so one might expect $|S|$
separate conditions. It does not.

### 5.2 Derivation

The weak form is linear in the terms, so (1) holds with
$\hat F=\sum_k\hat F_k$ and the mass balance (4) becomes

$$\frac{d}{dt}\int_\Omega f_h=\sum_k\hat F_k\Big|_a^b
 =\sum_{k\notin S}\hat F_k\Big|_a^b,\tag{10}$$

since sealed terms contribute nothing at the wall. That is the trivial half.

For the nontrivial half, apply (3) to the **last cell** $I_N$, whose right edge
*is* the wall $b$:

$$\frac{dm_N}{dt}=\hat F_b-\hat F_{N-1/2}
 =\sum_{k\notin S}\hat F_k(b)\;-\;\sum_{k}\hat F_k(x_{N-1/2}).$$

Note the asymmetry: at the *interior* edge $x_{N-1/2}$ there are no flags, so
**all** terms contribute; at the wall only the unsealed ones do. At steady state
$dm_N/dt=0$, hence

$$\sum_{k}F_k\big(x_{N-1/2}\big)=\sum_{k\notin S}F_k(b).$$

The left side is the full physical flux, evaluated one edge inside. Let that edge
approach the wall (equivalently, use that the discrete solution's trace is
continuous to $O(h^{p})$ across the last cell):

$$\sum_{k}F_k(b)=\sum_{k\notin S}F_k(b)
\quad\Longrightarrow\quad
\boxed{\ \sum_{k\in S}F_k(b)=0\ }\tag{11}$$

One equation. The individual sealed fluxes are *not* separately zero — only their
sum is. The mechanism is that sealing removes a term's ability to carry flux
*out*, forcing the solution to arrange itself so that the flux it would have
carried is not needed; with several sealed channels the solution has only one
scalar (its wall log-slope) with which to satisfy them, and (11) is the one
equation that scalar solves.

### 5.3 Emergent Robin

Take $S=\{\text{drag},\text{diffusion}\}$ with $F=Af+D\partial_x f$. Then (11)
reads

$$A f+D\,\partial_x f=0\quad\text{at the wall}
\qquad\Longleftrightarrow\qquad
\frac{f'}{f}\bigg|_{\rm wall}=-\frac{A}{D}.\tag{12}$$

This is a **Robin condition** — and nobody wrote one. It *emerged* from sealing
two terms that share a single wall. This is the correct physics: the wall is
impermeable, and an impermeable wall in the presence of drag does not give
$f'=0$; it gives the drag–diffusion balance (12), which is exactly the
Maxwell–Boltzmann-like exponential foot that a no-flux kinetic equilibrium must
have. Sealing gets it for free; imposing $f'=0$ by hand would be *wrong*.

### 5.4 The measured verification

The toy $\partial_t f=\partial_x(2x f+D_1f'+D_2f')$ on $[0,2]$ was run over all
eight sealing subsets. Writing $A(2)=4$, all predictions come from (11):

| $S$ | (11) reads | predicted $f'/f$ | measured | predicted leak | measured |
|---|---|---|---|---|---|
| $\{A,D_1,D_2\}$ | $Af+(D_1{+}D_2)f'=0$ | $-A/(D_1{+}D_2)=-2.67$ | $-2.60$ | $0$ | $\dot M=0$ |
| $\{D_1,D_2\}$ | $(D_1{+}D_2)f'=0$ | $f'=0$ | $f'=0$ | $A f(2)=+4$ | $+4$ |
| $\{A,D_1\}$ | $Af+D_1f'=0$ | $-A/D_1=-4$ | $-3.69$ | $D_2f'=-2$ | $-1.98$ |
| $\{A,D_2\}$ | $Af+D_2f'=0$ | $-A/D_2=-8$ | $-6.3$ | $D_1f'=-8$ | $-7.75$ |
| $\{D_1\}$ | $D_1f'=0$ | $f'=0$ | $f'=0$ | $Af(2)=+4$ | $+3.884$ |
| $\{D_2\}$ | $D_2f'=0$ | $f'=0$ | $f'=0$ | $Af(2)=+4$ | $+3.884$ |

Three points worth extracting.

**The data over-determine the rule and it passes.** All six rows are
simultaneously consistent with a *single* parameter set $D_1=1$, $D_2=1/2$,
$A(2)=4$, $f(2)\approx1$. Row 1 fixes $D_1+D_2=1.5$; row 3 fixes $D_1=1$; row 4
fixes $D_2=1/2$; rows 2–4 then predict the leaks $+4$, $-2$, $-8$ with no further
freedom, and all three are measured. The sum rule is not fitted; it is tested.

**Sealing one diffusion channel silences both.** Rows 5 and 6 are a corollary,
not a coincidence. (11) with $S=\{D_1\}$ gives $D_1f'=0$, hence $f'=0$, hence
$D_2f'=0$ *automatically* — because both channels are proportional to the **same**
quantity $f'$. So only the drag leaks, and it leaks the same amount either way.
The two runs use genuinely different matrices, yet agree to three digits
($3.884$), because they enforce the same single condition.

**The residual errors are the trace error, not a failure of the rule.** The
measured log-slopes fall short by $2.6\%$, $7.8\%$, $21\%$ for
$\gamma=2.67,4,8$ — a deficit scaling as $\gamma^2$ to within $20\%$
($0.026/\gamma^2=0.0037$, $0.078/\gamma^2=0.0049$, $0.21/\gamma^2=0.0033$). This
is the expected $O((\gamma h)^2)$ error of resolving an exponential wall layer
$f\sim e^{\gamma x}$ with a polynomial across the last cell. The sum rule (11) is
**exact on the numerical fluxes**; it is only the post-processed pointwise
quantity $f'/f$ that carries $O(h^2)$. The steeper the required layer, the more
the wall must be resolved — which is precisely why the seed file annotates the
$-8$ case "under-resolved".

---

## 6. Robin conditions: a flux coefficient, not a log-derivative

### 6.1 Where a Robin term enters

A Robin condition adds a wall flux proportional to the wall value. In the weak
form there is exactly one slot for it — the bracket in (1):

$$\hat F^{\rm R}_b=r_b\,f^-(b),\qquad \hat F^{\rm R}_a=r_a\,f^+(a).\tag{13}$$

`gen_robin_cmat` writes precisely this: a rank-one block
$-\,r_a/h\cdot\texttt{to\_left}$ on cell $0$ and $+\,r_b/h\cdot\texttt{to\_right}$
on the last cell — the *same* rank-one endpoint outer product that a free `div`
flux writes, with $r$ in place of $c$.

### 6.2 The units, and why the confusion is dangerous

Because $r$ sits where a flux sits, dimensional analysis is forced:

$$[r]=\frac{[F]}{[f]}=\textbf{flux per unit }f .$$

For $f$ a phase-space density and $F$ a phase-space flux in a velocity
coordinate, **$r$ is a velocity** (times whatever Jacobian the term carries). It
is *not* a logarithmic derivative.

Contrast with the emergent Robin (12): there the natural object is
$\gamma=f'/f$, with $[\gamma]=1/[x]$. These are different quantities with
different dimensions:

$$r\ \sim\ \frac{\text{flux}}{f}\qquad\text{versus}\qquad \gamma=\frac{f'}{f}\sim\frac{1}{\text{length}} .$$

Passing $A/D$ (a log-derivative) to `set_right_robin` is dimensionally wrong.
The correct argument is the *term's own wall flux coefficient*. Concretely, in
ICRF_1D the drag term is `term_div(c)` with $c(v)=-v^2\eta(v)$, and the historical
line was `set_right_robin(v_max^2*eta(v_max))` $=-c(v_{\max})$ — a velocity-like
quantity, as required.

### 6.3 "Free bracket + Robin" is *identically* "sealed bracket"

This is pure algebra, and it explains the bit-identical measurements.

At the right wall, a `div` term with coefficient $c$ contributes to the bracket:

$$\text{free (\texttt{none})}:\quad \hat F_b=c(b)\,f^-,
\qquad\qquad
\text{sealed (\texttt{bothsides})}:\quad \hat F_b=0 .$$

Adding a Robin block (13) to the free term gives

$$\hat F_b=\big(c(b)+r_b\big)\,f^- .$$

Hence

$$\boxed{\;\text{free}(c)\;+\;\text{Robin}\big(r=-c(b)\big)\;\equiv\;\text{sealed}(c)\;}\tag{14}$$

and this is an identity of *matrices*, not of solutions. Both constructions
multiply the **same** rank-one block $\ell_R\otimes\ell_R/h$; the free-plus-Robin
version carries coefficient $c(b)+r_b$, the sealed version has no block at all.
When $r_b=-c(b)$ exactly, the two axpys cancel to $0$ in IEEE arithmetic and the
assembled matrices are bit-identical.

This predicts, correctly, the graded agreement that was measured:

* **Toy problem: $7\times10^{-15}$, not $0$** — because $c(b)$ came from a
  quadrature evaluation of the coefficient function while $r$ was a hand-typed
  literal, so the two differ in the last bits.
* **ICRF_1D at $t=200$ and LHCD_2D: bit-identical** — because there
  $r=v_{\max}^2\eta(v_{\max})$ and $c(v_{\max})=-v_{\max}^2\eta(v_{\max})$ are the
  *same expression* evaluated at the same point, so the cancellation is exact.

### 6.4 Why sealing is strictly better than the Robin spelling

(14) says the two are equivalent when the Robin value is right. Sealing is
preferable because it is right *by construction*:

1. **No hand evaluation.** One need not compute $c$ at the wall; the assembler
   already knows it.
2. **Exact for variable and time-dependent coefficients.** If $c=c(x,t)$, a
   literal Robin value is stale the moment the coefficient is recomputed;
   deleting the block is not.
3. **Sign-proof.** The commonest failure is the sign in (14).
4. **Composes.** With several terms, sealing the set $S$ realizes the sum rule
   (11) automatically; reproducing it with Robin values requires getting
   $\sum_{k\in S}c_k(b)$ right by hand.

This is why the endpoint of the FENRIS work was `bothsides` on every outermost
`div` and **no Robin lines at all** — verified bit-identical wherever a correct
Robin already existed, and a genuine fix for LHCD_1D where none did.

### 6.5 Genuine physical Robin walls

None of this forbids a real Robin condition. A partially absorbing wall
$\alpha f+\beta q=0$ with physically given $\alpha,\beta$ is realized by sealing
the `div` and adding a Robin block with the physical $r$. The sum rule then reads

$$\sum_{k\in S}F_k(b)\;+\;r\,f(b)=0,$$

i.e. the Robin block is simply *one more member of the sealed set, whose flux is
$r f$*. §7 shows the sign condition it must satisfy: $r\le0$ at the right wall
(absorption), and $r\ge0$ at the left; the opposite sign injects energy.

---

## 7. Adjoint consistency and dissipativity: *deriving* the pairing rule

§4 asserted that exactly one of $\widehat f,\widehat q$ must be prescribed. Here
that rule is derived — not from counting degrees of freedom, but from demanding
that the operator not manufacture energy.

### 7.1 The LDG energy identity

Take $D=1$. Sum (7) over cells with $v=f_h$ and (6) with $w=q_h$. Using the
edge-regrouping identity

$$\sum_j\Big(v^-_{j+1/2}\hat F_{j+1/2}-v^+_{j-1/2}\hat F_{j-1/2}\Big)
=-\sum_{\text{int }e}\hat F_e\,[\![v]\!]_e\;+\;v^-(b)\hat F_b-v^+(a)\hat F_a,$$

and the cellwise product rule
$\sum_j\int_{I_j}(f'q+q'f)=\sum_j\int_{I_j}(fq)'
=-\sum_{\text{int}}[\![fq]\!]+(fq)^-(b)-(fq)^+(a)$,
the two summed equations add to

$$\frac12\frac{d}{dt}\|f_h\|^2+\|q_h\|^2
=\underbrace{\sum_{\text{int }e}\Big[(\{\!\{q\}\!\}-\widehat q)[\![f]\!]
 +(\{\!\{f\}\!\}-\widehat f)[\![q]\!]\Big]}_{\text{interior}}
\;+\;B_b+B_a,\tag{15}$$

where the identity $[\![fq]\!]=\{\!\{f\}\!\}[\![q]\!]+\{\!\{q\}\!\}[\![f]\!]$ was
used.

**Interior.** With the alternating flux $\widehat f=f^-$, $\widehat q=q^+$ one has
$\{\!\{f\}\!\}-\widehat f=\tfrac12[\![f]\!]$ and
$\{\!\{q\}\!\}-\widehat q=-\tfrac12[\![q]\!]$, so the bracket is
$-\tfrac12[\![q]\!][\![f]\!]+\tfrac12[\![f]\!][\![q]\!]=0$: the interior
contribution **cancels identically**. This is the defining property of LDG, and
in ASGarD it is automatic because $\mathrm{Grad}=-\mathrm{Div}^{\mathsf T}$ swaps
the upwind side (§3.3). The user cannot break it.

### 7.2 The wall term, in one formula

Everything therefore reduces to the two wall terms. At the right wall,

$$B_b=-(fq)^-+f^-\widehat q_b+q^-\widehat f_b .$$

Complete the square:

$$\boxed{\;B_b=\widehat f_b\,\widehat q_b\;-\;\big(f^--\widehat f_b\big)\big(q^--\widehat q_b\big)\;}\tag{16}$$

(expand to check). This single expression generates the entire §4 table.

For $B_b\le0$ for *all* data — which is what stability requires, since nothing
constrains the signs of $f^-$ and $q^-$ independently — both terms must be
controlled, and the only way to kill them with homogeneous choices is:

* kill $\widehat f_b\widehat q_b$ by setting **one** trace to zero, and
* kill $(f^--\widehat f_b)(q^--\widehat q_b)$ by setting the **other** trace to
  the interior value.

That is exactly the pairing rule, now derived rather than asserted. Evaluating
(16) in the four cases:

| pairing | $\widehat f_b\widehat q_b$ | $(f^-{-}\widehat f)(q^-{-}\widehat q)$ | $B_b$ | verdict |
|---|---|---|---|---|
| Neumann: $\widehat q=0$, $\widehat f=f^-$ | $0$ | $0$ | $\;0\;$ | dissipative, $\dot M=0$ |
| Dirichlet: $\widehat f=0$, $\widehat q=q^-$ | $0$ | $0$ | $\;0\;$ | dissipative, mass leaves |
| both zero | $0$ | $f^-q^-$ | $-(fq)^-$ | **indefinite** |
| both free | $f^-q^-$ | $0$ | $+(fq)^-$ | **indefinite** |

Both *consistent* pairings give $B_b=0$ — the wall contributes exactly nothing to
the energy budget — so (15) becomes

$$\frac12\frac{d}{dt}\|f_h\|^2=-\|q_h\|^2\;\le\;0,$$

unconditional dissipation, at any $h$, any $k$. Both *inconsistent* pairings
leave $\pm(fq)^-$, a product of two independently-signed quantities: whenever
$f$ and $q$ have the offending relative sign at the wall, the operator pumps
energy in. This is the mechanism of the measured blow-up to $1.4\times10^{6}$.

### 7.3 Two predictions this formula makes, both observed

**Locality.** $B_a$ and $B_b$ are separate terms; corrupting one leaves the other
at $0$. Hence `{div bothsides, grad right}` must be ill-posed on one wall while
the other remains exactly Neumann — measured ($0.2228$ vs. the constant $0.2215$
on the healthy wall).

**Left–right asymmetry of the "both free" case.** At the left wall the available
traces are $f^+,q^+$ and the mirror of (16) gives $B_a=-\widehat f_a\widehat q_a
+(f^+-\widehat f_a)(q^+-\widehat q_a)$, so "both free" yields
$B_a=-(fq)^+$ while $B_b=+(fq)^-$ — **opposite signs at the two ends**. The two
walls are therefore anti-dissipative under complementary conditions, and the
solution is asymmetric even though the flags are symmetric. This is exactly the
recorded "upwind-asymmetric mongrel, left and right behave differently despite
symmetric flags" with decay rate $0.77$ belonging to no Dirichlet mode.

### 7.4 Inhomogeneous data and the sign of a Robin coefficient

(16) also handles non-zero data, which is how one checks that a *physical* Robin
wall is admissible.

* **Dirichlet data $g$:** $\widehat f_b=g$, $\widehat q_b=q^-$. Then
  $B_b=g\,q^-$ — a boundary work term controlled once $g$ is given. Fine.
* **Robin:** $\widehat q_b=r f^-$ with $\widehat f_b=f^-$. Then (16) gives
  $B_b=r\,(f^-)^2$, so **the operator is dissipative iff $r\le0$ at the right
  wall** (and $r\ge0$ at the left, by the mirror formula). A Robin coefficient of
  the wrong sign is an energy source of strength $|r|f^2$ — a positive feedback
  on the wall value, not a bounded perturbation.

That last line is the same pathology as the free-flux-on-inflow problem of §2.1,
seen from the energy side: the free bracket at an inflow wall *is* a Robin term
with $r=c(b)$ of the destabilizing sign, and it is the reason the uncancelled
drag bracket in ICRF_2D manufactured $63\%$ extra particles and got worse under
refinement.

### 7.5 Adjoint consistency, stated compactly

The matrix identity (8)–(9) is the finite-dimensional shadow of all of this. A
matched pair gives

$$\mathcal{L}=-\,T[\varphi]\,T[\varphi]^{\mathsf T},$$

a Gram form: **symmetric and negative semi-definite** for $D>0$, hence

* real non-positive spectrum — no spurious growing modes, and the stiff
  eigenvalues sit on the negative real axis where implicit and explicit
  integrators expect them;
* its kernel is $\{f:T[\varphi]^{\mathsf T}f=0\}$, which for $\varphi=$`bothsides`
  contains the constants (the Neumann null space) and for $\varphi=$`none` forces
  the wall values to zero (the Dirichlet spectrum, eigenvalue $(\pi/2)^2$);
* mass conservation for the sealed case is the companion statement
  $\mathbf 1^{\mathsf T}T[\texttt{bothsides}]=0$, which is (4) with $v\equiv1$.

A mismatched pair gives $-T[\varphi_1]T[\varphi_2]^{\mathsf T}$ with
$\varphi_1\ne\varphi_2$ — not symmetric, not sign-definite, complex eigenvalues
with positive real part available. The measured $1.4\times10^{6}$ is one of them.

---

## 8. What to remember

1. In DG, a boundary condition is a rule for **two numbers**, $\hat F_a$ and
   $\hat F_b$. Interior fluxes telescope exactly; nothing else can leak. (§2)
2. `bothsides` **deletes** the wall block, so zero flux is exact to machine
   precision at every level and degree — it is not an approximation. (§2.1)
3. A second-order term is a first-order *system*, so it has **two** independent
   wall traces: $\widehat q$ owned by `div`, $\widehat f$ owned by `grad`. They
   must be chosen as a complementary pair. (§3)
4. The flags must be **opposite** in the user-facing enum,
   $\varphi_g=\mathrm{flip}(\varphi_d)$ — which, since `grad` defaults to `none`,
   makes `{div bothsides, grad default}` the correct sealed pair. (§3.3, §4)
5. Sealing a set $S$ imposes **one** weak condition, $\sum_{k\in S}F_k=0$ at the
   wall — hence an emergent Robin $f'/f=-A/D$ when drag and diffusion are sealed
   together. (§5)
6. A Robin coefficient is a **flux per unit $f$** (a velocity), not a
   log-derivative; and free$(c)$ + Robin$(-c)$ is the *same matrix* as sealed. (§6)
7. The consistent pairings make the wall energy contribution vanish identically,
   $B_{\rm wall}=\widehat f\widehat q-(f^--\widehat f)(q^--\widehat q)=0$; the
   inconsistent ones leave $\pm(fq)^-$, of no fixed sign. That is the whole
   stability story. (§7)
