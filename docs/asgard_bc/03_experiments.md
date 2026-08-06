# Quantitative verification suite for the ASGarD boundary-condition claims

Every test below compares a **measured** number against an **independently known
analytic** value. All of them are re-runnable from the commands given.

## Summary

| # | test | predicted | measured | verdict |
|---|---|---|---|---|
| 1a | Neumann pair eigenvalue, 4×(L,k,D) | `D(kπ/L)²` | agrees | **PASS** 1e-9…1e-7 |
| 1b | Neumann pair mass conservation, levels 4–8 | exact | 9e-16…3e-15 | **PASS** (structural) |
| 1b | Neumann wall flux under refinement | → 0 | O(h³), 9.7e-8 → 2.4e-11 | **PASS** (converging) |
| 2 | Dirichlet pair decay rate, L=2 k=1 | (π/2)² = 2.467401 | 2.467401 | **PASS** 5e-9 |
| 2 | …second eigenvalue k=2, k=3 | π², (3π/2)² | agree | **PASS** 8e-8, 4e-7 |
| 2 | …second domain L=3, and D=0.4 | (π/3)², 0.4(π/2)² | agree | **PASS** 1e-9 |
| 2b | Dirichlet profile shape | `sin(πx/2)` | L2 dev 3.1e-07 at t=2 | **PASS** |
| 3a | sum rule wall slope, all 8 sealings, lvl 8 | `-A_S/D_S` | agrees | **PASS** ≤1.2e-5 |
| 3c | seal drag alone ⇒ **Dirichlet** `f(2)=0` | 0 | 1.7e-09 | **PASS** (new) |
| 3d | `dN/dt` = open-channel flux, all 8 | — | agrees | **PASS** ≤1.7e-4 |
| 3e | sum-rule slope under refinement, lvl 4–8 | → exact | O(h²), rate 2.00–2.02 | **PASS** (converging) |
| 4a | sum rule with **opposite-sign** drags | `f'/f = +0.66667` | +0.66667 | **PASS** 2.8e-7 (new) |
| 4a | sum rule with **exactly cancelling** drags | Neumann `f'=0` | 3.0e-13 | **PASS** (new) |
| 4b | sum rule with **spatially varying** D | only wall value enters | agrees | **PASS** ≤1.3e-5 (new) |
| 4c/d | **negative** individual diffusion coefficient | — | **blows up (interior)** | see §4d (new) |
| 5 | Robin(4) ≡ sealing the drag | identical | 2.1e-13 | **PASS** |
| 5 | analytic steady state `C e^{-x²}` | — | L2 rel 7.2e-08 | **PASS** |
| 6 | Dirichlet absolute mass vs Fourier series | 0.0410189977 | 0.0410189972 | **PASS** 1.3e-8 |
| 3b | seed: "-8 case measured -6.3, under-resolved" | — | **-7.99840 at the same resolution** | ⚠️ **SEED WRONG** |
| 6a | seed: ill-posed pair has "mass error 8e2" | — | **mass conserved to 4e-15 relative** | ⚠️ **SEED WRONG** |
| 6b | seed: bad-grad's other wall "exactly Neumann" | — | **15% off the clean Neumann** | ⚠️ **SEED TOO STRONG** |

Three seed claims do not reproduce as stated (§3b, §6a, §6b). None of them
overturns the mechanism or the sum rule — two are measurement artifacts in the
seed and one is an overstatement — but all three would mislead a reader
debugging a real solver, so they are called out in full.

## 0. How to run and how to measure

Toys (sources in the scratchpad, all `cmake --build build -j 4`):

| dir | equation | deck keys |
|---|---|---|
| `bc_toy` | `dt f = d/dx(2x f + f')` | `variant` 1/2/3 |
| `bc_matrix` | `dt f = d/dx(2x f + 1·f' + 0.5·f')` | `sealA`,`seal1`,`seal2` (1=bothsides, 0=none) |
| `bc_grad` | `dt f = d/dx(f')` | `gradflag` 0..5 |
| `bc_exact` (new) | `dt f = d/dx(D f')` on `[0,L]` | `pair` 0..3, `mode`, `xmax`, `diff` |
| `bc_sum` (new) | `dt f = d/dx(a1 f + a2 f + D1 f' + D2 f')` | `sA1,sA2,sD1,sD2`, `dvar`, `a2c`, `use2` |

```
SCRATCH=/private/tmp/claude-502/-Users-ahsan-Desktop-FokkerPlanck--claude-worktrees-trusting-lederberg-151603/a462d5b2-efc2-4bad-9cd7-4d2188feb27b/scratchpad
```

**GOTCHA (cost me a full round of bogus numbers).** A deck file is *only* read
when it is passed with `-if` / `-infile`. Writing

```
./build/bc_exact deck.txt -of out.h5      # WRONG - deck silently ignored
```

runs the solver with every deck key at its **default** value and no error
message. The correct form is

```
OMP_NUM_THREADS=2 KMP_BLOCKTIME=infinite \
  ./build/bc_exact -if deck.txt -of out.h5 -l 7 -time 1.0 -dt 1.e-4
```

Also note the documented precedence: values in the deck override CLI options
that appear *before* `-if`, so put CLI overrides *after* the `-if` argument.

**Measurement method.** All wall quantities are read off the **DG polynomial
itself**, not off a finite difference of a plot grid. For degree 2 the solution
is a quadratic on each cell, so sampling 10 interior points of the last cell and
fitting a quadratic recovers `f(wall)` and `f'(wall)` exactly (to round-off) for
the discrete solution. Masses and L2 norms use 8-point Gauss–Legendre per cell.
Helper: `$SCRATCH/an/an.py` (`an.wall`, `an.logslope`, `an.mass`, `an.moment`,
`an.l2`). This matters: the seed file's wall slopes were measured by finite
differences on a uniform 2001-point grid, which is why several of them looked
"under-resolved" (see §3).

---

## 1. Neumann pair `{div bothsides, grad default}` vs exact solutions

Toy `bc_exact`, `pair : 0`. Equation `dt f = D f''` on `[0,L]` with a
zero-flux (Neumann) wall. Two independent analytic facts are checked:
mass is conserved *exactly*, and `cos(k pi x/L)` is an eigenmode with
eigenvalue `D (k pi / L)^2`.

### 1a. Eigenvalue of the Neumann spectrum

Deck `pair : 0 / mode : -k / xmax : L / diff : D`; IC `cos(k pi x/L)`.
Level 7 (128 cells), degree 2, `dt = 1e-4`, CN. The decay rate is measured
from the projection `a(t) = ∫ f cos(k pi x/L) dx` at `t = 0.25/λ` and
`t = 1.25/λ` (exactly one e-fold apart):
`λ_meas = -ln(a2/a1)/(t2-t1)`.

| L | k | D | λ exact = D(kπ/L)² | λ measured | rel. err | shape L2 rel | mass drift |
|---|---|---|---|---|---|---|---|
| 2 | 1 | 1.0 | 2.467401 | 2.467401 | 5.1e-09 | 7.2e-08 | -4.0e-15 |
| 2 | 2 | 1.0 | 9.869604 | 9.869605 | 8.1e-08 | 5.8e-07 | 1.4e-15 |
| 3 | 1 | 1.0 | 1.096623 | 1.096623 | 1.0e-09 | 7.2e-08 | 1.5e-13 |
| 2 | 1 | 0.4 | 0.986960 | 0.986960 | 8.1e-10 | 7.2e-08 | 1.8e-14 |

"shape L2 rel" is `||f - a·cos(kπx/L)||₂ / ||a·cos(kπx/L)||₂`. The zero-mean IC
means the exact mass is 0 forever; the measured drift is at round-off.

**Verdict: PASS.** Four independent (L, k, D) combinations, eigenvalue correct
to 8–9 digits, profile to 7 digits, mass conserved to machine precision.

### 1b. Refinement: the wall flux converges to zero at O(h³)

Deck `pair : 0 / mode : 0 / xmax : 2.0 / diff : 1.0`, IC `1 + cos(pi x/2)`,
run to `T = 3`, `dt = 1e-4`. Exact solution
`f(x,t) = 1 + e^{-(π/2)² t} cos(π x/2)`; exact mass `= 2`; exact `f'(0) = f'(2) = 0`.

```
for L in 4 5 6 7 8; do
  OMP_NUM_THREADS=2 KMP_BLOCKTIME=infinite ./build/bc_exact -if neu.txt \
      -of neu_l$L.h5 -l $L -time 3.0 -dt 1.e-4
done
```

| level | h | mass error (exact 0) | L2 vs exact solution | \|f'(2)\| (exact 0) | \|f'(0)\| (exact 0) |
|---|---|---|---|---|---|
| 4 | 0.12500 | 8.9e-16 | 2.25e-08 | 9.66e-08 | 2.41e-07 |
| 5 | 0.06250 | 1.8e-15 | 2.82e-09 | 1.21e-08 | 3.02e-08 |
| 6 | 0.03125 | 1.8e-15 | 3.53e-10 | 1.51e-09 | 3.78e-09 |
| 7 | 0.01562 | 2.2e-15 | 4.96e-11 | 1.89e-10 | 4.72e-10 |
| 8 | 0.00781 | 2.7e-15 | 2.36e-11 | 2.42e-11 | 5.85e-11 |

Each halving of `h` divides the wall derivative by exactly 8 → **O(h³)**, the
optimal `degree+1` rate for degree 2. The L2 error follows the same rate until
it hits the CN time-error floor near 2e-11 at level 8.

**Verdict: PASS, and it is "weakly imposed and converging", not "wrong".**
The wall flux is never *identically* zero in the discrete solution (it is a
weak condition), but it converges to zero at the optimal rate while mass is
conserved to round-off at *every* level — i.e. the *global* conservation
statement is exact even when the *pointwise* wall derivative is not.

Note the left/right asymmetry: `|f'(0)|` is consistently 2.5× `|f'(2)|`. That
is the upwind alternation of the LDG chain, not an error — both converge at
the same rate.

---

## 2. Dirichlet pair `{div none, grad bothsides}` vs exact solutions

Toy `bc_exact`, `pair : 1`. Same measurement protocol, with
`a(t) = ∫ f sin(k pi x/L) dx` and IC `sin(k pi x/L)`.

First, a sanity cross-check that the new toy reproduces the old one: `bc_exact`
with `pair : 1, mode : 99, xmax : 2, diff : 1` at level 6, `t=1`, `dt=1e-3`
is **bit-identical** to `bc_grad` with `gradflag : 5` (mass `4.101895e-02`,
all sample values agree to every printed digit).

| L | k | D | λ exact = D(kπ/L)² | λ measured | rel. err | shape L2 rel |
|---|---|---|---|---|---|---|
| 2 | 1 | 1.0 | **2.467401** = (π/2)² | 2.467401 | 5.1e-09 | 1.1e-07 |
| 2 | 2 | 1.0 | 9.869604 = π² | 9.869605 | 8.1e-08 | 8.9e-07 |
| 3 | 1 | 1.0 | 1.096623 = (π/3)² | 1.096623 | 1.0e-09 | 1.1e-07 |
| 2 | 1 | 0.4 | 0.986960 | 0.986960 | 8.1e-10 | 1.1e-07 |
| 2 | 3 | 1.0 (level 8) | 22.206610 = (3π/2)² | 22.206619 | 4.1e-07 | 3.2e-07 |

This answers the "is (π/2)² a coincidence?" question three separate ways:
a **second eigenvalue** on the same domain (k=2 gives π², k=3 gives (3π/2)²,
both to 7 digits), a **different domain length** (L=3 gives (π/3)²), and a
**different diffusivity** (D=0.4 rescales λ exactly).

### 2b. Profile shape from a generic (non-eigenmode) start

Deck `pair : 1 / mode : 99` → IC `exp(-16 (x-0.7)²)`, which is not an
eigenmode. The solution must relax onto the ground mode `sin(π x/2)` and then
decay at exactly (π/2)².

| t | L2 shape deviation from sin(πx/2) | local decay rate | exact |
|---|---|---|---|
| 0.5 | 2.00e-02 | – | 2.467401 |
| 1.0 | 4.93e-04 | 2.467401 | 2.467401 |
| 1.5 | 1.22e-05 | 2.467401 | 2.467401 |
| 2.0 | 3.11e-07 | 2.467401 | 2.467401 |

The shape deviation falls by ~40× per 0.5 time unit, which is
`exp(-(λ2-λ1)·0.5) = exp(-7.402·0.5)⁻¹ ≈ 40` — i.e. it is the *physical*
decay of the second Dirichlet mode relative to the first, exactly as the exact
solution requires. Nothing here is a discretisation artifact.

**Verdict: PASS.** The seed file's claim "decay rate measured 2.467 vs exact
(π/2)² = 2.467" reproduces, and is much stronger than stated: 9 significant
digits, over 5 different (L, k, D) combinations, with the profile confirmed.

---

## 3. The SUM RULE, all 8 sealing combinations of `bc_matrix`

`dt f = d/dx(2x f + D1 f' + D2 f')`, `D1 = 1`, `D2 = 0.5`, on `[0,2]`.
Sealed set `S` ⊂ {A, D1, D2}. Sum-rule prediction at a wall `x_w`:

```
   ( Σ_{i∈S} a_i(x_w) ) f  +  ( Σ_{j∈S} D_j ) f'  =  0
```

which resolves into three regimes, all of which are tested:

| `D_S = Σ_S D_j` | `A_S = Σ_S a_i(2) = 4·[A sealed]` | condition at x=2 |
|---|---|---|
| > 0 | any | `f'/f = -A_S / D_S` (Robin) |
| = 0 | > 0 | `f(2) = 0` (**Dirichlet**) |
| = 0 | = 0 | none (fully open) |

Command (one of eight):

```
printf 'sealA : 1\nseal1 : 0\nseal2 : 1\n' > s101.txt
OMP_NUM_THREADS=2 KMP_BLOCKTIME=infinite ./build/bc_matrix -if s101.txt \
    -of s101.h5 -l 8 -time 1.0 -dt 1.e-4
```

### 3a. Wall log-slope — level 8 (256 cells), degree 2, t=1, dt=1e-4

| seal A,D1,D2 | A_S | D_S | predicted | measured | error |
|---|---|---|---|---|---|
| 1,1,1 | 4.0 | 1.50 | `f'/f = -2.66667` | **-2.66664** | 9.3e-06 |
| 1,1,0 | 4.0 | 1.00 | `f'/f = -4.00000` | **-3.99996** | 1.1e-05 |
| 1,0,1 | 4.0 | 0.50 | `f'/f = -8.00000` | **-7.99990** | 1.2e-05 |
| 1,0,0 | 4.0 | 0.00 | `f(2)  =  0` (Dirichlet) | **1.67e-09** | 3.4e-10 |
| 0,1,1 | 0.0 | 1.50 | `f'    =  0` (Neumann) | 1.4e-07 | 1.4e-07 |
| 0,1,0 | 0.0 | 1.00 | `f'    =  0` (Neumann) | 1.4e-07 | 1.4e-07 |
| 0,0,1 | 0.0 | 0.50 | `f'    =  0` (Neumann) | 1.4e-07 | 1.4e-07 |
| 0,0,0 | 0.0 | 0.00 | (no condition) | — | — |

**Verdict: PASS on all 8, to 5 significant figures.**

### 3b. ⚠️ CORRECTION TO THE SEED FILE

> The seed says: *"seal {A,D2} → f'/f = -A/D2 predicted -8 measured -6.3
> (under-resolved)"* and *"all sealed → predicted -2.67, measured -2.60"* and
> *"seal {A,D1} → predicted -4, measured -3.69"*.

**These deficits were a measurement artifact, not under-resolution.** Reading
`f'/f` off the DG polynomial of the last cell instead of finite-differencing a
uniform plot grid gives, *at the seed's own resolution* (level 6, 64 cells,
t=1, dt=1e-3):

| case | seed's number | this measurement, same run parameters | exact |
|---|---|---|---|
| all sealed | -2.60 | **-2.66627** | -2.66667 |
| seal {A,D1} | -3.69 | **-3.99931** | -4 |
| seal {A,D2} | **-6.3** | **-7.99840** | -8 |

The relative error at level 6 is 1.5e-4 – 2.0e-4 across all three, i.e. the
level-6 solution was *already* converged to 4 digits; the 21% shortfall in the
"-8" case came entirely from a one-sided finite difference over a 0.001-wide
plot interval against a solution whose wall slope is -8. Do not repeat that
claim; **there is no boundary layer that needs extra refinement here.**

### 3c. A new regime the seed missed: seal the drag *alone* ⇒ Dirichlet

Case `1,0,0` (drag sealed, both diffusions open) has `D_S = 0`, so the sum rule
degenerates to `a_A(2)·f(2) = 4 f(2) = 0` — a **Dirichlet** condition, not a
Robin one. Measured `f(2) = 1.7e-09` against an interior scale `f(0) ≈ 4.9`,
i.e. 3.4e-10 relative. This is a qualitatively distinct outcome of the same
rule and is worth stating in the document: *sealing a set with no `f'` channel
pins `f` at the wall rather than its slope.*

### 3d. The other half of the sum rule: dN/dt equals the OPEN-channel flux

Same eight runs. Prediction: the sealed channels contribute **exactly zero** to
the mass balance, so

```
   dN/dt  =  F_open(2) - F_open(0),   F_open = A_open·f + D_open·f'
```

with `A_open = 4·[A open]`, `D_open = Σ_{j∉S} D_j`, and `f, f'` taken from the
measured DG polynomial at each wall. `dN/dt` is measured by a **centred**
difference of the Gauss-quadrature mass, `(N(1.005) - N(0.995))/0.01`
(a forward difference over Δt=0.02 costs 2–7% and is what makes this look
noisy; it is a time-differencing error, not a physics error).

| seal A,D1,D2 | A_open | D_open | predicted dN/dt | measured dN/dt | rel err |
|---|---|---|---|---|---|
| 1,1,1 | 0.0 | 0.00 | 0.00000 | 0.00000 | 0 (exact) |
| 1,1,0 | 0.0 | 0.50 | -0.06891 | **-0.06891** | 1.1e-05 |
| 1,0,1 | 0.0 | 1.00 | -0.12133 | **-0.12133** | 1.2e-05 |
| 1,0,0 | 0.0 | 1.50 | -12.16792 | **-12.16871** | 6.5e-05 |
| 0,1,1 | 4.0 | 0.00 | 9.01295 | **9.01310** | 1.7e-05 |
| 0,1,0 | 4.0 | 0.50 | 9.01295 | **9.01310** | 1.7e-05 |
| 0,0,1 | 4.0 | 1.00 | 9.01294 | **9.01309** | 1.7e-05 |
| 0,0,0 | 4.0 | 1.50 | 4298.23853 | **4298.97126** | 1.7e-04 |

**Verdict: PASS on all 8, to 4–5 significant figures.** Note the "all sealed"
row is conserved to *exactly* 0 — mass conservation with everything sealed is
not approximate, it is structural.

Two seed claims confirmed sharply by this table:

* **"sealing one f'-channel silences both".** Rows `0,1,1`, `0,1,0`, `0,0,1`
  give **9.01295 / 9.01295 / 9.01294** — identical to 6 significant figures,
  even though `D_open` is 0, 0.5 and 1.0 respectively. The reason is visible in
  the mechanism: the sum rule forces `f'(2)=0` in all three, so the open
  diffusion channel carries `D_open · 0 = 0` regardless of its size, and the
  only leak is the un-sealed drag `4·f(2)`. The seed's "identical to 3 digits"
  is really identical to 6.
* **The fully-open case manufactures mass explosively**: `dN/dt = +4298` at
  t=1, versus `0` for the fully-sealed case. This is the toy analogue of the
  ICRF/LHCD "+63% particles" pathology.

### 3e. Refinement study: the sum-rule slope converges at O(h²)

`$SCRATCH/an/t4_conv.py`, one run per (case, level), `t=1.0`, `dt=1e-4`:

```
for L in 4 5 6 7 8; do
  OMP_NUM_THREADS=2 KMP_BLOCKTIME=infinite ./build/bc_matrix -if s101.txt \
      -of c101_l$L.h5 -l $L -time 1.0 -dt 1.e-4
done
```

| level | h | seal{A,D1,D2} (pred -2.66667) | rel err | seal{A,D1} (pred -4) | rel err | seal{A,D2} (pred -8) | rel err |
|---|---|---|---|---|---|---|---|
| 4 | 0.12500 | -2.660090 | 2.5e-03 | -3.988237 | 2.9e-03 | -7.969343 | 3.8e-03 |
| 5 | 0.06250 | -2.665049 | 6.1e-04 | -3.997193 | 7.0e-04 | -7.993195 | 8.5e-04 |
| 6 | 0.03125 | -2.666266 | 1.5e-04 | -3.999315 | 1.7e-04 | -7.998403 | 2.0e-04 |
| 7 | 0.01562 | -2.666567 | 3.7e-05 | -3.999831 | 4.2e-05 | -7.999613 | 4.8e-05 |
| 8 | 0.00781 | -2.666642 | 9.3e-06 | -3.999958 | 1.1e-05 | -7.999905 | 1.2e-05 |

Observed order in h: **2.00–2.02** in every column, over four halvings and a
factor 270 in error. The steepest condition (-8) converges at the same rate and
is only 1.5× less accurate than the shallowest (-2.67) at any given level.

**Verdict: PASS — "weakly imposed and converging", not "wrong".** This is the
crucial distinction. The wall condition is *never* satisfied pointwise by the
discrete solution at any finite resolution; it is satisfied in the limit, at a
clean and predictable rate. Compare with §1b where the *conservation* statement
was exact (1e-15) at every level: **conservation is structural, the pointwise
wall slope is asymptotic.** Any diagnostic that checks the wall slope must be
told what error to expect (≈ 2.5e-3 at level 4, 1e-5 at level 8).

---

## 4. NEW: does the sum rule survive mixed signs and spatially varying D?

Nobody had tested this. New toy `bc_sum`:

```
dt f = d/dx( a1 f + a2 f + D1 f' + D2 f' )   on [0,2]
a1 = 2x        a2 = a2c·x  (deck a2c)      D1 = 1      D2 from deck dvar
dvar: 0 -> 0.5        1 -> -0.3 (NEGATIVE)
      2 -> 0.25x  (varies, = 0.5 at the wall)
      3 -> 0.5+0.75x (varies, = 2.0 at the wall)
seal flags sA1 sA2 sD1 sD2 ; use2=0 removes a2 entirely
```

```
printf 'dvar : 3\nuse2 : 0\nsA1 : 1\nsD1 : 1\nsD2 : 1\n' > c3.txt
OMP_NUM_THREADS=2 KMP_BLOCKTIME=infinite ./build/bc_sum -if c3.txt \
    -of c3.h5 -l 8 -time 1.0 -dt 1.e-4
```

### 4a. MIXED SIGNS — the rule is a signed sum, confirmed

Level 8, t=1, dt=1e-4. `a1(2) = +4`, `a2(2) = 2·a2c = -1`.

| case | A_S(2) | D_S(2) | predicted f'/f | measured | rel err |
|---|---|---|---|---|---|
| control, a2 absent, all sealed | +4.00 | 1.50 | -2.66667 | -2.66664 | 9.3e-06 |
| a1 **and** a2 sealed (4 − 1 = 3) | +3.00 | 1.50 | **-2.00000** | **-1.99999** | 2.9e-06 |
| only a2 sealed, a1 open — **A_S < 0** | **-1.00** | 1.50 | **+0.66667** | **+0.66667** | 2.8e-07 |
| only a1 sealed, a2 open | +4.00 | 1.50 | -2.66667 | -2.66666 | 3.4e-06 |
| `a2c = -2.0`, both sealed — **exact cancellation** | **0.00** | 1.50 | 0 (Neumann) | 3.0e-13 | 2.0e-13 |

Leak check on the two partial-sealing rows (centred `dN/dt`):

| case | predicted dN/dt | measured dN/dt | rel err |
|---|---|---|---|
| only a2 sealed (a1 open, A_open = +4) | 70.56716 | **70.57151** | 6.2e-05 |
| only a1 sealed (a2 open, A_open = -1) | -0.06989 | **-0.06989** | 9.6e-07 |

**Verdict: PASS, and this is the sharpest available discriminator.** The row
with `A_S = -1` predicts a wall slope of the *opposite sign* (`f' > 0`,
f rising into the wall) and it is measured to 3e-07. The row with
`a2c = -2.0` predicts *exact cancellation* of two nonzero sealed drags into a
pure Neumann wall, and delivers `f' = 3e-13`. No "first term wins", "largest
term wins", or "sum of magnitudes" rule can produce either number. The sum
rule is a genuine **signed algebraic sum of the sealed terms' wall fluxes**.

### 4b. SPATIALLY VARYING diffusion — only the WALL value enters

| case (a2 absent, all sealed) | D_S at x=2 | predicted f'/f | measured | rel err |
|---|---|---|---|---|
| `D2 = 0.5` (constant) | 1.50 | -2.66667 | -2.66664 | 9.3e-06 |
| `D2 = 0.25x` (varies 0 → 0.5) | 1.50 | -2.66667 | **-2.66663** | 1.3e-05 |
| `D2 = 0.5 + 0.75x` (varies 0.5 → 2.0) | 3.00 | **-1.33333** | **-1.33333** | 2.0e-06 |

**Verdict: PASS.** Row 2 is the key one: `D2(x) = 0.25x` is a *completely
different function* from the constant `0.5` throughout the interior — it
changes the whole interior solution — yet it gives the *same* wall slope to 5
digits, because it takes the same value at the wall. The condition is
strictly **local to the wall**: `Σ_S a_i(x_w) f + Σ_S D_j(x_w) f' = 0`,
evaluated pointwise at `x_w`. Row 3 (D_S doubled at the wall) halves the slope
exactly as predicted.

### 4c. ⚠️ A NEGATIVE individual diffusion coefficient BLOWS UP

`dvar : 1` sets `D2 = -0.3`, so the *net* diffusion `D1 + D2 = 0.7` is still
positive and the PDE is well posed. Both runs (`all sealed` and
`seal a1+D1 only`) **diverged**:

```
ERROR: found 'inf' or 'nan' entries in the next time-step
```

giving nonsense (`f'/f` measured `+158.6` and `+273.4` against predictions
`-5.71429` and `-4`, and `dN/dt = -inf`). See §4d for the control that
isolates the cause.

### 4d. Control: the blow-up is the SIGN of the individual term, not the BC

Same deck (`use2 : 0`, everything sealed), level 6, `t = 0.2`, `dt = 1e-4`,
varying only `D2`:

| D2 | net D1+D2 | predicted f'/f | measured f'/f | rel err | mass at t=0.2 | max\|f\| |
|---|---|---|---|---|---|---|
| **+0.30** | 1.30 | -3.076923 | **-3.076568** | 1.2e-04 | 8.82e-01 | 6.4e-01 |
| **-0.05** | 0.95 | -4.210526 | -0.970942 | 7.7e-01 | 2.14e+13 | 2.0e+13 |
| **-0.30** | 0.70 | -5.714286 | +39.87 | 8.0e+00 | 5.87e+75 | 4.3e+76 |

The `+0.3` row has the same magnitude as the `-0.3` row and behaves perfectly
(1.2e-4 at level 6, exactly the O(h²) error of §3e). Even a *tiny* negative
piece, `D2 = -0.05` against `D1 = +1`, blows up by 13 orders of magnitude in
t = 0.2. And the growth is **interior**, not at the wall:

| t | max\|f\| | located at x | mass |
|---|---|---|---|
| 0.02 | 2.09e+01 | 1.095 | 1.97e+01 |
| 0.05 | 1.98e+03 | 1.095 | 2.05e+03 |
| 0.10 | 4.18e+06 | 1.220 | 4.59e+06 |

**Verdict: the sum rule is not falsified here — it is untestable, because the
discretisation is unstable before the boundary can matter.** The finding is
still a real and useful one for the document:

> **You cannot split a diffusion operator into a positive and a negative
> `term_div/term_grad` chain, even when the net diffusivity is positive.**
> Each chain gets its own upwind alternation (`c ± |c|`) built from *its own*
> coefficient; a negative coefficient picks the anti-diffusive pairing and the
> term is unconditionally unstable in the interior. Terms must be arranged so
> that **every individual** `f'` channel has a non-negative coefficient. The
> sum rule then describes the boundary behaviour of that legal set.

This is worth stating explicitly because the sum rule otherwise *invites* the
mistake: §4a shows drags cancel across terms with opposite signs perfectly
well, so a reader could reasonably expect diffusions to do the same. They do
not.

---

## 5. `bc_toy`: Robin ≡ sealing, and the full analytic steady state

`dt f = d/dx(2x f + f')` on `[0,2]`. This one has a closed-form **steady
state**: zero flux everywhere ⇒ `2x f + f' = 0` ⇒ `f = C e^{-x²}`, and mass is
conserved so `C` is fixed by the IC. IC `exp(-4(x-1)²)`, whose exact mass is
`(√π/4)(erf(5.2) + erf(2.8)) = 0.8820813908`.

Level 7 (128 cells), `t = 4.0`, `dt = 1e-4`. Script `$SCRATCH/an/t6_robin.py`.

| test | predicted | measured | verdict |
|---|---|---|---|
| variant 1 (seal drag) vs variant 2 (drag free + `set_left_robin(0)`, `set_right_robin(4)`) | identical matrices ⇒ identical solutions | max abs difference over 4001 points **2.1e-13** (relative) | PASS |
| variant 1 profile vs analytic `C e^{-x²}` | 0 | L2 relative **7.2e-08** | PASS |
| variant 1 mass at t=4 vs exact IC mass | 0.8820813908 | **0.8820813908** (all 10 digits) | PASS |
| variant 1 wall condition `4f + f'` | 0 | 9.1e-06, i.e. `f'/f = -3.999504` vs exact -4 | PASS (1.2e-04, O(h²)) |
| variant 3 (drag free, **no** Robin) mass at t=4 | should be 0.88208 | **1333.97** — grew by a factor **1512** | FAIL, as designed |
| variant 3 wall state | — | `f(2) = 666.96`, `f'(2) = 7.4e-07`, so `4f + f' = 2668` | — |

**Verdict: seed claim 2 reproduces.** Robin with `r` = the drag's wall flux
coefficient is numerically indistinguishable from sealing the drag term
(2e-13 ≈ accumulated round-off over 40 000 Crank–Nicolson steps).

Variant 3 is the diagnostic picture of the production bug: the chain is sealed
so `f'(2) ≈ 0` (7e-07, a clean Neumann), but the drag is open, so it pumps
`4·f(2) = 2668` units of mass per unit time *into* the domain through the right
wall. The wall condition is not "wrong" — each term is doing exactly what its
own flag says. The failure is that the *set* of flags does not make the sum
vanish. That is the whole content of the sum rule, seen from the failure side.

---

## 6. All six chain pairings, refined — which pathologies are real

`bc_grad`, `dt f = d/dx(f')` on `[0,2]`, IC `exp(-16(x-0.7)²)`, `t = 1`,
`dt = 1e-4`, levels 5/6/7. Exact IC mass
`= ½√(π/16)·(erf(5.2)+erf(2.8)) = 0.443096843048`.

```
printf 'gradflag : 1\n' > g1.txt
OMP_NUM_THREADS=2 KMP_BLOCKTIME=infinite ./build/bc_grad -if g1.txt \
    -of g1_l7.h5 -l 7 -time 1.0 -dt 1.e-4
```

| pairing | lvl | mass at t=1 | \|mass − 0.4430968430\| | f(0) | f(2) |
|---|---|---|---|---|---|
| **div both, grad default** (NEUMANN) | 5 | 0.4430968430 | **4.4e-16** | 0.237950 | 0.205123 |
| | 6 | 0.4430968430 | **4.4e-16** | 0.237950 | 0.205123 |
| | 7 | 0.4430968430 | **5.0e-16** | 0.237950 | 0.205123 |
| **div both, grad both** (ill-posed) | 5 | 0.4430968430 | 2.9e-11 | 7.83e+02 | 2.30e+04 |
| | 6 | 0.4430968425 | 5.1e-10 | 3.09e+03 | 1.79e+05 |
| | 7 | 0.4430968368 | 6.3e-09 | 1.23e+04 | **1.42e+06** |
| **div both, grad right** | 5 | 0.4430968430 | 1.4e-11 | 0.200238 | 2.28e+04 |
| | 6 | 0.4430968427 | 3.9e-10 | 0.201031 | 1.79e+05 |
| | 7 | 0.4430968377 | 5.4e-09 | 0.201417 | **1.42e+06** |
| **div both, grad left** | 5 | 0.4430968430 | 1.5e-11 | 8.14e+02 | 0.207965 |
| | 6 | 0.4430968429 | 1.2e-10 | 3.21e+03 | 0.206527 |
| | 7 | 0.4430968421 | 9.3e-10 | **1.28e+04** | 0.205821 |
| **div none, grad default** (mongrel) | 5 | 0.2410542550 | 2.0e-01 | 0.183654 | -3.93e-03 |
| | 6 | 0.2457131709 | 2.0e-01 | 0.188406 | -1.96e-03 |
| | 7 | 0.2476339900 | 2.0e-01 | 0.190281 | -9.78e-04 |
| **div none, grad both** (DIRICHLET) | 5 | 0.0410189972 | (see below) | -5.10e-07 | 1.01e-06 |
| | 6 | 0.0410189972 | | -6.37e-08 | 1.26e-07 |
| | 7 | 0.0410189972 | | -7.97e-09 | 1.58e-08 |

Independent check on the Dirichlet row: summing the exact Fourier series
`Σ_k b_k e^{-(kπ/2)²t} ∫₀² sin(kπx/2)dx` (k up to 400) for this IC gives
**0.0410189977** at t=1; the solver gives **0.0410189972** at *every* level —
agreement to **1.3e-08**. So the Dirichlet pair reproduces not just the decay
*rate* (§2) but the absolute decayed mass.

### 6a. ⚠️ SECOND CORRECTION TO THE SEED FILE

> The seed says: *"{div bothsides, grad bothsides} → ILL-POSED ... f → 1.4e6,
> **mass error 8e2**"*.

The `f → 1.4e6` reproduces exactly (level 7, t=1). **The mass error does
not.** Mass is conserved to 6.3e-09 — and that 6.3e-09 is round-off relative
to a field of magnitude 1.4e+06, i.e. 4e-15 in relative terms. Mass
conservation with `div bothsides` is *structural*: the div term's boundary
block is simply absent, so there is no mechanism by which mass can leave,
whatever the grad flag does. The pathology of this pairing is **pointwise
divergence, not mass loss**, and it grows as `h^-3` under refinement
(2.3e4 → 1.8e5 → 1.4e6 per level halving, factor ≈ 7.9 ≈ 8).

Practical consequence, which matters for how people debug these solvers:
**a mass-conservation diagnostic will not catch this bug.** You need a
magnitude or `max|f|` check as well.

### 6b. ⚠️ THIRD CORRECTION: "the other wall stays exactly Neumann" is too strong

> The seed says: *"{div bothsides, grad right/left} → ill-posed on THAT WALL
> ONLY; the other wall stays exactly Neumann (0.2228 vs const 0.2215). Local
> corruption."*

Half of this reproduces: with `grad right`, `f(0)` stays **bounded and
converging** (0.200238 → 0.201031 → 0.201417, first differences halving, so
O(h) toward ≈ 0.2018) while `f(2)` diverges as h⁻³. But the good wall does
**not** agree with the clean Neumann run: the clean Neumann gives
`f(0) = 0.237950` (level-independent to 6 digits), against **0.2014**. That is
a **15% discrepancy**, not "exactly Neumann".

The correct statement is: *the divergence is confined to the bad wall, but the
solution everywhere — including the good wall — is wrong, because the huge
boundary spike redistributes the (conserved) mass.* Calling it "local
corruption" understates it; only the *blow-up* is local.

### 6c. The mongrel confirmed: the two walls disagree despite symmetric flags

`div none, grad default` gives symmetric flags on both walls, yet:
`f(0)` converges to ≈ 0.1922 (a finite, Neumann-like value) while `f(2)`
converges to **0** at O(h) (-3.93e-03 → -1.96e-03 → -9.78e-04, exact halving),
i.e. Dirichlet-like — **and slightly negative**, so this pairing produces
negative densities. Mass is lost (0.443 → ≈0.248) but the mass *deficit*
itself is only weakly level-dependent. This is the upwind alternation acting
asymmetrically, exactly as the seed described. Confirmed, including the
negative values.

---

## 7. Reproducing the whole suite

```
SCRATCH=/private/tmp/claude-502/-Users-ahsan-Desktop-FokkerPlanck--claude-worktrees-trusting-lederberg-151603/a462d5b2-efc2-4bad-9cd7-4d2188feb27b/scratchpad
PY=/Users/ahsan/venvs/asgardpy/bin/python

# build (bc_exact and bc_sum are new; the other three already exist)
for d in bc_toy bc_matrix bc_grad bc_exact bc_sum; do
  cmake -S $SCRATCH/$d -B $SCRATCH/$d/build -DCMAKE_BUILD_TYPE=Release \
        -Dasgard_DIR=/Users/ahsan/venvs/asgardpy/lib/cmake/asgard
  cmake --build $SCRATCH/$d/build -j 4
done

$PY $SCRATCH/an/t1_dirichlet.py     # §2   Dirichlet vs exact eigenmodes
$PY $SCRATCH/an/t2_neumann.py       # §1   Neumann vs exact + refinement
$PY $SCRATCH/an/t3_summ.py 8 1.0 1.e-4   # §3a,3c,3d  sum rule, all 8
$PY $SCRATCH/an/t3_summ.py 6 1.0 1.e-3   # §3b  seed's own resolution
$PY $SCRATCH/an/t4_conv.py          # §3e  sum-rule refinement
$PY $SCRATCH/an/t5_stress.py        # §4   mixed sign / varying D
$PY $SCRATCH/an/t6_robin.py         # §5   Robin equivalence + steady state
$PY $SCRATCH/an/t7_bad.py           # §6   all six pairings refined
```

Shared measurement helper: `$SCRATCH/an/an.py`. Total runtime ≈ 25 min.
Every solver invocation inside these scripts sets
`OMP_NUM_THREADS=2 KMP_BLOCKTIME=infinite` (macOS libomp segfaults otherwise).

New toy sources: `$SCRATCH/bc_exact/main.cpp`, `$SCRATCH/bc_sum/main.cpp`.

### Recommended regression checks to keep

If only three numbers are kept as a permanent regression test, keep these —
they are cheap, exact, and each catches a different class of failure:

1. **`bc_exact pair:0 mode:0`, mass at t=3 vs 2.0** → must be < 1e-14 at *any*
   level. Catches any change that reintroduces a boundary block on a sealed
   div. (Structural; no tolerance tuning needed.)
2. **`bc_exact pair:1 mode:1`, decay rate vs (π/2)²** → must be < 1e-6 at
   level 7. Catches a broken LDG flag flip on `operation_type::grad`.
3. **`bc_matrix sealA:1 seal1:0 seal2:1`, `f'(2)/f(2)` vs -8** → must be
   < 5e-5 at level 8 (it is 1.2e-5). Catches any change to how the sealed set
   is summed, and is the most sensitive of the three.

Add a `max|f| < 10` assertion alongside (1): per §6a, mass conservation alone
does not detect the ill-posed `{div bothsides, grad bothsides}` pairing.
