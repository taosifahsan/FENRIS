# Novice reader review — `asgard_boundary_conditions.tex`

**Reviewer profile.** Plasma physicist. Comfortable with PDEs, weak forms in the
abstract, Fokker–Planck. Has never used DG in anger, has never opened ASGarD,
and did not consult the source, the session notes, or any other file while
writing this. Everything below is what the document alone supports.

**Verdict up front.** The document is genuinely good at explaining *what the
boundary flags mean*. It is not yet usable for the design task it promises,
because it never states the one thing you need before you can write a single
line of code: **the sign and the split of the coefficients you hand to
`term_div` and `term_grad`.** It uses two opposite sign conventions for the
letter `F`, in the same document, both labelled "the flux", and it contradicts
itself outright on the sign of a real production coefficient. I got the *flags*
right with reasonable confidence. I could not get the *coefficients* right at
all, and I would not have caught this by pattern-matching, because the two
example snippets in the document disagree with each other.

---

## Part 1 — The two exercises

### Setup common to both

The PDE I am given is

```
∂_t f = ∂_v ( A(v) f + D(v) ∂_v f )     on [0, v_max],  D > 0
```

**Step 0: get into the document's convention.** §1 ("The one sign convention")
and eq. (1) say that terms are applied with α = −1, that `term_div(c)`
assembles the weak form of +∂_x(cf), and that the realised equation is therefore

> ∂_t f = −∂_x F,  F = c f,  dN/dt = F(a) − F(b)

§8.1 step 1 repeats this: "write the flux F_k that appears in
∂_t(mf) = −∂_x Σ_k F_k".

So to match my equation I must write it as ∂_t f = −∂_v Σ F_k with

```
Σ F_k = −( A f + D ∂_v f )
     ⇒ F_drag = −A(v) f       ⇒ c_drag = −A(v)
     ⇒ F_diff = −D(v) ∂_v f
```

**Step 0b: split the diffusion across the chain.** §4.2 says a chain
`{div, grad}` is the matrix product A = A_div A_grad and the ODE contribution is
−A_div A_grad f. §3.4 shows term_grad(c) acts as +c ∂_x. So the chain realises

```
∂_t f = −∂_v ( c_div · c_grad · ∂_v f )
```

and to get physical diffusion +∂_v(D ∂_v f) I need **c_div · c_grad = −D**.

I want to flag immediately that **the document never states this identity
anywhere.** I had to assemble it from three separate sections. And it directly
contradicts eq. (14), the LDG system the document itself sets up:

> q = D ∂_x f  (the grad equation),  ∂_t f = ∂_x q  (the div equation)

Read literally, eq. (14) says c_div = +1, c_grad = +D. That gives
∂_t f = −∂_v(D ∂_v f) — **anti-diffusion**. So either eq. (14) is written in a
convention the document abandoned two sections earlier, or my derivation is
wrong. The only evidence that settles it is a *variable name* in a code snippet
in §7.1: `neg_I_sinth = -I(th)*sin(th)` on the div of the pitch-angle diffusion
chain. That negative sign supports my reading. **A variable name in an example
is not documentation.**

I proceed with c_div · c_grad = −D and mark it as my single largest guess.

---

### Exercise 1 — perfectly reflecting at both ends

**Reasoning, following §8.1 verbatim:**

1. *Write the physical fluxes.* Two flux-carrying terms:
   F_drag = −A f and F_diff = −D f'. No mass weight in this problem
   (no Jacobian was given), so m = 1.
2. *Decide the physical wall law.* "Perfectly reflecting" = no particles cross =
   **total flux zero** at both walls. §8.2 row 1 confirms this is the
   "Reflecting / zero flux (impermeable wall)" case.
3. *Choose the sealed set.* S = **all** flux-carrying terms, at both walls.
   Flag each with `bothsides` on its outermost `div`.
4. *Fix each chain's parity.* The diffusion chain's `div` carries `bothsides`,
   so its `grad` must carry `none` (which §4.3(i) says is `term_grad`'s default).
5. *Check diffusion coefficient signs.* — see below, I could not do this step.
6. *Run the audit* (§8.3) — see below.

**My answer:**

```cpp
// drag: first-order, sealed at both walls
pde += term_1d({ term_div(c_A, boundary_type::bothsides) });        // c_A = -A(v)

// diffusion: two-link chain, one condition, carried by the div
pde += term_1d({ term_div(c_d, boundary_type::bothsides),           // c_d * c_g = -D(v)
                 term_grad(c_g, boundary_type::none) });            // none is the default
```

**What this realises**, per §5.1 and §5.3: sealing the set imposes **one** weak
condition per wall, Σ_{k∈S} F_k = 0, i.e.

```
A f + D f' = 0   ⇒   f'/f = −A/D   at v = 0 and at v = v_max
```

an *emergent Robin*, not f' = 0. §5.3 says explicitly this is the correct
physics of an impermeable wall in the presence of drag, and that imposing f' = 0
by hand would be wrong. Good — this is the part of the document that taught me
something I did not already know.

Note the wall law is sign-convention-invariant: if I got the overall sign of
c_drag and c_diff both backwards, `(+A)f + (+D)f' = 0` is the same equation. So
my *flag* answer survives my confusion about signs; my *coefficient* answer
does not.

**Confidence.**
- On the flags: **high, ~85%.**
- On the coefficient signs and the chain split: **low, ~40%.**
- On being able to type this into a compiler and have it build: **~20%**, because
  I do not know the argument order or the defaults (see guess (d)).

**Every point where I had to guess:**

| # | Guess | Why the document did not settle it |
|---|---|---|
| a | c_drag = −A(v), not +A(v) | §1/§2.2 say ∂_t f = −∂_x(cf); §2.3 says "the FENRIS sign convention, ∂_t f = ∂_x F"; eq. (14) says ∂_t f = ∂_x q. Three statements, two conventions, same letter F. |
| b | c_div · c_grad = −D | Never stated. Contradicted by eq. (14). Supported only by the variable name `neg_I_sinth`. |
| c | *How* to split −D between the two links (c_d = −1, c_g = D? c_d = −D, c_g = 1? symmetric?) | Never discussed. The only example (ICRF_1D) puts the Jacobian v² on **both** links, which suggests the split is not free — but it is never said what governs it. |
| d | The `flux_type` argument | The document never names the default flux for `term_div` or `term_grad`, and never tells me which flux I should write. §4.2 warns that if the two chain factors are not "bidiagonal in opposite directions" the product is **silently wrong**, and that `check_chain` does not verify this. So I am told a silent-wrong-answer trap exists and given no instruction for avoiding it. The two example snippets differ: ICRF_1D omits the flux, ICRF_2D writes `flux::upwind`. |
| e | That sealing **both** ends of a *first-order* term (the drag) is legitimate | A physicist's immediate objection: a first-order advection operator admits exactly one boundary condition, at the inflow. Sealing both ends of it looks over-determined. The document's rule "each factor can carry at most one wall condition; a second-order operator needs exactly one" is stated only for chains and never addressed for a standalone div. I eventually reassured myself from §5.3 (the composite is what carries the condition, and a sealed set with D_S = 0 degenerates to Dirichlet rather than blowing up) — but that took a cross-reference the document does not make. |
| f | Whether the flags at v = 0 do anything at all | §3.3/P5 say a coefficient vanishing at the endpoint makes the flag inert. My A and D are unspecified at v = 0. If this were a real spherical FP problem with a v² Jacobian, §7.1 says the v = 0 flag is a no-op. The document is good on this point; I flag it only because the exercise as posed cannot be answered without knowing A(0), D(0), and the document does not tell me to ask. |

---

### Exercise 2 — absorbing (f = 0) at v_max, reflecting at v = 0

**Reasoning:**

The two walls are algebraically independent — §4.5 says the bracket in eq. (7)
is a sum of two separate endpoint terms, and §4.4 says B_a and B_b are separate
terms in the energy budget. So I can design each wall on its own.

*Left wall (v = 0), reflecting.* Same as Exercise 1: seal every flux-carrying
term at the left only. §3.2 tells me the flag names the wall where the flux **is**
pinned, so `boundary_type::left` seals the left and leaves the right free.

*Right wall (v_max), Dirichlet f = 0.* §4.3(ii) and §8.2 row 2: the pair is
`{div none, grad bothsides}` — the div free (so flux leaves, as it must at an
absorbing wall), the grad pinning the trace of f to zero.

*Combining them on one chain.* The div must be sealed-left / free-right →
`boundary_type::left`. The grad must impose nothing at the left and Dirichlet at
the right → user-facing `boundary_type::right`.

*Cross-check with the parity mnemonic* (§4.6, eq. 27): a legal chain has
φ_grad = flip(φ_div). §3.4's flip maps `left ↔ right`. φ_div = `left`
⇒ φ_grad = `right`. ✔ The two rules agree, which raised my confidence
substantially.

*Cross-check with the audit* (§8.3 step 3): count `bothsides`-equivalent
conditions per link **per wall**.
- Left wall: div sealed ✔ (one), grad free ✔ (zero). Total = 1. ✔
- Right wall: div free ✔ (zero), grad Dirichlet ✔ (one). Total = 1. ✔

**My answer:**

```cpp
// drag: sealed at the left, free at the right
pde += term_1d({ term_div(c_A, boundary_type::left) });             // c_A = -A(v)

// diffusion: sealed-left div, Dirichlet-right grad
pde += term_1d({ term_div(c_d, boundary_type::left),
                 term_grad(c_g, boundary_type::right) });
```

**What this realises:** at v = 0, Σ_{k∈S} F_k = 0 ⇒ f'/f = −A/D (reflecting,
emergent Robin, no leak). At v_max, f = 0 with the flux free ⇒ particles are
absorbed and mass leaves at whatever rate the solution generates.

**Confidence: medium, ~65–70%.** Lower than Exercise 1, for two reasons that are
findings in their own right:

**Doubt 1 — which wall does `grad right` actually pin?** The document never says
this in prose. §3.4's table says `left`/`right` mean "same, that wall only",
which reads as *the wall named in the flag*. But §4.5 says:

> "The flip maps `right` → `left`, so `{div bothsides, grad right}` writes the
> grad's Dirichlet block at one wall only"

— and then does not say **which** wall. Read literally, "the flip maps right →
left" invites the conclusion that the block lands at the **left**. I could only
resolve this by going to the numerical table in §11.6, seeing that the row
`div both, grad right` blows up at f(2) = 1.42e+06 while f(0) stays at 0.2014,
and reasoning backwards: the over-determined (bad) wall must be the one where
both conditions land, so `grad right`'s Dirichlet block is at the **right**.
**The single sentence I most needed for this exercise had to be reverse-engineered
from the position of a blow-up in a verification table.**

**Doubt 2 — the recipe fails its own audit, and I cannot tell whether that
matters.** §8.2's Dirichlet row instructs: "advective terms left `none`". So my
drag is free at v_max. But §8.3 audit step 5 says:

> "For every **unsealed** advective term, does the characteristic leave the
> domain at that wall? If it points inward, the free flux is a positive feedback
> on the wall value (§3, §6) and will manufacture density."

In a Fokker–Planck problem the drag at v_max points **inward** (particles slow
down). That is the generic case, not an edge case. So the document's recommended
Dirichlet recipe fails the document's own audit step 5, every time, for exactly
the class of equation the document is about.

I *suspect* it is harmless — if the grad pins f(v_max) = 0, then the free drag
flux c·f^int(wall) → 0 with it, so the "positive feedback on the wall value" has
nothing to feed on. But the document never says this. P9 states the pathology
unconditionally ("It does *not* improve with refinement"), with no exception
carved out for the case where another term is simultaneously pinning the wall
value to zero. **I do not know whether my Exercise 2 answer is correct or is the
document's headline failure mode.** That is the sharpest single gap I found.

**Doubt 3 — no measurement of this configuration exists.** §11.6 tests six
pairings. Every one of them uses `bothsides` on the div. The mixed one-sided
configuration that Exercise 2 requires — different physics at the two walls,
which is surely the commonest real design task — is **never demonstrated,
never measured, and never shown as code anywhere in the document.** The
one-sided cases that *are* tested (`div both, grad right`, `div both, grad left`)
are the *illegal* ones, presented as failures. A reader pattern-matching off the
tables would conclude that one-sided grad flags are dangerous, which is the
opposite of what §4.5 intends.

**Points where I had to guess in Exercise 2:**

| # | Guess |
|---|---|
| g | `grad right` pins the trace at the **right** wall (not the left, as "the flip maps right → left" suggests). Resolved only via the §11.6 table. |
| h | A free (unsealed) drag at a Dirichlet wall does not trigger P9, because f is pinned to 0 there. Never stated; contradicts audit step 5 as written. |
| i | No penalty term is needed for the Dirichlet wall. Inferred from §11.2 showing `{div none, grad bothsides}` reproduces (π/2)² to 5e−09 without one, not from any statement. §3.6 discusses penalties at length without ever saying "you do not normally need one". |
| j | That "outermost div" (§8.1 step 3) means the `div`, i.e. the **first** element written in the chain list. §4.2 says the stored operator is A_div A_grad, so the div is the left/outer factor. But §3.7's footnote says `chain_level` "defaults to −1, i.e. the outermost link" — and −1 in C++ idiom means the **last** element, which is the `grad`. These two uses of "outermost" point at opposite links. |

---

## Part 2 — Stumbling points, ordered by how badly they blocked me

### 1. Two opposite sign conventions for the letter *F*, both called "the flux" — **hard blocker**

§2.2, eq. (1):
> "∂_t f = −∂_x F,  F = c f  ⟹  dN/dt = F(a) − F(b)"

§2.3, twenty lines later:
> "Take the scalar conservation law **in the FENRIS sign convention**,
> ∂_t f = ∂_x F with F the flux in the direction of increasing x."

§4.1, eq. (14):
> "q = D ∂_x f (the grad equation), ∂_t f = ∂_x q (the div equation)"

These cannot all describe the realised equation. I *think* §2.3 and §4.1 are
describing the pre-α matrix (what A represents) while §2.2 and §8.1 describe the
post-α realised ODE — but §2.3 calls its version "the FENRIS sign convention",
which is exactly the phrase that would make a reader adopt it as *the*
convention. This is the first decision anyone doing the design task has to make
and it is the one the document handles worst.

**What would unblock me:** one sentence, in §2.2, in bold: *"Throughout this
document, F always denotes the flux in the realised equation ∂_t f = −∂_x F.
The weak forms in §2.3 and §4.1 are written for the assembled matrix A, before
the α = −1 is applied; their ∂_x has the opposite sign. If you pass c to
term_div, the physical flux is −cf."* Then fix eq. (14) or annotate it.

### 2. The document contradicts itself on the sign of a real production coefficient — **hard blocker**

§6.1:
> "In ICRF_1D the drag term is `term_div(c)` with **c(v) = −v²η(v)**"

§7, code listing, same term, same file:
> `pde += term_1d({term_div(eta_v2, boundary_type::bothsides)});      // c = eta * v^2`

`+ηv²` and `−ηv²`. One of these is wrong and I have no way to tell which. This
is the exact quantity Exercise 1 asks me to produce. The §6.1 version is
internally corroborated (`set_right_robin(v_max²·η(v_max)) = −c(v_max)` only
works if c = −ηv²), so I lean that way — but then the code comment in §7 is
false, and §7 is the section that is supposed to teach me what coefficient to
pass.

**What would unblock me:** reconcile them, and add the definition of `eta_v2`.

### 3. The chain product's coefficient identity is never stated — **hard blocker**

Nowhere does the document say what
`term_1d({term_div(c_d), term_grad(c_g)})` realises as a PDE. I had to
reconstruct `∂_t f = −∂_x(c_d c_g ∂_x f)` from §3.4 (grad ≈ +c∂_x), §4.2
(A = A_div A_grad), and §1 (α = −1). The one place that looks like it answers the
question, eq. (14), gives the opposite sign.

**What would unblock me:** a single displayed equation early in §4:
*"chain {div(c_d), grad(c_g)} under mass m realises
 m ∂_t f = −∂_x( c_d · (1/m) · c_g ∂_x f ), so for a physical diffusivity D you
 pass c_d c_g = −mD."* One line. It is the most important line the document
does not contain.

### 4. P6 ("check the sign of every diffusion coefficient") is unusable because I don't know which coefficient it means — **hard blocker, and it is step 5 of the official recipe**

> "Every individual f' channel must have a **non-negative** coefficient"
> "Measured: with D_1 = +1 fixed, D_2 = +0.3 behaves perfectly; D_2 = −0.05 …
> blows up by 13 orders of magnitude"

Under my reconstruction the product c_d·c_g of a *correct* diffusion chain is
**−D**, i.e. negative. So is P6 telling me my correct chain is unstable? P6 must
mean the *physical* diffusivity D, not the coefficient I type — but it says
"coefficient", and the whole document is about what you type. Worse, the
verification table for P6 lists `D_2` values without ever showing the code line
that produced them, so I cannot calibrate.

**What would unblock me:** state P6 in terms of a named, unambiguous quantity —
"the physical diffusivity D_k = −c_div c_grad / m must be ≥ 0 for every chain
separately" — and show the code line for one of the `D_2 = −0.05` runs.

### 5. `flux_type` is never specified, and the document says getting it wrong fails silently — **blocker for writing code**

§4.2:
> "ASGarD relies on the two factors being numerically *bidiagonal in opposite
> directions* … `check_chain` … does **not** verify that the two side fluxes
> point opposite ways, and it cannot know the sign of a variable coefficient.
> [INFERENCE]: if the pairing is wrong … the out-of-band entries are silently
> dropped and you get a consistent-looking but wrong operator, with no warning."

I am warned of a silent-wrong-answer trap and then given no rule for avoiding
it, no statement of the default `flux_type`, and two examples that disagree
(`term_div(eta_v2, boundary_type::bothsides)` vs
`term_div(neg_I_sinth, flux::upwind, bc::bothsides)`). §8.1's six-step recipe
does not mention `flux_type` at all.

**What would unblock me:** add a step 0 to the recipe: "write `flux_type::upwind`
on both links of every chain; the internal flip makes them alternate correctly
(§3.4). Never write `central` on one link and a side flux on the other. If your
coefficient changes sign inside the domain, [do X]." Also: name the defaults.

### 6. §4.5 never says which wall a one-sided `grad` flag acts on — **blocker for Exercise 2**

> "The flip maps `right` → `left`, so `{div bothsides, grad right}` writes the
> grad's Dirichlet block at one wall only, and the other wall keeps its clean
> Neumann pairing."

Two plausible readings (block at right / block at left), and the sentence's own
phrasing points at the wrong one. Resolvable only from a table 400 lines later.

**What would unblock me:** "…writes the grad's Dirichlet block **at the right
wall**". Six words.

### 7. §8.2's Dirichlet recipe fails §8.3's audit step 5 — **contradiction, unresolved**

§8.2: "advective terms left `none`" at a Dirichlet wall.
§8.3 step 5: an unsealed advective term whose characteristic points inward
"will manufacture density".
P9: "It does *not* improve with refinement."

For a Fokker–Planck drag at v_max these are simultaneously in force. Nothing
reconciles them.

**What would unblock me:** a sentence in §8.2's Dirichlet row: "the free
advective flux at a Dirichlet wall is harmless because f(wall) → 0 kills it;
audit step 5 applies only where the wall value is not otherwise pinned" — *if*
that is in fact true. If it is not true, say so, because then the recipe is
wrong.

### 8. "Outermost" is used for both ends of a chain — **ambiguity**

§8.1 step 3: "flag every member of S with `bothsides` on its **outermost** `div`"
(the div is the left factor of A_div A_grad, so outermost = first-written).
§3.7 footnote: "`chain_level` defaults to −1, i.e. the **outermost** link"
(−1 idiomatically = last = the grad).

**What would unblock me:** pick one and define it once — "outermost = the
left-most factor of the matrix product = the first entry in the chain list".

### 9. No end-to-end worked example — **structural gap**

The abstract promises "a reader can design boundary conditions for *new* terms
rather than pattern-match existing ones." But there is not one complete example
anywhere going from *a stated PDE* to *complete code*. The closest is three
elided lines from ICRF_1D (§7) whose coefficient sign contradicts §6.1. Both of
my exercises would have taken ten minutes instead of an hour if §8 ended with:
"Worked example: ∂_t f = ∂_v(Af + D∂_v f) on [0, v_max], reflecting both ends.
Here is the complete code, here is what each argument is, here is the realised
wall law."

### 10. Symbols and terms used before or without definition

- **`P_L`** — Executive summary table: "build `−c g P_L/√h` as a source vector".
  `P_L` is never defined. §2.5 later introduces ℓ_L = P(−1); I assume they are
  the same object, but the summary is explicitly billed as "self-contained and
  meant to be read alone."
- **"trace projectors"** (§2.5 title) — the section defines four rank-one outer
  products and never explains in what sense they project, or why the plural
  noun "projector" is the right word. The content is fine; the title is jargon
  that made me expect something I then did not find.
- **"upwind alternation"** — used in §11.1 ("the consistent 2.5× ratio
  |f'(0)|/|f'(2)| is the upwind alternation of the LDG chain, not an error") and
  in P6, never defined. §4.4 defines "the alternating flux f̂ = f⁻, q̂ = q⁺" but
  never connects the phrase to it. In §11.1 the phrase is doing real work — it
  is the *entire* explanation for why a quantity that ought to be symmetric is
  off by a factor of 2.5 — and I have no way to check it. See #11.
- **`p`** in "continuous to O(h^p) across the last cell" (§5.2) — polynomial
  degree? degree+1? Never said, and the whole sum-rule derivation rests on it.
- **`rebuld_chain`** (§4.2) — presented as a source symbol. Typo, or the actual
  misspelled function name? A novice cannot tell, and it affects whether I can
  grep for it.
- **`bc::` vs `boundary_type::`**, **`flux::` vs `flux_type::`** — the two code
  snippets use different spellings for what must be the same enums. I do not
  know which is real (a `using` alias? a typo?).
- **`term_robin(r_l, r_r)`** appears as a thing "you write" in the summary
  table, but §6 says the actual API is `set_left_robin`/`set_right_robin` on an
  existing chain, and that a standalone `term_robin` has an extra requirement
  about mass factors in the other dimensions. The reference card names the
  wrong function.

### 11. Claims I am simply asked to trust

- **§11.1**: "The consistent 2.5× ratio |f'(0)|/|f'(2)| is the upwind
  alternation of the LDG chain, **not an error**." A clean factor of 2.5 in a
  symmetric problem is exactly what a careful reader would flag as a bug. It is
  dismissed in one clause with an undefined term and no derivation. Where does
  2.5 come from? Is it degree-dependent? This is the one number in the
  verification section I actively disbelieve, purely because nothing supports it.
- **§4.4**: "Sum … over cells … use the edge-regrouping identity and the
  cellwise product rule … The two summed equations add to [eq. 20]." I could not
  reproduce this. The *next* step (completing the square in eq. 21) I checked
  and it is correct — so the document has the good and the bad step adjacent,
  which makes the gap more conspicuous, not less.
- **§2.4**: "the quadrature array already carries the ½, leaving 2/h" —
  unverifiable without the source. Minor.
- **§2.6**: "it is dissipative: f·(−Af) = −(c/2) Σ_edges [[f]]² ≤ 0" — asserted,
  not derived, and as written it cannot hold with boundary terms present.
- **§5.2**: the derivation assumes **steady state** ("At steady state
  dm_N/dt = 0"), then concludes with eq. (24) written as an exact implication —
  and the sum rule is thereafter applied throughout the document to
  time-dependent runs without comment. §5.5 does eventually admit the pointwise
  condition is only asymptotic, which partly rescues this, but the steady-state
  assumption is never revisited.

### 12. Tables I could not act on

- **§8.2 "Worked cases"** is the table I most wanted, and it is the one I could
  not use, because every cell describes *flags* and no cell describes
  *coefficients*. "`bothsides` on every outermost `div`" tells me the third
  argument. It does not tell me the first, which is the one I got wrong.
- **§8.3 "The audit"** has a column "flux F_k" with entries like
  "c_drag(x)f" and "+4f(b), inward" — but "inward" at the right wall with a
  **positive** value means the audit table is written in the ∂_t f = +∂_x F
  convention, opposite to §8.1 step 1 which the reader was told to use one page
  earlier. I spent real time convincing myself the leak-rate signs still came
  out consistent (they do, because F = −G flips both sides). A reader who does
  not do that algebra will get the sign of their leak rate backwards.

### 13. Jargon that could be plainer with no loss

- "the trace projectors" → "the four edge blocks"
- "adjoint consistency" — I was primed to look for this and it does not appear.
  Credit where due.
- "a Gram form" (§4.6) — actually *is* glossed adequately ("−BB^T, symmetric
  and negative semi-definite"). Fine as written.
- "upwind alternation" → "the LDG rule that the div takes its trace from one
  side and the grad from the other" (say it once, then the short phrase is fine).
- "bit-identically" (§1, §6.2) — precise and useful, keep it.

---

## Part 3 — What the document does well (do not change these)

1. **§2.7, the telescoping argument, is excellent.** "Every unit of mass that
   appears or disappears in an ASGarD run passed through one of exactly two
   numbers" reframed the whole subject for me in one sentence. I followed the
   derivation completely, without gaps, and I now understand *why* DG boundary
   conditions are what they are rather than just what to type. This is the
   single most valuable page in the document.

2. **The "zero flux ≠ zero derivative" thread.** §5.3's emergent Robin, the
   "dirichlet flux trap" paragraph in §3.2, and the last row of §8.2 ("Pure
   Neumann f' = 0 (rarely what you want)") together demolish a mistake I would
   certainly have made. I came in assuming a reflecting wall meant f' = 0. I now
   know it does not, and I know why. Keep all three, including the redundancy —
   the repetition is what made it land.

3. **§2.4's derivation of 1/h and 2/h.** Short, complete, checkable, and it
   ends with a dimensional sanity check. This is the model the rest of the
   document should follow.

4. **§2.6's interior formula is verifiable.** I checked the upwind collapse
   (c > 0, s = +1 → lower = −c/h·from_left, diag = +c/h·to_right) by hand from
   the given ℓ, r, s definitions and it works. Likewise eq. (10)'s upwind flux
   collapses correctly from the jump/average definitions in §2.1. Sections that
   let a reader check the arithmetic build enormous trust.

5. **The `[INFERENCE]` / `[UNRESOLVED]` tagging, and §11.9 "Corrections to
   earlier claims".** Both are rare and both are exactly right. Being told
   "mass is conserved to 4e−15 *while f diverges to 1.4e+06*, and this corrects
   an earlier claim that it loses mass" is worth more than a confident-sounding
   document. Do not remove the corrections section when it gets stale — extend it.

6. **§9's P1–P11 and §11's verification tables.** The pitfalls are concrete,
   numbered, individually citable, and several are things no amount of theory
   would have warned me about (P7, the `-if` deck flag, is not even about DG and
   is obviously the product of someone losing a day to it). The verification
   tables have predictions *and* measurements *and* relative errors, which is how
   it should be done.

7. **The parity mnemonic (§4.6) cross-checking the primary rule.** In
   Exercise 2 I derived the flags two independent ways and got the same answer.
   That is the moment my confidence went from "guessing" to "probably right".
   Having a redundant check is worth the extra page.

8. **§5.4, "signed and local".** The measurement where the negative drag flips
   the sign of the wall slope to +0.66667, and the one where changing D_2's
   *interior* profile while fixing its *wall value* leaves the answer unchanged
   to 5 digits, are the two most persuasive numbers in the document. They rule
   out every competing hypothesis I could have invented. Keep exactly as is.

---

## Summary for the author

Both exercises are answerable **at the level of flags** — and I got both, with
the parity mnemonic as an independent check. That part of the document works.

Neither exercise is answerable **at the level of coefficients**, and Exercise 2
has one genuine unresolved contradiction (recipe vs. audit step 5) that I could
not settle from the text.

If you change three things, change these:

1. State, once, in a displayed equation, what
   `chain{div(c_d), grad(c_g)}` under mass m realises as a PDE, with signs.
   Then make eq. (14) agree with it or annotate it as pre-α.
2. Reconcile §6.1's `c(v) = −v²η(v)` with §7's `// c = eta * v^2`.
3. Add one complete, compilable, end-to-end example: stated PDE → full code →
   realised wall law. Ideally the reflecting Fokker–Planck of Exercise 1, since
   it is the canonical FENRIS case.

And add the six words to §4.5 that say which wall a one-sided `grad` flag
acts on.
