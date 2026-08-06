# Adversarial review of `asgard_boundary_conditions.tex`

Reviewed against `01_code_analysis.md`, `02_weak_form.md`, `03_experiments.md`,
`00_session_findings.md`, and — where the two sources disagreed — against the
ASGarD 0.9.1 headers in `~/venvs/asgardpy/include/` and the four FENRIS solvers
directly.

Line numbers are `.tex` line numbers as of the reviewed revision (2022 lines).

Verdict in one line: **the numbers are clean and the corrections from `03` have
all landed, but the document carries two mutually contradictory sign conventions
for the wall flux, and every downstream sign statement — the leak rate, the audit
table, and the Robin dissipativity condition — inherits the wrong one.** Four
BLOCKERs, nine MAJORs.

---

## BLOCKERS

### B1. Two incompatible sign conventions for `F`, used interchangeably

**Where:** `\eqref{eq:massbal}` (l.251–256, and the card at l.101–105) versus
`\eqref{eq:weak}` (l.287–292), `\eqref{eq:cellledger}` (l.418–421),
`\eqref{eq:master}` (l.429–432) and `\eqref{eq:openleak}` (l.1023–1027).

`\S`sign derives, correctly and with a source cross-check,

```
    d_t f = -d_x F,  F = c f     =>   dN/dt = F(a) - F(b).            (eq:massbal)
```

`\S`oneterm then restarts from "the FENRIS sign convention, `d_t f = d_x F`"
(l.271–272, imported verbatim from `02_weak_form.md` §1) and telescopes to

```
    d/dt \int f = F_b - F_a.                                          (eq:master)
```

These are opposite. `\eqref{eq:weak}` writes `\int v d_t f_h = ...`, which is
false: the object being assembled is `A`, and the ODE is `d_t f = -A f`. The
`alpha = -1` is applied once, in `\S`sign, and then silently dropped for the
whole of `\S\S`weak–sum. Consequences that are not cosmetic:

* `\S`flags, l.504–506: "Then \eqref{eq:master} reads `Ṅ ∋ c f^int_wall`". Under
  `eq:master` a free wall with `c(b) > 0` *adds* mass. Under `eq:massbal` it
  *removes* mass. The physics the paragraph then asserts (ICRF's inward drag
  manufactures particles) is only true under `eq:massbal`: FENRIS's drag has
  `c = -v^2 eta < 0` at `v_max` (verified in `ICRF_1D.cpp:262`), so
  `dN/dt|_b = -F(b) = +v^2 eta f > 0`. Read through `eq:master` the same term
  would *lose* mass, i.e. the document's own explanation of the +63 % pathology
  is sign-inverted relative to its own master equation.
* `\eqref{eq:openleak}` and audit item 1 (l.1542–1544), "the sum of the unsealed
  column is your leak rate, exactly", carries the `eq:master` sign and is
  therefore backwards relative to `\S`sign.
* The audit table's drag row (l.1531) reads `F_k` at the wall as "`+4f(b)`,
  inward". Under `\S`sign's `F = cf`, `+4f(b)` at the *right* wall is
  **outward**. The table is internally consistent with `eq:master` and
  inconsistent with the boxed rule the document tells you to read first.

This is the single most damaging defect for a solver-design document: the whole
point of `\S`sign is that the reader will get the sign wrong otherwise, and then
five sections proceed in the other convention.

**Fix.** Keep `F = c f` (the `term_div` coefficient) as the *only* meaning of
`F` in the document, and pay the `alpha = -1` once and visibly:

* rewrite `\eqref{eq:weak}` with `(A f)` on the left, not `d_t f`;
* rewrite `\eqref{eq:cellledger}` as `(A f)|_{cell j} = F_{j+1/2} - F_{j-1/2}`;
* rewrite `\eqref{eq:master}` as `dN/dt = -(F_b - F_a) = F_a - F_b`, and say in
  one sentence that the minus is the `alpha = -1`;
* flip the sign in `\eqref{eq:openleak}` and in audit item 1;
* recompute the audit table's "`F_k` at the wall (sign)" column: the FENRIS-style
  drag that manufactures particles has `c(b) < 0`, so the entry should be
  `-4 f(b)` with the leak `dN/dt = -F(b) = +4 f(b)`, inward.

Then check that `\S`sum's `F = Af + D_1 f' + D_2 f'` (l.1005) and the `03` leak
tables are re-signed to match; `03 §3d` measured `dN/dt = F(2) - F(0)` in the
`eq:master` convention, so either the toy's `term_div` coefficients are `-2x`
(likely) and the tables are fine once `F` is redefined, or the tables need their
signs flipped. State which, once, in `\S`verify.

---

### B2. The Robin dissipativity condition is sign-inverted, and both worked examples violate the version printed

**Where:** card l.174; `\S`robin "Dissipativity: the sign of `r`" l.1255–1266;
recipe worked-cases table l.1498–1499; "What to remember" item 6 l.2008–2009.

The document says: *dissipative iff `r <= 0` at the right wall, `r >= 0` at the
left.* The correct condition is the **opposite**.

Derivation, independent of `\eqref{eq:Bb}`. `gen_robin_cmat` adds
`+r_b/h * to_right` to the assembled matrix `A` — the document establishes this
correctly at l.1188 ("the same block with the same sign and the same `1/h`"), so
the Robin block is exactly a free `div` block with `c = r`. Then

```
    f^T (escale * r * to_right) f = r * (f^-(b))^2
```

and since the ODE is `d_t f = -A f`,

```
    (1/2) d/dt ||f||^2 = -f^T A f  ⊃  -r (f^-(b))^2,
```

which is dissipative **iff `r >= 0` at the right wall** (and, by the mirror
block `-r_a/h * to_left`, iff `r_a <= 0` at the left).

Three independent confirmations that the printed sign is wrong:

1. **`\S`robin's own equivalence.** Sealing corresponds to `r = -c(wall)`
   (l.1233–1234, correct). ICRF_1D's drag has `c = -v^2 eta` (`ICRF_1D.cpp:262`,
   and the document itself says so at l.1209), so the historical
   `set_right_robin(v_max^2 * eta(v_max))` is `r > 0` at the right wall. That
   configuration was verified bit-identical to sealing, i.e. it is the *known
   good* one — yet it violates the document's stated `r <= 0`.
2. **LHCD_2D.** `term_div(-0.5, upwind, ...)` with the former
   `set_*_robin(0.5)` (`LHCD_2D.cpp:169–181`): again `r = +0.5 > 0` at the right
   wall, again the known-good value.
3. **The pathology statement itself.** l.1264–1266: "the free bracket at an
   inflow wall is a Robin term with `r = c(b)` of the destabilising sign". With
   the printed rule, destabilising means `r > 0`, i.e. `c(b) > 0`. But ICRF's
   uncancelled drag bracket, the one that manufactured +63 % particles, has
   `c(b) < 0`. Under the document's own rule the production bug would have been
   *stabilising*.

**Where the error enters:** l.1257, "Evaluate \eqref{eq:Bb} with `q̂_b = r f^-`".
`\eqref{eq:Bb}` lives in the `d_t f = +d_x q` convention (B1), whereas
`gen_robin_cmat`'s `r` is the coefficient in `d_t f = -d_x(cf)`. The correct
substitution is `q̂_b = -r f^-`, giving `B_b = -r (f^-)^2`.

**Fix.** Replace `q̂_b = r f^-` with `q̂_b = -r f^-`, replace `B_b = r(f^-)^2`
with `B_b = -r(f^-)^2`, and change all four occurrences of the rule to **`r >= 0`
at the right wall, `r <= 0` at the left**. Then add the sanity line: *sealing a
term with `c(b) < 0` is spelled `r = -c(b) > 0`, which is on the dissipative
side, as it must be.* Also fix l.1219 (`dN/dt|_a = +r_a f(a)`): that is correct
under `eq:massbal`, so it becomes a useful cross-check once B1 is fixed — an
absorbing left wall needs `r_a < 0`, matching the corrected rule.

This is a BLOCKER because `\S`recipe tells a designer to pass `r <= 0` for a
partially absorbing right wall. Doing so builds a positive feedback of strength
`|r| f^2` on the wall value — the exact failure mode the document was written to
prevent.

---

### B3. The `\S`mass code listing prints the wrong sign for the ICRF_1D coefficients, contradicting `\S`robin

**Where:** l.1326–1329, the `ICRF_1D.cpp` listing and its comments
`// c = eta * v^2` and `// c = zeta * v^2`; contradicted by l.1209 in the same
document, which correctly says `c(v) = -v^2 eta(v)`.

Ground truth, `FENRIS/ICRF_1D/src/ICRF_1D.cpp:260–287`:

```cpp
auto eta_v2  = ... f[i] = - pow(v[i],2) * plasma.eta(v[i]);    // c = -eta*v^2
auto zeta_v2 = ... f[i] = - pow(v[i],2) * plasma.zeta(v[i]);   // c = -zeta*v^2
```

`01_code_analysis.md` §5.2 dropped both minus signs and the `.tex` copied it.
This is not a typo in a comment: `\S`mass is the section whose entire thesis is
"the coefficient you hand to `term_div` is the mass-weighted flux coefficient",
and the sign is half of that coefficient. With `c = +zeta*v^2` the chain
`M^{-1}A_div M^{-1}A_grad` realises `d_t f = -(1/v^2) d_v(zeta v^2 d_v f)` —
**anti-diffusion**. A reader copying the printed listing to build a new term gets
an unconditionally unstable operator.

It also destroys the only worked example of `\eqref{eq:robineq}`: with
`c = +v^2 eta`, sealing would correspond to `r = -v^2 eta`, and the historical
`set_right_robin(+v_max^2 eta)` would be the *wrong* sign — contradicting the
measured bit-identity.

**Fix.** Correct the two comments to `// c = -eta*v^2` and `// c = -zeta*v^2`,
and add one sentence: *the div coefficient of a diffusion chain carries the
opposite sign to the grad coefficient; that sign flip is what puts the two links
on opposite upwind sides (see B4).* Then propagate the correction to
`01_code_analysis.md` §5.2, which is the source of the error.

---

### B4. "The user cannot get the LDG alternation wrong" is false, and the document contradicts itself about it two pages later

**Where:** `\S`gradflip l.586–588 ("the interior LDG alternation is automatic and
the user cannot get it wrong"); `\S`energy l.906–908 ("in ASGarD it is automatic
because `Grad = -Div^T` swaps the upwind side. *The user cannot break it.*");
contradicted by P6, l.1619–1635, and by the `\S`product caveat, l.818–825.

The alternation is **not** a property of the transpose alone. Working the
assembly out for `c` constant and `flux_type::upwind` on both links:

* `div` with `c > 0`: `lower = -(c/h) from_left`, `diag = +(c/h) to_right`, i.e.
  `q̂ = q^-` (trace from the left).
* `grad` with `c > 0`: the `term_1d` ctor stores `downwind`; assembling with
  `s = -1` gives `diag = -(c/h) to_left`, `upper = +(c/h) from_right`, and after
  `G = -Ã^T`, `f̂ = f^-` — **also from the left**. Substituted into
  `\eqref{eq:energy}` the interior bracket is `[[f]][[q]]`, not `0`. No
  alternation.

What actually produces the alternation in FENRIS is that the div coefficient is
**negative** and the grad coefficient positive (`-zeta v^2` against `+v^2`,
`-I sin th` against `+L sin th`, `-0.5` against `+x`, `-B_cv` against
`mass_x` — all four solvers). A negative `c` flips `c ± |c|`, moving the div's
trace to `q^+`, and only then is the pair alternating.

So the alternation depends jointly on the two declared `flux_type`s *and* on the
sign of each link's coefficient — which is precisely what P6 says
(l.1622–1623: "Each chain gets its own upwind alternation (`c ± |c|`) built from
*its own* coefficient; a negative coefficient picks the anti-diffusive pairing")
and what the `\S`product caveat says (`check_chain` "does not verify that the two
side fluxes point opposite ways, and it cannot know the sign of a variable
coefficient"). P6's measured blow-up — `max|f| = 4.3e76`, interior, at
`x ≈ 1.1` — *is* a user breaking the interior alternation. The document asserts
in two places that this is impossible and then measures it happening.

**Fix.** Delete both "the user cannot get it wrong / cannot break it" sentences.
Replace with: *the transpose supplies half the alternation (it swaps
`from_left`/`from_right`); the other half is the sign of each link's flux
coefficient, and the `term_1d` ctor's `upwind ↔ downwind` flip only makes the
default pairing work when the two coefficients have opposite signs. Nothing
checks this — see `\S`product and P6.* Add the sign requirement to the recipe as
a step (currently step 5 only forbids negative *individual* diffusivities; it
does not say that `c_div` and `c_grad` must have opposite signs).

---

## MAJOR

### M1. The "primary" rule as worded is false for one-sided flags; the claimed exact equivalence with the parity rule does not hold

**Where:** abstract l.63–64; card l.136–140 and the mnemonic note l.155–158;
`\S`product boxed rule l.804–808; `\S`parity l.974–979; recipe step 4
l.1459–1463; audit item 3 l.1549–1551; remember item 3 l.1990–1996.

I checked the equivalence the brief asked about. The two framings **are**
equivalent when stated set-theoretically, and the reason is exactly the one the
brief suspected: let `S_d ⊂ {L,R}` be the walls at which the `div` seals its flux
and `S_g` the walls at which the `grad` pins `f`. Both are read off the flag by
the *same* encoding — `none ↦ ∅`, `left ↦ {L}`, `right ↦ {R}`,
`bothsides ↦ {L,R}` — and `flip` is exactly set complement on that encoding.
Hence

```
    exactly one condition per wall   <=>   S_g = complement(S_d)   <=>   phi_g = flip(phi_d).
```

I verified the wall bookkeeping against the measurements: `grad right` flips to
`left`, whose assembled block lands on the *right* wall, so `term_grad(c,right)`
pins `f` at the **right** wall — and `03 §6` shows `{div both, grad right}`
diverging at `f(2)`, not `f(0)`. Consistent.

**But the document does not state the primary rule set-theoretically.** It states
it as *"exactly one of the two factors must carry `bothsides` at each wall"*
(l.807), *"count `bothsides` across its two links; must be exactly one per wall"*
(l.1550), *"exactly one of its two links must carry `bothsides`"* (l.1461). Two
divergences follow:

* **`{div left, grad right}` is a legal, well-posed pairing in which *neither*
  link carries `bothsides`** — Neumann at the left wall, Dirichlet at the right.
  It satisfies `\eqref{eq:parity}` and it is exactly the configuration you need
  for a domain with an impermeable inner wall and an absorbing outer wall, which
  is a common kinetic setup. The document's primary rule rejects it and its audit
  step 3 will flag it as a bug.
* **`periodic`.** `flip(periodic) = periodic`, so `\eqref{eq:parity}` correctly
  admits `{div periodic, grad periodic}`. The condition-counting rule assigns
  periodic neither a seal nor a pin, so it reads as "zero conditions per wall" —
  under-determined — which is wrong. The document never says how `periodic`
  interacts with the pairing rule at all.

Also a wording slip that runs through all of these: *"each factor can carry at
most **one wall condition**"* (l.63, l.136, l.805, l.1991) is wrong even for the
canonical case — `bothsides` carries a condition at *two* walls. The intended
statement is "at most one condition **per wall**".

**Fix.** State the primary rule per wall and per flag-as-a-wall-set: *at each
wall, exactly one of the two links must name that wall. The `div` names walls
where the flux is sealed; the `grad` names walls where `f` is pinned. `bothsides`
names both, `left`/`right` names one, `none` names none.* Then
`\eqref{eq:parity}` follows literally, `{div left, grad right}` is admitted, and
audit step 3 becomes "for each wall, exactly one of the two flags names it".
Add a sentence on `periodic`: it is outside both framings and must be matched
`periodic`↔`periodic`.

### M2. `\eqref{eq:gram}`–`\eqref{eq:parity}`: the Gram-form derivation is wrong, and its spectral corollaries do not hold for any FENRIS term

**Where:** `\S`parity, l.959–992.

The document writes `A_div A_grad = -T[phi_d] T[flip(phi_g)]^T` and calls it a
Gram form "precisely when the two factors coincide". The two factors **never**
coincide in practice, for three independent reasons:

1. **Different coefficients.** ICRF_1D is `term_div(-zeta v^2)` with
   `term_grad(+v^2)`; ICRF_2D is `term_div(-I sin th)` with
   `term_grad(L sin th)`. The `T`s are built from different functions.
2. **Different stored flux types.** The div stores `upwind`, the grad stores
   `downwind` (the ctor flip). `T_{s=+1}[phi]` and `T_{s=-1}[phi]` are different
   matrices even with identical `c` and identical flags — as B4 shows, that
   asymmetry is load-bearing.
3. **Per-link `M^{-1}`.** `\S`mass establishes that the stored chain is
   `(M^{-1}A_div)(M^{-1}A_grad)`, not `A_div A_grad`.

The correct statement, which I verified block by block for constant `D`, is
`A_div = D * A_grad^T` when `c_div = -D c_grad` — i.e. the Gram structure needs
the **opposite-sign** coefficient pairing of B4, not coinciding factors. With
`M^{-1}` interposed the operator is still self-adjoint and negative
semi-definite, but only in the `M`-weighted inner product `<u,v>_M = u^T M v`.

Everything the section then asserts — "real non-positive spectrum, so no spurious
growing modes and the stiff eigenvalues sit on the negative real axis where
implicit and explicit integrators expect them" — is therefore stated for an
operator that no FENRIS solver assembles.

**Fix.** Either (a) restate `\eqref{eq:gram}` with the sign and flux
qualifications and prove `A_div = D A_grad^T` for the matched case, then say the
symmetry is in the `M`-inner product; or (b) demote the whole Gram argument to a
remark on the constant-coefficient model problem and stop using it as the
*derivation* of `\eqref{eq:parity}`. Option (b) is cheaper and honest —
`\eqref{eq:parity}` is already fully established by the wall-set argument of M1,
which needs no matrix algebra at all.

### M3. "A boundary condition is a rule for those two numbers and nothing else" is contradicted by the document's own P2

**Where:** card l.109–113; `\S`telescope l.445–448 ("There is no other place to
put one. Any discussion of boundary conditions that is not a discussion of `F̂_a`
and `F̂_b` is misdirected."); remember item 1 l.1984–1986.

The `grad` link's trace `f̂` **is** a boundary condition and it does **not**
appear in `\eqref{eq:master}`. That is precisely why the ill-posed pair conserves
mass to `4e-15` while `f` diverges to `1.4e6` (P2, l.1578–1587), and why the
Dirichlet pair's spectrum is a boundary property invisible to the mass balance.
The master statement is about the *mass balance*, not about boundary conditions
in general, and the document's most useful debugging advice (P2: check `max|f|`,
mass conservation will not tell you) is the direct refutation of the sentence.

**Fix.** Scope the claim: *for the conservative content of a term, a boundary
condition is a rule for `F̂_a` and `F̂_b`. A second-order term has two more wall
degrees of freedom, the traces `f̂_a` and `f̂_b`, which do not enter the mass
balance at all — see `\S`chain and P2.* Delete "any discussion ... is
misdirected".

### M4. "A `term_md` may carry a flux in at most one dimension" is unsupported by the cited evidence and false for the FENRIS solvers

**Where:** l.1390–1398.

Cited evidence: `term_md::flux_dim()` "returns the *first* dimension with a flux"
and the inhomogeneous-flux API asserts `flux_dim() != -1`. Neither establishes
"at most one". I read `asgard_pde.hpp:694–712`: `flux_dim()` loops and returns
the first hit; the only `rassert`s on the `+=` path are `flux_dim() != -1` and
"the flux function has to be constant in the dimension of `flux_dim()`". Nothing
forbids a second flux dimension; the doxygen comment claims "only one such is
allowed" but no code enforces it.

And it is violated in production. `ICRF_2D.cpp:739–742`:

```cpp
pde += term_md({div_dx,    res, grad_C});   // div_dx: flux in x;  grad_C: flux in theta
pde += term_md({div_theta, res, grad_E});
```

This `mode::chain` `term_md` carries a flux in `x` *and* in `theta`.
`flux_dim()` silently returns `0`. So the document's structural claim, and the
"a physically two-directional operator is built as *several* `term_md`s"
sentence that follows it (l.1395–1396), are both contradicted by the solver the
document uses as its worked 2-D example.

**Fix.** Replace with what is actually true: *a separable `term_md` carries a
flux in one dimension per `term_1d` factor, and `flux_dim()` returns the first
one it finds; nothing enforces uniqueness. A `mode::chain` `term_md` can and does
carry fluxes in several dimensions (ICRF_2D's QL operator). The inhomogeneous
`boundary_flux` API will attach to `flux_dim()`'s first hit, silently, which is
a trap for chain-mode terms.* Mark the last clause `[INFERENCE]` unless tested.

### M5. The central rule is declared inapplicable to `term_md` chains, and nothing is offered in its place — yet that is what the motivating bug lived in

**Where:** l.1400–1404.

The document says `mode::chain` `term_md`s are applied sequentially at run time
and "the one-condition-per-factor rule of `\S`chain is a statement about
`term_1d` chains." Full stop. But ICRF_2D's and LHCD_2D's QL operators — the
terms whose `bc::none` flags produced the +63 % particle manufacture and the
LHCD false steady state that motivated the whole document — are exactly
`mode::chain` `term_md`s (`ICRF_2D.cpp:739–742`, `LHCD_2D.cpp:318–331`), built
from a `div_dx` with `bc::bothsides` and bare `term_grad`s at the default flag.

A reader who has internalised `\S`chain has no rule for these at all. Nor does
`\S`recipe: step 3 says "flag every member of `S` with `bothsides` on its
*outermost* `div`", and `grad_B = term_md({term_grad{B_x}, term_volume{B_theta}})`
has no `div`.

**Fix.** Add a subsection. The intermediate vector *is* materialised for a
`term_md` chain, so each link's wall blocks act on a real field; state whether
the same parity applies (it evidently does — the FENRIS QL chain uses
`div bothsides` + `grad` default, the Neumann parity — but say so and say why),
and state explicitly what a bare `term_grad` link at default `none` contributes
at the wall. If this was not established, say so and tag it `[UNRESOLVED]`
rather than leaving the gap silent.

### M6. Recipe row "Dirichlet `f = g != 0`" does not realise a Dirichlet condition

**Where:** worked-cases table, l.1489–1493.

The row's own "realised condition" cell says "prescribed wall **flux**". That is
correct — `boundary_type::left` + `left_boundary_flux(g)` prescribes `F̂_a = g`,
per `\S`flags l.695–718. It is not `f(a) = g` unless the term is a bare advective
`div`, where `F = cf` makes the two interchangeable. For a diffusion chain they
are different conditions, and the document has an entire paragraph (`\S`flags,
"The 'dirichlet flux' trap", l.520–524) warning that confusing them is "the single
most common way to mis-flag a term". The recipe table then commits the confusion
in its own row heading.

**Fix.** Rename the row "Prescribed inflow flux `F = g`" and add a separate row
for inhomogeneous Dirichlet: `{div none, grad bothsides}` with the boundary flux
attached at the `grad` link via `boundary_flux::chain_level(d)` (the mechanism is
already described at l.713–718 but never connected to the recipe). If that
combination was not tested, tag it `[INFERENCE]`.

### M7. Recipe step 4 permits an over-determined wall without warning

**Where:** l.1455–1463.

Step 3 chooses a sealed set `S` (imposing `sum_{k in S} F_k = 0` at the wall).
Step 4 then says: "If the term is *not* in `S` and you nonetheless want a
Dirichlet wall from it, that is `{div none, grad bothsides}`." Doing both at the
same wall gives *two* conditions on one scalar — the sum-rule Robin from `S`,
plus the `f̂ = 0` pin from the other chain's `grad` — which is `\S`four(iii),
blow-up. Nothing in steps 3–4 or in the audit forbids it; audit item 3 only
counts flags *within* a term.

**Fix.** Add to step 4: *a `grad bothsides` anywhere at a wall imposes `f = 0`
at that wall for the whole problem. It is mutually exclusive with a non-empty
sealed set at that wall. Choose one law per wall.* Add the corresponding cross-
term check to the audit: "at each wall, either `S` is non-empty **or** some
`grad` carries `bothsides` — never both."

### M8. The sum-rule derivation assumes steady state; the rule is then used unconditionally

**Where:** l.1031–1048.

The derivation sets `dm_N/dt = 0` ("At steady state") and only then obtains
`sum_{k in S} F_k(b) = 0`. But `\eqref{eq:sumrule}` is stated unconditionally,
the design recipe applies it to arbitrary transients, and the verifying runs
(`03 §3a`, level 8, `t = 1`) are not steady states.

The rule does survive the transient, but by a different argument: `m_N = O(h)`,
so `dm_N/dt = O(h)` and the pointwise residual is `O(h)` on top of the `O(h^2)`
trace error, without any steady-state assumption. That argument is the honest
one and it also explains why `\S`"Weakly imposed and converging" measures
`O(h^2)` rather than exactness.

**Fix.** Replace "At steady state `dm_N/dt = 0`" with the `m_N = O(h)` argument,
or state explicitly that `\eqref{eq:sumrule}` is a steady-state / `h → 0`
statement and that in a fast transient the wall residual is larger than the
`2.5e-3`-at-level-4 figure quoted at l.1147.

### M9. `\S`recipe runs out on both of the test cases I was asked to apply it to

**(i) A wall that absorbs a fixed fraction of the incident flux.**

Step 2 offers four wall laws; a fixed-fraction absorber is not one of them, and
the closest, `alpha f + beta F = 0`, is not the same thing. The recipe never says
how to translate "absorbs fraction `alpha` of the incident flux" into
`alpha f + beta F = 0`, and that translation is not trivial: the DG wall exposes
only `f(wall)` and one *net* flux, never separated incoming and outgoing halves.
It requires a modelling assumption (e.g. a half-space kinetic flux `f v̄ / 4`)
that the recipe does not mention. Then the "Partially absorbing" row runs out in
four more places:

* **The value of `r` is left circular.** `r = (absorbed flux)/f(wall)`. If the
  incident flux is itself proportional to `f(wall)` this closes; the recipe does
  not say that this is a requirement, and does not say what to do when it is not.
* **The sign is wrong** (B2): following the row as printed gives an
  anti-dissipative wall.
* **`r` must be mass-weighted, and the document never says so.** I read
  `asgard_term_build.cpp:1110–1120`: when a mass is set, `gen_robin_cmat`'s
  output is passed through `bmass->solve(...)`. So `r` carries the same Jacobian
  weight as a `div`'s `c`. `\S`robin says only that the Robin block "is added
  after the multiplication, so it is not composed with the other links" —
  which reads as "no mass" and is exactly wrong. Recipe step 1 insists on
  including the Jacobian for `F_k`; the Robin row says "the *physical* flux
  coefficient", the opposite instruction.
* **`r` is a scalar and requires a chain.** `set_left_robin`/`set_right_robin`
  take a single `P` and `rassert(is_chain(), ...)` (`asgard_pde.hpp:694–704`).
  You cannot attach a Robin to a bare advective `term_div`, you cannot make `r`
  vary in time, and you cannot make it a function of anything. The footnote at
  l.1249–1250 mentions the chain requirement; the recipe row does not, and
  nothing anywhere says `r` is a compile-time-fixed scalar.

Net: the recipe does not give an unambiguous answer for this case. It runs out at
step 2 (no rule for translating a reflection coefficient into `alpha f + beta F`)
and again at the Robin row (magnitude, sign, weighting, and the scalar/chain
restrictions are all unspecified or wrong).

**(ii) A 2-D term whose flux at the wall depends on the other coordinate.**

The recipe is written entirely in 1-D; the word "dimension" does not appear in
`\S`recipe. Applying it:

* Step 3 asks for a sealed set `S` *per wall*. Seal/free is a per-term,
  per-dimension **binary**. There is no way to seal a term over part of a wall,
  so any wall law whose coefficient varies along the wall is unreachable by
  flags. `\S`md's one sentence "the sum rule applies per wall per dimension"
  (l.1396–1398) is the only guidance and it does not address variation *along* a
  wall.
* The Robin escape hatch fails too: `r` is a scalar (above), so a wall-varying
  Robin coefficient is impossible in the API.
* The one mechanism that *does* admit dependence on the other coordinate is
  `boundary_flux`, which requires the supplied function to be constant only in
  the **flux** dimension and may therefore vary in the others
  (`asgard_pde.hpp:1288–1289`). The recipe never mentions this, and the
  document's own paraphrase — "you prescribe a wall *value*, not a wall profile"
  (l.701–702) — actively misleads: you cannot prescribe a profile *across* the
  wall in the flux direction, but you *can* prescribe one *along* the wall in the
  other directions. That is exactly what case (ii) needs.
* A genuinely non-separable wall flux cannot be represented at all; it must be
  decomposed into separable `term_md`s, and the recipe says nothing about how
  many terms that costs or how the sum rule then closes.
* Cross-derivative fluxes — `Gamma_x` containing `d_theta f` — are the real
  form of case (ii) in this codebase, and they are exactly M4/M5's chain-mode
  `term_md`s, which the recipe does not cover. The ICRF_2D source comment
  (`ICRF_2D.cpp:576–600`) contains the actual reasoning for this case (the QL
  tensor is rank-1, so zeroing both flux components forces `v·grad f = 0` and the
  cross term drops out). That is a genuinely useful design pattern for case (ii),
  it is measured, and the document does not contain it.

**Fix.** Add a "when the recipe does not apply" subsection naming these four
boundaries explicitly: non-separable wall data, coefficients varying along a
wall, cross-derivative fluxes, and chain-mode `term_md`s. Add the ICRF_2D
rank-1 argument as the worked 2-D case. Correct the `boundary_flux`
"value, not profile" sentence.

---

## MINOR

### m1. `0.2018` and `0.2014` are used interchangeably, and `0.2018` is presented as measured

l.882 and l.1592 say `≈0.2018`; l.1903 and l.1957 say `≈0.2014`. Per `03 §6b`
these are different objects: `0.201417` is the level-7 **measurement**,
`≈0.2018` is the `O(h)` **extrapolated limit**. P3 (l.1592) says "Measured: the
good wall sits at `≈0.2018`" — that number was not measured. Fix: quote
`0.201417` at level 7 (extrapolating to `≈0.2018`) and use one form throughout.

### m2. The mongrel's left wall is called "Neumann-like" — the same overstatement P3 exists to correct

l.870–872 and l.1904: `f(0) → ≈0.192` is called "a finite Neumann-like value".
The clean Neumann is `0.237950`; `0.192` is **19 % off** — a larger discrepancy
than the 15 % that P3 correctly refuses to call "exactly Neumann". Fix: apply the
same standard — "a finite, bounded value (`≈0.192`), 19 % below the clean Neumann
wall, so 'Neumann-like' only in that it does not vanish".

### m3. "Sealing one diffusion channel silences all of them" lost `01`'s `[INFERENCE]` tag and over-generalises

l.1102–1108. `01_code_analysis.md` §8.5 tags the mechanism `[INFERENCE]`
("both diffusion terms are chains whose `grad` factor produces the same discrete
`q`"). The `.tex` states it untagged, and the heading says "all of them" where
the tested case is two channels sharing an identical `grad` factor. Two
different diffusion chains do **not** produce the same discrete `q` if their
`grad` coefficients differ.

The clean fix removes the need for a tag: this is not a separate fact, it is the
three-regimes table. Sealing any `S` with `D_S(x_w) != 0` forces
`f'(x_w) = -A_S f / D_S`; if `A_S = 0` that is `f'(x_w) = 0`, which zeroes the
wall flux of *every* purely-diffusive channel evaluated at the same wall,
whatever its coefficient. State it that way and cite the measured
`9.01295 / 9.01295 / 9.01294` as the confirmation.

### m4. The `2.5x` left/right ratio of `|f'|` is attributed to a mechanism that was not verified

l.1713–1715: "The consistent `2.5x` ratio `|f'(0)|/|f'(2)|` is the upwind
alternation of the LDG chain, not an error". `03 §1b` asserts the same without
evidence beyond both walls converging at the same rate. It is plausible (the two
walls sit on opposite sides of the alternation), but it is an inference. Tag it
`\INF` or drop the attribution and keep only the measured fact (same rate, ratio
level-independent).

### m5. "`4e-15` relative" is relative to `max|f|`, not to the conserved quantity

l.188–189, l.1581–1582, l.1951. `6.3e-9 / 1.4e6 = 4.5e-15` is the error relative
to the **field magnitude**. Relative to the conserved mass `0.443` it is
`1.4e-8` — still negligible, but seven orders larger, and the reader of a
conservation diagnostic will compute the latter. Also worth noting that the
absolute error *grows* about 10–20x per level (`2.9e-11 → 5.1e-10 → 6.3e-9`),
which strengthens P2's point rather than weakening it. Fix: say "relative to
`max|f|`" and quote both.

### m6. `\S`recipe's "Dirichlet `f=0`" row leaves the advective terms `none` without addressing the inflow question

l.1483–1487: "advective terms left `none`". If the advective `c` points inward at
that wall this is the destabilising Robin of P9 / `\S`robin, mitigated only
because the `grad`'s pin drives `f(wall) → 0`. The recipe's own audit step 5
("for every *unsealed* advective term, does the characteristic leave the
domain?") would flag the row it prescribes. Add a sentence resolving it.

### m7. Missing corroboration that is stronger than what is cited

`ICRF_2D.cpp:610–614` records a *measured* history that is the sharpest available
evidence for `\S`robin's central claim ("`r` is a flux coefficient, not a
log-derivative"): passing `r = A_cv/B_cv = 2*x_max` — which is precisely the
log-derivative `A/D` the section warns against — "overshoots ~14x and the solve
diverges", while `r = A_cv` and plain `bothsides` "agree to solver tolerance".
That is a direct experimental refutation of the dimensional confusion, in the
production code, and the document does not use it.

### m8. `set_right_robin`'s preconditions are in a footnote, not in the recipe

`rassert(is_chain(), ...)`: a Robin cannot be attached to a bare `term_div`. In
the footnote at l.1249–1250, absent from the recipe row that tells you to call it.

### m9. The refinement table drops a column

l.1791–1803: the `seal{A,D_2}` relative-error column present in `03 §3e`
(`3.8e-3, 8.5e-4, 2.0e-4, 4.8e-5, 1.2e-5`) is missing, leaving an asymmetric
table. It is the column that supports "the steepest condition converges at the
same rate" (l.1158).

### m10. `term_grad(c, left/right)` is missing from the card, and which wall it pins is never stated

The card table (l.117–133) omits the one-sided `grad` rows. `\S`gradflip explains
that `right → left` internally, which invites the reading that `term_grad(c,
right)` pins the **left** wall. It pins the right wall; `03 §6`'s `grad right`
row (`f(2)` diverging) is the confirmation. State it once, explicitly.

---

## NITS

* **n1.** l.71–74, "Every quantitative claim is tagged as *measured*,
  *structural*, or `[INFERENCE]`." Most body numbers carry no tag; they are
  merely cross-referenced to `\S`verify. Either add the tags or soften the claim.
* **n2.** P7 (l.1637–1643, the `-if` deck-file gotcha) is a fact about the
  verification harness, not about ASGarD boundary conditions. It belongs in
  `\S`verify's "how to run", not in a pitfalls list a solver designer reads.
* **n3.** l.183–197 promises "the five things that will bite you" and `\S`pitfalls
  delivers eleven. Say "five of the eleven in `\S`pitfalls".
* **n4.** `\S`four(iv)/`\S`energy, l.945–954: "This is exactly the measured
  mongrel." The energy argument predicts one anti-dissipative wall; the
  measurement shows net mass *loss* (`0.443 → 0.248`). The agreement is
  qualitative (asymmetry) only. "Consistent with" rather than "exactly".
* **n5.** l.1382–1388 ("a flag on the outermost `div` of dimension `d` controls
  dimension `d`'s wall and nothing else") is a structural inference from the
  tensor form and is not tagged. It is almost certainly right; tag it anyway,
  since neighbouring structural claims (`\S`sampling, `\S`md) are explicitly
  labelled "not separately measured".

---

## What I checked and found clean

Stated explicitly, per the brief, rather than padded with invented findings.

* **Every number in `\S`verify reconciles with `03_experiments.md`.** I checked
  all eight sum-rule rows, all eight `dN/dt` rows, the five-row refinement table,
  both eigenvalue tables, the six-pairing table, the negative-diffusion control,
  and the Robin/steady-state table. The only defect found is the dropped column
  of m9.
* **No stale `00_session_findings.md` number survives.** I grepped for `2.60`,
  `3.69`, `6.3` (as a slope), `0.2228`, `0.2215`, `3.884`, `8e2`, `5.3e-4`,
  `2e-6`, `7e-15`, `0.77`, `5.6e2`, `+63%`, `0.0043`. The only occurrences of the
  first three are inside the "Corrections to earlier claims" table, correctly
  labelled as the earlier, wrong values. The three corrected values
  (`-2.66627`, `-3.99931`, `-7.99840`), the `4e-15` conservation of the ill-posed
  pair, and the 15 %-off good wall all appear correctly.
* **`\eqref{eq:robineq}` and "sealing corresponds to `r = -c(wall)`" are
  correct.** I re-read `gen_robin_cmat` (`asgard_coefficients_mats.hpp:390–394`)
  against the `none` branches (`:305–319`, `:336–350`): the blocks and signs are
  identical, `escale = 1/dx` in both, so free(`c`) + Robin(`-c`) cancels
  bit-exactly. Both worked examples check out numerically: ICRF_1D
  `c = -v^2 eta`, `r = +v_max^2 eta`; LHCD_2D `c = -0.5`, `r = +0.5`. (The
  *dissipativity* rule attached to these is wrong — B2 — but the equivalence is
  right.)
* **`term_grad`'s default flag really is `none`** (`asgard_pde.hpp:175`), so the
  `{div bothsides, grad default}` idiom is correctly identified.
* **"All four FENRIS solvers converged on `bothsides` on every outermost `div`,
  no Robin lines"** (l.849) is true: all live `set_*_robin` calls are gone from
  ICRF_1D, ICRF_2D, LHCD_1D and LHCD_2D; only comments remain.
* **The `flux_type::none == central` finding (P4)** reproduces: the enumerator
  has no initialiser and the dispatch at `asgard_coefficients_mats.hpp:123–128`
  has no `return`.
* **The two framings of the div/grad rule are genuinely equivalent** in the sense
  the brief asked about (the `flip` map *is* set complement on the wall sets, and
  `grad`'s `none` default makes `{div bothsides, grad default}` the sealed pair).
  The divergence is not in the mathematics but in the document's *wording* of the
  primary rule — see M1.
* **No finding at BLOCKER severity in `\S`verify, `\S`mass's mass-matrix
  algebra, or `\S`penalty.** The `M^{-1}`-per-link result, the `v^2`-cancellation
  argument, and the Nitsche reading of the penalty switch all check out against
  the source.
