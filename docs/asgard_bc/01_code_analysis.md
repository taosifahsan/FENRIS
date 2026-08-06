# How ASGarD actually builds boundary conditions — a source-level walkthrough

Audience: a physicist who knows DG weak forms but has never read this codebase.

**Source of truth.** All line numbers refer to the installed ASGarD 0.9.1 tree

* headers: `/Users/ahsan/venvs/asgardpy/include/asgard_*.hpp`
* sources: `/Users/ahsan/venvs/asgardpy/src/asgard_*.cpp`

The headers in `include/` and `src/` are byte-identical apart from `assert`↔`expect`,
so header line numbers are valid for either path. The copy vendored inside FENRIS
(`/Users/ahsan/Desktop/FokkerPlanck/FENRIS/asgard/`) is a slightly older revision:
`asgard_coefficients_mats.hpp` is identical, but the `.cpp` line numbers are shifted
by roughly 10–20 lines. Every claim below was checked against the venv tree.

Anything I inferred rather than read is tagged **[INFERENCE]**. Anything I could not
settle from the source is tagged **[UNRESOLVED]**.

---

## 0. The one sign convention everything hangs on

Read this first or every sign below will look wrong.

The assembled term matrices are applied to the state with **alpha = −1**:

```cpp
// asgard_discretization.cpp:556  (in ode_rhs_base)
terms.apply(group, grid, conn, -1, in, 0, out);
```

(The implicit/Euler path does the same: `terms.apply(group, -term_scal.value, ...)`,
`asgard_discretization.cpp:630`.)

So if `A` is the matrix a term assembles, the ODE actually integrated is

    d f / d t  =  − A f  +  sources.

And, as shown in §1, `term_div(c)` assembles `A` = the DG weak form of `+∂ₓ(c f)`.
Therefore a `term_div` with coefficient `c` contributes

    ∂ₜ f = − ∂ₓ F ,        F = c · f      (F evaluated with the numerical trace)

i.e. **the coefficient you hand to `term_div` is the conservative flux coefficient,
and `F = c·f` is the thing that flows through the wall.** Total mass obeys

    dN/dt = − F(x_right) + F(x_left) = − c f |_right + c f |_left.

Cross-check against the shipped examples, which is how I pinned the sign:
`src/pde/sinwav.cpp` solves `∂ₜf + ∂ₓf = 0` (a *right*-moving wave, inflow on the
left) using `term_div(+1, upwind, boundary_type::left)` with a `left_boundary_flux`
— inflow pinned on the left, outflow free on the right, which is only correct if
`term_div(+1)` contributes `−∂ₓf`. Likewise `src/pde/continuity_2d.cpp` adds
`term_div(1)` per dimension for `∂ₜf + ∇·f = s` and then adds `+f_x, +f_y, +f_t` as
sources, which balances only with the `−1`.

---

## 1. One term → one matrix

### 1.1 The basis, and why 1/dx and 2/dx appear

`legendre_vals` builds Legendre polynomials on the reference interval `ξ ∈ [−1,1]`
and rescales `P_k → √(2k+1) P_k` (`asgard_quadrature.cpp:58-66`), so

    (1/2) ∫_{−1}^{1} P_i P_j dξ = δ_ij .

On a physical cell of width `dx` the basis functions are

    φ_j(x) = P_j(ξ) / √dx ,    ξ = 2(x − x_center)/dx ,

which makes `∫_cell φ_i φ_j dx = δ_ij` — the cell mass matrix is the identity, which
is why nothing ever inverts a mass matrix unless the user asked for one (§5).

Two scalings follow immediately, and they are exactly the two constants in the code:

```cpp
// asgard_coefficients_mats.hpp:187-188
P const escale = P{1} / dx; // edge scale
P const vscale = P{2} / dx; // volume scale
```

* **Edge**: a trace product `φ_r(edge)·φ_c(edge) = P_r(±1)P_c(±1)/dx`. The stored
  edge blocks hold only the reference numbers `P_r(±1)P_c(±1)`, so the physical
  block needs the factor `1/dx`. That is `escale`.
* **Volume**: `dφ_c/dx = (2/dx)·P′_c(ξ)/√dx`, and `∫_cell dx = (dx/2)∫dξ`. The array
  `legw` already carries the `(1/2)·quadrature-weight`, so
  `∫ φ_r′ φ_c dx = (2/dx) · [ (1/2)∫ P_r′ P_c dξ ]`. That is `vscale`.

Note there is no `√dx` left over anywhere: the operator is dimensionally
`1/length`, as `∂ₓ` should be.

### 1.2 The four edge blocks

```cpp
// asgard_transformations.cpp:55-58   (legendre_basis constructor)
smmat::gemm_outer_inc(pdof, lP_L.data(), lP_L.data(), to_left);
smmat::gemm_outer_inc(pdof, lP_L.data(), lP_R.data(), from_left);
smmat::gemm_outer_inc(pdof, lP_R.data(), lP_L.data(), from_right);
smmat::gemm_outer_inc(pdof, lP_R.data(), lP_R.data(), to_right);
```

`gemm_outer_inc(n, x, y, A)` does `A[c*n + r] += x[r]*y[c]`
(`asgard_small_mats.hpp:593-600`), and blocks are column-major with **row = test
index, column = trial index** (confirmed by `smmat::gemv`, `asgard_small_mats.hpp:209`).
`lP_L = P(−1)`, `lP_R = P(+1)`. So, with `r` = test, `c` = trial:

| block | entry (r,c) | meaning |
|---|---|---|
| `to_left`    | `P_r(−1) P_c(−1)` | my test trace at my LEFT edge × **my own** value there |
| `from_left`  | `P_r(−1) P_c(+1)` | my test trace at my LEFT edge × **left neighbour's** value there |
| `from_right` | `P_r(+1) P_c(−1)` | my test trace at my RIGHT edge × **right neighbour's** value there |
| `to_right`   | `P_r(+1) P_c(+1)` | my test trace at my RIGHT edge × **my own** value there |

Mnemonic: `to_*` = the trace comes from *this* cell (diagonal block); `from_*` = the
trace comes from the *neighbour* (off-diagonal block).

### 1.3 The block-tridiagonal container

`(A f)_i = lower(i)·f_{i−1} + diag(i)·f_i + upper(i)·f_{i+1}`, with a documented
wrap-around:

```cpp
// asgard_block_matrix.hpp:317-320
 * The three diagonals have the names lower, diag, upper.
 * The entry for lower(0) is actually the top-right entry (0, n-1) corresponding
 * to periodic boundary condition. Similarly upper(n-1) is the bottom left entry.
```

The storage slots for the wrap blocks always exist (the "full" 1-D connection
pattern always links left-most to right-most, `asgard_grid_1d.hpp:316-327`); for
non-periodic problems they are simply zero.

### 1.4 The interior of a `div`

```cpp
// asgard_coefficients_mats.hpp:239-249
P const left  = 0.5 * (... rhs at the left edge of cell i ...);
P const right = 0.5 * (... rhs at the right edge of cell i ...);
P const left_abs  = fscale * std::abs(left);
P const right_abs = fscale * std::abs(right);

smmat::axpy(nblock, escale * (-left - left_abs), basis.from_left,  coeff.lower(i));
smmat::axpy(nblock, escale * (-left + left_abs), basis.to_left,    coeff.diag(i));
smmat::axpy(nblock, escale * (right + right_abs), basis.to_right,  coeff.diag(i));
smmat::axpy(nblock, escale * (right - right_abs), basis.from_right, coeff.upper(i));
```

with `fscale = static_cast<int>(flux)` ∈ {+1 upwind, 0 central, −1 downwind}
(`:185`). The volume block is written separately by the `apply_volume` lambda
(`:210-222`) as `−c·∫φ_r′φ_c dx` (the `gemm_tn<-1>` is a subtract).

Putting the two together, row `r` of cell `i` is

    (A f)_r = − ∫ φ_r′ c f dx + φ_r(x_R)·(ĉf)_R − φ_r(x_L)·(ĉf)_L ,

which is the standard DG weak form of `+∂ₓ(c f)` with numerical flux `ĉf`. With
`flux_type::upwind` and `c>0` the coefficients collapse to `lower = −c/dx·from_left`
and `diag = +c/dx·to_right`: the trace is taken from the **left** at every edge.
Combined with the global `−1` of §0 that is genuine upwinding for `∂ₜf = −c∂ₓf`
(right-moving wave), and it makes the scheme dissipative: a short calculation gives
`f·(−A f) = −(c/2)Σ_edges [f]² < 0`.

Consequence that matters for BCs: **interior edges always contribute a matched pair
of blocks** (one `to_*` in cell `i`'s diagonal, one `from_*` in the neighbour's
off-diagonal). Testing against the constant function `φ_0`, the two members of each
pair cancel exactly. Interior transport therefore conserves mass to round-off, no
matter what the flags say. Only the two domain-edge blocks are unpaired, and that is
where all boundary physics lives.

---

## 2. What each `boundary_type` does at assembly time (div)

The whole boundary treatment is three `switch` statements inside a single
`#pragma omp single` block (`asgard_coefficients_mats.hpp:255-351`). Interior cells
`1 … N−2` are done in the parallel loop above; cells `0` and `N−1` are done here.

The flags themselves:

```cpp
// asgard_pde.hpp:109-121
enum class boundary_type
{
  periodic,   //! periodic boundary conditions
  left,       //! fixed flux on the left end of the boundary
  right,      //! fixed flux on the right end of the boundary
  bothsides,  //! fixed flux at both ends of the boundary
  none        //! do not fix the flux on either end of the domain
};
```

### 2.1 The left wall

```cpp
// asgard_coefficients_mats.hpp:305-319
switch (boundary) {
  case boundary_type::none:
  case boundary_type::right: // free on the left
    smmat::axpy(nblock, -escale * (c at xleft), basis.to_left, coeff.diag(0));
    break;
  case boundary_type::periodic: {
    P const left     = 0.5 * (c at xleft);
    P const left_abs = fscale * std::abs(left);
    smmat::axpy(nblock, escale * (-left - left_abs), basis.from_left, coeff.lower(0));
    smmat::axpy(nblock, escale * (-left + left_abs), basis.to_left,   coeff.diag(0));
    }
    break;
  default: // dirichlet flux, nothing to set
    break;
};
```

(I have elided the `(rtype == is_const) ? rhs_const : rhs_vals[0][0]` ternaries for
readability; they select constant vs. sampled coefficient and nothing else.)

### 2.2 The right wall (`:336-350`) is the mirror image

```cpp
  case boundary_type::none:
  case boundary_type::left: // free on the right
    smmat::axpy(nblock, escale * (c at xright), basis.to_right, coeff.diag(rmost));
```

### 2.3 Reading of each case

* **`none`** — an unpaired block `∓(c/dx)·to_*` is written into the boundary cell's
  own diagonal. Testing against `φ_0` this equals the flux `F = c·f_interior(wall)`.
  This is a *free / outflow / extrapolation* wall: the flux is whatever the interior
  trace says it is. Note two details: the coefficient is the **full** `c`, not `c/2`,
  and `|c|` never appears — upwinding is abandoned at the wall. So even if `c` points
  *inward* (a physically incoming characteristic, which would need data supplied) the
  code still uses the interior value and runs silently. **[INFERENCE]** that this is a
  latent well-posedness trap rather than intentional; the source has no guard.

* **`bothsides`** — the `default:` arm, commented `// dirichlet flux, nothing to set`.
  **No `axpy` is executed.** The block is omitted, which means the numerical flux at
  that wall is identically `F = 0`. This is the "seal". Nothing is added to the matrix;
  the condition is enforced by *absence*.

* **`left` / `right`** — same as `bothsides` but on one wall only. The flag names the
  wall where the flux **is** pinned: `boundary_type::left` falls into `default:` in the
  left switch (sealed) and into the free arm in the right switch (`case left:` at
  `:338`). This matches the doxygen "fixed flux on the left end".

* **`periodic`** — the ordinary *interior*-edge formula is written, but into
  `lower(0)` / `upper(rmost)`, which are the wrap-around blocks (§1.3). The domain
  edge becomes an ordinary interior edge joining cell `N−1` to cell `0`, upwinding
  and all. Mass is conserved by the same pairwise cancellation as any interior edge.

* **The mid-block at `:321-333`** — when `N > 1`, cell 0's *right* edge and cell
  `N−1`'s *left* edge are ordinary interior edges and are written unconditionally.
  Only the two outermost edges are governed by the switch. The `num_cells == 1`
  case is special-cased (`:263-264`, `:321`): the single cell is both left-most and
  right-most and the mid-block is skipped.

### 2.4 Where the coefficient is sampled

For a variable coefficient the sampling grid is `num_quad + 1` points per cell —
the cell's **left edge** plus the interior quadrature points (`:164-182`). The wall
values used above are `rhs_vals[0][0] = c(xleft)` and `rhs_raw.vals.back() = c(xright)`,
i.e. `c` evaluated exactly **at the domain endpoint**. Practical consequence for
FENRIS: with a spherical Jacobian, `c ∝ v²` vanishes at `v = 0`, so at the `v = 0`
wall `none` and `bothsides` assemble the *same* matrix — the seal is a no-op there.
The `v = v_max` wall is the one that matters.

---

## 3. Why the flag is flipped for `grad`

```cpp
// asgard_coefficients_mats.hpp:130-148
if constexpr (optype == operation_type::grad) {
  // the grad operation flips the fixed and free boundary conditions
  switch (boundary) {
    case boundary_type::bothsides: boundary = boundary_type::none;      break;
    case boundary_type::none:      boundary = boundary_type::bothsides; break;
    case boundary_type::right:     boundary = boundary_type::left;      break;
    case boundary_type::left:      boundary = boundary_type::right;     break;
    default: // periodic, do nothing since it is symmetric anyway
      break;
  }
}
```

and then, after the whole div-style matrix has been assembled,

```cpp
// asgard_coefficients_mats.hpp:355-366
if constexpr (optype == operation_type::grad)
{
  // take the negative transpose of div
  for (int64_t r = 0; r < coeff.nrows() - 1; r++) {
    smmat::neg_transp_swap(basis.pdof, coeff.lower(r + 1), coeff.upper(r));
    smmat::neg_transp(basis.pdof, coeff.diag(r));
  }
  smmat::neg_transp(basis.pdof, coeff.diag(coeff.nrows() - 1));
  smmat::neg_transp_swap(basis.pdof, coeff.lower(0), coeff.upper(coeff.nrows() - 1));
}
```

So `G = −Ã ᵀ`, where `Ã` is a div matrix built with the **flipped** flag. (The last
line transposes the periodic wrap pair, which is why the flip leaves `periodic`
alone.) `neg_transp` also automatically converts upwind into downwind, which is the
other half of the LDG alternation.

**What the flip accomplishes.** Work out one wall. `to_left` is symmetric, so if
`Ã` received the free block `−(c/dx)·to_left` in `diag(0)`, then `G = −Ãᵀ` receives
`+(c/dx)·to_left`. Expanding `G`'s volume part likewise gives `+c∫φ_r ∂ₓφ_c`. So

    (G f)_r = c ∫ φ_r ∂ₓ f dx + c φ_r(x_0) f(x_0) + … .

Compare with the textbook DG discretisation of `c ∂ₓ f` in "strong" form,

    ∫ φ_r c ∂ₓ f + φ_r(x_R) c (f̂_R − f_R⁻) − φ_r(x_L) c (f̂_L − f_L⁺) ,

and the extra term is exactly the left-wall correction with **f̂ = 0**. Therefore:

> For a `grad`, the presence of the boundary block means the numerical trace of `f`
> at that wall is pinned to **zero (Dirichlet)**; its absence means the trace is taken
> from the interior (no condition).

And because the flag was flipped, "block present" corresponds to the **user** writing
`bothsides`. The flip exists precisely so that the user-facing vocabulary stays
uniform across `div` and `grad`:

| user flag | on a `div` | on a `grad` |
|---|---|---|
| `bothsides` | impose: flux `F = c f = 0` (Neumann-like) | impose: trace `f = 0` (Dirichlet) |
| `none` | no condition, flux from interior trace | no condition, trace from interior |
| `left` / `right` | same, that wall only | same, that wall only |

In LDG language the flip implements the *alternating* choice of traces: the auxiliary
variable and the primal variable must take their traces from opposite sides / opposite
conventions, otherwise the pair `(div, grad)` is not a consistent discretisation of a
second derivative. ASGarD applies the same logic to the wall: exactly one of the two
factors is allowed to carry the boundary condition.

There is a second, separate flip you should know about, applied at `term_1d`
construction time:

```cpp
// asgard_pde.hpp:719-724   (private term_1d helper ctor)
if (optype_ == operation_type::grad) {
  if (flux_ == flux_type::upwind)        flux_ = flux_type::downwind;
  else if (flux_ == flux_type::downwind) flux_ = flux_type::upwind;
}
```

so a user who writes `flux_type::upwind` on *both* the div and the grad of a chain
gets stored fluxes that are opposite — the alternation — without having to think
about it. This is what the chain example in the docs does (`asgard_pde.hpp:314-315`).

---

## 4. How a `term_1d` chain `{div, grad}` becomes one operator

**Yes, it is a matrix product, formed before the solve, and the intermediate
quantity never exists as a field.**

The doxygen says so up front:

```cpp
// asgard_pde.hpp:384-386
 * The second mode is to represent a chain of simple terms.
 * The operators in the chain will be multiplied together using small-matrix
 * logic in a local cell-by-cell algorithm.
```

The implementation is `term_manager<P>::rebuld_chain` (`asgard_term_build.cpp:988-1092`).
It builds the **last** link first and multiplies leftwards:

```cpp
// asgard_term_build.cpp:1044-1074 (abridged)
build_raw_mat(tentry, d, num_chain - 1, level, hier, bmass, *diag0, *tri0);
for (int i = num_chain - 2; i > 0; i--) {
  build_raw_mat(tentry, d, i, level, hier, bmass, raw_diag, raw_tri);
  ...
  gemm_block_tri(basis.pdof, raw_tri, *tri0, *tri1);   // tri1 = A_i * accumulated
  std::swap(tri0, tri1);
}
build_raw_mat(tentry, d, 0, level, hier, bmass, *diag1, *tri1);
... gemm_block_tri(basis.pdof, *tri1, *tri0, raw_tri); // raw_tri = A_0 * (A_1 ... A_{n-1})
```

so for `term_1d chain{ div, grad }` the stored operator is

    A = A_div · A_grad ,     and the ODE contribution is  − A_div A_grad f.

`q = A_grad f` is a *virtual* vector: it is never assembled, never stored, never given
its own boundary condition. It exists only as the middle of a matrix product that was
collapsed at build time.

**This is the whole reason `div` and `grad` flags interact.** The wall condition is
not applied to a field `q`; it is baked into rows/columns of two matrices that are then
multiplied. Each factor can contribute *one* condition at each wall:

* `A_grad`'s boundary block (present ⇔ user wrote `bothsides`) pins `f`'s trace to 0.
* `A_div`'s boundary block (absent ⇔ user wrote `bothsides`) pins the flux `c·q` to 0.

A second-order operator on an interval needs exactly **one** condition per wall. Hence
the four observed regimes, which now follow mechanically:

| chain flags | conditions at each wall | result |
|---|---|---|
| `div bothsides`, `grad none` | flux `c q = 0`, trace free | **Neumann**, exactly one condition ✓ |
| `div none`, `grad bothsides` | flux free (interior `q`), trace `f = 0` | **Dirichlet**, exactly one condition ✓ |
| `div bothsides`, `grad bothsides` | flux = 0 **and** `f` = 0 | two conditions → over-determined → blow-up |
| `div none`, `grad none` | neither | under-determined; the wall behaviour is whatever the upwind bias happens to give, and is left/right asymmetric |

The measured "ill-posed on THAT WALL ONLY" for `{div bothsides, grad right}` is the
same statement restricted to one endpoint: the flip maps `right → left`, so the grad
writes its Dirichlet block only at one wall, and the other wall keeps its clean
Neumann pairing.

**A caveat about the product.** `gemm_block_tri` is documented and implemented to
compute only three bands:

```cpp
// asgard_block_matrix.hpp:544-547
 * The assumption is that the matrices are upper and lower tri-diagonal but it is unclear
 * which is which. Thus, the algorithm multiplies the matrices but ignores the entries
 * outside of the three diagonals.
```

(`asgard_block_matrix.cpp:537-582` never forms `A.lower·B.lower` or `A.upper·B.upper`.)
The product of two general tridiagonals is pentadiagonal; ASGarD relies on the two
factors being numerically *bidiagonal in opposite directions* (upwind div has only
`lower`+`diag`, downwind grad only `diag`+`upper`), so the true product really is
tridiagonal. `term_1d::check_chain` (`asgard_pde.hpp:730-745`) enforces "at most two
non-central fluxes, never a central mixed with a side flux", but it does **not** verify
that the two side fluxes point opposite ways, and it cannot know the sign of a variable
coefficient. **[INFERENCE]** If the pairing is wrong — or if the coefficient changes
sign inside the domain while an upwind flux is declared — the out-of-band entries are
silently dropped and you get a consistent-looking but wrong operator, with no warning.
This is a good thing for Agent 3 to probe.

---

## 5. `term_md`, the tensor product, and the mass matrix

### 5.1 Tensor product

A separable `term_md` holds one `term_1d` per dimension (`asgard_pde.hpp:1081-1096`).
Each dimension's `term_1d` (chain-collapsed as in §4) is converted from the
cell-local block form to the hierarchical-wavelet sparse form
(`hier.tri2hierarchical(...)`, `asgard_term_build.cpp:711-716`) and stored in
`term_entry::coeffs[d]` (`asgard_term_build.hpp:43`). Application is a Kronecker
product sweep:

```cpp
// asgard_term_manager.cpp:67-68
block_cpu(basis.pdof, grid, conns, tme.perm, tme.coeffs,
          al, in, be, out, kwork);
```

i.e. the multi-dimensional operator is `⊗_d A_d`, with `term_identity` dimensions
skipped (`kronmult::permutes` remaps to the active directions,
`asgard_kronmult.hpp:41-46`). The `conn_fill` upper/lower bookkeeping in `permutes`
is about the sparse-grid hierarchical connectivity, not about physics.

Two important consequences of the tensor structure for boundary conditions:

* A `term_md` may carry a flux in **at most one** dimension (`term_md::flux_dim()`,
  `asgard_pde.hpp:1264-1279`, returns the *first* dimension with a flux; the
  inhomogeneous-flux API asserts `flux_dim() != -1`, `:1285-1289`). Boundary
  conditions therefore live in exactly one direction per term, tensored with plain
  mass matrices in the others.
* Because it is a tensor product, sealing the 1-D factor seals the whole
  multi-dimensional term: the wall integral factorises as (1-D wall block) ⊗ (mass in
  the other dims), and setting the 1-D block to zero kills the whole thing.

A `term_md` of `mode::chain` is different: those links are *not* multiplied at build
time; they are applied sequentially at run time through a workspace
(`asgard_term_manager.cpp:103-118`). Only `term_1d` chains are collapsed into a single
matrix.

### 5.2 Where the mass enters

`pde_scheme::set_mass` (`asgard_pde.hpp:1580-1583`) sets a global, per-dimension,
positive, time-independent volume weight; `term_md::set_mass` (`:1229-1233`) sets a
per-term one. The mass matrix is a genuine Galerkin object, `M_{rc} = ∫ φ_r m(x) φ_c dx`
per cell (`gen_volume_mat`, `asgard_coefficients_mats.hpp:454-492`), Cholesky-factored
once (`asgard_block_matrix.cpp:212-233`) and then **inverted into every link of every
term**:

```cpp
// asgard_term_build.cpp:870-875   (end of build_raw_mat, runs for every chain link)
if (bmass) {
  if (t1d.is_diagonal() and not t1d.is_identity())
    bmass->solve(basis.pdof, raw_diag);
  else
    bmass->solve(basis.pdof, raw_tri);
}
```

`solve` left-multiplies each block row by `M_i^{-1}` (`asgard_block_matrix.cpp:283+`).
So the stored operator for a single term is `M^{-1}A`, and for a chain it is
`(M^{-1}A_0)(M^{-1}A_1)…` — the mass is inverted **once per link**, not once per term.

Combining with §0, the equation actually solved is

    m(x) · ∂ₜ f = − ∂ₓ ( c f )   for a single `term_div(c)`,

equivalently `∂ₜ(m f) = −∂ₓ(c f)`. **The conserved quantity is `∫ m f dx` and its flux
is `c f`.** That is why the coefficient handed to `term_div` is effectively
mass-weighted: to represent a physical flux `Γ` in a metric with weight `m`, you must
pass `c = m·Γ/f`, i.e. the Jacobian-weighted flux coefficient.

This is exactly what FENRIS does. From
`/Users/ahsan/Desktop/FokkerPlanck/FENRIS/ICRF_1D/src/ICRF_1D.cpp`:

```cpp
257:  pde.set_mass({term_volume(v2)});                      // m = v^2
271:  pde += term_1d({term_div(eta_v2, boundary_type::bothsides)});   // c = eta * v^2
284:  term_1d div_grad({
285:      term_div(zeta_v2, boundary_type::bothsides),      // c = zeta * v^2
286:      term_grad(v2, boundary_type::none)                // and v^2 again on the grad
287:  });
```

The `v²` on the *grad* is the same story one level down: because `M^{-1}` is applied
per link, `M^{-1}A_grad ≈ (1/v²)(v² ∂_v) = ∂_v`. The user writes `v²` on both factors
and the inverse mass cancels one of them in each.

And the flag pairing is the canonical Neumann pair from §4: `div bothsides` +
`grad none`. What gets sealed is `v²Γ` — the mass-weighted flux, which is exactly the
quantity whose wall value controls `d/dt ∫ v² f dv`.

---

## 6. `gen_robin_cmat` — the Robin argument is a flux coefficient

The entire function body that touches the matrix is four lines:

```cpp
// asgard_coefficients_mats.hpp:390-394
if (robin_left != 0)
  smmat::axpy(n2, -robin_left / dx, basis.to_left, coeff[0]);

if (robin_right != 0)
  smmat::axpy(n2, robin_right / dx, basis.to_right, coeff[num_cells - 1]);
```

Put this next to the `boundary_type::none` branches of §2:

```
free left  wall  (div):   axpy(-escale * c,  to_left,  diag(0))          // :308
robin left       :        axpy(-robin_left/dx, to_left, coeff[0])        // :391

free right wall (div):    axpy( escale * c,  to_right, diag(rmost))      // :339
robin right      :        axpy( robin_right/dx, to_right, coeff[nc-1])   // :394
```

They are **the same block with the same sign and the same `1/dx`**, differing only in
which number multiplies it. Therefore:

> `gen_robin_cmat(r_left, r_right)` re-installs exactly the "free/outflow" boundary
> block that `boundary_type::none` would have written, with a user-chosen coefficient
> in place of the term's own `c`.

So the Robin argument has the **units and meaning of a flux coefficient** — it is the
`c` in `F = c·f(wall)`, i.e. an effective velocity — **not** a log-derivative `f′/f`
and not a ratio of diffusion to advection. Setting `r` equal to the wall value of a
sealed `div`'s coefficient restores that term's wall flux exactly, which is why the
Robin-vs-seal equivalence measured in the session was bit-identical rather than merely
close: the two constructions produce literally the same floating-point `axpy`.

Assembly context (`asgard_term_build.cpp:1110-1120`): the Robin block is added to the
chain product **after** the multiplication, so it is *not* composed with the other
links. It acts directly on `f`. Standalone `term_robin` goes through
`build_raw_mat`'s `case operation_type::robin:` (`:861-864`) and writes into a
*diagonal* matrix; `term_1d::set_left_robin/set_right_robin` (`asgard_pde.hpp:694-704`)
attach one to an existing chain and require the term to be a chain. The doxygen
(`asgard_pde.hpp:286-291`) says a standalone Robin `term_md` "should have the same form
as the div-grad, but with the robin term in place of the div-grad" — i.e. it must carry
the same mass factors in the other dimensions, otherwise the tensor structure does not
match.

Sign check: with `robin_left = r`, the left-wall block is `−(r/dx)·to_left`, identical
to a free wall with `c = r`, giving `F(x_left) = r·f(x_left)` and (via §0)
`dN/dt |_left = + r f(x_left)`.

---

## 7. `term_penalty` — why `bothsides` means the opposite

Two structural differences from `div`/`grad`:

1. **No volume integral.** `apply_volume` is a no-op for penalty
   (`asgard_coefficients_mats.hpp:212`, `// the penalty term does not include a volume
   integral`), and the constant-coefficient precompute is skipped (`:191`). Also, a
   penalty coefficient may not be spatially varying (`static_assert` at `:120-121`).

2. **The interior blocks are a pure jump form**, not an upwind split:

```cpp
// asgard_coefficients_mats.hpp:231-235
smmat::axpy(nblock, -escale * rhs_const, basis.from_left,  coeff.lower(i));
smmat::axpy(nblock,  escale * rhs_const, basis.to_left,    coeff.diag(i));
smmat::axpy(nblock,  escale * rhs_const, basis.to_right,   coeff.diag(i));
smmat::axpy(nblock, -escale * rhs_const, basis.from_right, coeff.upper(i));
```

Assembled over an edge this is `p·[[φ]]·[[f]]`, so `fᵀPf = +p Σ_edges [f]² ≥ 0`, and
with the global `−1` of §0 the contribution to `∂ₜf` is dissipative. Algebraically it
is exactly the "upwinding part" of a div: `div_upwind(c) = div_central(c) +
(|c|/2)·penalty_form`.

Now the boundary switch:

```cpp
// asgard_coefficients_mats.hpp:268-279
switch (boundary) {
  case boundary_type::bothsides:
  case boundary_type::left: // dirichelt on the left
    smmat::axpy(nblock, escale * rhs_const, basis.to_left, coeff.diag(0));
    break;
  case boundary_type::periodic:
    smmat::axpy(nblock, -escale * rhs_const, basis.from_left, coeff.lower(0));
    smmat::axpy(nblock,  escale * rhs_const, basis.to_left,   coeff.diag(0));
    break;
  default: // free flux, no penalty applied
    break;
};
```

`bothsides` is now in the arm that **adds** a block; `none` is the `default:` that adds
nothing. Compare with §2 where `bothsides` was the `default:` that added nothing.

The apparent inversion dissolves once you notice that the added block is
`+(p/dx)·to_left`, i.e. `p·φ_r(x_0)·(f(x_0) − 0)` — a one-sided jump against a
*zero exterior state*. That is a Nitsche-style penalty enforcement of **f = 0** at the
wall, not a flux condition. So:

> The user-facing meaning is unchanged — `bothsides` still means "impose a condition
> at both walls". What differs is *which* condition (Dirichlet `f = 0` for a penalty,
> zero flux for a div) and therefore whether it is implemented by adding a block or by
> omitting one.

Note the penalty flag is **not** flipped: the flip at `:130` is guarded by
`optype == operation_type::grad`, and a penalty is its own `operation_type`.

Two assembly details that follow from this and are worth flagging:

* `term_1d::set_penalty(p)` on a `div` or `grad` adds a penalty matrix in
  `data_mode::increment` **using the same boundary flag as the parent**
  (`asgard_term_build.cpp:838-841` for div, `:851-854` for grad). For a
  `grad(bothsides)` that is coherent: the grad half flips to `none` and writes its
  Dirichlet block, the penalty half sees `bothsides` and adds its Dirichlet pin —
  both impose `f = 0`. For a **`div(bothsides)` + penalty**, however, the div omits its
  flux block (Neumann) while the penalty adds a Dirichlet pin — two conditions on the
  same wall. **[INFERENCE]** this combination is over-determined in the same way as
  `{div bothsides, grad bothsides}`; I read the dispatch directly but did not test the
  consequence.
* A penalty attached to a whole *chain* takes its flux and boundary flag from
  `chain_.back()` — the last link, normally the `grad`
  (`asgard_term_build.cpp:1095-1107`; the `rassert` message at `asgard_pde.hpp:671-672`
  says so explicitly). For the canonical Dirichlet pair `{div none, grad bothsides}`
  the penalty inherits `bothsides` and pins `f = 0`, which is consistent. For the
  Neumann pair `{div bothsides, grad none}` it inherits `none` and adds no wall
  penalty, also consistent. The design is coherent for the two canonical pairings.

---

## 8. Things the session notes do not yet cover

### 8.1 `flux_type::none` is numerically identical to `flux_type::central` — it does not work

```cpp
// asgard_pde.hpp:65-75
enum class flux_type
{
  upwind   = 1,
  central  = 0,
  downwind = -1,
  //! (experimental) div/grad term but without the edge fluxes, used for adding artificial viscosity
  none
};
```

`none` has no initialiser, so by the C++ rule it takes `downwind + 1 = 0` — **the same
value as `central`**. They are indistinguishable enumerators. (Verified by compiling
this exact enum: `none=0`, `central=0`, `none == central` is `true`.) Two consequences,
both read directly:

* `fscale = static_cast<int>(flux)` is `0` for `none`, i.e. a central flux.
* The dispatch at the top of `gen_tri_cmat`

  ```cpp
  // asgard_coefficients_mats.hpp:123-128
  if constexpr (optype == operation_type::div or optype == operation_type::grad) {
    if (flux == flux_type::none) {
      gen_no_flux_cmat<P, optype, rtype, dmode>(basis, xleft, xright, level,
                                                rhs, rhs_const, rhs_raw, coeff);
    }
  }
  ```

  has **no `return`**. Execution falls through to `coeff.resize_and_zero(...)` at `:155`
  and the full central-flux matrix is assembled on top. So the volume-only matrix is
  computed and thrown away. Also, because the test fires for `central` as well, every
  central-flux div/grad pays for a wasted `gen_no_flux_cmat` call.

  In `data_mode::replace` (the only mode reachable for div/grad —
  `data_mode::increment` is used only for penalty, `asgard_term_build.cpp:839,852,1104`,
  and penalty is excluded by the `if constexpr`) the stale data is zeroed, so this is
  waste, not corruption. **[INFERENCE]** on the conclusion, but a strong one: the
  `no_flux` path is currently dead, and asking for `flux_type::none` silently gives you
  a central flux with full edge terms — including a *central-flux* boundary block, which
  is not what "no edge fluxes" would mean. No FENRIS solver uses `flux_type::none`
  (all four use `upwind` or the default), so nothing here is currently affected; it is a
  trap for future work.

### 8.2 Inhomogeneous boundary fluxes are a **source vector**, not a matrix change

`left_boundary_flux` / `right_boundary_flux` / `sym_boundary_flux`
(`asgard_pde.hpp:861-961`) are attached to a `term_md` with `+=`
(`:1281-1292`). They require the term to have a flux dimension and the supplied
separable function to be **constant in that dimension** (`:1288-1289`) — you prescribe
a wall *value*, not a wall profile.

The construction is at `asgard_term_build.cpp:890-931`:

```cpp
P scale = P{1} / std::sqrt( (xright[d] - xleft[d]) / num_cells );   // = 1/sqrt(dx)
if (t1d.is_penalty()) scale = -scale;   // penalty flips the sign
...
smmat::axpy(pdof, - rhs_left * scale * fc, basis.leg_left, bentry.consts[d].data());
...
smmat::axpy(pdof,  rhs_right * scale * fc, basis.leg_right,
            bentry.consts[d].data() + num_entries - pdof);
```

This is precisely the omitted boundary block of §2 with the unknown trace `f(wall)`
replaced by the prescribed value: the free-wall block `−(c/dx)·to_left·f` acting on a
prescribed value `g` gives `−c·g·P_L/√dx`, which is what the `axpy` writes. It is then
added to the ODE right-hand side with the **negated** weight, matching the `−1` on the
terms:

```cpp
// asgard_term_sources.cpp:222, 237-239
sweights.push_back(-alpha);
...
rechain(bc, P{-1}, y);
```

So the recipe is: `boundary_type::left` (seal the left wall) **plus** a
`left_boundary_flux` **equals** an inhomogeneous prescribed flux there. Sealing alone
is the homogeneous special case. `sinwav.cpp` is the worked example.

Chain handling: `boundary_flux::chain_level(d)` says which link of the `term_1d` chain
the condition attaches to, and the vector is then pushed through the **outer** links by
multiplying with their raw matrices:

```cpp
// asgard_term_build.cpp:881-887
if (bentry.flux.chain_level(d) > clink) {
  if (t1d.is_diagonal()) raw_diag.inplace_gemv(basis.pdof, bentry.consts[d], t1);
  else                   raw_tri.inplace_gemv(basis.pdof, bentry.consts[d], t1);
}
```

Since links are built from `num_chain-1` down to `0`, "clink smaller" means "further
left in the product", so a Dirichlet value injected at the `grad` (inner) level is
subsequently multiplied by the `div` matrix — exactly the propagation you would do by
hand in LDG. `chain_level` defaults to `-1` (`asgard_pde.hpp:870`), i.e. attach at the
outermost link.

For `term_md` chains the analogous propagation happens at apply time in the `rechain`
lambda (`asgard_term_sources.cpp:29-45`).

### 8.3 Periodic

* Written into the wrap blocks `lower(0)` / `upper(N-1)`, using the *interior* upwind
  formula, so a periodic wall is a genuine interior edge and conserves by the same
  pairwise cancellation.
* The storage slots always exist (`asgard_block_matrix.hpp:317-320`), and the sparse
  "full" 1-D connection pattern always connects left-most to right-most
  (`asgard_grid_1d.hpp:316-327`), so a periodic term costs nothing extra in pattern
  terms but every tri-diagonal term pays for the two extra slots.
* The `grad` flip deliberately skips `periodic` ("do nothing since it is symmetric
  anyway", `:145`), and the wrap pair is transposed by the final
  `neg_transp_swap(coeff.lower(0), coeff.upper(nrows-1))` at `:365`.
* **[UNRESOLVED]** I did not verify what happens if one term in a `term_md` declares
  `periodic` in a dimension and another declares `bothsides` in the same dimension.
  Nothing in `gen_tri_cmat` or `term_manager` cross-checks flags between terms.

### 8.4 Adaptivity

```cpp
// asgard_term_build.cpp:645-651
int level = grid.current_level(d);           // required level
if (t1d.change() == changes_with::none)
  level = max_level;                         // build up to the max
```

* Time-independent terms (`changes_with::none`, which covers every constant- or
  function-coefficient div/grad) are assembled **once at `max_level`**, transformed to
  the hierarchical wavelet basis (`tri2hierarchical`, `asgard_term_build.cpp:716`),
  and never rebuilt. Adaptation then changes only which hierarchical basis functions
  are active; the operator seen by the solver is the Galerkin **restriction** of the
  fixed max-level operator to the active subspace.
* **[INFERENCE]** This is why sealing survives refinement exactly. The level-0 constant
  basis function is always in the sparse grid, so the row that expresses global mass
  balance is always present, and a block that was absent at max level is absent in every
  restriction. Conservation is a property of the max-level operator, inherited by all
  coarser active sets.
* Time-dependent terms (`changes_with::time` — moment-coupled, electric-field,
  Lenard-Bernstein) are rebuilt at `grid.current_level(d)`, so their `escale = 1/dx`
  and their wall blocks change as the grid adapts. The wall *location* and the
  seal/no-seal structure are unchanged, but the discrete operator is not the restriction
  of a single fixed matrix. **[INFERENCE]** on the practical consequence; I read the
  branch directly.
* Mass matrices follow the same pattern with a `lmass` fallback built at the current
  level (`asgard_term_build.cpp:679-691`).

### 8.5 Smaller things worth knowing

* **The "sum rule" has a trivial mechanism.** All terms are accumulated into one RHS
  (`asgard_term_manager.cpp:103-120`, the `b = 1` at `:120` switches from overwrite to
  accumulate). Each *open* term deposits its own unpaired wall block; each *sealed*
  term deposits nothing. Testing the assembled RHS against `φ_0` gives
  `dN/dt = −Σ_{open terms} c_t f(wall)`. There is no per-term boundary condition
  anywhere in the code — there is only one flux budget per wall, and the flags choose
  which terms are in it. That is the sum rule.
* **Why sealing one diffusion channel silences both.** **[INFERENCE]** In the toy,
  both diffusion terms are chains whose `grad` factor produces the same discrete
  `q ≈ ∂ₓ f`. Their wall fluxes are `D₁·q_wall` and `D₂·q_wall` — the *same* number
  times different constants. Any condition forcing one to vanish forces `q_wall = 0`
  and hence kills the other. This predicts the measured identical-to-3-digits behaviour
  of "seal {D1}" and "seal {D2}", with the residual leak being the still-open drag,
  `4·f(2) ≈ 3.884`.
* **`term_1d` chain hygiene.** Identity links are stripped, a single-non-identity chain
  collapses to that term, and chains of chains throw
  (`asgard_pde.hpp:524-561`). So `{div, identity, grad}` and `{div, grad}` build the
  same matrix.
* **Volume/`identity` terms never touch boundaries.** `term_1d::is_diagonal()`
  (`:679-682`) is true for everything except div/grad/penalty, and diagonal matrices
  have no edge blocks at all.
* **`gen_no_flux_cmat` takes no `boundary` argument** (`:20-22`) — consistent with its
  intent of having no edge terms at all, and moot given §8.1.
* **Nothing validates flag combinations.** There is no check anywhere that a
  `{div, grad}` chain carries exactly one `bothsides`. `{div bothsides, grad bothsides}`
  compiles, builds, runs, and blows up numerically. `{div none, grad none}` compiles,
  builds, runs, and quietly produces the wrong physics. This is the single biggest
  usability hazard in the boundary API and is worth stating loudly in the final
  document.

---

## Appendix: a one-page cheat sheet

| you write | assembly effect | physical condition |
|---|---|---|
| `term_div(c, none)` | write `∓(c/dx)·to_*` in the boundary cell's diagonal | free: `F = c·f_interior(wall)` |
| `term_div(c, bothsides)` | write nothing | sealed: `F = 0` at both walls |
| `term_div(c, left)` | write nothing at left, free block at right | `F = 0` at the left wall only |
| `term_div(c, periodic)` | interior formula into the wrap blocks | wall becomes an interior edge |
| `term_grad(c, none)` | flag flips to `bothsides` → write nothing → after `−transpose`, no wall correction | trace of `f` free |
| `term_grad(c, bothsides)` | flag flips to `none` → block written → after `−transpose`, `+c φ_r(wall)f(wall)` | trace of `f` pinned to 0 (Dirichlet) |
| `term_penalty(p, bothsides)` | **add** `+(p/dx)·to_*` | Nitsche pin `f = 0` |
| `term_penalty(p, none)` | add nothing | free |
| `term_robin(r_l, r_r)` | add `−(r_l/dx)to_left`, `+(r_r/dx)to_right` | reinstate a free wall with flux coefficient `r` |
| `+= left_boundary_flux(g)` | build `−c·g·P_L/√dx` as a source, added with weight `−1` | inhomogeneous prescribed wall flux |

Canonical second-order pairings:

* Neumann (zero flux): `{ term_div(c, bothsides), term_grad(D, none) }`
* Dirichlet (`f = 0`): `{ term_div(c, none), term_grad(D, bothsides) }`
* Anything else: over- or under-determined.
