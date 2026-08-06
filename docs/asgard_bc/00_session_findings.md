# Verified findings from the 2026-08-05 session (seed for the BC document)

All claims below were measured in this repo, not inferred. Toys live in
/private/tmp/claude-502/.../scratchpad/{bc_toy,bc_matrix,bc_grad}.
ASGarD headers: /Users/ahsan/venvs/asgardpy/include/asgard_*.hpp
Key file: asgard_coefficients_mats.hpp (gen_tri_cmat, gen_robin_cmat),
asgard_pde.hpp (boundary_type enum, term_robin, set_left/right_robin).

## Mechanism (read from gen_tri_cmat)
- Each term assembles a block-tridiagonal matrix: volume blocks + edge blocks.
- Interior edges always get blocks (upwind: c +/- |c|); they cancel pairwise in
  the mass balance, so interior transport conserves identically.
- At the two DOMAIN edges the flag decides:
    boundary_type::none      -> an edge block IS written from the interior trace
                                ("free"/outflow: flux = c * q_interior)
    boundary_type::bothsides -> "dirichlet flux, nothing to set": NO axpy.
                                The block is omitted = flux prescribed 0.
    left/right               -> same, one side only.
- gen_robin_cmat re-adds a boundary block: -r/dx * to_left on cell 0,
  +r/dx * to_right on the last cell. So a Robin argument is a FLUX coefficient
  (multiplies f at the wall), not a log-derivative.
- CRITICAL: for operation_type::grad the flag is FLIPPED before assembly
  (bothsides<->none, left<->right). This is the LDG alternation.

## Experimentally established (toy: dt f = d/dx(2x f + D1 f' + D2 f'), [0,2])
1. SUM RULE. Sealing a set S of terms enforces (sum of S's fluxes) -> 0 at the
   wall, ONE condition per wall, not one per term. Verified on all 8 sealing
   combinations of a 3-term equation:
     all sealed      -> f'/f = -A/(D1+D2): predicted -2.67, measured -2.60; dN/dt = 0
     seal {D1,D2}    -> f' = 0 (Neumann); leak +4 f(2) (predicted +4)
     seal {A,D1}     -> EMERGENT ROBIN f'/f = -A/D1: predicted -4, measured -3.69;
                        leak -1.98 vs predicted -2 (through open D2)
     seal {A,D2}     -> f'/f = -A/D2 predicted -8 measured -6.3 (under-resolved);
                        leak -7.75 vs predicted -8
     seal {D1} or {D2} alone -> both give f'=0 and +3.884 leak, IDENTICAL to 3
                        digits (sealing one f'-channel silences both)
2. Robin(r) with r = the drag term's wall flux coefficient is EXACTLY equivalent
   to sealing the drag term. Verified bit-identical (7e-15) in the toy, and
   bit-identical in ICRF_1D (t=200) and LHCD_2D production solvers.
3. Chain pairings (pure diffusion dt f = d/dx(f') on [0,2], exact steady f=const):
     {div bothsides, grad default} -> NEUMANN. flat steady state, walls at the
         constant, max dev 5.3e-4, mass error 2e-6.  THE consistent sealed pair.
     {div none,      grad bothsides} -> DIRICHLET f=0. decay rate measured 2.467
         vs exact ground mode (pi/2)^2 = 2.467.  THE consistent Dirichlet pair.
     {div bothsides, grad bothsides} -> ILL-POSED (Neumann AND Dirichlet on the
         same wall): f -> 1.4e6, mass error 8e2.
     {div bothsides, grad right/left} -> ill-posed on THAT WALL ONLY; the other
         wall stays exactly Neumann (0.2228 vs const 0.2215).  Local corruption.
     {div none, grad default} -> neither; upwind-asymmetric mongrel, left and
         right behave differently despite symmetric flags. Decay rate 0.77,
         profile not a Dirichlet mode (dev 5.6e2). Produced negative mass in
         3-term tests.
4. Penalty terms invert the convention: on term_penalty, bothsides ADDS
   boundary blocks (Dirichlet-style pin on f) rather than deleting them.

## Production consequences measured in this repo
- ICRF_2D originally: diffusion sealed, drag free, QL divergences free
  (bc::none). The uncancelled inward drag bracket MANUFACTURED particles:
  +63% by t=100 (level 5, x_max=6), worse under refinement.
- LHCD_2D originally: QL divergences free -> steady leak 0.0043%/tau, and a
  FALSE steady state (energy pinned, tail frozen) because RF input balanced
  wall loss.
- Final configuration in all four solvers: bothsides on every outermost div,
  no Robin lines. Verified bit-identical to the Robin construction where the
  Robin was correct; fixes LHCD_1D where no Robin existed.
