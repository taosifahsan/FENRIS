"""ICRF quasilinear and collisional coefficient plots.

ICRF's coefficients are *not* computed in Python.  They are built once by
``src/build_tables.cpp`` (a compiled stage in the CMake build) and written to
``output_data/tables/*.bin`` as flat float64 dumps.  This module reads those
tables back, reconstructs the physical coefficients from them, and plots them.

(LHCD's equivalent file computes its diffusion coefficient analytically from
the input deck instead -- see ``LHCD_2D/plot/coefficients.py``.  Same role,
opposite implementation: table lookup versus closed form.)

Cartesian ``(v_parallel, v_perp)`` view only, signed-log color only -- see
``plot_common/static.py`` for why.  These figures never read a snapshot, so
they take no :class:`~plot_common.reader.SnapshotCache` and cost nothing to
compute in parallel with the expensive stages.

What lives here, in dependency order:

1. **Table layout** -- what the four ``.bin`` files contain and how to index
   them.  The raw byte reading is :func:`plot_common.reader.read_binary_array`;
   this module supplies the shape.
2. **Collisional physics from the deck** -- per-species coefficient arrays and
   the analytic collisional equilibrium, which the initial condition needs.
3. **Bounce-average measures** -- ``lambda(theta_0)``, the orbit weight that
   makes velocity-space marginals integrate to conserved particle number.
4. **The coefficients themselves** -- ``ql_coefficients`` and
   ``collision_coefficients``, derived from the tables.
5. **Figures** -- the 2x2 quasilinear panel, the 3-panel collisional panel, and
   the orbit-factor line plot.

Used by:
  - ``ICRF_2D/plot/diagnostics.py`` -- the initial-condition overlay and
    bounce-orbit weight
  - ``tools/run.sh``                -- all three figures

Depends on: :mod:`plot_common.reader` (deck + binary input),
:mod:`plot_common.static` (all drawing).  Its own Gauss-Legendre quadrature
(mirroring the solver's own ``src/GL.hpp``) is local to this file -- nothing
else in this project needs it.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

# Locate the directory that holds plot_common by walking up from this file,
# instead of hardcoding a parent depth -- so reorganizing the project tree
# (as when everything moved into FENRIS/) cannot silently break the import.
for _parent in Path(__file__).resolve().parents:
    if (_parent / "plot_common").is_dir():
        sys.path.insert(0, str(_parent))
        break
else:
    raise ImportError("plot_common not found above " + __file__)

from plot_common.runtime import bootstrap

PATHS = bootstrap(__file__)

import matplotlib.pyplot as plt
import numpy as np

from plot_common.reader import option_vector, read_binary_array, read_options
from plot_common.static import contour2d, line1d, save_png

# Electron mass in atomic mass units -- sets the electron/ion mass ratio in
# the collision operator.
ELECTRON_MASS_AMU = 5.4461702149e-4
# 2/sqrt(pi): the normalization of the error-function derivative, which
# appears throughout the collisional coefficients.
GAUSS_FACT = 2.0 / math.sqrt(math.pi)
# Coefficient figures are resampled onto this many points per axis,
# independent of the tables' own resolution -- the tables are built coarse
# for speed, but contour plots need a fine grid to look smooth.
COEFF_PLOT_POINTS = 1024

# Panel names and their LaTeX titles.  ``<X_ql>`` denotes a bounce average.
QL_TABLES = (
    ("B", r"$\langle B_{\mathrm{ql}}\rangle$"),
    ("C", r"$\langle C_{\mathrm{ql}}\rangle$"),
    ("E", r"$\langle E_{\mathrm{ql}}\rangle$"),
    ("F", r"$\langle F_{\mathrm{ql}}\rangle$"),
)

COLLISION_TABLES = (
    ("A", r"$\langle A_{\mathrm{c}}\rangle$"),
    ("B", r"$\langle B_{\mathrm{c}}\rangle$"),
    ("F", r"$\langle F_{\mathrm{c}}\rangle$"),
)


# ---------------------------------------------------------------------------
# Section 1: table layout
# ---------------------------------------------------------------------------
#
# ``build_tables.cpp`` writes four headerless files into output_data/tables/:
#
#   parameters.bin  12 doubles describing the grid and the plasma species
#   res_tab.bin     the 2-D resonance table, row-major (nv, npitch)
#   L_tab.bin       the 1-D orbit-time table, length npitch
#   I_tab.bin       the 1-D pitch-scattering table, length npitch
#
# Because the files carry no shape information, parameters.bin must be read
# first -- it is what tells us how to reshape the others.


def table_parameters(table_dir=None):
    """Read ``parameters.bin`` into a named dictionary.

    The file is exactly 12 float64 values in a fixed order, matching the
    write order in ``build_tables.cpp``.  The size is asserted rather than
    trusted: a mismatch means the tables were built by a different version of
    the generator, and silently misreading them would produce
    plausible-looking but wrong coefficients everywhere downstream.

    Fields:
      ``nv``, ``npitch``     -- table grid dimensions
      ``xmin``/``xmax``      -- normalized-speed range (x = v / v_thermal)
      ``pamin``/``pamax``    -- pitch-angle range in radians
      ``omega``              -- RF wave angular frequency
      ``Omega_s0``           -- species cyclotron frequency at the magnetic axis
      ``z_a``, ``m_a``, ``T_a`` -- charge, mass (amu), temperature of the
                                resonant species
      ``logLambda_aa``       -- Coulomb logarithm for self-collisions
    """
    table_dir = table_dir or PATHS.tables
    values = read_binary_array(table_dir, "parameters.bin")
    if values.size != 12:
        raise ValueError("parameters.bin must contain 12 doubles; regenerate tables")
    return {
        "nv": int(round(values[0])),
        "npitch": int(round(values[1])),
        "xmin": values[2],
        "xmax": values[3],
        "pamin": values[4],
        "pamax": values[5],
        "omega": values[6],
        "Omega_s0": values[7],
        "z_a": values[8],
        "m_a": values[9],
        "T_a": values[10],
        "logLambda_aa": values[11],
    }


def table_grids(table_dir=None):
    """Return the tables' **native** ``(x_grid, pitch_grid)`` coordinates.

    These are the coordinates the stored values actually sit on, so they are
    what any interpolation must treat as the source grid.  Both are uniform
    by construction in ``build_tables.cpp``, hence ``linspace``.
    """
    params = table_parameters(table_dir)
    x_grid = np.linspace(params["xmin"], params["xmax"], params["nv"])
    pitch_grid = np.linspace(params["pamin"], params["pamax"], params["npitch"])
    return x_grid, pitch_grid


def plot_grids(table_dir=None, n=COEFF_PLOT_POINTS):
    """Return a fine ``(x_grid, pitch_grid)`` for plotting.

    Deliberately independent of the tables' native resolution: the tables
    are built as coarsely as accuracy allows (generating them is expensive),
    while contour plots need a fine grid or the level curves look faceted.
    The gap is bridged by the interpolators below.
    """
    params = table_parameters(table_dir)
    x_grid = np.linspace(params["xmin"], params["xmax"], n)
    pitch_grid = np.linspace(params["pamin"], params["pamax"], n)
    return x_grid, pitch_grid


def interp_1d_table(table_dir, name, x_new):
    """Read a 1-D table and linearly interpolate it onto ``x_new``.

    Used for ``L_tab.bin`` and ``I_tab.bin``, both of which are functions of
    pitch angle only.
    """
    _, pitch_old = table_grids(table_dir)
    values = read_binary_array(table_dir, name)
    return np.interp(x_new, pitch_old, values)


def interp_2d_table(table_dir, name, x_new, pitch_new):
    """Read a 2-D table and bilinearly interpolate it onto a new grid.

    The stored array is flat, so it is reshaped to ``(nv, npitch)`` using the
    native dimensions -- row-major, matching the C++ write order.

    Fast path: when the requested grid *is* the native grid, return the
    values untouched.  This matters because the boundary-flux diagnostic
    evaluates coefficients at native resolution and would otherwise pay for a
    no-op interpolation on every snapshot.

    Otherwise interpolate separably -- first along ``x`` for every native
    pitch column, then along pitch for every new ``x`` row.  Two 1-D passes
    are used rather than a true 2-D interpolator because ``np.interp`` is
    C-speed and the intermediate array is small.
    """
    x_old, pitch_old = table_grids(table_dir)
    values = read_binary_array(table_dir, name).reshape(len(x_old), len(pitch_old))

    if np.array_equal(x_new, x_old) and np.array_equal(pitch_new, pitch_old):
        return values

    # Pass 1: resample along the velocity axis, keeping native pitch columns.
    by_x = np.empty((len(x_new), len(pitch_old)), dtype=float)
    for j in range(len(pitch_old)):
        by_x[:, j] = np.interp(x_new, x_old, values[:, j])

    # Pass 2: resample each row along the pitch axis.
    out = np.empty((len(x_new), len(pitch_new)), dtype=float)
    for i in range(len(x_new)):
        out[i, :] = np.interp(pitch_new, pitch_old, by_x[i, :])
    return out


def style_cartesian_axes(ax):
    """Label axes for the Cartesian velocity-space view.

    ``set_aspect("equal")`` is essential here: velocity space is isotropic,
    so an unequal aspect ratio would distort the resonance geometry the plot
    exists to show.
    """
    ax.set_xlabel(r"$v_\parallel/v_{ta}$", fontsize=13)
    ax.set_ylabel(r"$v_\perp/v_{ta}$", fontsize=13)
    ax.set_aspect("equal")


def velocity_mesh(table_dir=None):
    """Return ``(x_grid, pitch_grid, v_parallel, v_perp)`` meshes.

    Converts the polar plotting grid into Cartesian velocity components.
    ``indexing="ij"`` keeps the array layout ``(velocity, pitch)``, matching
    how the coefficient arrays are built.
    """
    x_grid, pitch_grid = plot_grids(table_dir)
    X, TH = np.meshgrid(x_grid, pitch_grid, indexing="ij")
    return x_grid, pitch_grid, X * np.cos(TH), X * np.sin(TH)


# ---------------------------------------------------------------------------
# Section 2: collisional physics from the input deck
# ---------------------------------------------------------------------------


def psi_collision(x):
    """Chandrasekhar-like function governing collisional drag and diffusion.

    Defined as ``(erf(x) - x*erf'(x)) / (2*x^2)``.

    The small-``x`` branch is a series expansion, not an optimization: the
    closed form evaluates ``0/0`` as ``x -> 0`` and loses all precision to
    catastrophic cancellation well before that (``erf(x)`` and ``x*erf'(x)``
    agree to leading order).  Below ``x = 1e-3`` the two-term Taylor series
    ``x * (2/sqrt(pi)) * (1/3 - x^2/5)`` is both exact to double precision and
    numerically stable.
    """
    if abs(x) < 1.0e-3:
        return x * GAUSS_FACT * (1.0 / 3.0 - x * x / 5.0)
    derf = GAUSS_FACT * math.exp(-x * x)
    return (math.erf(x) - x * derf) / (2.0 * x * x)


def collision_arrays(solver_input=None, table_dir=None):
    """Build per-background-species collision coefficient arrays.

    Returns ``(C, ell, mu)``, each an array with one entry per background
    species, electrons first:

      ``C``    collision-strength prefactor, normalized to self-collisions of
               the resonant species (so ``C`` is dimensionless and O(1))
      ``ell``  ratio of thermal speeds, ``sqrt(mu * T_a / T_species)`` -- the
               argument scaling for :func:`psi_collision`
      ``mu``   mass ratio ``m_species / m_a``

    Densities are normalized to sum to one, so ``C`` measures *relative*
    collisionality between species and the absolute density scale drops out
    (it is carried by the solver's time normalization instead).

    Electrons are handled separately from the ion loop because their density
    follows from quasineutrality (``n_e = sum(n_i z_i)``) rather than being
    specified in the deck, and their mass comes from a physical constant
    rather than a deck entry.
    """
    solver_input = solver_input or PATHS.solver_input
    options = read_options(solver_input)
    params = table_parameters(table_dir)
    # Per-species background properties, all vectors of equal length.
    z_bg = option_vector(options, "z_bg")
    m_bg = option_vector(options, "m_bg")
    n_bg = option_vector(options, "n_bg")
    T_bg = option_vector(options, "T_bg")
    log_bg = option_vector(options, "logLambda_bg")
    # Resonant-species properties come from the tables, not the deck, so that
    # they cannot disagree with what build_tables used.
    z_a = params["z_a"]
    m_a = params["m_a"]
    T_a = params["T_a"]
    log_aa = params["logLambda_aa"]
    T_e = float(options["T_e"])
    log_ea = float(options["logLambda_ea"])

    # Normalize densities: only ratios matter here.
    n_sum = np.sum(n_bg)
    n_bg = n_bg / n_sum

    # Quasineutrality fixes the electron density.
    n_e = np.sum(n_bg * z_bg)
    # Common denominator normalizing every species to a-a self-collisions.
    denom = z_a * z_a * log_aa

    C = [n_e * log_ea / denom]
    mu = [ELECTRON_MASS_AMU / m_a]
    ell = [math.sqrt(mu[0] * T_a / T_e)]
    for z, m, n, T, log_l in zip(z_bg, m_bg, n_bg, T_bg, log_bg):
        C.append(n * z * z * log_l / denom)
        mu.append(m / m_a)
        ell.append(math.sqrt(mu[-1] * T_a / T))
    return np.array(C), np.array(ell), np.array(mu)


# ---------------------------------------------------------------------------
# Gauss-Legendre quadrature
# ---------------------------------------------------------------------------
#
# Used only here, for the collisional-equilibrium integrals below.  Mirrors
# the solver's own ``src/GL.hpp`` so Python-side normalizations match what the
# C++ solver computed.

# Cache of quadrature rules by order.  Generating nodes and weights solves an
# eigenvalue problem, and the same handful of orders (16, 32, 64) is reused
# thousands of times while building the initial-condition overlay.
_GL_RULES = {}


def _gauss_legendre_rule(order):
    """Return cached ``(nodes, weights)`` for an ``order``-point rule.

    Arrays are marked read-only after generation: they are shared by every
    caller at this order, so an accidental in-place modification would
    silently corrupt every subsequent integral.
    """
    cached = _GL_RULES.get(order)
    if cached is None:
        cached = np.polynomial.legendre.leggauss(order)
        cached[0].setflags(write=False)
        cached[1].setflags(write=False)
        _GL_RULES[order] = cached
    return cached


def gauss_legendre(function, a, b, order, vectorized=False):
    """Integrate ``function`` over ``[a, b]`` with an ``order``-point rule.

    Gauss-Legendre nodes are defined on ``[-1, 1]``, so they are affinely
    mapped onto ``[a, b]`` as ``point = mid + half * node``, and the result
    scaled by ``half`` -- the Jacobian of that map.

    ``vectorized=True`` evaluates every node in one call, which matters when
    the integrand is itself expensive -- notably the nested equilibrium
    integrals below, where each evaluation runs another quadrature.
    """
    nodes, weights = _gauss_legendre_rule(order)
    mid = 0.5 * (a + b)
    half = 0.5 * (b - a)
    points = mid + half * nodes

    if vectorized:
        values = np.asarray(function(points), dtype=float)
        if values.shape != points.shape:
            raise ValueError("vectorized Gauss-Legendre callable changed shape")
    else:
        values = np.fromiter(
            (function(point) for point in points),
            dtype=float,
            count=order,
        )
    return half * float(np.dot(weights, values))


def drag_over_diffusion(solver_input=None, table_dir=None):
    """Return a callable ``A/B(x)``: summed collisional drag over diffusion.

    The no-RF equilibrium satisfies ``B df/dx + A f = 0``, so ``d ln f/dx =
    -A/B``.  Shared by :func:`collisional_equilibrium_shape` (which
    integrates it) and ``temperature.py`` (whose analytic initial
    temperature is ``T = 2 x B / A`` with ``E = x_0^2``).  A factory rather
    than a plain function so the table reads happen once, not per call.
    """
    C, ell, mu = collision_arrays(solver_input, table_dir)

    def ratio(xv):
        """A/B at one speed; 0 at x <= 0, where the coefficients vanish."""
        if xv <= 0.0:
            return 0.0
        A = 0.0
        B = 0.0
        for Cb, lb, mub in zip(C, ell, mu):
            px = psi_collision(lb * xv)
            A += (2.0 * Cb * lb * lb / mub) * xv * xv * px
            B += Cb * xv * px
        # Guard a vanishing denominator (all species negligible at this speed).
        return A / B if abs(B) > np.finfo(float).tiny else 0.0

    return ratio


def collisional_equilibrium_shape(x, solver_input=None, table_dir=None):
    """Return the no-RF collisional equilibrium distribution shape at ``x``.

    With RF off, the steady state satisfies ``B df/dx + A f = 0`` where ``A``
    and ``B`` are the summed collisional drag and diffusion coefficients. So

        f(x) = exp( -integral_0^x (A/B) dx' )

    which for a single Maxwellian background reduces to ``exp(-x^2/2)``; with
    several species at different temperatures it does not, hence the
    numerical integration.

    The integral is evaluated per grid point with 16-node Gauss-Legendre
    quadrature.  ``x <= 0`` is skipped (the value stays 1, matching the
    ``exp(-0)`` limit) because ``A/B`` is undefined there.
    """
    A_over_B = drag_over_diffusion(solver_input, table_dir)

    eq = np.ones_like(x, dtype=float)
    for i, xi in enumerate(x):
        if xi <= 0.0:
            continue
        integral = gauss_legendre(A_over_B, 0.0, xi, 16)
        eq[i] = math.exp(-integral)
    return eq


def velocity_equilibrium_norm(solver_input=None, table_dir=None):
    """Normalize the equilibrium over velocity: ``integral x^2 f_eq dx``.

    The ``x^2`` factor is the spherical velocity-space volume element.
    64-node quadrature (versus 16 for the inner shape integral) because this
    integrand is the product of two nontrivial functions over the full
    domain.

    ``vectorized=True`` lets the quadrature evaluate all nodes in one call,
    which matters because each evaluation runs the nested integral above.
    """
    solver_input = solver_input or PATHS.solver_input
    options = read_options(solver_input)
    x_max = float(options["x_max"])
    norm = gauss_legendre(
        lambda x: x * x * collisional_equilibrium_shape(
            x, solver_input, table_dir
        ),
        0.0,
        x_max,
        64,
        vectorized=True,
    )
    if not norm > 0.0:
        raise ValueError("velocity equilibrium normalization is non-positive")
    return norm


def collisional_pitch_norm(solver_input=None, table_dir=None):
    """Normalize over pitch angle, matching the solver's own quadrature.

    This mirrors ``Coefficients::collisional_th_norm()`` in the C++ solver
    deliberately: if the two disagreed, plotted distributions would not
    integrate to the particle number the solver conserves.

    The difficulty is that the integrand ``sin(theta) * cos_eps * L(theta)``
    has a narrow peak at the trapped-passing boundary, where ``L`` (the orbit
    time) spikes.  Plain quadrature would step over it.  The solver's
    approach, reproduced here:

    1. Locate the peak by taking the ``argmax`` of ``L`` over the first half
       of the table (the boundary always lies below 90 degrees).
    2. Split the integral there.
    3. Substitute ``|theta - theta_turn| = u^2`` on each side.  This clusters
       Gauss-Legendre nodes near the peak -- the substitution's Jacobian
       ``2u du`` vanishes there, concentrating resolution exactly where the
       integrand varies fastest.

    The final factor of 2 accounts for the symmetric other half of pitch space.
    """
    solver_input = solver_input or PATHS.solver_input
    options = read_options(solver_input)
    # eps_mass regularizes cos(theta) at 90 degrees, where a bounce orbit
    # degenerates; it must match the solver's value.
    eps_mass = float(options.get("eps_mass", "0.0"))
    params = table_parameters(table_dir)
    pitch_grid = np.linspace(
        params["pamin"], params["pamax"], params["npitch"]
    )
    L_table = read_binary_array(table_dir or PATHS.tables, "L_tab.bin")
    if L_table.size != pitch_grid.size:
        raise ValueError("L_tab.bin size does not match parameters.bin")

    # Index of 90 degrees in the table, clamped to the last entry.
    half_end = min(
        params["npitch"] - 1,
        int(math.floor(
            (0.5 * math.pi - params["pamin"])
            * (params["npitch"] - 1)
            / (params["pamax"] - params["pamin"])
        )),
    )
    turn_index = int(np.argmax(L_table[:half_end + 1]))
    theta_turn = float(pitch_grid[turn_index])
    if not 0.0 < theta_turn < 0.5 * math.pi:
        raise ValueError("could not locate trapped-passing peak in L table")

    def mass_theta(theta):
        """The pitch-space mass element at one angle."""
        L_value = float(np.interp(theta, pitch_grid, L_table))
        cos_eps = math.sqrt(math.cos(theta)**2 + eps_mass**2)
        return math.sin(theta) * cos_eps * L_value

    # The 2*u factors are the Jacobian of theta = theta_turn -/+ u^2.
    def left_integrand(u):
        """The mass element below the peak, in the clustered variable u."""
        return 2.0 * u * mass_theta(theta_turn - u * u)

    def right_integrand(u):
        """The mass element above the peak, in the clustered variable u."""
        return 2.0 * u * mass_theta(theta_turn + u * u)

    norm = 2.0 * (
        gauss_legendre(left_integrand, 0.0, math.sqrt(theta_turn), 32)
        + gauss_legendre(
            right_integrand,
            0.0,
            math.sqrt(0.5 * math.pi - theta_turn),
            32,
        )
    )
    if not norm > 0.0:
        raise ValueError("pitch equilibrium normalization is non-positive")
    return norm


def initial_condition_grid(x0, theta0, solver_input=None, table_dir=None):
    """Evaluate the normalized initial (no-RF equilibrium) distribution.

    This is what solution plots overlay as "the initial condition", so it
    must use the same normalization the solver does -- with the phase-space
    measure ``x_0^2 lambda(theta_0) sin(theta_0)``, not the naive
    ``x_0^2 sin(theta_0)``.  The ``lambda`` factor is why
    :func:`collisional_pitch_norm` exists rather than a plain ``sin``
    integral.

    Performance: ASGarD reconstruction grids repeat each ``x0`` value along
    the entire pitch axis, so evaluating the nested equilibrium integral per
    grid point would recompute the same value hundreds of times.
    ``np.unique`` with ``return_inverse`` evaluates once per *distinct* speed
    and scatters the result back -- typically a several-hundred-fold saving.

    The result is independent of ``theta0`` in shape (the equilibrium is
    isotropic); ``theta0`` enters only through the normalization.
    """
    x_norm = velocity_equilibrium_norm(solver_input, table_dir)
    theta_norm = collisional_pitch_norm(solver_input, table_dir)

    x_flat = np.asarray(x0).ravel()
    x_unique, inverse = np.unique(x_flat, return_inverse=True)
    eq_unique = collisional_equilibrium_shape(
        x_unique, solver_input, table_dir
    )
    eq_x = eq_unique[inverse].reshape(np.shape(x0))
    return eq_x / (theta_norm * x_norm)


# ---------------------------------------------------------------------------
# Section 3: bounce-average measures and velocity-space marginals
# ---------------------------------------------------------------------------


def bounce_pitch_weight(theta0, solver_input=None, table_dir=None):
    """Return the bounce-orbit pitch measure ``lambda * sin(theta_0)``.

    Bounce averaging integrates out a particle's position along its orbit,
    so each midplane pitch bin represents an amount of phase-space volume
    proportional to ``lambda = v * |cos(theta_0)| * tau_bounce``, where
    ``tau_bounce`` is what ``L_tab.bin`` stores.

    ``cos`` is regularized as ``sqrt(cos^2 + eps_mass^2)`` -- the same
    regularization the solver applies -- so the weight stays strictly
    positive at 90 degrees, where a bounce orbit degenerates to a point and
    the true ``cos`` vanishes.
    """
    solver_input = solver_input or PATHS.solver_input
    theta0 = np.asarray(theta0, dtype=float)
    options = read_options(solver_input)
    eps_mass = float(options.get("eps_mass", "0.0"))
    # Interpolate on a flat array then restore the original shape, so this
    # accepts scalars, vectors, and 2-D meshes alike.
    L = interp_1d_table(
        table_dir or PATHS.tables, "L_tab.bin", theta0.ravel()
    ).reshape(theta0.shape)
    cos_eps = np.sqrt(np.cos(theta0)**2 + eps_mass**2)
    return np.sin(theta0) * cos_eps * L


def _pitch_axis(theta0):
    """Return which array axis of a reconstruction mesh carries pitch angle.

    Determined empirically -- whichever axis ``theta0`` actually varies along
    is the pitch axis -- because ASGarD's axis order is not guaranteed across
    configurations, and guessing wrong would integrate over the wrong
    coordinate while still producing plausible-looking output.
    """
    theta_var_0 = (
        np.nanmax(np.abs(np.diff(theta0, axis=0))) if theta0.shape[0] > 1 else 0.0
    )
    theta_var_1 = (
        np.nanmax(np.abs(np.diff(theta0, axis=1))) if theta0.shape[1] > 1 else 0.0
    )
    return 0 if theta_var_0 > theta_var_1 else 1


def pitch_integral_weighted(f, x0, theta0, solver_input=None, table_dir=None):
    """Integrate out pitch angle, giving a marginal versus speed.

    Uses the bounce-orbit weight from :func:`bounce_pitch_weight` rather than
    a plain ``sin(theta)``, so that multiplying by ``x_0^2`` and integrating
    over ``x_0`` recovers the conserved global particle number the solver
    tracks.

    Returns ``(x_line, integral)`` sorted ascending in ``x``, because
    ASGarD's axis ordering is not guaranteed and an unsorted x-axis would
    draw as a scribble.
    """
    pitch_axis = _pitch_axis(theta0)
    weighted = f * bounce_pitch_weight(theta0, solver_input, table_dir)
    integral = np.trapezoid(weighted, x=theta0, axis=pitch_axis)
    # Collapsing the pitch axis leaves one x value per row; the mean is exact
    # because x is constant along that axis.
    x_line = np.mean(x0, axis=pitch_axis)

    order = np.argsort(x_line)
    return x_line[order], integral[order]


def velocity_integral_weighted(f, x0, theta0):
    """Integrate out speed, giving a marginal versus pitch angle (degrees).

    The ``x_0^2`` factor is the spherical volume element.  No bounce weight
    is applied here -- it belongs to the pitch coordinate, and this marginal
    keeps pitch as its free variable.
    """
    pitch_axis = _pitch_axis(theta0)
    velocity_axis = 1 - pitch_axis

    weighted = f * x0**2
    integral = np.trapezoid(weighted, x=x0, axis=velocity_axis)
    theta_line = np.mean(theta0, axis=velocity_axis)

    order = np.argsort(theta_line)
    # Degrees for display, matching the axis labels.
    return np.degrees(theta_line[order]), integral[order]


# ---------------------------------------------------------------------------
# Section 4: the coefficients themselves
# ---------------------------------------------------------------------------


def ql_coefficients(table_dir=None, solver_input=None):
    """Derive the quasilinear bounce-averaged coefficients from the tables.

    Returns ``{"B", "C", "E", "F"}``, each a ``(velocity, pitch)`` array on
    the fine plotting grid.

    Normalization note (this is the subtle part): the solver's ``ql()``
    returns ``eps_E`` times these arrays, and its conservative flux stores
    ``X_ql0 = lambda * ql()``.  So what is returned here is exactly the
    bounce average ``<X_ql> = X_ql0 / (lambda * eps_E)`` -- *neither* factor
    is included.  That makes these arrays directly comparable across runs
    with different wave amplitudes, which the raw solver quantities are not.

    The ``valid`` mask zeroes regions where the expressions are ill-defined
    rather than letting them produce infinities that would wreck the color
    scale: at ``x ~ 0`` (no resonance), at ``cos(theta) ~ 0`` (the
    ``1/costh`` factor in ``C`` diverges at 90 degrees), where ``L``
    vanishes, and if the wave frequency is nonpositive.
    """
    params = table_parameters(table_dir)
    x_grid, pitch_grid = plot_grids(table_dir)
    res = interp_2d_table(table_dir or PATHS.tables, "res_tab.bin", x_grid, pitch_grid)
    L = interp_1d_table(table_dir or PATHS.tables, "L_tab.bin", pitch_grid)

    V, TH = np.meshgrid(x_grid, pitch_grid, indexing="ij")
    sinth = np.sin(TH)
    cos_raw = np.cos(pitch_grid)
    # Broadcast pitch-only quantities along the velocity axis.
    costh = cos_raw[np.newaxis, :]
    omega = params["omega"]
    Omega_s0 = params["Omega_s0"]

    with np.errstate(divide="ignore", invalid="ignore"):
        sin2 = sinth * sinth
        # Dividing out L is what removes the lambda factor described above.
        B_base = res / L[np.newaxis, :]
        # B: diffusion in speed.
        B = B_base * V * V * sin2
        # C: the cross term. The (sin^2 - Omega/omega) factor changes sign
        # at the resonance condition, which is why these plots need a signed
        # (rather than purely positive) color scale.
        C = -B_base * V * sinth * (sin2 - Omega_s0 / omega) / costh
        # E: pitch-angle counterpart of C.
        E = sinth * C
        # F: the closure term C*E/B, guarded where B underflows.
        F = np.where(np.abs(B) >= 1.0e-30, C * E / B, 0.0)

    valid = (
        (V >= 1.0e-30)
        & (np.abs(costh) >= 1.0e-4)
        & (L[np.newaxis, :] > 1.0e-30)
        & (omega > 0.0)
    )
    return {
        "B": np.where(valid, B, 0.0),
        "C": np.where(valid, C, 0.0),
        "E": np.where(valid, E, 0.0),
        "F": np.where(valid, F, 0.0),
    }


def collision_coefficients(solver_input=None, table_dir=None):
    """Derive the collisional bounce-averaged coefficients.

    Returns ``{"A", "B", "F"}`` on the fine plotting grid, where ``A`` is
    drag, ``B`` is speed diffusion, and ``F`` is pitch-angle scattering.

    Like :func:`ql_coefficients`, these are bounce averages with ``lambda``
    divided out: ``<X_c> = X_c0 / lambda``.  For ``A`` and ``B`` the
    cancellation is exact and analytic (both contain ``lambda`` explicitly),
    so they end up independent of pitch angle -- hence the ``broadcast_to``.

    ``F`` is different: ``F_c0 = F_v * tan(theta_0) * I(theta_0)``, so
    dividing by ``lambda = cos(theta_0) L(theta_0)`` leaves
    ``sin*I / (cos^2 * L)``.  That looks divergent at 90 degrees but is
    finite: ``I`` vanishes as ``cos^2(theta_0)`` at the trapped-particle
    endpoint, so the ratio has a removable singularity.  The ``isfinite``
    filter below cleans up the floating-point residue of that cancellation.

    The speed loop is explicit (rather than vectorized) because
    :func:`psi_collision` branches on magnitude per element; the grid is
    small enough that this is not a bottleneck.
    """
    x_grid, pitch_grid = plot_grids(table_dir)
    L = interp_1d_table(table_dir or PATHS.tables, "L_tab.bin", pitch_grid)
    I = interp_1d_table(table_dir or PATHS.tables, "I_tab.bin", pitch_grid)
    C, ell, mu = collision_arrays(solver_input, table_dir)

    A_x = np.zeros_like(x_grid)
    B_x = np.zeros_like(x_grid)
    F_x = np.zeros_like(x_grid)
    for i, x in enumerate(x_grid):
        for Cb, lb, mub in zip(C, ell, mu):
            px = psi_collision(lb * x)
            A_x[i] += (2.0 * Cb * lb * lb / mub) * x * x * px
            B_x[i] += Cb * x * px
            # Small-x series for the same cancellation reason as psi_collision.
            if x < 1.0e-3:
                F_x[i] += Cb * lb * 2.0 / (3.0 * math.sqrt(math.pi))
            else:
                F_x[i] += Cb * (math.erf(lb * x) - px) / (2.0 * x)

    TH = pitch_grid[np.newaxis, :]
    sin_th = np.sin(TH)
    cos_th = np.cos(TH)
    with np.errstate(divide="ignore", invalid="ignore"):
        # A and B are pitch-independent after removing lambda.
        A = np.broadcast_to(A_x[:, np.newaxis], (len(x_grid), len(pitch_grid)))
        B = np.broadcast_to(B_x[:, np.newaxis], (len(x_grid), len(pitch_grid)))
        # F retains pitch structure; see the docstring on why this is finite.
        pitch_factor = (
            sin_th * I[np.newaxis, :]
            / (cos_th * cos_th * L[np.newaxis, :])
        )
        F = F_x[:, np.newaxis] * pitch_factor

    return {
        "A": np.where(np.isfinite(A), A, 0.0),
        "B": np.where(np.isfinite(B), B, 0.0),
        "F": np.where(np.isfinite(F), F, 0.0),
    }


# ---------------------------------------------------------------------------
# Section 5: figures
# ---------------------------------------------------------------------------
#
# All drawing goes through plot_common.static's generic contour2d/line1d, so
# these functions only choose data, layout, and labels.
#
# Note the shared `linewidths=None, extend="both"` on every coefficient
# panel: `None` yields matplotlib's default line width (coefficient panels
# are read as filled-ish line families and look right heavier than solution
# contours), and `extend="both"` adds the out-of-range arrows these fixed
# level sets need.


def plot_coefficients(table_dir=None, solver_input=None):
    """Build the 2x2 quasilinear coefficient figure (B, C, E, F).

    Cartesian view only.
    """
    x_grid, pitch_grid, x, y = velocity_mesh(table_dir)
    coeffs = ql_coefficients(table_dir, solver_input)

    fig, axs = plt.subplots(2, 2, figsize=(8.2, 5.25), constrained_layout=False)
    for ax, (name, title) in zip(axs.ravel(), QL_TABLES):
        contour2d(
            fig, ax, x, y, coeffs[name],
            title=title, style_axes=style_cartesian_axes,
            linewidths=None, extend="both",
        )
    fig.suptitle(
        r"$\langle X_{\mathrm{ql}}\rangle"
        r"=X_{\mathrm{ql}0}/(\lambda\,\epsilon_E)$",
        fontsize=14,
        y=0.985,
    )
    # Explicit margins rather than constrained_layout: the colorbars are
    # attached per-panel and constrained_layout fights them.
    fig.subplots_adjust(left=0.075, right=0.935, bottom=0.10, top=0.90,
                        wspace=0.36, hspace=0.34)

    return fig


def plot_collision_coefficients(solver_input=None, table_dir=None):
    """Build the 3-panel collisional coefficient figure (A, B, F).

    Cartesian view only.  Three panels do not tile a grid, so the layout is
    hand-placed: two on top, one centered below.  Each panel gets an
    explicitly positioned colorbar axes so all three stay the same size --
    letting matplotlib steal space per panel would leave the bottom,
    colorbar-less-neighbour panel wider than the others.
    """
    _, _, x, y = velocity_mesh(table_dir)
    coeffs = collision_coefficients(solver_input, table_dir)
    fig = plt.figure(figsize=(9.8, 6.1), constrained_layout=False)
    # Figure-fraction geometry for the hand-placed layout.
    ax_w = 0.335
    ax_h = 0.285
    cbar_w = 0.010
    cbar_pad = 0.012
    top_y = 0.64
    bottom_y = 0.14
    left_x = 0.065
    right_x = 0.575
    center_x = 0.5 - 0.5 * ax_w

    ax_positions = (
        (left_x, top_y, ax_w, ax_h),
        (right_x, top_y, ax_w, ax_h),
        (center_x, bottom_y, ax_w, ax_h),
    )
    axs = tuple(fig.add_axes(pos) for pos in ax_positions)
    # Each colorbar sits just right of its panel, same height.
    caxs = tuple(
        fig.add_axes((pos[0] + pos[2] + cbar_pad, pos[1], cbar_w, pos[3]))
        for pos in ax_positions
    )
    for ax, cax, (name, title) in zip(axs, caxs, COLLISION_TABLES):
        contour2d(
            fig, ax, x, y, coeffs[name],
            title=title, style_axes=style_cartesian_axes, cax=cax,
            linewidths=None, extend="both",
        )

    fig.suptitle(
        r"$\langle X_{\mathrm{c}}\rangle=X_{\mathrm{c}0}/\lambda$",
        fontsize=14,
        y=0.975,
    )

    return fig


def plot_li(table_dir=None):
    """Plot the two bounce-averaged orbit factors versus pitch angle.

    These are the combinations that actually enter the bounce-averaged
    operator, so plotting them directly is the quickest way to see whether
    the generated tables are sane:

      ``lambda = L cos(theta_0)``  -- the orbit-time weight
      ``chi    = I tan(theta_0)``  -- the pitch-scattering weight

    A ``symlog`` y-axis is used rather than the shared signed-log machinery,
    because these are single curves rather than fields and matplotlib's
    built-in symlog needs no custom norm.  Its ``linthresh`` is set six
    decades below the larger curve's peak, so the near-zero region stays
    readable without inventing structure.

    ``chi`` is computed only where ``cos(theta_0)`` is safely nonzero; the
    tangent diverges at 90 degrees, and the remaining entries stay at zero.
    """
    _, pitch = plot_grids(table_dir)
    pitch_deg = np.degrees(pitch)
    L = interp_1d_table(table_dir or PATHS.tables, "L_tab.bin", pitch)
    I = interp_1d_table(table_dir or PATHS.tables, "I_tab.bin", pitch)

    orbit_factor = L * np.cos(pitch)
    pitch_scattering_factor = np.zeros_like(I)
    away_from_midplane = np.abs(np.cos(pitch)) > 1.0e-12
    pitch_scattering_factor[away_from_midplane] = (
        I[away_from_midplane]
        * np.sin(pitch[away_from_midplane])
        / np.cos(pitch[away_from_midplane])
    )

    fig, ax = plt.subplots(1, 1, figsize=(5.8, 3.8), constrained_layout=True)
    plot_scale = max(
        float(np.max(np.abs(orbit_factor))),
        float(np.max(np.abs(pitch_scattering_factor))),
    )
    ax.set_yscale("symlog", linthresh=max(1.0e-12, 1.0e-6 * plot_scale))

    # scale="linear" here means "do not split into log branches" -- the axis
    # scaling above is matplotlib's own symlog, applied to the raw values.
    line1d(
        ax, pitch_deg,
        [
            (orbit_factor, r"$\lambda=L\cos\theta_0$",
             {"color": "#2d1e8f", "lw": 2.0}),
            (pitch_scattering_factor, r"$\chi=I\tan\theta_0$",
             {"color": "#cf1f63", "lw": 2.0}),
        ],
        scale="linear",
        legend=False,
    )
    ax.set_xlabel(r"$\theta_0$ [deg]", fontsize=13)
    ax.set_ylabel("bounce-averaged factors", fontsize=13)
    ax.set_xlim(pitch_deg[0], pitch_deg[-1])
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    title = (
        r"Orbit and pitch-scattering factors"
        "\n"
        r"$\lambda(\theta_0)=L(\theta_0)\cos\theta_0$, "
        r"$\chi(\theta_0)=I(\theta_0)\tan\theta_0$"
    )
    ax.set_title(title, fontsize=13)
    return fig


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------


def main():
    """CLI entry point: render the three static coefficient figures (no movie)."""
    parser = argparse.ArgumentParser(description="ICRF coefficient plots")
    parser.add_argument("--fig-dir", default=str(PATHS.figures))
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args()

    save_png(plot_coefficients(), args.fig_dir, "ql_coefficients", dpi=args.dpi)
    save_png(plot_collision_coefficients(), args.fig_dir, "collision_coefficients",
             dpi=args.dpi)
    save_png(plot_li(), args.fig_dir, "li_factors", dpi=args.dpi)


if __name__ == "__main__":
    main()
