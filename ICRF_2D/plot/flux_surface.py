"""ICRF's magnetic flux surfaces.

ICRF-only, and the reason a project-specific "flux surface" plotter exists at
all: LHCD has no magnetic geometry, so it has no counterpart to this file.

This is *static* geometry, reconstructed from the **build** deck.  It reads no
snapshot data and knows nothing about the distribution function -- it answers
"what magnetic configuration is this run assuming?", not "what is the plasma
doing?".  For the latter, see ``diagnostics.py``.

(Those two were briefly one file, since the boundary flux is also "flux".
They are separated because they answer unrelated questions, and because
splitting them lets ``diagnostics.py`` carry the same name and role in both
projects.)

The physics: the Solov'ev equilibrium is an analytic solution of the
Grad-Shafranov equation whose poloidal flux is

    psi(R, Z) = C R^4/8 + A (R^2 ln R - R^2/2) + B Z^2

with A, B, C fixed by the requested plasma shape.  It has two stationary
points on the midplane -- an O-point (the magnetic axis, confined plasma) and
an X-point (a saddle, through which the separatrix passes) -- and locating and
classifying those is most of the work here.

Used by:
  - ``tools/run.sh`` -- the trajectory figure

Depends on: :mod:`plot_common.reader` (the build deck).  No snapshot access,
no coefficients.  Deliberately does not route through
:mod:`plot_common.static`'s ``contour2d``: this is a bespoke geometry figure
with a fixed ``[0, 1]`` normalization, three separately styled contour levels,
O/X point markers, and a hand-built legend -- none of which the generic
signed-log field plotter is for.
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

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from plot_common.reader import read_options
from plot_common.static import save_png

# A plain white -> blue ramp for the confined-flux region, where psi_N runs
# 0 (separatrix) to 1 (magnetic axis).  This is a fixed positive-only
# colormap for a geometry figure, not part of the signed-log data-scale
# system in plot_common.static -- kept local since this is its only user.
POS_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "linear_white_to_blue",
    ("#ffffff", "#d9fbff", "#5ad7d1", "#1787bf", "#2d1e8f"),
)
POS_CMAP.set_bad("white")


def bisect_root(function, lo, hi, tolerance=1.0e-12, iterations=160):
    """Find a bracketed scalar root by bisection.

    Hand-rolled rather than using ``scipy.optimize.brentq`` so the plotting
    stack has no SciPy dependency -- these projects run on clusters where the
    Python environment is deliberately minimal.

    Bisection (not Newton) because it cannot diverge: given a sign change on
    ``[lo, hi]``, it always converges, and 160 iterations is far more than the
    ~50 needed to exhaust float64 precision.  Speed is irrelevant here (two
    roots per figure).

    Raises if the interval does not bracket a root, since silently returning
    an endpoint would put the magnetic axis in the wrong place.
    """
    f_lo = function(lo)
    f_hi = function(hi)
    # Exact hits at either end are valid roots.
    if f_lo == 0.0:
        return lo
    if f_hi == 0.0:
        return hi
    if f_lo * f_hi > 0.0:
        raise ValueError(f"root is not bracketed on [{lo}, {hi}]")

    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        f_mid = function(mid)
        if f_mid == 0.0 or hi - lo <= tolerance:
            return mid
        # Keep the half that still contains the sign change.
        if f_lo * f_mid <= 0.0:
            hi = mid
            f_hi = f_mid
        else:
            lo = mid
            f_lo = f_mid
    return 0.5 * (lo + hi)


def equilibrium_geometry(build_input=None):
    """Reconstruct the Solov'ev MHD equilibrium from the build deck.

    The Solov'ev equilibrium is an analytic solution of the Grad-Shafranov
    equation.  Its poloidal flux function is

        psi(R, Z) = C R^4/8 + A (R^2 ln R - R^2/2) + B Z^2

    with ``A``, ``B``, ``C`` fixed by the requested shape (aspect ratio
    ``eps``, elongation ``kappa``, triangularity ``delta``).

    The physics that matters for plotting: ``psi`` has two stationary points
    on the midplane ``Z = 0``, one an O-point (the magnetic axis, where the
    Hessian is definite -- confined plasma) and one an X-point (a saddle --
    the separatrix passes through it).  Which is which depends on the deck, so
    both are found and then *classified* by the sign of the Hessian
    determinant rather than assumed by position.

    Finding them: on ``Z = 0``, ``dpsi/dR = 0`` reduces to
    ``C R^2 + 4A ln R = 0``, called ``stationary`` below.  That function is
    negative at its own critical point ``R_critical = sqrt(-2A/C)`` and
    positive far to either side, so it has exactly two roots which are
    bracketed by expanding outward from ``R_critical`` and bisecting each side.

    Returns a dict carrying both the scalar geometry and the ``psi`` callables,
    so :func:`plot_trajectory` can evaluate flux anywhere without re-deriving
    the coefficients.
    """
    build_input = build_input or PATHS.build_input
    options = read_options(build_input)
    R0 = float(options["R0"])          # major radius
    eps = float(options["eps"])        # inverse aspect ratio a/R0
    kappa = float(options["kappa"])    # elongation
    delta = float(options["delta"])    # triangularity
    B0 = float(options["B0"])          # toroidal field on axis
    alpha = float(options["alpha"])
    psi_fraction = float(options["psi"])  # which surface to highlight

    a = eps * R0                      # minor radius
    R1 = R0 * (1.0 - eps)             # inboard midplane edge
    R2 = R0 * (1.0 + eps)             # outboard midplane edge
    Rt = R0 * (1.0 - delta * eps)     # radius of the top of the boundary
    Zt = kappa * a                    # height of the top
    # C, A, B follow from requiring psi = 0 on the boundary at R1 and at the
    # top point (Rt, Zt); this is the standard Solov'ev shaping construction.
    C = 8.0 / R0**4
    A = -(C * R1**4 / 8.0) / (R1**2 * math.log(R1) - 0.5 * R1**2)
    B = -(C * Rt**4 / 8.0 + A * (Rt**2 * math.log(Rt) - 0.5 * Rt**2)) / Zt**2
    F0_squared = (R0 * B0) ** 2

    # Without these signs the two-stationary-point topology does not exist
    # and the O/X classification below would be meaningless.
    if not (C > 0.0 and A < 0.0 and B != 0.0):
        raise ValueError("equilibrium does not have a nondegenerate two-root O/X topology")

    def psi(R, Z):
        """Poloidal flux function."""
        R = np.asarray(R)
        return C * R**4 / 8.0 + A * (R**2 * np.log(R) - 0.5 * R**2) + B * np.asarray(Z)**2

    def psi_R(R, Z):
        """dpsi/dR (Z-independent for this equilibrium)."""
        del Z
        R = np.asarray(R)
        return 0.5 * C * R**3 + 2.0 * A * R * np.log(R)

    def psi_Z(R, Z):
        """dpsi/dZ."""
        del R
        return 2.0 * B * np.asarray(Z)

    def psi_RR(R):
        """d2psi/dR2 on the midplane, used to classify O versus X."""
        return 1.5 * C * R**2 + 2.0 * A * (math.log(R) + 1.0)

    def stationary(R):
        """Zero exactly at the midplane stationary points of psi."""
        return C * R**2 + 4.0 * A * math.log(R)

    R_critical = math.sqrt(-2.0 * A / C)
    if stationary(R_critical) >= 0.0:
        raise ValueError("radial stationary points are absent or degenerate")

    # Bracket the inner root: walk toward R = 0 until `stationary` turns
    # positive.  Starting from a tiny fraction of R0 because ln R -> -inf.
    R_lo = 1.0e-12 * max(1.0, R0)
    for _ in range(12):
        if stationary(R_lo) > 0.0:
            break
        R_lo *= 0.1
    if stationary(R_lo) <= 0.0:
        raise ValueError("could not bracket the inner stationary point")

    # Bracket the outer root by doubling outward.
    R_hi = max(2.0 * R_critical, 2.0 * R0, R2, Rt)
    for _ in range(32):
        if stationary(R_hi) > 0.0:
            break
        R_hi *= 2.0
    if stationary(R_hi) <= 0.0:
        raise ValueError("could not bracket the outer stationary point")

    R_inner = bisect_root(stationary, R_lo, R_critical)
    R_outer = bisect_root(stationary, R_critical, R_hi)
    # Hessian determinant on the midplane is psi_RR * psi_ZZ = psi_RR * 2B.
    # Positive -> extremum (O-point); negative -> saddle (X-point).
    determinant_inner = psi_RR(R_inner) * (2.0 * B)
    determinant_outer = psi_RR(R_outer) * (2.0 * B)
    if determinant_inner > 0.0 and determinant_outer < 0.0:
        R_axis, R_xpoint = R_inner, R_outer
    elif determinant_outer > 0.0 and determinant_inner < 0.0:
        R_axis, R_xpoint = R_outer, R_inner
    else:
        raise ValueError("could not classify exactly one O-point and one X-point")

    psi_axis = float(psi(R_axis, 0.0))
    psi_separatrix = float(psi(R_xpoint, 0.0))
    # The surface to highlight, as a fraction of the way from separatrix to axis.
    psi_surface = psi_separatrix + psi_fraction * (psi_axis - psi_separatrix)
    # A single length scale for choosing plot limits.
    shape_scale = max(a, abs(Zt), abs(R2 - R_axis), abs(R_axis - R1))

    return {
        "options": options,
        "R0": R0, "a": a, "R1": R1, "R2": R2, "Rt": Rt, "Zt": Zt,
        "A": A, "B": B, "C": C, "alpha": alpha,
        "F0_squared": F0_squared,
        "psi_fraction": psi_fraction,
        "psi_axis": psi_axis,
        "psi_surface": psi_surface,
        "psi_separatrix": psi_separatrix,
        "R_axis": R_axis,
        "R_xpoint": R_xpoint,
        "shape_scale": shape_scale,
        "psi": psi, "psi_R": psi_R, "psi_Z": psi_Z,
    }


def plot_trajectory(build_input=None):
    """Plot the normalized flux ``psi_N``, with 1 at the O-point and 0 at the X.

    Normalizing this way makes the picture readable independent of the
    absolute flux scale: 1 is the magnetic axis, 0 is the separatrix, and
    everything between is confined.  Values outside ``[0, 1]`` are masked so
    the filled region shows exactly the confined plasma and nothing else.

    The highlighted surface is drawn twice, a thick pale line under a thinner
    dark one, giving it a halo so it stays visible against the filled
    contours.
    """
    eq = equilibrium_geometry(build_input)
    # Frame the plot generously around the geometry rather than the data.
    radial_margin = 1.45 * eq["shape_scale"]
    R_min = max(1.0e-5, min(eq["R_axis"], eq["R_xpoint"]) - radial_margin)
    R_max = max(eq["R_axis"], eq["R_xpoint"]) + radial_margin
    Z_limit = 1.45 * max(eq["shape_scale"], abs(eq["Zt"]))
    R_values = np.linspace(R_min, R_max, 700)
    Z_values = np.linspace(-Z_limit, Z_limit, 650)
    R_mesh, Z_mesh = np.meshgrid(R_values, Z_values)
    psi_values = eq["psi"](R_mesh, Z_mesh)
    psi_normalized = (
        (psi_values - eq["psi_separatrix"])
        / (eq["psi_axis"] - eq["psi_separatrix"])
    )
    confined_flux = np.ma.masked_outside(psi_normalized, 0.0, 1.0)

    fig, ax = plt.subplots(figsize=(7.2, 5.8), constrained_layout=True)
    flux_norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
    # antialiased=False avoids pale seams between adjacent filled bands.
    ax.contourf(
        R_mesh, Z_mesh, confined_flux,
        levels=np.linspace(0.0, 1.0, 31), cmap=POS_CMAP,
        norm=flux_norm, antialiased=False,
    )
    # The separatrix.
    ax.contour(
        R_mesh, Z_mesh, psi_normalized,
        levels=[0.0], colors=["#2d1e8f"], linewidths=2.4,
    )
    # The highlighted surface: pale halo, then the line itself.
    ax.contour(
        R_mesh, Z_mesh, psi_normalized,
        levels=[eq["psi_fraction"]], colors=["#eeeaf7"], linewidths=5.0,
    )
    ax.contour(
        R_mesh, Z_mesh, psi_normalized,
        levels=[eq["psi_fraction"]], colors=["#8b7bb8"], linewidths=2.8,
    )

    # O-point (filled circle) and X-point (cross), both on the midplane.
    ax.scatter(
        [eq["R_axis"]], [0.0], s=60, marker="o",
        facecolor="#2f6f73", edgecolor="#d9fbff", linewidth=1.0, zorder=8,
    )
    ax.scatter(
        [eq["R_xpoint"]], [0.0], s=95, marker="x",
        color="#2f6f73", linewidth=2.2, zorder=8,
    )

    # Proxy artists: contour sets and scatters do not produce usable legend
    # handles, so the legend entries are built by hand.
    legend_handles = [
        Line2D([0], [0], color="#8b7bb8", lw=3.0,
               label=rf"$\psi_N={eq['psi_fraction']:.3g}$"),
        Line2D([0], [0], color="#2d1e8f", lw=2.4,
               label=r"$\psi_N=0$"),
        Line2D([0], [0], marker="o", linestyle="None", color="#2f6f73",
               markerfacecolor="#2f6f73", label=r"$O$"),
        Line2D([0], [0], marker="x", linestyle="None", color="#2f6f73",
               markersize=8, markeredgewidth=1.8, label=r"$X$"),
    ]
    ax.legend(
        handles=legend_handles, loc="upper right", ncol=2, fontsize=8.5,
        framealpha=0.9, handlelength=1.8, columnspacing=1.0,
        borderpad=0.45, labelspacing=0.35,
    )

    # Standalone mappable: the filled contours use masked data, so a colorbar
    # taken from them would omit the masked ends of the range.
    flux_map = mpl.cm.ScalarMappable(norm=flux_norm, cmap=POS_CMAP)
    flux_map.set_array([])
    colorbar = fig.colorbar(
        flux_map, ax=ax, pad=0.018, fraction=0.032, shrink=0.72, aspect=24,
    )
    colorbar.set_label(r"$\psi_N$", fontsize=10)
    colorbar.set_ticks(np.linspace(0.0, 1.0, 6))
    colorbar.ax.tick_params(labelsize=8)
    ax.set_xlabel(r"$R$ [m]", fontsize=12)
    ax.set_ylabel(r"$Z$ [m]", fontsize=12)
    # Equal aspect: this is real poloidal-plane geometry, so distorting it
    # would misrepresent the plasma shape.
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="#1787bf", alpha=0.12)
    return fig


def main():
    """CLI entry point: render the static flux-surface figure (no movie)."""
    parser = argparse.ArgumentParser(description="ICRF flux surface plot")
    parser.add_argument("--fig-dir", default=str(PATHS.figures))
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args()
    save_png(plot_trajectory(), args.fig_dir, "trajectory", dpi=args.dpi)


if __name__ == "__main__":
    main()
