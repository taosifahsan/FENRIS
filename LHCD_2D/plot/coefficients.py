"""LHCD's quasilinear diffusion coefficient.

LHCD's coefficients are computed **analytically** from the input deck -- no
lookup tables, no compiled generator stage.  That is the one structural
difference from ``ICRF_2D/plot/coefficients.py``, which reads precomputed
``.bin`` tables: same role in the architecture, opposite implementation.

The physics: LH waves resonate over a band of *parallel* velocities, so the
quasilinear operator is a plain one-dimensional diffusion along
``w = x_parallel = x cos(theta)``,

    d/dw ( D_ql(w) df/dw )

and :func:`diffusion_window` is that ``D_ql`` -- essentially "on" between two
parallel-velocity cutoffs and "off" outside.  :func:`plot_diffusion` draws it
directly, in the Cartesian ``(v_parallel, v_perp)`` view, where the resonance
is a vertical band: the operator acts along ``w`` only, so the coefficient is
constant along every vertical line and the band's edges are the two cutoffs.

**Why there is no tensor figure.**  In the solver's ``(x, theta)`` chart the
same operator becomes ``div(D . grad f)`` with

    D_xx = D_ql cos^2(theta),   D_x,theta = -D_ql cos(theta) sin(theta),
    D_theta,theta = D_ql sin^2(theta)

-- the outer product of the parallel unit vector ``(cos, -sin)`` with itself.
Being rank-1 its determinant vanishes identically, so those three panels
carry no information beyond ``D_ql`` and the rotation angle: diffusion is
strictly one-dimensional in velocity space, which is exactly the physics of a
single resonant wave.  Plotting the scalar says it once instead of three
times.

Used by: ``tools/run.sh`` (the coefficient figure); ``growth.py`` and
``solution.py`` import the axis styling and the initial condition from here.

Depends on: :mod:`plot_common.reader` (the deck), :mod:`plot_common.static`
(``save_png``).
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

from plot_common.reader import option_float, read_options
from plot_common.static import contour2d, save_png


def style_cartesian_axes(ax):
    """Label axes for the Cartesian velocity-space view.

    ``set_aspect("equal")`` is essential: velocity space is isotropic, so an
    unequal aspect ratio would distort the resonance geometry the plot exists
    to show.
    """
    ax.set_xlabel(r"$v_\parallel/v_{te}$", fontsize=12)
    ax.set_ylabel(r"$v_\perp/v_{te}$", fontsize=12)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(labelsize=10)


def smooth_step(x, cut, width):
    """Logistic step: ~0 below ``cut``, ~1 above, over a scale of ``width``.

    ``width == 0`` degenerates to a hard boolean step, which the deck can
    request explicitly.

    The exponent is clipped to +/-700 because ``exp(710)`` overflows float64 to
    infinity; at |z| = 700 the logistic is already 0 or 1 to full precision, so
    clipping changes nothing except avoiding the overflow warning.
    """
    if width == 0.0:
        return np.asarray(x) > cut
    z = np.clip((np.asarray(x) - cut) / width, -700.0, 700.0)
    return 1.0 / (1.0 + np.exp(-z))


def diffusion_window(x, theta, solver_input=None):
    """Return the analytic LH quasilinear diffusion magnitude ``D_ql(x, theta)``.

    Two multiplied factors:

    1. **A parallel-velocity window.**  The wave resonates only for
       ``cut_x_parallel_min < x_parallel < cut_x_parallel_max``, where
       ``x_parallel = x cos(theta)``.  With nonzero ``smoothing_width`` the
       window's edges are logistic rather than sharp -- the product of a
       rising step at the lower cutoff and a falling one at the upper.
    2. **An optional outer-speed rolloff.**  ``cut_center < 1`` tapers the
       coefficient off above ``cut_center * x_max``, so RF power is not
       injected right at the domain boundary where it would contaminate the
       flux measurement.  ``cut_center >= 1`` disables it entirely, and then
       ``D_ql`` is a function of ``x_parallel`` alone.

    All the knobs come from the solver's own deck, so the plotted coefficient
    is the one the run actually used.
    """
    solver_input = solver_input or PATHS.solver_input
    options = read_options(solver_input)
    x_parallel_min = option_float(options, "cut_x_parallel_min")
    x_parallel_max = option_float(options, "cut_x_parallel_max")
    height = option_float(options, "cut_height")
    smoothing_width = float(options.get("smoothing_width", "0.02"))
    cut_center = float(options.get("cut_center", "1.0"))
    x_max = option_float(options, "x_max")

    x = np.asarray(x)
    x_parallel = x * np.cos(theta)
    if smoothing_width == 0.0:
        parallel = (
            (x_parallel_min < x_parallel) & (x_parallel < x_parallel_max)
        ).astype(float)
    else:
        # Rising edge at the lower cutoff times falling edge at the upper:
        # note the reversed argument order in the second call, which is what
        # makes it fall rather than rise.
        parallel = (
            smooth_step(x_parallel, x_parallel_min, smoothing_width)
            * smooth_step(x_parallel_max, x_parallel, smoothing_width)
        )

    if cut_center >= 1.0:
        x_cutoff = 1.0
    elif smoothing_width == 0.0:
        x_cutoff = (x / x_max < cut_center).astype(float)
    else:
        x_cutoff = 0.5 * (
            1.0 - np.tanh((x / x_max - cut_center) / smoothing_width)
        )
    return height * parallel * x_cutoff


def initial_condition_grid(x, theta=None):
    """The Maxwellian initial condition, ``pi^-3/2 exp(-x^2)``.

    Standard units ``x = v / sqrt(2T/m)`` (see src/LHCD_2D.cpp's
    normalization header); the 2 pi * x^2 sin(theta) integral of this is
    exactly 1.  Isotropic, hence ``theta``-independent -- accepted only so
    the signature matches ICRF's, whose initial condition genuinely depends
    on pitch angle through its bounce-orbit normalization.
    """
    del theta
    return math.pi ** (-1.5) * np.exp(-np.asarray(x)**2)


def plot_diffusion(solver_input=None, points=256):
    """The scalar ``D_ql`` the quasilinear operator diffuses with, in Cartesian.

    Evaluated on a synthetic ``(speed, pitch)`` grid spanning the full domain
    and mapped to ``(v_parallel, v_perp)`` -- the coefficient is an analytic
    function of coordinates, so it needs no snapshot data at all and costs
    nothing to run alongside the expensive stages.

    Reading it: the resonance appears as a vertical band, because ``D_ql``
    depends on ``x_parallel = x cos(theta)``.  Vertical means the coefficient
    is genuinely independent of ``v_perp`` -- which is the whole reason the
    perpendicular tail in ``density_vperp.py`` cannot come from the RF
    directly.  With ``cut_center < 1`` an outer-speed rolloff also curves the
    band off at large total speed.
    """
    solver_input = solver_input or PATHS.solver_input
    options = read_options(solver_input)
    x_max = option_float(options, "x_max")
    speed = np.linspace(0.0, x_max, points)
    pitch = np.linspace(0.0, math.pi, points)
    x, theta = np.meshgrid(speed, pitch, indexing="ij")

    diffusion = diffusion_window(x, theta, solver_input)
    vpar, vperp = x * np.cos(theta), x * np.sin(theta)

    fig, ax = plt.subplots(figsize=(5.8, 4.6), constrained_layout=True)
    contour2d(
        fig, ax, vpar, vperp, diffusion,
        style_axes=style_cartesian_axes,
        title=(
            r"LHCD quasilinear coefficient $D_{ql}$"
            "\n"
            r"$\frac{\partial}{\partial w}"
            r"\left(D_{ql}\frac{\partial f}{\partial w}\right)$,"
            r"  $w=v_\parallel$"
        ),
    )
    return fig


def main():
    """CLI entry point: render the static coefficient figure (no movie)."""
    parser = argparse.ArgumentParser(description="LHCD coefficient plot")
    parser.add_argument("--fig-dir", default=str(PATHS.figures))
    parser.add_argument("-n", "--points", type=int, default=256)
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args()
    save_png(plot_diffusion(points=args.points), args.fig_dir,
             "diffusion_coefficient", dpi=args.dpi)


if __name__ == "__main__":
    main()
