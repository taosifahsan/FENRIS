"""LHCD's quasilinear diffusion tensor.

LHCD's coefficients are computed **analytically** from six input-deck
parameters -- no lookup tables, no compiled generator stage.  That is the one
structural difference from ``ICRF_2D/plot/coefficients.py``, which reads
precomputed ``.bin`` tables: same role in the architecture, opposite
implementation.

The physics: LH waves resonate over a band of *parallel* velocities, so the
diffusion coefficient is a window in ``x_parallel = x cos(theta)`` --
essentially "on" between two parallel-velocity cutoffs and "off" outside.
:func:`diffusion_window` builds that scalar window; :func:`ql_coefficients`
rotates it from the parallel direction into the solver's ``(x, theta)``
coordinates, which turns one scalar into a symmetric 2-tensor.

Cartesian ``(v_parallel, v_perp)`` view, signed-log color -- see
``plot_common/static.py``.  Reads no snapshots, so this costs nothing to run
alongside the expensive stages.

Used by:
  - ``LHCD_2D/plot/diagnostics.py`` -- the boundary diffusion value
  - ``tools/run.sh``                -- the tensor figure

Depends on: :mod:`plot_common.reader` (the deck), :mod:`plot_common.static`
(drawing).
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

# Panel keys and their LaTeX titles, in display order.
TENSOR_PANELS = (
    ("xx", r"$D_{xx}$"),
    ("xtheta", r"$D_{x\theta}$"),
    ("thetatheta", r"$D_{\theta\theta}$"),
)


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
    """Return the analytic LH quasilinear diffusion magnitude ``D(x, theta)``.

    Two multiplied factors:

    1. **A parallel-velocity window.**  The wave resonates only for
       ``cut_x_parallel_min < x_parallel < cut_x_parallel_max``, where
       ``x_parallel = x cos(theta)``.  With nonzero ``smoothing_width`` the
       window's edges are logistic rather than sharp -- the product of a
       rising step at the lower cutoff and a falling one at the upper.
    2. **An optional outer-speed rolloff.**  ``cut_center < 1`` tapers the
       coefficient off above ``cut_center * x_max``, so RF power is not
       injected right at the domain boundary where it would contaminate the
       flux measurement.  ``cut_center >= 1`` disables it entirely.

    All six knobs come from the solver's own deck, so the plotted coefficient
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


def ql_coefficients(x, theta, solver_input=None):
    """Rotate the scalar diffusion into the ``(x, theta)`` diffusion tensor.

    Diffusion acts purely along the *parallel* direction, so as a tensor in
    ``(x, theta)`` it is the outer product of the parallel unit vector with
    itself, scaled by the scalar magnitude.  In these coordinates the parallel
    direction has components ``(cos(theta), -sin(theta))``, giving

        D_xx        = D cos^2(theta)
        D_xtheta    = -D cos(theta) sin(theta)
        D_thetatheta = D sin^2(theta)

    The cross term is negative because increasing ``theta`` moves *away* from
    the parallel direction.  Being a rank-1 tensor, its determinant vanishes
    identically -- diffusion is strictly one-dimensional in velocity space,
    which is exactly the physics of a single resonant wave.
    """
    diffusion = diffusion_window(x, theta, solver_input)
    cosine = np.cos(theta)
    sine = np.sin(theta)
    return {
        "xx": diffusion * cosine**2,
        "xtheta": -diffusion * cosine * sine,
        "thetatheta": diffusion * sine**2,
    }


def initial_condition_grid(x, theta=None):
    """The Maxwellian initial condition, ``(2 pi)^-3/2 exp(-x^2/2)``.

    Isotropic, hence ``theta``-independent -- accepted only so the signature
    matches ICRF's, whose initial condition genuinely depends on pitch angle
    through its bounce-orbit normalization.
    """
    del theta
    return (2.0 * math.pi) ** (-1.5) * np.exp(-0.5 * np.asarray(x)**2)


def plot_tensor(solver_input=None, points=256):
    """Three-panel figure: the diffusion tensor's independent components.

    Evaluated on a synthetic ``(speed, pitch)`` grid spanning the full domain
    rather than on a reconstruction grid -- the coefficient is an analytic
    function of coordinates, so it needs no snapshot data at all.
    """
    solver_input = solver_input or PATHS.solver_input
    options = read_options(solver_input)
    x_max = option_float(options, "x_max")
    speed = np.linspace(0.0, x_max, points)
    pitch = np.linspace(0.0, math.pi, points)
    x, theta = np.meshgrid(speed, pitch, indexing="ij")
    coeffs = ql_coefficients(x, theta, solver_input)

    vpar, vperp = x * np.cos(theta), x * np.sin(theta)
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.6), constrained_layout=True)
    for ax, (key, title) in zip(axes, TENSOR_PANELS):
        contour2d(fig, ax, vpar, vperp, coeffs[key], title=title,
                  style_axes=style_cartesian_axes)
    fig.suptitle("LHCD quasilinear diffusion tensor", fontsize=14)
    return fig


def main():
    parser = argparse.ArgumentParser(description="LHCD coefficient plot")
    parser.add_argument("--fig-dir", default=str(PATHS.figures))
    parser.add_argument("-n", "--points", type=int, default=256)
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args()
    save_png(plot_tensor(points=args.points), args.fig_dir,
             "diffusion_tensor", dpi=args.dpi)


if __name__ == "__main__":
    main()
