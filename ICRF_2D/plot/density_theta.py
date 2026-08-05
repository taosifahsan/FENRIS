"""The conserved pitch density ``n(theta_0) = lambda sin(theta_0) *`` (speed-integrated F_0).

The speed-integrated marginal (computed by ``density_x.py``'s shared
:func:`derive`) carries, by deliberate convention, only the ELIMINATED
coordinate's measure factor ``x_0^2`` (see ``bounce_pitch_weight``'s
docstring in ``coefficients.py``).  This file multiplies the surviving
bounce-orbit factor back in at the call site, producing the true conserved
density

    n(theta_0) = 2 pi lambda(theta_0) sin(theta_0) * integral F_0(x_0, theta_0) x_0^2 dx_0

whose *area is the full 3-D particle count* -- gyrophase ``2 pi``
included -- exactly 1 at t=0 by the initial-condition normalization
(the same convention as LHCD_2D).  ``integral n dtheta_0`` equals
``density_x.py``'s ``integral n dx_0`` frame by frame, and both are
``2 pi *`` ``diagnostics.py``'s number moment.  The multiplication happens here at
the call site, not inside the shared reduction -- both quantities are
legitimate and different callers need each.

**The dip to ~zero at theta_0 = 90 deg is expected, not a bug.**  The
bounce weight ``lambda = |cos(theta_0)| L(theta_0)`` (eps_mass-regularized)
vanishes for deeply trapped orbits mirroring at the midplane -- that is the
Jacobian of labeling orbits by their midplane pitch, i.e. an honest feature
of the conserved measure.  It is also why this stays a valid density only
while ``lambda`` depends on ``theta_0`` alone: if table generation ever
makes ``L_tab`` speed-dependent, this factorization breaks and the plot
becomes an uncontrolled approximation.

Drawn on a *linear* y-axis, unlike the speed density: the pitch density
spans no decades, and linear axes show the trapped-orbit structure -- the
poles and the 90-degree dip -- directly, with area = particles readable by
eye.

Used by: ``tools/run.sh`` (one of the parallel plotter processes).

Depends on: ``density_x.py`` (:func:`derive`, which computes the
speed-integrated marginal), ``coefficients.py``
(:func:`bounce_pitch_weight`), :mod:`plot_common.reader` (the cache),
:mod:`plot_common.static` (drawing), :mod:`plot_common.movie` (movie
rendering).
"""

from __future__ import annotations

import argparse
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

import numpy as np

from coefficients import bounce_pitch_quadrature, bounce_pitch_weight
from plot_common.movie import render_movie
from plot_common.reader import load_cache, load_snapshots
from plot_common.static import line1d, render_still, save_png
from density_x import derive


def derive_density(cache):
    """Weight the speed-integrated marginal into the conserved density n(theta_0).

    Reuses ``density_x.py``'s :func:`derive` (one reduction pass, shared
    across the marginal plotters) and multiplies each frame by
    the surviving-coordinate measure ``lambda(theta_0) sin(theta_0)``.  The
    marginal's pitch axis is stored in degrees for plotting; the weight and
    the particle-number integral both need radians, so convert once here.
    """
    data = derive(cache)
    branch = data["theta"]
    theta_deg = np.asarray(branch["x"], dtype=float)
    theta_rad = np.radians(theta_deg)
    # 2 pi: the gyrophase integral, made explicit so the area is the full
    # 3-D particle count rather than the bare 2-D weighted moment.
    weight = 2.0 * np.pi * bounce_pitch_weight(theta_rad)
    initial = weight * np.asarray(branch["initial"], dtype=float)
    frames = [weight * np.asarray(frame, dtype=float) for frame in branch["frames"]]
    # The plotted curves use the pointwise weight (they ARE the density);
    # the particle-number integrals must not: trapezoid through the
    # trapped-passing peak of lambda is resolution-dependent, so integrate
    # the un-weighted marginal against the exact hat-function quadrature.
    quad = 2.0 * np.pi * bounce_pitch_quadrature(theta_rad)
    numbers = [float(quad @ np.asarray(frame, dtype=float))
               for frame in branch["frames"]]

    # Fixed linear y-limits spanning the initial curve and every frame, so
    # a movie's axis does not rescale as the solution evolves.  Densities
    # can dip slightly negative from reconstruction ringing; keep the true
    # signed minimum rather than clamping it away.
    everything = np.concatenate([initial] + frames)
    finite = everything[np.isfinite(everything)]
    low, high = float(finite.min()), float(finite.max())
    margin = 0.08 * (high - low) if high > low else 1.0
    low, high = low - margin, high + margin
    return {
        "theta_deg": theta_deg,
        "initial": initial,
        "frames": frames,
        "numbers": numbers,
        "times": data["times"],
        "ylim": (low, high),
    }


# One size for the still and every movie frame -- stated once.
FIGSIZE = (5.9, 3.8)


def draw_frame(fig, ax, data, index):
    """One density frame: initial (dashed) versus current (solid), linear."""
    line1d(
        ax, data["theta_deg"],
        [
            (data["initial"], "initial", {"color": "#999999",
                                          "linestyle": "--", "lw": 1.8}),
            (data["frames"][index], "current", {"color": "#118ab2",
                                                "linestyle": "-", "lw": 2.0}),
        ],
        scale="linear", ylim=data["ylim"], legend=False,
    )
    ax.set_xlabel(r"$\theta_0$ [deg]", fontsize=13)
    ax.set_ylabel(r"$n(\theta_0)$", fontsize=13)
    ax.set_title(
        r"$n(\theta_0)=2\pi\lambda\sin\theta_0\int\mathcal{F}_0\,x_0^2\,dx_0$"
        rf":  $t={data['times'][index]:.2f}\,\tau_c$,"
        rf"  $N={data['numbers'][index]:.4f}$",
        fontsize=11,
    )
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()


def main():
    """CLI entry point: parse flags, load the data, render the figures.

    Giving neither ``--static`` nor ``--movie`` renders both -- that is how
    tools/run.sh invokes every plotter; either flag narrows a manual run to
    just that output.
    """
    parser = argparse.ArgumentParser(description="ICRF pitch-density plots")
    parser.add_argument("--static", action="store_true")
    parser.add_argument("--movie", action="store_true")
    parser.add_argument("-o", "--output", default=str(PATHS.snapshots))
    parser.add_argument("--cache", default=None,
                        help="load the shared cache.npz instead of reading snapshots")
    parser.add_argument("--fig-dir", default=str(PATHS.figures))
    parser.add_argument("-n", "--points", type=int, default=192)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--dpi", type=int, default=140)
    args = parser.parse_args()
    do_static = args.static or not (args.static or args.movie)
    do_movie = args.movie or not (args.static or args.movie)

    if args.cache:
        cache = load_cache(args.cache)
    else:
        cache = load_snapshots(args.output, args.points)
    data = derive_density(cache)

    # Bind the derived data into the (fig, ax, index) signature render_still
    # and render_movie expect (closures are fine: rendering is in-process).
    def draw(fig, ax, index):
        draw_frame(fig, ax, data, index)

    if do_static:
        save_png(render_still(draw, len(data["times"]) - 1, figsize=FIGSIZE),
                 args.fig_dir, "density_theta", dpi=220)
    if do_movie:
        render_movie(draw, len(data["times"]),
                     str(Path(args.fig_dir) / "density_theta.mp4"),
                     figsize=FIGSIZE, fps=args.fps, dpi=args.dpi)


if __name__ == "__main__":
    main()
