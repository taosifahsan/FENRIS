"""The pitch anisotropy ``A(theta_0)``: the shape of the pitch profile, not its size.

Start from the anisotropy density ``F_0 / <F_0>_theta`` (each speed shell
normalized by its own bounce-measure pitch average), then collapse the
speed axis with each shell's particle weight ``x_0^2 <F_0>_theta``.  The
weights telescope, leaving a ratio of two things the marginal machinery
already computes:

    A(theta_0) = m(theta_0) / <m>_lambda,

    m(theta_0)   = integral F_0(x_0, theta_0) x_0^2 dx_0
    <m>_lambda   = integral m lambda sin(theta_0) dtheta_0
                   / integral lambda sin(theta_0) dtheta_0.

``m`` is exactly ``density_x.py``'s speed-integrated marginal (the "theta"
branch of its shared :func:`derive`), and the denominator is its pitch
AVERAGE -- an average, not an integral, which is what makes A
dimensionless and normalized: ``A = 1`` at every angle iff the
distribution is isotropic, so the t = 0 Maxwellian draws a flat line at
one -- a built-in correctness check in the same spirit as the 3/2
equipartition energy.  RF perpendicular heating then reads directly:
``A > 1`` near theta_0 = 90 deg (over-populated pitches), ``A < 1`` along
the field, and the area-weighted average is pinned to 1 by construction.

Equivalently, in terms of the conserved pitch density
``n(theta_0) = 2 pi lambda sin(theta_0) m(theta_0)`` (what the old plot
drew), the 2 pi cancels and

    A(theta_0) = [ n(theta_0) / N ] / [ lambda sin(theta_0) / Lambda ],

    N = integral n dtheta_0,     Lambda = integral lambda sin dtheta_0

-- the OBSERVED pitch share of the particles divided by the ISOTROPIC
pitch share, i.e. the fraction of phase space that pitch owns.  A ratio
of two probability distributions over theta_0: it answers "how over- or
under-represented is this pitch relative to its phase-space volume?".
This form makes explicit why the measure's 90-degree zero is absent from
A (numerator and reference vanish together; the ratio stays finite) and
why the lambda-weighted mean of every frame is exactly 1 (both
distributions integrate to 1 against the same measure).

Replaces the pitch density ``n(theta_0)`` plot, which multiplied the same
marginal by the bounce weight ``lambda sin(theta_0)`` instead of dividing
by an average: that drew the conserved measure -- poles, the 90-degree
trapped-orbit dip -- on top of the physics, and the bulk's sheer particle
count kept the curve looking near-Maxwellian however anisotropic the tail
grew.  Dividing by ``<m>`` removes the amount and keeps the shape.  (The
90-degree dip was the measure's zero, so it belongs to n and NOT to A:
its absence here is correct, not a regression.)

Drawn on a *linear* y-axis around the reference line at 1: the whole
point is which side of 1 each angle sits on, and by how much.

Used by: ``tools/run.sh`` (one of the parallel plotter processes).

Depends on: ``density_x.py`` (:func:`derive`, the shared reduction),
``coefficients.py`` (:func:`bounce_pitch_quadrature`),
:mod:`plot_common.reader` (the cache), :mod:`plot_common.static`
(drawing), :mod:`plot_common.movie` (movie rendering).
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

from coefficients import bounce_pitch_quadrature
from plot_common.movie import render_movie
from plot_common.reader import load_cache, load_snapshots
from plot_common.static import line1d, render_still, save_png
from density_x import derive


def derive_anisotropy(cache):
    """Normalize the speed-integrated marginal into the anisotropy A(theta_0).

    Reuses ``density_x.py``'s :func:`derive` (one reduction pass, shared
    across the marginal plotters) and divides each frame by its own
    bounce-measure pitch average.  The average uses the exact hat-function
    quadrature: a plain trapezoid against pointwise ``lambda`` steps over
    the narrow trapped-passing peak, and the error would shift A's
    normalization with reconstruction resolution.
    """
    data = derive(cache)
    branch = data["theta"]
    theta_deg = np.asarray(branch["x"], dtype=float)
    theta_rad = np.radians(theta_deg)
    quad = bounce_pitch_quadrature(theta_rad)
    weights = quad / float(np.sum(quad))

    def normalize(marginal):
        marginal = np.asarray(marginal, dtype=float)
        return marginal / float(weights @ marginal)

    initial = normalize(branch["initial"])
    frames = [normalize(frame) for frame in branch["frames"]]

    # Fixed linear y-limits spanning the reference line and every frame, so
    # a movie's axis does not rescale as the solution evolves.
    everything = np.concatenate([initial] + frames)
    finite = everything[np.isfinite(everything)]
    low, high = min(float(finite.min()), 1.0), max(float(finite.max()), 1.0)
    margin = 0.08 * (high - low) if high > low else 0.1
    return {
        "theta_deg": theta_deg,
        "initial": initial,
        "frames": frames,
        "times": data["times"],
        "ylim": (low - margin, high + margin),
    }


# One size for the still and every movie frame -- stated once.
FIGSIZE = (5.9, 3.8)


def draw_frame(fig, ax, data, index):
    """One anisotropy frame: the isotropic reference versus the current shape."""
    ax.axhline(1.0, color="#bbbbbb", lw=0.9)
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
    ax.set_ylabel(r"$A(\theta_0)$", fontsize=13)
    ax.set_title(
        r"$A(\theta_0)=m(\theta_0)/\langle m\rangle_{\lambda}$,"
        r"  $m=\int\mathcal{F}_0\,x_0^2\,dx_0$"
        rf":  $t={data['times'][index]:.2f}\,\tau_c$",
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
    parser = argparse.ArgumentParser(description="ICRF pitch-anisotropy plots")
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
    data = derive_anisotropy(cache)

    # Bind the derived data into the (fig, ax, index) signature render_still
    # and render_movie expect (closures are fine: rendering is in-process).
    def draw(fig, ax, index):
        draw_frame(fig, ax, data, index)

    if do_static:
        save_png(render_still(draw, len(data["times"]) - 1, figsize=FIGSIZE),
                 args.fig_dir, "anisotropy", dpi=220)
    if do_movie:
        render_movie(draw, len(data["times"]),
                     str(Path(args.fig_dir) / "anisotropy.mp4"),
                     figsize=FIGSIZE, fps=args.fps, dpi=args.dpi)


if __name__ == "__main__":
    main()
