"""The speed marginal (``vel_smoothed``) and the shared reduction machinery.

Two 1-D reductions of the same 2-D solution, computed from one snapshot cache
pass:

  ``vel``    -- integrate out pitch angle, giving a marginal versus speed ``x``
  ``theta``  -- integrate out speed, giving a marginal versus pitch angle
               (degrees)

The one physics difference from ``ICRF_2D/plot/smoothed.py``: LHCD's pitch
measure is plain ``sin(theta)``, because LHCD has no magnetic geometry and so
no bounce-orbit weight.  Everything else -- the reduction machinery, the fixed
y-limits, the drawing -- is the same idea.

Used by: ``tools/run.sh`` (one of the parallel plotter processes), and
``theta_smoothed.py``, which imports :func:`derive`, :func:`plot_static`,
and :func:`plot_movie` from here and only picks the other reduction.

Depends on: :mod:`plot_common.reader` (the cache), :mod:`plot_common.static`
(drawing), :mod:`plot_common.movie` (parallel frame rendering),
``coefficients.py`` (the initial condition).  The grid-orientation and
trapezoid-weight helpers are small enough to keep local to this file.
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

from coefficients import initial_condition_grid
from plot_common.movie import render_movie
from plot_common.reader import load_cache, load_snapshots, numerical_display_floor
from plot_common.static import line1d, render_still, save_png

REDUCTIONS = ("vel", "theta")


def _grid_axes(x, theta):
    """Return ``(velocity_axis, pitch_axis)`` for a reconstruction mesh.

    ASGarD's axis order is not guaranteed across configurations, so it is
    determined empirically: whichever axis ``theta`` actually varies along is
    the pitch axis.  Guessing wrong would integrate over the wrong coordinate
    while still producing plausible-looking output.
    """
    change0 = np.nanmax(np.abs(np.diff(theta, axis=0))) if theta.shape[0] > 1 else 0.0
    change1 = np.nanmax(np.abs(np.diff(theta, axis=1))) if theta.shape[1] > 1 else 0.0
    pitch_axis = 0 if change0 > change1 else 1
    return 1 - pitch_axis, pitch_axis


def _trapezoid_weights(coord):
    """Composite-trapezoid integration weights for a 1-D coordinate vector.

    Returns weights ``w`` such that ``sum(w * f)`` approximates
    ``integral(f dcoord)``.  Interior points get half the span to each
    neighbour; the endpoints get half of their single adjacent interval.
    Precomputing these once turns each frame's reduction into a single
    matrix-vector product instead of a per-frame ``np.trapezoid`` call.
    """
    coord = np.asarray(coord, dtype=float)
    if coord.size < 2:
        return np.ones_like(coord)
    weights = np.empty_like(coord)
    weights[0] = 0.5 * (coord[1] - coord[0])
    weights[-1] = 0.5 * (coord[-1] - coord[-2])
    weights[1:-1] = 0.5 * (coord[2:] - coord[:-2])
    return weights


def _integral_spec(x0, theta0, reduction):
    """Build the axis/order/weight spec for one reduction, once per movie.

    The weights depend only on the fixed reconstruction grid, not on any
    frame's data, so they are computed once and reused by :func:`_apply` for
    every frame.
    """
    velocity_axis, pitch_axis = _grid_axes(x0, theta0)

    if reduction == "theta":
        theta = np.take(theta0, 0, axis=velocity_axis)
        x = np.take(x0, 0, axis=pitch_axis)
        order = np.argsort(x)
        x = x[order]
        # x^2 is the spherical velocity-space volume element.
        weights = _trapezoid_weights(x) * x * x
        return np.degrees(theta), {"axis": velocity_axis, "order": order,
                                   "weights": weights}

    x = np.take(x0, 0, axis=pitch_axis)
    theta = np.take(theta0, 0, axis=velocity_axis)
    order = np.argsort(theta)
    theta = theta[order]
    # Plain sin(theta): LHCD has no orbit geometry, hence no bounce weight.
    weights = _trapezoid_weights(theta) * np.sin(theta)
    return x, {"axis": pitch_axis, "order": order, "weights": weights}


def _apply(f, spec):
    """Reduce one 2-D frame to a 1-D marginal using a precomputed spec."""
    f_line = np.moveaxis(f, spec["axis"], -1)[..., spec["order"]]
    return np.asarray(f_line @ spec["weights"], dtype=float)


def _semilog_magnitudes(values):
    """Flatten a signed array into its finite |value| pieces, for a y-limit.

    Distinct from the *drawing* split in :func:`plot_common.static.line1d`:
    this is purely a magnitude scan used to choose an axis range, so both
    signs fold into one array.
    """
    data = np.asarray(values, dtype=float)
    positive = data[np.isfinite(data) & (data > 0.0)]
    negative = -data[np.isfinite(data) & (data < 0.0)]
    pieces = [p for p in (positive, negative) if p.size]
    if not pieces:
        return np.array([], dtype=float)
    return np.concatenate(pieces)


def _ylim(initial, frames, floor):
    """Signed-log y-limits spanning the initial curve and every frame.

    Fixed up front (not per-frame) so a movie's axis does not rescale as the
    solution evolves.  The floor bounds the bottom so numerical noise cannot
    stretch the axis into a decade with no real structure.
    """
    spectra = [_semilog_magnitudes(initial)]
    spectra.extend(_semilog_magnitudes(frame) for frame in frames)
    spectra = [values for values in spectra if values.size]
    if not spectra:
        return None
    return floor, float(np.max(np.concatenate(spectra))) * 1.1


def _linear_ylim(initial, frames, pad=0.08):
    """Fixed y-limits for a linear-scale marginal, spanning every frame.

    Unlike :func:`_ylim`'s positive noise floor -- needed because a log axis
    cannot show zero or negative values -- a linear axis has nothing to hide,
    so this keeps the true signed min/max.  Padding is a fraction of the
    spanned range rather than a multiplicative factor, so it behaves sensibly
    even when the curve is flat or straddles zero.
    """
    values = np.concatenate(
        [np.asarray(initial, dtype=float).ravel()]
        + [np.asarray(frame, dtype=float).ravel() for frame in frames]
    )
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    low, high = float(finite.min()), float(finite.max())
    margin = pad * (high - low) if high > low else (abs(high) * pad or 1.0)
    return low - margin, high + margin


def derive(cache, solver_input=None):
    """Compute both marginals from the shared snapshot cache.

    No snapshot is opened here -- every frame is already in the cache.
    """
    floor = numerical_display_floor(solver_input or PATHS.solver_input)
    f_initial = initial_condition_grid(cache.x)

    result = {"times": cache.times, "floor": floor}
    for reduction in REDUCTIONS:
        x, spec = _integral_spec(cache.x, cache.y, reduction)
        initial = _apply(f_initial, spec)
        frames = [_apply(frame, spec) for frame in cache.frames]
        # theta is plotted on a linear y-axis (see draw_frame); vel stays
        # signed-log, so each needs its own kind of fixed y-limit.
        ylim = (_linear_ylim(initial, frames) if reduction == "theta"
               else _ylim(initial, frames, floor))
        result[reduction] = {
            "x": x,
            "initial": initial,
            "frames": frames,
            "ylim": ylim,
        }
    return result


_LABELS = {
    "vel": (r"$x=v/v_{te}$",
           r"$\int f(x,\theta)\sin\theta\,d\theta$"),
    "theta": (r"$\theta$ [deg]",
             r"$\int f(x,\theta)x^2\,dx$"),
}


# One size for the still and every movie frame -- stated once.
FIGSIZE = (5.9, 3.8)


def draw_frame(fig, ax, data, reduction, index):
    """Draw one marginal frame: initial (dashed) versus current (solid).

    theta is linear (its magnitude doesn't span decades, and a linear axis
    shows its shape more directly); vel stays signed-log.
    """
    branch = data[reduction]
    xlabel, title = _LABELS[reduction]
    scale = "linear" if reduction == "theta" else "log"
    line1d(
        ax, branch["x"],
        [
            (branch["initial"], "initial",
             {"color": "#171717", "linestyle": "--", "lw": 1.8}),
            (branch["frames"][index], "current",
             {"color": "#118ab2", "linestyle": "-", "lw": 2.0}),
        ],
        scale=scale, ylim=branch["ylim"], legend=False,
    )
    time_label = rf"time, $t = {data['times'][index]:.2f}\,\tau_c$"
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel("amplitude", fontsize=13)
    ax.set_title(f"{title}:  {time_label}", fontsize=12)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()


def main():
    parser = argparse.ArgumentParser(description="LHCD speed-marginal plots")
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
    data = derive(cache)

    def draw(fig, ax, index):
        draw_frame(fig, ax, data, "vel", index)

    if do_static:
        save_png(render_still(draw, len(data["times"]) - 1, figsize=FIGSIZE),
                 args.fig_dir, "vel_smoothed", dpi=220)
    if do_movie:
        render_movie(draw, len(data["times"]),
                     str(Path(args.fig_dir) / "vel_smoothed.mp4"),
                     figsize=FIGSIZE, fps=args.fps, dpi=args.dpi)


if __name__ == "__main__":
    main()
