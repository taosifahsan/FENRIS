"""The conserved speed density ``n(x_0)`` and the shared reduction machinery.

Owns the marginal reduction of the 2-D solution: the ``_integral_spec`` /
:func:`derive` machinery lives here, and ``temperature.py`` imports it (one
reduction pass covers both when run from the same cache).  By deliberate
convention (see
``bounce_pitch_weight``'s docstring in ``coefficients.py``) each reduction
inside :func:`derive` carries only the ELIMINATED coordinate's measure
factor, never the surviving one's -- that keeps the shape-preserving
marginal available to ``temperature.py``, whose log-derivative diagnostic
must not have the thermal core sent to zero by an ``x_0^2`` factor.  This
file's own plot then multiplies the surviving factor back in at the call
site, producing the true conserved density

    n(x_0) = 2 pi x_0^2 * integral F_0(x_0, theta_0) lambda(theta_0) sin(theta_0) dtheta_0

whose *area is the full 3-D particle count* -- gyrophase ``2 pi``
included -- exactly 1 at t=0 by the initial-condition normalization
(the same convention as LHCD_2D); ``integral n dx_0`` is ``2 pi *``
``moments.py``'s number moment, frame by frame.  The multiplication happens
here at the call site, not inside the shared reduction -- both quantities
are legitimate and different callers need each.

Drawn on a signed-log y-axis, matching the shape plots: the density spans
decades, and :func:`plot_common.static.line1d` splits any negative
reconstruction ringing into its own branch rather than hiding it.  The
measure's zero at ``x_0 = 0`` dives below the display floor.

Used by: ``tools/run.sh`` (one of the parallel plotter processes).

Used also by: ``temperature.py``, which imports :func:`derive` (and
``FIGSIZE``) from here.

Depends on: ``coefficients.py`` (the bounce weight and the initial
condition), :mod:`plot_common.reader` (the cache), :mod:`plot_common.static`
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

from coefficients import bounce_pitch_quadrature, initial_condition_grid
from plot_common.movie import render_movie
from plot_common.reader import load_cache, load_snapshots, numerical_display_floor
from plot_common.static import line1d, render_still, save_png

REDUCTIONS = ("vel", "theta")


def _grid_axes(x0, theta0):
    """Return ``(velocity_axis, pitch_axis)`` for a reconstruction mesh.

    ASGarD's axis order is not guaranteed across configurations, so it is
    determined empirically: whichever axis ``theta0`` actually varies along is
    the pitch axis (``x0`` is constant along that axis on a rectilinear
    reconstruction grid, and vice versa).  Guessing wrong would integrate over
    the wrong coordinate while still producing plausible-looking output, so
    this is checked rather than assumed.  Matches ``coefficients.py``'s
    ``_pitch_axis`` exactly -- both are the same check, kept local to each
    file rather than shared.
    """
    theta_change_0 = np.nanmax(np.abs(np.diff(theta0, axis=0)))
    theta_change_1 = np.nanmax(np.abs(np.diff(theta0, axis=1)))
    pitch_axis = 0 if theta_change_0 > theta_change_1 else 1
    return 1 - pitch_axis, pitch_axis


def _trapezoid_weights(coord):
    """Composite-trapezoid integration weights for a 1-D coordinate vector.

    Returns weights ``w`` such that ``sum(w * f)`` approximates
    ``integral(f dcoord)``.  Interior points get half the span to each
    neighbour; the two endpoints get half of their single adjacent interval.
    Handles nonuniform spacing correctly, which adaptive reconstruction grids
    need.
    """
    coord = np.asarray(coord, dtype=float)
    if coord.size < 2:
        return np.ones_like(coord)
    weights = np.empty_like(coord)
    weights[0] = 0.5 * (coord[1] - coord[0])
    weights[-1] = 0.5 * (coord[-1] - coord[-2])
    weights[1:-1] = 0.5 * (coord[2:] - coord[:-2])
    return weights


def _integral_spec(x0, theta0, reduction, solver_input, table_dir):
    """Build the axis/order/weight spec for one reduction, once per movie.

    Returns ``(x_axis_values, spec)`` where ``spec`` is reused by
    :func:`_apply` for every frame -- the weights depend only on the fixed
    reconstruction grid, not on the frame's data.
    """
    velocity_axis, pitch_axis = _grid_axes(x0, theta0)

    if reduction == "theta":
        theta = np.mean(theta0, axis=velocity_axis)
        x = np.mean(x0, axis=pitch_axis)
        order = np.argsort(x)
        x = x[order]
        # x^2 is the spherical velocity-space volume element.
        weights = _trapezoid_weights(x) * x * x
        return np.degrees(theta), {"axis": velocity_axis, "order": order,
                                   "weights": weights}

    x = np.mean(x0, axis=pitch_axis)
    theta = np.mean(theta0, axis=velocity_axis)
    order = np.argsort(theta)
    theta = theta[order]
    # Bounce-orbit measure lambda(theta)*sin(theta), matching the solver's
    # conserved phase-space mass -- integrated exactly against this grid's
    # hat functions (see bounce_pitch_quadrature: pointwise lambda under a
    # plain trapezoid steps over the trapped-passing peak, and the error
    # moves with reconstruction resolution).
    weights = bounce_pitch_quadrature(theta, solver_input, table_dir)
    return x, {"axis": pitch_axis, "order": order, "weights": weights}


def _apply(f, spec):
    """Reduce one 2-D frame to a 1-D marginal using a precomputed spec."""
    f_line = np.moveaxis(f, spec["axis"], -1)[..., spec["order"]]
    return np.asarray(f_line @ spec["weights"], dtype=float)


def _semilog_magnitudes(values):
    """Flatten a signed array into its finite |value| pieces, for a y-limit.

    Distinct from the *drawing* split in ``plot_common.static.line1d``: this
    is purely a magnitude scan used to choose an axis range, so both signs
    fold into one array rather than staying separate for plotting.
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
    solution evolves.  The floor sets the lower bound so numerical noise does
    not stretch the axis into a decade with no real structure.
    """
    spectra = [_semilog_magnitudes(initial)]
    spectra.extend(_semilog_magnitudes(frame) for frame in frames)
    spectra = [values for values in spectra if values.size]
    if not spectra:
        return None
    finite = np.concatenate(spectra)
    return floor, float(np.max(finite)) * 1.1


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


def derive(cache, solver_input=None, table_dir=None):
    """Compute both marginals from the shared snapshot cache.

    One reduction spec per marginal is built from the cache's fixed
    coordinate mesh, then applied to every already-read frame -- no snapshot
    is opened here.
    """
    floor = numerical_display_floor(solver_input or PATHS.solver_input)
    f_initial = initial_condition_grid(cache.x, cache.y, solver_input, table_dir)

    result = {"times": cache.times, "floor": floor}
    for reduction in REDUCTIONS:
        x, spec = _integral_spec(cache.x, cache.y, reduction, solver_input, table_dir)
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



def derive_density(cache):
    """Weight the pitch-integrated marginal into the conserved density n(x_0).

    Reuses :func:`derive` above (one reduction pass, shared with
    ``temperature.py``) and multiplies each frame by
    the surviving-coordinate measure ``x_0^2``.  Also integrates each frame's
    density so the drawing can print the running particle number -- the
    quantity whose conservation this plot exists to display.
    """
    data = derive(cache)
    branch = data["vel"]
    x = np.asarray(branch["x"], dtype=float)
    # 2 pi: the gyrophase integral, made explicit so the area is the full
    # 3-D particle count rather than the bare 2-D weighted moment.
    weight = 2.0 * np.pi * x * x
    initial = weight * np.asarray(branch["initial"], dtype=float)
    frames = [weight * np.asarray(frame, dtype=float) for frame in branch["frames"]]
    numbers = [float(np.trapezoid(frame, x)) for frame in frames]

    # Signed-log y-limits, same convention as the shape plots: the floor
    # bounds the bottom so measure zeros and reconstruction noise dive below
    # the axis instead of compressing it, and the top spans every frame so a
    # movie's axis does not rescale as the solution evolves.
    everything = np.concatenate([np.abs(initial)] + [np.abs(fr) for fr in frames])
    finite = everything[np.isfinite(everything) & (everything > 0.0)]
    low, high = float(data["floor"]), float(finite.max()) * 1.1
    return {
        "x": x,
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
        ax, data["x"],
        [
            (data["initial"], "initial", {"color": "#999999",
                                          "linestyle": "--", "lw": 1.8}),
            (data["frames"][index], "current", {"color": "#118ab2",
                                                "linestyle": "-", "lw": 2.0}),
        ],
        scale="log", ylim=data["ylim"], legend=False,
    )
    ax.set_xlabel(r"$x_0$", fontsize=13)
    ax.set_ylabel(r"$n(x_0)$", fontsize=13)
    ax.set_title(
        r"$n(x_0)=2\pi x_0^2\int\mathcal{F}_0\,\lambda(\theta_0)\sin\theta_0\,d\theta_0$"
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
    parser = argparse.ArgumentParser(description="ICRF speed-density plots")
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
                 args.fig_dir, "density_x", dpi=220)
    if do_movie:
        render_movie(draw, len(data["times"]),
                     str(Path(args.fig_dir) / "density_x.mp4"),
                     figsize=FIGSIZE, fps=args.fps, dpi=args.dpi)


if __name__ == "__main__":
    main()
