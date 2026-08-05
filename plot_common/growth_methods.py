"""The fractional growth rate ``gamma_f = d(ln f)/dt`` -- 1-D and 2-D.

The one genuinely identical piece across all four projects, so it lives here
and each ``plot/growth.py`` is a thin adapter supplying the initial-condition
shape and the labels.  The moments (particle conservation, energy, power) are
NOT here: their measures differ per project, so each keeps its own
``plot/moments.py``.

Two outputs:

  ``--static``  one overall time average over the whole run.
  ``--movie``   a short (1 tau_c) rolling average per frame.

**The time average is computed from the endpoints only.**  Since
``gamma_f = d(ln f)/dt``, its time average telescopes exactly:

    <gamma_f>_T = (1/T) integral_0^T d(ln f)/dt dt = ln[f(T)/f(0)] / T

so the whole frame stack is unnecessary for it -- and the result is immune to
pixels that dip below the noise floor partway through, which make a
frame-by-frame average silently report a mean over a shorter interval than
its label claims.

``f(0)`` is not read back from the first snapshot alone.  Where the initial
condition sits below the floor -- the fast-tail region, exactly the part
worth measuring -- that reconstruction is noise and the log ratio would mask
out.  But ``f(0)`` is an input: each project passes the closed-form SHAPE
from its ``initial_condition.hpp``, and the amplitude is calibrated against
the solver's own frame 0 where that IS resolvable.  Calibrating rather than
reproducing the solver's normalization means this never has to track the
solver's quadrature, domain, or measure convention -- and it is
self-checking, since a wrong shape would not produce a constant ratio.

Used by: each project's ``plot/growth.py``, in turn by ``tools/run.sh``.

Depends on: :mod:`plot_common.reader` (cache, deck, noise floor),
:mod:`plot_common.static` (drawing), :mod:`plot_common.movie` (rendering).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_common.movie import render_movie
from plot_common.reader import (
    load_cache,
    load_snapshots,
    load_snapshots_1d,
    numerical_display_floor,
    option_float,
    read_options,
)
from plot_common.static import (
    contour2d,
    movie_scale_range,
    save_png,
)

# Width, in simulation time, of the centered moving average applied to movie
# frames.  Frame-to-frame growth rates divide two nearly equal numbers and are
# extremely noisy; averaging over a fixed time window makes the physical trend
# visible.  The static figure does not use this -- see the module docstring.
GROWTH_AVERAGE_WIDTH = 1.0


def _calibrated_initial(cache, shape, floor):
    """The initial condition: solver's own frame 0, extended by ``shape``.

    Where frame 0 clears the floor it is used directly -- that is the state
    the run actually started from, and it departs from the closed form by up
    to ~10% at the reconstruction's worst points.  Below the floor the
    reconstruction is noise, so ``shape`` takes over, rescaled by the median
    ratio between the two across the resolvable region.
    """
    shape = np.asarray(shape, dtype=float)
    start = np.asarray(cache.frames[0], dtype=float)
    known = (start > floor) & (shape > 0.0)
    scale = float(np.median(start[known] / shape[known])) if np.any(known) else 0.0
    return np.where(start > floor, start, shape * scale)


def endpoint_average(cache, shape, floor):
    """``ln[f(T)/f(0)] / T`` per point, NaN where ``f(T)`` is unresolvable."""
    times = np.asarray(cache.times, dtype=float)
    total_time = float(times[-1] - times[0])
    start = _calibrated_initial(cache, shape, floor)
    final = np.asarray(cache.frames[-1], dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        average = np.where(final > floor,
                           np.log(final / start) / total_time, np.nan)
    return average, float(times[0]), float(times[-1])


# ---------------------------------------------------------------------------
# One dimension
# ---------------------------------------------------------------------------


def derive_1d(cache, solver_input, initial_shape):
    """Per-pair rates for the movie, plus the endpoint time average."""
    floor = numerical_display_floor(solver_input)
    options = read_options(solver_input)
    v = np.asarray(cache.x, dtype=float)

    rates, times = [], []
    for index in range(len(cache.frames) - 1):
        dt = cache.times[index + 1] - cache.times[index]
        if dt <= 0.0:
            continue
        prev = np.asarray(cache.frames[index], dtype=float)
        nxt = np.asarray(cache.frames[index + 1], dtype=float)
        # Below the numerical floor (solver tolerances or machine error,
        # whichever is larger) the "solution" is noise, and noise/noise
        # ratios manufacture growth that is not there -- so the rate of a
        # sub-floor region is defined as exactly zero.
        valid = (np.abs(prev) > floor) & (np.abs(nxt) > floor)
        with np.errstate(divide="ignore", invalid="ignore"):
            raw = (nxt - prev) / (dt * np.abs(prev))
        rates.append(np.where(valid, raw, 0.0))
        times.append(cache.times[index + 1])

    average, _, _ = endpoint_average(cache, initial_shape(v, options), floor)
    return {"v": v, "rates": rates, "times": times, "average": average,
            "span": cache.times[-1] - cache.times[0]}


FIGSIZE_1D = (6.5, 4.2)


def draw_frame_1d(fig, ax, data, index, xlabel):
    """One instantaneous growth-rate frame, fixed symmetric axes."""
    ax.axhline(0.0, color="#999999", lw=0.8)
    rate = np.asarray(data["rates"][index], dtype=float)
    ax.plot(data["v"], rate, color="#118ab2", lw=1.6)
    ax.set_xlim(data["v"][0], data["v"][-1])
    # Per-frame symmetric limits (99.5th percentile so one noisy point does
    # not set the scale): the rate collapses over the run, and a fixed range
    # sized for the violent early frames would flatten the late ones.
    frame_peak = float(np.nanpercentile(np.abs(rate), 99.5)) or 1.0
    ax.set_ylim(-1.1 * frame_peak, 1.1 * frame_peak)
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(r"$\partial_t f \,/\, |f|$  $[1/\tau]$", fontsize=13)
    ax.set_title(rf"growth rate:  $t = {data['times'][index]:.2f}\,\tau$",
                 fontsize=12)
    ax.grid(alpha=0.25)
    fig.tight_layout()


def plot_average_1d(data, xlabel):
    """Run-averaged logarithmic growth rate versus velocity."""
    fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    ax.axhline(0.0, color="#999999", lw=0.8)
    ax.plot(data["v"], data["average"], color="#2d1e8f", lw=1.8)
    ax.set_xlim(data["v"][0], data["v"][-1])
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(r"$\langle \partial_t \ln|f| \rangle$  $[1/\tau]$",
                  fontsize=13)
    ax.set_title(rf"mean growth rate, $t = 0..{data['span']:.0f}\,\tau$",
                 fontsize=12)
    ax.grid(alpha=0.25)
    return fig


def main_1d(paths, *, description, xlabel, initial_shape):
    """CLI entry point for a 1-D project's ``plot/growth.py``.

    ``initial_shape(v, options)`` returns the *unnormalized* shape of
    ``initial_condition.hpp``'s ``initial_f0``; only the shape has to be
    right, since the amplitude is calibrated against the run.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--static", action="store_true")
    parser.add_argument("--movie", action="store_true")
    parser.add_argument("-o", "--output", default=str(paths.snapshots))
    parser.add_argument("--cache", default=None,
                        help="load the shared cache.npz instead of reading snapshots")
    parser.add_argument("--fig-dir", default=str(paths.figures))
    parser.add_argument("-n", "--points", type=int, default=0,
                        help="reconstruction points (0 = deck num_points / 2)")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--dpi", type=int, default=140)
    args = parser.parse_args()
    do_static = args.static or not (args.static or args.movie)
    do_movie = args.movie or not (args.static or args.movie)

    if args.cache:
        cache = load_cache(args.cache)
    else:
        points = args.points
        if points <= 0:
            options = read_options(paths.solver_input)
            points = int(option_float(options, "num_points", 256) / 2)
        cache = load_snapshots_1d(args.output, points)
    data = derive_1d(cache, paths.solver_input, initial_shape)

    if do_static:
        save_png(plot_average_1d(data, xlabel), args.fig_dir,
                 "growth_average", dpi=220)
    if do_movie:
        # Bind the derived data into the (fig, ax, index) signature
        # render_movie expects (closures are fine: rendering is in-process).
        def draw(fig, ax, index):
            draw_frame_1d(fig, ax, data, index, xlabel)

        render_movie(draw, len(data["rates"]),
                     str(Path(args.fig_dir) / "growth_rate.mp4"),
                     figsize=FIGSIZE_1D, fps=args.fps, dpi=args.dpi)


# ---------------------------------------------------------------------------
# Two dimensions
# ---------------------------------------------------------------------------


def _short_time_average(frames, centers, width):
    """Centered moving average of a frame series, over a fixed *time* window.

    Averaging over a fixed time width (rather than a fixed frame count) keeps
    the physical smoothing scale constant even when snapshot spacing varies.
    Returns ``(averaged_frames, (first_time, last_time) bounds)``.

    The stack and its finite mask are built once, before the loop:
    reprocessing every frame per window it appears in -- the naive approach --
    costs a large multiple for a wide window over dense snapshots.
    """
    if not frames:
        return [], []
    centers = np.asarray(centers, dtype=float)
    half_width = 0.5 * float(width)
    stack = np.asarray(frames, dtype=np.float32)
    stack_finite = np.isfinite(stack)

    averaged_frames = []
    bounds = []
    for center in centers:
        selected = np.flatnonzero(np.abs(centers - center) <= half_width + 1.0e-14)
        first, last = int(selected[0]), int(selected[-1]) + 1
        block = stack[first:last]
        finite = stack_finite[first:last]
        # Accumulate in float64: summing many float32 values loses precision.
        total = np.sum(np.where(finite, block, 0.0), axis=0, dtype=np.float64)
        count = np.count_nonzero(finite, axis=0)
        average = np.full(block.shape[1:], np.nan, dtype=np.float32)
        valid = count > 0
        average[valid] = total[valid] / count[valid]
        averaged_frames.append(average)
        bounds.append((float(centers[first]), float(centers[last - 1])))
    return averaged_frames, bounds


def derive_2d(cache, floor, shape):
    """Movie frames (smoothed per-pair rates) plus the endpoint time average.

    ``cache.frame_pairs()`` supplies each interior snapshot exactly once, so
    the movie frames are essentially free once the cache exists -- versus the
    double read a naive per-pair implementation would cost.

    A movie pixel only counts where both endpoints are finite *and* above the
    noise floor: dividing by an ``f`` that is numerical noise yields a
    meaningless enormous rate.
    """
    raw_frames = []
    centers = []
    for index, previous, current, dt in cache.frame_pairs():
        reliable = (
            np.isfinite(previous) & np.isfinite(current)
            & (previous > floor) & (current > floor)
        )
        growth = np.full(np.shape(current), np.nan, dtype=np.float32)
        growth[reliable] = (
            (current[reliable] - previous[reliable]) / (dt * current[reliable])
        )
        raw_frames.append(growth)
        centers.append(0.5 * (cache.times[index] + cache.times[index + 1]))

    frames, bounds = _short_time_average(raw_frames, centers,
                                         GROWTH_AVERAGE_WIDTH)
    average, time_start, time_end = endpoint_average(cache, shape, floor)
    return {
        "frames": frames,
        "bounds": bounds,
        "average": np.ma.masked_invalid(average),
        "scale": movie_scale_range(frames, subsample=4096),
        "time_start": time_start,
        "time_end": time_end,
    }


FIGSIZE_2D = (5.4, 4.5)


def draw_frame_2d(fig, ax, vpar, vperp, style_axes, growth, index, symbol):
    """Draw one short-time-averaged growth-rate frame."""
    time_previous, time_current = growth["bounds"][index]
    contour2d(
        fig, ax, vpar, vperp, growth["frames"][index],
        fixed_range=growth["scale"], filled=True,
        style_axes=style_axes, extend="both",
        title=(
            r"$1\,\tau_c$-averaged growth rate, "
            rf"$\gamma_f=(\Delta {symbol}/\Delta t)/{symbol}$"
            "\n"
            rf"$t={time_previous:.3f}\rightarrow{time_current:.3f}\,\tau_c$"
        ),
    )
    fig.subplots_adjust(left=0.13, right=0.88, bottom=0.13, top=0.84)


def plot_average_2d(vpar, vperp, style_axes, growth, symbol):
    """Overall time-averaged growth-rate contour (data-derived scale)."""
    fig, ax = plt.subplots(figsize=FIGSIZE_2D)
    contour2d(
        fig, ax, vpar, vperp, growth["average"], filled=True,
        style_axes=style_axes, extend="both",
        title=(
            r"Time-averaged growth rate, "
            rf"$\langle\gamma_f\rangle_t=\langle(\Delta {symbol}"
            rf"/\Delta t)/{symbol}\rangle_t$"
            "\n"
            rf"$t={growth['time_start']:.3f}\rightarrow"
            rf"{growth['time_end']:.3f}\,\tau_c$"
        ),
    )
    fig.subplots_adjust(left=0.13, right=0.88, bottom=0.13, top=0.84)
    return fig


def main_2d(paths, *, description, symbol, style_axes, initial_shape):
    """CLI entry point for a 2-D project's ``plot/growth.py``.

    ``initial_shape(cache)`` returns the *unnormalized* initial condition on
    the cache grid; its amplitude is calibrated against the run.
    """
    from plot_common.static import cartesian_mesh

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--static", action="store_true")
    parser.add_argument("--movie", action="store_true")
    parser.add_argument("-o", "--output", default=str(paths.snapshots))
    parser.add_argument("--cache", default=None,
                        help="load the shared cache.npz instead of reading snapshots")
    parser.add_argument("--fig-dir", default=str(paths.figures))
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

    floor = numerical_display_floor(paths.solver_input)
    growth = derive_2d(cache, floor, initial_shape(cache))
    vpar, vperp = cartesian_mesh(cache.x, cache.y)

    if do_static:
        save_png(plot_average_2d(vpar, vperp, style_axes, growth, symbol),
                 args.fig_dir, "growth_rate", dpi=220)
    if do_movie:
        def draw(fig, ax, index):
            draw_frame_2d(fig, ax, vpar, vperp, style_axes, growth, index,
                          symbol)

        render_movie(draw, len(growth["frames"]),
                     str(Path(args.fig_dir) / "growth_rate.mp4"),
                     figsize=FIGSIZE_2D, fps=args.fps, dpi=args.dpi)
