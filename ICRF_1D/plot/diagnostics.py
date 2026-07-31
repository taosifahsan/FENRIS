"""Growth-rate diagnostics, one dimension: instantaneous movie + time average.

The 1-D version of the growth pieces of ``ICRF_2D/plot/diagnostics.py``:

1. **Growth-rate movie** -- gamma(v) = (f_next - f_prev) / (dt * |f_prev|),
   the relative growth per collision time, one frame per consecutive
   snapshot pair.  Wherever the solution sits below the numerical error
   floor the rate is defined as exactly zero: sub-floor values are solver
   noise, and noise/noise ratios manufacture growth that is not there.
2. **Average growth** (static) -- the full-span time-average of the
   logarithmic rate, with the same sub-floor-is-zero rule.

3. **Particle conservation** (static) -- relative change of the number
   moment over the run.
4. **Energy and power** (static) -- the energy moment's history and its time
   derivative, the absorbed power.

Used by: the ``plots`` CMake target (and by hand).

Depends on: :mod:`plot_common.reader` (the 1-D snapshot cache, deck, floor),
:mod:`plot_common.static` (``line1d``, ``save_png``), :mod:`plot_common.movie`
(parallel frame rendering).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Locate the directory that holds plot_common by walking up from this file,
# instead of hardcoding a parent depth -- so reorganizing the project tree
# cannot silently break the import.
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

from plot_common.movie import render_movie
from plot_common.reader import (
    load_cache,
    load_snapshots_1d,
    numerical_display_floor,
    option_float,
    read_options,
)
from plot_common.static import save_png


def _moments(cache, solver_input=None):
    """Number and energy histories under the isotropic speed measure.

    N(t) = int f v^2 dv;  E(t) = int (m/2) v^2 f v^2 dv, with the minority
    mass m from the deck -- the same moments the original run.py used for its
    norm and spread, tracked over every frame.
    """
    options = read_options(solver_input or PATHS.solver_input)
    mass = option_float(options, "m")
    v = np.asarray(cache.x, dtype=float)
    dv = v[1] - v[0]
    number, energy = [], []
    for frame in cache.frames:
        f = np.asarray(frame, dtype=float)
        number.append(float(np.sum(f * v**2) * dv))
        energy.append(float(0.5 * mass * np.sum(f * v**4) * dv))
    return np.array(number), np.array(energy)


def derive(cache, solver_input=None):
    """Per-pair growth rates and the run-averaged rate, from the cache."""
    floor = numerical_display_floor(solver_input or PATHS.solver_input)
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
        # ratios manufacture growth that is not there -- so the growth rate
        # of a sub-floor region is defined as exactly zero.
        valid = (np.abs(prev) > floor) & (np.abs(nxt) > floor)
        with np.errstate(divide="ignore", invalid="ignore"):
            raw = (nxt - prev) / (dt * np.abs(prev))
        rates.append(np.where(valid, raw, 0.0))
        times.append(cache.times[index + 1])

    # Full-span time-average of the *logarithmic* rate, with the same rule as
    # the movie: sub-floor pairs contribute exactly zero.  ln(R)/dt rather
    # than (R-1)/dt so the frames where the heating front arrives (R huge)
    # do not swamp the mean; zero-not-excluded means every velocity averages
    # over the same window, so the curve has no per-velocity window seams.
    frames = [np.asarray(f, dtype=float) for f in cache.frames]
    log_sum = np.zeros_like(frames[0])
    for index in range(len(frames) - 1):
        dt = cache.times[index + 1] - cache.times[index]
        if dt <= 0.0:
            continue
        prev, nxt = frames[index], frames[index + 1]
        valid = (np.abs(prev) > floor) & (np.abs(nxt) > floor)
        with np.errstate(divide="ignore", invalid="ignore"):
            step = np.log(np.abs(nxt) / np.abs(prev))
        log_sum += np.where(valid, step, 0.0)
    average = log_sum / (cache.times[-1] - cache.times[0])

    number, energy = _moments(cache, solver_input)
    frame_times = np.asarray(cache.times, dtype=float)
    power = np.gradient(energy, frame_times)

    span = cache.times[-1] - cache.times[0]
    return {"v": v, "rates": rates, "times": times,
            "average": average, "span": span,
            "frame_times": frame_times, "number": number,
            "energy": energy, "power": power}


# One size for the still and every movie frame -- stated once.
FIGSIZE = (6.5, 4.2)


def draw_frame(fig, ax, data, index):
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
    ax.set_xlabel(r"$v/v_{th}$", fontsize=13)
    ax.set_ylabel(r"$\partial_t f \,/\, |f|$  $[1/\tau]$", fontsize=13)
    ax.set_title(rf"growth rate:  $t = {data['times'][index]:.2f}\,\tau$",
                 fontsize=12)
    ax.grid(alpha=0.25)
    fig.tight_layout()


def plot_average(data):
    """Run-averaged logarithmic growth rate versus velocity."""
    fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    ax.axhline(0.0, color="#999999", lw=0.8)
    ax.plot(data["v"], data["average"], color="#2d1e8f", lw=1.8)
    ax.set_xlim(data["v"][0], data["v"][-1])
    ax.set_xlabel(r"$v/v_{th}$", fontsize=13)
    ax.set_ylabel(r"$\langle \partial_t \ln|f| \rangle$  $[1/\tau]$",
                  fontsize=13)
    ax.set_title(rf"mean growth rate, $t = 0..{data['span']:.0f}\,\tau$ (0 below the noise floor)",
                 fontsize=12)
    ax.grid(alpha=0.25)
    return fig


def plot_particles(data):
    """Relative particle-number change versus time: the conservation check."""
    fig, ax = plt.subplots(figsize=(6.5, 4.0), constrained_layout=True)
    ax.axhline(0.0, color="#999999", lw=0.8)
    ax.plot(data["frame_times"], 100.0 * (data["number"] / data["number"][0] - 1.0),
            color="#2d1e8f", lw=1.8)
    ax.set_xlabel("simulation time", fontsize=13)
    ax.set_ylabel(r"$\Delta N / N_0$  [%]", fontsize=13)
    ax.set_title("particle conservation", fontsize=12)
    ax.grid(alpha=0.25)
    return fig


def plot_energy_power(data):
    """Total energy and the absorbed power (its time derivative), stacked."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.5, 6.2), sharex=True,
                                   constrained_layout=True)
    ax1.plot(data["frame_times"], data["energy"], color="#2d1e8f", lw=1.8)
    ax1.set_ylabel("total energy", fontsize=13)
    ax1.set_title("energy and absorbed power", fontsize=12)
    ax1.grid(alpha=0.25)
    ax2.axhline(0.0, color="#999999", lw=0.8)
    ax2.plot(data["frame_times"], data["power"], color="#118ab2", lw=1.8)
    ax2.set_xlabel("simulation time", fontsize=13)
    ax2.set_ylabel(r"power, $dE/dt$", fontsize=13)
    ax2.grid(alpha=0.25)
    return fig


def main():
    parser = argparse.ArgumentParser(description="ICRF_1D growth diagnostics")
    parser.add_argument("--static", action="store_true")
    parser.add_argument("--movie", action="store_true")
    parser.add_argument("-o", "--output", default=str(PATHS.snapshots))
    parser.add_argument("--cache", default=None,
                        help="load the shared cache.npz instead of reading snapshots")
    parser.add_argument("--fig-dir", default=str(PATHS.figures))
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
            options = read_options(PATHS.solver_input)
            points = int(option_float(options, "num_points", 256) / 2)
        cache = load_snapshots_1d(args.output, points)
    data = derive(cache)

    if do_static:
        save_png(plot_average(data), args.fig_dir, "growth_average", dpi=220)
        save_png(plot_particles(data), args.fig_dir, "particle_loss", dpi=220)
        save_png(plot_energy_power(data), args.fig_dir, "energy_power", dpi=220)
    if do_movie:
        def draw(fig, ax, index):
            draw_frame(fig, ax, data, index)

        render_movie(draw, len(data["rates"]),
                     str(Path(args.fig_dir) / "growth_rate.mp4"),
                     figsize=FIGSIZE, fps=args.fps, dpi=args.dpi)


if __name__ == "__main__":
    main()
