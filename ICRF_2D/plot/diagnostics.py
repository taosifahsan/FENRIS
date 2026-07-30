"""Growth rate, particle conservation, energy, and power.

Four outputs, all derived from the one shared snapshot cache -- no snapshot is
read a second time for any of these:

  ``growth``        -- the fractional rate ``(df/dt)/f``, time-averaged.
                       ``--static`` shows one overall time average;
                       ``--movie`` shows a short (1 tau_c) rolling average per
                       frame.
  ``particle_loss``  -- the fractional change in the solver's own conserved
                       phase-space moment (weighted particle number).
  ``energy_power``    -- the same conserved-moment machinery's second moment
                       (energy), plus its smoothed time-derivative (power).

**On the missing radial flux.** This file used to also compute the boundary
radial flux ``J_x`` at the outer velocity wall, by least-squares extrapolating
``f`` and its derivatives into the last grid cell and combining them with the
collisional and quasilinear coefficients evaluated there.  That was the most
intricate and most fragile part of the diagnostics and never produced a
physically reasonable result, so it has been removed rather than left in
place emitting numbers nobody trusts.  Particle loss answers the same
question -- "are particles escaping, and how fast?" -- directly from a
conserved moment, with no extrapolation and no coefficient evaluation at the
boundary.

``LHCD_2D/plot/diagnostics.py`` is the direct counterpart -- same filename,
same role, same four figures.  The two differ only in the phase-space measure
(ICRF carries the bounce-orbit weight ``lambda(theta_0)``; LHCD has no orbit
geometry and so uses plain ``sin(theta)``) and in that ICRF's measure comes
from a coefficient table while LHCD's is a closed form.  The smoothing/growth
math itself is identical and kept local to each file rather than shared,
since it is short enough to read in place.

Used by: ``ICRF_2D/plot/show_all.py``.

Depends on: :mod:`plot_common.reader` (the cache, the noise floor),
:mod:`plot_common.static` (drawing), :mod:`plot_common.movie` (parallel frame
rendering), ``coefficients.py`` (the bounce-orbit weight and species
temperature).
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

import matplotlib.pyplot as plt
import numpy as np

from coefficients import bounce_pitch_weight, table_parameters
from plot_common.movie import render_movie
from plot_common.reader import load_snapshots, numerical_display_floor
from plot_common.static import contour2d, line1d, movie_scale_range, save_png

# Width, in simulation time, of the centered moving average applied to
# growth-rate movie frames.  Frame-to-frame growth rates divide two nearly
# equal numbers and are extremely noisy; averaging over a fixed time window
# makes the physical trend visible.
GROWTH_AVERAGE_WIDTH = 1.0


# ---------------------------------------------------------------------------
# Growth rate
# ---------------------------------------------------------------------------


def _short_time_average(frames, centers, width):
    """Centered moving average of a frame series, over a fixed *time* window.

    Averaging over a fixed time width (rather than a fixed frame count) keeps
    the physical smoothing scale constant even when snapshot spacing varies
    through a run.  Returns ``(averaged_frames, (first_time, last_time))``
    pairs, one per input frame, so a caller can label each smoothed frame with
    the interval it actually represents.

    The frame list is stacked once, and the finite mask computed once, before
    the loop: reprocessing every frame per window it appears in -- the naive
    approach -- costs a large multiple for a wide window over densely spaced
    snapshots.
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
        total = np.sum(np.where(finite, block, 0.0), axis=0, dtype=np.float64)
        count = np.count_nonzero(finite, axis=0)
        average = np.full(block.shape[1:], np.nan, dtype=np.float32)
        valid = count > 0
        average[valid] = total[valid] / count[valid]
        averaged_frames.append(average)
        bounds.append((float(centers[first]), float(centers[last - 1])))
    return averaged_frames, bounds


def _time_weighted_pixel_average(frames, weights):
    """Collapse a frame stack to one field, weighted by each frame's duration.

    Used to fold the (already short-time-averaged) growth movie down to a
    single static time-averaged figure.  Pixels that were never finite in any
    frame come back masked rather than zero, since zero would draw as a real
    measured value.
    """
    values = np.asarray(frames, dtype=float)
    valid = np.isfinite(values)
    weights = np.asarray(weights, dtype=float)[:, None, None]
    numerator = np.sum(np.where(valid, values * weights, 0.0), axis=0)
    denominator = np.sum(np.where(valid, weights, 0.0), axis=0)
    average = np.full(values.shape[1:], np.nan, dtype=float)
    resolved = denominator > 0.0
    average[resolved] = numerator[resolved] / denominator[resolved]
    return np.ma.masked_invalid(average)


def _growth_frames(cache, floor):
    """Build one raw fractional-growth-rate frame per consecutive pair.

    ``cache.frame_pairs()`` supplies each interior snapshot's reconstruction
    exactly once, already read by the shared cache -- this is why the growth
    rate is essentially free once the cache exists, versus the double read a
    naive per-pair implementation would cost.
    """
    frames = []
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
        frames.append(growth)
        centers.append(0.5 * (cache.times[index] + cache.times[index + 1]))
    return frames, centers


def _growth(cache, floor):
    raw_frames, centers = _growth_frames(cache, floor)
    smoothed_frames, bounds = _short_time_average(
        raw_frames, centers, GROWTH_AVERAGE_WIDTH
    )
    weights = [max(0.0, end - start) for start, end in bounds]
    average = _time_weighted_pixel_average(smoothed_frames, weights)
    scale = movie_scale_range(smoothed_frames, subsample=4096)
    return {
        "frames": smoothed_frames,
        "bounds": bounds,
        "average": average,
        "scale": scale,
        "time_start": bounds[0][0] if bounds else None,
        "time_end": bounds[-1][1] if bounds else None,
    }


# ---------------------------------------------------------------------------
# Particle number, energy, and power
# ---------------------------------------------------------------------------
#
# Both moments use the solver's own phase-space measure
# x^2 * lambda(theta) * sin(theta), so drift in "number" is physical loss
# through the boundary, not a mismatch of conventions with what the solver
# itself conserves.


def _standard_grid_transform(x0, theta0):
    """Determine the reorientation needed to get ascending [theta, x] axes.

    ASGarD's axis order and coordinate direction are not guaranteed, and
    ``np.trapezoid`` requires ascending coordinates.  Computed once from the
    cache's shared coordinate mesh (identical for every frame), then applied
    per frame by :func:`_apply_standard_grid` -- cheaper than recomputing the
    orientation on every snapshot, and exactly equivalent since the
    coordinates do not change frame to frame.
    """
    theta_change_0 = np.nanmax(np.abs(np.diff(theta0, axis=0)))
    theta_change_1 = np.nanmax(np.abs(np.diff(theta0, axis=1)))
    pitch_axis = 0 if theta_change_0 > theta_change_1 else 1
    if pitch_axis == 0:
        x = np.mean(x0, axis=0)
        theta = np.mean(theta0, axis=1)
    else:
        x = np.mean(x0, axis=1)
        theta = np.mean(theta0, axis=0)
    x_order = np.argsort(x)
    theta_order = np.argsort(theta)
    return {
        "pitch_axis": pitch_axis,
        "x": x[x_order],
        "theta": theta[theta_order],
        "x_order": x_order,
        "theta_order": theta_order,
    }


def _apply_standard_grid(f, transform):
    """Reorient one frame into ascending ``[theta, x]`` order."""
    values = f if transform["pitch_axis"] == 0 else f.T
    return values[np.ix_(transform["theta_order"], transform["x_order"])]


def _moments(cache, solver_input, table_dir):
    """Compute ``(number, energy)`` per frame from the reconstruction cache.

    Both are nested-trapezoid integrals over the solver's phase-space measure.
    Energy carries an extra ``x^2`` and a factor of ``T_a``: since
    ``x = v/v_ta`` and ``v_ta^2 = 2 T_a/m_a``, a single particle's kinetic
    energy is exactly ``T_a x^2``, so the result is in keV.
    """
    transform = _standard_grid_transform(cache.x, cache.y)
    x, theta = transform["x"], transform["theta"]
    pitch_weight = bounce_pitch_weight(theta, solver_input, table_dir)
    T_a = table_parameters(table_dir)["T_a"]

    numbers = []
    energies = []
    for frame in cache.frames:
        f = _apply_standard_grid(frame, transform)
        number = np.trapezoid(
            np.trapezoid(f * x[np.newaxis, :]**2, x, axis=1) * pitch_weight, theta
        )
        energy = T_a * np.trapezoid(
            np.trapezoid(f * x[np.newaxis, :]**4, x, axis=1) * pitch_weight, theta
        )
        numbers.append(float(number))
        energies.append(float(energy))
    return np.array(numbers), np.array(energies)


def _smooth_history_and_derivative(times, values, bandwidth_points=4.0, degree=3):
    """Smooth a scalar time history and return its derivative too.

    Used for energy -> power: differentiating raw snapshot data amplifies
    reconstruction noise badly, so a local polynomial is fit at each point and
    differentiated instead.  For each time point, fit a low-order polynomial
    to all the data, weighted by a Gaussian centered on that point; the
    constant term is the smoothed value there, the linear term (after
    unscaling) is the derivative there.

    Bandwidth is ``bandwidth_points`` median snapshot spacings, so it adapts
    to output cadence.  Gaussian weights (rather than a hard window) avoid the
    small kinks a hard window produces whenever it gains or loses a snapshot.
    """
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    count = len(times)
    if count == 0:
        return values.copy(), values.copy()
    if count == 1:
        return values.copy(), np.zeros_like(values)

    positive_steps = np.diff(times)
    positive_steps = positive_steps[positive_steps > 0.0]
    if positive_steps.size == 0:
        return values.copy(), np.zeros_like(values)
    bandwidth = bandwidth_points * float(np.median(positive_steps))
    smoothed = np.empty_like(values)
    derivative = np.empty_like(values)

    for i in range(count):
        scaled_time = (times - times[i]) / bandwidth
        fit_weight = np.exp(-0.25 * scaled_time**2)
        use = fit_weight > 1.0e-6
        local_degree = min(degree, int(np.count_nonzero(use)) - 1)
        coefficients = np.polynomial.polynomial.polyfit(
            scaled_time[use], values[use], local_degree, w=fit_weight[use]
        )
        smoothed[i] = coefficients[0]
        derivative[i] = coefficients[1] / bandwidth

    return smoothed, derivative


def _conservation(cache, solver_input, table_dir):
    numbers, energies = _moments(cache, solver_input, table_dir)
    number0 = numbers[0]
    if abs(number0) <= np.finfo(float).tiny:
        raise ValueError("initial weighted particle number is zero")
    energy_raw = energies / number0
    energy, power = _smooth_history_and_derivative(cache.times, energy_raw)
    return {
        "times": cache.times,
        "delta_number": numbers / number0 - 1.0,
        "energy_raw": energy_raw,
        "energy": energy,
        "power": power,
    }


# ---------------------------------------------------------------------------
# Derive
# ---------------------------------------------------------------------------


def derive(cache, solver_input=None, table_dir=None):
    floor = numerical_display_floor(solver_input or PATHS.solver_input)
    return {
        "growth": _growth(cache, floor),
        "conservation": _conservation(cache, solver_input, table_dir),
        "vpar_vperp": None,  # filled by callers that already have the mesh
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def draw_growth_frame(fig, ax, vpar, vperp, style_axes, growth, index):
    """Draw one short-time-averaged growth-rate frame."""
    frame = growth["frames"][index]
    time_previous, time_current = growth["bounds"][index]
    contour2d(
        fig, ax, vpar, vperp, frame, fixed_range=growth["scale"], filled=True,
        style_axes=style_axes, extend="both",
        title=(
            r"$1\,\tau_c$-averaged growth rate, "
            r"$\gamma_f=(\Delta\mathcal{F}_0/\Delta t)/\mathcal{F}_0$"
            "\n"
            rf"$t={time_previous:.3f}\rightarrow{time_current:.3f}\,\tau_c$"
        ),
    )
    fig.subplots_adjust(left=0.13, right=0.88, bottom=0.13, top=0.84)


def plot_growth_static(vpar, vperp, style_axes, growth):
    """Overall time-averaged growth-rate contour (data-derived scale)."""
    fig, ax = plt.subplots(figsize=(5.4, 4.5))
    contour2d(
        fig, ax, vpar, vperp, growth["average"], filled=True,
        style_axes=style_axes, extend="both",
        title=(
            r"Time-averaged growth rate, "
            r"$\langle\gamma_f\rangle_t=\langle(\Delta\mathcal{F}_0/\Delta t)"
            r"/\mathcal{F}_0\rangle_t$"
            "\n"
            rf"$t={growth['time_start']:.3f}\rightarrow"
            rf"{growth['time_end']:.3f}\,\tau_c$"
        ),
    )
    fig.subplots_adjust(left=0.13, right=0.88, bottom=0.13, top=0.84)
    return fig


def plot_particle_loss(conservation):
    """Fractional weighted-particle loss versus time (linear axes)."""
    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    line1d(
        ax, conservation["times"],
        [(100.0 * conservation["delta_number"], None,
          {"color": "#118ab2", "lw": 2.1})],
        scale="linear", legend=False,
    )
    ax.axhline(0.0, color="#666666", lw=0.8)
    ax.set_title(r"Particle conservation")
    ax.set_xlabel(r"simulation time [$\tau_c$]")
    ax.set_ylabel(r"$\Delta N / N_0$  [%]")
    ax.grid(alpha=0.25)
    return fig


def plot_energy_power(conservation):
    """Two-panel figure: stored energy, and net power ``dE/dt`` (linear axes)."""
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 6.8), constrained_layout=True,
                             sharex=True)
    line1d(
        axes[0], conservation["times"],
        [
            (conservation["energy_raw"], "snapshots",
             {"color": "#2d1e8f", "lw": 0.0, "marker": "o", "ms": 2.5,
              "alpha": 0.24}),
            (conservation["energy"], "local cubic smoothing",
             {"color": "#2d1e8f", "lw": 2.1}),
        ],
        scale="linear", legend=False,
    )
    axes[0].set_title(r"Bounce-weighted fast-ion energy")
    axes[0].set_ylabel(r"$E_\lambda/N_\lambda(0)$ [keV]")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    line1d(
        axes[1], conservation["times"],
        [(conservation["power"], None, {"color": "#2d1e8f", "lw": 2.1})],
        scale="linear", legend=False,
    )
    axes[1].axhline(0.0, color="#666666", lw=0.8)
    axes[1].set_title(r"Net fast-ion power, $P_\lambda=dE_\lambda/dt$")
    axes[1].set_xlabel(r"simulation time [$\tau_c$]")
    axes[1].set_ylabel(r"$P_\lambda/N_\lambda(0)$ [keV/$\tau_c$]")
    axes[1].grid(alpha=0.25)
    return fig


# Per-worker state for the growth movie.
_VPAR = _VPERP = _STYLE_AXES = _GROWTH = None


def _init_growth_worker(vpar, vperp, style_axes, growth):
    global _VPAR, _VPERP, _STYLE_AXES, _GROWTH
    _VPAR, _VPERP, _STYLE_AXES, _GROWTH = vpar, vperp, style_axes, growth


def _draw_growth_frame_task(task):
    index = task["index"]
    fig = plt.figure(figsize=(5.4, 4.5))
    ax = fig.add_subplot(1, 1, 1)
    draw_growth_frame(fig, ax, _VPAR, _VPERP, _STYLE_AXES, _GROWTH, index)
    fig.savefig(f"{task['frame_dir']}/frame_{index:06d}.png", dpi=task["dpi"])
    plt.close(fig)


def plot_growth_movie(vpar, vperp, style_axes, growth, output_file, *,
                      workers=None, fps=8, dpi=140):
    return render_movie(
        _draw_growth_frame_task, len(growth["frames"]), output_file,
        fps=fps, dpi=dpi, workers=workers,
        initializer=_init_growth_worker, initargs=(vpar, vperp, style_axes, growth),
    )


def main():
    from coefficients import style_cartesian_axes
    from plot_common.static import cartesian_mesh

    parser = argparse.ArgumentParser(description="ICRF diagnostics plots")
    parser.add_argument("--static", action="store_true")
    parser.add_argument("--movie", action="store_true")
    parser.add_argument("-o", "--output", default=str(PATHS.snapshots))
    parser.add_argument("--fig-dir", default=str(PATHS.figures))
    parser.add_argument("-n", "--points", type=int, default=192)
    parser.add_argument("-j", "--workers", type=int, default=0)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--dpi", type=int, default=140)
    args = parser.parse_args()
    do_static = args.static or not (args.static or args.movie)
    do_movie = args.movie or not (args.static or args.movie)

    cache = load_snapshots(args.output, args.points, workers=args.workers)
    data = derive(cache)
    vpar, vperp = cartesian_mesh(cache.x, cache.y)

    if do_static:
        save_png(plot_growth_static(vpar, vperp, style_cartesian_axes, data["growth"]),
                 args.fig_dir, "growth_rate", dpi=220)
        save_png(plot_particle_loss(data["conservation"]), args.fig_dir,
                 "particle_loss", dpi=220)
        save_png(plot_energy_power(data["conservation"]), args.fig_dir,
                 "energy_power", dpi=220)
    if do_movie:
        out = str(Path(args.fig_dir) / "growth_rate.mp4")
        plot_growth_movie(vpar, vperp, style_cartesian_axes, data["growth"], out,
                          workers=args.workers, fps=args.fps, dpi=args.dpi)


if __name__ == "__main__":
    main()
