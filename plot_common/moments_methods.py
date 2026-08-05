"""Particle conservation, energy, and power -- 1-D and 2-D.

The third of the shared method modules, alongside
:mod:`plot_common.grid_methods` and :mod:`plot_common.growth_methods`.  Each
project's ``plot/moments.py`` is a thin wrapper that supplies the one thing
that genuinely differs -- the phase-space measure (the Jacobian) -- plus its
figure labels.

Two static figures:

  ``particle_loss`` -- relative change of the number moment over the run.
                       Because this is the measure the solver itself
                       conserves, drift here is physical loss through the
                       boundary rather than a mismatch of conventions.
  ``energy_power``  -- the energy moment's history and its time derivative,
                       the absorbed power.

**The Jacobian is injected, not branched on.**  In 2-D the theta part of
the phase-space Jacobian arrives as ``jacobian_th``: a *quadrature vector*
(integration weights with the measure folded in) rather than a pointwise
weight, so LHCD's closed-form ``sin(theta)`` and ICRF's tabulated bounce
weight ``lambda(theta_0)`` are the same code path -- see
:func:`trapezoid_weights` for turning a pointwise measure into one.  (A
quadrature vector, deliberately: integrating ICRF's pointwise ``lambda`` by
trapezoid through its trapped-passing peak is resolution-dependent; the
exact hat-function quadrature is not.)  The speed part ``x^2`` is identical
in all four projects and lives here.  In 1-D the number and energy weights
arrive as plain per-point arrays.

**Energy is dimensionless, in units of the background temperature T.**
With the standardized velocity ``x = v / sqrt(2T/m)`` a single particle's
kinetic energy is exactly ``T x^2``, so ``E/(N T) = <x^2>`` -- and a
Maxwellian gives 3/2, the equipartition value, in every project.  That
shared 3/2 at t = 0 is a free cross-project correctness check, and it is
why there is no per-project energy prefactor anywhere in this module.

**On the missing radial flux.**  This used to also compute the boundary
radial flux ``J_x`` at the outer velocity wall, by least-squares
extrapolating ``f`` and its derivatives into the last grid cell and combining
them with the coefficients evaluated there.  That was the most intricate and
most fragile part of the diagnostics and never produced a physically
reasonable result, so it was removed rather than left in place emitting
numbers nobody trusts.  Particle loss answers the same question -- "are
particles escaping, and how fast?" -- directly from a conserved moment, with
no extrapolation.

Used by: each project's ``plot/moments.py``, in turn by ``tools/run.sh``.

Depends on: :mod:`plot_common.reader` (the cache, the deck),
:mod:`plot_common.static` (``line1d``, ``save_png``).
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from plot_common.reader import (
    load_cache,
    load_snapshots,
    load_snapshots_1d,
    option_float,
    read_options,
)
from plot_common.static import line1d, save_png


def trapezoid_weights(coordinate):
    """Quadrature vector ``w`` with ``g @ w == np.trapezoid(g, coordinate)``.

    Lets a caller fold a pointwise measure into the quadrature (multiply
    ``w`` by it) instead of into the integrand, which is what makes one
    implementation serve both a closed-form measure and a tabulated one.
    """
    coordinate = np.asarray(coordinate, dtype=float)
    weights = np.zeros_like(coordinate)
    if coordinate.size < 2:
        return weights
    weights[0] = 0.5 * (coordinate[1] - coordinate[0])
    weights[-1] = 0.5 * (coordinate[-1] - coordinate[-2])
    weights[1:-1] = 0.5 * (coordinate[2:] - coordinate[:-2])
    return weights


def _smooth_history_and_derivative(times, values, bandwidth_points=4.0, degree=3):
    """Smooth a scalar time history and return its derivative too.

    Used for energy -> power: differentiating raw snapshot data amplifies
    reconstruction noise badly, so a local polynomial is fit at each point and
    differentiated instead.  For each time point, fit a low-order polynomial
    to all the data weighted by a Gaussian centered there; the constant term
    is the smoothed value, the linear term (unscaled) the derivative.

    Bandwidth is ``bandwidth_points`` *median* snapshot spacings -- median
    rather than mean so a few long gaps do not inflate it.  Gaussian weights
    rather than a hard window avoid the small kinks a hard window produces
    whenever it gains or loses a snapshot.
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
        # Drop negligibly weighted points: they cannot affect the fit but cost
        # time and can worsen the conditioning of the least-squares solve.
        use = fit_weight > 1.0e-6
        local_degree = min(degree, int(np.count_nonzero(use)) - 1)
        coefficients = np.polynomial.polynomial.polyfit(
            scaled_time[use], values[use], local_degree, w=fit_weight[use]
        )
        smoothed[i] = coefficients[0]
        derivative[i] = coefficients[1] / bandwidth

    return smoothed, derivative


# ---------------------------------------------------------------------------
# One dimension
# ---------------------------------------------------------------------------


def derive_1d(cache, solver_input, number_weight, energy_weight):
    """Number and energy histories, plus power as the energy's derivative.

    ``energy_weight`` should be built so the moment comes out in units of
    the background temperature T (see the module header): the weight is the
    number weight times ``x^2`` in standard units, with no 1/2.
    """
    options = read_options(solver_input)
    v = np.asarray(cache.x, dtype=float)
    dv = v[1] - v[0]
    n_weight = number_weight(v, options)
    e_weight = energy_weight(v, options)

    number, energy = [], []
    for frame in cache.frames:
        f = np.asarray(frame, dtype=float)
        number.append(float(np.sum(f * n_weight) * dv))
        energy.append(float(np.sum(f * e_weight) * dv))
    number = np.array(number)
    energy = np.array(energy)
    frame_times = np.asarray(cache.times, dtype=float)
    return {"frame_times": frame_times, "number": number, "energy": energy,
            "power": np.gradient(energy, frame_times)}


def plot_particles_1d(data):
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


def plot_energy_power_1d(data):
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


def _parse(description, paths, default_points):
    """The flag surface every plotter shares.

    Both figures here are static; ``--static``/``--movie`` are accepted for a
    uniform surface across plotters, and ``--movie`` alone renders nothing.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--static", action="store_true")
    parser.add_argument("--movie", action="store_true")
    parser.add_argument("-o", "--output", default=str(paths.snapshots))
    parser.add_argument("--cache", default=None,
                        help="load the shared cache.npz instead of reading snapshots")
    parser.add_argument("--fig-dir", default=str(paths.figures))
    parser.add_argument("-n", "--points", type=int, default=default_points)
    parser.add_argument("--fps", type=int, default=8, help=argparse.SUPPRESS)
    parser.add_argument("--dpi", type=int, default=140, help=argparse.SUPPRESS)
    return parser.parse_args()


def main_1d(paths, *, description, number_weight, energy_weight):
    """CLI entry point for a 1-D project's ``plot/moments.py``.

    ``number_weight(v, options)`` and ``energy_weight(v, options)`` return the
    per-point weights of the two moments under that project's measure -- the
    1-D Jacobian, injected.
    """
    args = _parse(description, paths, default_points=0)
    if args.movie and not args.static:
        return

    if args.cache:
        cache = load_cache(args.cache)
    else:
        points = args.points
        if points <= 0:
            options = read_options(paths.solver_input)
            points = int(option_float(options, "num_points", 256) / 2)
        cache = load_snapshots_1d(args.output, points)
    data = derive_1d(cache, paths.solver_input, number_weight, energy_weight)

    save_png(plot_particles_1d(data), args.fig_dir, "particle_loss", dpi=220)
    save_png(plot_energy_power_1d(data), args.fig_dir, "energy_power", dpi=220)


# ---------------------------------------------------------------------------
# Two dimensions
# ---------------------------------------------------------------------------


def _standard_grid_transform(x0, theta0):
    """Determine the reorientation giving ascending ``[theta, x]`` axes.

    ASGarD's axis order and direction are not guaranteed, and
    ``np.trapezoid`` needs ascending coordinates.  Computed once from the
    cache's shared coordinate mesh (identical for every frame), then applied
    per frame -- exactly equivalent to recomputing it per snapshot, since the
    coordinates never change, and considerably cheaper.
    """
    change0 = np.nanmax(np.abs(np.diff(theta0, axis=0))) if theta0.shape[0] > 1 else 0.0
    change1 = np.nanmax(np.abs(np.diff(theta0, axis=1))) if theta0.shape[1] > 1 else 0.0
    pitch_axis = 0 if change0 > change1 else 1
    velocity_axis = 1 - pitch_axis
    speed = np.take(x0, 0, axis=pitch_axis)
    pitch = np.take(theta0, 0, axis=velocity_axis)
    x_order = np.argsort(speed)
    theta_order = np.argsort(pitch)
    return {
        "pitch_axis": pitch_axis,
        "x": speed[x_order],
        "theta": pitch[theta_order],
        "x_order": x_order,
        "theta_order": theta_order,
    }


def _apply_standard_grid(f, transform):
    """Reorient one frame into ascending ``[theta, x]`` order."""
    values = f.T if transform["pitch_axis"] == 1 else f
    return values[np.ix_(transform["theta_order"], transform["x_order"])]


def _moments_2d(cache, jacobian_th):
    """``(number, energy)`` per frame, over the solver's phase-space measure.

    Nested integrals over the speed Jacobian ``x^2`` and the theta
    quadrature; energy carries an extra ``x^2`` and nothing else, so it is
    ``E/(N T)``-ready (see the module header).
    """
    transform = _standard_grid_transform(cache.x, cache.y)
    x, theta = transform["x"], transform["theta"]
    quadrature = jacobian_th(theta)

    numbers = []
    energies = []
    for frame in cache.frames:
        f = _apply_standard_grid(frame, transform)
        number = np.trapezoid(f * x[np.newaxis, :]**2, x, axis=1) @ quadrature
        energy = np.trapezoid(f * x[np.newaxis, :]**4, x, axis=1) @ quadrature
        numbers.append(float(number))
        energies.append(float(energy))
    return np.array(numbers), np.array(energies)


def derive_2d(cache, jacobian_th):
    """Number and energy histories, both normalized to the initial number.

    Dividing by N(0) makes the traces per-initial-particle, so runs with
    different absolute normalizations plot on comparable scales; energy is
    then smoothed and differentiated for the power panel.
    """
    numbers, energies = _moments_2d(cache, jacobian_th)
    number0 = numbers[0]
    if abs(number0) <= np.finfo(float).tiny:
        raise ValueError("initial particle number is zero")
    energy_raw = energies / number0
    energy, power = _smooth_history_and_derivative(cache.times, energy_raw)
    return {
        "times": cache.times,
        "delta_number": numbers / number0 - 1.0,
        "energy_raw": energy_raw,
        "energy": energy,
        "power": power,
    }


def plot_particle_loss_2d(conservation):
    """Fractional particle loss versus time (linear axes)."""
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


def plot_energy_power_2d(conservation, energy_title, energy_ylabel,
                         power_title, power_ylabel):
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
    axes[0].set_title(energy_title)
    axes[0].set_ylabel(energy_ylabel)
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    line1d(
        axes[1], conservation["times"],
        [(conservation["power"], None, {"color": "#2d1e8f", "lw": 2.1})],
        scale="linear", legend=False,
    )
    axes[1].axhline(0.0, color="#666666", lw=0.8)
    axes[1].set_title(power_title)
    axes[1].set_xlabel(r"simulation time [$\tau_c$]")
    axes[1].set_ylabel(power_ylabel)
    axes[1].grid(alpha=0.25)
    return fig


def main_2d(paths, *, description, jacobian_th,
            energy_title, energy_ylabel, power_title, power_ylabel):
    """CLI entry point for a 2-D project's ``plot/moments.py``.

    ``jacobian_th(theta)`` returns the theta part of the phase-space
    Jacobian as a quadrature vector: a pointwise measure folded into
    :func:`trapezoid_weights`, or a tabulated exact quadrature.  Energy is
    reported in units of the background temperature T (module header); there
    is no per-project prefactor.
    """
    args = _parse(description, paths, default_points=192)
    if args.movie and not args.static:
        return

    if args.cache:
        cache = load_cache(args.cache)
    else:
        cache = load_snapshots(args.output, args.points)
    conservation = derive_2d(cache, jacobian_th)

    save_png(plot_particle_loss_2d(conservation), args.fig_dir,
             "particle_loss", dpi=220)
    save_png(plot_energy_power_2d(conservation, energy_title, energy_ylabel,
                                  power_title, power_ylabel),
             args.fig_dir, "energy_power", dpi=220)
