"""The perpendicular density ``n(v_perp) = 2 pi v_perp * integral f dv_par``.

The companion marginal to ``density_vpar.py``: the same cylindrical volume
element ``2 pi v_perp dv_par dv_perp``, integrated along the OTHER axis.  So
``integral n dv_perp`` is again the full 3-D particle count -- 1 at t=0 by
the initial-condition normalization, and equal frame by frame to
``density_vpar.py``'s count, which is the cross-check to reach for if
either reduction is ever in doubt.  Fixed-``v_perp`` lines cut diagonally
across the solver's spherical ``(x, theta)`` grid, so each frame is
bilinearly interpolated along them rather than reduced by a per-axis weight.

**The analytic background is the closure ansatz itself.**  The 2D->1D
reduction of ``Readings/2D-1D-Fokker-Planck-Derivation.pdf`` closes the
perpendicular moments by ASSUMING a Maxwellian perpendicular profile at
every parallel speed,

    f(v_par, v_perp) = pi^{-1} e^{-v_perp^2} F(v_par)

(in the standard units ``x = v / sqrt(2T/m)`` -- see src/LHCD_2D.cpp's
normalization header; the derivation document uses the older
``sqrt(T/m)`` convention, where the same ansatz reads
``(2 pi)^{-1} e^{-v_perp^2/2}``).  Substituting it into the definition
above collapses the ``v_par`` integral into ``integral F dv_par = N``
and leaves

    n(v_perp) = N * 2 v_perp e^{-v_perp^2}

The shape integrates to 1 on its own -- but only over the HALF-LINE.
Unlike ``v_par``, ``v_perp`` is a radius: its range is capped from below at
0, never negative.  That is what makes the normalization work, and it is a
live trap rather than a pedantic note, because ``v_perp e^{-v_perp^2}``
is odd, so integrating it over the full line returns 0 (and a
symmetric-grid reduction would silently normalize against that).  Every
``v_perp`` grid here therefore starts at exactly 0.

Note what is NOT in it: the RF
window, ``D_w``, ``Zi``, any of it.  The quasilinear diffusion
``D_ql = D_w e_par e_par^T`` pushes strictly along ``v_par``, so under the
ansatz LHCD cannot touch this marginal at all -- the prediction is that
``n(v_perp)`` stays exactly its initial Maxwellian for all time.

That makes this the DIRECT measurement of the closure error, where
``density_vpar.py`` only shows it indirectly (as an amplitude gap against
the reduced steady state).  Any daylight between the solid and dashed
curves here is precisely the physics the reduction throws away: the
resonance is in ``v_par`` only, so the RF raises the TOTAL speed
``x = hypot(v_par, v_perp)`` at fixed ``v_perp``, and pitch-angle
scattering then rotates that gain into ``v_perp`` at fixed ``x``.
Repeating the loop ratchets particles outward -- each window transit adds
``5^2 - 3^2 = 16`` to ``x^2`` regardless of how fast the particle already
is.

Read the two plots together, and mind the SIGN of the feedback: every
collision coefficient falls like ``1/x^3``, so a fatter-than-Maxwellian
``v_perp`` means larger ``x`` at fixed ``v_par`` and therefore WEAKER
parallel collisionality, not stronger.  The reduction assumes a Maxwellian
``v_perp``, so it over-estimates collisionality and UNDER-predicts the
plateau: the measured ``n(v_par)`` does not converge up to the reduced
steady state from below, it crosses it (fill fraction passed 1.0 near
t = 39 and reached 1.11 by t = 50) and keeps climbing.  Neither marginal
has a steady state in this deck -- see the growth check below.

**The mechanism is confirmed causally, not just inferred.**  Setting
``Zi = -1.0`` in the deck makes the pitch coefficient ``(Zi+1)/(4x^3)``
identically zero -- the Lorentz operator switches off while everything else
is untouched -- and the tail vanishes completely: ``n(v_perp)/Maxwellian``
goes 2.22 -> 1.000 at ``v_perp=4`` and 22.8 -> 0.99 at ``v_perp=5``, flat to
within 1% wherever the reconstruction can resolve it.  Perpendicular
heating over 50 tau_c drops 17-fold (``<v_perp^2>`` gains 0.0033 instead of
0.056) and saturates instead of rising linearly.  With no rotation at fixed
speed there is no ratchet, so the RF can only push along ``v_par`` and this
marginal stays put -- exactly as the closure ansatz says it should.

**No saturation** (with pitch scattering ON).
``n(v_perp=6)/Maxwellian`` runs 15 / 86 / 254 / 517 at
t = 20 / 30 / 40 / 50, and ``<v_perp^2>`` climbs steadily with no rollover.
The reason is structural: the per-transit energy gain is speed-independent
(``16``), while the collisional loss over one transit falls with speed, so
the ratchet has no collisional cutoff below the ``x = 10`` wall.  That is a
property of THIS quasilinear operator -- a window in ``v_par`` with no
``v_perp`` dependence and no upper energy limit -- not a physical
prediction; real LH spectra truncate it via finite ``k_perp``, relativistic
detuning, and radial losses.  Trust the curve only where it is above the
reconstruction floor: by ``v_perp >~ 7`` the frames ring negative.

(Units note: the measured numbers in the two paragraphs above were taken
before the 2026-08 standardization to ``v_th = sqrt(2T/m)`` and quote the
old ``sqrt(T/m)`` coordinates -- divide velocities by sqrt(2), times by
2 sqrt(2).  The physical statements are convention-independent: the
per-transit energy gain is 8 T in either language.)

Used by: ``tools/run.sh`` (one of the parallel plotter processes).

Depends on: :mod:`plot_common.reader` (the cache and the deck),
:mod:`plot_common.static` (drawing), :mod:`plot_common.movie` (movie
rendering), scipy (bilinear frame interpolation).
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
from scipy.interpolate import RegularGridInterpolator

from plot_common.movie import render_movie
from plot_common.reader import (
    load_cache,
    load_snapshots,
    numerical_display_floor,
)
from plot_common.static import line1d, render_still, save_png


def _oriented_axes(x0, theta0):
    """Sorted 1-D ``(x, theta)`` axes plus the reorientation for each frame.

    Empirical axis detection (same check the other 2-D plotters use):
    whichever axis ``theta0`` varies along is the pitch axis; guessing wrong
    would interpolate the transposed field while still looking plausible.
    """
    change0 = np.nanmax(np.abs(np.diff(theta0, axis=0))) if theta0.shape[0] > 1 else 0.0
    change1 = np.nanmax(np.abs(np.diff(theta0, axis=1))) if theta0.shape[1] > 1 else 0.0
    pitch_axis = 0 if change0 > change1 else 1
    x = np.take(x0, 0, axis=pitch_axis)
    theta = np.take(theta0, 0, axis=1 - pitch_axis)
    x_order, theta_order = np.argsort(x), np.argsort(theta)

    def orient(frame):
        values = frame if pitch_axis == 1 else frame.T
        return values[np.ix_(x_order, theta_order)]

    return x[x_order], theta[theta_order], orient


def analytic_maxwellian(vperp):
    """The closure ansatz's prediction (un-normalized shape).

    See the module header: ``2 v_perp e^{-v_perp^2}`` in the standard
    sqrt(2T/m) units, with no dependence on the RF window or the collision
    parameters.  Kept as its own function so the shape is stated exactly
    once, and so its independence from the deck is visible in the signature.
    """
    vperp = np.asarray(vperp, dtype=float)
    return 2.0 * vperp * np.exp(-vperp * vperp)


def derive_density(cache, solver_input=None):
    """Reduce every frame to n(v_perp) and build the normalized overlay.

    The fixed-``v_perp`` lines are diagonals of the ``(x, theta)`` grid, so
    the reduction interpolates each frame bilinearly at
    ``x = hypot(v_par, v_perp)``, ``theta = atan2(v_perp, v_par)`` and
    integrates ``2 pi f v_perp`` over ``v_par`` by trapezoid.  The query
    points depend only on the fixed grid, so they are built once and reused
    for every frame.
    """
    x, theta, orient = _oriented_axes(cache.x, cache.y)
    x_max = float(x[-1])
    floor = numerical_display_floor(solver_input or PATHS.solver_input)

    vperp = np.linspace(0.0, x_max, 401)
    # One v_par grid per v_perp, spanning the disk |v| <= x_max symmetrically
    # about v_par = 0 (the RF window sits on the positive side only, so the
    # two halves are not interchangeable and both must be integrated).
    span = np.sqrt(np.maximum(x_max**2 - vperp**2, 0.0))
    vpar = np.linspace(-1.0, 1.0, 401)[np.newaxis, :] * span[:, np.newaxis]
    xq = np.clip(np.hypot(vpar, vperp[:, np.newaxis]), x[0], x[-1])
    thq = np.clip(np.arctan2(vperp[:, np.newaxis], vpar), theta[0], theta[-1])
    query = np.stack([xq.ravel(), thq.ravel()], axis=-1)

    def reduce(frame):
        values = RegularGridInterpolator((x, theta), orient(np.asarray(frame, float)),
                                         bounds_error=False, fill_value=0.0)(query)
        integrand = values.reshape(vpar.shape)
        return 2.0 * np.pi * vperp * np.trapezoid(integrand, x=vpar, axis=1)

    frames = [reduce(frame) for frame in cache.frames]
    numbers = [float(np.trapezoid(frame, vperp)) for frame in frames]

    # The analytic background, normalized to the INITIAL total so its area
    # is the particle count the run started with -- same convention as
    # density_vpar.py, so the two overlays carry the same number.
    shape = analytic_maxwellian(vperp)
    analytic = shape * (numbers[0] / float(np.trapezoid(shape, vperp)))

    # Signed-log y-limits spanning the analytic curve and every frame, same
    # convention as the other density plots.
    everything = np.concatenate([np.abs(analytic)] + [np.abs(fr) for fr in frames])
    finite = everything[np.isfinite(everything) & (everything > 0.0)]
    return {
        "vperp": vperp,
        "analytic": analytic,
        "frames": frames,
        "numbers": numbers,
        "times": cache.times,
        "ylim": (floor, float(finite.max()) * 1.1),
    }


# One size for the still and every movie frame -- stated once.
FIGSIZE = (5.9, 3.8)


def draw_frame(fig, ax, data, index):
    """One frame: the closure ansatz (dashed) and the measured n(v_perp)."""
    line1d(
        ax, data["vperp"],
        [
            (data["analytic"], "closure ansatz (Maxwellian)", {"color": "#999999",
                                                               "linestyle": "--", "lw": 1.8}),
            (data["frames"][index], "current", {"color": "#118ab2",
                                                "linestyle": "-", "lw": 2.0}),
        ],
        scale="log", ylim=data["ylim"], legend=False,
    )
    ax.set_xlabel(r"$v_\perp/v_{te}$", fontsize=13)
    ax.set_ylabel(r"$n(v_\perp)$", fontsize=13)
    ax.set_title(
        r"$n(v_\perp)=2\pi v_\perp\int f\,dv_\parallel$"
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
    parser = argparse.ArgumentParser(description="LHCD perpendicular-velocity density plots")
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
                 args.fig_dir, "density_vperp", dpi=220)
    if do_movie:
        render_movie(draw, len(data["times"]),
                     str(Path(args.fig_dir) / "density_vperp.mp4"),
                     figsize=FIGSIZE, fps=args.fps, dpi=args.dpi)


if __name__ == "__main__":
    main()
