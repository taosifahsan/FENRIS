"""ICRF_1D local effective temperature: T(v) = -f dE/df, static + movie.

Split out of ``solution.py`` so tools/run.sh can render the temperature and
distribution movies as two parallel processes instead of one serial pair --
they were the two slowest movies in the batch and shared a critical path.

All the physics lives in :func:`solution.derive` (shared with the
distribution plots): the analytic steady state, the flat ``T_eff``
prediction, and the RF tail's ``~ v^3`` asymptote come from there; this file
only draws them.

Used by: ``tools/run.sh`` (one of the parallel plotter processes).

Depends on: :mod:`solution` (derive + local_temperature),
:mod:`plot_common.runtime`, :mod:`plot_common.reader`,
:mod:`plot_common.static`, :mod:`plot_common.movie`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Locate the directory that holds plot_common by walking up from this file,
# so reorganizing the project tree cannot silently break the import.
for _parent in Path(__file__).resolve().parents:
    if (_parent / "plot_common").is_dir():
        sys.path.insert(0, str(_parent))
        break
else:
    raise ImportError("plot_common not found above " + __file__)

from plot_common.runtime import bootstrap

PATHS = bootstrap(__file__)

import numpy as np

from plot_common.movie import render_movie
from plot_common.reader import (
    load_cache,
    load_snapshots_1d,
    option_float,
    read_options,
)
from plot_common.static import render_still, save_png
from solution import M_E, derive, local_temperature


# One size for the still and every movie frame -- stated once.
FIGSIZE = (7.2, 5.2)


def draw_frame(fig, ax, data, index):
    """One temperature frame: local T(v) of solved and steady f, with the
    flat T_eff prediction and the RF tail's ~ v^3 asymptote."""
    v, plasma = data["v"], data["plasma"]
    f = np.asarray(data["frames"][index], dtype=float)

    time_label = rf"$(v/v_{{th}},\ t/\tau = {data['times'][index]:.2f})$"
    # Strict on purpose: a deck typo crashes with the bad name rather
    # than silently falling back to a linear plot.
    draw = getattr(ax, data["scale_temp"])
    draw(v, data["temp_steady"], label=r"$T$ $(v/v_{th})$ steady")
    temp_frame = local_temperature(f, v, plasma.m)
    draw(v, temp_frame, "--", label=r"$T$ " + time_label)
    draw(v, np.full_like(v, plasma.effective_temperature()), label=r"$T_{eff}$")
    v_large = np.linspace(1 / M_E**0.5, v.max(), 100)
    draw(v_large, plasma.large_v_temperature(v_large), "k",
         label=r"$T_{large}\propto v^3$")
    ax.set_xlabel(r"$v/v_{th}$", fontsize=14)
    ax.set_ylabel(r"$T/T_e$", fontsize=14)
    ax.set_xlim([1e-2, v.max()])
    # Below half the T_eff line is finite-difference noise; the top tracks
    # the larger of the two temperature curves for this frame.
    finite = temp_frame[np.isfinite(temp_frame)]
    top = 1.2 * max(float(np.nanmax(data["temp_steady"])),
                    float(finite.max()) if finite.size else 0.0)
    ax.set_ylim(bottom=0.5 * plasma.effective_temperature(), top=top)
    ax.set_title("local effective temperature", fontsize=13)
    ax.legend(loc="best")
    ax.grid(True, which="major", linestyle=":")
    fig.tight_layout()


def main():
    """CLI entry point: parse flags, load the data, render the figures.

    Giving neither ``--static`` nor ``--movie`` renders both -- that is how
    tools/run.sh invokes every plotter; either flag narrows a manual run to
    just that output.
    """
    parser = argparse.ArgumentParser(description="ICRF_1D temperature plots")
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

    # Bind the derived data into the (fig, ax, index) signature render_still
    # and render_movie expect (closures are fine: rendering is in-process).
    def draw(fig, ax, index):
        draw_frame(fig, ax, data, index)

    if do_static:
        save_png(render_still(draw, len(data["frames"]) - 1, figsize=FIGSIZE),
                 args.fig_dir, "temperature", dpi=220)
    if do_movie:
        render_movie(draw, len(data["frames"]),
                     str(Path(args.fig_dir) / "temperature.mp4"),
                     figsize=FIGSIZE, fps=args.fps, dpi=args.dpi)


if __name__ == "__main__":
    main()
