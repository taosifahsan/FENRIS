"""Single-plot drawing: the signed-log 2-D contour, 1-D lines, and the
per-box refinement-level map.  Shared by both projects because these are
presentation techniques, not physics -- callers pass in finished arrays plus
labels, and get back drawn artists and saved files.

Coordinates are always Cartesian ``(v_parallel, v_perp)``; color is always
signed-log.  There is no linear-scale variant and no native ``(theta_0, x_0)``
view -- both were dropped deliberately to cut the figure count and the
axis/colorbar permutations in half.  See :func:`cartesian_mesh` for the one
coordinate transform every 2-D plotter uses.

Four sections:

1. **Colormap and scale** -- turning a data array into the four things
   matplotlib needs: masked values, contour levels, a norm, and a colormap.
   Both a data-derived and a fixed-range flavour exist (movies must hold one
   scale across all frames; stills derive theirs per figure).
2. **Colorbars and ticks** -- readable tick placement on a logarithmic scale,
   fiddly enough to deserve its own section.
3. **Generic drawers** -- :func:`contour2d`, :func:`line1d`, and
   :func:`level_contour`.  Every 2-D field in both projects (solution, growth
   rate, coefficients) goes through ``contour2d``; every line plot (marginals,
   particle loss, energy/power, orbit factors) goes through ``line1d``.  The
   per-box refinement level map is deliberately its own function -- see
   :func:`level_contour`.
4. **Output** -- PNG saving and timestamped run directories.

Used by: every plotter in ``ICRF_2D/plot/`` and ``LHCD_2D/plot/``.

Depends on: ``matplotlib``, ``numpy``.  Deliberately does *not* import
``plot_common.reader`` -- drawing must not depend on where the data came from.

Why "signed log": the distribution function spans many orders of magnitude
*and* changes sign.  A plain log scale cannot show negative values; a linear
scale hides everything except the peak.  So each sign gets its own
logarithmic branch, joined by a white band at zero whose width is set by the
numerical noise floor -- values inside that band are not physically
meaningful (see :func:`plot_common.reader.numerical_display_floor`).
"""

from __future__ import annotations

import datetime as _datetime
import math
import os
import shutil

import matplotlib as mpl
import numpy as np


# ---------------------------------------------------------------------------
# Section 1: colormap and scale
# ---------------------------------------------------------------------------
#
# Positive values run white -> cyan -> blue; negative values run white ->
# orange -> dark red.  White always means "at or below the noise floor", so
# the eye reads distance-from-white as significance in both directions.

# Default dynamic range of a signed-log scale when no explicit noise floor is
# supplied.  ICRF's historical value; LHCD passes 7.0 where it wants a wider
# range.
SIGNED_LOG_DECADES = 6.0


def mask_exact_zero(values):
    """Mask entries that are exactly zero so they render as white.

    Exact zeros are almost always "nothing here" rather than a measured value
    of zero, and on a logarithmic scale they have no representable position
    at all.  Masking is preferable to clipping: a clipped zero would paint as
    the smallest representable magnitude and read as real signal.
    """
    return np.ma.masked_where(values == 0.0, values)


def signed_log_cmap(split):
    """Diverging colormap whose white center sits at normalized ``split``.

    ``split`` is where zero falls on the 0..1 colorbar axis.  For a symmetric
    range that is 0.5, but signed-log ranges are usually lopsided (far more
    positive dynamic range than negative, say), and the white band must sit
    where zero actually is or the colors lie about sign.

    The two degenerate cases (``split`` at either end) are special-cased to
    one-sided ramps, because interpolating a color stop at exactly 0.0 or 1.0
    alongside its neighbour produces a discontinuity.

    ``N=4096`` rather than matplotlib's default 256: with a strongly one-sided
    range, ``split`` can land within a couple of entries of a 256-color table
    and the intended white center quantizes away to red or blue.
    """
    split = float(np.clip(split, 0.0, 1.0))
    eps = 1.0e-6
    if split <= eps:
        # Zero at the bottom: everything is positive, so ramp white -> blue.
        colors = (
            (0.00, "#ffffff"),
            (0.10, "#d9fbff"),
            (0.42, "#5ad7d1"),
            (0.72, "#1787bf"),
            (1.00, "#2d1e8f"),
        )
    elif split >= 1.0 - eps:
        # Zero at the top: everything is negative, so ramp dark red -> white.
        colors = (
            (0.00, "#4b0f2f"),
            (0.28, "#cf1f63"),
            (0.58, "#ff8a5c"),
            (0.90, "#ffe2d1"),
            (1.00, "#ffffff"),
        )
    else:
        # Two-sided: stops below `split` are placed as fractions of the
        # negative span, stops above as fractions of the positive span, so
        # both branches get the same internal color progression regardless of
        # how lopsided the split is.
        colors = (
            (0.00, "#4b0f2f"),
            (0.55 * split, "#cf1f63"),
            (0.88 * split, "#ff8a5c"),
            (split, "#ffffff"),
            (split + 0.12 * (1.0 - split), "#d9fbff"),
            (split + 0.42 * (1.0 - split), "#5ad7d1"),
            (split + 0.72 * (1.0 - split), "#1787bf"),
            (1.00, "#2d1e8f"),
        )
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "signed_log_split", colors, N=4096
    )
    cmap.set_bad("white")
    return cmap


class SignedLogNorm(mpl.colors.Normalize):
    """Logarithmic on each sign, with a white zero band at ``split``.

    Maps data values to the 0..1 range matplotlib colormaps expect:

        vmin (most negative) -> 0.0
        0                    -> split
        vmax (most positive) -> 1.0

    Within each sign the mapping is logarithmic in ``|value|``, so equal
    *ratios* occupy equal colorbar distance -- which is the whole point when
    the data spans many decades.

    ``linthresh`` is the magnitude below which values are treated as zero
    (the numerical noise floor).  Everything inside it collapses into the
    white band rather than being stretched across the many decades between
    the floor and true zero, which do not exist in the data.
    """

    def __init__(self, linthresh, vmin, vmax, split):
        # clip=False so out-of-range values come back masked (and draw as
        # white) rather than being silently pinned to the endpoints.
        super().__init__(vmin=vmin, vmax=vmax, clip=False)
        self.linthresh = linthresh
        self.split = split
        # Precompute the logarithms once; __call__ runs per frame per pixel.
        self._log_eps = math.log10(linthresh)
        self._log_neg = math.log10(-vmin) if vmin < 0.0 else self._log_eps
        self._log_pos = math.log10(vmax) if vmax > 0.0 else self._log_eps
        # Guard against a zero span (a one-sided range), which would divide
        # by zero below.  eps keeps the division finite; the branch is unused.
        self._neg_span = max(self._log_neg - self._log_eps, np.finfo(float).eps)
        self._pos_span = max(self._log_pos - self._log_eps, np.finfo(float).eps)

    def __call__(self, value, clip=None):
        """Map data values into 0..1 colormap coordinates."""
        values = np.ma.asarray(value)
        data = np.asarray(values.filled(np.nan), dtype=float)
        # "Valid" means finite and not already masked by the caller.
        valid = np.isfinite(data) & ~np.ma.getmaskarray(values)
        result = np.full(data.shape, np.nan, dtype=float)

        neg = valid & (data < 0.0)
        if np.any(neg):
            # Clamp at linthresh so values inside the noise band land at the
            # branch's inner edge instead of running off to -inf.
            log_abs = np.log10(np.maximum(np.abs(data[neg]), self.linthresh))
            # Larger |value| -> closer to 0.0 (the most-negative end).
            result[neg] = self.split * (self._log_neg - log_abs) / self._neg_span

        pos = valid & (data > 0.0)
        if np.any(pos):
            log_abs = np.log10(np.maximum(data[pos], self.linthresh))
            # Larger value -> closer to 1.0.
            result[pos] = (
                self.split
                + (1.0 - self.split) * (log_abs - self._log_eps) / self._pos_span
            )

        zero = valid & (data == 0.0)
        if np.any(zero):
            result[zero] = self.split

        return np.ma.array(np.clip(result, 0.0, 1.0), mask=~valid)

    def inverse(self, value):
        """Map 0..1 colormap coordinates back to data values.

        Matplotlib needs this to label colorbar ticks: it asks "what data
        value sits at this fraction of the bar?"
        """
        values = np.asarray(value)
        result = np.zeros_like(values, dtype=float)

        neg = values < self.split
        if np.any(neg):
            frac = values[neg] / max(self.split, np.finfo(float).eps)
            result[neg] = -10.0 ** (self._log_neg - frac * self._neg_span)

        pos = values > self.split
        if np.any(pos):
            frac = (
                (values[pos] - self.split)
                / max(1.0 - self.split, np.finfo(float).eps)
            )
            result[pos] = 10.0 ** (self._log_eps + frac * self._pos_span)

        return result


def _signed_log_from_range(values, vmin, vmax, decades, n, linthresh_floor):
    """Shared core of the signed-log scale, given an already-chosen range.

    Split out so the data-derived (:func:`signed_log_scale`) and fixed-range
    (:func:`fixed_signed_log_scale`) entry points cannot drift apart.

    Returns ``(masked_values, levels, norm, cmap)``.
    """
    masked = mask_exact_zero(values)

    # The noise floor: either supplied explicitly (from the solver tolerance)
    # or taken as `decades` below the largest magnitude present.
    max_abs = max(abs(vmin), abs(vmax), np.finfo(float).tiny)
    if linthresh_floor > 0.0:
        linthresh = max(linthresh_floor, np.finfo(float).tiny)
    else:
        linthresh = max(max_abs * 10.0 ** (-decades), np.finfo(float).tiny)

    # Collapse a branch entirely if its extreme does not clear the floor --
    # otherwise a pure-noise undershoot would get its own logarithmic branch.
    vmin = vmin if vmin < -linthresh else 0.0
    vmax = vmax if vmax > linthresh else 0.0
    if vmin == 0.0 and vmax == 0.0:
        # Everything was inside the floor; fall back to a nominal symmetric
        # range so matplotlib still has something drawable.
        vmin, vmax = -1.0, 1.0
        linthresh = 1.0e-6

    # How many decades each branch spans, and hence where zero sits.
    neg_range = max(0.0, math.log10(-vmin / linthresh)) if vmin < 0.0 else 0.0
    pos_range = max(0.0, math.log10(vmax / linthresh)) if vmax > 0.0 else 0.0
    total_range = max(neg_range + pos_range, np.finfo(float).eps)
    split = neg_range / total_range

    # Divide the requested contour levels between branches in proportion to
    # their decade spans; at least 2 per non-empty branch so it renders.
    n_neg = max(2, int(round(n * neg_range / total_range))) if neg_range > 0.0 else 0
    n_pos = max(2, int(round(n * pos_range / total_range))) if pos_range > 0.0 else 0

    level_parts = []
    if vmin < 0.0:
        # Negative levels run from the extreme inward to the floor.  `min`
        # guards the case where the extreme is itself inside the floor.
        neg_stop = min(linthresh, -vmin)
        level_parts.append(-np.logspace(np.log10(-vmin), np.log10(neg_stop), n_neg))
    if vmin < 0.0 < vmax:
        # An explicit zero level draws the white band's boundary.
        level_parts.append(np.array([0.0]))
    if vmax > 0.0:
        pos_start = min(linthresh, vmax)
        level_parts.append(np.logspace(np.log10(pos_start), np.log10(vmax), n_pos))

    # `unique` also sorts, which matplotlib requires of contour levels.
    levels = np.unique(np.concatenate(level_parts))
    norm = SignedLogNorm(linthresh=linthresh, vmin=vmin, vmax=vmax, split=split)
    return masked, levels, norm, signed_log_cmap(split)


def signed_log_scale(values, decades=SIGNED_LOG_DECADES, n=96, linthresh_floor=0.0):
    """Signed-log scale with the range derived from ``values`` themselves.

    Use for still figures, where each figure may set its own scale.  For
    movies use :func:`fixed_signed_log_scale` instead, so the colorbar does
    not change meaning between frames.

    Degenerate inputs are handled before the shared core runs: an all-masked
    array gets a nominal symmetric range, and a constant array is widened
    toward zero so it has a drawable span.
    """
    masked = mask_exact_zero(values)
    finite = masked.compressed()
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        vmin, vmax = -1.0, 1.0
    else:
        vmin, vmax = np.min(finite), np.max(finite)
        if vmin == vmax:
            # A constant field: extend the range to zero so there is a span
            # to draw, keeping the constant at one end.
            if vmin < 0.0:
                vmax = 0.0
            elif vmax > 0.0:
                vmin = 0.0
            else:
                vmin, vmax = -1.0, 1.0
    return _signed_log_from_range(values, vmin, vmax, decades, n, linthresh_floor)


def fixed_signed_log_scale(values, vmin, vmax, decades=SIGNED_LOG_DECADES, n=96,
                           display_floor=None):
    """Signed-log scale over a caller-supplied ``[vmin, vmax]``.

    Movies scan every frame once up front (see :func:`movie_scale_range`) and
    then pass that global range here for each frame, so a given color means
    the same value throughout the movie.
    """
    floor = display_floor if display_floor is not None else 0.0
    return _signed_log_from_range(values, vmin, vmax, decades, n, floor)


def finite_range(values):
    """Return ``(min, max)`` over finite nonzero entries, padded if degenerate.

    Zeros are excluded because they carry no scale information on a log axis
    (and are masked in the drawing anyway).  A degenerate result -- no finite
    data, or a single repeated value -- is padded into a real interval so
    downstream scale code always receives something drawable.
    """
    data = np.asarray(values, dtype=float)
    finite = data[np.isfinite(data) & (data != 0.0)]
    if finite.size == 0:
        return -1.0, 1.0
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    if lo == hi:
        pad = max(abs(lo), 1.0) * 1.0e-6
        return lo - pad, hi + pad
    return lo, hi


def movie_scale_range(frames, percentile=None, subsample=None):
    """Return one ``(vmin, vmax)`` covering every frame of a movie.

    A movie must fix its color scale up front: if each frame rescaled to its
    own extremes, a color would mean a different value in every frame and the
    animation would show scale changes rather than physics.

    ``percentile`` (e.g. ``99.5``) uses a symmetric percentile pair instead of
    the absolute extremes, which is the right choice for growth-rate movies
    where a handful of near-noise-floor pixels produce enormous outliers.

    ``subsample`` takes every N-th element when estimating percentiles, which
    keeps this scan cheap for large frame stacks; extremes are unaffected by
    subsampling often enough that the saving is worth it.
    """
    pooled = []
    for frame in frames:
        data = np.asarray(frame, dtype=float).ravel()
        data = data[np.isfinite(data) & (data != 0.0)]
        if data.size == 0:
            continue
        if subsample is not None and subsample > 1 and data.size > subsample:
            data = data[::max(1, data.size // subsample)]
        pooled.append(data)
    if not pooled:
        return -1.0, 1.0
    combined = np.concatenate(pooled)
    if percentile is not None:
        lo = float(np.percentile(combined, 100.0 - percentile))
        hi = float(np.percentile(combined, percentile))
    else:
        lo = float(np.min(combined))
        hi = float(np.max(combined))
    if lo == hi:
        pad = max(abs(lo), 1.0) * 1.0e-6
        return lo - pad, hi + pad
    return lo, hi


def contour_line_levels(values, levels):
    """Restrict contour levels to the range the data actually spans.

    Matplotlib happily accepts levels outside the data range and simply draws
    nothing for them -- but they still appear in the colorbar, implying
    structure that is not present.  This trims to the levels that will
    produce a line.

    Falls back to eight evenly spaced levels when fewer than two survive
    (which happens for a nearly constant field), and to a single level when
    the field is exactly constant.
    """
    data = np.ma.asarray(values)
    finite = data.compressed()
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return levels

    lo = np.min(finite)
    hi = np.max(finite)
    in_range = levels[(levels >= lo) & (levels <= hi)]
    if in_range.size >= 2:
        return in_range
    if lo < hi:
        return np.linspace(lo, hi, 8)
    return np.array([lo])


def semilog_line_values(values):
    """Split a signed array into positive and ``|negative|`` branches.

    A log-scaled 1-D axis cannot show negative values, so a signed curve is
    drawn as two curves: the positive part, and the absolute value of the
    negative part (styled differently by the caller).  Each branch is masked
    where the other applies, so matplotlib breaks the line rather than
    connecting across the sign change.
    """
    data = np.asarray(values, dtype=float)
    positive = np.ma.masked_where(~(data > 0.0), data)
    negative = np.ma.masked_where(~(data < 0.0), -data)
    return positive, negative


def cartesian_mesh(x, theta):
    """Convert native ``(x, theta)`` coordinates to Cartesian ``(v_par, v_perp)``.

    The one coordinate transform every 2-D plotter uses: ``v_par = x cos(theta)``,
    ``v_perp = x sin(theta)``.  ``x`` and ``theta`` may be 1-D coordinate
    vectors (in which case they are meshed first) or already-meshed 2-D
    arrays.

    Returns ``(v_par, v_perp)`` as 2-D arrays.
    """
    x = np.asarray(x, dtype=float)
    theta = np.asarray(theta, dtype=float)
    if x.ndim == 1 and theta.ndim == 1:
        x, theta = np.meshgrid(x, theta, indexing="ij")
    return x * np.cos(theta), x * np.sin(theta)


# ---------------------------------------------------------------------------
# Section 2: colorbars and logarithmic ticks
# ---------------------------------------------------------------------------


def signed_log_side_ticks(lo, hi, max_ticks):
    """Choose decade ticks (powers of ten) spanning ``[lo, hi]``.

    Returns at most ``max_ticks`` values, thinning by taking every N-th
    decade when the span is wide, while always keeping the top decade so the
    labelled range reaches the data.
    """
    if hi < lo:
        return np.array([])

    # Inner decades: ceil of the low end, floor of the high end.
    lo_exp = int(np.ceil(np.log10(lo)))
    hi_exp = int(np.floor(np.log10(hi)))
    if hi_exp < lo_exp:
        # Less than one full decade spanned; label the top only.
        return np.array([hi])

    exps = list(range(lo_exp, hi_exp + 1))
    if len(exps) > max_ticks:
        step = int(np.ceil(len(exps) / max_ticks))
        exps = exps[::step]
        if exps[-1] != hi_exp:
            exps.append(hi_exp)
    return np.array([10.0 ** e for e in exps])


def prune_close_ticks(ticks, norm, min_gap=0.12):
    """Drop ticks whose *drawn positions* would overlap.

    Ticks well separated in value can still be adjacent on a signed-log bar
    (the branches compress hard near the noise floor).  Spacing is therefore
    checked in normalized colorbar coordinates, not in data units.

    When two ticks collide the later one replaces the earlier, which keeps
    the range endpoints -- the most informative labels -- rather than
    dropping them.
    """
    if len(ticks) <= 2:
        return np.array(ticks)

    kept = []
    positions = []
    for tick in ticks:
        pos = float(norm([tick])[0])
        if not positions or abs(pos - positions[-1]) >= min_gap:
            kept.append(tick)
            positions.append(pos)
        else:
            kept[-1] = tick
            positions[-1] = pos
    return np.array(kept)


def signed_log_ticks(linthresh, vmin, vmax, norm, max_ticks=7):
    """Build the tick list for a two-sided signed-log colorbar.

    Composition: the exact extremes (so the reader sees the true range),
    decade ticks within each branch, and zero when both signs are present.
    The per-branch budget is half the total, less one for zero.

    Inner ticks start one decade *above* ``linthresh`` on a two-sided bar,
    because a tick exactly at the noise floor sits on the white band's edge
    and reads as spurious precision.
    """
    ticks = []
    side_ticks = max(1, (max_ticks - 1) // 2)
    tick_low = linthresh * 10.0 if vmin < 0.0 < vmax else linthresh
    if vmin < 0.0:
        ticks.append(np.array([vmin]))
        neg = signed_log_side_ticks(tick_low, -vmin, side_ticks)
        if neg.size:
            # Negate and reverse so the negative branch stays ascending.
            ticks.append(-neg[::-1])
    if vmin < 0.0 < vmax:
        ticks.append(np.array([0.0]))
    if vmax > 0.0:
        pos = signed_log_side_ticks(tick_low, vmax, side_ticks)
        if pos.size:
            ticks.append(pos)
        ticks.append(np.array([vmax]))
    ticks = np.unique(np.concatenate(ticks)) if ticks else np.array([0.0])
    return prune_close_ticks(ticks, norm)


def signed_log_tick_label(value, _=None):
    """Format a colorbar tick as compact LaTeX.

    Exact powers of ten render as ``10^n`` (no redundant ``1x``); everything
    else as ``m e n``.  The 2%% tolerance catches mantissas that are 1.0 only
    after rounding, and the ``>= 10`` renormalization handles a mantissa that
    rounds up past the decade boundary (9.99 -> 10 becomes 1.0 with n+1).

    The second parameter exists because matplotlib's ``FuncFormatter`` passes
    a tick index; it is unused.
    """
    if value == 0.0:
        return "0"
    sign = "-" if value < 0.0 else ""
    abs_value = abs(value)
    exponent = int(np.floor(np.log10(abs_value)))
    mantissa = abs_value / 10.0 ** exponent
    rounded_mantissa = float(f"{mantissa:.2g}")
    if rounded_mantissa >= 10.0:
        rounded_mantissa /= 10.0
        exponent += 1
    if abs(mantissa - 1.0) < 0.02:
        return rf"${sign}10^{{{exponent}}}$"
    if abs(rounded_mantissa - 1.0) < 0.02:
        return rf"${sign}10^{{{exponent}}}$"
    return rf"${sign}{rounded_mantissa:.2g}\mathrm{{e}}{{{exponent}}}$"


def negative_log_tick_label(value, index=None):
    """Label an all-negative log colorbar, whose axis carries ``|value|``."""
    return signed_log_tick_label(-value, index)


def compact_colorbar_kwargs():
    """Geometry for a colorbar that does not crowd its axes.

    These figures pack several panels together, so colorbars are deliberately
    slim and tight against their axes.
    """
    return {"shrink": 0.64, "fraction": 0.032, "pad": 0.008, "aspect": 24}


def polish_colorbar(cbar):
    """Shrink colorbar tick text to match the compact geometry."""
    cbar.ax.tick_params(labelsize=7, length=3, pad=1)


def colorbar_target(ax, cax):
    """Choose between stealing space from ``ax`` and using an explicit ``cax``.

    Multi-panel figures pre-allocate a dedicated colorbar axes (``cax``) so
    that panels stay aligned; single-panel figures let matplotlib take the
    space from the plot axes.
    """
    if cax is None:
        return {"ax": ax, **compact_colorbar_kwargs()}
    return {"cax": cax}


def one_sided_log_ticks(lo, hi, max_ticks=6, min_endpoint_gap=0.12):
    """Decade ticks for a one-sided log bar, without crowding the top.

    Keeps the exact maximum only when it is visibly clear of the highest
    decade tick: a maximum of 1.13 next to ``10^0`` would print two labels at
    nearly the same height, so the rounder ``10^0`` is preferred.
    """
    ticks = signed_log_side_ticks(lo, hi, max_ticks)
    ticks = np.unique(ticks[np.isfinite(ticks)])
    if ticks.size == 0:
        return np.array([hi])

    log_norm = mpl.colors.LogNorm(vmin=lo, vmax=hi)
    endpoint_gap = abs(float(log_norm(hi)) - float(log_norm(ticks[-1])))
    if not np.isclose(ticks[-1], hi) and endpoint_gap >= min_endpoint_gap:
        ticks = np.append(ticks, hi)
    return ticks


def _one_sided_log_colorbar(fig, comp, ax, cax, lo, hi, formatter):
    """Draw a plain ``LogNorm`` colorbar for data of a single sign.

    A standalone ``ScalarMappable`` is used rather than the contour artist,
    because the artist carries the signed-log norm whose unused branch would
    otherwise appear on the bar.
    """
    sm = mpl.cm.ScalarMappable(
        norm=mpl.colors.LogNorm(vmin=lo, vmax=hi), cmap=comp.cmap
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, **colorbar_target(ax, cax))
    cbar.set_ticks(one_sided_log_ticks(lo, hi))
    cbar.formatter = mpl.ticker.FuncFormatter(formatter)
    cbar.update_ticks()
    polish_colorbar(cbar)
    return cbar


def add_signed_log_colorbar(fig, comp, ax, cax=None):
    """Attach a signed-log colorbar, collapsing to one-sided when possible.

    When the data turned out to be all one sign, a two-sided bar would devote
    half its length to an empty branch -- so those cases are redrawn as a
    simple log bar over the populated branch only.
    """
    if isinstance(comp.norm, SignedLogNorm) and comp.norm.vmin >= 0.0:
        lo = max(comp.norm.linthresh, np.finfo(float).tiny)
        # Nudge hi above lo so LogNorm always has a nonzero span.
        hi = max(comp.norm.vmax, lo * (1.0 + np.finfo(float).eps))
        return _one_sided_log_colorbar(
            fig, comp, ax, cax, lo, hi, signed_log_tick_label
        )

    if isinstance(comp.norm, SignedLogNorm) and comp.norm.vmax <= 0.0:
        lo = max(comp.norm.linthresh, np.finfo(float).tiny)
        hi = max(-comp.norm.vmin, lo * (1.0 + np.finfo(float).eps))
        # Axis carries magnitudes; the formatter restores the minus sign.
        return _one_sided_log_colorbar(
            fig, comp, ax, cax, lo, hi, negative_log_tick_label
        )

    sm = mpl.cm.ScalarMappable(norm=comp.norm, cmap=comp.cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, **colorbar_target(ax, cax))
    cbar.set_ticks(signed_log_ticks(
        comp.norm.linthresh, comp.norm.vmin, comp.norm.vmax, comp.norm
    ))
    cbar.formatter = mpl.ticker.FuncFormatter(signed_log_tick_label)
    cbar.update_ticks()
    polish_colorbar(cbar)
    return cbar


# ---------------------------------------------------------------------------
# Section 3: the generic drawers
# ---------------------------------------------------------------------------


def contour2d(fig, ax, xx, yy, values, *, fixed_range=None, filled=False,
              title=None, style_axes=None, floor=0.0, cax=None,
              restrict_levels=False, linewidths=0.85, extend=None,
              decades=SIGNED_LOG_DECADES, title_fontsize=13):
    """Draw a signed-log 2-D contour plot on ``ax`` and attach its colorbar.

    This is the single entry point for every 2-D field in both projects:
    solution snapshots, growth rates, and quasilinear/collisional
    coefficients.

    Parameters
    ----------
    fig, ax
        Target figure and axes.  ``fig`` is needed because the colorbar is
        attached to the figure, not the axes.
    xx, yy, values
        Coordinate arrays and the field.  Passed straight to matplotlib, so
        either 1-D coordinate vectors or 2-D meshes work.
    fixed_range
        ``(vmin, vmax)`` to hold the scale constant -- movies pass the result
        of :func:`movie_scale_range` here.  ``None`` derives the range from
        ``values`` (correct for stills).
    filled
        ``True`` uses ``contourf`` (solid bands), ``False`` uses ``contour``
        (lines).  Lines are the default because they let overlapping
        structure stay visible.
    title, style_axes
        The physics-specific parts, supplied by the caller.  ``style_axes``
        is a callable taking ``ax`` that sets labels/limits/aspect -- this is
        how project-specific axis conventions stay out of this shared module.
    floor
        Noise floor below which values render as white.  Threaded into the
        scale computation rather than applied afterwards, so it also sets the
        signed-log branch boundaries.
    cax
        Explicit colorbar axes for multi-panel layouts; ``None`` steals space
        from ``ax``.
    restrict_levels
        Trim levels to the data's actual span (see
        :func:`contour_line_levels`).  Solution plots want this; coefficient
        plots deliberately keep the full level set so panels stay comparable.
    linewidths
        Line width for unfilled contours.  ``None`` omits the argument so
        matplotlib's default applies -- the coefficient panels rely on that.
    extend
        Matplotlib's out-of-range arrow behaviour (``"both"`` for the
        coefficient panels, whose fixed levels can clip).
    decades
        Dynamic range of the signed-log scale when no ``floor`` is set.

    Returns the contour artist, so callers can add labels or further
    colorbars.
    """
    if fixed_range is None:
        masked, levels, norm, cmap = signed_log_scale(
            values, decades=decades, linthresh_floor=floor
        )
    else:
        masked, levels, norm, cmap = fixed_signed_log_scale(
            values, fixed_range[0], fixed_range[1],
            decades=decades, display_floor=floor,
        )

    if restrict_levels:
        levels = contour_line_levels(masked, levels)
    draw_kwargs = {"levels": levels, "norm": norm, "cmap": cmap}
    if extend is not None:
        draw_kwargs["extend"] = extend
    if filled:
        artist = ax.contourf(xx, yy, masked, **draw_kwargs)
    else:
        # Only pass linewidths when asked, so `None` means "matplotlib
        # default" rather than "zero-width invisible lines".
        if linewidths is not None:
            draw_kwargs["linewidths"] = linewidths
        artist = ax.contour(xx, yy, masked, **draw_kwargs)

    if title is not None:
        ax.set_title(title, fontsize=title_fontsize)
    # White background so masked (sub-floor) regions read as empty, matching
    # the colormaps' `set_bad("white")`.
    ax.set_facecolor("white")
    if style_axes is not None:
        style_axes(ax)

    add_signed_log_colorbar(fig, artist, ax, cax=cax)
    return artist


def line1d(ax, x, series, *, scale="linear", ylim=None, style_axes=None,
           title=None, legend=True, title_fontsize=13, legend_fontsize=8):
    """Draw one or more 1-D curves on ``ax``, linearly or on a log y-axis.

    Used for every line plot in both projects: reduced (marginal) solution
    curves, orbit/pitch-scattering factors, particle-loss and energy
    histories.

    ``series`` is a list of ``(values, label, style_kwargs)`` triples, where
    ``style_kwargs`` goes straight to ``ax.plot`` (color, linestyle, and so
    on).  A ``None`` label omits that curve from the legend.

    On ``scale="log"`` each signed curve is split into a positive branch and
    an ``|negative|`` branch (see :func:`semilog_line_values`), because a log
    axis cannot represent negative values.  The negative branch keeps the same
    linestyle as the positive one but is always drawn in a fixed pink
    (``#cf1f63``), regardless of the series' own color, and its label is
    suffixed ``(negative)`` -- so a sign flip reads as "the alarming color"
    rather than blending into whatever the series happened to be colored.
    This matches the original hand-written negative-branch convention
    exactly; it is not a caller-configurable style.
    """
    NEGATIVE_COLOR = "#cf1f63"
    artists = []
    for values, label, style in series:
        style = dict(style or {})
        if scale == "log":
            positive, negative = semilog_line_values(values)
            # Positive branch keeps the caller's styling verbatim.
            if np.ma.count(positive):
                artists.extend(ax.plot(x, positive, label=label, **style))
            # Negative branch: caller's linestyle/width, fixed pink color.
            if np.ma.count(negative):
                negative_style = dict(style)
                negative_style["color"] = NEGATIVE_COLOR
                negative_label = None if label is None else f"{label} (negative)"
                artists.extend(
                    ax.plot(x, negative, label=negative_label, **negative_style)
                )
        else:
            artists.extend(ax.plot(x, values, label=label, **style))

    if scale == "log":
        ax.set_yscale("log")
    if ylim is not None:
        ax.set_ylim(*ylim)
    if title is not None:
        ax.set_title(title, fontsize=title_fontsize)
    if style_axes is not None:
        style_axes(ax)
    # Only build a legend when at least one curve was actually labelled,
    # otherwise matplotlib emits an empty-legend warning.
    if legend and any(label is not None for _, label, _ in series):
        ax.legend(fontsize=legend_fontsize)
    return artists


def level_contour(ax, xx, yy, levels, *, cmap_name="Blues",
                  colorbar_label="local refinement level", fig=None,
                  cmap_low=0.02, cmap_high=0.92):
    """Draw the adaptive-grid refinement-level map as a filled contour plot.

    Deliberately *not* folded into :func:`contour2d`: the field here is a
    small integer (how many times the grid has been refined at each point),
    so its coloring is categorical, not a continuous scale.  Each integer
    level gets exactly one flat color band and one integer colorbar tick.

    How the banding works: contour boundaries are placed at half-integers
    (``arange(min, max+2) - 0.5``), so band *k* covers ``[k-0.5, k+0.5)`` and
    therefore contains exactly the cells whose level is ``k``.  Colors are
    sampled from ``Blues`` between ``cmap_low`` and ``cmap_high`` rather than
    the full range, because the extreme ends are near-white (invisible) and
    near-black (indistinguishable from adjacent bands).

    This reproduces the existing adaptive-grid visual exactly -- it is a
    relocation, not a redesign.  In particular the colorbar deliberately does
    **not** use this module's compact multi-panel styling
    (:func:`compact_colorbar_kwargs`/:func:`polish_colorbar`): the grid figure
    is always a single standalone panel, and the hand-written original used
    matplotlib's own default colorbar sizing with only ``pad=0.02`` set.

    Returns the ``contourf`` artist.
    """
    level_field = np.asarray(levels)
    min_level = int(np.min(level_field))
    max_level = int(np.max(level_field))
    # One band per integer level, with boundaries offset by half.
    contour_levels = np.arange(min_level, max_level + 2) - 0.5
    band_count = len(contour_levels) - 1
    base = mpl.colormaps[cmap_name]
    colors = [
        base(value)
        for value in np.linspace(cmap_low, cmap_high, max(band_count, 1))
    ]
    artist = ax.contourf(xx, yy, level_field, levels=contour_levels,
                         colors=colors)
    target_fig = fig if fig is not None else ax.get_figure()
    cbar = target_fig.colorbar(artist, ax=ax, pad=0.02)
    # Integer ticks, one per band, centered in the band.
    cbar.set_ticks(np.arange(min_level, max_level + 1))
    cbar.set_label(colorbar_label)
    return artist


# ---------------------------------------------------------------------------
# Section 4: output -- PNGs and run directories
# ---------------------------------------------------------------------------


def save_png(fig, output_dir, stem, dpi=220):
    """Write ``fig`` to ``output_dir/stem.png`` and report the path.

    ``bbox_inches="tight"`` crops surrounding whitespace, which matters
    because these figures are dropped straight into notes and papers.  Note
    it makes the output pixel dimensions depend on the content.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{stem}.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"saved {path}")
    return path


def save_all_png(figures, output_dir, dpi=450):
    """Save an iterable of ``(stem, figure)`` pairs at publication DPI."""
    for stem, fig in figures:
        save_png(fig, output_dir, stem, dpi=dpi)


def timestamped_output_dir(base_dir):
    """Create and return a fresh ``base_dir/YYYY-MM-DD_HH-MM-SS`` directory.

    Every run writes into its own directory so results are never silently
    overwritten and can be compared after the fact.  A ``_01``, ``_02``, ...
    suffix is appended if the timestamp already exists (two runs starting in
    the same second).
    """
    stamp = _datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = os.path.join(base_dir, stamp)
    counter = 1
    while os.path.exists(path):
        path = os.path.join(base_dir, f"{stamp}_{counter:02d}")
        counter += 1
    os.makedirs(path, exist_ok=True)
    return path


def save_input_files(output_dir, *input_files):
    """Copy the input decks that produced a figure set alongside it.

    This is what makes a figure directory self-describing months later: the
    exact parameters are stored next to the plots.  ICRF passes both
    ``input_solver.txt`` and ``input_build.txt``; LHCD has no build deck and
    passes only the solver one.

    A missing file warns rather than raising -- losing the archive copy is
    not a reason to discard a completed plotting run.
    """
    os.makedirs(output_dir, exist_ok=True)
    saved = []
    for src in input_files:
        if not src:
            continue
        if not os.path.exists(src):
            print(f"warning: input file not found: {src}")
            continue
        dst = os.path.join(output_dir, os.path.basename(src))
        shutil.copyfile(src, dst)
        print(f"saved {dst}")
        saved.append(dst)
    return saved
