"""Movie rendering: parallel frames over timesteps, then ffmpeg encoding.

The encoder is resolved *before* frames are rendered.  Encoding happens only
after every frame exists, so an absent or H.264-less ffmpeg would otherwise
waste the whole rendering pass before reporting itself.

Every movie in both projects is "one frame per timestep, timesteps are
independent" -- so there is exactly one rendering strategy here
(:func:`render_movie`), not one per plotter.  Workers render PNG frames into a
temp directory; one ffmpeg call stitches them in order.  There is no
``FuncAnimation`` path: that API is inherently serial, and every movie here
benefits from parallelizing over frames.

Used by: every plotter's ``plot_movie`` in ``ICRF_2D/plot/`` and
``LHCD_2D/plot/``, and ``show_all.py`` in both.

Depends on: ``plot_common.runtime`` (worker pool), an ``ffmpeg`` binary or
``imageio-ffmpeg``.  Does not import ``plot_common.static`` -- what a frame
draws is entirely the caller's ``draw_task``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

from plot_common.runtime import process_pool, worker_count


# ---------------------------------------------------------------------------
# ffmpeg discovery and encoding
# ---------------------------------------------------------------------------

# Cache for resolve_ffmpeg(): probing an ffmpeg binary costs a subprocess
# launch, and the answer cannot change within one process.
_FFMPEG_CHOICE = None


def encoder_list(binary):
    """Return one ffmpeg binary's encoder listing, or ``""`` if it fails.

    Failure (missing binary, non-zero exit) is reported as an empty string
    rather than an exception, because the caller is probing several
    candidates and a failed probe simply means "not this one".
    """
    try:
        result = subprocess.run(
            [binary, "-hide_banner", "-encoders"],
            capture_output=True, text=True, check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return result.stdout


def resolve_ffmpeg():
    """Pick an ffmpeg binary, preferring one that can encode H.264.

    Why this exists: cluster ffmpeg modules are frequently built *without*
    libx264, because x264 is GPL.  Such a build still writes a structurally
    valid ``.mp4`` -- using MPEG-4 Part 2, a 1999 codec -- but browsers and
    QuickTime cannot decode it and display a **green frame**.  The failure is
    silent and only visible when someone tries to watch the result, after a
    full render has completed.

    So: probe the system ffmpeg first, then ``imageio-ffmpeg``'s bundled
    static build (which always includes libx264 and installs via pip into a
    venv, needing no module changes).  Prefer whichever has a real H.264
    encoder.

    Returns ``(binary, h264_encoder_name_or_None)``.  A ``None`` encoder name
    tells callers to warn loudly before falling back to MPEG-4.
    """
    global _FFMPEG_CHOICE
    if _FFMPEG_CHOICE is not None:
        return _FFMPEG_CHOICE

    candidates = []
    system = shutil.which("ffmpeg")
    if system:
        candidates.append(system)
    try:
        import imageio_ffmpeg

        bundled = imageio_ffmpeg.get_ffmpeg_exe()
        if bundled and os.path.exists(bundled) and bundled not in candidates:
            candidates.append(bundled)
    except Exception:
        # imageio-ffmpeg is optional; its absence just means one fewer
        # candidate, not an error.
        pass

    fallback = None
    for binary in candidates:
        listing = encoder_list(binary)
        if not listing:
            continue
        # libopenh264 (Cisco's BSD implementation) is an acceptable
        # substitute when libx264 is unavailable for licensing reasons.
        for name in ("libx264", "libopenh264"):
            if name in listing:
                _FFMPEG_CHOICE = (binary, name)
                return _FFMPEG_CHOICE
        # Remember the first working binary in case none has H.264.
        if fallback is None:
            fallback = binary

    _FFMPEG_CHOICE = (fallback or system, None)
    return _FFMPEG_CHOICE


def movie_encoder_problem(output_file):
    """Return why this movie target cannot be written, or ``None`` if it can.

    Called *before* frames are rendered.  Encoding happens only after every
    frame exists, so without this pre-flight an absent ffmpeg wastes an
    entire rendering pass before reporting itself.

    Only ``.mp4`` needs ffmpeg; ``.gif`` goes through Pillow and always works.
    """
    if os.path.splitext(output_file)[1].lower() != ".mp4":
        return None
    binary, _ = resolve_ffmpeg()
    if binary is None:
        return (
            "ffmpeg was not found on PATH, so .mp4 output is impossible. "
            "Load an ffmpeg module, run 'pip install imageio-ffmpeg', or "
            "use a .gif output extension instead."
        )
    return None


def _mpeg4_fallback_warning(output_file):
    """Warn loudly that a movie was written in an unplayable codec.

    This warning exists because the failure it describes is otherwise
    invisible: the file is valid, the run reports success, and the problem
    only surfaces as a green rectangle in a video player later.
    """
    print(
        f"warning: {os.path.basename(output_file)} was encoded as MPEG-4 "
        "Part 2 because no available ffmpeg has an H.264 encoder. Browsers "
        "and QuickTime cannot play it and will show a green frame. Fix with "
        "'pip install imageio-ffmpeg'.",
        flush=True,
    )


def encode_png_frames(frame_dir, frame_count, output_file, fps, threads=None):
    """Encode ``frame_%06d.png`` files in ``frame_dir`` into one movie.

    ``.mp4`` goes through ffmpeg with a codec ladder (H.264 first, MPEG-4 as a
    warned fallback); any other extension is written as an animated GIF via
    Pillow, needing no external binary.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    ext = os.path.splitext(output_file)[1].lower()
    if ext == ".mp4":
        problem = movie_encoder_problem(output_file)
        if problem is not None:
            raise RuntimeError(problem)
        binary, h264 = resolve_ffmpeg()
        pattern = os.path.join(frame_dir, "frame_%06d.png")
        last_error = None

        # Quality settings per codec.  Both are explicit because ffmpeg's
        # MPEG-4 default is 200 kbps, which destroys thin contour lines at
        # plot resolution -- roughly a tenth of what this content needs.
        candidates = []
        if h264 is not None:
            candidates.append((h264, ["-crf", "18"] if h264 == "libx264" else []))
        candidates.append(("mpeg4", ["-qscale:v", "2"]))

        for codec, quality in candidates:
            cmd = [
                binary, "-y", "-framerate", str(max(fps, 1)),
                "-i", pattern, "-vcodec", codec,
                # yuv420p is the pixel format every player supports; ffmpeg
                # would otherwise pick one that many cannot decode.
                "-pix_fmt", "yuv420p",
            ]
            cmd.extend(quality)
            if threads is not None:
                cmd.extend(["-threads", str(max(1, int(threads)))])
            cmd.append(output_file)
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                               stderr=subprocess.PIPE)
                if codec == "mpeg4":
                    _mpeg4_fallback_warning(output_file)
                return output_file
            except Exception as exc:
                last_error = exc
        raise RuntimeError(
            "MP4 output failed with every available ffmpeg encoder."
        ) from last_error

    # --- GIF path: no external binary required ----------------------------
    from PIL import Image

    frame_paths = [
        os.path.join(frame_dir, f"frame_{i:06d}.png")
        for i in range(frame_count)
    ]
    # Adaptive palette conversion keeps GIF size manageable for line art.
    images = [Image.open(path).convert("P", palette=Image.Palette.ADAPTIVE)
              for path in frame_paths]
    duration_ms = int(round(1000.0 / max(fps, 1)))
    images[0].save(
        output_file,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
    )
    for image in images:
        image.close()
    return output_file


# ---------------------------------------------------------------------------
# Parallel frame rendering
# ---------------------------------------------------------------------------


def render_movie(draw_task, frame_count, output_file, *, fps=8, dpi=180,
                 workers=None, initializer=None, initargs=()):
    """Render ``frame_count`` frames in parallel, then encode them into a movie.

    Parameters
    ----------
    draw_task
        A **module-level** function (must be picklable under the ``spawn``
        start method) taking one dict ``{"index": i, "frame_dir": str,
        "dpi": dpi}``.  It must draw frame ``i`` and save it as
        ``frame_dir/frame_{i:06d}.png``.  Any data the frame needs beyond its
        index (a shared cache, a fixed color scale, ...) must arrive via
        ``initializer``/``initargs``, not be captured in a closure -- closures
        over large objects do not survive being pickled to a worker.
    frame_count
        Number of frames to render, indices ``0 .. frame_count - 1``.
    output_file
        Final movie path.  Its extension selects the encoder (``.mp4`` via
        ffmpeg, anything else via Pillow GIF).
    workers
        Pool width; ``None``/0 uses every logical CPU.
    initializer, initargs
        Passed straight to :func:`plot_common.runtime.process_pool`.  This is
        how large shared state (a :class:`~plot_common.reader.SnapshotCache`,
        a fixed movie-wide color scale) reaches every worker exactly **once**
        rather than once per frame -- the pool initializer runs one time per
        worker process, not one time per task.

    The encoder is pre-flighted with :func:`movie_encoder_problem` before any
    frame is rendered, so a missing H.264 encoder is reported immediately
    rather than after the whole render completes.
    """
    problem = movie_encoder_problem(output_file)
    if problem is not None:
        raise RuntimeError(problem)

    frame_dir = tempfile.mkdtemp(prefix="movie_frames_")
    try:
        tasks = [
            {"index": i, "frame_dir": frame_dir, "dpi": dpi}
            for i in range(frame_count)
        ]
        pool_width = min(worker_count(workers), max(1, len(tasks)))
        if pool_width == 1 or len(tasks) <= 1:
            # Serial path: still run the initializer once, in-process, since
            # draw_task assumes whatever state it sets up (e.g. a shared cache
            # stashed in a module global) already exists.  The pool path below
            # gets this for free -- ProcessPoolExecutor runs the initializer
            # once per worker automatically.
            if initializer is not None:
                initializer(*initargs)
            for task in tasks:
                draw_task(task)
        else:
            with process_pool(
                pool_width, initializer=initializer, initargs=initargs,
            ) as pool:
                # chunksize=1: frame costs vary (grid refinement, data
                # complexity), so larger chunks would strand work at the tail.
                list(pool.map(draw_task, tasks, chunksize=1))

        threads = max(1, worker_count(workers) // max(1, frame_count))
        encode_png_frames(frame_dir, frame_count, output_file, fps,
                          threads=threads)
    finally:
        shutil.rmtree(frame_dir, ignore_errors=True)

    # Report like save_png does: a run that produced movies silently looks
    # like a run that skipped them.
    print(f"saved {output_file} ({frame_count} frames)", flush=True)
    return output_file
