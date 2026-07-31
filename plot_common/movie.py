"""Movie rendering: draw frames in a loop, then one ffmpeg call.

Deliberately serial.  Measured on the real workloads, a process pool saved
roughly nothing on 1-D movies (pool startup ate the gains) and under a minute
on a full 2-D batch -- not worth carrying spawn, pickling, and per-plotter
worker boilerplate.  The only parallelism left anywhere in the plotting stack
is asgard's own OpenMP inside snapshot reconstruction.

The encoder binary comes from imageio-ffmpeg (a pip-installed static build
that always includes libx264), with a system ffmpeg as fallback.  A missing
binary is reported *before* frames are rendered, not after.

Used by: every plotter's ``plot_movie``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

import matplotlib.pyplot as plt


def require_ffmpeg():
    """Return the ffmpeg binary path, or raise with a clear instruction.

    Prefers imageio-ffmpeg's bundled binary: it is a static build that
    always includes libx264, whereas system/module ffmpegs are sometimes
    built without it (x264 is GPL) and then silently produce movies most
    players show as a green frame.  A system ffmpeg is the fallback when the
    package is absent.

    Checked before rendering: encoding happens only after every frame
    exists, so a missing ffmpeg would otherwise waste the whole rendering
    pass before reporting itself.
    """
    try:
        import imageio_ffmpeg

        binary = imageio_ffmpeg.get_ffmpeg_exe()
        if binary and os.path.exists(binary):
            return binary
    except Exception:
        pass
    binary = shutil.which("ffmpeg")
    if binary is None:
        raise RuntimeError(
            "No ffmpeg available; it is required for movie output. "
            "Run 'pip install imageio-ffmpeg' (preferred) or install ffmpeg."
        )
    return binary


def encode_png_frames(frame_dir, frame_count, output_file, fps):
    """Encode ``frame_%06d.png`` files in ``frame_dir`` into an H.264 movie.

    ``-pix_fmt yuv420p`` because it is the one pixel format every player
    supports; ``-crf 18`` because ffmpeg's default rate destroys thin
    contour lines at plot resolution.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    subprocess.run(
        [require_ffmpeg(), "-y",
         "-framerate", str(max(fps, 1)),
         "-i", os.path.join(frame_dir, "frame_%06d.png"),
         "-vcodec", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
         output_file],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return output_file


def render_movie(draw_frame, frame_count, output_file, *, figsize,
                 fps=8, dpi=140):
    """Draw ``frame_count`` frames serially, then encode them into a movie.

    Owns everything about a frame except its content: the temp directory,
    the figure lifecycle (create, save, close), and the ``frame_%06d.png``
    naming contract read back by :func:`encode_png_frames` below.  The fixed
    ``figsize x dpi`` (never ``bbox_inches="tight"``) keeps every frame's
    pixel dimensions identical and even, which H.264 requires.

    Parameters
    ----------
    draw_frame
        Any callable (closures welcome -- there is no pickling) taking
        ``(fig, ax, index)``: draw frame ``index`` onto the provided axes.
    frame_count
        Number of frames, indices ``0 .. frame_count - 1``.
    output_file
        Final ``.mp4`` path.
    figsize
        Matplotlib figure size, fixed for the whole movie.
    """
    require_ffmpeg()   # fail before rendering, not after

    frame_dir = tempfile.mkdtemp(prefix="movie_frames_")
    try:
        for index in range(frame_count):
            fig, ax = plt.subplots(figsize=figsize)
            draw_frame(fig, ax, index)
            fig.savefig(os.path.join(frame_dir, f"frame_{index:06d}.png"),
                        dpi=dpi)
            plt.close(fig)
        encode_png_frames(frame_dir, frame_count, output_file, fps)
    finally:
        shutil.rmtree(frame_dir, ignore_errors=True)

    # Report like save_png does: a run that produced movies silently looks
    # like a run that skipped them.
    print(f"saved {output_file} ({frame_count} frames)", flush=True)
    return output_file
