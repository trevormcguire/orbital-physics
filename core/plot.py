
import os
import shutil
import tempfile
import subprocess
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from typing import Literal

from core.engine import SimulationEngine

def plot_orbits(
    engine: SimulationEngine,
    every_n: int = 1,
    plane: Literal["xy", "xz", "yz"] = "xy",
    separate: bool = False,
    with_velocity: bool = True,
    equal_axes: bool = True,
    labels: bool = True,
    alpha: float = 0.9,
    linewidth: float = 1.5,
    markersize: float = 50,
    last_k: Optional[int] = None,
    savepath: Optional[str] = None,
    show: bool = True,
    show_barycenter: bool = True,
    barycenter_trail: bool = False,
    bary_marker: str = "x",
    bary_size: float = 120,
):
    """
    Plot orbits using engine.history, and annotate the system barycenter.

    New args:
        show_barycenter: draw the current barycenter as a marker.
        barycenter_trail: also draw the barycenter's history (projected onto the plane).
        bary_marker: matplotlib marker for the barycenter (e.g., 'x', '*', 'P').
        bary_size: marker size for the barycenter.
    """
    # --- plane indices ---
    idx_map = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}
    if plane not in idx_map:
        raise ValueError("plane must be one of 'xy', 'xz', 'yz'")
    ix, iy = idx_map[plane]

    objs = list(engine.objects)
    uuids = [o.uuid for o in objs]
    masses = np.array([o.mass for o in objs], dtype=float)
    Mtot = masses.sum()

    # --- build trajectories per object (subsampled) ---
    trajs = {}
    # use the shortest history length across bodies (just in case)
    T = min(len(engine.history[u]) for u in uuids)
    slicer = slice(None if last_k is None else -int(last_k))
    step = max(1, int(every_n))

    for o in objs:
        arr = np.array(engine.history[o.uuid], dtype=float)[:T]
        arr = arr[slicer][::step]
        trajs[o.uuid] = arr

    # --- barycenter history (same slicing/subsampling) ---
    # R_cm(t) = sum_i m_i * r_i(t) / M
    # stack trajectories aligned along time
    stacks = [np.array(engine.history[u], dtype=float)[:T] for u in uuids]
    Rcm = (np.tensordot(masses, np.stack(stacks, axis=0), axes=(0, 0)) / Mtot)
    Rcm = Rcm[slicer][::step]  # apply same slicing/subsampling

    # --- figure/axes ---
    if separate:
        n = len(objs)
        cols = 2 if n > 1 else 1
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
        axes = np.atleast_1d(axes).ravel()
    else:
        fig, ax = plt.subplots(figsize=(8, 8))
        axes = np.array([ax])

    def label_for(o):
        return f"{o.uuid[:6]} (m={o.mass:.2e})"

    # --- plotting ---
    for obj, ax in zip(objs, (axes if separate else [axes[0]] * len(objs))):
        traj = trajs[obj.uuid]
        if traj.shape[0] == 0:
            continue

        x = traj[:, ix]
        y = traj[:, iy]
        ax.plot(x, y, alpha=alpha, linewidth=linewidth, label=(label_for(obj) if labels else None))
        cx, cy = x[-1], y[-1]
        ax.scatter([cx], [cy], s=markersize, marker="o")

        if with_velocity and isinstance(obj.velocity, np.ndarray):
            vx, vy = obj.velocity[ix], obj.velocity[iy]
            vnorm = np.hypot(vx, vy) + 1e-12
            span = max(np.ptp(x), np.ptp(y), 1.0)
            L = 0.05 * span  # cap arrow length to avoid autoscale blowups
            ax.arrow(
                cx,
                cy, L * vx / vnorm,
                L * vy / vnorm,
                head_width=0.08 * L,
                length_includes_head=True,
                linewidth=1.0
            )

        ax.set_xlabel(plane[0])
        ax.set_ylabel(plane[1])
        ax.grid(True, alpha=0.2)
        if equal_axes:
            ax.set_aspect("equal", adjustable="datalim")

    # --- barycenter annotation (on all axes) ---
    if show_barycenter and Rcm.shape[0] > 0:
        bx = Rcm[:, ix]
        by = Rcm[:, iy]
        for ax in axes:
            if barycenter_trail and len(bx) > 1:
                ax.plot(bx, by, linestyle="--", linewidth=1.2, alpha=0.7, label=("barycenter trail" if labels else None))
            ax.scatter([bx[-1]], [by[-1]], s=bary_size, marker=bary_marker, zorder=5, label=("barycenter" if labels else None))

    if not separate and labels:
        axes[0].legend(frameon=False, loc="best")
    elif separate:
        for ax in axes:
            if labels:
                ax.legend(frameon=False, loc="best")

    axes[0].set_title(f"Orbital Trajectories ({plane}-plane), every {every_n} steps")

    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig, axes


def render_orbit_video_no_deps(
    engine,
    out_path: str = "orbits.mp4",
    plane: str = "xy",
    fps: int = 30,
    duration_s: Optional[float] = None,
    frame_every_n: int = 1,
    separate: bool = False,
    with_velocity: bool = False,
    labels: bool = True,
    show_barycenter: bool = True,
    barycenter_trail: bool = True,
    dpi: int = 150,
    pad_frac: float = 0.08,
    tmp_dir: Optional[str] = None,
    cleanup: bool = True,
    # If True, we override aspect AFTER plotting to avoid warnings/jitter
    enforce_equal_aspect: bool = True,
    every_n: int = 1
):
    """
    Renders frames by repeatedly calling your existing `plot_orbits(engine_view, ...)`
    and stitches them with system ffmpeg (no extra Python deps).
    """
    # ------------ choose frames ------------
    uuids = list(engine.history.keys())
    T_full = min(len(engine.history[u]) for u in uuids)

    if duration_s is not None:
        total_frames = max(1, int(round(fps * duration_s)))
        stride = max(1, int(np.ceil(T_full / total_frames)))
    else:
        stride = max(1, int(frame_every_n))
        total_frames = (T_full - 1) // stride

    frame_indices = list(range(2, T_full + 1, stride))  # start at 2 so trails exist
    if duration_s is not None and len(frame_indices) > total_frames:
        frame_indices = frame_indices[:total_frames]

    # ------------ global axis limits (steady camera) ------------
    plane_idx = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}
    if plane not in plane_idx:
        raise ValueError("plane must be one of {'xy','xz','yz'}")
    ix, iy = plane_idx[plane]

    all_x, all_y = [], []
    for u in uuids:
        arr = np.asarray(engine.history[u])
        all_x.append(arr[:, ix])
        all_y.append(arr[:, iy])
    all_x = np.concatenate(all_x)
    all_y = np.concatenate(all_y)
    x_min, x_max = float(all_x.min()), float(all_x.max())
    y_min, y_max = float(all_y.min()), float(all_y.max())
    dx, dy = (x_max - x_min), (y_max - y_min)
    pad_x = pad_frac * (dx if dx > 0 else 1.0)
    pad_y = pad_frac * (dy if dy > 0 else 1.0)
    x_lim = (x_min - pad_x, x_max + pad_x)
    y_lim = (y_min - pad_y, y_max + pad_y)

    # ------------ temp frame dir ------------
    made_tmp = False
    if tmp_dir is None:
        tmp_dir = tempfile.mkdtemp(prefix="orbit_frames_")
        made_tmp = True
    os.makedirs(tmp_dir, exist_ok=True)
    pattern = os.path.join(tmp_dir, "frame_%06d.png")

    # Lazy import to avoid circulars; adapt to where your plotter lives:
    from core.plot import plot_orbits  # <-- change to your actual import

    # Tiny inner “engine view” with truncated history (no `types` usage)
    class EngineView:
        def __init__(self, objects, history):
            self.objects = objects
            self.history = history

    # ------------ render frames ------------
    for f_idx, t_idx in enumerate(frame_indices):
        eview = EngineView(
            objects=engine.objects,
            history={u: engine.history[u][:t_idx] for u in uuids},
        )
        # Avoid double aspect logic by passing equal_axes=False; we'll enforce below
        fig, axes = plot_orbits(
            eview,
            every_n=every_n,
            plane=plane,
            separate=separate,
            with_velocity=with_velocity,
            equal_axes=False,        # <--- important to avoid warning spam
            labels=labels,
            last_k=None,
            savepath=None,
            show=False,
            show_barycenter=show_barycenter,
            barycenter_trail=barycenter_trail,
        )

        # lock camera & aspect (use adjustable='box' so limits are respected)
        ax_list = axes if isinstance(axes, np.ndarray) else np.array([axes])
        for ax in ax_list:
            ax.set_xlim(*x_lim)
            ax.set_ylim(*y_lim)
            if enforce_equal_aspect:
                ax.set_aspect("equal", adjustable="box")

        # Save WITHOUT tight bounding, which often produces odd pixel widths
        fig.savefig(pattern % f_idx, dpi=dpi, bbox_inches=None)
        plt.close(fig)

    # ------------ stitch with ffmpeg (pad to even dims) ------------
    ffmpeg = shutil.which("ffmpeg")
    ext = os.path.splitext(out_path)[1].lower()
    ok = False
    try:
        if ffmpeg:
            if ext not in {".mp4", ".mov", ".mkv", ".gif"}:
                ext = ".mp4"
                out_path = os.path.splitext(out_path)[0] + ext

            if ext == ".gif":
                # palette pass (GIF doesn't require even dims, but keep consistent)
                palette = os.path.join(tmp_dir, "palette.png")
                cmd1 = [
                    ffmpeg, "-y", "-i", os.path.join(tmp_dir, "frame_%06d.png"),
                    "-vf", "palettegen=stats_mode=single",
                    palette
                ]
                cmd2 = [
                    ffmpeg, "-y", "-framerate", str(fps),
                    "-i", os.path.join(tmp_dir, "frame_%06d.png"),
                    "-i", palette,
                    "-lavfi", "paletteuse=dither=sierra2_4a",
                    "-loop", "0",
                    out_path
                ]
                subprocess.run(cmd1, check=True)
                subprocess.run(cmd2, check=True)
                ok = True
            else:
                # H.264 requires even dims; use pad (not scale) to preserve crispness
                vf = "pad=ceil(iw/2)*2:ceil(ih/2)*2"
                cmd = [
                    ffmpeg, "-y",
                    "-framerate", str(fps),
                    "-i", os.path.join(tmp_dir, "frame_%06d.png"),
                    "-vf", vf,
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                    out_path
                ]
                subprocess.run(cmd, check=True)
                ok = True
    except subprocess.CalledProcessError:
        ok = False

    info = {
        "frames": len(frame_indices),
        "fps": fps,
        "path": out_path if ok else tmp_dir,
        "duration_s": len(frame_indices) / fps,
        "stitched": ok,
        "ffmpeg": bool(ffmpeg),
        "frame_dir": tmp_dir,
    }

    if ok and cleanup and made_tmp:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    if not ok:
        print(
            "\nFrames were written to:", tmp_dir,
            "\nCouldn't stitch automatically (ffmpeg missing or failed).",
            "\nTry this (pads to even dims):\n"
            f'  ffmpeg -y -framerate {fps} -i "{os.path.join(tmp_dir, "frame_%06d.png")}" '
            '-vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -c:v libx264 -pix_fmt yuv420p "orbits.mp4"\n'
        )

    return info
