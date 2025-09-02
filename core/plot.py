import numpy as np
import matplotlib.pyplot as plt
from typing import Literal, Optional

def plot_orbits(
    engine,
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

        ax.set_xlabel(plane[0]); ax.set_ylabel(plane[1]); ax.grid(True, alpha=0.2)
        if equal_axes:
            ax.set_aspect("equal", adjustable="datalim")

    # --- barycenter annotation (on all axes) ---
    if show_barycenter and Rcm.shape[0] > 0:
        bx = Rcm[:, ix]; by = Rcm[:, iy]
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
