"""One picture per planned grasp: what the planner saw and why it chose (or rejected) a base pose.

Called by BasePlacement.find_standpoint(..., viz_dir=...). Left: top view in the map frame. Right: side view along
the gripper's approach. Needs matplotlib (in the mm-gpu image).
"""
import math
from pathlib import Path

import numpy as np

from scene_io import base_to_map_xy

VIEW_R = 1.8            # top view: metres around the grasp
SIDE_HALF_W = 0.15      # side view: points within this distance of the approach plane
REASONS = {
    "ok": "a base pose was found",
    "no_ik_solution": "no base pose nearby lets the arm reach this gripper pose without colliding",
    "outside_cone": "reachable only with the target outside the arm's ±30° cone",
    "cannot_park": "reachable, but Kachaka cannot park at any of those spots",
    "not_robust": "reachable, but no spot survives at least half of the 27 navigation errors",
}


def _occupancy(ax, yaml_path):
    """Kachaka's 2D map as the background (free = white, occupied = dark, unknown = light grey)."""
    import yaml
    from PIL import Image
    y = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8"))
    img = np.asarray(Image.open(Path(yaml_path).parent / y["image"]).convert("L"), float)
    r, (ox, oy) = float(y["resolution"]), y["origin"][:2]
    H, W = img.shape
    ax.imshow(img, cmap="gray", vmin=0, vmax=255, alpha=0.35, origin="upper",
              extent=(ox, ox + W * r, oy, oy + H * r), zorder=0)


def _footprint(ax, pose, cfg, **kw):
    """Kachaka's body rectangle and heading at pose (x, y, yaw), map frame."""
    from matplotlib.patches import Polygon
    k = cfg["robot"]["kachaka"]; L, Wd = k["size_m"][:2]; cx, cy = k.get("body_center_xy", [0.0, 0.0])
    x, y, t = pose
    c = np.array([[cx - L / 2, cy - Wd / 2], [cx + L / 2, cy - Wd / 2], [cx + L / 2, cy + Wd / 2], [cx - L / 2, cy + Wd / 2]])
    R = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
    ax.add_patch(Polygon(c @ R.T + [x, y], closed=True, **kw))
    ax.annotate("", xy=(x + 0.3 * math.cos(t), y + 0.3 * math.sin(t)), xytext=(x, y),
                arrowprops=dict(arrowstyle="->", color=kw.get("edgecolor", "k"), lw=1.5))


def draw(path, cfg, world, scene, res, pinch_map, occupancy=None, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Rectangle

    base = tuple(scene.base_pose_in_map)
    g = np.asarray(pinch_map[:3, 3], float); approach = np.asarray(pinch_map[:3, 2], float)
    P = np.asarray(world["means3D"], float)
    tgt = np.asarray(world["target_mask"], bool) if "target_mask" in world else np.zeros(len(P), bool)
    wrist = np.asarray(world["from_wrist"], bool) if "from_wrist" in world else np.zeros(len(P), bool)
    near = (np.abs(P[:, 0] - g[0]) < VIEW_R) & (np.abs(P[:, 1] - g[1]) < VIEW_R) & (P[:, 2] > 0.03)

    fig, (ax, sx) = plt.subplots(1, 2, figsize=(16, 7.5), gridspec_kw={"width_ratios": [1.15, 1]})

    # ---------------- top view (map frame)
    if occupancy:
        _occupancy(ax, occupancy)
    for m, c, s, lab in ((near & ~tgt & ~wrist, "#9e9e9e", 1, "map points"), (near & wrist & ~tgt, "#4a90d9", 1, "wrist camera"),
                         (near & tgt, "#d0021b", 4, "target")):
        if m.any():
            ax.scatter(P[m, 0], P[m, 1], s=s, c=c, label=lab, zorder=2, linewidths=0)
    bt = base[2]
    for b in scene.boxes_base:
        if b["name"] == "floor":
            continue
        cxy = base_to_map_xy(np.asarray(b["center"][:2])[None], base)[0]; t = b["yaw"] + bt
        dx, dy = b["dims"][0] / 2, b["dims"][1] / 2
        c = np.array([[-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]])
        R = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
        ax.add_patch(Polygon(c @ R.T + cxy, closed=True, facecolor="#f4a6a6", edgecolor="none", alpha=0.35, zorder=1))
    if scene.esdf is not None:
        d = scene.esdf["dims"]
        ax.add_patch(Rectangle((g[0] - d[0] / 2, g[1] - d[1] / 2), d[0], d[1], fill=False, ls="--", ec="#d0021b", lw=1,
                               zorder=3, label="1 cm grid (fine obstacles)"))

    # candidate base poses from the whole-body IK, coloured by the step that stopped them
    Q = np.asarray(res.get("q_success") or np.zeros((0, 9)), float).reshape(-1, 9)
    U = np.asarray(res.get("uniq") or np.zeros((0, 9)), float).reshape(-1, 9)
    park = np.asarray(res.get("parked_ok") or [], bool); rob = res.get("rob") or []
    if len(Q):
        q = base_to_map_xy(Q[:, :2], base)
        ax.scatter(q[:, 0], q[:, 1], s=10, c="#bdbdbd", marker="x", zorder=4, label=f"IK solution (grey only: target outside the ±30° cone)")
    groups = {"cannot park": ([], "#f5a623", "x"), "not checked": ([], "#7fb3e6", "o"),
              "fails some of 27 errors": ([], "#f8e71c", "o"), "27/27": ([], "#2e7d32", "o")}
    for i in range(len(U)):
        if i < len(park) and not park[i]:
            k = "cannot park"
        elif i >= len(rob) or rob[i] is None:
            k = "not checked"
        else:
            k = "27/27" if rob[i] >= 1.0 else "fails some of 27 errors"
        groups[k][0].append(i)
    for k, (idx, c, mk) in groups.items():
        if idx:
            q = base_to_map_xy(U[idx, :2], base)
            ax.scatter(q[:, 0], q[:, 1], s=28, c=c, marker=mk, edgecolors="k" if mk == "o" else None, linewidths=0.4,
                       zorder=5, label=f"{k} ({len(idx)})")
            if k == "fails some of 27 errors":
                for i, p in zip(idx, q):
                    ax.annotate(f"{round(rob[i] * 27)}/27", p, fontsize=7, xytext=(3, 3), textcoords="offset points")
    _footprint(ax, base, cfg, facecolor="none", edgecolor="#555555", ls="--", lw=1.5, zorder=6)
    ax.annotate("robot now", base[:2], fontsize=9, xytext=(-30, -18), textcoords="offset points")
    if res.get("ok"):
        ch = res["chosen_map"]
        _footprint(ax, ch, cfg, facecolor="#2e7d32", alpha=0.35, edgecolor="#2e7d32", lw=2, zorder=7)
        ax.annotate("chosen", ch[:2], fontsize=10, color="#2e7d32", fontweight="bold", xytext=(6, -14), textcoords="offset points")
    ax.scatter([g[0]], [g[1]], marker="*", s=260, c="#ffd400", edgecolors="k", zorder=8, label="gripper (pinch point)")
    a = approach[:2]
    if np.linalg.norm(a) > 1e-3:
        a = a / np.linalg.norm(a) * 0.25
        ax.annotate("", xy=g[:2], xytext=g[:2] - a, arrowprops=dict(arrowstyle="->", color="k", lw=2), zorder=8)
    lo = np.minimum(g[:2], base[:2]) - 0.8; hi = np.maximum(g[:2], base[:2]) + 0.8
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_aspect("equal")
    ax.set_xlabel("map x (m)"); ax.set_ylabel("map y (m)"); ax.set_title("Top view (map frame)")
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=2, fontsize=8, frameon=False)
    for hd in getattr(leg, "legend_handles", None) or getattr(leg, "legendHandles", []):
        if hasattr(hd, "set_sizes"):
            hd.set_sizes([40])

    # ---------------- side view along the approach
    h = approach[:2] if np.linalg.norm(approach[:2]) > 0.2 else g[:2] - np.asarray(base[:2])
    h = h / (np.linalg.norm(h) + 1e-9); lat = np.array([-h[1], h[0]])
    s = (P[:, :2] - g[:2]) @ h; w = (P[:, :2] - g[:2]) @ lat
    m = (np.abs(w) < SIDE_HALF_W) & (s > -1.2) & (s < 0.5)
    for mm, c, sz in ((m & ~tgt & ~wrist, "#9e9e9e", 2), (m & wrist & ~tgt, "#4a90d9", 2), (m & tgt, "#d0021b", 6)):
        if mm.any():
            sx.scatter(s[mm], P[mm, 2], s=sz, c=c, linewidths=0)
    d2 = np.array([approach[:2] @ h, approach[2]])
    grip = np.array([0.0, g[2]]) - d2 * 0.12               # gripper_base, 0.12 m behind the pinch point
    sx.plot([grip[0], 0.0], [grip[1], g[2]], c="k", lw=4, solid_capstyle="round")
    sx.annotate("", xy=(0.0, g[2]), xytext=(grip[0] - d2[0] * 0.15, grip[1] - d2[1] * 0.15),
                arrowprops=dict(arrowstyle="->", color="k", lw=1.5))
    sx.scatter([0.0], [g[2]], marker="*", s=260, c="#ffd400", edgecolors="k", zorder=5)
    sx.axhline(0.0, c="k", lw=0.8)
    sx.set_xlim(-1.2, 0.5); sx.set_ylim(-0.05, max(1.3, g[2] + 0.3)); sx.set_aspect("equal")
    sx.set_xlabel("along the approach (m), 0 = gripper"); sx.set_ylabel("height above floor (m)")
    sx.set_title(f"Side view: points within ±{int(SIDE_HALF_W * 100)} cm of the approach line")

    info = (f"{len(Q)} IK solutions · {res.get('n_cone', 0)} in the ±30° cone · {int(park.sum())} parkable · "
            f"{res.get('n_checked', 0)} checked against 27 errors")
    why = REASONS.get(res.get("reason"), "")
    fig.suptitle(f"{title}  —  {why}\n{info}", fontsize=12)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
