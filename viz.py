"""One picture per planned grasp: what the planner saw and why it chose (or rejected) a base pose.

Called by BasePlacement.find_standpoint(..., viz_dir=...). Left: top view in the map frame. Middle: side view along
the gripper's approach. Right: 3D close-up of the obstacles the planner uses near the gripper (the boxes fitted to the
point cloud, and the 1 cm grid). Needs matplotlib (in the mm-gpu image).
"""
import math
from pathlib import Path

import numpy as np

from scene_io import base_to_map_xy

VIEW_R = 1.8            # top view: metres around the grasp
SIDE_HALF_W = 0.15      # side view: points within this distance of the approach plane
CLOSE_R = 0.6           # 3D close-up: metres around the gripper
BOX_C, GRID_C = "#c0392b", "#e74c3c"
REASONS = {
    "ok": "a base pose was found",
    "no_ik_solution": "no base pose nearby lets the arm reach this gripper pose without colliding",
    "outside_cone": "reachable only with the target outside the arm's ±30° cone",
    "cannot_park": "reachable, but Kachaka cannot park at any of those spots",
    "not_robust": "reachable, but no spot survives at least half of the navigation errors",
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


def _boxes_map(scene):
    """Obstacle boxes (not the floor) as (name, 8 corners in the map frame, dims)."""
    base = tuple(scene.base_pose_in_map); out = []
    for b in scene.boxes_base:
        if b["name"] == "floor":
            continue
        c = np.asarray(b["center"], float); d = np.asarray(b["dims"], float) / 2; t = b["yaw"] + base[2]
        sx, sy, sz = np.meshgrid([-1, 1], [-1, 1], [-1, 1], indexing="ij")
        loc = np.c_[sx.ravel() * d[0], sy.ravel() * d[1], sz.ravel() * d[2]]
        R = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
        cxy = base_to_map_xy(c[:2][None], base)[0]
        out.append((b["name"], np.c_[loc[:, :2] @ R.T + cxy, loc[:, 2] + c[2]], d * 2))
    return out


def _grid_obstacles(scene):
    """Centres of the 1 cm grid cells the planner treats as obstacle (signed distance <= 0), map frame."""
    e = scene.esdf
    if e is None or "sdf" not in e:
        return np.zeros((0, 3))
    sdf = np.asarray(e["sdf"], np.float32); n = np.array(sdf.shape); v = float(e["voxel"])
    c = np.r_[base_to_map_xy(np.asarray(e["center"][:2])[None], tuple(scene.base_pose_in_map))[0], e["center"][2]]
    ijk = np.argwhere(sdf <= 0)
    return c - n * v / 2 + (ijk + 0.5) * v


def _box_frame(C):
    """Corner 0 and the three edge vectors of a box from _boxes_map (corner index = 4·ix + 2·iy + iz)."""
    return C[0], C[4] - C[0], C[2] - C[0], C[1] - C[0]


def _box_dist(C, p):
    """Distance from point p to the (rotated) box; 0 inside."""
    o, *E = _box_frame(C)
    u = [np.dot(p - o, e) / np.dot(e, e) for e in E]                     # 0..1 inside along each edge
    return float(np.linalg.norm([max(0.0, -t, t - 1.0) * np.linalg.norm(e) for t, e in zip(u, E)]))


def _plane_cut(C, g, h, s_lo, s_hi):
    """Where the vertical plane through g along h crosses the box: (s0, s1, z0, z1) or None."""
    o, ex, ey, _ = _box_frame(C)
    ss = np.linspace(s_lo, s_hi, 341); q = g[:2] + ss[:, None] * h - o[:2]
    a = q @ ex[:2] / np.dot(ex[:2], ex[:2]); b = q @ ey[:2] / np.dot(ey[:2], ey[:2])
    inside = (a >= 0) & (a <= 1) & (b >= 0) & (b <= 1)
    if not inside.any():
        return None
    return ss[inside].min(), ss[inside].max(), C[:, 2].min(), C[:, 2].max()


# 12 edges of a box whose corners come from meshgrid([-1,1]^3, indexing="ij")
_EDGES = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]


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

    boxes = _boxes_map(scene); grid = _grid_obstacles(scene)
    fig = plt.figure(figsize=(23, 8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1, 1.05])
    ax = fig.add_subplot(gs[0]); sx = fig.add_subplot(gs[1]); ax3 = fig.add_subplot(gs[2], projection="3d")

    # ---------------- top view (map frame)
    if occupancy:
        _occupancy(ax, occupancy)
    for m, c, s, lab in ((near & ~tgt & ~wrist, "#9e9e9e", 1, "map points"), (near & wrist & ~tgt, "#4a90d9", 1, "wrist camera"),
                         (near & tgt, "#d0021b", 4, "target")):
        if m.any():
            ax.scatter(P[m, 0], P[m, 1], s=s, c=c, label=lab, zorder=2, linewidths=0)
    for k, (_, C, _) in enumerate(boxes):
        ax.add_patch(Polygon(C[[0, 2, 6, 4], :2], closed=True, facecolor="#f4a6a6", edgecolor=BOX_C, lw=0.5, alpha=0.35,
                             zorder=1, label="obstacle boxes (from the points)" if k == 0 else None))
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
    K = int(res.get("n_perturb", 27))
    groups = {"cannot park": ([], "#f5a623", "x"), "not checked": ([], "#7fb3e6", "o"),
              "fails some navigation errors": ([], "#f8e71c", "o"), "passes all": ([], "#2e7d32", "o")}
    for i in range(len(U)):
        if i < len(park) and not park[i]:
            k = "cannot park"
        elif i >= len(rob) or rob[i] is None:
            k = "not checked"
        else:
            k = "passes all" if rob[i] >= 1.0 else "fails some navigation errors"
        groups[k][0].append(i)
    for k, (idx, c, mk) in groups.items():
        if idx:
            q = base_to_map_xy(U[idx, :2], base)
            ax.scatter(q[:, 0], q[:, 1], s=28, c=c, marker=mk, edgecolors="k" if mk == "o" else None, linewidths=0.4,
                       zorder=5, label=f"{k} ({len(idx)})")
            if k == "fails some navigation errors":
                for i, p in zip(idx, q):
                    ax.annotate(f"{round(rob[i] * K)}/{K}", p, fontsize=7, xytext=(3, 3), textcoords="offset points")
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
    for _, C, _ in boxes:                                     # boxes cut by the vertical plane of the approach
        cut = _plane_cut(C, g, h, -1.2, 0.5)
        if cut is not None:
            s0, s1, z0, z1 = cut
            sx.add_patch(Rectangle((s0, z0), s1 - s0, z1 - z0, facecolor="#f4a6a6", edgecolor=BOX_C, lw=0.8, alpha=0.3, zorder=1))
    if len(grid):
        gw = (grid[:, :2] - g[:2]) @ lat
        gk = np.abs(gw) < 0.006                                  # grid cells on the approach plane
        sx.scatter((grid[gk, :2] - g[:2]) @ h, grid[gk, 2], s=3, c=GRID_C, marker="s", alpha=0.35, linewidths=0, zorder=2)
    d2 = np.array([approach[:2] @ h, approach[2]])
    grip = np.array([0.0, g[2]]) - d2 * 0.12               # gripper_base, 0.12 m behind the pinch point
    sx.plot([grip[0], 0.0], [grip[1], g[2]], c="k", lw=4, solid_capstyle="round")
    sx.annotate("", xy=(0.0, g[2]), xytext=(grip[0] - d2[0] * 0.15, grip[1] - d2[1] * 0.15),
                arrowprops=dict(arrowstyle="->", color="k", lw=1.5))
    sx.scatter([0.0], [g[2]], marker="*", s=260, c="#ffd400", edgecolors="k", zorder=5)
    sx.axhline(0.0, c="k", lw=0.8)
    sx.set_xlim(-1.2, 0.5); sx.set_ylim(-0.05, max(1.3, g[2] + 0.3)); sx.set_aspect("equal")
    sx.set_xlabel("along the approach (m), 0 = gripper"); sx.set_ylabel("height above floor (m)")
    sx.set_title(f"Side view: points within ±{int(SIDE_HALF_W * 100)} cm; boxes and grid cut by the approach plane")
    from matplotlib.lines import Line2D
    sx.legend(handles=[Line2D([], [], ls="", marker="o", ms=5, c="#9e9e9e", label="map points"),
                       Line2D([], [], ls="", marker="o", ms=5, c="#4a90d9", label="wrist camera"),
                       Line2D([], [], ls="", marker="o", ms=5, c="#d0021b", label="target"),
                       Rectangle((0, 0), 1, 1, facecolor="#f4a6a6", edgecolor=BOX_C, alpha=0.5, label="obstacle box"),
                       Line2D([], [], ls="", marker="s", ms=5, c=GRID_C, alpha=0.6, label="1 cm grid: obstacle")],
              loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=3, fontsize=8, frameon=False)

    # ---------------- 3D close-up: the obstacles the planner actually uses near the gripper
    c3 = (np.abs(P[:, 0] - g[0]) < CLOSE_R) & (np.abs(P[:, 1] - g[1]) < CLOSE_R) & (P[:, 2] > 0.02)
    rng = np.random.default_rng(0)
    for mm, c, sz in ((c3 & ~tgt & ~wrist, "#9e9e9e", 1), (c3 & wrist & ~tgt, "#4a90d9", 1), (c3 & tgt, "#d0021b", 3)):
        idx = np.flatnonzero(mm)
        if len(idx):
            idx = rng.choice(idx, min(len(idx), 5000), replace=False)
            ax3.scatter(P[idx, 0], P[idx, 1], P[idx, 2], s=sz, c=c, linewidths=0, alpha=0.6, depthshade=False)
    if len(grid):
        gi = rng.choice(len(grid), min(len(grid), 6000), replace=False)
        ax3.scatter(grid[gi, 0], grid[gi, 1], grid[gi, 2], s=2, c=GRID_C, marker="s", alpha=0.15, linewidths=0, depthshade=False)
    close = []
    for name, C, d in boxes:
        lo3, hi3 = C.min(0), C.max(0)
        if (lo3[0] < g[0] + CLOSE_R and hi3[0] > g[0] - CLOSE_R and lo3[1] < g[1] + CLOSE_R and hi3[1] > g[1] - CLOSE_R):
            close.append((_box_dist(C, g), name, C, d))
    close.sort(key=lambda x: x[0])
    for k, (gap, name, C, d) in enumerate(close):
        top3 = k < 3                                              # the three boxes closest to the gripper: bold, numbered
        for i, j in _EDGES:
            ax3.plot(*zip(C[i], C[j]), c=BOX_C, lw=1.8 if top3 else 0.5, alpha=0.95 if top3 else 0.45)
        if top3:
            t = C[np.argmax(C[:, 2])]
            ax3.text(t[0], t[1], C[:, 2].max() + 0.02, f"[{k + 1}]", fontsize=10, color=BOX_C, fontweight="bold")
    lines = [f"[{k + 1}] {name}  {d[0]*100:.1f} × {d[1]*100:.1f} × {d[2]*100:.1f} cm, "
             + ("contains the gripper point" if gap == 0 else f"{gap*100:.1f} cm from the gripper point")
             for k, (gap, name, C, d) in enumerate(close[:3])]
    wc = f", c{int(world['wrist_class'])} = wrist camera" if "wrist_class" in world else ""
    ax3.text2D(0.0, -0.08, f"Closest boxes (name as in the scene: c<class>_o<object>_<index>{wc}):\n"
               + ("\n".join(lines) if lines else "none within the close-up"), transform=ax3.transAxes, fontsize=8, va="top")
    ax3.scatter([g[0]], [g[1]], [g[2]], marker="*", s=220, c="#ffd400", edgecolors="k", depthshade=False)
    gb = g - approach * 0.12
    ax3.plot([gb[0], g[0]], [gb[1], g[1]], [gb[2], g[2]], c="k", lw=4)
    ax3.plot([gb[0] - approach[0] * 0.15, gb[0]], [gb[1] - approach[1] * 0.15, gb[1]], [gb[2] - approach[2] * 0.15, gb[2]], c="k", lw=1)
    zt = max(1.2, g[2] + 0.4)
    ax3.set_xlim(g[0] - CLOSE_R, g[0] + CLOSE_R); ax3.set_ylim(g[1] - CLOSE_R, g[1] + CLOSE_R); ax3.set_zlim(0, zt)
    ax3.set_box_aspect((2 * CLOSE_R, 2 * CLOSE_R, zt))
    ax3.view_init(elev=22, azim=math.degrees(math.atan2(-h[1], -h[0])) + 35)
    ax3.set_xlabel("x"); ax3.set_ylabel("y"); ax3.set_zlabel("z")
    ax3.set_title(f"3D close-up (±{CLOSE_R:.1f} m): boxes and 1 cm grid the planner uses")

    info = (f"{len(Q)} IK solutions · {res.get('n_cone', 0)} in the ±30° cone · {int(park.sum())} parkable · "
            f"{res.get('n_checked', 0)} checked against {res.get('n_perturb', 27)} navigation errors")
    why = REASONS.get(res.get("reason"), "")
    fig.suptitle(f"{title}  —  {why}\n{info}", fontsize=12)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
