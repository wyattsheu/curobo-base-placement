"""Inputs of this module → what the base-placement planner needs.

Scope: upstream has already decided WHAT to grasp and the FINAL gripper pose (position + orientation), and found
that the robot cannot do it from where it stands. This module turns

    • the 3D semantic SLAM map (.npz, SLAM_DOCKER_HANDOFF.md §4: means3D, semantic_ids [+ class-name .json])
    • Kachaka's localisation           base_pose_in_map = (x, y, yaw)
    • the decided grasp                grasp position (x, y, z) + orientation (wxyz) in the map frame
    • which points are the target      the SLAM labels the target object's region in the 3D map; those points are
                                       the thing we grasp, never an obstacle

into a PlanningScene, everything expressed in the *base* frame (Kachaka where it stands now):

    boxes_base   obstacle boxes  {"name", "center": [x, y, z], "dims": [dx, dy, dz], "yaw": rad} → cuRobo cuboids
    known        2D mask of map cells known to the SLAM (floor or something seen there) → never park in unknown space
    grasp_pos_base, grasp_quat_base   the decided gripper pose, unchanged, just re-expressed in the base frame
"""
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------- frames
def rot2(t):
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s], [s, c]])


def map_to_base_xy(xy, base_pose_in_map):
    """Points (N, 2) in map → base frame."""
    bx, by, bt = base_pose_in_map
    return (np.asarray(xy) - [bx, by]) @ rot2(bt)          # R(-bt) · (p - b)


def base_to_map_xy(xy, base_pose_in_map):
    bx, by, bt = base_pose_in_map
    return np.asarray(xy) @ rot2(bt).T + [bx, by]


def base_to_map_pose(p, base_pose_in_map):
    """A base-frame pose (x, y, yaw) → map frame. This is what Kachaka's Navigate_to_pose expects."""
    xy = base_to_map_xy(np.array([p[:2]]), base_pose_in_map)[0]
    return float(xy[0]), float(xy[1]), float((p[2] + base_pose_in_map[2] + math.pi) % (2 * math.pi) - math.pi)


def quat_map_to_base(q_wxyz, base_yaw):
    """Orientation in map → base frame: q_base = Rz(-yaw) ⊗ q_map."""
    w1, z1 = math.cos(-base_yaw / 2), math.sin(-base_yaw / 2)
    w2, x2, y2, z2 = q_wxyz
    return [w1 * w2 - z1 * z2, w1 * x2 - z1 * y2, w1 * y2 + z1 * x2, w1 * z2 + z1 * w2]


# ---------------------------------------------------------------------------- data
@dataclass
class KnownMask:
    """Cells the SLAM has seen (map frame; origin = map coords of cell [0, 0] centre, row = y index)."""
    known: np.ndarray
    resolution: float
    origin_xy: tuple
    base_pose_in_map: tuple

    def is_known_base(self, xy_base):
        """Are base-frame points (N, 2) on known cells? Off the map counts as unknown."""
        m = base_to_map_xy(np.atleast_2d(xy_base), self.base_pose_in_map)
        ij = np.floor((m - self.origin_xy) / self.resolution + 0.5).astype(int)
        H, W = self.known.shape
        inside = (ij[:, 0] >= 0) & (ij[:, 0] < W) & (ij[:, 1] >= 0) & (ij[:, 1] < H)
        out = np.zeros(len(m), bool)
        out[inside] = self.known[ij[inside, 1], ij[inside, 0]]
        return out

    def bbox_base(self):
        """Extent (x0, x1, y0, y1) of the known cells, in the base frame."""
        jj, ii = np.nonzero(self.known)
        b = map_to_base_xy(np.c_[ii, jj] * self.resolution + self.origin_xy, self.base_pose_in_map)
        return b[:, 0].min(), b[:, 0].max(), b[:, 1].min(), b[:, 1].max()


@dataclass
class PlanningScene:
    boxes_base: list                    # obstacles in the base frame
    grasp_pos_base: tuple               # decided gripper position (tool_frame) in the base frame
    grasp_quat_base: tuple              # decided gripper orientation (wxyz) in the base frame
    base_pose_in_map: tuple             # Kachaka localisation (x, y, yaw)
    known: KnownMask                    # known area
    esdf: dict = None                   # near the grasp: signed distance grid instead of boxes (esdf_grid), or None


def load_request(path, cfg=None):
    """request.json written by upstream:
       {"base_pose_in_map": [x, y, yaw],
        "grasp_pose_in_map": {"position": [x, y, z], "quaternion_wxyz": [w, x, y, z]},
        "target": {"mask_key": "target_mask"}     # npz field (N,) bool: the SLAM's labelled target region   ← preferred
               or {"class": "can"}                # class name (SLAM .json) or semantic id: that class near the grasp point
       }"""
    req = json.loads(Path(path).read_text(encoding="utf-8"))
    return normalize_request(req, cfg)


def normalize_request(req, cfg=None):
    """Accept the decided gripper pose either in the map frame (grasp_pose_in_map) or, as mm_actions works, in the
    Piper base frame of the robot standing at base_pose_in_map (grasp_pose_in_piper_base). The latter is moved to the
    map frame through the arm mount (config robot.arm_mount_xyz / arm_mount_rpy_deg) — the same transform as
    piper_kinematic.piper_point_to_kachaka. Everything downstream uses grasp_pose_in_map."""
    if "grasp_pose_in_map" in req or "grasp_pose_in_piper_base" not in req:
        return req
    if cfg is None:
        raise ValueError("grasp_pose_in_piper_base needs the config (arm mount)")
    from robot_model import arm_mount
    g = req["grasp_pose_in_piper_base"]; bx, by, bt = req["base_pose_in_map"]
    mxyz, mq = arm_mount(cfg)
    p_k = np.asarray(_qrot(mq, g["position"])) + np.asarray(mxyz)                  # piper base → Kachaka base
    q_k = _qmul(mq, g["quaternion_wxyz"])
    xy = base_to_map_xy(np.array([p_k[:2]]), (bx, by, bt))[0]                       # Kachaka base → map
    q_m = _qmul([math.cos(bt / 2), 0.0, 0.0, math.sin(bt / 2)], q_k)
    out = dict(req); out["grasp_pose_in_map"] = {"position": [float(xy[0]), float(xy[1]), float(p_k[2])], "quaternion_wxyz": list(map(float, q_m))}
    return out


def _qmul(a, b):
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    return [w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2]


def _qrot(q, v):
    w, x, y, z = q; p = _qmul(_qmul(q, [0.0, *v]), [w, -x, -y, -z])
    return p[1:]


def load_class_names(path):
    """Class-name table next to the map, in any of the formats seen in the lab:
    {"0": "floor", ...} (flat) · {"segmentation": [{"id": 0, "class": "chair"}, ...]} (SLAM_DOCKER_HANDOFF §4)
    · {"labels": ["wall", "floor", ...], "instances": [...]} (exported instance table). Returns {id: name}."""
    p = Path(path)
    if not p.exists():
        return {}
    j = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(j, dict) and "segmentation" in j:
        return {int(e["id"]): e["class"] for e in j["segmentation"]}
    if isinstance(j, dict) and "labels" in j:
        return {i: n for i, n in enumerate(j["labels"])}
    return {int(k): v for k, v in j.items()}


# ---------------------------------------------------------------------------- helpers
def _rot_xy(xy, t):
    c, s = math.cos(t), math.sin(t)
    return np.c_[c * xy[:, 0] - s * xy[:, 1], s * xy[:, 0] + c * xy[:, 1]]


def _dilate(mask, n):
    out = mask.copy()
    for _ in range(n):
        o = out.copy()
        o[1:] |= out[:-1]; o[:-1] |= out[1:]; o[:, 1:] |= out[:, :-1]; o[:, :-1] |= out[:, 1:]
        out = o
    return out


def _close(mask, n):
    """Morphological closing: fills the notches a slanted edge leaves on a grid (obstacles grow by < 1 cell)."""
    if n <= 0:
        return mask
    p = np.pad(mask, n + 1)
    return ~_dilate(~_dilate(p, n), n)[n + 1:-n - 1, n + 1:-n - 1]


def decompose(mask, fill=0.7, min_cells=4, max_empty=None):
    """Cover the True cells of a 2D grid with few axis-aligned boxes, tolerant to jagged edges: each connected
    component is covered by its bounding box when that box is at least `fill` occupied and covers at most `max_empty`
    empty cells (so an L-shaped union of two touching objects is not filled in as one big block), otherwise it is
    cut in half along its longest side and each half is handled the same way. Returns [(lo_idx, hi_idx)] inclusive."""
    from scipy import ndimage
    out = []

    def rec(sub, off):
        idx = np.argwhere(sub)
        if len(idx) == 0:
            return
        lo, hi = idx.min(0), idx.max(0)
        box = sub[tuple(slice(a, b + 1) for a, b in zip(lo, hi))]
        if (box.mean() >= fill and (max_empty is None or box.size - box.sum() <= max_empty)) or box.size <= min_cells:
            out.append((off + lo, off + hi)); return
        ax = int(np.argmax(hi - lo)); mid = (lo[ax] + hi[ax] + 1) // 2
        a_ = [slice(None)] * sub.ndim; b_ = [slice(None)] * sub.ndim
        a_[ax] = slice(0, mid); b_[ax] = slice(mid, None)
        rec(sub[tuple(a_)], off.copy())
        o2 = off.copy(); o2[ax] += mid
        rec(sub[tuple(b_)], o2)

    lab, n = ndimage.label(mask)
    for i, sl in enumerate(ndimage.find_objects(lab)):
        if sl is not None:
            rec(lab[sl] == i + 1, np.array([x.start for x in sl]))           # this component only
    return out


def dominant_angle(xy, bin_m=0.01):
    """Orientation (radians, [0, pi/2)) of an object's footprint from its raw points, by minimum projection entropy:
    when the axes line up with the object's edges, the projected points pile up into sharp peaks."""
    xy = np.asarray(xy, float); xy = xy - xy.mean(0)

    def score(t):
        r = _rot_xy(xy, -t); e = 0.0
        for k in (0, 1):
            h = np.bincount(np.floor((r[:, k] - r[:, k].min()) / bin_m).astype(int)).astype(float)
            p = h[h > 0] / h.sum(); e -= (p * np.log(p)).sum()
        return e
    coarse = min(np.radians(np.arange(0, 90, 0.5)), key=score)
    return float(np.mod(min(coarse + np.radians(np.arange(-0.5, 0.51, 0.05)), key=score), np.pi / 2))


def height_levels(top, foot, gap, min_cells):
    """Height levels of one object from its footprint cells' top heights (e.g. a table top, a board standing on it, a
    cabinet next to it). Sorted tops are split wherever two consecutive values differ by more than `gap`; a level only
    counts if its cells form ONE connected patch of at least `min_cells` — scattered edge cells that caught only a few
    low points do not. Cells of a level that does not count are rounded UP to the next level (never less obstacle).
    Returns [(lo, hi, surface)]: level k holds the cells with top >= lo (its points reach up to hi) and is filled up
    to `surface`, the level's own top surface (95th percentile of its cells' tops — not the highest point of anything
    standing on it)."""
    from scipy import ndimage
    vals = np.sort(top[foot])
    ranges = [(a[0], a[-1]) for a in np.split(vals, np.nonzero(np.diff(vals) > gap)[0] + 1)]
    keep = []
    for a, b in ranges:
        lab, n = ndimage.label(foot & (top >= a - 1e-9) & (top <= b + 1e-9))
        if n and np.bincount(lab.ravel())[1:].max() >= min_cells:
            keep.append((b, float(np.percentile(vals[(vals >= a) & (vals <= b)], 95))))
    keep = keep or [(vals[-1], vals[-1])]
    keep[-1] = (vals[-1], keep[-1][1])                         # the top level takes every cell above it, too
    return [(float(vals[0]) if k == 0 else float(keep[k - 1][0]) + 1e-6, float(h), s) for k, (h, s) in enumerate(keep)]


def objects_to_boxes(P, sem, v, min_vox, xy_margin, z_margin, fill=0.7, level_gap=0.06, level_min_cells=8, max_empty_m2=0.04,
                     keep_out=None, workers=0):
    """Points (+ optional semantic ids) → obstacle boxes. Per class, 3D-connected voxel clusters (≥ min_vox voxels) are
    objects; without semantic ids everything is one class, so touching things (a board on a table, a cabinet next to
    it) become ONE object. Each object is therefore also split by height: every footprint cell gets the height of its
    highest point, the cell heights are grouped into levels (height_levels), and the boxes are stacked level by level
    — the cells of level k (and everything taller) get a box from the top of level k-1 up to the top of level k.
    So a table top stays at table height instead of growing to the cabinet's height and swallowing the grasp.
    Boxes lie on the object's own orientation and take the tight extent of its points (+ margin).
    keep_out = (xy (M, 2), z_bottom): the target's footprint. Filling holes in an object's outline would otherwise
    close the gap the target's (removed) points leave, e.g. a bottle standing right in front of a box; layers whose
    top is above z_bottom skip those cells, layers below (the table under the bottle) are kept.
    Objects are independent: with workers > 1 they are boxed in parallel processes (a pool kept for the process's
    lifetime); the boxes and their names come out in the same order either way."""
    from scipy import ndimage
    jobs = []
    for c in np.unique(sem):
        Pc = P[sem == c]
        if len(Pc) < min_vox:
            continue
        o = np.floor(Pc.min(0) / v) * v
        ijk = np.floor((Pc - o) / v).astype(int)
        grid = np.zeros(ijk.max(0)[::-1] + 1, bool); grid[ijk[:, 2], ijk[:, 1], ijk[:, 0]] = True
        lab, n = ndimage.label(grid, structure=np.ones((3, 3, 3)))
        pl = lab[ijk[:, 2], ijk[:, 1], ijk[:, 0]]
        counts = np.bincount(lab.ravel(), minlength=n + 1)
        jobs += [(c, obj, Pc[pl == obj]) for obj in range(1, n + 1) if counts[obj] >= min_vox]   # fewer: noise
    args = (v, xy_margin, z_margin, fill, level_gap, level_min_cells, max_empty_m2, keep_out)
    results = None
    if workers > 1 and len(jobs) > 1:                                           # biggest objects first, results in order
        from concurrent.futures.process import BrokenProcessPool
        try:
            pool = _box_pool(workers)
            futs = {k: pool.submit(_object_boxes, jobs[k][2], *args) for k in sorted(range(len(jobs)), key=lambda k: -len(jobs[k][2]))}
            results = [futs[k].result() for k in range(len(jobs))]
        except BrokenProcessPool:        # e.g. the calling script has no `if __name__ == "__main__":` guard (spawn)
            import warnings
            warnings.warn("box fitting pool failed; boxing serially (guard the calling script with __main__)")
            _POOLS.pop(workers, None)
    if results is None:
        results = [_object_boxes(Po, *args) for _, _, Po in jobs]
    out = []
    for (c, obj, _), boxes in zip(jobs, results):
        for cen, dims, yaw in boxes:
            out.append({"name": f"c{c}_o{obj}_{len(out)}", "center": cen, "dims": dims, "yaw": yaw})
    return out


_POOLS = {}


def _box_pool(workers):
    """Process pool for objects_to_boxes, made once per process; spawned, so no CUDA or thread state is inherited."""
    if workers not in _POOLS:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        import atexit
        _POOLS[workers] = ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"))
        atexit.register(_POOLS[workers].shutdown)                               # before the interpreter tears down
    return _POOLS[workers]


def _ping(_):
    return 0


def warm_box_pool(workers):
    """Start the box-fitting workers now (each imports this module once), so the first request does not wait for them."""
    if workers > 1:
        list(_box_pool(workers).map(_ping, range(4 * workers)))


def _object_boxes(Po, v, xy_margin, z_margin, fill, level_gap, level_min_cells, max_empty_m2, keep_out):
    """Boxes [(centre, dims, yaw)] of one object's points (objects_to_boxes)."""
    from scipy import ndimage
    out = []
    t = dominant_angle(Po[:, :2]) if len(Po) > 20 else 0.0
    Pr = np.c_[_rot_xy(Po[:, :2], -t), Po[:, 2]]
    lo = Pr[:, :2].min(0) - v / 2
    ij = np.floor((Pr[:, :2] - lo) / v).astype(int)
    seen = np.zeros(ij.max(0)[::-1] + 1, bool); seen[ij[:, 1], ij[:, 0]] = True
    foot = ndimage.binary_fill_holes(_close(seen, 1))                          # points only lie on surfaces
    top = np.full(seen.shape, -np.inf); np.maximum.at(top, (ij[:, 1], ij[:, 0]), Pr[:, 2])
    near = ndimage.distance_transform_edt(~seen, return_distances=False, return_indices=True)
    top = top[near[0], near[1]]                                                 # filled-in cells: nearest seen cell
    levels = height_levels(top, foot, level_gap, level_min_cells)
    ko = None
    if keep_out is not None and len(keep_out[0]):                             # the target's footprint, this object's grid
        kij = np.floor((_rot_xy(keep_out[0], -t) - lo) / v).astype(int)
        ok = (kij[:, 0] >= 0) & (kij[:, 0] < foot.shape[1]) & (kij[:, 1] >= 0) & (kij[:, 1] < foot.shape[0])
        if ok.any():
            ko = np.zeros(foot.shape, bool); ko[kij[ok, 1], kij[ok, 0]] = True; ko = _dilate(ko, 1)
    z_floor = None
    for k, (h_lo, h_hi, surf) in enumerate(levels):
        mask = foot & (top >= h_lo - 1e-9)
        if ko is not None and surf > keep_out[1]:                               # a layer reaching above the target's
            mask &= ~ko                                                         # bottom never fills the target's place
        for a, b in decompose(mask, fill=fill, max_empty=max_empty_m2 / v ** 2):
            x0, y0 = lo + a[::-1] * v; x1, y1 = lo + (b[::-1] + 1) * v
            m = (Pr[:, 0] >= x0) & (Pr[:, 0] <= x1) & (Pr[:, 1] >= y0) & (Pr[:, 1] <= y1) & (Pr[:, 2] <= h_hi + 1e-9)
            if k > 0:
                m &= Pr[:, 2] > z_floor                                         # only this level's points set its extent
            if not m.any():
                continue
            lo_q, hi_q = np.percentile(Pr[m], [0.5, 99.5], axis=0)              # robust to single noisy points
            x0, y0 = max(x0, lo_q[0] - xy_margin), max(y0, lo_q[1] - xy_margin)
            x1, y1 = min(x1, hi_q[0] + xy_margin), min(y1, hi_q[1] + xy_margin)
            z0 = lo_q[2] - z_margin if k == 0 else z_floor                       # stacked on the level below
            z1 = (hi_q[2] if len(levels) == 1 else surf) + z_margin               # one level: as tight as its points
            if z1 <= z0:
                continue
            cxy = _rot_xy(np.array([[(x0 + x1) / 2, (y0 + y1) / 2]]), t)[0]
            out.append(([float(cxy[0]), float(cxy[1]), float((z0 + z1) / 2)], [float(x1 - x0), float(y1 - y0), float(z1 - z0)], float(t)))
        z_floor = surf + z_margin if len(levels) > 1 else None                  # only a next level stacks on it
    return out


def esdf_grid(P, center, half, half_z, voxel, gap, margin, keep_out=None):
    """Obstacle points around the grasp → a signed distance grid (metres, negative inside), axis-aligned in the map frame,
    centred on `center`, 2·half × 2·half × 2·half_z, in cuRobo's VoxelGrid layout (x, y, z; voxel centres at
    (i + 0.5)·voxel − half). Every voxel column is filled from its lowest to its highest point, cut wherever two
    consecutive points are more than `gap` apart: a surface seen from above becomes solid below it, but a carton's floor
    stays low next to its wall and a stray point above a table does not fill the space between. Distances are reduced
    by `margin` in every direction; occupying whole voxels already moves a surface out by half a voxel on average.
    keep_out = (xy (M, 2), z_bottom, grow): as for the boxes, nothing is filled inside the target's footprint (grown by
    `grow`) above its bottom — the SLAM blurs the support up into the object, right where the fingers close."""
    from scipy.ndimage import distance_transform_edt
    n = np.array([round(2 * half / voxel), round(2 * half / voxel), round(2 * half_z / voxel)])
    lo = np.asarray(center, float) - n * voxel / 2
    occ = np.zeros(n, bool)
    if len(P):
        ijk = np.clip(np.floor((P - lo) / voxel).astype(int), 0, n - 1)
        o = np.lexsort((ijk[:, 2], ijk[:, 1], ijk[:, 0]))
        ijk = ijk[o]
        new = np.r_[True, (np.diff(ijk[:, 0]) != 0) | (np.diff(ijk[:, 1]) != 0) | (np.diff(ijk[:, 2]) * voxel > gap)]
        start = np.flatnonzero(new); end = np.r_[start[1:], len(ijk)] - 1
        for a, b in zip(start, end):
            occ[ijk[a, 0], ijk[a, 1], ijk[a, 2]:ijk[b, 2] + 1] = True
    if keep_out is not None and len(keep_out[0]):
        from scipy.ndimage import binary_dilation
        fij = np.floor((np.asarray(keep_out[0], float) - lo[:2]) / voxel).astype(int)
        ok = np.all((fij >= 0) & (fij < n[:2]), axis=1)
        foot = np.zeros(n[:2], bool); foot[fij[ok, 0], fij[ok, 1]] = True
        r = int(np.ceil(keep_out[2] / voxel)); yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
        foot = binary_dilation(foot, structure=xx ** 2 + yy ** 2 <= r * r)
        k0 = int(np.clip(np.floor((keep_out[1] - lo[2]) / voxel), 0, n[2]))
        occ[foot, k0:] = False
    if occ.any():
        sdf = np.where(occ, -(distance_transform_edt(occ) - 0.5), distance_transform_edt(~occ) - 0.5) * voxel - margin
    else:
        sdf = np.full(n, 10.0)
    return {"sdf": sdf.astype(np.float16), "center": [float(c) for c in center], "dims": (n * voxel).tolist(),
            "voxel": float(voxel), "occupied": int(occ.sum())}


def boxes_map_to_base(boxes, base_pose_in_map):
    out = []
    for b in boxes:
        c = np.asarray(b["center"], float)
        xy = map_to_base_xy(c[None, :2], base_pose_in_map)[0]
        out.append({"name": b["name"], "center": [float(xy[0]), float(xy[1]), float(c[2])], "dims": list(map(float, b["dims"])),
                    "yaw": float(b.get("yaw", 0.0)) - base_pose_in_map[2]})
    return out


# ---------------------------------------------------------------------------- the one entry point
def load_occupancy(yaml_path):
    """Kachaka's 2D map in ROS map_server form (yaml + png, e.g. the lab export's occupancy_<site>.yaml) → KnownMask
    of FREE cells (map frame). This is where Kachaka itself can drive; the lab's Stage1 uses the same map."""
    import yaml
    from PIL import Image
    y = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8"))
    img = np.asarray(Image.open(Path(yaml_path).parent / y["image"]).convert("L"), float)
    occ = img / 255.0 if y.get("negate", 0) else (255.0 - img) / 255.0
    free = (occ < y.get("free_thresh", 0.196))[::-1]                         # row 0 = lowest y
    r = float(y["resolution"]); ox, oy = y["origin"][:2]
    return free, r, (ox + r / 2, oy + r / 2)


def window_reach_mask(xy_map, base_pose_in_map, cfg):
    """Points (map frame) some part of the robot could touch from a base pose inside the search window (± navigation
    error): within the arm's reach bound (robot_model.arm_reach) + the shoulder's offset + crop_margin_m of the window.
    Anything farther can never matter to the planner, so it is dropped before the (slow) box fitting."""
    from robot_model import arm_reach
    pivot, reach, _ = arm_reach(cfg)
    S = cfg["search"]; b = map_to_base_xy(xy_map, base_pose_in_map)
    gx = np.maximum.reduce([S["window_x"][0] - b[:, 0], np.zeros(len(b)), b[:, 0] - S["window_x"][1]])
    gy = np.maximum.reduce([S["window_y"][0] - b[:, 1], np.zeros(len(b)), b[:, 1] - S["window_y"][1]])
    lim = reach + math.hypot(*pivot[:2]) + cfg["robustness"]["perturb_xy_m"] * math.sqrt(2) + cfg["map"].get("crop_margin_m", 0.1)
    return np.hypot(gx, gy) <= lim


def scene_from_slam(npz, base_pose_in_map, grasp_pos_map, grasp_quat_map, cfg, target=None, class_names=None,
                    ignore_classes=("floor",), occupancy=None):
    """3D semantic SLAM map + decided grasp → PlanningScene (base frame).
    npz: path or dict with means3D (N, 3) float32 [m, SLAM map frame], semantic_ids (N,) uint8,
         optional logit_opacities (N, 1), optional target mask (N,) bool. The SLAM map frame must be the frame
         Kachaka localises in.
    target: {"mask_key": name of the npz field marking the target's points}  ← the SLAM's own labelling, preferred
            {"class": name or id}  → points of that class within target_object_radius_m of the grasp point
            None                   → fallback: points in an upright cylinder around the grasp point (radius
                                     target_clear_radius_m, ± target_clear_half_height_m) — fits an upright object
    class_names: {id: name} (the .json saved next to the .npz); classes may be given by name or id.
    occupancy: Kachaka's 2D map (yaml path). If given, the base may only park on its FREE cells; otherwise the area
               the 3D points have seen is used (the 3D map often has no floor points where nothing stands)."""
    M = cfg["map"]
    data = np.load(npz) if not isinstance(npz, dict) else npz
    P = np.asarray(data["means3D"], np.float64)
    sem = np.asarray(data["semantic_ids"]).reshape(-1).astype(int) if "semantic_ids" in data else np.zeros(len(P), int)
    target = target or {}
    if target.get("mask_key"):
        is_tgt = np.asarray(data[target["mask_key"]]).reshape(-1).astype(bool)
    elif target.get("instance_id") is not None:                                 # needs the map's instance_ids
        is_tgt = np.asarray(data["instance_ids"]).reshape(-1) == int(target["instance_id"])
    else:
        is_tgt = np.zeros(len(P), bool)
    if "logit_opacities" in data:                                               # drop nearly transparent splats
        k = np.asarray(data["logit_opacities"]).reshape(-1) > 0.0; P, sem, is_tgt = P[k], sem[k], is_tgt[k]
    by_name = {v: int(k) for k, v in (class_names or {}).items()}
    cid = lambda c: by_name.get(c, c) if isinstance(c, str) else c
    ign = [cid(c) for c in ignore_classes if not isinstance(cid(c), str)]
    g = np.asarray(grasp_pos_map, float)
    v = M["voxel_m"]

    # known area: cells with floor points or with something in the robot's height band; plus the robot's own footprint
    low = P[:, 2] < M["floor_z_m"]
    band = (P[:, 2] >= M["known_band_m"][0]) & (P[:, 2] <= M["known_band_m"][1])
    xy0 = np.floor(P[:, :2].min(0) / v) * v - v; xy1 = np.ceil(P[:, :2].max(0) / v) * v + v
    W, H = (np.round((xy1 - xy0) / v).astype(int) + 1)
    known = np.zeros((H, W), bool)
    for Q in (P[low], P[band]):
        ij = np.floor((Q[:, :2] - xy0) / v).astype(int); known[ij[:, 1], ij[:, 0]] = True
    known = _dilate(known, M["known_close_cells"])
    org = (xy0[0] + v / 2, xy0[1] + v / 2)
    from robot_model import footprint_xy
    fp = base_to_map_xy(footprint_xy(cfg, 9, 7), base_pose_in_map)
    ij = np.floor((fp - org) / v + 0.5).astype(int); known[ij[:, 1], ij[:, 0]] = True   # (assumes the robot is on the map)
    known_mask = KnownMask(known, v, org, tuple(base_pose_in_map))
    if occupancy is not None:
        free, r, o = load_occupancy(occupancy)
        ij = np.floor((fp - o) / r + 0.5).astype(int)
        ok = (ij[:, 0] >= 0) & (ij[:, 0] < free.shape[1]) & (ij[:, 1] >= 0) & (ij[:, 1] < free.shape[0])
        free = free.copy(); free[ij[ok, 1], ij[ok, 0]] = True                  # where it stands now is drivable
        known_mask = KnownMask(free, r, o, tuple(base_pose_in_map))

    # the target object: the SLAM's labelled region if given, else its class near the grasp point, else a radius
    d = np.linalg.norm(P - g, axis=1)
    if target.get("mask_key") or target.get("instance_id") is not None:
        how = "slam_mask" if target.get("mask_key") else "instance_id"
    elif target.get("class") is not None:
        is_tgt = (sem == cid(target["class"])) & (d < M["target_object_radius_m"]); how = "class_near_grasp"
    else:
        is_tgt = ((np.linalg.norm(P[:, :2] - g[:2], axis=1) < M["target_clear_radius_m"])
                  & (np.abs(P[:, 2] - g[2]) < M.get("target_clear_half_height_m", 0.10))); how = "radius_fallback"
    # segmentation is never exact at the target's surface: points hugging the target (within target_grow_m, inside its
    # height range) are the target too — otherwise a mislabelled rim stays an obstacle right where the fingers go
    grow = M.get("target_grow_m", 0.0)
    if grow > 0 and is_tgt.any():
        from scipy.spatial import cKDTree
        z0, z1 = np.percentile(P[is_tgt, 2], [2, 98])
        cand = ~is_tgt & ~low & (P[:, 2] >= z0) & (P[:, 2] <= z1) & (np.linalg.norm(P[:, :2] - g[:2], axis=1) < 0.5)
        if cand.any():
            d_t, _ = cKDTree(P[is_tgt]).query(P[cand], distance_upper_bound=grow)
            is_tgt[np.nonzero(cand)[0][np.isfinite(d_t)]] = True
    # obstacles: everything above the floor within roi_radius_m of the grasp point, except the target and ignored classes
    obst = ~low & ~is_tgt & ~np.isin(sem, ign) & (np.linalg.norm(P[:, :2] - g[:2], axis=1) <= M["roi_radius_m"])
    if M.get("crop_to_window", False):
        obst &= window_reach_mask(P[:, :2], base_pose_in_map, cfg)
    # around the grasp a 1 cm signed distance grid instead of boxes (map.esdf_half_m > 0): the fingers go within
    # 1-3 cm of the clutter there, closer than 4 cm-grid boxes can tell apart
    grid = None
    if M.get("esdf_half_m", 0.0) > 0:
        h, hz = M["esdf_half_m"], M["esdf_half_z_m"]
        ing = (np.abs(P[:, 0] - g[0]) < h) & (np.abs(P[:, 1] - g[1]) < h) & (np.abs(P[:, 2] - g[2]) < hz) & (P[:, 2] >= M["esdf_min_z_m"])
        ko = (P[is_tgt, :2], float(np.percentile(P[is_tgt, 2], 2)), grow) if is_tgt.any() else None
        grid = esdf_grid(P[obst & ing], g, h, hz, M["esdf_voxel_m"], M["esdf_gap_m"], M.get("esdf_margin_m", 0.005), ko)
        obst &= ~ing
    boxes = objects_to_boxes(P[obst], sem[obst], v, M["min_voxels"], M["box_xy_margin_m"], M["box_z_margin_m"],
                             fill=M.get("box_fill", 0.7), max_empty_m2=M.get("box_max_empty_m2", 0.04), level_gap=M.get("level_gap_m", 0.06),
                             level_min_cells=M.get("level_min_cells", 8),
                             keep_out=(P[is_tgt, :2], float(np.percentile(P[is_tgt, 2], 2))) if is_tgt.any() else None,
                             workers=M.get("box_workers", 0))
    boxes.append({"name": "floor", "center": [org[0] + W * v / 2, org[1] + H * v / 2, -0.025], "dims": [W * v + 2, H * v + 2, 0.05], "yaw": 0.0})

    gb = map_to_base_xy(g[None, :2], base_pose_in_map)[0]
    sc = PlanningScene(boxes_map_to_base(boxes, base_pose_in_map), (float(gb[0]), float(gb[1]), float(g[2])),
                       tuple(quat_map_to_base(grasp_quat_map, base_pose_in_map[2])), tuple(base_pose_in_map),
                       known_mask)
    if grid is not None:                                                        # grid pose: map → base frame
        cb = map_to_base_xy(np.asarray(grid["center"][:2])[None], base_pose_in_map)[0]
        grid["center"] = [float(cb[0]), float(cb[1]), float(grid["center"][2])]; grid["yaw"] = -float(base_pose_in_map[2])
        sc.esdf = grid
    sc.target_points = int(is_tgt.sum()); sc.target_source = how; sc.known_source = "occupancy" if occupancy is not None else "slam_points"
    return sc
