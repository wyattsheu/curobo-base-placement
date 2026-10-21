"""A Grasp Lab Global capture (mm_system_prod, `mm_actions.lab_global`) → this project's input. Read only: nothing in
mm_system_prod is changed, its saved files are read as they are.

A run folder `main_ws/grasp_logs/<run>/` holds
    global_pose.json  map_from_arm: Piper base → the map (4×4), alignment_mode (manual_se3 / icp), aligned_map_folder
    clouds.npz        base_T_cam (wrist camera → Piper base at the capture), scene_pts / object_pts (cropped clouds)
    input_depth.png   the wrist depth image (uint16, mm); seg_mask.png the target's pixels; summary.json intrinsics
    grasps.npz        poses (K, 4, 4): GraspGen grasps in the Piper base frame, confidence (K,)
    plan.npz          chosen_grasp, grasp (joint path; its last row is where the arm actually went)   [optional]

Frames
    Grasp Lab Global's map ("native metric") applies the same alignment_*.yaml as lab_map.py: identical x, y; its z is
    the raw plane-normal height, ours has the floor at 0, so  ours = theirs − (0, 0, floor_height_m).
    A GraspGen grasp frame is gripper_base turned −90° about z (checked on executed runs: the arm's final pose by FK
    equals chosen_grasp · Rz(90°) exactly), so  pinch_center = grasp · Rz(90°) · (0, 0, tool_offset_m).
Merging (no registration here; the alignment is used as saved): what the wrist camera saw replaces the SLAM map
    inside its view. A SLAM point is dropped when it projects onto a valid depth pixel and lies more than CARVE_TOL_M in
    front of the measured surface: the camera saw through it, so it is no longer there (scenes change after mapping —
    a moved carton stays in the SLAM map). Points behind the surface, outside the image or on missing depth stay.
    SLAM objects of the capture's class (summary.json label, e.g. bottle) with at least STALE_VIEW_FRAC of their points
    inside the camera image are dropped whole, hidden parts included — only with --drop-stale, a one-off for this test
    data (map803 scanned 9/30 with several bottles on the table, captures from 10/08 with one); a map scanned together
    with the capture does not need it.
    The depth image (up to MAX_DEPTH_M, thinned to WRIST_VOXEL_M) is then added. target_mask = segmented pixels within
    TARGET_NEAR_M of Grasp Lab's own cleaned object cloud (clouds.npz object_pts) and SLAM points within
    map.target_grow_m of it. Segmented pixels farther from it are depth-edge mixed pixels and are left out entirely.

    python grasp_lab.py --lab-map <map dir> --run <run dir> [--run ...] --out <dir> [--top 6] [--drop-stale]
writes <dir>/<run>.npz (+ .json class names) and <dir>/<run>_requests.jsonl: one request per line for
`run_demo.py --serve`, Kachaka starting where it stood for the capture, the executed grasp first, then GraspGen's best.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from robot_model import load_config, arm_mount
import scene_io

HERE = Path(__file__).parent
CARVE_TOL_M = 0.05          # a SLAM point must be this far in front of the measured depth to be dropped (alignment + depth noise)
MAX_DEPTH_M = 1.5           # depth farther than this is neither added nor used to drop SLAM points
WRIST_VOXEL_M = 0.01
STALE_VIEW_FRAC = 0.3       # SLAM object of the target's class this much inside the camera image → stale, dropped whole
TARGET_NEAR_M = 0.015       # a segmented depth pixel counts as the target only this close to the cleaned object cloud
RZ90 = np.array([[0.0, -1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])


def _T(R=np.eye(3), t=(0.0, 0.0, 0.0)):
    M = np.eye(4); M[:3, :3] = R; M[:3, 3] = t
    return M


def _quat_wxyz(R):
    w = math.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1.0 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1.0 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1.0 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    return [w, x, y, z]


def load_capture(run, floor_height_m, map_dir=None):
    """→ dict: map_from_arm in OUR map frame, wrist points in the map frame, grasps (pinch_center poses in the Piper
    base frame, best first), and the alignment's own metadata."""
    run = Path(run)
    g = json.loads((run / "global_pose.json").read_text(encoding="utf-8"))
    if g.get("map_from_arm") is None:
        raise ValueError(f"{run}: not aligned (global_pose.json has no map_from_arm)")
    if map_dir and g.get("aligned_map_folder") and Path(g["aligned_map_folder"]).name != Path(map_dir).name:
        raise ValueError(f"{run}: aligned to {g['aligned_map_folder']}, not {map_dir}")
    M = np.array(g["map_from_arm"], float); M[2, 3] -= floor_height_m
    c = np.load(run / "clouds.npz")
    to_map = lambda p: np.asarray(p, float).reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]
    cap = {"run": run.name, "map_from_arm": M, "alignment_mode": g.get("alignment_mode"),
           "alignment_metrics": g.get("alignment_metrics"), "placement_estimate": g.get("placement_estimate"),
           "scene_map": to_map(c["scene_pts"]), "object_map": to_map(c["object_pts"]), "grasps": []}
    if (run / "input_depth.png").exists() and "base_T_cam" in c:
        from PIL import Image
        summ = json.loads((run / "summary.json").read_text(encoding="utf-8"))
        cap["label"] = summ.get("label") or None
        K = next(st["intrinsics"] for st in summ["stages"] if "intrinsics" in st)
        cap["depth_m"] = np.asarray(Image.open(run / "input_depth.png"), float) / 1000.0
        cap["K"] = (K["fx"], K["fy"], K["cx"], K["cy"])
        cap["map_from_cam"] = M @ np.asarray(c["base_T_cam"], float)
        cap["seg"] = np.asarray(Image.open(run / "seg_mask.png")) > 0 if (run / "seg_mask.png").exists() else None
    if (run / "plan.npz").exists():
        cap["executed_grasp"] = np.load(run / "plan.npz")["chosen_grasp"]
    if (run / "grasps.npz").exists():
        z = np.load(run / "grasps.npz")
        order = np.argsort(-z["confidence"], kind="stable")
        cap["grasps"] = [(z["poses"][i], float(z["confidence"][i])) for i in order]
    return cap


def pinch_pose(grasp, tool_offset_m):
    """GraspGen grasp (Piper base frame) → pinch_center pose (Piper base frame)."""
    return np.asarray(grasp, float) @ RZ90 @ _T(t=(0.0, 0.0, tool_offset_m))


def kachaka_pose(cfg, map_from_arm):
    """Where Kachaka stood for the capture: (x, y, yaw) in the map, projected to the floor."""
    mxyz, mq = arm_mount(cfg)
    w, x, y, z = mq
    Rm = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                   [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                   [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    B = map_from_arm @ np.linalg.inv(_T(Rm, mxyz))
    return [float(B[0, 3]), float(B[1, 3]), float(math.atan2(B[1, 0], B[0, 0]))]


def checks(cfg, data, cap):
    """How believable the alignment is, measured against the SLAM map (numbers only; nothing is corrected)."""
    from scipy.spatial import cKDTree
    tree = cKDTree(np.asarray(data["means3D"], float))
    d_scene, _ = tree.query(cap["scene_map"]) if len(cap["scene_map"]) else (np.zeros(0), None)
    M = cap["map_from_arm"]
    mz = arm_mount(cfg)[0][2]
    return {"scene_to_slam_median_m": float(np.median(d_scene)) if len(d_scene) else None,
            "scene_within_5cm": float((d_scene < 0.05).mean()) if len(d_scene) else None,
            "arm_base_height_m": float(M[2, 3]), "arm_mount_height_m": float(mz),
            "arm_base_tilt_deg": float(np.degrees(np.arccos(np.clip(M[2, 2], -1.0, 1.0))))}


def depth_cloud(cap):
    """The wrist depth image → map-frame points (one per WRIST_VOXEL_M voxel) and which of them are the target."""
    d = cap["depth_m"]; fx, fy, cx, cy = cap["K"]
    v, u = np.nonzero((d > 0) & (d < MAX_DEPTH_M)); z = d[v, u]
    pc = np.c_[(u - cx) * z / fx, (v - cy) * z / fy, z]
    T = cap["map_from_cam"]; pm = pc @ T[:3, :3].T + T[:3, 3]
    tgt = cap["seg"][v, u] if cap.get("seg") is not None else np.zeros(len(pm), bool)
    if tgt.any() and len(cap["object_map"]):                                   # mask pixels off the object's surface are
        from scipy.spatial import cKDTree                                       # depth-edge mixed pixels: neither target
        d_o, _ = cKDTree(cap["object_map"]).query(pm[tgt], distance_upper_bound=TARGET_NEAR_M)   # nor obstacle
        ok = ~tgt.copy(); ok[np.flatnonzero(tgt)[np.isfinite(d_o)]] = True
        pm, tgt = pm[ok], tgt[ok]
    k = np.floor(pm / WRIST_VOXEL_M).astype(np.int64); k -= k.min(0)
    span = k.max(0) + 1
    key = (k[:, 0] * span[1] + k[:, 1]) * span[2] + k[:, 2]                   # one integer per voxel, (x, y, z) order
    _, first, inv = np.unique(key, return_index=True, return_inverse=True)      # first pixel of every voxel
    is_t = np.bincount(inv.reshape(-1), weights=tgt.astype(float), minlength=len(first)) > 0   # any target pixel in it
    return pm[first], is_t


def in_image(cap, P):
    """Map points that project inside the wrist image (in front of the camera)."""
    T = np.linalg.inv(cap["map_from_cam"]); pc = P @ T[:3, :3].T + T[:3, 3]
    fx, fy, cx, cy = cap["K"]; H, W = cap["depth_m"].shape
    z = np.maximum(pc[:, 2], 1e-9); u = pc[:, 0] / z * fx + cx; v = pc[:, 1] / z * fy + cy
    return (pc[:, 2] > 0.05) & (u >= 0) & (u < W) & (v >= 0) & (v < H)


def stale_instances(cap, data, names):
    """SLAM instances of the capture's class mostly inside the camera image: the camera shows what is there now."""
    cls = {v: int(k) for k, v in (names or {}).items()}.get(cap.get("label"))
    if cls is None or "depth_m" not in cap:
        return []
    P = np.asarray(data["means3D"], float); inst = np.asarray(data["instance_ids"]); sem = np.asarray(data["semantic_ids"])
    cand = np.unique(inst[(sem == cls) & (inst >= 0)])
    return [int(i) for i in cand if in_image(cap, P[inst == i]).mean() >= STALE_VIEW_FRAC]


def seen_through(cap, P):
    """SLAM points the wrist camera saw through: valid depth at their pixel, and more than CARVE_TOL_M in front of it."""
    T = np.linalg.inv(cap["map_from_cam"]); pc = P @ T[:3, :3].T + T[:3, 3]
    fx, fy, cx, cy = cap["K"]; d = cap["depth_m"]; H, W = d.shape
    out = np.zeros(len(P), bool)
    k = pc[:, 2] > 0.05
    u = np.round(pc[k, 0] / pc[k, 2] * fx + cx).astype(int); v = np.round(pc[k, 1] / pc[k, 2] * fy + cy).astype(int)
    inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    idx = np.flatnonzero(k)[inside]; dm = d[v[inside], u[inside]]
    out[idx] = (dm > 0) & (dm < MAX_DEPTH_M) & (pc[idx, 2] < dm - CARVE_TOL_M)
    return out


def free_class(sem):
    """A semantic id the map does not use, for the wrist points (highest free one below 255). They must not share a class
    with any map points: objects are grouped per class, and wrist points merged with, say, the map's unlabelled points
    (255) become one object reaching from the floor to the table, boxed as tall solid blocks."""
    used = set(np.unique(np.asarray(sem)).tolist())
    return next((c for c in range(254, -1, -1) if c not in used), 255)


def fuse(cfg, data, cap, names=None, drop_stale=False):
    """SLAM map with the wrist camera's view replacing it → the npz fields scene_io reads (target_mask marks the object).
    Without a depth image, falls back to SLAM ∪ the recorded clouds."""
    from scipy.spatial import cKDTree
    P = np.asarray(data["means3D"], float)
    stale = stale_instances(cap, data, names) if drop_stale else []
    if "depth_m" in cap:
        drop = seen_through(cap, P) | np.isin(np.asarray(data["instance_ids"]), stale)
        W, obj = depth_cloud(cap)
    else:
        drop = np.zeros(len(P), bool)
        W = np.vstack([cap["scene_map"], cap["object_map"]])
        obj = np.r_[np.zeros(len(cap["scene_map"]), bool), np.ones(len(cap["object_map"]), bool)]
    keep = ~drop
    near_obj = np.zeros(len(P), bool)
    if obj.any():
        dist, _ = cKDTree(W[obj]).query(P, distance_upper_bound=cfg["map"]["target_grow_m"])
        near_obj = np.isfinite(dist)
    wc = free_class(data["semantic_ids"])
    return {"means3D": np.vstack([P[keep], W]).astype(np.float32),
            "semantic_ids": np.r_[np.asarray(data["semantic_ids"])[keep], np.full(len(W), wc)].astype(np.uint8),
            "instance_ids": np.r_[np.asarray(data["instance_ids"])[keep], np.full(len(W), -1)].astype(np.int32),
            "target_mask": np.r_[near_obj[keep], obj],
            "from_wrist": np.r_[np.zeros(int(keep.sum()), bool), np.ones(len(W), bool)],
            "n_slam_dropped": np.int64(drop.sum()), "stale_instances": np.asarray(stale, np.int64), "wrist_class": np.int64(wc)}


def requests(cfg, cap, top=6, min_sep_m=0.01, min_sep_deg=10.0):
    """Requests in the map frame: the executed grasp first (if any), then GraspGen's best distinct grasps."""
    M, off = cap["map_from_arm"], cfg["robot"]["tool_offset_m"]
    start = kachaka_pose(cfg, M)
    cands = ([("executed", cap["executed_grasp"], None)] if "executed_grasp" in cap else []) + \
            [(f"graspgen{i}", G, conf) for i, (G, conf) in enumerate(cap["grasps"])]
    out, kept = [], []
    for name, G, conf in cands:
        T = M @ pinch_pose(G, off)
        dup = any(np.linalg.norm(T[:3, 3] - K[:3, 3]) < min_sep_m and
                  np.degrees(np.arccos(np.clip((np.trace(T[:3, :3].T @ K[:3, :3]) - 1) / 2, -1, 1))) < min_sep_deg for K in kept)
        if dup:
            continue
        kept.append(T)
        out.append({"base_pose_in_map": start,
                    "grasp_pose_in_map": {"position": T[:3, 3].tolist(), "quaternion_wxyz": _quat_wxyz(T[:3, :3])},
                    "target": {"mask_key": "target_mask"}, "grasp": name, "confidence": conf, "run": cap["run"]})
        if len(out) >= top + ("executed_grasp" in cap):
            break
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--lab-map", required=True, help="the map folder the capture was aligned to")
    ap.add_argument("--run", action="append", required=True, help="Grasp Lab run folder (repeatable)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--top", type=int, default=6, help="GraspGen grasps per run besides the executed one")
    ap.add_argument("--drop-stale", action="store_true", help="test data only: drop SLAM objects of the target's class "
                    "that the camera should see (map and capture from different days)")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    from lab_map import load_lab_map
    data, names, _ = load_lab_map(a.lab_map)
    occ = sorted(Path(a.lab_map).glob("occupancy_*.yaml"))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for run in a.run:
        cap = load_capture(run, float(data["floor_height_m"]), a.lab_map)
        f = out / f"{cap['run']}.npz"
        F = fuse(cfg, data, cap, names, a.drop_stale)
        np.savez_compressed(f, **F)
        f.with_suffix(".json").write_text(json.dumps({str(k): v for k, v in names.items()}, ensure_ascii=False), encoding="utf-8")
        R = requests(cfg, cap, a.top)
        with open(out / f"{cap['run']}_requests.jsonl", "w", encoding="utf-8") as fh:
            for r in R:
                fh.write(json.dumps({**r, "npz": str(f), "occupancy": str(occ[0]) if occ else None}) + "\n")
        print(json.dumps({"run": cap["run"], "alignment_mode": cap["alignment_mode"], "start": R[0]["base_pose_in_map"] if R else None,
                          "requests": len(R), "slam_points_dropped": int(F["n_slam_dropped"]),
                          "stale_instances_dropped": F["stale_instances"].tolist(), "target_points": int(F["target_mask"].sum()),
                          "wrist_points": int(F["from_wrist"].sum()), **checks(cfg, data, cap)}))


if __name__ == "__main__":
    main()
