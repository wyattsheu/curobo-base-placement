"""Near the grasp (the ESDF box: ±0.25 m, ±0.35 m in z, z ≥ 0.15): how much of the volume the boxes claim is really empty
(farther than 3 cm from any obstacle point), and the same for the 1 cm ESDF grid (sdf < 0)."""
import sys, json, copy, numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import scene_io
from robot_model import load_config
from scipy.spatial import cKDTree
from scipy.ndimage import map_coordinates

def in_boxes(p, boxes):
    m = np.zeros(len(p), bool)
    for b in boxes:
        if b["name"] == "floor": continue
        c, s = np.cos(-b["yaw"]), np.sin(-b["yaw"]); d = p - b["center"]
        l = np.c_[c * d[:, 0] - s * d[:, 1], s * d[:, 0] + c * d[:, 1], d[:, 2]]
        m |= np.all(np.abs(l) <= np.asarray(b["dims"]) / 2, axis=1)
    return m

def measure(cfg_on, npz, base, g, target, occ, names=None):
    cfg_off = copy.deepcopy(cfg_on); cfg_off["map"]["esdf_half_m"] = 0.0
    so = scene_io.scene_from_slam(npz, base, g["position"], g["quaternion_wxyz"], cfg_off, target=target, occupancy=occ, class_names=names)
    se = scene_io.scene_from_slam(npz, base, g["position"], g["quaternion_wxyz"], cfg_on, target=target, occupancy=occ, class_names=names)
    G = np.array(so.grasp_pos_base); M = cfg_on["map"]; h, hz = M["esdf_half_m"], M["esdf_half_z_m"]
    ax = [np.arange(G[0] - h, G[0] + h, 0.01), np.arange(G[1] - h, G[1] + h, 0.01), np.arange(max(G[2] - hz, M["esdf_min_z_m"]), G[2] + hz, 0.01)]
    X = np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3)
    # obstacle points the planner sees (base frame): the box scene's input = everything minus target / floor
    D = np.load(npz) if not isinstance(npz, dict) else npz
    P = np.asarray(D["means3D"], float); bx, by, bt = base; c, s = np.cos(-bt), np.sin(-bt)
    Pb = np.c_[c * (P[:, 0] - bx) - s * (P[:, 1] - by), s * (P[:, 0] - bx) + c * (P[:, 1] - by), P[:, 2]]
    near = np.all(np.abs(Pb - G) < [h + 0.1, h + 0.1, hz + 0.1], axis=1) & (Pb[:, 2] > M["floor_z_m"])
    dist = cKDTree(Pb[near]).query(X)[0] if near.any() else np.full(len(X), 9.0)
    ib = in_boxes(X, so.boxes_base)
    e = se.esdf; sdf = e["sdf"].astype(float); n = np.array(sdf.shape); v = e["voxel"]; cc = np.array(e["center"]); y = e["yaw"]
    d = X - cc; cy, sy = np.cos(-y), np.sin(-y); loc = np.c_[cy * d[:, 0] - sy * d[:, 1], sy * d[:, 0] + cy * d[:, 1], d[:, 2]]
    ie = map_coordinates(sdf, (loc / v + n / 2 - 0.5).T, order=1, mode="nearest") < 0
    empty = dist > 0.03
    return {"box_vol_L": ib.sum() / 1000, "box_empty_frac": float((ib & empty).sum() / max(ib.sum(), 1)),
            "esdf_vol_L": ie.sum() / 1000, "esdf_empty_frac": float((ie & empty).sum() / max(ie.sum(), 1)),
            "free_lost_to_boxes": float((ib & empty).sum() / max(empty.sum(), 1)), "free_lost_to_esdf": float((ie & empty).sum() / max(empty.sum(), 1)),
            "esdf_not_box_L": float((ie & ~ib).sum() / 1000)}

if __name__ == "__main__":
    S, CFG = sys.argv[1], sys.argv[2]; cfg = load_config(CFG)
    out = []
    if S != "-":
        reqs = [json.loads(l) for l in open(f"{S}/fused/all.jsonl")]
        for q in [reqs[0], reqs[24]]:
            out.append(("fused " + q["run"][9:15], measure(cfg, q["npz"], q["base_pose_in_map"], q["grasp_pose_in_map"], q["target"], q["occupancy"])))
    if len(sys.argv) > 3:
        from bench_lab_maps import load_site
        from bench_roi import requests, STARTS
        for site in sys.argv[4:]:
            data, objs, occ = load_site(f"{sys.argv[3]}/{site}")
            if not objs: continue
            for sname, r0, r1 in STARTS:
                for q in requests(cfg, data, objs[0], occ, r0, r1):
                    out.append((f"{site[:6]} {sname} g{q['grasp']}", measure(cfg, data, q["base_pose_in_map"], q["grasp_pose_in_map"], q["target"], occ)))
    for name, r in out:
        print(f"{name:18s} boxes {r['box_vol_L']:5.1f} L, {100*r['box_empty_frac']:3.0f}% empty | ESDF {r['esdf_vol_L']:5.1f} L, {100*r['esdf_empty_frac']:3.0f}% empty, {r['esdf_not_box_L']:4.1f} L not in boxes | free space claimed: boxes {100*r['free_lost_to_boxes']:4.1f}%, ESDF {100*r['free_lost_to_esdf']:4.1f}%", flush=True)
    print("@@JSON@@" + json.dumps(out))
