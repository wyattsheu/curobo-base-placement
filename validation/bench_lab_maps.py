"""Do fewer seeds pick as good a station as many seeds, on the lab's real SLAM maps? (needs cuRobo + GPU)

  python validation/bench_lab_maps.py <maps_root> [site ...]

For each site under <maps_root> (the lab export folders: *_sem_color_fine.npz, *_instance_semantic_table_fine.json,
alignment_*.yaml, occupancy_*.yaml) and up to MAX_OBJECTS small objects on it:
  - points → Kachaka map frame with the alignment; the target = points inside the object's bounding box
  - Kachaka starts on the most open free cell about 1 m from the object, facing it
  - N_GRASPS gripper poses at the object centre (approach every 36 deg, horizontal and 45 deg down)
Candidate stations come from one whole-body IK pass (128 seeds, today's setting); every scoring setting and the
reference (32 seeds) score the same stations. Each setting picks a station (best robustness over all grasps, nearest
on ties, as in plan()); regret = reference's best score - reference's score of that pick.
"""
import glob
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
from lab_map import load_alignment, map3d_to_map, load_lab_map  # noqa: E402
from robot_model import load_config  # noqa: E402
import scene_io  # noqa: E402
from bench_multi_grasp import make_grasps, build, candidates, score_pool, sync  # noqa: E402

GRASPABLE = ("bottle", "remote control", "mm chocolate", "book", "headphones", "box", "laptop", "cup", "can")
MAX_OBJECTS = 2
N_GRASPS = 20
K_MAX = 8                       # stations scored per grasp (nearest first)
SCORER_SEEDS = [16, 8, 4]       # 16 = today's config.yaml
REF_SEEDS = 32                  # reference (64 is too slow with the ~350 boxes of a real map)


def one(d, pat):
    return sorted(glob.glob(os.path.join(d, pat)))[0]


def load_site(d, voxel=0.02):
    """→ (data for scene_from_slam, objects, occupancy yaml). With *_instance_labels.npz the verified lab_map loader is
    used (semantic floor, per-point instance ids, target by instance id); without it the floor is the biggest 'floor'
    instance's centre and the target falls back to the radius around the grasp point."""
    table = json.loads(Path(one(d, "*_instance_semantic_table_fine.json")).read_text(encoding="utf-8"))
    A = load_alignment(one(d, "alignment_*.yaml"))
    labels = table["labels"]
    insts = [i for i in table["instances"] if not i.get("filtered")]
    occ = [f for f in sorted(glob.glob(os.path.join(d, "occupancy_*.yaml"))) if "native" not in f] or \
        sorted(glob.glob(os.path.join(d, "occupancy_*.yaml")))
    if glob.glob(os.path.join(d, "*_instance_labels.npz")):
        data, _, _ = load_lab_map(d)
        P, inst = data["means3D"], data["instance_ids"]
        def obj_points(i):
            return P[inst == i["inst_id"]]
        target_of = lambda i: {"instance_id": int(i["inst_id"])}
    else:
        pts = np.load(one(d, "*_sem_color_fine.npz"))["points"].astype(float)
        xy, h = map3d_to_map(pts, A)
        big = max((i for i in insts if labels[i["class_id"]] == "floor"), key=lambda i: i["num_voxels"])
        floor = float(map3d_to_map(np.array(big["center"])[None], A)[1][0])
        P = np.c_[xy, h - floor]
        _, keep = np.unique(np.floor(P / voxel).astype(np.int64), axis=0, return_index=True)
        keep = np.sort(keep); raw, P = pts[keep], P[keep]
        data = {"means3D": P.astype(np.float32)}
        def obj_points(i):
            return P[np.all((raw >= np.array(i["bbox_min"]) - 0.01) & (raw <= np.array(i["bbox_max"]) + 0.01), axis=1)]
        target_of = lambda i: None
    objs = []
    for i in insts:
        c = labels[i["class_id"]]
        if c not in GRASPABLE:
            continue
        q = obj_points(i)
        if len(q) < 20:
            continue
        lo, hi = np.percentile(q, 5, 0), np.percentile(q, 95, 0)
        pos = np.median(q, 0)
        if (hi - lo).max() > (0.30 if c == "box" else 0.45) or not 0.25 <= pos[2] <= 1.10:
            continue
        objs.append({"inst": int(i["inst_id"]), "cls": c, "pos": pos.astype(float).tolist(),
                     "ext": (hi - lo).round(2).tolist(), "target": target_of(i)})
    objs.sort(key=lambda o: (o["cls"] == "box", GRASPABLE.index(o["cls"]), o["inst"]))
    return data, objs[:MAX_OBJECTS], occ[0]


def start_pose(occ_yaml, obj_xy, r_min=0.9, r_max=1.3):
    """Most open free cell r_min..r_max from the object, facing it."""
    from scipy.ndimage import distance_transform_edt
    free, r, o = scene_io.load_occupancy(occ_yaml)
    clear = distance_transform_edt(free) * r
    jj, ii = np.nonzero(free)
    xy = np.c_[ii, jj] * r + o
    d = np.hypot(*(xy - obj_xy).T)
    m = (d >= r_min) & (d <= r_max)
    if not m.any():
        return None
    k = np.argmax(np.where(m, clear[jj, ii], -1))
    return [float(xy[k, 0]), float(xy[k, 1]), float(math.atan2(obj_xy[1] - xy[k, 1], obj_xy[0] - xy[k, 0]))]


def pick(pool, rob):
    """plan()'s rule without the arm plan: best robustness, nearest on ties."""
    if not len(pool):
        return None
    k = min(range(len(pool)), key=lambda i: (-rob[i], math.hypot(pool[i][1][0], pool[i][1][1])))
    return k


def run_scene(cfg, scene, label):
    """One set of candidate stations (whole-body IK, 128 seeds) scored by every setting and by the reference."""
    grasps = make_grasps(scene, N_GRASPS, pitches_deg=(0, 45))
    row = {"scene": label, "n_boxes": len(scene.boxes_base), "settings": []}
    scores = {}
    pool = None
    for seeds in SCORER_SEEDS + [REF_SEEDS]:
        pl, setup = build(cfg, scene, **{"profile.scorer_seeds": seeds, "profile.scorer_group": 4})
        if pool is None:                                    # candidates once, with today's settings
            t = time.perf_counter()
            pool = [(gi, st) for gi, g in enumerate(grasps) for st in candidates(pl, g, K_MAX, stop_on_zero=True)]
            sync(); row["ik_s"] = time.perf_counter() - t; row["pool"] = len(pool)
            row["n_grasps_feasible"] = len({gi for gi, _ in pool})
            print(f"  {label}: {len(scene.boxes_base)} boxes, {row['pool']} stations from "
                  f"{row['n_grasps_feasible']}/{len(grasps)} grasps, ik {row['ik_s']:.1f}s", flush=True)
            if not pool:
                return row
        t = time.perf_counter()
        scores[seeds] = np.array(score_pool(pl, [(grasps[gi], st) for gi, st in pool]))
        sync(); dt = time.perf_counter() - t
        print(f"    scorer_seeds {seeds}: setup {setup:.1f}s (group {pl.group}), score {dt:.1f}s", flush=True)
        row["settings"].append({"scorer_seeds": seeds, "group": pl.group, "setup_s": setup, "score_s": dt})
        del pl; torch.cuda.empty_cache()
    ref = scores[REF_SEEDS]; best = float(ref.max())
    floor = cfg["robustness"]["min_accept"]
    for st in row["settings"]:
        r = scores[st["scorer_seeds"]]; k = pick(pool, r)
        st.update({"picked_grasp": pool[k][0], "picked_own": float(r[k]), "picked_ref": float(ref[k]),
                   "regret": best - float(ref[k]), "mae": float(np.abs(r - ref).mean()),
                   "mean_err": float((r - ref).mean()),
                   "accept_agree": float(((r >= floor) == (ref >= floor)).mean())})
    row["ref_best"] = best
    row["ref_n_accept"] = int((ref >= floor).sum())
    return row


def main(root, sites):
    cfg = load_config(HERE.parent / "config.yaml")
    sites = sites or sorted(os.path.basename(p) for p in glob.glob(os.path.join(root, "*")) if glob.glob(os.path.join(p, "*_sem_color_fine.npz")))
    rows = []
    for site in sites:
        d = os.path.join(root, site)
        try:
            data, objs, occ = load_site(d)
        except (IndexError, FileNotFoundError, ValueError) as e:
            print(f"skip {site}: {e}", flush=True); continue
        print(f"{site}: {[(o['cls'], o['inst'], round(o['pos'][2], 2)) for o in objs]}", flush=True)
        for o in objs:
            start = start_pose(occ, np.array(o["pos"][:2]))
            if start is None:
                print(f"skip {site}/{o['cls']}#{o['inst']}: no free cell near it", flush=True); continue
            # the grasp orientation is replaced per grasp; any valid quaternion works here
            scene = scene_io.scene_from_slam(data, start, o["pos"], [1.0, 0.0, 0.0, 0.0], cfg,
                                             target=o["target"], occupancy=occ)
            label = f"{site}/{o['cls']}#{o['inst']}"
            t = time.perf_counter()
            row = run_scene(cfg, scene, label)
            row.update({"target_points": scene.target_points, "obj_z": round(o["pos"][2], 2), "ext": o["ext"],
                        "wall_s": time.perf_counter() - t})
            rows.append(row)
            print(json.dumps(row, default=float), flush=True)
    print("@@JSON@@" + json.dumps(rows, default=float))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
