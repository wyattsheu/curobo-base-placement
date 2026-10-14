"""Smaller search window + cropping the map to what the arm can reach from it: time and chosen station vs today.
Needs cuRobo + GPU.

  python validation/bench_roi.py <maps_root> [site ...]

For each site, its first graspable object, Kachaka starting near it (0.6–0.9 m, "already beside the target") and
farther (0.9–1.3 m), and 10 gripper poses (approach every 72 deg, horizontal and 45 deg down), every setting below
plans every request with a long-running planner (solvers built once per setting, as run_demo.py --serve does):
  current      search window ±1.5 m, every point within roi_radius_m of the grasp boxed
  crop 1.5     same window, points no station in it can reach are dropped before box fitting (must not change results)
  crop 0.8/0.5 smaller window ("only move a little") + crop
"""
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
from robot_model import load_config  # noqa: E402
import scene_io  # noqa: E402
from bench_lab_maps import load_site, start_pose  # noqa: E402
from bench_multi_grasp import make_grasps  # noqa: E402

SETTINGS = [("current", 1.5, False), ("crop 1.5", 1.5, True), ("crop 0.8", 0.8, True), ("crop 0.5", 0.5, True)]
STARTS = [("near", 0.6, 0.9), ("far", 0.9, 1.3)]


def requests(cfg, data, o, occ, r0, r1):
    start = start_pose(occ, np.array(o["pos"][:2]), r0, r1)
    if start is None:
        return []
    sc = scene_io.scene_from_slam(data, start, o["pos"], [1, 0, 0, 0], cfg, target=o["target"], occupancy=occ)
    t = start[2]; qz = [math.cos(t / 2), 0.0, 0.0, math.sin(t / 2)]
    return [{"base_pose_in_map": start, "grasp_pose_in_map": {"position": o["pos"], "quaternion_wxyz": scene_io._qmul(qz, g["quat"])},
             "target": o["target"], "grasp": i} for i, g in enumerate(make_grasps(sc, 20, pitches_deg=(0, 45))) if i % 2 == 0]


def main(root, sites):
    from placement import BasePlacementPlanner
    cfg0 = load_config(HERE.parent / "config.yaml")
    rows = []
    for site in sites:
        data, objs, occ = load_site(f"{root}/{site}")
        if not objs:
            continue
        o = objs[0]
        reqs = {name: requests(cfg0, data, o, occ, r0, r1) for name, r0, r1 in STARTS}
        for tag, w, crop in SETTINGS:
            cfg = copy.deepcopy(cfg0)
            cfg["search"]["window_x"] = [-w, w]; cfg["search"]["window_y"] = [-w, w]; cfg["map"]["crop_to_window"] = crop
            torch.cuda.empty_cache()
            pl = BasePlacementPlanner(cfg)
            for sname, R in reqs.items():
                for req in R:
                    g = req["grasp_pose_in_map"]
                    t0 = time.perf_counter()
                    sc = scene_io.scene_from_slam(data, req["base_pose_in_map"], g["position"], g["quaternion_wxyz"], cfg,
                                                  target=req["target"], occupancy=occ)
                    t_scene = time.perf_counter() - t0
                    pl.set_scene(sc); res = pl.plan()
                    T = res["timing_s"]
                    row = {"site": site, "obj": f"{o['cls']}#{o['inst']}", "start": sname, "grasp": req["grasp"], "setting": tag,
                           "ok": res["ok"], "reason": res["reason"], "robustness": res.get("robustness"),
                           "chosen_map": res.get("chosen_map"), "n_boxes": len(sc.boxes_base), "boxes_used": T.get("boxes_used"),
                           "scene_s": t_scene, "scoring_s": T["scoring_s"], "arm_plan_s": T["arm_plan_s"],
                           "plan_s": sum(T[k] for k in ("wholebody_ik_s", "filter_s", "scoring_s", "arm_plan_s")),
                           "total_s": time.perf_counter() - t0, "n_checked": res["n_checked"]}
                    rows.append(row)
                    print(json.dumps(row, default=float), flush=True)
            del pl
    print("@@JSON@@" + json.dumps(rows, default=float))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
