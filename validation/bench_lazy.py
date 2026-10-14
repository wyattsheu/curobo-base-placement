"""The optimiser only where it can change the answer (profile.lazy_optimiser) vs the previous way: same requests,
same chosen station? and how much faster? Needs cuRobo + GPU.

  python validation/bench_lazy.py <maps_root> [site ...]

Requests as in bench_roi.py (first graspable object, 10 grasps, Kachaka starting 0.6–0.9 m and 0.9–1.3 m away), one
long-running planner per setting. "previous" runs twice: its own run-to-run differences are the noise floor.
"""
import copy
import json
import math
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
from robot_model import load_config  # noqa: E402
import scene_io  # noqa: E402
from bench_lab_maps import load_site  # noqa: E402
from bench_roi import requests, STARTS  # noqa: E402

SETTINGS = [("previous", False, None), ("previous again", False, None), ("lazy b27", True, 27), ("lazy b54", True, 54)]


def main(root, sites):
    from placement import BasePlacementPlanner
    cfg0 = load_config(HERE.parent / "config.yaml")
    reqs = []
    for site in sites:
        data, objs, occ = load_site(f"{root}/{site}")
        if objs:
            reqs += [(site, sname, data, occ, r) for sname, r0, r1 in STARTS for r in requests(cfg0, data, objs[0], occ, r0, r1)]
    rows = []
    for tag, lazy, ob in SETTINGS:
        cfg = copy.deepcopy(cfg0); cfg["profile"]["lazy_optimiser"] = lazy
        if ob:
            cfg["profile"]["optimiser_batch"] = ob
        torch.cuda.empty_cache()
        pl = BasePlacementPlanner(cfg)
        for site, sname, data, occ, req in reqs:
            g = req["grasp_pose_in_map"]
            sc = scene_io.scene_from_slam(data, req["base_pose_in_map"], g["position"], g["quaternion_wxyz"], cfg,
                                          target=req["target"], occupancy=occ)
            pl.set_scene(sc)
            t = time.perf_counter(); res = pl.plan(); dt = time.perf_counter() - t
            row = {"setting": tag, "site": site, "start": sname, "grasp": req["grasp"], "ok": res["ok"], "reason": res["reason"],
                   "chosen": res.get("chosen_base"), "robustness": res.get("robustness"), "below": res.get("below_threshold"),
                   "plan_s": dt, "scoring_s": res["timing_s"]["scoring_s"], "n_checked": res["n_checked"]}
            rows.append(row); print(json.dumps(row, default=float), flush=True)
        del pl
    key = lambda r: (r["site"], r["start"], r["grasp"])
    ref = {key(r): r for r in rows if r["setting"] == "previous"}
    for tag, *_ in SETTINGS[1:]:
        R = [r for r in rows if r["setting"] == tag]
        same = sum((r["ok"] == ref[key(r)]["ok"]) and (not r["ok"] or math.dist(r["chosen"][:2], ref[key(r)]["chosen"][:2]) < 0.01
                   and abs(r["chosen"][2] - ref[key(r)]["chosen"][2]) < 0.02) for r in R)
        sc = [r["scoring_s"] for r in R if r["scoring_s"] > 0]; s0 = [ref[key(r)]["scoring_s"] for r in R if r["scoring_s"] > 0]
        print(f"{tag:<15} same result as 'previous' {same}/{len(R)}; scoring total {sum(sc):.1f}s vs {sum(s0):.1f}s", flush=True)
    print("@@JSON@@" + json.dumps(rows, default=float))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
