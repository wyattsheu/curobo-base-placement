"""Bounded scoring (profile.bound_scoring) and 9 instead of 27 perturbations (robustness.perturb_points): time and
chosen station against today, on the same requests. Needs cuRobo + GPU.

  python validation/bench_bound.py --lab <maps_root> <site> [<site> ...]      # bench_roi's requests (as profile_plan.py)
  python validation/bench_bound.py --requests requests.jsonl                  # e.g. grasp_lab.py output

Settings: 27 (today), 27+bound, 9, 9+bound, 27 again (same call sequence: identical) and 27 reversed (today, requests in
reverse order: cuRobo's own random state differs per request — the method's seed luck, the yardstick for any change
that alters the call sequence). Every setting plans every request with one long-running planner. A station
chosen with 9 points is then re-scored with all 27 by the 27-point planner (what it would have scored today).
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
from robot_model import load_config  # noqa: E402
import scene_io  # noqa: E402

SETTINGS = [("27", 27, False), ("27+bound", 27, True), ("9", 9, False), ("9+bound", 9, True), ("27 again", 27, False),
            ("27 reversed", 27, False)]


def load_requests(a, cfg):
    """→ [(name, scene builder)]"""
    out = []
    if a.requests:
        for k, line in enumerate(open(a.requests)):
            q = json.loads(line)
            name = f"{q.get('run', 'req')[-12:]}:{q.get('grasp', k)}"
            out.append((name, lambda c, q=q: scene_io.scene_from_slam(
                q["npz"], q["base_pose_in_map"], q["grasp_pose_in_map"]["position"], q["grasp_pose_in_map"]["quaternion_wxyz"],
                c, target=q["target"], occupancy=q.get("occupancy"))))
    if a.lab:
        from bench_lab_maps import load_site
        from bench_roi import requests, STARTS
        root, sites = a.lab[0], a.lab[1:]
        for site in sites:
            data, objs, occ = load_site(f"{root}/{site}")
            if not objs:
                continue
            for sname, r0, r1 in STARTS:
                for q in requests(cfg, data, objs[0], occ, r0, r1):
                    g = q["grasp_pose_in_map"]
                    out.append((f"{site[:6]}:{sname}:{q['grasp']}", lambda c, q=q, g=g, data=data, occ=occ: scene_io.scene_from_slam(
                        data, q["base_pose_in_map"], g["position"], g["quaternion_wxyz"], c, target=q["target"], occupancy=occ)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests"); ap.add_argument("--lab", nargs="+")
    ap.add_argument("--settings", nargs="+", help="subset of: " + ", ".join(t for t, _, _ in SETTINGS))
    a = ap.parse_args()
    from placement import BasePlacementPlanner
    cfg0 = load_config(HERE.parent / "config.yaml")
    reqs = load_requests(a, cfg0)
    scenes = [(n, f(cfg0)) for n, f in reqs]                          # same scenes for every setting
    rows, chosen9 = [], {}
    for tag, pts, bound in [x for x in SETTINGS if not a.settings or x[0] in a.settings]:
        cfg = copy.deepcopy(cfg0); cfg["robustness"]["perturb_points"] = pts; cfg["profile"]["bound_scoring"] = bound
        torch.cuda.empty_cache()
        pl = BasePlacementPlanner(cfg)
        for name, sc in (scenes[::-1] if tag == "27 reversed" else scenes):
            t0 = time.perf_counter(); pl.set_scene(sc); r = pl.plan(); t = time.perf_counter() - t0
            T = r["timing_s"]
            row = {"req": name, "setting": tag, "ok": r["ok"], "reason": r["reason"], "below": r["below_threshold"],
                   "robustness": r["robustness"], "chosen": r.get("chosen_base"), "n_checked": r["n_checked"],
                   "n_cand": int(sum(r["parked_ok"])), "scoring_s": T["scoring_s"], "plan_s": t}
            rows.append(row); print(json.dumps(row), flush=True)
            if pts == 9 and r["ok"]:
                chosen9[(name, tag)] = r["chosen_base"]
        del pl
    # re-score the 9-point choices with all 27 points
    cfg = copy.deepcopy(cfg0); cfg["robustness"]["perturb_points"] = 27
    pl = BasePlacementPlanner(cfg)
    for name, sc in scenes:
        keys = [k for k in chosen9 if k[0] == name]
        if not keys:
            continue
        pl.set_scene(sc)
        for k in keys:
            s27 = float(pl.robustness(np.array([chosen9[k]]))[0])
            for row in rows:
                if row["req"] == name and row["setting"] == k[1]:
                    row["score27_of_choice"] = s27
            print(json.dumps({"req": name, "setting": k[1], "score27_of_choice": s27}), flush=True)
    print("@@JSON@@" + json.dumps(rows))


if __name__ == "__main__":
    main()
